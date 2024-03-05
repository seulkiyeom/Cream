import torch
import torch.nn as nn

from model.utils import get_flops
from model.arch_modif import replace_layer

class BottleneckReader():
    def __init__(self, model, criterion, data_loader, modules, init_lamb, device, maxflops, targetflops, lr=0.6, steps=200, batch_size=1, beta=7,
                 loss_scaler=None, clip_grad=0.1, clip_mode=None):
        self.model = model
        self.model_criterion = criterion
        self.device = device
        self.data_loader = data_loader
        self.lr = lr
        self.beta = beta
        self.batch_size = batch_size
        self.train_steps = steps
        self.unique_alphas = UniqueAlphaMasks(self.device)
        self.unique_alphas.create_alpha(init_lamb)
        self.used_masks = [True for _ in range(len(init_lamb))] 
        self.loss_scaler=loss_scaler 
        self.clip_grad=clip_grad
        self.clip_mode=clip_mode

        self.bottlenecks = []
        self.original_layers = []
        self.sequentials = []
        for i, m in enumerate(modules):
            b = Bottleneck(m.type, m.conv_info, lambda_in_idx=None, lambda_out_idx=[i], alphas=self.unique_alphas, forward_with_mask=True) ##이거 아직 다 완성 안됨
            self.bottlenecks.append(b)
            self.original_layers.append(m.module)
            self.sequentials.append(nn.Sequential(m.module, b))

        print(f"Used masks: {[i for i in range(len(self.used_masks)) if self.used_masks[i]]}")

        for i in range(len(self.used_masks)): #if mask == Flase, then pass through 100%
            if not self.used_masks[i]:
                self.unique_alphas.set_lambda(torch.ones(len(init_lamb[i])), i)

        for b in self.bottlenecks:
            b.update_lambdas()

        # init loss:
        flops = self.compute_flops(ignore_mask=True) #(un)pruned target modules only (해당하는 values의 CN + BN만 포함됨)
        #maxflops = non-target modules + unpruned target modules
        #targetflops = non-target modules + pruned target modules

        self.base_flops = maxflops - flops
        print(f"Base flops: {int(self.base_flops):,}")
        self.target_flops = targetflops
        print(f"Target flops: {int(targetflops):,}")
        self.max_flops = maxflops
        print(f"Max flops: {int(maxflops):,}")


    def compute_flops(self, ignore_mask=False):
        flops = 0
        for b in self.bottlenecks:
            flops += b.compute_flops(ignore_mask=ignore_mask)
        return flops
    
    def get_attribution_score(self): #training with bottleneck
        # Attach layer and train the bottleneck
        print(f'nb unique bottlenecks: {sum(self.used_masks)}')
        for i in range(len(self.bottlenecks)):
            replace_layer(self.model, self.original_layers[i], self.sequentials[i]) #bottleneck 싹다 넣어주고
        self._train_bottleneck() #training 시작

        return self.best_attribution_score

    def _train_bottleneck(self):
        params = self.unique_alphas.get_params(self.used_masks)
        optimizer = torch.optim.Adam(lr=self.lr, params=params)
        self.best_attribution_score = []
        self.model.eval() #weight freeze
        optimizer.zero_grad()
        best_loss = 999999
        best_epoch = 1
        i = 0
        accumulation_steps = 1

        while i < self.train_steps:
            for (inputs, targets) in self.data_loader:
                if i == self.train_steps: break
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                print(f"=== Batch {int((i+1))}/{int(self.train_steps)}")
                
                for b in self.bottlenecks:
                    b.update_lambdas()
                out = self.model(inputs)
                # loss = self.model_criterion(out, targets)
                
                loss = self.calc_loss(inputs, targets.detach(), i, self.train_steps, verbose=True)/accumulation_steps
                loss.backward()
                # self.logger.info(f"loss = {loss:.3f}")
                
                if loss < best_loss and i>0.6*self.train_steps:
                    best_epoch = i+1
                    best_loss = float(loss.detach().clone().cpu())
                    self.best_attribution_score = []
                    for j in range(self.unique_alphas.len()):
                        lamb = self.unique_alphas.get_lambda(j)
                        self.best_attribution_score.append(lamb.detach().clone().cpu().tolist()) #self.best_attribution_score에 best pruning ratio가 저장
                optimizer.step()
                optimizer.zero_grad()

                # # show params
                # accumulated_step = ((i+1) / accumulation_steps)
                # if accumulated_step < 20 or accumulated_step%100==0: # after 20 epochs, values don't change that much
                #     attrib_list_str = "attribution_score[0:12]: " + ('' if sum(self.used_masks)==1 else '\n')
                #     for j in range(self.unique_alphas.len()):
                #         lmd_layer = self.unique_alphas.get_lambda(j).detach().clone().cpu().numpy()
                #         attrib_list_str += ('[ ' + ' '.join("{:.2f} ".format(lmbd) for lmbd in lmd_layer[0:12]) + ']\n')
                #     self.logger.info(attrib_list_str)
                i+=1

        print(f'===\nBest loss was {best_loss:.2f} at iteration {best_epoch}\n')

    def calc_loss(self, inputs, targets, step, maxstep, verbose=False):

        for b in self.bottlenecks:
            b.update_lambdas()
        ce_loss_total  = 0
        for j in range(inputs.size(0)): #for each single image
            img = inputs[j].unsqueeze(0)
            batch = img.expand(self.batch_size, -1, -1, -1), targets[j].expand(self.batch_size)
            out = self.model(batch[0]) #forward-pass using different noises with self.batch_size
            ce_loss_total += self.model_criterion(out, batch[1])/inputs.size(0)
        pruning_loss_total, bool_loss_total = self.calc_loss_terms()
        loss = ce_loss_total + self.beta * pruning_loss_total
        if verbose:
            print(f"loss = {ce_loss_total:.3f} + {self.beta * pruning_loss_total:.3f} = {loss:.3f}")
        return loss

    def calc_loss_terms(self):
        """ Calculate the loss terms """

        flops = self.base_flops + self.compute_flops()
        
        # pruning_loss = abs(self.target_flops - flops) / self.target_flops
        if flops > self.target_flops:
            pruning_loss = (flops-self.target_flops) / (self.max_flops-self.target_flops)
        else:
            pruning_loss = 1-(flops/self.target_flops)
        print(f'total flops: {int(flops):,}')

        bool_loss = 0
        for i in range(self.unique_alphas.len()):
            if self.used_masks[i]:
                lamb = self.unique_alphas.get_lambda(i)
                nb_filters = lamb.size(0)
                bool_loss += (torch.sum(torch.abs(lamb-torch.round(lamb)))/nb_filters)
        bool_loss /= sum(self.used_masks)
        
        return pruning_loss, bool_loss


class UniqueAlphaMasks(nn.Module):
    """
    If you input a new init_lamb, it creates a new alpha mask
    parameter (nn.Parameter) from it, and it return the associated lambda.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.alphas = []
        self.sigmoid = nn.Sigmoid()

    def len(self):
        return len(self.alphas)

    def get_lambda(self, pos):
        return self.sigmoid(self.alphas[pos])
    
    def set_lambda(self, init_lamb, layer):
        if isinstance(init_lamb, list):
            init_lamb = torch.tensor(init_lamb, dtype=torch.float32)
    
        init_alpha = 1*(-torch.log((1 / (init_lamb + 1e-8)) - 1)).to(self.device)
        self.alphas[layer].requires_grad = False
        self.alphas[layer].copy_(init_alpha.detach().clone())

    def create_alpha(self, init_lamb):
        for lamb in init_lamb:
            init_alpha = (-torch.log((1 / (lamb + 1e-8)) - 1)).to(self.device)
            alpha = nn.Parameter(init_alpha.detach().clone())
            self.alphas.append(alpha)

    def get_params(self, used_masks=None):
        """
        Return the list of params (for training)
        """
        return self.alphas if used_masks is None else [self.alphas[i] for i in range(self.len()) if used_masks[i]]

class Bottleneck(nn.Module):
    """
    The Attribution Bottleneck.
    Is inserted in a existing model to suppress information, parametrized by a suppression mask alpha.
    
    Let's M be the module before the bottleneck.
    Args:
        - module_type: type of M (Conv2d, Linear, BatchNorm, etc...)
        - info: info on M necessary to compute the FLOPS
        - lambda_in: list of lambda masks that restrict the input flow of M
        - lamb_out: list of lambda masks that restrict the output flow of M

    Remarks:
        - if you don't want the module output to be pruned, use lamb_out=None
        - if you don't want the module input to be pruned, use lamb_in=None
        - if module_type is BatchNorm or ReLu, then lambda_in == lamb_out
        - if there is no skip connections, then len(lambda_in) == len(lamb_out) = 1,
        meaning that no masks are concatenated.
    """
    def __init__(self, module_type, info, lambda_in_idx = None, lambda_out_idx = None, alphas = None, forward_with_mask=True):
        super().__init__()
        self.module_type = module_type
        self.lambda_in_idx = lambda_in_idx
        self.lambda_out_idx = lambda_out_idx
        self.alphas = alphas
        self.info = info
        self.update_lambdas()
        self.forward_with_mask = False if lambda_out_idx is None else forward_with_mask

    def update_lambdas(self):
        self.lambda_in = None if self.lambda_in_idx is None else [self.alphas.get_lambda(i) for i in self.lambda_in_idx]
        self.lambda_out = None if self.lambda_out_idx is None else [self.alphas.get_lambda(i) for i in self.lambda_out_idx]

    def forward(self, r):
        """ Restrict the information from r by reducing the signal amplitude, using the mask alpha """
        if not self.forward_with_mask: return r
        prev = 0
        for i, lmb in enumerate(self.lambda_out):
            partial_r = r[:, prev:prev+lmb.size(0)]#.to(lmb.device)
            lamb = lmb.unsqueeze(1).unsqueeze(1)
            partial_out = lamb*partial_r
            z = torch.cat((z, partial_out), dim=1) if i>0 else partial_out
            prev += lmb.size(0)
        return z

    def compute_flops(self, ignore_mask=False):
        if ignore_mask:
            return get_flops(self.module_type, self.info, None, None)
        return get_flops(self.module_type, self.info, self.lambda_in, self.lambda_out)

class ModuleInfo():
    """
        Save the informations from a module (to avoid having to collect them everytime)
    """
    def __init__(self, module):
        self.type = type(module)
        self.module = module 
        self.conv_info = {'h':0, 'w':0, 'k':1, 'cin':0, 'cout':0, 'has_bias': False} #kernel size = 1 always

    def conv_info(self):
        return self.conv_info

    def module(self):
        return self.module

class ModulesInfo:
    """
    A wrapper for a list of modules.
    Allows to collect data for all modules in one feed-forward.
    Args:
        - model: the model to collect data from
        - modulesInfo: a list of ModuleInfo objects
        - input_img_size: the size of the input image, for the feed-forwarding
        - device: the device of the model
    """
    def __init__(self, model, modulesInfo, input_img_size=None, device='cuda'):
        self.model = model.eval()
        self.device = device
        self.modulesInfo = modulesInfo
        if input_img_size:
            input_image = torch.randn(1, 3, input_img_size, input_img_size).cuda()
            self.feed(input_image)

    
    def _make_feed_hook(self, i):
        def hook(m, x, z):
            self.modulesInfo[i].conv_info['cin'] = int(x[0].size(1))
            self.modulesInfo[i].conv_info['cout'] = int(z.size(1))
            self.modulesInfo[i].conv_info['has_bias'] = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Conv2d):
                self.modulesInfo[i].conv_info['h'] = int(z.size(2)) if len(z.size())>2 else 1
                self.modulesInfo[i].conv_info['w'] = int(z.size(3)) if len(z.size())>2 else 1
                # self.modulesInfo[i].conv_info['k'] = 1 #kernel size는 무조건 1
            elif isinstance(m, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                self.modulesInfo[i].conv_info['h'] = int(z.size(2)) if len(z.size())>2 else 1
                self.modulesInfo[i].conv_info['w'] = int(z.size(3)) if len(z.size())>2 else 1
                # self.modulesInfo[i].conv_info['k'] = int(x[0].size(2))//int(z.size(2))
            else:
                self.modulesInfo[i].conv_info['h'] = int(x[0].size(2)) if len(x[0].size())>2 else 1
                self.modulesInfo[i].conv_info['w'] = int(x[0].size(3)) if len(x[0].size())>2 else 1
        return hook

    def feed(self, input_image):
        hook_handles = [e.module.register_forward_hook(self._make_feed_hook(i)) for i, e in enumerate(self.modulesInfo)]

        if self.device is not None:
            self.model.to(self.device)
            self.model(input_image.to(self.device))
        else:
            self.model(input_image)

        for handle in hook_handles:
            handle.remove()

    def get_conv_info(self, i):
            return self.modulesInfo[i].info
    
    def modules(self):
        return [m for m in self.modulesInfo]

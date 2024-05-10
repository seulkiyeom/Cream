from typing import Iterable, Optional
from losses import DistillationLoss

import torch
import torch.nn as nn
import numpy as np

from thop import profile
from bottleneck import BottleneckReader, ModulesInfo, ModuleInfo
from model.arch_modif import prune_layer, get_module_in_model
from model.common import CrossEntropyLabelSmooth
from engine import train_one_epoch, evaluate
from model.prune import prune
import utils
from timm.utils import get_state_dict

def prune_model(model: torch.nn.Module, 
                args, 
                data_loader: Iterable, 
                optimizer: torch.optim.Optimizer,
                lr_scheduler, 
                model_ema,
                device: torch.device, 
                gpu,
                loss_scaler=None
                ): #args.finetune
    clip_grad = args.clip_grad if args.clip_grad else 0.1
    clip_mode = args.clip_mode if args.clip_mode else None
    pr = 0.3 # 나중에 변수로 선언해야함!!!! 꼭
    nb_batches = 200
    beta = 6
    CLASSES = 1000 #label_smooth: 0.1
    criterion = CrossEntropyLabelSmooth(CLASSES, 0.1).cuda()

    maxflops, _ = compute_flops_and_params(model, input_image_size=224, device=device) #compute original flops
    targetflops = 302.65 * 10**6
    print(f"Pruning ratio: {(1-(targetflops/maxflops)):.2f}: Current FLOPs: {maxflops:.0f}, Target FLOPs: {targetflops:.0f}")

    modules, init_lamb = get_modules(model, device)
    reader = BottleneckReader(model, criterion, data_loader, modules.modules(), init_lamb, device, maxflops, targetflops, steps = nb_batches, beta=beta, 
                              loss_scaler=loss_scaler, clip_grad=clip_grad, clip_mode=clip_mode)
    attribution_score = reader.get_attribution_score()

    import pickle
    # with open('outfile', 'wb') as fp: #when saving
    #     pickle.dump(self.best_attribution_score, fp)
    with open ('outfile', 'rb') as fp: #when loading
        attribution_score = pickle.load(fp)

    # select the indexes to preserve
    preserved_indexes_all = select_index_flops(attribution_score, targetflops, reader)
        
    # test_stat = evaluate(data_loader, model, device)
    
    attrib_list_str = "attribution_score[0:12]: \n"
    for j in range(reader.unique_alphas.len()):
        tmp = reader.unique_alphas.get_lambda(j).detach().clone().cpu().numpy()[0:12]
        attrib_list_str += ('[ ' + ' '.join("{:.2f} ".format(lmbd) for lmbd in tmp) + ']\n')
    print(f'{attrib_list_str}')

    reader.remove_layer()

    #pruning
    prune(model.module, preserved_indexes_all)#, layers_to_prune)
    if model_ema:
        prune(model_ema.ema, preserved_indexes_all)#, layers_to_prune))

    #save pruned model
    filename = args.prune.split(".")[0] + '_pruned.' + args.prune.split(".")[1]
    model_without_ddp = model
    if args.distributed:
        model_without_ddp = model.module

    utils.save_on_master({
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'model_ema': get_state_dict(model_ema),
        'scaler': loss_scaler.state_dict(),
        'args': args,
    }, filename)

    # test_stat = evaluate(data_loader, model, device)
    print(f'Finish Pruning!')

    return model


    # percent_prune = torch.zeros(len(attribution_score), dtype=torch.float32)
    # for i in range(len(attribution_score)):
    #     percent_prune[i] = len(list(filter((0.2).__le__, attribution_score[i]))) / len(attribution_score[i]) * 100

    # ##모든 architecture를 통들어서 value dimension에 있는 redundant ratio (100이면 모든 channel 사용 10이면 해당 layer의 전체 채널 중 10%만 유의미)
    # tensor([96.8750, 93.7500, 96.8750, 87.5000, 95.3125, 96.8750, 93.7500, 84.3750,
    #         87.5000, 93.7500, 93.7500, 89.0625, 82.2917, 80.2083, 77.0833, 69.7917,
    #         83.3333, 78.1250, 77.0833, 72.9167, 68.7500, 75.0000, 87.5000, 62.5000]) #순서는 앞 layer's value 부터 차례대로 (총 6개의 layer x 각 layer 마다 4개의 value = 24개)
    # #앞으로 가면 갈수록 중요한 channel이 많고 뒤로가면 갈수록 적음 (즉, 뒤로가면 갈수록 redundant가 많음)
    

def get_modules(model, device):
    module_list = []
    init_lamb = []
    for name, module in model.named_modules():
        if 'mix' in name and isinstance(module, nn.ModuleList): #select values that ara target for pruning only
            for i in range(len(module)): #head 개수만큼 load
                module_list.append(ModuleInfo(module[i]))
                init_lamb.append(torch.tensor([0.9] * module[i].weight.size(0), dtype=torch.float32))
                # init_lamb.append(torch.tensor([0.9] * module[i].c.weight.size(0), dtype=torch.float32)) #Cream에서 제공하는 Conv + BN 쓰고 싶을 경우
    modules = ModulesInfo(model, module_list, input_img_size=224, device=device)
    return modules, init_lamb


def compute_flops_and_params(model, input_image_size, device):
    """
        Compute flops and number of parameters for the loaded model
    """
    input_image = torch.randn(1, 3,input_image_size, input_image_size).to(device, non_blocking=True)
    flops, params = profile(model.module.to(device), inputs=(input_image,), verbose=False)

    return [int(flops), int(params)]

def select_index_flops(attribution_score, target_flops, r):
    """
        Args:
            - attribution_score: attribution score for each filter in each layer (list of list)
            - target_flops: target flops for the pruned model 
            - r: BottleneckReader
    """
    with torch.no_grad():
        # 1. we find a threshold to have the right number of flops, using dychotomy
        print(f'Looking for optimal threshold...')
        
        thres = 0.5
        delta = 0.25

        attrib = [[1 if e>thres else 0 for e in l] for l in attribution_score]
        base_flops = r.base_flops
        flops = base_flops
        iteration = 0
        while abs(flops-target_flops)>50000 and iteration<50:
            print(f'Testing threshold {thres}')
            attrib = [[1 if e>thres else 0 for e in l] for l in attribution_score]
            # make sure that nothing is 100% pruned
            for i in range(len(attrib)):
                if sum(attrib[i])==0:
                    attrib[i][np.argmax(attribution_score[i])] = 1

            # pseudo-prune model with attrib
            r.update_alpha_with(attrib)
            flops = base_flops + r.compute_flops()

            print(f'Distance to target: {int(abs(flops-target_flops)):,}')
            if flops > target_flops: thres += delta
            else: thres -= delta
            delta /= 2
            iteration +=1
        # 2. once we found the right threshold, we select the indexes to prune
        from itertools import groupby
        preserved_indexes_all = [[bool(e) for e in l] for l in attrib]
        preserved_indexes_all = [[j,i] for j in range(len(preserved_indexes_all)) for i in range(len(preserved_indexes_all[j])) if preserved_indexes_all[j][i]]
        preserved_indexes_all = [[i[1] for i in e] for _,e in groupby(preserved_indexes_all, lambda x: x[0])]

        return preserved_indexes_all

from typing import Iterable, Optional
from losses import DistillationLoss

import torch
import torch.nn as nn

from thop import profile
from bottleneck import BottleneckReader, ModulesInfo, ModuleInfo
from model.arch_modif import prune_layer, get_module_in_model
from model.common import CrossEntropyLabelSmooth

def prune(model: torch.nn.Module,
          data_loader: Iterable, optimizer: torch.optim.Optimizer,
          device: torch.device, 
          loss_scaler=None, clip_grad=0.1, clip_mode=None):
    pr = 0.3 # 나중에 변수로 선언해야함!!!! 꼭
    nb_batches = 200
    beta = 6
    CLASSES = 1000 #label_smooth: 0.1
    criterion = CrossEntropyLabelSmooth(CLASSES, 0.1).cuda()

    maxflops, _ = compute_flops_and_params(model, input_image_size=224, device=device) #compute original flops
    targetflops = 307.72 * 10**6
    print(f"Pruning ratio: {(1-(targetflops/maxflops)):.2f}: Current FLOPs: {maxflops:.0f}, Target FLOPs: {targetflops:.0f}")

    modules, init_lamb = get_modules(model, device)
    reader = BottleneckReader(model, criterion, data_loader, modules.modules(), init_lamb, device, maxflops, targetflops, steps = nb_batches, beta=beta, 
                              loss_scaler=loss_scaler, clip_grad=clip_grad, clip_mode=clip_mode)
    attribution_score = reader.get_attribution_score()
    
    print(f'here')

    percent_prune = torch.zeros(len(attribution_score), dtype=torch.float32)
    for i in range(len(attribution_score)):
        percent_prune[i] = len(list(filter((0.2).__le__, attribution_score[i]))) / len(attribution_score[i]) * 100

    ##모든 architecture를 통들어서 value dimension에 있는 redundant ratio (100이면 모든 channel 사용 10이면 해당 layer의 전체 채널 중 10%만 유의미)
    tensor([96.8750, 93.7500, 96.8750, 87.5000, 95.3125, 96.8750, 93.7500, 84.3750,
            87.5000, 93.7500, 93.7500, 89.0625, 82.2917, 80.2083, 77.0833, 69.7917,
            83.3333, 78.1250, 77.0833, 72.9167, 68.7500, 75.0000, 87.5000, 62.5000]) #순서는 앞 layer's value 부터 차례대로 (총 6개의 layer x 각 layer 마다 4개의 value = 24개)
    #앞으로 가면 갈수록 중요한 channel이 많고 뒤로가면 갈수록 적음 (즉, 뒤로가면 갈수록 redundant가 많음)
    

def get_modules(model, device):
    module_list = []
    init_lamb = []
    for name, module in model.named_modules():
        if 'vs' in name and isinstance(module, nn.ModuleList): #select values that ara target for pruning only
            for i in range(len(module)): #head 개수만큼 load
                module_list.append(ModuleInfo(module[i]))
                init_lamb.append(torch.tensor([0.9] * module[i].c.weight.size(0), dtype=torch.float32))
    modules = ModulesInfo(model, module_list, input_img_size=224, device=device)
    return modules, init_lamb


def compute_flops_and_params(model, input_image_size, device):
    """
        Compute flops and number of parameters for the loaded model
    """
    input_image = torch.randn(1, 3,input_image_size, input_image_size).to(device, non_blocking=True)
    flops, params = profile(model.module.to(device), inputs=(input_image,), verbose=False)

    return [int(flops), int(params)]


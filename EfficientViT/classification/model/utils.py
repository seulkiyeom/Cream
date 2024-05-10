import torch
import torch.nn as nn
import numpy as np

def val2tuple(x: any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    # 입력 값을 리스트로 변환
    if isinstance(x, list):
        x_list = x
    elif isinstance(x, tuple):
        x_list = list(x)
    else:
        x_list = [x]

    # repeat elements if necessary
    if len(x_list) < min_len:
        if idx_repeat < 0:  # 음수 인덱스 처리
            idx_repeat += len(x_list)
        x_list.extend(x_list[idx_repeat] for _ in range(min_len - len(x_list)))

    return tuple(x_list)

def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2
    
def split_layer(total_channels, num_groups):
    split = [int(np.ceil(total_channels / num_groups)) for _ in range(num_groups)]
    split[num_groups - 1] += total_channels - sum(split)
    return split

def get_flops(module_type, info, lamb_in=None, lamb_out=None, train=True):
    """
    Remark: these functions are adapted to match the thop library
    """
    cin = info['cin'] if lamb_in is None else sum([torch.sum(l) for l in lamb_in])
    cout = info['cout'] if lamb_out is None else sum([torch.sum(l) for l in lamb_out])

    if not train:
        if cin is not None: cin = float(cin)
        if cout is not None: cout = float(cout)
    # print()
    # print(cin)
    # print(cout)
    # print()

    assert(cin<=info['cin'] and cout<=info['cout'])
    hw = info['h'] * info['w']
    return get_flops_conv2d(info, cin, cout, hw) + get_flops_batchnorm2d(info, cin, cout, hw) #Conv2d + BN
    # if module_type is nn.Conv2d:
    #     return get_flops_conv2d(info, cin, cout, hw)
    # elif module_type is nn.Linear:
    #     return get_flops_linear(info, cin, cout, hw)
    # elif module_type is nn.BatchNorm2d:
    #     return get_flops_batchnorm2d(info, cin, cout, hw)
    # elif module_type is nn.ReLU:
    #     return get_flops_ReLU(info, cin, cout, hw)
    # elif module_type is nn.AdaptiveAvgPool2d:
    #     return get_flops_AdaptiveAvgPool2d(info, cin, cout, hw)
    # elif module_type is nn.AvgPool2d:
    #     return get_flops_AvgPool2d(info, cin, cout, hw)
    # else: return 0

def get_flops_conv2d(info, cin, cout, hw):
    total_flops = 0
    kk = info['k']**2 
    # bias = Cout x H x W if model has bias, else 0
    bias_flops = 0 #cout*hw if info['has_bias'] else 0
    # Cout x Cin x H x W x Kw x Kh
    return cout * cin * hw * kk + bias_flops

def get_flops_linear(info, cin, cout, hw):
    #bias_flops = cout if info['has_bias'] else 0
    return cin * cout #* hw + bias_flops

def get_flops_batchnorm2d(info, cin, cout, hw):
    return hw * cin * 2

def get_flops_ReLU(info, cin, cout, hw):
    return 0 #hw * cin

def get_flops_AdaptiveAvgPool2d(info, cin, cout, hw):
    kk = info['k']**2
    return (kk+1) * hw * cout

def get_flops_AvgPool2d(info, cin, cout, hw):
    return hw * cout
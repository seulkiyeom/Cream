# --------------------------------------------------------
# EfficientViT Model Architecture
# Copyright (c) 2022 Microsoft
# Build the EfficientViT Model
# Written by: Xinyu Liu
# --------------------------------------------------------
import torch
import itertools

from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite
import time

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim, input_resolution):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, resolution=input_resolution)
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim, resolution=input_resolution)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0, resolution=input_resolution // 2)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h, resolution):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h, resolution=resolution)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0, resolution=resolution)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

class RelPos2d(torch.nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))

        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)
        
    def generate_2d_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H*W, 2) #(H*W 2)
        mask = grid[:, None, :] - grid[None, :, :] #(H*W H*W 2)
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]  #(n H*W H*W)
        return mask
    
    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :] #(l l)
        mask = mask.abs() #(l l)
        # mask[mask >= l/2] = 0 #여기 슬기 수정 ????????????????????????
        mask = mask * self.decay[:, None, None]  #(n l l)
        return mask
    
    def forward(self, slen, activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:

            retention_rel_pos = self.decay.exp()

        elif chunkwise_recurrent:
            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])

            retention_rel_pos = (mask_h, mask_w)

        else:
            mask = self.generate_2d_decay(slen[0], slen[1]) #(n l l)
            retention_rel_pos = mask

        return retention_rel_pos

class CascadedGroupAttention(torch.nn.Module): #슬기꺼 seulki (reuse attention + trainingab) (실험 진행 안함, 다시 할 것)
    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 kernels=[5, 5, 5, 5],        
                 init_value=2,
                heads_ranges=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        self.qk = Conv2d_BN(dim, dim // 2, resolution=resolution)
        self.dws = Conv2d_BN(dim // 4, dim // 4, kernels[0], 1, kernels[0]//2, groups=dim // 4, resolution=resolution)

        vs = []
        for _ in range(num_heads):
            vs.append(Conv2d_BN(dim // (num_heads), self.d, resolution=resolution))
        self.vs = torch.nn.ModuleList(vs)
        
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        points = list(range(resolution))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = abs(p1 - p2)
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        # points = list(itertools.product(range(resolution), range(resolution)))
        # N = len(points)
        # attention_offsets = {}
        # idxs = []
        # for p1 in points:
        #     for p2 in points:
        #         offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
        #         if offset not in attention_offsets:
        #             attention_offsets[offset] = len(attention_offsets)
        #         idxs.append(attention_offsets[offset])
        # self.attention_biases = torch.nn.Parameter(
        #     torch.zeros(1, len(attention_offsets)))
        # self.register_buffer('attention_bias_idxs',
        #                      torch.LongTensor(idxs).view(N, N))

        self.attention_biases_h = torch.nn.Parameter(torch.zeros(1, len(attention_offsets)))
        self.attention_biases_w = torch.nn.Parameter(torch.zeros(1, len(attention_offsets)))

        self.register_buffer('attention_bias_idxs_h', torch.LongTensor(idxs).view(N, N))
        self.register_buffer('attention_bias_idxs_w', torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab_h'):
            del self.ab_h, self.ab_w #0행렬이 만들어져야됨
        else:
            self.ab_h = self.attention_biases_h[:, self.attention_bias_idxs_h]
            self.ab_w = self.attention_biases_w[:, self.attention_bias_idxs_w]

    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape

        trainingab_h = self.attention_biases_h[:, self.attention_bias_idxs_h]
        trainingab_w = self.attention_biases_w[:, self.attention_bias_idxs_w]

        feat = self.qk(x)
        q, k = feat.view(B, -1, H, W).split([C // 4, C // 4], dim=1) # B, C/h, H, W
        # q, k = feat.view(B, -1, H, W).split([C // 2, C // 2], dim=1) # B, C/h, H, W
        q = self.dws(q)
        
        #along with width
        qr_w = q.permute(0, 2, 3, 1) #(b, h, w, d1)
        kr_w = k.permute(0, 2, 3, 1) #(b, h, w, d1)        
        attn_w = (
                (qr_w @ kr_w.transpose(-2, -1)) * self.scale
                +
                (trainingab_w if self.training else self.ab_w)
            ) #(b, h, w_q, w_k)
        attn_w = attn_w.softmax(dim=-1) # BNN

        #along with height
        qr_h = qr_w.permute(0, 2, 1, 3) #(b, w, h, d1)
        kr_h = kr_w.permute(0, 2, 1, 3) #(b, w, h, d1)        
        attn_h = (
                (qr_h @ kr_h.transpose(-2, -1)) * self.scale
                +
                (trainingab_h if self.training else self.ab_h)
            ) #(b, w, h_q, h_k)
        attn_h = attn_h.softmax(dim=-1) # BNN

        feats_in = x.chunk(len(self.vs), dim=1)
        feats_out = []

        feat = feats_in[0]
        for i, vs in enumerate(self.vs):
            if i > 0: # add the previous output to the input
                feat = feat + feats_in[i] #cascading 방식
            v = vs(feat).permute(0, 2, 3, 1) # (b, h, w, d2)
            v = torch.matmul(attn_w, v) # (b, h, w, d2)
            v = v.permute(0, 2, 1, 3) # (b, w, h, d2)
            feats = torch.matmul(attn_h, v) # (b, w, h, d2)
            feats_out.append(feats.permute(0, 3, 2, 1)) # (b, c, h, w)
        x = self.proj(torch.cat(feats_out, 1))

        # x = 0.5 * x + 0.5 * x.mean(dim=[2,3], keepdim=True) #Uniform attention
        return x


# class CascadedGroupAttention(torch.nn.Module): #슬기꺼 seulki (reuse attention + decay mask) (실험 진행 안함, 다시 할 것)
#     r""" Cascaded Group Attention.

#     Args:
#         dim (int): Number of input channels.
#         key_dim (int): The dimension for query and key.
#         num_heads (int): Number of attention heads.
#         attn_ratio (int): Multiplier for the query dim for value dimension.
#         resolution (int): Input resolution, correspond to the window size.
#         kernels (List[int]): The kernel size of the dw conv on query.
#     """
#     def __init__(self, dim, key_dim, num_heads=8,
#                  attn_ratio=4,
#                  resolution=14,
#                  kernels=[5, 5, 5, 5],        
#                  init_value=2,
#                 heads_ranges=4):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = key_dim ** -0.5
#         self.key_dim = key_dim
#         self.d = int(attn_ratio * key_dim)
#         self.attn_ratio = attn_ratio

#         self.qk = Conv2d_BN(dim, dim // 2, resolution=resolution)
#         self.dws = Conv2d_BN(dim // 4, dim // 4, kernels[0], 1, kernels[0]//2, groups=dim // 4, resolution=resolution)

#         vs = []
#         for _ in range(num_heads):
#             vs.append(Conv2d_BN(dim // (num_heads), self.d, resolution=resolution))
#         self.vs = torch.nn.ModuleList(vs)
        
#         self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
#             self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

#         single_num_heads = 1
#         self.Relpos = RelPos2d(dim, single_num_heads, init_value, heads_ranges)
#         self.chunkwise_recurrent = True
    
#     def generate_1d_decay(self, l):
#         '''
#         generate 1d decay mask, the result is l*l
#         '''
#         index = torch.arange(l)
#         mask = index[:, None] - index[None, :] #(l l)
#         mask = mask.abs() #(l l)
#         # mask = mask * self.decay[:, None, None]  #(n l l)
#         return mask

#     def forward(self, x):  # x (B,C,H,W)
#         B, C, H, W = x.shape

#         mask = self.Relpos((H, W), chunkwise_recurrent=self.chunkwise_recurrent)[0].unsqueeze(0)

#         feat = self.qk(x)
#         q, k = feat.view(B, -1, H, W).split([C // 4, C // 4], dim=1) # B, C/h, H, W
#         # q, k = feat.view(B, -1, H, W).split([C // 2, C // 2], dim=1) # B, C/h, H, W
#         q = self.dws(q)
        
#         #along with width
#         qr_w = q.permute(0, 2, 3, 1) #(b, h, w, d1)
#         kr_w = k.permute(0, 2, 3, 1) #(b, h, w, d1)        
#         attn_w = (
#                 (qr_w @ kr_w.transpose(-2, -1)) * self.scale
#                 # +
#                 # (trainingab_w if self.training else self.ab_w)
#             ) #(b, h, w_q, w_k)
#         attn_w = attn_w + mask
#         attn_w = attn_w.softmax(dim=-1) # BNN

#         #along with height
#         qr_h = qr_w.permute(0, 2, 1, 3) #(b, w, h, d1)
#         kr_h = kr_w.permute(0, 2, 1, 3) #(b, w, h, d1)        
#         attn_h = (
#                 (qr_h @ kr_h.transpose(-2, -1)) * self.scale
#                 # +
#                 # (trainingab_h if self.training else self.ab_h)
#             ) #(b, w, h_q, h_k)
#         attn_h = attn_h + mask
#         attn_h = attn_h.softmax(dim=-1) # BNN

#         feats_in = x.chunk(len(self.vs), dim=1)
#         feats_out = []

#         feat = feats_in[0]
#         for i, vs in enumerate(self.vs):
#             if i > 0: # add the previous output to the input
#                 feat = feat + feats_in[i] #cascading 방식
#             v = vs(feat).permute(0, 2, 3, 1) # (b, h, w, d2)
#             v = torch.matmul(attn_w, v) # (b, h, w, d2)
#             v = v.permute(0, 2, 1, 3) # (b, w, h, d2)
#             feats = torch.matmul(attn_h, v) # (b, w, h, d2)
#             feats_out.append(feats.permute(0, 3, 2, 1)) # (b, c, h, w)
#         x = self.proj(torch.cat(feats_out, 1))

#         # x = 0.5 * x + 0.5 * x.mean(dim=[2,3], keepdim=True) #Uniform attention
#         return x

# class CascadedGroupAttention(torch.nn.Module): #슬기꺼 seulki reuse attention 적용됨 (현재 이거 안씀) Max accuracy: 71.79% (reuse_test.txt 참조)
#     r""" Cascaded Group Attention.

#     Args:
#         dim (int): Number of input channels.
#         key_dim (int): The dimension for query and key.
#         num_heads (int): Number of attention heads.
#         attn_ratio (int): Multiplier for the query dim for value dimension.
#         resolution (int): Input resolution, correspond to the window size.
#         kernels (List[int]): The kernel size of the dw conv on query.
#     """
#     def __init__(self, dim, key_dim, num_heads=8,
#                  attn_ratio=4,
#                  resolution=14,
#                  kernels=[5, 5, 5, 5],):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = key_dim ** -0.5
#         self.key_dim = key_dim
#         self.d = int(attn_ratio * key_dim)
#         self.attn_ratio = attn_ratio

#         self.qk = Conv2d_BN(dim, dim // 2, resolution=resolution)
#         self.dws = Conv2d_BN(dim // 4, dim // 4, kernels[0], 1, kernels[0]//2, groups=dim // 4, resolution=resolution)

#         vs = []
#         for _ in range(num_heads):
#             vs.append(Conv2d_BN(dim // (num_heads), self.d, resolution=resolution))
#         self.vs = torch.nn.ModuleList(vs)
        
#         self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
#             self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

#         points = list(itertools.product(range(resolution), range(resolution)))
#         N = len(points)
#         attention_offsets = {}
#         idxs = []
#         for p1 in points:
#             for p2 in points:
#                 offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
#                 if offset not in attention_offsets:
#                     attention_offsets[offset] = len(attention_offsets)
#                 idxs.append(attention_offsets[offset])
#         self.attention_biases = torch.nn.Parameter(
#             torch.zeros(num_heads, len(attention_offsets)))
#         self.register_buffer('attention_bias_idxs',
#                              torch.LongTensor(idxs).view(N, N))

#     @torch.no_grad()
#     def train(self, mode=True):
#         super().train(mode)
#         if mode and hasattr(self, 'ab'):
#             del self.ab
#         else:
#             self.ab = self.attention_biases[:, self.attention_bias_idxs]

#     def forward(self, x):  # x (B,C,H,W)
#         B, C, H, W = x.shape
#         trainingab = self.attention_biases[:, self.attention_bias_idxs]

#         feat = self.qk(x)
#         q, k = feat.view(B, -1, H, W).split([C // 4, C // 4], dim=1) # B, C/h, H, W
#         # q, k = feat.view(B, -1, H, W).split([C // 2, C // 2], dim=1) # B, C/h, H, W
#         q = self.dws(q)
#         q, k = q.flatten(2), k.flatten(2) # B, C/h, N
#         attn = (
#                 (q.transpose(-2, -1) @ k) * self.scale
#                 +
#                 (trainingab[0] if self.training else self.ab[0])
#             )
#         attn = attn.softmax(dim=-1) # BNN

#         feats_in = x.chunk(len(self.vs), dim=1)
#         feats_out = []

#         feat = feats_in[0]
#         for i, vs in enumerate(self.vs):
#             if i > 0: # add the previous output to the input
#                 feat = feat + feats_in[i] #cascading 방식
#             v = vs(feat)
#             v = v.flatten(2) # B, C/h, N
#             feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W) # BCHW
#             feats_out.append(feat)
#         x = self.proj(torch.cat(feats_out, 1))

#         # x = 0.5 * x + 0.5 * x.mean(dim=[2,3], keepdim=True) #Uniform attention
#         return x
    
# class CascadedGroupAttention(torch.nn.Module): #기존 original cascadedGroupattention from CVPR2023 Max accuracy: 71.40% (original_test.txt 참조)
#     r""" Cascaded Group Attention.

#     Args:
#         dim (int): Number of input channels.
#         key_dim (int): The dimension for query and key.
#         num_heads (int): Number of attention heads.
#         attn_ratio (int): Multiplier for the query dim for value dimension.
#         resolution (int): Input resolution, correspond to the window size.
#         kernels (List[int]): The kernel size of the dw conv on query.
#     """
#     def __init__(self, dim, key_dim, num_heads=8,
#                  attn_ratio=4,
#                  resolution=14,
#                  kernels=[5, 5, 5, 5],):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = key_dim ** -0.5
#         self.key_dim = key_dim
#         self.d = int(attn_ratio * key_dim)
#         self.attn_ratio = attn_ratio

#         qkvs = []
#         dws = []
#         for i in range(num_heads):
#             qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))
#             dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim, resolution=resolution))
#         self.qkvs = torch.nn.ModuleList(qkvs)
#         self.dws = torch.nn.ModuleList(dws)
#         self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
#             self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

#         points = list(itertools.product(range(resolution), range(resolution)))
#         N = len(points)
#         attention_offsets = {}
#         idxs = []
#         for p1 in points:
#             for p2 in points:
#                 offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
#                 if offset not in attention_offsets:
#                     attention_offsets[offset] = len(attention_offsets)
#                 idxs.append(attention_offsets[offset])
#         self.attention_biases = torch.nn.Parameter(
#             torch.zeros(num_heads, len(attention_offsets)))
#         self.register_buffer('attention_bias_idxs',
#                              torch.LongTensor(idxs).view(N, N))

#     @torch.no_grad()
#     def train(self, mode=True):
#         super().train(mode)
#         if mode and hasattr(self, 'ab'):
#             del self.ab
#         else:
#             self.ab = self.attention_biases[:, self.attention_bias_idxs]

#     def forward(self, x):  # x (B,C,H,W)
#         B, C, H, W = x.shape
#         trainingab = self.attention_biases[:, self.attention_bias_idxs]
#         feats_in = x.chunk(len(self.qkvs), dim=1)
#         feats_out = []
#         feat = feats_in[0]
#         for i, qkv in enumerate(self.qkvs):
#             if i > 0: # add the previous output to the input
#                 feat = feat + feats_in[i]
#             feat = qkv(feat)
#             q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1) # B, C/h, H, W
#             q = self.dws[i](q)
#             q, k, v = q.flatten(2), k.flatten(2), v.flatten(2) # B, C/h, N
#             attn = (
#                 (q.transpose(-2, -1) @ k) * self.scale
#                 +
#                 (trainingab[i] if self.training else self.ab[i])
#             )
#             attn = attn.softmax(dim=-1) # BNN
#             feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W) # BCHW
#             feats_out.append(feat)
#         x = self.proj(torch.cat(feats_out, 1))
#         return x


class LocalWindowAttention(torch.nn.Module):
    r""" Local Window Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution
        
        window_resolution = min(window_resolution, resolution)
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                attn_ratio=attn_ratio, 
                                resolution=window_resolution,
                                kernels=kernels,)

    def forward(self, x):
        H = W = self.resolution
        B, C, H_, W_ = x.shape
        # Only check this for classifcation models
        assert H == H_ and W == W_, 'input feature has wrong size, expect {}, got {}'.format((H, W), (H_, W_))
               
        if H <= self.window_resolution and W <= self.window_resolution:
            # start_time = time.time()
            x = self.attn(x)
            # print("cascaded method: --- %s seconds ---" % (time.time() - start_time))
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)
            # start_time = time.time()
            x = self.attn(x)
            # print("cascaded method: --- %s seconds ---" % (time.time() - start_time))
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,
                       C).transpose(2, 3).reshape(B, pH, pW, C)
            if padding:
                x = x[:, :H, :W].contiguous()
            x = x.permute(0, 3, 1, 2)
        return x


class EfficientViTBlock(torch.nn.Module):    
    """ A basic EfficientViT building block.

    Args:
        type (str): Type for token mixer. Default: 's' for self-attention.
        ed (int): Number of input channels.
        kd (int): Dimension for query and key in the token mixer.
        nh (int): Number of attention heads.
        ar (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, type,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
            
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn0 = Residual(FFN(ed, int(ed * 2), resolution))

        if type == 's':
            self.mixer = Residual(LocalWindowAttention(ed, kd, nh, attn_ratio=ar, \
                    resolution=resolution, window_resolution=window_resolution, kernels=kernels))
                
        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn1 = Residual(FFN(ed, int(ed * 2), resolution))

    def forward(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))


class EfficientViT(torch.nn.Module):
    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 stages=['s', 's', 's'],
                 embed_dim=[64, 128, 192],
                 key_dim=[16, 16, 16],
                 depth=[1, 2, 3],
                 num_heads=[4, 4, 4],
                 window_size=[7, 7, 7],
                 kernels=[5, 5, 5, 5],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 distillation=False,):
        super().__init__()

        resolution = img_size
        # Patch embedding
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1, resolution=resolution), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1, resolution=resolution // 2), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1, resolution=resolution // 4), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1, resolution=resolution // 8))

        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []

        # Build EfficientViT blocks
        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            for d in range(dpth):
                eval('self.blocks' + str(i+1)).append(EfficientViTBlock(stg, ed, kd, nh, ar, resolution, wd, kernels))
            if do[0] == 'subsample':
                # Build EfficientViT downsample block
                #('Subsample' stride)
                blk = eval('self.blocks' + str(i+2))
                resolution_ = (resolution - 1) // do[1] + 1
                blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i], resolution=resolution)),
                                    Residual(FFN(embed_dim[i], int(embed_dim[i] * 2), resolution)),))
                blk.append(PatchMerging(*embed_dim[i:i + 2], resolution))
                resolution = resolution_
                blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1], resolution=resolution)),
                                    Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2), resolution)),))
        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)
        
        # Classification head
        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x

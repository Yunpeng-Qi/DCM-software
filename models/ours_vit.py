from copy import deepcopy
import math
import sys
from turtle import forward

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchvision
from data_compression.distributions.uniform_noised import (NoisyDeepFactorized,
                                                           NoisyNormal)
from data_compression.entropy_models import (
    ContinuousConditionalEntropyModel, ContinuousUnconditionalEntropyModel)
from data_compression.layers import (SFT, Downsample, QmapDownsample, QmapFuse,
                                     QmapUpsample, ResAttn, ResAttnSplit,
                                     ResBlock, ResBlockDown, ResBlocks,
                                     ResBlockUp, SFTResblk, Upsample)
from data_compression.prob import GaussianConditional, LogisticConditional
from data_compression.quantization import UniformQuantization
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn

# --- functions for swin transformer ---

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    将input tensor x 切分成多个窗口, 每个窗口的大小为window_size x window_size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    将切分后的窗口张量 windows 恢复为原始的图像张量 x
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads     # 每个注意力头对应的通道数
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nHeads

        # get pair-wise relative position index for each token inside the window
        # 计算窗口内token之间的相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww 相对位置索引tensor
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 对相对位置偏置参数表进行截断正态分布初始化，以初始化相对位置偏置参数
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))    # 计算q,k点积

        # 引入相对位置信息到attention计算中，以考虑tokens之间的空间关系
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]  # 获取mask的窗口数
            # print(attn.shape, attn.view(B_ // nW, nW, self.num_heads, N, N).shape, mask.shape)
            # normal: torch.Size([2048, 8, 64, 64]) torch.Size([8, 256, 8, 64, 64]) torch.Size([256, 64, 64])
            # group: torch.Size([2048, 8, 64, 64]) torch.Size([1, 2048, 8, 64, 64]) torch.Size([2048, 64, 64])
            # 将注意力矩阵按窗口进行分组，并添加掩码 simply add?
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn) # avoid overfitting

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):  # SW-MSA
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size, 上下左右的padding
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:     # 通过对特征图移位，并给Attention设置mask来间接实现shifted window
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift  
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 use_shift_window=True):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else (window_size // 2)*int(use_shift_window), 
                # shift_size=0,   # todo : we keep attention happens in each window for 'GroupViT' - > Is it necessary?
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function. 包含了SW-MSA的Atten mask计算以及对模型块的遍历

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        # 定义window并分配标识
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class PatchUnEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # self.proj = nn.ConvTranspose2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = Upsample(in_chans, embed_dim)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding: no need for padding when up-sampling
        # _, _, H, W = x.size()
        # if W % self.patch_size[1] != 0:
        #     x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        # if H % self.patch_size[0] != 0:
        #     x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


# --- functions for plain vit ---

# helpers

# 对输入参数进行处理，如果 t 已经是一个元组（tuple），则保持不变，如果 t 不是元组，则创建一个包含两个相同元素的元组
def pair(t):    
    return t if isinstance(t, tuple) else (t, t)

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),              
            nn.Dropout(dropout),    
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class GroupAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() # nn.Identity()是恒等映射，将输入返回为输出

    def forward(self, x, msk=None):             # bx256x1024
        qkv = self.to_qkv(x).chunk(3, dim=-1)   # bx256x1024
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # bx16x256x64 切分为多头

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale    # bx16x256x256

        if msk != None:
            # only calculate attention in the masked region
            b, _, _ = x.shape
            msk = msk.reshape(b, -1).float().unsqueeze(1)  # bx1x256
            tmp_unique = msk.unique()   # .unique()消除数据中的重复值
            dots_msk = torch.zeros_like(dots)                           # bx16x256x256
            # for each in tmp_unique:
            #     tmp = (msk==each).float()   # bx1x256
            #     tmp = torch.matmul(tmp.transpose(1, 2), tmp).unsqueeze(1)
            #     dots_msk += tmp
            # torchvision.utils.save_image(dots_msk[:,0].unsqueeze(1), 'dots_msk.png')

            # ↓ this is faster for no matrixs multiplication in the loops
            _, h, n, _ = dots.shape
            dots_msk = torch.zeros((b, len(tmp_unique), 1, n)).cuda()
            for idx, each in enumerate(tmp_unique):
                dots_msk[:,idx] = (msk==each).float()
            # 将注意力限制在dots_msk指定的区域
            dots_msk = torch.matmul(dots_msk.transpose(-1, -2), dots_msk).sum(dim=1).unsqueeze(1)
            # torchvision.utils.save_image(dots_msk, 'dots_msk.png')
            dots_msk = dots_msk.repeat(1, h, 1, 1)
            
            dots[dots_msk==0] = float('-inf')

        attn = self.attend(dots)    # softmax (dim=-1)
        attn = self.dropout(attn)   # dropout

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GroupTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, GroupAttention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, msk=None):
        for attn, ff in self.layers:
            x = attn(x, msk=msk) + x
            x = ff(x) + x
        return x


# --- functions for main framework ---
## ------------------------------------------
## --- GroupViT ---
## ------------------------------------------
class FactorizedPriorGroupViT(nn.Module):
    '''
        The first 2 blocks use swin-block without patch merging.
        And the rest use plain transformer block.
        Codes are modified based on Swin-Transformer and Swin-Transformer-Segmentation.

        Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    '''
    def __init__(self, hyper_channels=None, in_channel=3, out_channel=3,
            patch_size=2, embed_dims_swin=[128,192,256], embed_dim_vit=320, 
            depths=[2,4,4,2], num_heads=[8,12,16,16],
            window_size=8, mlp_ratio=2, qkv_bias=True, qk_scale=None,
            drop_rate=0., attn_drop_rate=0., 
            norm_layer=nn.LayerNorm, 
            patch_norm=False,   # default is True.
            use_checkpoint=False, use_shift_window=False,
            dim_head=32):
        super().__init__()

        self.patch_norm = patch_norm

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_channel, embed_dim=embed_dims_swin[0],
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        init_size = 16
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, init_size, init_size))

        # build 3 swin blocks inside a block (e.g. 32 pixels)
        # 2x - 8x8 self-attention, 4x - 8x8 self-attention, 8x - 4x4 self-attention

        # --- encoder ---
        # build swin blocks
        self.g_a_swin_2x = BasicLayer(
                dim=embed_dims_swin[0],
                depth=depths[0],
                num_heads=num_heads[0],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.,
                norm_layer=norm_layer,
                # downsample=PatchMerging,
                downsample=None,
                use_checkpoint=use_checkpoint,
                # use_shift_window=True
                use_shift_window=use_shift_window
                )
        self.g_a_down_2xto4x = Downsample(embed_dims_swin[0], embed_dims_swin[1], 2)
        self.g_a_swin_4x = BasicLayer(
                dim=embed_dims_swin[1],
                depth=depths[1],
                num_heads=num_heads[1],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.,
                norm_layer=norm_layer,
                # downsample=PatchMerging, 
                downsample=None,
                use_checkpoint=use_checkpoint,
                # use_shift_window=True
                use_shift_window=use_shift_window
                )
        self.g_a_down_4xto8x = Downsample(embed_dims_swin[1], embed_dims_swin[2], 2)
        self.g_a_swin_8x = BasicLayer(
                dim=embed_dims_swin[2],
                depth=depths[2],
                num_heads=num_heads[2],
                window_size=window_size // 2,   # self-attention inside 32x32 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.,
                norm_layer=norm_layer,
                downsample=None, 
                use_checkpoint=use_checkpoint,
                use_shift_window=use_shift_window)
        self.g_a_down = Downsample(embed_dims_swin[2], embed_dim_vit, 2)

        # build vit blocks
        self.g_a_vit = GroupTransformer(
            embed_dim_vit, depths[3], num_heads[3], dim_head, embed_dim_vit*mlp_ratio, drop_rate)

        # --- decoder ---
        # build vit blocks
        self.g_s_vit = GroupTransformer(
            embed_dim_vit, depths[3], num_heads[3], dim_head, embed_dim_vit*mlp_ratio, drop_rate)

        self.g_s_up = Upsample(embed_dim_vit, embed_dims_swin[2], 2)

        # build swin blocks
        self.g_s_swin_8x = BasicLayer(
                dim=embed_dims_swin[2],
                depth=depths[2],
                num_heads=num_heads[2],
                window_size=window_size // 2,   # self-attention inside 32x32 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.,
                norm_layer=norm_layer,
                downsample=None, 
                use_checkpoint=use_checkpoint,
                use_shift_window=use_shift_window)
        self.g_s_up_8xto4x = Upsample(embed_dims_swin[2], embed_dims_swin[1], 2)
        self.g_s_swin_4x = BasicLayer(
                dim=embed_dims_swin[1],
                depth=depths[1],
                num_heads=num_heads[1],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.,
                norm_layer=norm_layer,
                downsample=None, 
                use_checkpoint=use_checkpoint,
                # use_shift_window=True
                use_shift_window=use_shift_window
                )
        self.g_s_up_4xto2x = Upsample(embed_dims_swin[1], embed_dims_swin[0], 2)
        self.g_s_swin_2x = BasicLayer(
                dim=embed_dims_swin[0],
                depth=depths[0],
                num_heads=num_heads[0],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.,
                norm_layer=norm_layer,
                downsample=None, 
                use_checkpoint=use_checkpoint,
                # use_shift_window=True
                use_shift_window=use_shift_window
                )
        self.g_s_final = Upsample(embed_dims_swin[0], out_channel, 2)

        # entropy model
        self.quant = UniformQuantization(step=1)
        self.em = ContinuousUnconditionalEntropyModel(
            NoisyDeepFactorized(batch_shape=(embed_dim_vit,)))

    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4

    def g_a(self, x, msk):
        # patch embedding 
        x = self.patch_embed(x)                 # B C H/2 W/2
        _, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)        # B (H/2 * W/2) C

        # swin blocks
        _, _, _, x, Wh, Ww = self.g_a_swin_2x(x, H, W)
        x = rearrange(x, 'b (h w) c -> b c h w', h=Wh, w=Ww)  # B C H/2 W/2
        x = self.g_a_down_2xto4x(x)
        x = rearrange(x, 'b c h w -> b (h w) c')    # B (H/4 W/4) C
        _, _, _, x, Wh, Ww = self.g_a_swin_4x(x, Wh // 2, Ww // 2)
        x = rearrange(x, 'b (h w) c -> b c h w', h=Wh, w=Ww)  # B C H/4 W/4
        x = self.g_a_down_4xto8x(x)
        x = rearrange(x, 'b c h w -> b (h w) c')    # B (H/8 W/8) C
        _, _, _, x, Wh, Ww = self.g_a_swin_8x(x, Wh // 2, Ww // 2)
        x = rearrange(x, 'b (h w) c -> b c h w', h=Wh, w=Ww)  # B C H/8 W/8
        x = self.g_a_down(x)            # # B C H/16 W/16

        # vit blocks
        _, _, H_l, W_l = x.shape
        # pos embedding
        # pos_emb = F.interpolate(self.pos_embedding, (H_l, W_l), mode='bicubic')
        # x += pos_emb
        x = rearrange(x, 'b c h w -> b (h w) c')        # B (H/16 W/16) C
        y = self.g_a_vit(x, msk)
        y = rearrange(y, 'b (h w) c -> b c h w', h=H_l, w=W_l)        # B C H/16 W/16

        # # just use swin blocks
        # x = rearrange(x, 'b c h w -> b (h w) c')    # B (H/16 W/16) C
        # _, _, _, x, Wh, Ww = self.g_a_vit(x, Wh // 2, Ww // 2)
        # x = rearrange(x, 'b (h w) c -> b c h w', h=Wh, w=Ww)  # B C H/16 W/16
        # y = x

        return y

    def g_s(self, y_hat, msk):
        y_tmp = y_hat
        _, _, H_l, W_l = y_tmp.shape
        y_tmp = rearrange(y_tmp, 'b c h w -> b (h w) c')    # B (H/16 W/16) C

        # vit blocks
        x_hat = self.g_s_vit(y_tmp, msk)                         # B (H/16 W/16) C
        # # just use swin blocks
        # x_hat, _, _, _, _, _ = self.g_s_vit(y_tmp, H_l, W_l)   # B (H/16*W/16) C

        x_hat = rearrange(x_hat, 'b (h w) c -> b c h w', h=H_l, w=W_l)  # B C H/16 W/16
        x_hat = self.g_s_up(x_hat)         # B C H/8 W/8

        _, _, H, W = x_hat.shape
        x_hat = x_hat.flatten(2).transpose(1, 2)    # B (H/8*W/8) C
        x_hat, _, _, _, _, _ = self.g_s_swin_8x(x_hat, H, W)   # B (H/8*W/8) C
        _, _, C = x_hat.shape
        x_hat = x_hat.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()    # B C H/8 W/8
        x_hat = self.g_s_up_8xto4x(x_hat)

        _, _, H, W = x_hat.shape
        x_hat = x_hat.flatten(2).transpose(1, 2)    # B (H/4*W/4) C
        x_hat, _, _, _, _, _ = self.g_s_swin_4x(x_hat, H, W)   # B (H/4*W/4) C
        _, _, C = x_hat.shape
        x_hat = x_hat.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()    # B C H/4 W/4
        x_hat = self.g_s_up_4xto2x(x_hat)

        _, _, H, W = x_hat.shape
        x_hat = x_hat.flatten(2).transpose(1, 2)    # B (H/2*W/2) C
        x_hat, _, _, _, _, _ = self.g_s_swin_2x(x_hat, H, W)   # B (H/2*W/2) C
        _, _, C = x_hat.shape
        x_hat = x_hat.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()    # B C H/2 W/2
        x_hat = self.g_s_final(x_hat)

        return x_hat

    def init_tables(self):
        for m in self.modules():
            if hasattr(m, '_init_tables'):
                m._init_tables()

    def fix_tables(self):
        for m in self.modules():
            if hasattr(m, '_fix_tables'):
                m._fix_tables()


class MeanScaleHyperpriorGroupViT(FactorizedPriorGroupViT):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, hyper_channels=None, in_channel=3, out_channel=3,
            patch_size=2, embed_dims_swin=[128,192,256], embed_dim_vit=320, 
            depths=[2,4,4,2], num_heads=[8,12,16,16],
            window_size=8, mlp_ratio=2, qkv_bias=True, qk_scale=None,
            drop_rate=0., attn_drop_rate=0., 
            norm_layer=nn.LayerNorm, 
            patch_norm=False,   # default is True.
            use_checkpoint=False, use_shift_window=False,
            dim_head=32):
        super().__init__(hyper_channels, in_channel, out_channel,
            patch_size, embed_dims_swin, embed_dim_vit, 
            depths, num_heads,
            window_size, mlp_ratio, qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, 
            norm_layer, 
            patch_norm, 
            use_checkpoint, use_shift_window,
            dim_head)
        # hyper encoder/decoder均为CNN
        y_channel = embed_dim_vit
        m = []
        for i in range(len(hyper_channels)):
            Ci = hyper_channels[i]
            if i == 0:
                m.append(nn.Conv2d(y_channel, Ci, 3, 1, 1))
            else:
                Cim1 = hyper_channels[i - 1]
                m.append(nn.ReLU())
                m.append(nn.Conv2d(Cim1, Ci, 5, 2, 2))
        self.h_a = nn.Sequential(*m)

        m = []
        for i in range(len(hyper_channels))[::-1]:
            Ci = hyper_channels[i]
            if i == 0:
                m.append(nn.Conv2d(Ci, y_channel * 2, 3, 1, 1))
            else:
                Cim1 = hyper_channels[i - 1]
                m.append(nn.ConvTranspose2d(Ci, Cim1, 5, 2, 2, 1))
                m.append(nn.ReLU())
        self.h_s = nn.Sequential(*m)

        scale_min, scale_max, num_scales = 0.11, 256, 64
        offset = math.log(scale_min)
        factor = (math.log(scale_max) - math.log(scale_min))/(num_scales - 1)
        scale_table = torch.exp(offset + factor * torch.arange(num_scales))
        self.y_em = ContinuousConditionalEntropyModel(
            NoisyNormal, param_tables=dict(loc=[0], scale=scale_table.tolist()))
        self.z_em = ContinuousUnconditionalEntropyModel(
            NoisyDeepFactorized(batch_shape=(hyper_channels[-1],)))

    def forward(self, x, noisy=False, keep_bits_batch=False, msk=None, only_rec_fg=False): 
        y = self.g_a(x, msk)
        y = torch.clamp(y, min=-255.5, max=256.49)

        z = self.h_a(y)  
        z_hat, z_indexes = self.quant(z, noisy=noisy)     # entropy model must use noisy input.

        y_means, y_scales = self.h_s(z_hat).chunk(2, 1)
        _, y_indexes = self.quant(y, offset=y_means, noisy=noisy)    # noisy for entropy estimation
        y_hat, _ = self.quant(y, offset=y_means, noisy=False)           # noisy or ste
        y_loc = torch.zeros(1).to(y.device)
        bits = self.y_em(y_indexes, loc=y_loc, scale=y_scales, keep_batch=keep_bits_batch)
        side_bits = self.z_em(z_indexes, keep_batch=keep_bits_batch)

        if only_rec_fg:
            b,c,h,w = y.shape
            y_hat[(msk==0).broadcast_to((b,c,h,w))] = 0

        x_hat = self.g_s(y_hat, msk)

        return {
            "x_hat": x_hat,
            "bits": {"y": bits, "z": side_bits},
        }

    def soft_then_hard(self):
        modules_tobe_fixed = [
            self.patch_embed,
            self.g_a_swin_2x,
            self.g_a_down_2xto4x,
            self.g_a_swin_4x,
            self.g_a_down_4xto8x,
            self.g_a_swin_8x,
            self.g_a_down,
            self.g_a_vit,
            self.h_a, 
        ]
        
        for m in modules_tobe_fixed:
            for p in m.parameters():
                p.requires_grad = False


class ChannelAutoRegressiveGroupViT(MeanScaleHyperpriorGroupViT):
    '''
    Channel Autoregressive Entropy model.
    '''
    def __init__(self, hyper_channels=None, in_channel=3, out_channel=3,
            patch_size=2, embed_dims_swin=[128,192,256], embed_dim_vit=320, 
            depths=[2,4,4,2], num_heads=[8,12,16,16],
            window_size=8, mlp_ratio=2, qkv_bias=True, qk_scale=None,
            drop_rate=0., attn_drop_rate=0., 
            norm_layer=nn.LayerNorm,    # nn.Identity,
            patch_norm=False,   # default is True.
            use_checkpoint=False, use_shift_window=False,
            dim_head=32, 
            splits=10):
        super().__init__(hyper_channels, in_channel, out_channel,
            patch_size, embed_dims_swin, embed_dim_vit, 
            depths, num_heads,
            window_size, mlp_ratio, qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, 
            norm_layer, 
            patch_norm, 
            use_checkpoint, use_shift_window,
            dim_head)
        raise NotImplementedError

        M = embed_dim_vit
        split_channels = M // splits
        assert M % splits == 0
        self.splits = splits
        self.split_channels = split_channels

        # todo : here the implements of Minnen's ChAR and SwinT are different.
        # todo : h_s of Minnen's generate hyper with split_channels channels.
        # todo : h_s of SwinT's generate hyper with hyper_channels[-1].
        # todo : In the future, please make sure which is better.
        # new h_s
        m = []
        for i in range(len(hyper_channels))[::-1]:
            Ci = hyper_channels[i]
            if i == 0:
                m.append(nn.Conv2d(Ci, split_channels*2, 3, 1, 1))
            else:
                Cim1 = hyper_channels[i - 1]
                m.append(nn.ConvTranspose2d(Ci, Cim1, 5, 2, 2, 1))
                m.append(nn.ReLU())
        self.h_s = nn.Sequential(*m)

        self.char_m_means = nn.ModuleList()
        for idx in range(splits):
            # change 3x3conv to 1x1conv?
            self.char_m_means.append(nn.Sequential(
                nn.Conv2d(split_channels*2 + idx*split_channels, M, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(M, M // 2, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(M // 2, split_channels, 3, 1, 1)
            ))
        self.char_m_scales = deepcopy(self.char_m_means)

    def forward(self, x, noisy=False, keep_bits_batch=False, msk=None, only_rec_fg=False): 
        y = self.g_a(x, msk)
        y = torch.clamp(y, min=-255.5, max=256.49)

        z = self.h_a(y)  
        z_hat, z_indexes = self.quant(z, noisy=noisy)     # entropy model must use noisy input.

        y_hyper = self.h_s(z_hat)

        # channel autoregressive
        y_indexes = torch.zeros_like(y)
        y_means, y_scales = [], []
        for idx in range(self.splits):
            if idx == 0:
                y_means_tmp, y_scales_tmp = y_hyper.chunk(2, 1)
            else:
                y_means_tmp = self.char_m_means[idx](
                    torch.cat([y_hyper, y_indexes[:,:idx*self.split_channels,...]], dim=1))
                y_scales_tmp = self.char_m_scales[idx](
                    torch.cat([y_hyper, y_indexes[:,:idx*self.split_channels,...]], dim=1))
            y_tmp = y[:,idx*self.split_channels:(idx+1)*self.split_channels,...]
            _, y_indexes[:,idx*self.split_channels:(idx+1)*self.split_channels,...] = self.quant(
                y_tmp, offset=y_means_tmp, noisy=noisy)
            
            y_means.append(y_means_tmp)
            y_scales.append(y_scales_tmp)

        y_means = torch.cat(y_means, dim=1)
        y_scales = torch.cat(y_scales, dim=1)

        _, y_indexes = self.quant(y, offset=y_means, noisy=noisy)    # noisy for entropy estimation
        y_hat, _ = self.quant(y, offset=y_means, noisy=False)           # noisy or ste
        y_loc = torch.zeros(1).to(x.device)
        bits = self.y_em(y_indexes, loc=y_loc, scale=y_scales, keep_batch=keep_bits_batch)
        side_bits = self.z_em(z_indexes, keep_batch=keep_bits_batch)

        if only_rec_fg:
            b,c,h,w = y.shape
            y_hat[(msk==0).broadcast_to((b,c,h,w))] = 0 # 将y_hat中msk==0的地方设置为0

        x_hat = self.g_s(y_hat, msk)

        return {
            "x_hat": x_hat,
            "bits": {"y": bits, "z": side_bits},
        }


## ------------------------------------------
## --- Transformer Based Transform Coding ---
## ------------------------------------------
# Original Paper: https://openreview.net/pdf?id=IDwN6xjHnK8
# Referenced code base: https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation

class TransformerBasedTransformCodingHyper(nn.Module):
    def __init__(self, 
            in_channel=3, out_channel=3,
            embed_dim_g=[128,192,256,320], depths_g=[2,2,5,1], num_heads_g=[8,12,16,16],
            embed_dim_h=[192,192], depths_h=[5,1], num_heads_h=[12,12],
            window_size_g=8, window_size_h=4, 
            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, 
            patch_norm=False,   # default: True
            use_checkpoint=False):
        super().__init__()

        self.patch_norm = patch_norm

        # . encoder
        # transform
        self.g_a_pe = nn.ModuleList()  # pe = patch embed
        self.g_a_m = nn.ModuleList()
        in_channels = [in_channel] + embed_dim_g[:-1]
        for i in range(len(embed_dim_g)):
            patchembed = PatchEmbed(
                patch_size=2, in_chans=in_channels[i], embed_dim=embed_dim_g[i],
                norm_layer=norm_layer if self.patch_norm else None)
            self.g_a_pe.append(patchembed)

            layer = BasicLayer(
                dim=embed_dim_g[i],
                depth=depths_g[i],
                num_heads=num_heads_g[i],
                window_size=window_size_g,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint)
            self.g_a_m.append(layer)
        
        # hyper - conv based
        y_channel = embed_dim_g[-1]
        embed_dim_h = embed_dim_h + [embed_dim_h[-1]]
        m = []
        for i in range(len(embed_dim_h)):
            Ci = embed_dim_h[i]
            if i == 0:
                m.append(nn.Conv2d(y_channel, Ci, 3, 1, 1))
            else:
                Cim1 = embed_dim_h[i - 1]
                m.append(nn.ReLU())
                m.append(nn.Conv2d(Cim1, Ci, 5, 2, 2))
        self.h_a_m = nn.Sequential(*m)

        # .decoder
        # transform
        self.g_s_m = nn.ModuleList()
        self.g_s_pe = nn.ModuleList()  # pe = patch embed
        out_channels = embed_dim_g[::-1][1:] + [out_channel]
        for i in range(len(embed_dim_g)):
            layer = BasicLayer(
                dim=embed_dim_g[::-1][i],
                depth=depths_g[::-1][i],
                num_heads=num_heads_g[::-1][i],
                window_size=window_size_g,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint)
            self.g_s_m.append(layer)

            norm_ = norm_layer if self.patch_norm else None
            # if i == len(embed_dim_g) - 1:   # todo : should there be a norm layer for the final un-sampling?
            #     norm_ = None
            patchembed = PatchUnEmbed(
                patch_size=2, in_chans=embed_dim_g[::-1][i], embed_dim=out_channels[i],
                norm_layer=norm_)
            self.g_s_pe.append(patchembed)

        # hyper - conv based
        m = []
        for i in range(len(embed_dim_h))[::-1]:
            Ci = embed_dim_h[i]
            if i == 0:
                m.append(nn.Conv2d(Ci, y_channel * 2, 3, 1, 1))
            else:
                Cim1 = embed_dim_h[i - 1]
                m.append(nn.ConvTranspose2d(Ci, Cim1, 5, 2, 2, 1))
                m.append(nn.ReLU())
        self.h_s_m = nn.Sequential(*m)

        # .entropy model
        scale_min, scale_max, num_scales = 0.11, 256, 64
        offset = math.log(scale_min)
        factor = (math.log(scale_max) - math.log(scale_min))/(num_scales - 1)
        scale_table = torch.exp(offset + factor * torch.arange(num_scales))
        self.y_em = ContinuousConditionalEntropyModel(
            NoisyNormal, param_tables=dict(loc=[0], scale=scale_table.tolist()))
        self.z_em = ContinuousUnconditionalEntropyModel(
            NoisyDeepFactorized(batch_shape=(embed_dim_h[-1],)))

        self.quant = UniformQuantization(step=1)

    def forward(self, x, noisy=True, keep_bits_batch=False, msk=None, only_rec_fg=False):
        y = self.g_a(x)
        y = torch.clamp(y, min=-255.5, max=256.49)

        z = self.h_a(y)  
        z_hat, z_indexes = self.quant(z, noisy=noisy)     # entropy model must use noisy input.

        y_means, y_scales = self.h_s(z_hat).chunk(2, 1)
        _, y_indexes = self.quant(y, offset=y_means, noisy=noisy)    # noisy for entropy estimation
        y_hat, _ = self.quant(y, offset=y_means, noisy=False)           # noisy or ste
        y_loc = torch.zeros(1).to(y.device)
        # bits = self.y_em(y_indexes, loc=y_loc, scale=y_scales, keep_batch=keep_bits_batch)
        bits, log_probs = self.y_em(
            y_indexes, draw=True, loc=y_loc, scale=y_scales, keep_batch=keep_bits_batch)
        side_bits = self.z_em(z_indexes, keep_batch=keep_bits_batch)

        if only_rec_fg:
            b,c,h,w = y.shape
            y_hat[(msk==0).broadcast_to((b,c,h,w))] = 0

        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "bits": {"y": bits, "z": side_bits},
            'log_probs': log_probs
        }

    def g_a(self, x):
        for pe, layer in zip(self.g_a_pe, self.g_a_m):
            x = pe(x)
            _, _, Wh, Ww = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
            _, _, _, x, Wh, Ww = layer(x, Wh, Ww)
            x = rearrange(x, 'b (h w) c -> b c h w', h=Wh, w=Ww)
        return x

    def g_s(self, x):
        for pe, layer in zip(self.g_s_pe, self.g_s_m):
            _, _, Wh, Ww = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
            _, _, _, x, Wh, Ww = layer(x, Wh, Ww)
            x = rearrange(x, 'b (h w) c -> b c h w', h=Wh, w=Ww)
            x = pe(x)
        return x

    def h_a(self, x):
        # hyper - conv based
        x = self.h_a_m(x)
        return x

    def h_s(self, x):
        # hyper - conv based
        x = self.h_s_m(x)
        return x

    def init_tables(self):
        for m in self.modules():
            if hasattr(m, '_init_tables'):
                m._init_tables()

    def fix_tables(self):
        for m in self.modules():
            if hasattr(m, '_fix_tables'):
                m._fix_tables()
        
    def soft_then_hard(self):
        modules_tobe_fixed = [
            self.g_a_pe,
            self.g_a_m,
            self.h_a_m,
        ]
        
        for m in modules_tobe_fixed:
            for p in m.parameters():
                p.requires_grad = False
    
    def freeze_transform(self):
        modules_tobe_fixed = [
            self.g_a_pe,
            self.g_a_m,
            self.g_s_pe,
            self.g_s_m,
        ]
        
        for m in modules_tobe_fixed:
            for p in m.parameters():
                p.requires_grad = False


class TransformerBasedTransformCodingChAR(TransformerBasedTransformCodingHyper):
    def __init__(self, 
        in_channel=3, out_channel=3,
        embed_dim_g=[128,192,256,320], depths_g=[2,2,5,1], num_heads_g=[8,12,16,16],
        embed_dim_h=[192,192], depths_h=[5,1], num_heads_h=[12,12],
        window_size_g=8, window_size_h=4, 
        mlp_ratio=4, qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, 
        patch_norm=False,   # default: True
        use_checkpoint=False, 
        splits=10
        ):
        super().__init__(in_channel, out_channel, embed_dim_g, depths_g, num_heads_g,
            embed_dim_h, depths_h, num_heads_h, window_size_g, window_size_h, 
            mlp_ratio, qkv_bias, qk_scale, norm_layer, patch_norm, use_checkpoint)
        
        # raise NotImplementedError
        M = embed_dim_g[-1]
        split_channels = M // splits
        assert M % splits == 0
        self.splits = splits
        self.split_channels = split_channels

        self.char_m_means = nn.ModuleList()
        for idx in range(splits):
            self.char_m_means.append(nn.Sequential(
                nn.Conv2d(split_channels*2 + idx*split_channels, M, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(M, M // 2, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(M // 2, split_channels, 3, 1, 1)
            ))
        self.char_m_scales = deepcopy(self.char_m_means)

    def forward(self, x, noisy=True, keep_bits_batch=False, msk=None, only_rec_fg=False):
        y = self.g_a(x)
        y = torch.clamp(y, min=-255.5, max=256.49)

        z = self.h_a(y)  
        z_hat, z_indexes = self.quant(z, noisy=noisy)     # entropy model must use noisy input.

        y_hyper = self.h_s(z_hat)

        # channel autoregressive
        y_indexes = torch.zeros_like(y)
        y_means, y_scales = [], []
        for idx in range(self.splits):
            y_hyper_tmp = \
                y_hyper[:,idx*self.split_channels*2:(idx+1)*self.split_channels*2,...]
            y_means_tmp = self.char_m_means[idx](
                    torch.cat([y_hyper_tmp, y_indexes[:,:idx*self.split_channels,...]], dim=1))
            y_scales_tmp = self.char_m_scales[idx](
                    torch.cat([y_hyper_tmp, y_indexes[:,:idx*self.split_channels,...]], dim=1))

            y_tmp = y[:,idx*self.split_channels:(idx+1)*self.split_channels,...]
            _, y_indexes[:,idx*self.split_channels:(idx+1)*self.split_channels,...] = self.quant(
                y_tmp, offset=y_means_tmp, noisy=noisy)
            y_means.append(y_means_tmp)
            y_scales.append(y_scales_tmp)

        y_means = torch.cat(y_means, dim=1)
        y_scales = torch.cat(y_scales, dim=1)

        _, y_indexes = self.quant(y, offset=y_means, noisy=noisy)    # noisy for entropy estimation
        y_hat, _ = self.quant(y, offset=y_means, noisy=False)           # noisy or ste
        y_loc = torch.zeros(1).to(y.device)
        bits = self.y_em(y_indexes, loc=y_loc, scale=y_scales, keep_batch=keep_bits_batch)
        side_bits = self.z_em(z_indexes, keep_batch=keep_bits_batch)


        if only_rec_fg:
            b,c,h,w = y.shape
            y_hat[(msk==0).broadcast_to((b,c,h,w))] = 0
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "bits": {"y": bits, "z": side_bits},
        }


## ----------------------------------------------------------------
## --- Group Attention based Transformer Based Transform Coding ---
## ----------------------------------------------------------------
class GroupSwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix_normal, mask_matrix_shift):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix_normal: Attention mask for normal attention.
            mask_matrix: Attention mask for attention with cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix_shift
        else:
            shifted_x = x
            attn_mask = mask_matrix_normal

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class GroupBasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 use_shift_window=True):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            GroupSwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else (window_size // 2)*int(use_shift_window), 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W, group_mask=None):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        def vis_msk(x, p):
            x = torch.permute(x, (0,3,1,2)).float()
            x = (x - x.min()) / (x.max() - x.min())
            torchvision.utils.save_image(x, p)

        group_mask_normal = F.interpolate(group_mask.float(), (Hp, Wp), mode='nearest')
        group_mask_normal = torch.permute(group_mask_normal, (0,2,3,1)).to(img_mask.device).int()
        # generate attn_mask_normal
        mask_windows_normal = window_partition(group_mask_normal, self.window_size)  # nW, window_size, window_size, 1
        # vis_msk(mask_windows_normal, 'mask_windows_normal.png')
        mask_windows_normal = mask_windows_normal.view(-1, self.window_size * self.window_size)
        attn_mask_normal = mask_windows_normal.unsqueeze(1) - mask_windows_normal.unsqueeze(2)
        attn_mask_normal = attn_mask_normal.masked_fill(
            attn_mask_normal != 0, float(-100.0)).masked_fill(attn_mask_normal == 0, float(0.0))
        # vis_msk(group_mask_normal, 'group_mask_normal.png')
        # vis_msk(attn_mask_normal.unsqueeze(3).float(), 'attn_mask_normal.png')

        # generate attn_mask_shift
        group_mask_shift = torch.roll(group_mask_normal, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        img_mask = repeat(img_mask, 'b h w c -> (repeat b) h w c', repeat=group_mask_shift.shape[0]).contiguous()
        img_mask += group_mask_shift.clone()
        mask_windows_shift = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        # vis_msk(mask_windows_shift, 'mask_windows_shift.png')
        mask_windows_shift = mask_windows_shift.view(-1, self.window_size * self.window_size)
        attn_mask_shift = mask_windows_shift.unsqueeze(1) - mask_windows_shift.unsqueeze(2)
        attn_mask_shift = attn_mask_shift.masked_fill(
            attn_mask_shift != 0, float(-100.0)).masked_fill(attn_mask_shift == 0, float(0.0))
        # print(group_mask.shape, attn_mask_shift.shape)
        # vis_msk(group_mask_shift, 'group_mask_shift.png')
        # vis_msk(img_mask, 'img_mask_shift.png')
        # vis_msk(attn_mask_shift.unsqueeze(3).float(), 'attn_mask_shift.png')

        # if 32 in img_mask.shape:
        #     # raise
        #     if 8 in img_mask.shape:
        #         raise
        # raise

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                # x = checkpoint.checkpoint(blk, x, attn_mask)
                x = checkpoint.checkpoint(blk, x, attn_mask_normal, attn_mask_shift)
            else:
                x = blk(x, attn_mask_normal, attn_mask_shift)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class GroupTransformerBasedTransformCodingHyper(nn.Module):
    def __init__(self, 
            in_channel=3, out_channel=3,
            embed_dim_g=[128,192,256,320], depths_g=[2,2,5,1], num_heads_g=[8,12,16,16],
            embed_dim_h=[192,192], depths_h=[5,1], num_heads_h=[12,12],
            window_size_g=8, window_size_h=4, 
            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, 
            patch_norm=False,   # default: True
            use_checkpoint=False):
        super().__init__()


        self.patch_norm = patch_norm

        # . encoder
        # transform
        self.g_a_pe = nn.ModuleList()  # pe = patch embed
        self.g_a_m = nn.ModuleList()
        in_channels = [in_channel] + embed_dim_g[:-1]
        for i in range(len(embed_dim_g)):
            patchembed = PatchEmbed(
                patch_size=2, in_chans=in_channels[i], embed_dim=embed_dim_g[i],
                norm_layer=norm_layer if self.patch_norm else None)
            self.g_a_pe.append(patchembed)

            layer = GroupBasicLayer(
                dim=embed_dim_g[i],
                depth=depths_g[i],
                num_heads=num_heads_g[i],
                window_size=window_size_g,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint)
            self.g_a_m.append(layer)
        
        # hyper - conv based
        y_channel = embed_dim_g[-1]
        embed_dim_h = embed_dim_h + [embed_dim_h[-1]]
        m = []
        for i in range(len(embed_dim_h)):
            Ci = embed_dim_h[i]
            if i == 0:
                m.append(nn.Conv2d(y_channel, Ci, 3, 1, 1))
            else:
                Cim1 = embed_dim_h[i - 1]
                m.append(nn.ReLU())
                m.append(nn.Conv2d(Cim1, Ci, 5, 2, 2))
        self.h_a_m = nn.Sequential(*m)

        # .decoder
        # transform
        self.g_s_m = nn.ModuleList()
        self.g_s_pe = nn.ModuleList()  # pe = patch embed
        out_channels = embed_dim_g[::-1][1:] + [out_channel]
        for i in range(len(embed_dim_g)):
            layer = GroupBasicLayer(
                dim=embed_dim_g[::-1][i],
                depth=depths_g[::-1][i],
                num_heads=num_heads_g[::-1][i],
                window_size=window_size_g,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint)
            self.g_s_m.append(layer)

            norm_ = norm_layer if self.patch_norm else None
            # if i == len(embed_dim_g) - 1:   # todo : should there be a norm layer for the final un-sampling?
            #     norm_ = None
            patchembed = PatchUnEmbed(
                patch_size=2, in_chans=embed_dim_g[::-1][i], embed_dim=out_channels[i],
                norm_layer=norm_)
            self.g_s_pe.append(patchembed)

        # hyper - conv based
        m = []
        for i in range(len(embed_dim_h))[::-1]:
            Ci = embed_dim_h[i]
            if i == 0:
                m.append(nn.Conv2d(Ci, y_channel * 2, 3, 1, 1))
            else:
                Cim1 = embed_dim_h[i - 1]
                m.append(nn.ConvTranspose2d(Ci, Cim1, 5, 2, 2, 1))
                m.append(nn.ReLU())
        self.h_s_m = nn.Sequential(*m)

        # .entropy model
        scale_min, scale_max, num_scales = 0.11, 256, 64
        offset = math.log(scale_min)
        factor = (math.log(scale_max) - math.log(scale_min))/(num_scales - 1)
        scale_table = torch.exp(offset + factor * torch.arange(num_scales))
        self.y_em = ContinuousConditionalEntropyModel(
            NoisyNormal, param_tables=dict(loc=[0], scale=scale_table.tolist()))
        self.z_em = ContinuousUnconditionalEntropyModel(
            NoisyDeepFactorized(batch_shape=(embed_dim_h[-1],)))

        self.quant = UniformQuantization(step=1)

    def forward(self, x, noisy=True, keep_bits_batch=False, msk=None, only_rec_fg=False, encrypt_msk=None):
        y = self.g_a(x, msk)
        y = torch.clamp(y, min=-255.5, max=256.49)

        z = self.h_a(y)  
        z_hat, z_indexes = self.quant(z, noisy=noisy)     # entropy model must use noisy input.

        y_means, y_scales = self.h_s(z_hat).chunk(2, 1)
        _, y_indexes = self.quant(y, offset=y_means, noisy=noisy)    # noisy for entropy estimation
        y_hat, _ = self.quant(y, offset=y_means, noisy=False)           # noisy or ste
        y_loc = torch.zeros(1).to(y.device)
        # bits = self.y_em(y_indexes, loc=y_loc, scale=y_scales, keep_batch=keep_bits_batch)
        bits, log_probs = self.y_em(
            y_indexes, draw=True, loc=y_loc, scale=y_scales, keep_batch=keep_bits_batch)
        side_bits = self.z_em(z_indexes, keep_batch=keep_bits_batch)

        if only_rec_fg:
            b,c,h,w = y.shape
            y_hat[(msk==0).broadcast_to((b,c,h,w))] = 0

        if encrypt_msk is not None:
            encrypt_msk = torch.from_numpy(encrypt_msk).bool()
            encrypt_msk = einops.repeat(encrypt_msk, 'h w -> b c h w', b=y_hat.shape[0], c=y_hat.shape[1])
            y_hat[encrypt_msk] = y_hat[encrypt_msk][torch.randperm(y_hat[encrypt_msk].size(0))]

        x_hat = self.g_s(y_hat, msk)

        return {
            "x_hat": x_hat,
            "bits": {"y": bits, "z": side_bits},
            'log_probs': log_probs
        }

    def g_a(self, x, msk):
        for pe, layer in zip(self.g_a_pe, self.g_a_m):
            x = pe(x)
            _, _, Wh, Ww = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
            _, _, _, x, Wh, Ww = layer(x, Wh, Ww, group_mask=msk)
            x = rearrange(x, 'b (h w) c -> b c h w', h=Wh, w=Ww)
        return x

    def g_s(self, x, msk):
        for pe, layer in zip(self.g_s_pe, self.g_s_m):
            _, _, Wh, Ww = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
            _, _, _, x, Wh, Ww = layer(x, Wh, Ww, group_mask=msk)
            x = rearrange(x, 'b (h w) c -> b c h w', h=Wh, w=Ww)
            x = pe(x)
        return x

    def h_a(self, x):
        # hyper - conv based
        x = self.h_a_m(x)
        return x

    def h_s(self, x):
        # hyper - conv based
        x = self.h_s_m(x)
        return x

    def init_tables(self):
        for m in self.modules():
            if hasattr(m, '_init_tables'):
                m._init_tables()

    def fix_tables(self):
        for m in self.modules():
            if hasattr(m, '_fix_tables'):
                m._fix_tables()
        

class GroupChARTTC(GroupTransformerBasedTransformCodingHyper):
    def __init__(self, 
                in_channel=3, out_channel=3, 
                embed_dim_g=[128, 192, 256, 320], depths_g=[2, 2, 5, 1], num_heads_g=[8, 12, 16, 16], 
                embed_dim_h=[192, 192], depths_h=[5, 1], num_heads_h=[12, 12], 
                window_size_g=8, window_size_h=4, 
                mlp_ratio=4, qkv_bias=True, qk_scale=None, 
                norm_layer=nn.LayerNorm, 
                patch_norm=False, 
                use_checkpoint=False,
                splits=10):
        super().__init__(in_channel, out_channel, embed_dim_g, 
                        depths_g, num_heads_g, embed_dim_h, 
                        depths_h, num_heads_h, window_size_g, 
                        window_size_h, mlp_ratio, qkv_bias, 
                        qk_scale, norm_layer, patch_norm, use_checkpoint)

        M = embed_dim_g[-1]
        self.splits = splits
        split_channels = M // splits
        assert M % splits == 0
        self.splits = splits
        self.split_channels = split_channels

        self.char_m_means = nn.ModuleList()
        for idx in range(splits):
            self.char_m_means.append(nn.Sequential(
                # nn.Conv2d(split_channels*2 + idx*split_channels, M, 3, 1, 1),
                nn.Conv2d(split_channels*2 + idx*split_channels, M, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(M, M, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(M, split_channels, 1, 1, 0)
            ))
        self.char_m_scales = deepcopy(self.char_m_means)

    def forward(self, x, noisy=True, keep_bits_batch=False, msk=None, only_rec_fg=False, encrypt_msk=None):
        y = self.g_a(x, msk)
        y = torch.clamp(y, min=-255.5, max=256.49)

        z = self.h_a(y)  
        z_hat, z_indexes = self.quant(z, noisy=noisy)     # entropy model must use noisy input.
        y_hyper = self.h_s(z_hat)

        # channel autoregressive
        y_slices = y.chunk(self.splits, dim=1)
        y_hyper_slices = y_hyper.chunk(self.splits, dim=1)
        y_hat_slices, bits, log_probs = [], [], []
        for idx in range(self.splits):
            y_slice = y_slices[idx]
            y_hyper_slice = y_hyper_slices[idx]
            y_context_slices = y_hat_slices[:idx]
            support = torch.cat([y_hyper_slice] + y_context_slices, dim=1)

            y_mean = self.char_m_means[idx](support)
            y_scale = self.char_m_scales[idx](support)
            _, y_indexes = self.quant(y_slice, offset=y_mean, noisy=noisy)
            y_hat_slice, _ = self.quant(y_slice, offset=y_mean, noisy=False)  # STE
            y_loc = torch.zeros(1).to(x.device)
            # slice_bits = self.y_em(y_indexes, loc=y_loc, scale=y_scale)
            slice_bits, slice_log_probs = self.y_em(
                y_indexes, draw=True, loc=y_loc, scale=y_scale)
            bits.append(slice_bits)
            y_hat_slices.append(y_hat_slice)
            log_probs.append(slice_log_probs)
        bits = sum(bits)
        y_hat = torch.cat(y_hat_slices, dim=1)
        log_probs = torch.cat(log_probs, dim=1)

        side_bits = self.z_em(z_indexes, keep_batch=keep_bits_batch)

        if only_rec_fg:
            b,c,h,w = y.shape
            y_hat[(msk==0).broadcast_to((b,c,h,w))] = 0

        if encrypt_msk is not None:
            encrypt_msk = torch.from_numpy(encrypt_msk).bool()
            encrypt_msk = einops.repeat(encrypt_msk, 'h w -> b c h w', b=y_hat.shape[0], c=y_hat.shape[1])
            y_hat[encrypt_msk] = y_hat[encrypt_msk][torch.randperm(y_hat[encrypt_msk].size(0))]

        x_hat = self.g_s(y_hat, msk)

        return {
            "x_hat": x_hat,
            "bits": {"y": bits, "z": side_bits},
            'log_probs': log_probs
        }



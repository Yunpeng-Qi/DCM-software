import torch
import torch.nn as nn
import torch.nn.functional as F

import math, sys
from pathlib import Path

from data_compression.entropy_models import ContinuousUnconditionalEntropyModel, \
    ContinuousConditionalEntropyModel
from data_compression.distributions.uniform_noised import NoisyNormal
from data_compression.distributions.uniform_noised import NoisyDeepFactorized
from data_compression.quantization import UniformQuantization
from data_compression.layers import ResBlock, ResBlocks, Downsample, Upsample
from data_compression.layers.sft import SFT, SFTResblk, QmapFuse, QmapDownsample, QmapUpsample


class RDT_CheckerCube(RDT_Checkerboard):
    def __init__(self, args):
        super().__init__(args)
        self.M = args.transform_channels[-1]

    def forward(self, x, noisy=True, trainer=None, return_y=False):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_indexes = self.quant(z, noisy=noisy)
        y_hyper = self.h_s(z_hat)

        # checkercube mask generation
        H, W = y.shape[2:]
        mask_checkboard = torch.ones(1, 4, H // 2, W // 2).to(y.device)
        mask_checkboard[:, 1:3, :, :] = 0
        mask_checkboard = F.pixel_shuffle(mask_checkboard, 2)  # mask = 0, the first layer
        mask_CheckerCube_layer = torch.cat([mask_checkboard, 1 - mask_checkboard], dim=1)
        mask_CheckerCube = mask_CheckerCube_layer.repeat(1, self.M // 2, 1, 1)

        # First layer
        ## hyper only
        y_context_l1 = torch.cat((y_hyper, torch.zeros_like(y_hyper)), dim=1)
        y_mean_l1, y_scale_l1 = self.entropy_parameters(y_context_l1).chunk(2, 1)

        # Second layer
        y_hat_l1, _ = self.quant(y, noisy=noisy)
        y_ar_l2 = self.context_model(y_hat_l1 * (1 - mask_CheckerCube))
        y_context_l2 = torch.cat((y_hyper, y_ar_l2), dim=1)
        y_mean_l2, y_scale_l2 = self.entropy_parameters(y_context_l2).chunk(2, 1)

        y_means = (1 - mask_CheckerCube) * y_mean_l1 + mask_CheckerCube * y_mean_l2
        y_scales = (1 - mask_CheckerCube) * y_scale_l1 + mask_CheckerCube * y_scale_l2

        y_hat, y_indexes = self.quant(y, noisy=noisy)
        y_loc = torch.zeros(1).to(x.device)
        bits = self.y_em(y_indexes - y_means, loc=y_loc, scale=y_scales)
        side_bits = self.z_em(z_indexes)
        x_hat = self.g_s(y_hat)

        return x_hat, bits, side_bits
        # if trainer is not None:
        #     assert x.shape == x_hat.shape
        #     B, C, H, W = x.shape
        #     distortion = trainer['distortion_fn'](x, x_hat)
        #     bpp = (bits + side_bits) / (B * H * W)
        #     loss = bpp + distortion * self.lmbda
        #     trainer['optimizer'].zero_grad()
        #     loss.backward()
        #     trainer['optimizer'].step()
        # elif return_y:
        #     return x_hat, y_hat, bits, side_bits
        # else:
        #     return x_hat, bits, side_bits





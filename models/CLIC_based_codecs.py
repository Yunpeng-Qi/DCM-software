import imp
import math
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import einops
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchac

from data_compression.distributions.uniform_noised import (NoisyDeepFactorized,
                                                           NoisyNormal)
from data_compression.entropy_models import (
    ContinuousConditionalEntropyModel, ContinuousUnconditionalEntropyModel)
from data_compression.layers import (SFT, Downsample, QmapDownsample, QmapFuse,
                                     QmapUpsample, ResAttn, ResAttnSplit,
                                     ResBlock, ResBlockDown, ResBlockUp,
                                     SFTResblk, Upsample)
from data_compression.quantization import UniformQuantization
from data_compression.prob import GaussianConditional, LogisticConditional
from data_compression.layers.gdn import GDN

class Spatially_adaptive_ImgComNet(nn.Module):
    def __init__(self, args, inplace=False, prior_nc=64, sft_ks=3):
        super().__init__()
        self.N = args.N
        self.M = args.M

        self.qmap_ga1 = nn.Sequential(
            nn.Conv2d(4, prior_nc * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(prior_nc * 4, prior_nc * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(prior_nc * 2, prior_nc, kernel_size=3, stride=1, padding=1),
        )
        self.qmap_ga2 = QmapDownsample(prior_nc, prior_nc)
        self.qmap_ga3 = QmapDownsample(prior_nc, prior_nc)
        self.qmap_ga4 = QmapDownsample(prior_nc, prior_nc)

        self.ga_sft1 = SFT(self.N, prior_nc, sft_ks)
        self.ga_sft2 = SFT(self.N, prior_nc, sft_ks)
        self.ga_sft3 = SFT(self.N, prior_nc, sft_ks)
        self.ga_sft_res1 = SFTResblk(self.M, prior_nc, ks=sft_ks)
        self.ga_sft_res2 = SFTResblk(self.M, prior_nc, ks=sft_ks)

        self.g_analysis = nn.Sequential(OrderedDict([
            ('conv_down1', ResBlockDown(3, self.N)),
            ('down_attn1', ResAttnSplit(self.N, self.N, groups=2)),
            ('conv_res1', ResBlock(self.N, self.N)),
            ('conv_down2', ResBlockDown(self.N, self.N)),
            ('down_attn2', ResAttnSplit(self.N, self.N, groups=2)),
            ('conv_res2', ResBlock(self.N, self.N)),
            ('conv_down3', ResBlockDown(self.N, self.N)),
            ('down_attn3', ResAttnSplit(self.N, self.N, groups=2)),
            ('conv_res3', ResBlock(self.N, self.N)),
            ('conv_down4', Downsample(self.N, self.M)),
            ('down_attn4', ResAttnSplit(self.M, self.M, groups=2)),
        ]))
        
        self.qmap_generate = nn.Sequential(
            nn.ConvTranspose2d(self.N, self.N//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(self.N//2, self.N//4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.N//4, self.N//4, kernel_size=3, stride=1, padding=1)
        )

        self.qmap_gs1 = nn.Sequential(
            nn.Conv2d(self.M + self.N // 4, prior_nc * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(prior_nc * 4, prior_nc * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(prior_nc * 2, prior_nc, kernel_size=3, stride=1, padding=1)
        )
        self.qmap_gs2 = QmapUpsample(prior_nc, prior_nc)
        self.qmap_gs3 = QmapUpsample(prior_nc, prior_nc)
        self.qmap_gs4 = QmapUpsample(prior_nc, prior_nc)

        self.gs_sft_res1 = SFTResblk(self.M, prior_nc, ks=sft_ks)
        self.gs_sft_res2 = SFTResblk(self.M, prior_nc, ks=sft_ks)
        self.gs_sft1 = SFT(self.N, prior_nc, ks=sft_ks)
        self.gs_sft2 = SFT(self.N, prior_nc, ks=sft_ks)
        self.gs_sft3 = SFT(self.N, prior_nc, ks=sft_ks)

        self.g_synthesis = nn.Sequential(OrderedDict([
            ('up_attn2', ResAttnSplit(self.M, self.M, groups=2)),
            ('conv_res4', ResBlock(self.M, self.M)),
            ('conv_up4', ResBlockUp(self.M, self.N)),
            ('conv_res3', ResBlock(self.N, self.N)),
            ('conv_up3', ResBlockUp(self.N, self.N)),
            ('up_attn1', ResAttnSplit(self.N, self.N, groups=2)),
            ('conv_res2', ResBlock(self.N, self.N)),
            ('conv_up2', ResBlockUp(self.N, self.N)),
            ('conv_res1', ResBlock(self.N, self.N)),
            ('conv_up1', Upsample(self.N, 3)),
        ]))

        self.h_analysis = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.M, self.N, kernel_size=3, stride=1, padding=1)),
            ('relu1', nn.LeakyReLU(inplace=inplace)),
            ('conv2', nn.Conv2d(self.N, self.N, kernel_size=5, stride=2, padding=2)),
            ('relu2', nn.LeakyReLU(inplace=inplace)),
            ('conv3', nn.Conv2d(self.N, self.N, kernel_size=5, stride=2, padding=2)),
            ('down_attn', ResAttn(self.N, self.N)),
        ]))

        self.h_synthesis = nn.Sequential(OrderedDict([
            ('unconv1', nn.ConvTranspose2d(self.N, self.M, kernel_size=5, stride=2, padding=2, output_padding=1)),
            ('relu1', nn.LeakyReLU(inplace=inplace)),
            ('unconv2', nn.ConvTranspose2d(self.M, self.M + self.M // 2, kernel_size=5, stride=2, padding=2, output_padding=1)),
            ('relu2', nn.LeakyReLU(inplace=inplace)),
            ('unconv3', nn.ConvTranspose2d(self.M + self.M // 2, self.M * 2, kernel_size=3, stride=1, padding=1)),
            # ('resblock', ResBlock(M * 2, M * 2)),
        ]))

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(self.M * 12 // 3, self.M * 10 // 3, 1, 1, 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.M * 10 // 3, self.M * 8 // 3, 1, 1, 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.M * 8 // 3, self.M * 6 // 3, 1, 1, 0),
        )

        self.context_model_layer2 = nn.Sequential(
            nn.Conv2d(self.M, self.M, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.M, self.M, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.M, self.M * 2, 3, 1, 1)
        )

        self.quant = UniformQuantization(step=1)
        
        scale_min, scale_max, num_scales = 0.11, 256, 64
        offset = math.log(scale_min)
        factor = (math.log(scale_max) - math.log(scale_min))/(num_scales - 1)
        scale_table = torch.exp(offset + factor * torch.arange(num_scales))
        self.y_em = ContinuousConditionalEntropyModel(
            NoisyNormal, param_tables=dict(loc=[0], scale=scale_table.tolist()))
        self.z_em = ContinuousUnconditionalEntropyModel(
            NoisyDeepFactorized(batch_shape=(self.N,)))

        self.debug = False
    
    def forward(self, x, qmap, noisy=True, keep_bits_batch=False):
        x_enc = x * 2 - 1
        # y = self.g_a(x)
        qmap_enc = self.qmap_ga1(torch.cat([qmap, x_enc], dim=1))
        # qmap_enc = self.qmap_fuse_enc(torch.cat([qmap, x_enc], dim=1))
        # qmap_enc = self.qmap_ga1(qmap_enc)
        x_enc = self.g_analysis.conv_down1(x_enc)
        x_enc = self.g_analysis.down_attn1(x_enc)
        x_enc = self.g_analysis.conv_res1(x_enc)
        x_enc = self.ga_sft1(x_enc, qmap_enc)

        qmap_enc = self.qmap_ga2(qmap_enc)
        x_enc = self.g_analysis.conv_down2(x_enc)
        x_enc = self.g_analysis.down_attn2(x_enc)
        x_enc = self.g_analysis.conv_res2(x_enc)
        x_enc = self.ga_sft2(x_enc, qmap_enc)

        qmap_enc = self.qmap_ga3(qmap_enc)
        x_enc = self.g_analysis.conv_down3(x_enc)
        x_enc = self.g_analysis.down_attn3(x_enc)
        x_enc = self.g_analysis.conv_res3(x_enc)
        x_enc = self.ga_sft3(x_enc, qmap_enc)

        qmap_enc = self.qmap_ga4(qmap_enc)    
        x_enc = self.g_analysis.conv_down4(x_enc)
        x_enc = self.g_analysis.down_attn4(x_enc)
        x_enc = self.ga_sft_res1(x_enc, qmap_enc)
        y = self.ga_sft_res2(x_enc, qmap_enc)
        y = torch.clamp(y, min=-255.5, max=256.49)

        z = self.h_analysis(y)
        z_hat, z_indexes = self.quant(z, noisy=noisy)
        y_hyper = self.h_synthesis(z_hat)

        # 3D checkboard mask generation
        H, W = y.shape[2:]
        mask_checkboard = torch.ones(1, 4, H // 2, W // 2).to(x.device)
        mask_checkboard[:, 1:3, :, :] = 0
        mask_checkboard = F.pixel_shuffle(mask_checkboard, 2) # mask = 0, the first layer
        ThreeD_mask_checkboard_layer = torch.cat([mask_checkboard, 1-mask_checkboard], dim=1)
        ThreeD_mask_checkboard = ThreeD_mask_checkboard_layer.repeat(1, self.M // 2, 1, 1)

        # First layer
        ## hyper only
        y_context_l1 = torch.cat((y_hyper, torch.zeros_like(y_hyper)), dim=1)
        y_mean_l1, y_scale_l1 = self.entropy_parameters(y_context_l1).chunk(2, 1)

        # Second layer
        y_hat_l1, _ = self.quant(y, offset=y_mean_l1, noisy=noisy)
        y_ar_l2 = self.context_model_layer2(y_hat_l1 * (1 - ThreeD_mask_checkboard))
        y_context_l2 = torch.cat((y_hyper, y_ar_l2), dim=1)
        y_mean_l2, y_scale_l2 = self.entropy_parameters(y_context_l2).chunk(2, 1)

        y_means = (1 - ThreeD_mask_checkboard) * y_mean_l1 + ThreeD_mask_checkboard * y_mean_l2
        y_scales = (1 - ThreeD_mask_checkboard) * y_scale_l1 + ThreeD_mask_checkboard * y_scale_l2

        y_hat, y_indexes = self.quant(y, offset=y_means, noisy=noisy)
        y_loc = torch.zeros(1).to(x.device)
        bits = self.y_em(y_indexes, loc=y_loc, scale=y_scales, keep_batch=keep_bits_batch)
        side_bits = self.z_em(z_indexes, keep_batch=keep_bits_batch)
        
        # x_hat = self.g_s(y_hat)
        w = self.qmap_generate(z_hat)
        w = self.qmap_gs1(torch.cat([w, y_hat], dim=1))

        x_dec = self.gs_sft_res1(y_hat, w)
        x_dec = self.gs_sft_res2(x_dec, w)
        
        w = self.qmap_gs2(w)
        x_dec = self.g_synthesis.up_attn2(x_dec)
        x_dec = self.g_synthesis.conv_res4(x_dec)
        x_dec = self.g_synthesis.conv_up4(x_dec)
        x_dec = self.g_synthesis.conv_res3(x_dec)
        x_dec = self.gs_sft1(x_dec, w)

        w = self.qmap_gs3(w)
        x_dec = self.g_synthesis.conv_up3(x_dec)
        x_dec = self.g_synthesis.up_attn1(x_dec)
        x_dec = self.g_synthesis.conv_res2(x_dec)
        x_dec = self.gs_sft2(x_dec, w)

        w = self.qmap_gs4(w)
        x_dec = self.g_synthesis.conv_up2(x_dec)
        x_dec = self.g_synthesis.conv_res1(x_dec)
        x_dec = self.gs_sft3(x_dec, w)

        x_dec = self.g_synthesis.conv_up1(x_dec)
        x_hat = (x_dec + 1) / 2.

        # if return_y:
        #     return x_hat, y_hat, bits, side_bits
        # else:
        #     return x_hat, bits, side_bits

        return {
            "x_hat": x_hat,
            "bits": {"y": bits, "z": side_bits},
        }

    def init_tables(self):
        for m in self.modules():
            if hasattr(m, '_init_tables'):
                m._init_tables()

    def fix_tables(self):
        for m in self.modules():
            if hasattr(m, '_fix_tables'):
                m._fix_tables()

    def compress(self, x, q_factor, reconstruct=False):
        # self.device = device
        self.device = x.device
        """Compresses an image tensor."""
        x = x * 2 - 1
        if isinstance(q_factor, float):
            qmap = torch.tensor(q_factor).repeat(1, 1, x.size(2), x.size(3)).float()
        else:
            qmap = cv2.imread(q_factor) # 0 ~ 255
            if len(qmap.shape) == 3:
                qmap = qmap[...,0]
            qmap = torch.from_numpy(qmap).float() / 255
            qmap = qmap.unsqueeze(0).unsqueeze(0)
        qmap = qmap.to(self.device)
        # y = self.g_a(x)
        qmap = self.qmap_ga1(torch.cat([qmap, x], dim=1))
        x = self.g_analysis.conv_down1(x)
        x = self.g_analysis.down_attn1(x)
        x = self.g_analysis.conv_res1(x)
        x = self.ga_sft1(x, qmap)

        qmap = self.qmap_ga2(qmap)
        x = self.g_analysis.conv_down2(x)
        x = self.g_analysis.down_attn2(x)
        x = self.g_analysis.conv_res2(x)
        x = self.ga_sft2(x, qmap)

        qmap = self.qmap_ga3(qmap)
        x = self.g_analysis.conv_down3(x)
        x = self.g_analysis.down_attn3(x)
        x = self.g_analysis.conv_res3(x)
        x = self.ga_sft3(x, qmap)

        qmap = self.qmap_ga4(qmap)    
        x = self.g_analysis.conv_down4(x)
        x = self.g_analysis.down_attn4(x)
        x = self.ga_sft_res1(x, qmap)
        y = self.ga_sft_res2(x, qmap)
        y = torch.clamp(y, min=-255.5, max=256.49)

        z = self.h_analysis(y)
        z_hat, z_indexes = self.quant(z, noisy=False)
        y_hyper = self.h_synthesis(z_hat)
        string_l1, string_l2 = self._compress_CheckerCube(y, y_hyper)
        side_string = self.z_em.compress(z_indexes)
        strings = [string_l1, string_l2, side_string]

        if self.debug:
            self.enc_var.update({
                'z_hat': z_hat,
            })
        return strings

    def _compress_CheckerCube(self, y, y_hyper):
        if self.debug:
            import numpy as np
            enc_sever_enc_var = np.load('237_encoder_tensor.npy', allow_pickle=True)
            print(enc_sever_enc_var['y_scale_l1'])
            print(list(enc_sever_enc_var.keys()))
        # 3D checkboard mask generation
        H, W = y.shape[2:]
        mask_checkboard = torch.ones(1, 4, H // 2, W // 2).to(self.device)
        mask_checkboard[:, 1:3, :, :] = 0
        mask_checkboard = F.pixel_shuffle(mask_checkboard, 2)  # mask = 0, the first layer
        ThreeD_mask_checkboard_layer = torch.cat([mask_checkboard, 1 - mask_checkboard], dim=1)
        ThreeD_mask_checkboard = ThreeD_mask_checkboard_layer.repeat(1, self.M // 2, 1, 1)
        # First layer
        ## hyper only
        y_context_l1 = torch.cat((y_hyper, torch.zeros_like(y_hyper)), dim=1)
        y_mean_l1, y_scale_l1 = self.entropy_parameters(y_context_l1).chunk(2, 1)

        # Second layer
        y_hat_l1, _ = self.quant(y, offset=y_mean_l1, noisy=False)
        y_ar_l2 = self.context_model_layer2(y_hat_l1 * (1 - ThreeD_mask_checkboard))
        y_context_l2 = torch.cat((y_hyper, y_ar_l2), dim=1)
        y_mean_l2, y_scale_l2 = self.entropy_parameters(y_context_l2).chunk(2, 1)

        y_means = (1 - ThreeD_mask_checkboard) * y_mean_l1 + ThreeD_mask_checkboard * y_mean_l2
        y_scales = (1 - ThreeD_mask_checkboard) * y_scale_l1 + ThreeD_mask_checkboard * y_scale_l2
        y_indexes = self.quant.quantize(y, offset=y_means)

        # 3D checkboard mask generation
        B, cy, H, W = y.shape[:]
        mask1 = (1 - ThreeD_mask_checkboard).bool()
        mask2 = ThreeD_mask_checkboard.bool()

        # First Layer
        y_scale_l1 = y_scales.masked_select(mask1).reshape(cy, H * W // 2)
        if self.debug:
            print('Cross platform debugging')
            import numpy as np
            enc_sever_enc_var = np.load('237_encoder_tensor.npy', allow_pickle=True)

            print((y_hyper-enc_sever_enc_var['y_hyper']).abs().mean())
            print((y_scale_l1-enc_sever_enc_var['y_scale_l1']).abs().mean())
            assert y_hyper.equal(enc_sever_enc_var['y_hyper'])
            assert y_scale_l1.equal(enc_sever_enc_var['y_scale_l1'])
        # import numpy as np
        # dict_tmp = {'y_scale_l1':y_scale_l1,
        #             'y_hyper': y_hyper}
        # np.save('./237_encoder_tensor.npy', dict_tmp)
        y_loc = torch.zeros(1).to(self.device)
        y_l1_indexes = y_indexes.masked_select(mask1).reshape(cy, H * W // 2)
        string_l1 = self.y_em.compress(y_l1_indexes, loc=y_loc, scale=y_scale_l1)
        # Second Layer
        y_scale_l2 = y_scales.masked_select(mask2).reshape(cy, H * W // 2)
        y_l2_indexes = y_indexes.masked_select(mask2).reshape(cy, H * W // 2)
        string_l2 = self.y_em.compress(y_l2_indexes, loc=y_loc, scale=y_scale_l2)
        if self.debug:


            self.enc_var.update({
                'y_hyper': y_hyper,
                'y_ar_l2': y_ar_l2,
                'y_context_l2': y_context_l2,
                'y_scale_l1': y_scale_l1,
                'y_scale_l2': y_scale_l2,
                'y_hat_l1': y_hat_l1 * (1 - ThreeD_mask_checkboard),
                'y_l1_indexes': y_l1_indexes,
                'string_l1': string_l1
            })
        return string_l1, string_l2

    def decompress(self, strings, shape):
        # self.device = device
        """Decompresses an image tensor."""
        string_l1, string_l2, side_string = strings
        factor = 64
        z_shape = [int(math.ceil(s / factor)) for s in shape]
        z_indexes = self.z_em.decompress(side_string, z_shape)
        z_hat = self.quant.dequantize(z_indexes)

        y_hyper = self.h_synthesis(z_hat)
        self.device = y_hyper.device
        if self.debug:
            print('Debugging...')
            assert z_hat.equal(self.enc_var['z_hat'])
            assert y_hyper.equal(self.enc_var['y_hyper'])

        y_hat = self._decompress_CheckerCube(string_l1, string_l2, y_hyper)

        w = self.qmap_generate(z_hat)
        w = self.qmap_gs1(torch.cat([w, y_hat], dim=1))
        x_hat = self.gs_sft_res1(y_hat, w)
        x_hat = self.gs_sft_res2(x_hat, w)
        
        w = self.qmap_gs2(w)
        x_hat = self.g_synthesis.up_attn2(x_hat)
        x_hat = self.g_synthesis.conv_res4(x_hat)
        x_hat = self.g_synthesis.conv_up4(x_hat)
        x_hat = self.g_synthesis.conv_res3(x_hat)
        x_hat = self.gs_sft1(x_hat, w)

        w = self.qmap_gs3(w)
        x_hat = self.g_synthesis.conv_up3(x_hat)
        x_hat = self.g_synthesis.up_attn1(x_hat)
        x_hat = self.g_synthesis.conv_res2(x_hat)
        x_hat = self.gs_sft2(x_hat, w)

        w = self.qmap_gs4(w)
        x_hat = self.g_synthesis.conv_up2(x_hat)
        x_hat = self.g_synthesis.conv_res1(x_hat)
        x_hat = self.gs_sft3(x_hat, w)

        x_hat = self.g_synthesis.conv_up1(x_hat)
        x_hat = (x_hat + 1) / 2.

        return x_hat

    def _decompress_CheckerCube(self, string_l1, string_l2, y_hyper):
        # 3D checkboard mask generation
        B, C, H, W = y_hyper.shape[:]
        cy = C//2
        y_hat = torch.zeros(B, cy, H, W).to(self.device)
        mask_checkboard = torch.ones(1, 4, H // 2, W // 2).to(self.device)
        mask_checkboard[:, 1:3, :, :] = 0
        mask_checkboard = F.pixel_shuffle(mask_checkboard, 2)  # mask = 0, the first layer
        ThreeD_mask_checkboard_layer = torch.cat([mask_checkboard, 1 - mask_checkboard], dim=1)
        ThreeD_mask_checkboard = ThreeD_mask_checkboard_layer.repeat(1, cy // 2, 1, 1)
        mask1 = (1 - ThreeD_mask_checkboard).bool()
        mask2 = ThreeD_mask_checkboard.bool()
        ## First layer
        y_context_l1 = torch.cat((y_hyper, torch.zeros_like(y_hyper)), dim=1)
        y_mean_l1, y_scale_l1 = self.entropy_parameters(y_context_l1).chunk(2, 1)
        y_scale_l1 = y_scale_l1.masked_select(mask1).reshape(cy, H * W // 2)
        y_mean_l1 = y_mean_l1.masked_select(mask1).reshape(cy, H * W // 2)
        y_loc = torch.zeros(1).to(self.device)
        y_l1_indexes = self.y_em.decompress(string_l1, loc=y_loc, scale=y_scale_l1)
        y_hat_l1 = self.quant.dequantize(y_l1_indexes, offset=y_mean_l1)
        y_hat[mask1.broadcast_to((B, cy, H, W))] = y_hat_l1.float().reshape(-1).to(self.device)

        ## Second layer
        y_ar_l2 = self.context_model_layer2(y_hat)
        if self.debug:
            y_hat_l1 = y_hat
        y_context_l2 = torch.cat((y_hyper, y_ar_l2), dim=1)
        y_mean_l2, y_scale_l2 = self.entropy_parameters(y_context_l2).chunk(2, 1)
        y_scale_l2 = y_scale_l2.masked_select(mask2).reshape(cy, H * W // 2)
        y_mean_l2 = y_mean_l2.masked_select(mask2).reshape(cy, H * W // 2)
        y_l2_indexes = self.y_em.decompress(string_l2, loc=y_loc, scale=y_scale_l2)
        y_hat_l2 = self.quant.dequantize(y_l2_indexes, offset=y_mean_l2)
        y_hat[mask2.broadcast_to((B, cy, H, W))] = y_hat_l2.float().reshape(-1).to(self.device)

        if self.debug:

            assert y_scale_l1.equal(self.enc_var['y_scale_l1'])
            assert string_l1 == self.enc_var['string_l1']
            print((y_l1_indexes - self.enc_var['y_l1_indexes']).abs().mean())
            assert y_l1_indexes.equal(self.enc_var['y_l1_indexes'])
            print((y_hat_l1 - self.enc_var['y_hat_l1']).abs().mean())
            assert y_hat_l1.equal(self.enc_var['y_hat_l1'])
            print((y_ar_l2 - self.enc_var['y_ar_l2']).abs().mean())
            assert y_ar_l2.equal(self.enc_var['y_ar_l2'])
            assert y_context_l2.equal(self.enc_var['y_context_l2'])
            print((y_scale_l2 - self.enc_var['y_scale_l2']).abs().mean())
            assert y_scale_l2.equal(self.enc_var['y_scale_l2'])
        return y_hat


    def group_compress(self, x, group_msk, q_factor):
        # self.device = device
        self.device = x.device
        """Compresses an image tensor."""
        x = x * 2 - 1
        if isinstance(q_factor, float):
            qmap = torch.tensor(q_factor).repeat(1, 1, x.size(2), x.size(3)).float()
        else:
            qmap = cv2.imread(q_factor) # 0 ~ 255
            if len(qmap.shape) == 3:
                qmap = qmap[...,0]
            qmap = torch.from_numpy(qmap).float() / 255
            qmap = qmap.unsqueeze(0).unsqueeze(0)
        qmap = qmap.to(self.device)

        # y = self.g_a(x)
        qmap = self.qmap_ga1(torch.cat([qmap, x], dim=1))
        x = self.g_analysis.conv_down1(x)
        x = self.g_analysis.down_attn1(x)
        x = self.g_analysis.conv_res1(x)
        x = self.ga_sft1(x, qmap)

        qmap = self.qmap_ga2(qmap)
        x = self.g_analysis.conv_down2(x)
        x = self.g_analysis.down_attn2(x)
        x = self.g_analysis.conv_res2(x)
        x = self.ga_sft2(x, qmap)

        qmap = self.qmap_ga3(qmap)
        x = self.g_analysis.conv_down3(x)
        x = self.g_analysis.down_attn3(x)
        x = self.g_analysis.conv_res3(x)
        x = self.ga_sft3(x, qmap)

        qmap = self.qmap_ga4(qmap)    
        x = self.g_analysis.conv_down4(x)
        x = self.g_analysis.down_attn4(x)
        x = self.ga_sft_res1(x, qmap)
        y = self.ga_sft_res2(x, qmap)
        y = torch.clamp(y, min=-255.5, max=256.49)

        z = self.h_analysis(y)
        z_hat, z_indexes = self.quant(z, noisy=False)
        
        y_hyper = self.h_synthesis(z_hat)
        side_string = self.z_em.compress(z_indexes)

        # string_l1, string_l2 = self._compress_CheckerCube(y, y_hyper)
        strings = []
        for group_idx in group_msk.unique():
            if group_idx == -1: # -1 means not compress
                continue
            single_group_msk = group_msk==group_idx
            y_tmp = y.clone()
            string_l1, string_l2 = self._group_compress_CheckerCube(y_tmp, y_hyper, single_group_msk)
            strings.append(string_l1)
            strings.append(string_l2)

        strings.append(side_string)
        return strings

    def group_decompress(self, strings, shape, group_msk, groups_tobe_decode=None):

        # self.device = device
        """Decompresses an image tensor."""
        # string_l1, string_l2, side_string = strings
        side_string = strings[-1]

        factor = 64
        z_shape = [int(math.ceil(s / factor)) for s in shape]
        z_indexes = self.z_em.decompress(side_string, z_shape)
        z_hat = self.quant.dequantize(z_indexes)

        y_hyper = self.h_synthesis(z_hat)
        self.device = y_hyper.device
        # # y_hat = self._decompress_CheckerCube(string_l1, string_l2, y_hyper)
        by, _, hy, wy = y_hyper.shape
        y_hat = torch.zeros((by, self.M, hy, wy)).to(self.device)

        if groups_tobe_decode:
            for group_idx in groups_tobe_decode:
                assert group_idx in group_msk.unique()
            group_idxs = groups_tobe_decode
        else:
            group_idxs = group_msk.unique().tolist()
            if -1 in group_idxs:
                group_idxs.remove(-1)

        assert -1 not in group_idxs

        string_idxs = range(len(group_idxs))

        for group_idx, string_idx in zip(group_idxs, string_idxs):
            single_group_msk = (group_msk==group_idx).to(y_hyper.device)
            # string_l1, string_l2 = strings[group_idx*2], strings[group_idx*2+1]
            string_l1 = strings[string_idx*2]
            string_l2 = strings[string_idx*2+1]
            y_hat_tmp = self._group_decompress_CheckerCube(string_l1, string_l2, y_hyper, single_group_msk)
            single_group_msk = einops.repeat(
                single_group_msk, 'b c h w -> b (repeat c) h w', repeat=self.M)
            y_hat[single_group_msk] = y_hat_tmp[single_group_msk]

        # generate qmap_msk
        qmap_msk = group_msk.clone().float().to(self.device)
        qmap_msk = F.interpolate(qmap_msk, scale_factor=16, mode='nearest')
        qmap_msk[qmap_msk>=0] = 1
        qmap_msk[qmap_msk==-1] = 0  # -1 means not compress. 
                                    # todo : change not compress to not transmit. 

        w = self.qmap_generate(z_hat)
        w = self.qmap_gs1(torch.cat([w, y_hat], dim=1))
        # x_hat = self.gs_sft_res1(y_hat, w)
        # x_hat = self.gs_sft_res2(x_hat, w)
        x_hat = self.gs_sft_res1(y_hat, w, qmap_msk)
        x_hat = self.gs_sft_res2(x_hat, w, qmap_msk)
        
        w = self.qmap_gs2(w)
        x_hat = self.g_synthesis.up_attn2(x_hat)
        x_hat = self.g_synthesis.conv_res4(x_hat)
        x_hat = self.g_synthesis.conv_up4(x_hat)
        x_hat = self.g_synthesis.conv_res3(x_hat)
        # x_hat = self.gs_sft1(x_hat, w)
        x_hat = self.gs_sft1(x_hat, w, qmap_msk)

        w = self.qmap_gs3(w)
        x_hat = self.g_synthesis.conv_up3(x_hat)
        x_hat = self.g_synthesis.up_attn1(x_hat)
        x_hat = self.g_synthesis.conv_res2(x_hat)
        # x_hat = self.gs_sft2(x_hat, w)
        x_hat = self.gs_sft2(x_hat, w, qmap_msk)

        w = self.qmap_gs4(w)
        x_hat = self.g_synthesis.conv_up2(x_hat)
        x_hat = self.g_synthesis.conv_res1(x_hat)
        # x_hat = self.gs_sft3(x_hat, w)
        x_hat = self.gs_sft3(x_hat, w, qmap_msk)

        x_hat = self.g_synthesis.conv_up1(x_hat)
        x_hat = (x_hat + 1) / 2.

        return x_hat

    def _group_compress_CheckerCube(self, y, y_hyper, single_group_msk):
        # pre-process the single_group_msk
        not_single_group_msk = einops.repeat(
            (1-(single_group_msk).int()).bool(), 
            'b c h w -> b (repeat c) h w', repeat=self.M)

        # 3D checkboard mask generation
        H, W = y.shape[2:]
        mask_checkboard = torch.ones(1, 4, H // 2, W // 2).to(self.device)
        mask_checkboard[:, 1:3, :, :] = 0
        mask_checkboard = F.pixel_shuffle(mask_checkboard, 2)  # mask = 0, the first layer
        ThreeD_mask_checkboard_layer = torch.cat([mask_checkboard, 1 - mask_checkboard], dim=1)
        ThreeD_mask_checkboard = ThreeD_mask_checkboard_layer.repeat(1, self.M // 2, 1, 1)

        # First layer
        ## hyper only
        y_context_l1 = torch.cat((y_hyper, torch.zeros_like(y_hyper)), dim=1)
        y_mean_l1, y_scale_l1 = self.entropy_parameters(y_context_l1).chunk(2, 1)
        # set elements outside the group the same value as y_mean_l1
        # to ensure the context module are equal in both compress and decompress.
        y[not_single_group_msk] = y_mean_l1[not_single_group_msk]

        # Second layer
        y_hat_l1, _ = self.quant(y, offset=y_mean_l1, noisy=False)
        y_ar_l2 = self.context_model_layer2(y_hat_l1 * (1 - ThreeD_mask_checkboard))
        y_context_l2 = torch.cat((y_hyper, y_ar_l2), dim=1)
        y_mean_l2, y_scale_l2 = self.entropy_parameters(y_context_l2).chunk(2, 1)

        y_means = (1 - ThreeD_mask_checkboard) * y_mean_l1 + ThreeD_mask_checkboard * y_mean_l2
        y_scales = (1 - ThreeD_mask_checkboard) * y_scale_l1 + ThreeD_mask_checkboard * y_scale_l2
        y_indexes = self.quant.quantize(y, offset=y_means)

        # 3D checkboard mask generation
        B, cy, H, W = y.shape[:]
        mask1 = (1 - ThreeD_mask_checkboard).bool()
        mask2 = ThreeD_mask_checkboard.bool()

        # change the mask1 and mask2 according to its corresponding group
        mask1[not_single_group_msk] = False
        mask2[not_single_group_msk] = False

        # First Layer
        n_elements = (single_group_msk.int()).sum() // 2
        # y_scale_l1 = y_scales.masked_select(mask1).reshape(cy, H * W // 2)
        y_scale_l1 = y_scales.masked_select(mask1).reshape(cy, n_elements)
        y_loc = torch.zeros(1).to(self.device)
        # y_l1_indexes = y_indexes.masked_select(mask1).reshape(cy, H * W // 2)
        y_l1_indexes = y_indexes.masked_select(mask1).reshape(cy, n_elements)
        string_l1 = self.y_em.compress(y_l1_indexes, loc=y_loc, scale=y_scale_l1)

        # Second Layer
        # y_scale_l2 = y_scales.masked_select(mask2).reshape(cy, H * W // 2)
        # y_l2_indexes = y_indexes.masked_select(mask2).reshape(cy, H * W // 2)
        y_scale_l2 = y_scales.masked_select(mask2).reshape(cy, n_elements)
        y_l2_indexes = y_indexes.masked_select(mask2).reshape(cy, n_elements)

        string_l2 = self.y_em.compress(y_l2_indexes, loc=y_loc, scale=y_scale_l2)
        return string_l1, string_l2

    def _group_decompress_CheckerCube(self, string_l1, string_l2, y_hyper, single_group_msk):
        # pre-process the single_group_msk
        not_single_group_msk = einops.repeat(
            (1-(single_group_msk).int()).bool(), 
            'b c h w -> b (repeat c) h w', repeat=self.M)

        # 3D checkboard mask generation
        B, C, H, W = y_hyper.shape[:]
        cy = C//2
        y_hat = torch.zeros(B, cy, H, W).to(self.device)
        mask_checkboard = torch.ones(1, 4, H // 2, W // 2).to(self.device)
        mask_checkboard[:, 1:3, :, :] = 0
        mask_checkboard = F.pixel_shuffle(mask_checkboard, 2)  # mask = 0, the first layer
        ThreeD_mask_checkboard_layer = torch.cat([mask_checkboard, 1 - mask_checkboard], dim=1)
        ThreeD_mask_checkboard = ThreeD_mask_checkboard_layer.repeat(1, cy // 2, 1, 1)
        mask1 = (1 - ThreeD_mask_checkboard).bool()
        mask2 = ThreeD_mask_checkboard.bool()
        # change the mask1 and mask2 according to its corresponding group
        mask1[not_single_group_msk] = False
        mask2[not_single_group_msk] = False

        ## First layer
        n_elements = (single_group_msk.int()).sum() // 2
        y_context_l1 = torch.cat((y_hyper, torch.zeros_like(y_hyper)), dim=1)
        y_mean_l1, y_scale_l1 = self.entropy_parameters(y_context_l1).chunk(2, 1)
        # Set elements outside the group the same value as y_mean_l1
        # to ensure the context module are equal in both compress and decompress.
        # In decompress process, a quantizaion should be performed.
        y_hat[not_single_group_msk] = y_mean_l1[not_single_group_msk]
        y_scale_l1 = y_scale_l1.masked_select(mask1).reshape(cy, n_elements)
        y_mean_l1 = y_mean_l1.masked_select(mask1).reshape(cy, n_elements)
        y_loc = torch.zeros(1).to(self.device)
        y_l1_indexes = self.y_em.decompress(string_l1, loc=y_loc, scale=y_scale_l1)
        y_hat_l1 = self.quant.dequantize(y_l1_indexes, offset=y_mean_l1)
        y_hat[mask1.broadcast_to((B, cy, H, W))] = y_hat_l1.float().reshape(-1).to(self.device)

        ## Second layer
        # y_ar_l2 = self.context_model_layer2(y_hat)
        y_ar_l2 = self.context_model_layer2((y_hat * (1 - ThreeD_mask_checkboard))) # bad.
        y_context_l2 = torch.cat((y_hyper, y_ar_l2), dim=1)
        y_mean_l2, y_scale_l2 = self.entropy_parameters(y_context_l2).chunk(2, 1)
        y_mean_l2 = y_mean_l2.masked_select(mask2).reshape(cy, n_elements)
        y_scale_l2 = y_scale_l2.masked_select(mask2).reshape(cy, n_elements)
        y_l2_indexes = self.y_em.decompress(string_l2, loc=y_loc, scale=y_scale_l2)
        y_hat_l2 = self.quant.dequantize(y_l2_indexes, offset=y_mean_l2)
        y_hat[mask2.broadcast_to((B, cy, H, W))] = y_hat_l2.float().reshape(-1).to(self.device)

        return y_hat


class ImgComNet(nn.Module):
    def __init__(self, args, inplace=False):
        super().__init__()
        self.N = args.N
        self.M = args.M

        self.g_analysis = nn.Sequential(OrderedDict([
            ('conv_down1', ResBlockDown(3, self.N)),
            ('down_attn1', ResAttnSplit(self.N, self.N, groups=2)),
            ('conv_res1', ResBlock(self.N, self.N)),
            ('conv_down2', ResBlockDown(self.N, self.N)),
            ('down_attn2', ResAttnSplit(self.N, self.N, groups=2)),
            ('conv_res2', ResBlock(self.N, self.N)),
            ('conv_down3', ResBlockDown(self.N, self.N)),
            ('down_attn3', ResAttnSplit(self.N, self.N, groups=2)),
            ('conv_res3', ResBlock(self.N, self.N)),
            ('conv_down4', Downsample(self.N, self.M)),
            ('down_attn4', ResAttnSplit(self.M, self.M, groups=2)),
        ]))

        self.g_synthesis = nn.Sequential(OrderedDict([
            ('up_attn2', ResAttnSplit(self.M, self.M, groups=2)),
            ('conv_res4', ResBlock(self.M, self.M)),
            ('conv_up4', ResBlockUp(self.M, self.N)),
            ('conv_res3', ResBlock(self.N, self.N)),
            ('conv_up3', ResBlockUp(self.N, self.N)),
            ('up_attn1', ResAttnSplit(self.N, self.N, groups=2)),
            ('conv_res2', ResBlock(self.N, self.N)),
            ('conv_up2', ResBlockUp(self.N, self.N)),
            ('conv_res1', ResBlock(self.N, self.N)),
            ('conv_up1', Upsample(self.N, 3)),
        ]))

        self.h_analysis = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.M, self.N, kernel_size=3, stride=1, padding=1)),
            ('relu1', nn.LeakyReLU(inplace=True)),
            ('conv2', nn.Conv2d(self.N, self.N, kernel_size=5, stride=2, padding=2)),
            ('relu2', nn.LeakyReLU(inplace=True)),
            ('conv3', nn.Conv2d(self.N, self.N, kernel_size=5, stride=2, padding=2)),
            ('down_attn', ResAttn(self.N, self.N)),
        ]))

        self.h_synthesis = nn.Sequential(OrderedDict([
            ('unconv1', nn.ConvTranspose2d(self.N, self.M, kernel_size=5, stride=2, padding=2, output_padding=1)),
            ('relu1', nn.LeakyReLU(inplace=True)),
            ('unconv2', nn.ConvTranspose2d(self.M, self.M + self.M // 2, kernel_size=5, stride=2, padding=2, output_padding=1)),
            ('relu2', nn.LeakyReLU(inplace=True)),
            ('unconv3', nn.ConvTranspose2d(self.M + self.M // 2, self.M * 2, kernel_size=3, stride=1, padding=1)),
            # ('resblock', ResBlock(M * 2, M * 2)),
        ]))

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(self.M * 12 // 3, self.M * 10 // 3, 1, 1, 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.M * 10 // 3, self.M * 8 // 3, 1, 1, 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.M * 8 // 3, self.M * 6 // 3, 1, 1, 0),
        )

        self.context_model_layer2 = nn.Sequential(
            nn.Conv2d(self.M, self.M, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.M, self.M, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.M, self.M * 2, 3, 1, 1)
        )

        self.quant = UniformQuantization(step=1)
        
        scale_min, scale_max, num_scales = 0.11, 256, 64
        offset = math.log(scale_min)
        factor = (math.log(scale_max) - math.log(scale_min))/(num_scales - 1)
        scale_table = torch.exp(offset + factor * torch.arange(num_scales))
        self.y_em = ContinuousConditionalEntropyModel(
            NoisyNormal, param_tables=dict(loc=[0], scale=scale_table.tolist()))
        self.z_em = ContinuousUnconditionalEntropyModel(
            NoisyDeepFactorized(batch_shape=(self.N,)))
    
    def forward(self, x, noisy=True, trainer=None, return_y=False):
        x_enc = x * 2 - 1
        y = self.g_analysis(x_enc)
        z = self.h_analysis(y)
        z_hat, z_indexes = self.quant(z, noisy=noisy)
        y_hyper = self.h_synthesis(z_hat)

        # 3D checkboard mask generation
        H, W = y.shape[2:]
        mask_checkboard = torch.ones(1, 4, H // 2, W // 2).to(self.device)
        mask_checkboard[:, 1:3, :, :] = 0
        mask_checkboard = F.pixel_shuffle(mask_checkboard, 2) # mask = 0, the first layer
        ThreeD_mask_checkboard_layer = torch.cat([mask_checkboard, 1-mask_checkboard], dim=1)
        ThreeD_mask_checkboard = ThreeD_mask_checkboard_layer.repeat(1, self.M // 2, 1, 1)

        # First layer
        ## hyper only
        y_context_l1 = torch.cat((y_hyper, torch.zeros_like(y_hyper)), dim=1)
        y_mean_l1, y_scale_l1 = self.entropy_parameters(y_context_l1).chunk(2, 1)

        # Second layer
        y_hat_l1, _ = self.quant(y, offset=y_mean_l1, noisy=noisy)
        y_ar_l2 = self.context_model_layer2(y_hat_l1 * (1 - ThreeD_mask_checkboard))
        y_context_l2 = torch.cat((y_hyper, y_ar_l2), dim=1)
        y_mean_l2, y_scale_l2 = self.entropy_parameters(y_context_l2).chunk(2, 1)

        y_means = (1 - ThreeD_mask_checkboard) * y_mean_l1 + ThreeD_mask_checkboard * y_mean_l2
        y_scales = (1 - ThreeD_mask_checkboard) * y_scale_l1 + ThreeD_mask_checkboard * y_scale_l2

        y_hat, y_indexes = self.quant(y, offset=y_means, noisy=noisy)
        y_loc = torch.zeros(1).to(self.device)
        bits = self.y_em(y_indexes, loc=y_loc, scale=y_scales)
        side_bits = self.z_em(z_indexes)
        
        x_dec = self.g_synthesis(y_hat)
        x_hat = (x_dec + 1) / 2.

        if return_y:
            return x_hat, y_hat, bits, side_bits
        else:
            return x_hat, bits, side_bits
    
    def init_tables(self):
        for m in self.modules():
            if hasattr(m, '_init_tables'):
                m._init_tables()

    def fix_tables(self):
        for m in self.modules():
            if hasattr(m, '_fix_tables'):
                m._fix_tables()

    def compress(self, x, qmap, reconstruct=False):
        """Compresses an image tensor."""
        x = x * 2 - 1
        # y = self.g_a(x)
        qmap = self.qmap_fuse_enc(torch.cat([qmap, x], dim=1))
        qmap = self.qmap_ga1(qmap)
        x = self.g_analysis.conv_down1(x)
        x = self.g_analysis.down_attn1(x)
        x = self.g_analysis.conv_res1(x)
        x = self.ga_sft1(x, qmap)

        qmap = self.qmap_ga2(qmap)
        x = self.g_analysis.conv_down2(x)
        x = self.g_analysis.down_attn2(x)
        x = self.g_analysis.conv_res2(x)
        x = self.ga_sft2(x, qmap)

        qmap = self.qmap_ga3(qmap)
        x = self.g_analysis.conv_down3(x)
        x = self.g_analysis.down_attn3(x)
        x = self.g_analysis.conv_res3(x)
        x = self.ga_sft3(x, qmap)

        qmap = self.qmap_ga4(qmap)    
        x = self.g_analysis.conv_down4(x)
        x = self.g_analysis.down_attn4(x)
        x = self.ga_sft_res1(x, qmap)
        y = self.ga_sft_res2(x, qmap)

        # if not hasattr(self, 'z_em'):
        #     y_indexes = self.quant.quantize(y)
        #     string = self.y_em.compress(y_indexes)
        #     side_string = ''
        # else:
        z = self.h_a(y)
        z_hat, z_indexes = self.quant(z, noisy=False)
        y_means, y_scales = self.h_s(z_hat).chunk(2, 1)
        y_indexes = self.quant.quantize(y, offset=y_means)
        y_loc = torch.zeros(1).to(self.device)
        string = self.y_em.compress(y_indexes, loc=y_loc, scale=y_scales)
        side_string = self.z_em.compress(z_indexes)
        strings = [string, side_string]
        # if reconstruct:
        #     if not hasattr(self, 'z_em'):
        #         y_means = None
        #     y_hat = self.quant.dequantize(y_indexes, offset=y_means)
        #     x_hat = self.g_s(y_hat)
        #     return strings, x_hat
        # else:
        return strings

    def decompress(self, strings, shape):
        """Decompresses an image tensor."""
        string, side_string = strings
        factor = 64
        # if not hasattr(self, 'z_em'):
        #     y_shape = [int(math.ceil(s / factor)) * 16 for s in shape]
        #     y_indexes = self.y_em.decompress(string, y_shape)
        #     y_hat = self.quant.dequantize(y_indexes)
        # else:
        z_shape = [int(math.ceil(s / factor)) for s in shape]
        z_indexes = self.z_em.decompress(side_string, z_shape)
        z_hat = self.quant.dequantize(z_indexes)
        y_means, y_scales = self.h_s(z_hat).chunk(2, 1)
        y_loc = torch.zeros(1).to(self.device)
        y_indexes = self.y_em.decompress(string, loc=y_loc, scale=y_scales)
        y_hat = self.quant.dequantize(y_indexes, offset=y_means)
        
        # x_hat = self.g_s(y_hat)
        w = self.qmap_generate(z_hat)
        w = self.qmap_fuse_dec(torch.cat([w, y_hat], dim=1))
        x_hat = self.gs_sft_res1(y_hat, w)
        x_hat = self.gs_sft_res2(x_hat, w)
        
        w = self.qmap_gs1(w)
        x_hat = self.g_synthesis.up_attn2(x_hat)
        x_hat = self.g_synthesis.conv_res4(x_hat)
        x_hat = self.g_synthesis.conv_up4(x_hat)
        x_hat = self.g_synthesis.conv_res3(x_hat)
        x_hat = self.gs_sft1(x_hat, w)

        w = self.qmap_gs2(w)
        x_hat = self.g_synthesis.conv_up3(x_hat)
        x_hat = self.g_synthesis.up_attn1(x_hat)
        x_hat = self.g_synthesis.conv_res2(x_hat)
        x_hat = self.gs_sft2(x_hat, w)

        w = self.qmap_gs3(w)
        x_hat = self.g_synthesis.conv_up2(x_hat)
        x_hat = self.g_synthesis.conv_res1(x_hat)
        x_hat = self.gs_sft3(x_hat, w)

        x_hat = self.g_synthesis.conv_up1(x_hat)
        x_hat = (x_hat + 1) / 2

        return x_hat


class ImgComNet_Lossless_GMM(ImgComNet):
    def __init__(self, args, inplace=False):
        super().__init__(args)
        self.N = args.N
        self.M = args.M
        u_channel = 64
        self.K = 5
        self.p_num = 3
        self.res_fea = Upsample(self.N, u_channel)
        self.res_entropy_parameters = nn.Sequential(
            nn.Conv2d(u_channel * 2, u_channel * 2 // 3, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(u_channel * 2 // 3, u_channel, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(u_channel, 3 * self.K * self.p_num, 3, 1, 1))
        self.res_em = GaussianConditional()
        self.res_context_model = nn.Sequential(
            nn.Conv2d(3, 3 * 4, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(3 * 4, 3 * 8, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(3 * 8, u_channel, 3, 1, 1))
        self.softmax = nn.Softmax(1)

    def g_s_res(self, y_hat):
        for i in range(len(self.g_synthesis) - 1):
            y_hat = self.g_synthesis[i](y_hat)
        u = self.res_fea(y_hat)
        x_hat = self.g_synthesis[i+1](y_hat)
        return x_hat, u

    def _reshape(self, x):
        if x is None:
            return None
        N, _, H, W = x.shape
        return x.reshape(N, self.K * self.p_num, 3, H, W)

    def res_entropy(self, x, NN_out):
        r_means = NN_out[:, :self.K, :, :, :].chunk(self.K, 1)
        r_scales = NN_out[:, self.K: 2*self.K, :, :, :].chunk(self.K, 1)
        r_ws = NN_out[:, 2*self.K: 3*self.K, :, :, :].chunk(self.K, 1)
        r_w_softmax = self.softmax(torch.cat(r_ws, dim=1))
        res_likelihood = torch.zeros_like(x)
        for i in range(self.K):
            res_likelihood += self.res_em(x, r_scales[i].squeeze(), r_means[i].squeeze()) * r_w_softmax[:, i]

        return res_likelihood

    def forward(self, x, hx, wx, noisy=True, trainer=None, return_y=False):
        x_enc = x * 2 - 1
        y = self.g_analysis(x_enc)
        z = self.h_analysis(y)
        z_hat, z_indexes = self.quant(z, noisy=noisy)
        y_hyper = self.h_synthesis(z_hat)

        # 3D checkboard mask generation
        H, W = y.shape[2:]
        mask_checkboard = torch.ones(1, 4, H // 2, W // 2).to(self.device)
        mask_checkboard[:, 1:3, :, :] = 0
        mask_checkboard = F.pixel_shuffle(mask_checkboard, 2)  # mask = 0, the first layer
        ThreeD_mask_checkboard_layer = torch.cat([mask_checkboard, 1 - mask_checkboard], dim=1)
        ThreeD_mask_checkboard = ThreeD_mask_checkboard_layer.repeat(1, self.M // 2, 1, 1)

        # First layer
        ## hyper only
        y_context_l1 = torch.cat((y_hyper, torch.zeros_like(y_hyper)), dim=1)
        y_mean_l1, y_scale_l1 = self.entropy_parameters(y_context_l1).chunk(2, 1)

        # Second layer
        y_hat_l1, _ = self.quant(y, offset=y_mean_l1, noisy=noisy)
        y_ar_l2 = self.context_model_layer2(y_hat_l1 * (1 - ThreeD_mask_checkboard))
        y_context_l2 = torch.cat((y_hyper, y_ar_l2), dim=1)
        y_mean_l2, y_scale_l2 = self.entropy_parameters(y_context_l2).chunk(2, 1)

        y_means = (1 - ThreeD_mask_checkboard) * y_mean_l1 + ThreeD_mask_checkboard * y_mean_l2
        y_scales = (1 - ThreeD_mask_checkboard) * y_scale_l1 + ThreeD_mask_checkboard * y_scale_l2

        _, y_indexes = self.quant(y, offset=y_means, noisy=noisy)
        y_hat, _ = self.quant(y, offset=y_means, noisy=False) # STE
        y_loc = torch.zeros(1).to(self.device)
        bits = self.y_em(y_indexes, loc=y_loc, scale=y_scales)
        side_bits = self.z_em(z_indexes)

        x_dec, u = self.g_s_res(y_hat)
        x_hat = (x_dec + 1) / 2.

        # residual compression
        x_ori = x[:, :, :hx, :wx].mul(255).round().clamp(0, 255)
        if trainer is None:
            torch.cuda.empty_cache()
            del y, z, y_hat, z_hat, ThreeD_mask_checkboard, y_means, y_scales, x_dec, x_enc, y_ar_l2, y_context_l1, \
                ThreeD_mask_checkboard_layer, mask_checkboard, y_hyper, y_mean_l1, y_scale_l1, y_mean_l2, y_scale_l2, \
                y_context_l2
            x_recon = x_hat[:, :, :hx, :wx].mul(255).round().clamp(0, 255)
        else:
            x_recon = x_hat[:, :, :hx, :wx].mul(255)
            x_recon = x_recon - (x_recon - x_recon.round()).detach()
            x_recon = x_recon.clamp(0, 255)
        res = x_ori - x_recon
        # checkboard mask generation
        H, W = res.shape[2:]
        mask_checkboard = torch.ones(1, 4, H // 2, W // 2).to(self.device)
        mask_checkboard[:, 1:3, :, :] = 0
        mask_checkboard = F.pixel_shuffle(mask_checkboard, 2)  # mask = 0, the first layer
        if mask_checkboard.shape[-2] != res.shape[-2]:
            mask_checkboard = torch.cat([mask_checkboard, mask_checkboard[:,:,-2,:].unsqueeze(-2)], dim=-2)
        if mask_checkboard.shape[-1] != res.shape[-1]:
            mask_checkboard = torch.cat([mask_checkboard, mask_checkboard[:,:,:,-2].unsqueeze(-1)], dim=-1)
        mask_CheckerCube = torch.cat([mask_checkboard, 1 - mask_checkboard, mask_checkboard], dim=1)
        # First layer
        r_context_l1 = torch.cat((u[:, :, :hx, :wx], torch.zeros_like(u[:, :, :hx, :wx])), dim=1)
        r_out_l1 = self.res_entropy_parameters(r_context_l1)
        # Second layer
        r_ar_l2 = self.res_context_model(res * (1 - mask_CheckerCube))
        r_context_l2 = torch.cat((u[:, :, :hx, :wx], r_ar_l2), dim=1)
        if trainer is None:
            torch.cuda.empty_cache()
            del r_context_l1, r_ar_l2
        r_out_l2 = self.res_entropy_parameters(r_context_l2)
        r_out_l1, r_out_l2 = self._reshape(r_out_l1), self._reshape(r_out_l2)
        mask_CheckerCube = mask_CheckerCube.unsqueeze(1)
        if trainer is None:
            torch.cuda.empty_cache()
            del mask_checkboard, x_recon, u, r_context_l2
        r_out = (1 - mask_CheckerCube) * r_out_l1 + mask_CheckerCube * r_out_l2

        res_likelihood = self.res_entropy(res, r_out)
        res_bits = -torch.sum(torch.log2(res_likelihood))

        if return_y:
            return x_hat, y_hat, bits, side_bits, res_bits
        else:
            return x_hat, bits, side_bits, res_bits

    def compress(self, x_ori, hx, wx, lossy_device, lossless_device, reconstruct=False):
        self.lossy_device = lossy_device
        self.lossless_device = lossless_device
        """Compresses an image tensor."""
        x = x_ori * 2 - 1
        y = self.g_analysis(x)
        z = self.h_analysis(y)
        z_hat, z_indexes = self.quant(z, noisy=False)
        y_hyper = self.h_synthesis(z_hat)
        string_y1, string_y2, y_hat = self._compress_y_CheckerCube(y, y_hyper)
        side_string = self.z_em.compress(z_indexes)
        x_hat, u = self.g_s_res(y_hat)
        u = u.to(self.lossless_device)
        x_hat = (x_hat + 1) / 2.
        del y, z, z_hat, z_indexes, y_hat
        x_ori = x_ori[:, :, :hx, :wx].mul(255).round().clamp(0, 255)
        x_recon = x_hat[:, :, :hx, :wx].mul(255).round().clamp(0, 255)
        res = x_ori - x_recon
        res = res.to(self.lossless_device)
        string_r1, string_r2, r_min, r_max = self._compress_res_CheckerCube(res, u, hx, wx)
        strings = [string_y1, string_y2, side_string, string_r1, string_r2]

        # if reconstruct:
        #     if not hasattr(self, 'z_em'):
        #         y_means = None
        #     y_hat = self.quant.dequantize(y_indexes, offset=y_means)
        #     x_hat = self.g_s(y_hat)
        #     return strings, x_hat
        # else:
        return strings, r_min, r_max

    def _compress_y_CheckerCube(self, y, y_hyper):
        # 3D checkboard mask generation
        H, W = y.shape[2:]
        mask_checkboard = torch.ones(1, 4, H // 2, W // 2).to(self.lossy_device)
        mask_checkboard[:, 1:3, :, :] = 0
        mask_checkboard = F.pixel_shuffle(mask_checkboard, 2)  # mask = 0, the first layer
        ThreeD_mask_checkboard_layer = torch.cat([mask_checkboard, 1 - mask_checkboard], dim=1)
        ThreeD_mask_checkboard = ThreeD_mask_checkboard_layer.repeat(1, self.M // 2, 1, 1)
        # First layer
        ## hyper only
        y_context_l1 = torch.cat((y_hyper, torch.zeros_like(y_hyper)), dim=1)
        y_mean_l1, y_scale_l1 = self.entropy_parameters(y_context_l1).chunk(2, 1)

        # Second layer
        y_hat_l1, _ = self.quant(y, offset=y_mean_l1, noisy=False)
        y_ar_l2 = self.context_model_layer2(y_hat_l1 * (1 - ThreeD_mask_checkboard))
        y_context_l2 = torch.cat((y_hyper, y_ar_l2), dim=1)
        y_mean_l2, y_scale_l2 = self.entropy_parameters(y_context_l2).chunk(2, 1)

        y_means = (1 - ThreeD_mask_checkboard) * y_mean_l1 + ThreeD_mask_checkboard * y_mean_l2
        y_scales = (1 - ThreeD_mask_checkboard) * y_scale_l1 + ThreeD_mask_checkboard * y_scale_l2
        y_hat, y_indexes = self.quant(y, offset=y_means, noisy=False)

        # 3D checkboard mask generation
        B, cy, H, W = y.shape[:]
        mask1 = (1 - ThreeD_mask_checkboard).bool()
        mask2 = ThreeD_mask_checkboard.bool()

        # First Layer
        y_scale_l1 = y_scales.masked_select(mask1).reshape(cy, H * W // 2)
        y_loc = torch.zeros(1).to(self.lossy_device)
        y_l1_indexes = y_indexes.masked_select(mask1).reshape(cy, H * W // 2)
        string_l1 = self.y_em.compress(y_l1_indexes, loc=y_loc, scale=y_scale_l1)
        # Second Layer
        y_scale_l2 = y_scales.masked_select(mask2).reshape(cy, H * W // 2)
        y_l2_indexes = y_indexes.masked_select(mask2).reshape(cy, H * W // 2)
        string_l2 = self.y_em.compress(y_l2_indexes, loc=y_loc, scale=y_scale_l2)
        return string_l1, string_l2, y_hat

    def _compress_res_CheckerCube(self, res, u, hx, wx):
        # checkboard mask generation
        H, W = res.shape[2:]
        mask_checkboard = torch.ones(1, 4, H // 2, W // 2).to(self.lossless_device)
        mask_checkboard[:, 1:3, :, :] = 0
        mask_checkboard = F.pixel_shuffle(mask_checkboard, 2)  # mask = 0, the first layer
        if mask_checkboard.shape[-2] != res.shape[-2]:
            mask_checkboard = torch.cat([mask_checkboard, mask_checkboard[:, :, -2, :].unsqueeze(-2)], dim=-2)
        if mask_checkboard.shape[-1] != res.shape[-1]:
            mask_checkboard = torch.cat([mask_checkboard, mask_checkboard[:, :, :, -2].unsqueeze(-1)], dim=-1)
        mask_CheckerCube = torch.cat([mask_checkboard, 1 - mask_checkboard, mask_checkboard], dim=1)

        # First layer
        r_context_l1 = torch.cat((u[:, :, :hx, :wx], torch.zeros_like(u[:, :, :hx, :wx])), dim=1)
        r_out_l1 = self.res_entropy_parameters.to(self.lossless_device)(r_context_l1)
        r_out_l1 = r_out_l1.to(self.lossless_device)

        # Second layer
        r_ar_l2 = self.res_context_model.to(self.lossless_device)(res * (1 - mask_CheckerCube))
        torch.cuda.empty_cache()
        r_context_l2 = torch.cat((u[:, :, :hx, :wx], r_ar_l2), dim=1)
        # torch.cuda.empty_cache()
        del r_context_l1, r_ar_l2
        r_out_l2 = self.res_entropy_parameters.to(self.lossless_device)(r_context_l2)
        r_out_l1, r_out_l2 = self._reshape(r_out_l1), self._reshape(r_out_l2)
        mask1 = (1 - mask_CheckerCube).bool()
        mask2 = mask_CheckerCube.bool()
        mask_CheckerCube = mask_CheckerCube.unsqueeze(1)
        torch.cuda.empty_cache()
        del mask_checkboard, u, r_context_l2
        r_out = (1 - mask_CheckerCube) * r_out_l1.to(self.lossless_device) + mask_CheckerCube * r_out_l2
        del mask_CheckerCube

        cr, hr, wr = res.shape[1:]
        r_min = res.min().int().item()
        r_max = res.max().int().item()
        # sample = torch.arange(r_min, r_max + 1, dtype=torch.float).view(-1, 1, 1, 1).repeat(1, cr, hr, wr)
        sample = torch.arange(r_min, r_max + 1, dtype=torch.float).view(-1, 1, 1, 1).repeat(1, cr, hr, wr).to(self.lossless_device)
        torch.cuda.empty_cache()
        pmf_r = self.res_entropy(sample, r_out)
        pmf_r = pmf_r.permute(1, 2, 3, 0)  # C, H, W, L
        cdf_r = float_pmf_to_integer_cdf(pmf_r)  # C, H, W, L+1
        inp_r = res - r_min
        ## First layer
        inp_r1 = inp_r.masked_select(mask1).reshape(cr, hr * wr // 2)
        inp_r1 = inp_r1.to(torch.int16).cpu()
        cdf_r1 = cdf_r.masked_select(mask1.permute(1, 2, 3, 0))
        cdf_r1 = cdf_r1.reshape(cr, hr * wr // 2, -1).cpu()
        ## Second layer
        inp_r2 = inp_r.masked_select(mask2).reshape(cr, hr * wr // 2)
        inp_r2 = inp_r2.to(torch.int16).cpu()
        cdf_r2 = cdf_r.masked_select(mask2.permute(1, 2, 3, 0))
        cdf_r2 = cdf_r2.reshape(cr, hr * wr // 2, -1).cpu()

        string_r1 = torchac.encode_int16_normalized_cdf(cdf_r1, inp_r1)  # arithmetical coding
        del r_out, inp_r1, cdf_r1, pmf_r, cdf_r, inp_r
        cdf_r2 = cdf_r2[:,:hr * wr // 2 - 1,:]
        inp_r2 = inp_r2[:,:hr * wr // 2 - 1]
        string_r2 = torchac.encode_int16_normalized_cdf(cdf_r2, inp_r2)  # arithmetical coding

        return string_r1, string_r2, r_min, r_max

    def decompress(self, strings, shape, r_min, r_max, lossy_device, lossless_device):
        self.lossy_device = lossy_device
        self.lossless_device = lossless_device
        hx, wx = shape
        string_y1, string_y2, side_string, string_r1, string_r2 = strings
        factor = 64
        z_shape = [int(math.ceil(s / factor)) for s in shape]
        z_indexes = self.z_em.decompress(side_string, z_shape)
        z_hat = self.quant.dequantize(z_indexes)
        y_hyper = self.h_synthesis(z_hat)
        y_hat = self._decompress_y_CheckerCube(string_y1, string_y2, y_hyper)
        del string_y1, string_y2, side_string, z_shape, z_indexes, y_hyper, z_hat
        x_hat, u = self.g_s_res(y_hat)
        x_hat = (x_hat + 1) / 2.
        x_hat = x_hat.to(self.lossless_device)

        res = self._decompress_res_CheckerCube(string_r1, string_r2, u, hx, wx, r_min, r_max)
        res = res.round().clamp(r_min, r_max).to(self.lossless_device)
        x_recon = x_hat[:, :, :hx, :wx].mul(255).round().clamp(0, 255).to(self.lossless_device)
        return x_recon + res

    def _decompress_y_CheckerCube(self, string_l1, string_l2, y_hyper):
        # 3D checkboard mask generation
        B, C, H, W = y_hyper.shape[:]
        cy = C//2
        y_hat = torch.zeros(B, cy, H, W).to(self.lossy_device)
        mask_checkboard = torch.ones(1, 4, H // 2, W // 2).to(self.lossy_device)
        mask_checkboard[:, 1:3, :, :] = 0
        mask_checkboard = F.pixel_shuffle(mask_checkboard, 2)  # mask = 0, the first layer
        ThreeD_mask_checkboard_layer = torch.cat([mask_checkboard, 1 - mask_checkboard], dim=1)
        ThreeD_mask_checkboard = ThreeD_mask_checkboard_layer.repeat(1, cy // 2, 1, 1)
        mask1 = (1 - ThreeD_mask_checkboard).bool()
        mask2 = ThreeD_mask_checkboard.bool()
        ## First layer
        y_context_l1 = torch.cat((y_hyper, torch.zeros_like(y_hyper)), dim=1)
        y_mean_l1, y_scale_l1 = self.entropy_parameters(y_context_l1).chunk(2, 1)
        y_scale_l1 = y_scale_l1.masked_select(mask1).reshape(cy, H * W // 2)
        y_mean_l1 = y_mean_l1.masked_select(mask1).reshape(cy, H * W // 2)
        y_loc = torch.zeros(1).to(self.lossy_device)
        y_l1_indexes = self.y_em.decompress(string_l1, loc=y_loc, scale=y_scale_l1)
        y_hat_l1 = self.quant.dequantize(y_l1_indexes, offset=y_mean_l1)
        y_hat[mask1.broadcast_to((B, cy, H, W))] = y_hat_l1.float().reshape(-1).to(self.lossy_device)

        ## Second layer
        y_ar_l2 = self.context_model_layer2(y_hat)
        y_context_l2 = torch.cat((y_hyper, y_ar_l2), dim=1)
        y_mean_l2, y_scale_l2 = self.entropy_parameters(y_context_l2).chunk(2, 1)
        y_scale_l2 = y_scale_l2.masked_select(mask2).reshape(cy, H * W // 2)
        y_mean_l2 = y_mean_l2.masked_select(mask2).reshape(cy, H * W // 2)
        y_l2_indexes = self.y_em.decompress(string_l2, loc=y_loc, scale=y_scale_l2)
        y_hat_l2 = self.quant.dequantize(y_l2_indexes, offset=y_mean_l2)
        y_hat[mask2.broadcast_to((B, cy, H, W))] = y_hat_l2.float().reshape(-1).to(self.lossy_device)
        return y_hat

    def _decompress_res_CheckerCube(self, string_l1, string_l2, u, hx, wx, r_min, r_max):
        u = u.to(self.lossless_device)
        # checkboard mask generation
        mask_checkboard = torch.ones(1, 4, hx // 2, wx // 2).to(self.lossless_device)
        mask_checkboard[:, 1:3, :, :] = 0
        mask_checkboard = F.pixel_shuffle(mask_checkboard, 2)  # mask = 0, the first layer
        if mask_checkboard.shape[-2] != hx:
            mask_checkboard = torch.cat([mask_checkboard, mask_checkboard[:, :, -2, :].unsqueeze(-2)], dim=-2)
        if mask_checkboard.shape[-1] != wx:
            mask_checkboard = torch.cat([mask_checkboard, mask_checkboard[:, :, :, -2].unsqueeze(-1)], dim=-1)
        mask_CheckerCube = torch.cat([mask_checkboard, 1 - mask_checkboard, mask_checkboard], dim=1)
        mask1 = (1 - mask_CheckerCube).bool()
        mask2 = mask_CheckerCube.bool()

        ## Init y_hat
        res = torch.zeros(u.shape[0], 3, hx, wx).to(self.lossless_device)
        sample = torch.arange(r_min, r_max + 1, dtype=torch.float).view(-1, 1, 1, 1).repeat(1, 3, hx, wx).to(self.lossless_device)
        ## First layer
        r_context_l1 = torch.cat((u[:, :, :hx, :wx], torch.zeros_like(u[:, :, :hx, :wx])), dim=1)
        r_out_l1 = self.res_entropy_parameters.to(self.lossless_device)(r_context_l1)
        r_out_l1 = self._reshape(r_out_l1)
        pmf_r = self.res_entropy(sample, r_out_l1)
        pmf_r = pmf_r.permute(1, 2, 3, 0)  # C, H, W, L
        cdf_r = float_pmf_to_integer_cdf(pmf_r)  # C, H, W, L+1
        cdf_r = cdf_r.masked_select(mask1.permute(1, 2, 3, 0))
        cdf_r1 = cdf_r.reshape(3, hx * wx // 2, -1).cpu()
        inp_r1 = torchac.decode_int16_normalized_cdf(cdf_r1, string_l1)
        res[mask1.broadcast_to((u.shape[0], 3, hx, wx))] = inp_r1.float().reshape(-1).to(self.lossless_device) + r_min
        ## Second layer
        r_ar_l2 = self.res_context_model.to(self.lossless_device)(res * (1 - mask_CheckerCube))
        r_context_l2 = torch.cat((u[:, :, :hx, :wx], r_ar_l2), dim=1)
        r_out_l2 = self.res_entropy_parameters.to(self.lossless_device)(r_context_l2)
        r_out_l2 = self._reshape(r_out_l2)
        pmf_r = self.res_entropy(sample, r_out_l2)
        pmf_r = pmf_r.permute(1, 2, 3, 0)  # C, H, W, L
        cdf_r = float_pmf_to_integer_cdf(pmf_r)  # C, H, W, L+1
        cdf_r = cdf_r.masked_select(mask2.permute(1, 2, 3, 0))
        cdf_r2 = cdf_r.reshape(3, hx * wx // 2, -1).cpu()
        inp_r2 = torchac.decode_int16_normalized_cdf(cdf_r2, string_l2)
        res[mask2.broadcast_to((u.shape[0], 3, hx, wx))] = inp_r2.float().reshape(-1).to(self.lossless_device) + r_min
        return res


class ImgComNet_Lossless_Logistic(ImgComNet_Lossless_GMM):
    def __init__(self, args, inplace=False):
        super().__init__(args)
        self.res_em = LogisticConditional()

def float_pmf_to_integer_cdf(pmf, precision=16):
    """
    :param pmf: (..., L)
    :param precision: 16
    :return: cdf: (..., L + 1)
    """
    cdf = torch.cumsum(pmf, dim=-1)
    cdf = cdf / cdf[..., -1:]  # normalize
    cdf[..., -1] = 1  # make sure the final vaule are exactly 1
    cdf = F.pad(cdf, (1, 0), mode='constant')  # the "float_unnormalized_cdf" in torchac

    max_value, Lp = 2.0 ** precision, cdf.shape[-1]
    cdf = cdf * (max_value - (Lp - 1))
    cdf = cdf.round().to(torch.int16)
    cdf.add_(torch.arange(Lp, dtype=torch.int16, device=cdf.device))
    return cdf  # the "integer_normalized_cdf " in torchac
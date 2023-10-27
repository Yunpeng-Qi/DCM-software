import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from compressai.models import (
    Cheng2020Anchor,
)
from compressai.layers import (
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from detectron2.layers import (
    Conv2d,
    DeformConv,
    FrozenBatchNorm2d,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)
import math
from data_compression.quantization import UniformQuantization
from data_compression.distributions.uniform_noised import (NoisyDeepFactorized,
                                                           NoisyNormal)
from data_compression.entropy_models import (
    ContinuousConditionalEntropyModel, ContinuousUnconditionalEntropyModel)

cfgs = {
    1: (128,),
    2: (128,),
    3: (128,),
    4: (192,),
    5: (192,),
    6: (192,),
}

class E0ToE3CompressModel(nn.Module):
    def __init__(self, N=128):
        super(E0ToE3CompressModel, self).__init__()
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
        )

    def forward(self, x):
        y = self.g_a(x)
        return y

    def inference(self, x):
        y = self.g_a(x)
        return y

class Adapter(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Adapter, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.adapter = Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(self.adapter(x))
        return out

class CodingNet(Cheng2020Anchor):
    def __init__(self, N=128, **kwargs):
        super().__init__(N=N, **kwargs)
        self.g_a = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
        )

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        params = self.h_s(z_hat)
        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))

        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}


def _load_model(
        metric, quality, pretrained=False, progress=True, **kwargs
):
    if quality not in cfgs:
        raise ValueError(f'Invalid quality value "{quality}"')

    if pretrained:
        if (
                architecture not in model_urls
                or metric not in model_urls[architecture]
                or quality not in model_urls[architecture][metric]
        ):
            raise RuntimeError("Pre-trained model not yet available")

        url = model_urls[architecture][metric][quality]
        state_dict = load_state_dict_from_url(url, progress=progress)
        state_dict = load_pretrained(state_dict)
        model = model_architectures[architecture].from_state_dict(state_dict)
        return model

    model = CodingNet(*cfgs[quality], **kwargs)
    # logger = logging.getLogger("detectron2")
    # logger.info("CompressAI's net constructure is:\n{}".format(model))
    return model

def _load_cube_model(
        metric, quality, pretrained=False, progress=True, **kwargs
):
    if quality not in cfgs:
        raise ValueError(f'Invalid quality value "{quality}"')

    if pretrained:
        if (
                architecture not in model_urls
                or metric not in model_urls[architecture]
                or quality not in model_urls[architecture][metric]
        ):
            raise RuntimeError("Pre-trained model not yet available")

        url = model_urls[architecture][metric][quality]
        state_dict = load_state_dict_from_url(url, progress=progress)
        state_dict = load_pretrained(state_dict)
        model = model_architectures[architecture].from_state_dict(state_dict)
        return model

    model = RDT_CheckerCube_base(*cfgs[quality], **kwargs)
    # logger = logging.getLogger("detectron2")
    # logger.info("CompressAI's net constructure is:\n{}".format(model))
    return model

def build_feature_coding(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        quality (int): Quality levels (1: lowest, highest: 6)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 6:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 6)')

    return _load_model(
        metric, quality, pretrained, progress, **kwargs
    )

def build_cube_feature_coding(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        quality (int): Quality levels (1: lowest, highest: 6)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 6:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 6)')

    return _load_cube_model(
        metric, quality, pretrained, progress, **kwargs
    )

def build_feature_extractor():
    model = E0ToE3CompressModel()
    return model

def build_feature_adapter(in_channel, out_channel):
    model = Adapter(in_channel, out_channel)
    # logger = logging.getLogger("detectron2")
    # logger.info("Adapter's net constructure is:\n{}".format(model))
    return model


class RDT_CheckerCube_base(CodingNet):
    """
    g_a, g_s, h_a, h_s are all from Cheng2020
    entropy_parameters is from JointAutoregressive to predict the params of Gaussian Distribution
    """
    def __init__(self, args):
        super().__init__(args)
        # self.M = args.transform_channels[-1] # M=192
        self.M = 192
        self.context_model = nn.Sequential(
            nn.Conv2d(self.N, self.N, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.N, self.N, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.N, self.N * 2, 3, 1, 1)
        )
        scale_min, scale_max, num_scales = 0.11, 256, 64
        offset = math.log(scale_min)
        factor = (math.log(scale_max) - math.log(scale_min))/(num_scales - 1)
        scale_table = torch.exp(offset + factor * torch.arange(num_scales))
        self.y_em = ContinuousConditionalEntropyModel(
            NoisyNormal, param_tables=dict(loc=[0], scale=scale_table.tolist()))
        self.z_em = ContinuousUnconditionalEntropyModel(
            NoisyDeepFactorized(batch_shape=(self.N,)))
        self.quant = UniformQuantization(step=1)

    # def forward(self, f, noisy=True):
    #     """
    #     input:
    #     f: feature extracted by E0ToE3CompressModel, 为对input 4倍下采样的结果
    #     noisy: whether training
    #     """
    #     y = self.g_a(f) # [1, 128, 64, 64]
    #     z = self.h_a(y) # 
    #     z_hat, z_indexes = self.quant(z, noisy=noisy)
    #     y_hyper = self.h_s(z_hat)   # 

    #     # checkercube mask generation
    #     H, W = y.shape[2:]
    #     mask_checkboard = torch.ones(1, 4, H // 2, W // 2).to(y.device)
    #     mask_checkboard[:, 1:3, :, :] = 0
        
    #     mask_checkboard = F.pixel_shuffle(mask_checkboard, 2)  # generate checkerboard mask [b, 1, H, W] 
    #     mask_CheckerCube_layer = torch.cat([mask_checkboard, 1 - mask_checkboard], dim=1)
    #     # mask_CheckerCube = mask_CheckerCube_layer.repeat(1, self.M // 2, 1, 1) # [1, 192, 32, 38]
    #     mask_CheckerCube = mask_CheckerCube_layer.repeat(1, self.N // 2, 1, 1) # different masks in odd / even layers

    #     # First layer: predict half of the entropy parameters of latent
    #     ## hyper only
    #     y_context_l1 = torch.cat((y_hyper, torch.zeros_like(y_hyper)), dim=1)
    #     y_mean_l1, y_scale_l1 = self.entropy_parameters(y_context_l1).chunk(2, 1) # split in dim 1 to 2 chunks

    #     # Second layer
    #     y_hat_l1, _ = self.quant(y, noisy=noisy)    # y_hat_anchor [1, 128, 32, 48]
    #     # g_cm with mask
    #     y_ar_l2 = self.context_model(y_hat_l1 * (1 - mask_CheckerCube)) # ctx_feat [1, 256, 32, 48]
    #     y_context_l2 = torch.cat((y_hyper, y_ar_l2), dim=1)
    #     y_mean_l2, y_scale_l2 = self.entropy_parameters(y_context_l2).chunk(2, 1) # y_hat_non_anchor

    #     y_means = (1 - mask_CheckerCube) * y_mean_l1 + mask_CheckerCube * y_mean_l2
    #     y_scales = (1 - mask_CheckerCube) * y_scale_l1 + mask_CheckerCube * y_scale_l2

    #     y_hat, y_indexes = self.quant(y, noisy=noisy)
    #     y_loc = torch.zeros(1).to(f.device)
    #     bits = self.y_em(y_indexes - y_means, loc=y_loc, scale=y_scales)
    #     side_bits = self.z_em(z_indexes)
    #     f_hat = self.g_s(y_hat)

    #     return f_hat, bits, side_bits

class RDT_CheckerCube(nn.Module):
    """
    g_a, g_s, h_a, h_s are all from Cheng2020
    entropy_parameters is from JointAutoregressive to predict the params of Gaussian Distribution
    """
    def __init__(self, args):
        super().__init__(args)
        self.feature_extractor = build_feature_extractor()
        self.feature_coding = build_cube_feature_coding(quality=3)
        self.feature_adapter = build_feature_adapter(128, 256)

    def forward(self, x, noisy=True):
        f = self.feature_extractor(x) # 4倍下采样
        f_hat, bits, side_bits = self.feature_coding(x, noisy=noisy)
        f_out = self.feature_adapter(f_hat)

        return f_out, bits, side_bits
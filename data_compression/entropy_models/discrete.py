import abc, math
import torch
import torch.nn as nn
import torch.nn.functional as F

import data_compression.ops as ops
from data_compression.distributions import helpers
try:
    from data_compression._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
    from data_compression import rans
except:
    pass
import numpy as np



class DiscreteUnconditionalEntropyModel(nn.Module):

    def __init__(self, prior):
        super().__init__()
        self.prior = prior

    def log_pmf(self):
        return self.prior.log_pmf

    def forward(self, one_hot):
        bits = torch.sum(one_hot * self.log_pmf().unsqueeze(1)) / (-math.log(2))
        return bits



class DiscreteConditionalEntropyModel(nn.Module):

    def __init__(self, prior_fn):
        super().__init__()
        self.prior_fn = prior_fn

    def log_pmf(self, logits):
        return self.prior_fn(logits).log_pmf

    def forward(self, one_hot, logits):
        bits = torch.sum(one_hot * self.log_pmf(logits)) / (-math.log(2))
        return bits






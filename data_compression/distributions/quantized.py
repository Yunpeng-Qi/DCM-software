import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.distribution import Distribution

import data_compression.ops as ops
from data_compression.distributions import helpers
from data_compression.distributions.distribution import DeepDistribution
from data_compression.distributions.deep_factorized import DeepFactorized
from data_compression.distributions.common import Normal, DeepNormal, \
    DeepLogistic, DeepLogisticMixture


def _logsum_expbig_minus_expsmall(big, small):
    """Stable evaluation of `Log[exp{big} - exp{small}]`.
    To work correctly, we should have the pointwise relation:  `small <= big`.
    Args:
    big: Floating-point `Tensor`
    small: Floating-point `Tensor` with same `dtype` as `big` and broadcastable
      shape.
    Returns:
    log_sub_exp: `Tensor` of same `dtype` of `big` and broadcast shape.
    """
    return big + torch.log1p(-torch.exp(small - big))


class QuantizedDistribution(nn.Module):
    """Distribution representing the quantization `Y = ceiling(X)`.
    #### Definition in Terms of Sampling
    ```
    1. Draw X
    2. Set Y <-- ceiling(X)
    3. If Y < low, reset Y <-- low
    4. If Y > high, reset Y <-- high
    5. Return Y
    ```
    #### Definition in Terms of the Probability Mass Function
    Given scalar random variable `X`, we define a discrete random variable `Y`
    supported on the integers as follows:
    ```
    P[Y = j] := P[X <= low],  if j == low,
           := P[X > high - 1],  j == high,
           := 0, if j < low or j > high,
           := P[j - 1 < X <= j],  all other j.
    ```
    Conceptually, without cutoffs, the quantization process partitions the real
    line `R` into half open intervals, and identifies an integer `j` with the
    right endpoints:
    ```
    R = ... (-2, -1](-1, 0](0, 1](1, 2](2, 3](3, 4] ...
    j = ...      -1      0     1     2     3     4  ...
    ```
    `P[Y = j]` is the mass of `X` within the `jth` interval.
    If `low = 0`, and `high = 2`, then the intervals are redrawn
    and `j` is re-assigned:
    ```
    R = (-infty, 0](0, 1](1, infty)
    j =          0     1     2
    ```
    `P[Y = j]` is still the mass of `X` within the `jth` interval.
    #### Examples
    We illustrate a mixture of discretized logistic distributions
    [(Salimans et al., 2017)][1]. This is used, for example, for capturing 16-bit
    audio in WaveNet [(van den Oord et al., 2017)][2]. The values range in
    a 1-D integer domain of `[0, 2**16-1]`, and the discretization captures
    `P(x - 0.5 < X <= x + 0.5)` for all `x` in the domain excluding the endpoints.
    The lowest value has probability `P(X <= 0.5)` and the highest value has
    probability `P(2**16 - 1.5 < X)`.
    Below we assume a `wavenet` function. It takes as `input` right-shifted audio
    samples of shape `[..., sequence_length]`. It returns a real-valued tensor of
    shape `[..., num_mixtures * 3]`, i.e., each mixture component has a `loc` and
    `scale` parameter belonging to the logistic distribution, and a `logits`
    parameter determining the unnormalized probability of that component.
    ```python
    tfd = tfp.distributions
    tfb = tfp.bijectors
    net = wavenet(inputs)
    loc, unconstrained_scale, logits = tf.split(net,
                                              num_or_size_splits=3,
                                              axis=-1)
    scale = tf.math.softplus(unconstrained_scale)
    # Form mixture of discretized logistic distributions. Note we shift the
    # logistic distribution by -0.5. This lets the quantization capture 'rounding'
    # intervals, `(x-0.5, x+0.5]`, and not 'ceiling' intervals, `(x-1, x]`.
    discretized_logistic_dist = tfd.QuantizedDistribution(
      distribution=tfd.TransformedDistribution(
          distribution=tfd.Logistic(loc=loc, scale=scale),
          bijector=tfb.Shift(shift=-0.5)),
      low=0.,
      high=2**16 - 1.)
    mixture_dist = tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(logits=logits),
      components_distribution=discretized_logistic_dist)
    neg_log_likelihood = -tf.reduce_sum(mixture_dist.log_prob(targets))
    train_op = tf.train.AdamOptimizer().minimize(neg_log_likelihood)
    ```
    After instantiating `mixture_dist`, we illustrate maximum likelihood by
    calculating its log-probability of audio samples as `target` and optimizing.
    #### References
    [1]: Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P. Kingma.
       PixelCNN++: Improving the PixelCNN with discretized logistic mixture
       likelihood and other modifications.
       _International Conference on Learning Representations_, 2017.
       https://arxiv.org/abs/1701.05517
    [2]: Aaron van den Oord et al. Parallel WaveNet: Fast High-Fidelity Speech
       Synthesis. _arXiv preprint arXiv:1711.10433_, 2017.
       https://arxiv.org/abs/1711.10433
    """

    def __init__(self,
                 base,
                 low=None,
                 high=None):
        """Construct a Quantized Distribution representing `Y = ceiling(X)`.
        Some properties are inherited from the distribution defining `X`. Example:
        `allow_nan_stats` is determined for this `QuantizedDistribution` by reading
        the `distribution`.
        Args:
          distribution:  The base distribution class to transform. Typically an
            instance of `Distribution`.
          low: `Tensor` with same `dtype` as this distribution and shape
            that broadcasts to that of samples but does not result in additional
            batch dimensions after broadcasting. Should be a whole number. Default
            `None`. If provided, base distribution's `prob` should be defined at
            `low`.
          high: `Tensor` with same `dtype` as this distribution and shape
            that broadcasts to that of samples but does not result in additional
            batch dimensions after broadcasting. Should be a whole number. Default
            `None`. If provided, base distribution's `prob` should be defined at
            `high - 1`. `high` must be strictly greater than `low`.
          validate_args: Python `bool`, default `False`. When `True` distribution
            parameters are checked for validity despite possibly degrading runtime
            performance. When `False` invalid inputs may silently render incorrect
            outputs.
          name: Python `str` name prefixed to Ops created by this class.
        Raises:
          TypeError: If `dist_cls` is not a subclass of
              `Distribution` or continuous.
          NotImplementedError:  If the base distribution does not implement `cdf`.
        """
        super().__init__()
        self._base = base
        self._low = low
        self._high = high

    @property
    def base(self):
        """Base distribution, p(x)."""
        return self._base

    @property
    def low(self):
        """Lowest value that quantization returns."""
        return self._low

    @property
    def high(self):
        """Highest value that quantization returns."""
        return self._high

    @property
    def batch_shape(self):
        return self.base.batch_shape

    def log_prob(self, y):
        # Changes of mass are only at the integers, so we must use tf.floor in our
        # computation of log_cdf/log_sf.  Floor now, since
        # tf.floor(y - 1) can incur unwanted rounding near powers of two, but
        # tf.floor(y) - 1 can't.
        y = torch.floor(y)
        if not (hasattr(self.base, 'log_cdf') or
                hasattr(self.base, 'cdf')):
            raise NotImplementedError(
              '`log_prob` not implemented unless the base distribution implements '
              '`log_cdf`')
        try:
            log_prob = self._log_prob_with_logsf_and_logcdf(y)
        except:
            log_prob = self._log_prob_with_logcdf(y)
        # log_prob = self._log_prob_with_logsf_and_logcdf(y)
        return ops.lower_bound(log_prob, math.log(1e-15))

    def _log_prob_with_logcdf(self, y):
        low = self._low
        high = self._high
        return _logsum_expbig_minus_expsmall(
            self._log_cdf(y, low=low, high=high),
            self._log_cdf(y - 1., low=low, high=high))

    def _log_prob_with_logsf_and_logcdf(self, y):
        """Compute log_prob(y) using log survival_function and cdf together."""
        # There are two options that would be equal if we had infinite precision:
        # Log[ sf(y - 1) - sf(y) ]
        #   = Log[ exp{logsf(y - 1)} - exp{logsf(y)} ]
        # Log[ cdf(y) - cdf(y - 1) ]
        #   = Log[ exp{logcdf(y)} - exp{logcdf(y - 1)} ]
        low = self._low
        high = self._high
        logsf_y = self._log_survival_function(y, low=low, high=high)
        logsf_y_minus_1 = self._log_survival_function(y - 1., low=low, high=high)
        logcdf_y = self._log_cdf(y, low=low, high=high)
        logcdf_y_minus_1 = self._log_cdf(y - 1., low=low, high=high)

        # Important:  Here we use select in a way such that no input is inf, this
        # prevents the troublesome case where the output of select can be finite,
        # but the output of grad(select) will be NaN.

        # In either case, we are doing Log[ exp{big} - exp{small} ]
        # We want to use the sf items precisely when we are on the right side of the
        # median, which occurs when logsf_y < logcdf_y.
        condition = logsf_y < logcdf_y
        big = torch.where(condition, logsf_y_minus_1, logcdf_y)
        small = torch.where(condition, logsf_y, logcdf_y_minus_1)

        return _logsum_expbig_minus_expsmall(big, small)

    def _log_cdf(self, y, low=None, high=None):
        low = self._low if low is None else low
        high = self._high if high is None else high
        # Recall the promise:
        # cdf(y) := P[Y <= y]
        #         = 1, if y >= high,
        #         = 0, if y < low,
        #         = P[X <= y], otherwise.

        # P[Y <= j] = P[floor(Y) <= j] since mass is only at integers, not in
        # between.
        j = torch.floor(y)

        result_so_far = self.base.log_cdf(j)

        # Re-define values at the cutoffs.
        if low is not None:
            const = torch.tensor([-np.inf], device=y.device, dtype=y.dtype)
            result_so_far = torch.where(j < low, const, result_so_far)

        if high is not None:
            const = torch.tensor([0], device=y.device, dtype=y.dtype)
            result_so_far = torch.where(j < high, result_so_far, const)

        return result_so_far

    def _cdf(self, y, low=None, high=None):
        low = self._low if low is None else low
        high = self._high if high is None else high

        # Recall the promise:
        # cdf(y) := P[Y <= y]
        #         = 1, if y >= high,
        #         = 0, if y < low,
        #         = P[X <= y], otherwise.

        # P[Y <= j] = P[floor(Y) <= j] since mass is only at integers, not in
        # between.
        j = torch.floor(y)

        # P[X <= j], used when low < X < high.
        result_so_far = self.base.cdf(j)

        # Re-define values at the cutoffs.
        if low is not None:
            const = torch.tensor([0], device=y.device, dtype=y.dtype)
            result_so_far = torch.where(j < low, const, result_so_far)

        if high is not None:
            const = torch.tensor([1], device=y.device, dtype=y.dtype)
            result_so_far = torch.where(j < high, result_so_far, const)

        return result_so_far

    def _log_survival_function(self, y, low=None, high=None):
        low = self._low if low is None else low
        high = self._high if high is None else high

        # Recall the promise:
        # survival_function(y) := P[Y > y]
        #                       = 0, if y >= high,
        #                       = 1, if y < low,
        #                       = P[X > y], otherwise.

        # P[Y > j] = P[ceiling(Y) > j] since mass is only at integers, not in
        # between.
        j = torch.ceil(y)

        # P[X > j], used when low < X < high.
        result_so_far = self.base.log_survival_function(j)

        # Re-define values at the cutoffs.
        if low is not None:
            const = torch.tensor([0], device=y.device, dtype=y.dtype)
            result_so_far = torch.where(j < low, const, result_so_far)

        if high is not None:
            const = torch.tensor([-np.inf], device=y.device, dtype=y.dtype)
            result_so_far = torch.where(j < high, result_so_far, const)

        return result_so_far


    def _survival_function(self, y, low=None, high=None):
        low = self._low if low is None else low
        high = self._high if high is None else high

        # Recall the promise:
        # survival_function(y) := P[Y > y]
        #                       = 0, if y >= high,
        #                       = 1, if y < low,
        #                       = P[X > y], otherwise.

        # P[Y > j] = P[ceiling(Y) > j] since mass is only at integers, not in
        # between.
        j = torch.ceil(y)

        # P[X > j], used when low < X < high.
        result_so_far = self.base.survival_function(j)

        # Re-define values at the cutoffs.
        if low is not None:
            result_so_far = torch.where(
                j < low, 1, result_so_far)

        if high is not None:
            result_so_far = torch.where(
                j < high, result_so_far, 0)

        return result_so_far



class QuantizedDeepFactorized(QuantizedDistribution):

    def __init__(self, low, high, **kwargs):
        super().__init__(DeepFactorized(**kwargs), low, high)


class QuantizedNormal(QuantizedDistribution):

    def __init__(self, **kwargs):
        super().__init__(Normal(**kwargs))


class QuantizedDeepLogistic(QuantizedDistribution):

    def __init__(self, low, high, **kwargs):
        super().__init__(DeepLogistic(**kwargs), low, high)

class QuantizedDeepLogisticMixture(QuantizedDistribution):

    def __init__(self, low, high, **kwargs):
        super().__init__(DeepLogisticMixture(**kwargs), low, high)
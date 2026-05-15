"""Core utilities — initialization and normalization primitives."""

import math

import torch


def trunc_normal_init_(
    tensor: torch.Tensor,
    std: float = 1.0,
    lower: float = -2.0,
    upper: float = 2.0,
) -> torch.Tensor:
    """JAX-compatible truncated normal initialization (in-place).

    Unlike PyTorch's nn.init.trunc_normal_, this correctly compensates the
    standard deviation so that the initialized tensor has the requested std.
    Based on JAX/Flax's default truncated normal initializer.
    """
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
            return tensor

        sqrt2 = math.sqrt(2)

        erf_lower = math.erf(lower / sqrt2)
        erf_upper = math.erf(upper / sqrt2)
        prob_mass = (erf_upper - erf_lower) / 2

        inv_sqrt_2pi = (2 * math.pi) ** -0.5
        phi_lower = inv_sqrt_2pi * math.exp(-0.5 * lower ** 2)
        phi_upper = inv_sqrt_2pi * math.exp(-0.5 * upper ** 2)

        trunc_var = (
            1.0
            - (upper * phi_upper - lower * phi_lower) / prob_mass
            - ((phi_lower - phi_upper) / prob_mass) ** 2
        )
        comp_std = std / math.sqrt(trunc_var)

        tensor.uniform_(erf_lower, erf_upper)
        tensor.erfinv_()
        tensor.mul_(sqrt2 * comp_std)
        tensor.clamp_(lower * comp_std, upper * comp_std)

    return tensor


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    """RMSNorm as a pure function with no learnable parameters.

    Computes root-mean-square normalization in float32 precision and casts
    the result back to the input dtype.
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)

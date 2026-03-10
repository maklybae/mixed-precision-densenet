"""
Differentiable Quantization (DQ) — Case U3: θ = [d, q_max]

Based on:
    Uhlich et al., "Mixed Precision DNNs: All you need is a good parametrization"
    https://arxiv.org/abs/1905.11452
"""

import math

import torch
import torch.nn as nn


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.round()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.floor()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return RoundSTE.apply(x)


def floor_ste(x: torch.Tensor) -> torch.Tensor:
    return FloorSTE.apply(x)


def quantize_to_pow2(x: torch.Tensor) -> torch.Tensor:
    return 2.0 ** round_ste(torch.log2(x))


def clamp_ste(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return x + (x.clamp(min_val, max_val) - x).detach()


class UniformQuantU3(nn.Module):
    def __init__(
        self,
        signed: bool = True,
        d_init: float = 2**-4,
        xmax_init: float = 1.0,
        d_min: float = 2**-8,
        d_max: float = 2**8,
        xmax_min: float = 1e-3,
        xmax_max: float = 10.0,
    ):
        super().__init__()
        self.signed = signed
        self.d_min = d_min
        self.d_max = d_max
        self.xmax_min = xmax_min
        self.xmax_max = xmax_max

        self.d = nn.Parameter(torch.tensor(float(d_init)))
        self.q_max = nn.Parameter(torch.tensor(float(xmax_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_clamped = self.d.clamp(self.d_min, self.d_max)
        d_q = quantize_to_pow2(d_clamped)

        xmax = self.q_max.clamp(self.xmax_min, self.xmax_max)

        if self.signed:
            x_min = -xmax
        else:
            x_min = torch.zeros_like(xmax)

        x_clamped = torch.clamp(x, x_min, xmax)
        return d_q * round_ste(x_clamped / d_q)

    @torch.no_grad()
    def bitwidth(self) -> float:
        d_clamped = self.d.clamp(self.d_min, self.d_max)
        d_q = quantize_to_pow2(d_clamped)
        xmax = self.q_max.clamp(self.xmax_min, self.xmax_max)

        n_levels = xmax / d_q + 1.0
        b = math.ceil(math.log2(max(n_levels.item(), 1.0)))
        if self.signed:
            b += 1
        return max(b, 2)

    def bitwidth_soft(self) -> torch.Tensor:
        """Used in penalty terms for bitwidth regularization."""
        d_clamped = self.d.clamp(self.d_min, self.d_max)
        xmax = self.q_max.clamp(self.xmax_min, self.xmax_max)

        b = torch.log2(xmax / d_clamped + 1.0)
        if self.signed:
            b = b + 1.0
        return b

    def extra_repr(self) -> str:
        return (
            f"signed={self.signed}, "
            f"d={self.d.item():.6f}, q_max={self.q_max.item():.4f}, "
            f"bitwidth≈{self.bitwidth()}"
        )


def init_weight_quantizer(
    weight: torch.Tensor,
    init_bitwidth: int = 4,
    d_min: float = 2**-8,
    d_max: float = 2**8,
    xmax_min: float = 1e-3,
    xmax_max: float = 10.0,
) -> UniformQuantU3:
    """Create a weight quantizer initialized from a pre-trained weight tensor."""
    w_absmax = weight.abs().max().item()
    n_levels = 2 ** (init_bitwidth - 1) - 1  # signed: 2^{b-1} - 1

    # Step size: power-of-two
    # Reference uses ceil for bw > 4, floor for bw <= 4
    if w_absmax < 1e-10:
        d_init = d_min
    elif init_bitwidth > 4:
        d_init = 2 ** math.ceil(math.log2(w_absmax / n_levels))
    else:
        d_init = 2 ** math.floor(math.log2(w_absmax / n_levels))
    d_init = max(min(d_init, d_max), d_min)

    # Dynamic range
    xmax_init = d_init * n_levels
    xmax_init = max(min(xmax_init, xmax_max), xmax_min)

    return UniformQuantU3(
        signed=True,
        d_init=d_init,
        xmax_init=xmax_init,
        d_min=d_min,
        d_max=d_max,
        xmax_min=xmax_min,
        xmax_max=xmax_max,
    )


def init_activation_quantizer(
    act_min: float = 0.0,
    act_max: float = 1.0,
    init_bitwidth: int = 4,
    d_min: float = 2**-8,
    d_max: float = 2**8,
    xmax_min: float = 1e-3,
    xmax_max: float = 10.0,
) -> UniformQuantU3:
    """Create an activation quantizer (unsigned, for post-ReLU activations)."""
    # For unsigned: n_levels = 2^b - 1
    n_levels = 2**init_bitwidth - 1

    dynamic_range = max(abs(act_max), abs(act_min))
    if dynamic_range < 1e-10:
        dynamic_range = 1.0

    d_init = 2 ** math.floor(math.log2(dynamic_range / n_levels))
    d_init = max(min(d_init, d_max), d_min)

    xmax_init = d_init * n_levels
    xmax_init = max(min(xmax_init, xmax_max), xmax_min)

    return UniformQuantU3(
        signed=False,
        d_init=d_init,
        xmax_init=xmax_init,
        d_min=d_min,
        d_max=d_max,
        xmax_min=xmax_min,
        xmax_max=xmax_max,
    )

"""Running normalisers and layer-normalised MLPs, ported from the xmimic reference.

xmimic keeps a running mean/var for every network input and for the world-model
residual output, updates them from data (never by gradients) and stores them with
the weights. Without them the world model sees joint speeds of order 10 rad/s
next to heights of order 0.3 m, which is the main reason a small MLP fits poorly.
"""
from __future__ import annotations

import torch
from torch import nn


class RunningNorm(nn.Module):
    """Streaming mean/var with the same merge rule as xmimic ``rn_update``."""

    def __init__(self, dim: int, eps: float = 1e-2):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(self.dim))
        self.register_buffer("var", torch.ones(self.dim))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: torch.Tensor, max_count: float = 2e8):
        x = x.reshape(-1, x.shape[-1]).float()
        count = float(self.count)
        if count >= max_count or x.shape[0] == 0:
            return self
        n = x.shape[0]
        mean, var = x.mean(0), x.var(0, unbiased=False)
        total = count + n
        delta = mean - self.mean
        self.mean += delta*n/total
        self.var = (self.var*count + var*n + delta.square()*count*n/total)/total
        self.count += n
        return self

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean)/(self.var + self.eps**2).sqrt()

    def std(self) -> torch.Tensor:
        return (self.var + self.eps**2).sqrt()

    def extra_repr(self) -> str:
        return f"dim={self.dim}, count={float(self.count):.0f}"


def normed_mlp(in_dim: int, hidden, out_dim: int, zero_last: bool = False) -> nn.Sequential:
    """Linear -> LayerNorm -> SiLU per hidden layer, plain Linear head (xmimic ``layer_norm=True``)."""
    layers: list[nn.Module] = []
    width = in_dim
    for size in hidden:
        layers += [nn.Linear(width, size), nn.LayerNorm(size), nn.SiLU()]
        width = size
    head = nn.Linear(width, out_dim)
    if zero_last:
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)

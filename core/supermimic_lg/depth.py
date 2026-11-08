"""Egocentric depth perception: pinhole rays plus a compact encoder.

The rays mimic a forward-down looking depth camera mounted on the trunk. Both
physics backends use :func:`depth_rays`, so the CPU validation run and the GPU
training run observe the same geometry.
"""
import math
import torch
from torch import nn

DEPTH_PITCH = 30.0          # degrees below horizontal
DEPTH_OFFSET = (0.16, 0.0, 0.05)   # trunk-local mount: forward, left, up


def parse_depth_size(text):
    """Parse ``"18x24"`` (height x width) or ``"18"`` (square)."""
    parts = str(text).lower().replace("*", "x").split("x")
    if len(parts) > 2:
        raise ValueError("depth size must look like 18x24")
    height = int(parts[0])
    width = int(parts[1]) if len(parts) == 2 else height
    if height <= 0 or width <= 0:
        raise ValueError("depth size must be positive")
    return height, width


def depth_rays(height, width, fov, pitch=DEPTH_PITCH, offset=DEPTH_OFFSET, device="cpu"):
    """Camera-local ray origins and unit directions, row-major ``[height, width]``.

    The camera looks along +X (robot forward), +Y left, +Z up, then pitches down
    by ``pitch`` degrees. All rays share one origin, like a real pinhole camera.
    """
    device = torch.device(device)
    tan_half = math.tan(math.radians(float(fov))*0.5)
    aspect = width/height
    rows = torch.arange(height, device=device, dtype=torch.float32)
    cols = torch.arange(width, device=device, dtype=torch.float32)
    v, u = torch.meshgrid(rows, cols, indexing="ij")
    x = ((u + .5)/width - .5)*2*tan_half*aspect
    y = (.5 - (v + .5)/height)*2*tan_half
    center = torch.stack((torch.ones_like(x), x, y), -1)
    angle = math.radians(float(pitch))
    c, s = math.cos(angle), math.sin(angle)
    down = torch.tensor(((c, 0., s), (0., 1., 0.), (-s, 0., c)), device=device)
    directions = center @ down.T
    directions = directions/directions.norm(dim=-1, keepdim=True)
    origins = torch.tensor(offset, device=device).expand(height*width, 3).clone()
    return origins, directions.reshape(-1, 3)


def depth_spec(size="18x24", fov=60.0, range_m=3.0, feat=32, pitch=DEPTH_PITCH,
               noise=0.0, dropout=0.0, latency=0):
    height, width = parse_depth_size(size)
    if noise < 0 or not 0 <= dropout < 1 or latency < 0:
        raise ValueError("depth noise/dropout/latency out of range")
    return dict(height=height, width=width, fov=float(fov), range=float(range_m),
                feat=int(feat), pitch=float(pitch), pixels=height*width,
                noise=float(noise), dropout=float(dropout), latency=int(latency))


def corrupt_depth(image, spec, generator=None):
    """Stereo-like range error plus random invalid pixels.

    ``noise`` is the coefficient of a quadratic range model (sigma = noise * z^2),
    which matches how stereo depth error grows with distance. Dropped pixels read
    as the maximum range, exactly like a returned "no reading".
    """
    noise, dropout = spec.get("noise", 0.), spec.get("dropout", 0.)
    if not noise and not dropout:
        return image
    out = image
    if noise:
        sigma = noise*out.clamp_min(1e-3).square()
        out = out + sigma*torch.randn(out.shape, device=out.device, generator=generator)
    if dropout:
        lost = torch.rand(out.shape, device=out.device, generator=generator) < dropout
        out = torch.where(lost, torch.full_like(out, spec["range"]), out)
    return out.clamp(0., spec["range"])


class DepthLagBuffer:
    """Delays the sensed depth by a fixed number of control steps."""

    def __init__(self, latency, like):
        self.latency = int(latency)
        self.slots = [like.clone() for _ in range(self.latency+1)]

    def push(self, image, ids=None):
        if ids is None:
            self.slots = self.slots[1:] + [image.clone()] if self.latency else [image.clone()]
        else:
            for slot in self.slots:
                slot[ids] = image[ids]
        return self.slots[0]


class DepthEncoder(nn.Module):
    """Small convolutional encoder: ``[N, H, W]`` metres to ``[N, feat]``."""

    def __init__(self, spec):
        super().__init__()
        self.spec = dict(spec)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(3))
        self.head = nn.Linear(32*9, spec["feat"])

    def forward(self, depth):
        """``depth`` is ``[N, H, W]`` metres; ``1.0`` marks a miss at max range."""
        normalized = (depth/self.spec["range"]).clamp(0., 1.)[:, None]
        features = self.conv(normalized).flatten(1)
        return torch.tanh(self.head(features))

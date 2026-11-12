"""Contact probability losses shared by collection diagnostics and WM fitting."""
import torch
from torch.nn import functional as F


def contact_bce(probabilities, targets):
    """Return per-state FP32 BCE, preserving the differentiable contact model.

    CUDA rejects probability BCE inside autocast even when its inputs are FP32.
    Disable autocast around the loss explicitly; network matmuls keep their AMP
    settings. Clamp after converting to FP32 so the upper bound stays below one.
    """
    with torch.autocast(device_type=probabilities.device.type, enabled=False):
        return F.binary_cross_entropy(probabilities.float().clamp(1e-6, 1-1e-6),
                                      targets.float(), reduction="none").mean(-1)

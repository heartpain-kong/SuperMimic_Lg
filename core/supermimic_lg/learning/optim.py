"""Optimizer steps and exception-safe parameter freezing for BPTT."""
from contextlib import contextmanager
import torch


def active_autocast_dtype(device_type):
    """Read AMP state while supporting the declared PyTorch 2.1 minimum."""
    if hasattr(torch, 'get_autocast_dtype'):
        return torch.get_autocast_dtype(device_type) if torch.is_autocast_enabled(device_type) else None
    if device_type == 'cuda':
        return torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else None
    if device_type == 'cpu':
        return torch.get_autocast_cpu_dtype() if torch.is_autocast_cpu_enabled() else None
    return None


@contextmanager
def frozen_parameters(*models):
    """Freeze parameters while preserving gradients with respect to model inputs."""
    original = [(p, p.requires_grad) for model in models for p in model.parameters()]
    try:
        for p, _ in original:
            p.requires_grad_(False)
        yield
    finally:
        for p, enabled in original:
            p.requires_grad_(enabled)


def optimizer_step(loss, model, optimizer, grad_clip=1., grad_value_clip=0.):
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite learning loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    if grad_value_clip:
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_value_clip)
    optimizer.step()
    # Several updates can share the runner's autocast scope. Cached low-
    # precision parameter copies must not survive an in-place optimizer step.
    torch.clear_autocast_cache()
    return norm.detach()

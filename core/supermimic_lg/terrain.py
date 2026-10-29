"""Progress gates for reference-driven terrain training."""
import torch


def curriculum_decisions(displacement, reference_displacement, steps, failed):
    """Promote only sustained, command-aligned travel beyond the spawn platform.

    Short reference tails and standing episodes hold their level. A physical
    failure always demotes; sufficiently long but stalled episodes also demote.
    """
    distance = reference_displacement.norm(dim=-1)
    direction = reference_displacement / distance.clamp_min(1e-6).unsqueeze(-1)
    progress = (displacement * direction).sum(dim=-1)
    eligible = (steps >= 50) & (distance >= 2.)
    up = eligible & ~failed & (progress >= torch.maximum(distance * .7, distance.new_tensor(2.)))
    down = failed | (eligible & (progress < distance * .4))
    return up, down, progress

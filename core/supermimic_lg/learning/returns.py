"""Soft-termination returns shared by tracking and velocity learning.

All transition tensors are time-major [H, B]; values include the bootstrap
state [H+1, B]. Termination[t] belongs to the transition s_t -> s_{t+1}.
"""
import torch


def survival_weights(termination):
    return torch.cat((torch.ones_like(termination[:1]), (1.-termination).cumprod(0)), 0)


def discounted_objective(rewards, termination, bootstrap, gamma=.99):
    """Per-trajectory analytic return; keep gradients through survival and V_H."""
    if rewards.shape != termination.shape or rewards.shape[0] < 1:
        raise ValueError("rewards and transition terminations must have equal nonempty [H,B] shapes")
    survival = survival_weights(termination)
    discounts = gamma ** torch.arange(len(rewards), device=rewards.device, dtype=rewards.dtype)
    return (discounts[:, None]*survival[:-1]*rewards).sum(0) + gamma**len(rewards)*survival[-1]*bootstrap


@torch.no_grad()
def lambda_returns(rewards, termination, values, gamma=.99, lam=.95):
    """GAE(lambda) regression targets, without dividing cumulative survivals."""
    if rewards.shape != termination.shape or values.shape != (len(rewards)+1, *rewards.shape[1:]):
        raise ValueError("expected rewards/done [H,B] and values [H+1,B]")
    targets = torch.empty_like(rewards)
    following = values[-1]
    for t in reversed(range(len(rewards))):
        following = rewards[t] + gamma*(1.-termination[t])*((1.-lam)*values[t+1]+lam*following)
        targets[t] = following
    return targets

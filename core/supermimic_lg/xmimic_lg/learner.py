"""XMimic-style optimization and averaged deployment policy for LgDog.

This is a PyTorch adaptation, not the native JAX/XLA G1 trainer.
"""
from dataclasses import dataclass
import math
import torch
from ..learning.velocity import VelocityLearner, VelocityLearningConfig
from .optim import MuonAdam
from .ema import PolicyEMA
from .dynamics import XMimicStateSpec
from .mirror import augment_windows


@dataclass(frozen=True)
class XMimicLearningConfig(VelocityLearningConfig):
    wm_batch_size: int = 512
    policy_batch_size: int = 256
    wm_horizon: int = 8
    horizon: int = 32
    actor_lr: float = 4e-4
    critic_lr: float = 4e-4
    critic_epochs: int = 5
    normalize_actor_loss: bool = False
    policy_ema_decay: float = .995
    wm_mirror_fraction: float = 0.
    lr_decay_iterations: int = 250000
    policy_lr_mult: float = 4.
    policy_lr_decay_start: int = 1400
    policy_lr_decay_length: int = 300

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.policy_ema_decay < 1 or not 0 <= self.wm_mirror_fraction <= 1:
            raise ValueError("invalid EMA decay or mirror fraction")
        if self.lr_decay_iterations < 1 or self.policy_lr_decay_start < 0 or self.policy_lr_decay_length < 1:
            raise ValueError("learning-rate schedule lengths must be positive, start >=0")
        if not math.isfinite(self.policy_lr_mult) or self.policy_lr_mult <= 0:
            raise ValueError("policy LR multiplier must be finite and positive")


class XMimicLearner(VelocityLearner):
    def __init__(self, world_model, policy, critic, task, cfg=None):
        cfg = cfg or XMimicLearningConfig()
        if world_model.encoder is not None:
            raise ValueError("XMimic-Lg currently supports blind proprioception; use SuperMimic for depth")
        super().__init__(world_model, policy, critic, task, cfg)
        world_model.spec = XMimicStateSpec(12)
        world_model.out_norm.eps = 1e-4
        self.optimizers = [MuonAdam(world_model, cfg.wm_lr),
                           MuonAdam(policy, cfg.actor_lr, betas=(.9, cfg.actor_beta2)),
                           torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)]
        self.ema = PolicyEMA(policy, cfg.policy_ema_decay)
        self.iteration = 0
        self.set_iteration(0)

    def set_iteration(self, iteration):
        self.iteration = int(iteration)
        cfg = self.cfg
        fraction = min(1., self.iteration/cfg.lr_decay_iterations)
        scale = .03+.97*.5*(1.+math.cos(math.pi*fraction))
        ramp = min(1., max(0., self.iteration-cfg.policy_lr_decay_start)/cfg.policy_lr_decay_length)
        policy_mult = cfg.policy_lr_mult+(1.-cfg.policy_lr_mult)*ramp
        for optimizer, rate in zip(self.optimizers, (cfg.wm_lr*scale, cfg.actor_lr*scale*policy_mult, cfg.critic_lr*scale)):
            for group in optimizer.param_groups:
                group["lr"] = rate

    def update_world_model(self, rows):
        return super().update_world_model(augment_windows(rows, self.cfg.wm_mirror_fraction))

    def update_policy(self, rows):
        metrics = super().update_policy(rows)
        with self.timings.measure('policy_ema'):
            self.ema.update(self.policy)
        metrics.update({"xmimic/ema_updates": self.ema.updates,
                        "xmimic/wm_lr": self.optimizers[0].param_groups[0]["lr"],
                        "xmimic/actor_lr": self.optimizers[1].param_groups[0]["lr"]})
        return metrics

    def state_dict(self):
        return {**super().state_dict(), "ema": self.ema.state_dict(), "iteration": self.iteration}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.ema.load_state_dict(state["ema"])
        self.set_iteration(state["iteration"])

    def export(self):
        return dict(policy_ema=self.ema.model.state_dict(),
                    deployment_policy="ema" if self.ema.updates else "raw")

"""Velocity WM and actor/critic learning, independent of collection and storage."""
from dataclasses import asdict, dataclass
import math
import torch
from torch.nn import functional as F

from .optim import frozen_parameters, optimizer_step, active_autocast_dtype
from .losses import contact_bce
from .profiling import LearnerTimings
from .returns import discounted_objective, lambda_returns
from .rollout import imagine
from ..tasks.velocity.schema import FRAME_DIM
from ..tasks.velocity.task import VelocityCondition
from ..tasks.velocity.imagination import imagine_batched, critic_features, CompiledRecurrentStep


@dataclass(frozen=True)
class VelocityLearningConfig:
    wm_batch_size: int = 128
    policy_batch_size: int = 128
    wm_horizon: int = 4
    horizon: int = 8
    wm_lr: float = 4e-4
    actor_lr: float = 3e-4
    critic_lr: float = 2e-4
    actor_beta2: float = .95
    gamma: float = .99
    gae_lambda: float = .95
    critic_epochs: int = 3
    critic_minibatches: int = 4
    grad_clip: float = 1.
    actor_grad_value: float = .1
    action_bound_weight: float = 1.
    reward_limit: float = 10.
    value_target_limit: float = 200.
    wm_loss_scale: str = "fixed"
    normalize_actor_loss: bool = True

    def __post_init__(self):
        for key in ("wm_batch_size", "policy_batch_size", "wm_horizon", "horizon",
                    "critic_epochs", "critic_minibatches"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        for key in ("wm_lr", "actor_lr", "critic_lr", "grad_clip"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("action_bound_weight", "actor_grad_value", "reward_limit", "value_target_limit"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and non-negative")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1 or not 0 <= self.actor_beta2 < 1:
            raise ValueError("invalid discount, GAE lambda or actor beta2")
        if self.wm_loss_scale not in ("fixed", "increment"):
            raise ValueError("invalid WM loss scale")

    @classmethod
    def from_args(cls, args):
        defaults = asdict(cls())
        values = {key: getattr(args, key, None) for key in defaults}
        values = {key: defaults[key] if value is None else value for key, value in values.items()}
        for key in ("wm_batch_size", "policy_batch_size"):
            values[key] = args.batch_size if getattr(args, key, None) is None else getattr(args, key)
        return cls(**values)


class VelocityLearner:
    def __init__(self, world_model, policy, critic, task, cfg=None):
        self.wm, self.policy, self.critic, self.task = world_model, policy, critic, task
        self.cfg = cfg or VelocityLearningConfig()
        self.optimizers = [torch.optim.AdamW(model.parameters(), lr=lr,
                            betas=(.9, self.cfg.actor_beta2) if model is policy else (.9, .999))
                           for model, lr in ((world_model, self.cfg.wm_lr),
                                             (policy, self.cfg.actor_lr), (critic, self.cfg.critic_lr))]
        self.counts = dict(wm=0, actor=0, critic=0)
        self.timings = LearnerTimings(next(policy.parameters()).device)
        self.rollout_mode = 'batched'
        self._compiled_step = None

    def fields(self, policy=False):
        names = (["state", "context", "history", "previous", "command", "phase"] if policy
                 else ["state", "next", "context", "action"])
        return names + (["depth"] if self.wm.encoder is not None else [])

    def context(self, rows):
        context = rows["context"][0]
        if "depth" in rows:
            spec = self.wm.encoder.spec
            image = rows["depth"][0].float().reshape(-1, spec["height"], spec["width"])
            context = torch.cat((context, self.wm.encode_depth(image)), -1)
        return context

    def update_world_model(self, rows):
        cfg = self.cfg
        state, context = rows["state"][0], self.context(rows)
        self.wm.update_norms(state, rows["action"][0], context, rows["next"][0])
        physical, contacts = [], []
        for t in range(cfg.wm_horizon):
            state = self.wm.step(state, rows["action"][t], context)
            state = torch.where(rows["valid"][t, :, None].bool(), state, rows["next"][t].detach())
            physical.append(self.wm.prediction_error(state, rows["next"][t]) if cfg.wm_loss_scale == "increment"
                            else self.wm.spec.normalized_sq_error(state[:, :37], rows["next"][t, :, :37]))
            contacts.append(contact_bce(state[:, 37:], rows["next"][t, :, 37:]))
        valid = rows["valid"]
        state_loss = (torch.stack(physical)*valid).sum()/valid.sum().clamp_min(1.)
        contact_loss = (torch.stack(contacts)*valid).sum()/valid.sum().clamp_min(1.)
        optimizer_step(state_loss+contact_loss, self.wm, self.optimizers[0], cfg.grad_clip)
        self.counts["wm"] += 1
        return {"wm/state_loss": state_loss.detach(), "wm/contact_loss": contact_loss.detach()}

    def update_policy(self, rows):
        cfg = self.cfg
        with torch.no_grad():
            context = self.context(rows)
        history = rows["history"][0]
        depth = history[:, -1, FRAME_DIM:] if "depth" in rows else None
        condition = VelocityCondition(rows["command"][0], rows["phase"][0], depth)
        with frozen_parameters(self.wm, self.critic):
            with self.timings.measure('actor_rollout'):
                if self.rollout_mode not in ('reference', 'batched', 'compiled'):
                    raise ValueError('unknown rollout execution mode')
                runtime = {}
                if self.rollout_mode == 'compiled':
                    if self._compiled_step is None:
                        self._compiled_step = CompiledRecurrentStep(self.wm,self.policy,self.task)
                    runtime['step_fn'] = self._compiled_step
                rollout_fn = imagine if self.rollout_mode == 'reference' else imagine_batched
                rollout = rollout_fn(self.wm, self.policy, self.task, rows["state"][0], context,
                    history, rows["previous"][0], condition, cfg.horizon, reward_limit=cfg.reward_limit, **runtime)
                last_features = critic_features(rollout, self.task, self.wm, context, last_only=True)
                bootstrap = self.critic(last_features[0])
                objective = discounted_objective(rollout.rewards, rollout.termination, bootstrap, cfg.gamma)
                if not torch.isfinite(objective).all() or float(objective.detach().abs().max()) > 1e4:
                    raise FloatingPointError("imagined return exploded (|J| > 1e4); inspect model accuracy")
                bounds = rollout.action_bounds.mean()
                actor_loss = -objective.mean()/(cfg.horizon if cfg.normalize_actor_loss else 1.)
                actor_loss = actor_loss + cfg.action_bound_weight*bounds
            with self.timings.measure('actor_backward'):
                grad = optimizer_step(actor_loss, self.policy, self.optimizers[1], cfg.grad_clip, cfg.actor_grad_value)
            self.counts["actor"] += 1

        with self.timings.measure('critic_targets'), torch.no_grad():
            features = critic_features(rollout, self.task, self.wm, context)
            # All values and the fixed GAE targets must use the same normalization
            # as the critic regression, including the final bootstrap state.
            self.critic.norm.update(features)
            normalized = self.critic.norm.apply(features)
            dtype = active_autocast_dtype(normalized.device.type)
            if dtype is not None:
                # The first Linear would cast every gathered minibatch to this
                # dtype anyway. Quantize the fixed regression batch once.
                normalized = normalized.to(dtype)
            values = self.critic.forward_normalized(normalized.flatten(0, 1)).reshape(cfg.horizon+1, -1)
            targets = lambda_returns(rollout.rewards, rollout.termination, values, cfg.gamma, cfg.gae_lambda)
            if cfg.value_target_limit:
                targets = targets.clamp(-cfg.value_target_limit, cfg.value_target_limit)
            inputs_x, targets_y = normalized[:-1].flatten(0, 1), targets.flatten()
        torch.clear_autocast_cache()  # Separate no-grad target evaluation from trainable critic forwards.
        total = len(targets_y)
        minibatches = min(cfg.critic_minibatches, total)
        loss = None
        with self.timings.measure('critic_fit'):
            for _ in range(cfg.critic_epochs):
                # tensor_split covers the remainder as well as N < minibatches.
                for index in torch.randperm(total, device=inputs_x.device).tensor_split(minibatches):
                    loss = F.mse_loss(self.critic.forward_normalized(inputs_x[index]), targets_y[index])
                    optimizer_step(loss, self.critic, self.optimizers[2], cfg.grad_clip)
                    self.counts["critic"] += 1
        return {"policy/actor_objective": objective.detach().mean(), "policy/actor_loss": actor_loss.detach(),
                "policy/action_bound_loss": bounds.detach(), "policy/critic_loss": loss.detach(),
                "policy/policy_grad_norm": grad, "policy/survival_end": rollout.survival[-1].detach().mean(),
                "policy/termination_mean": rollout.termination.detach().mean()}

    def state_dict(self):
        return {"config": asdict(self.cfg), "updates": dict(self.counts)}

    def load_state_dict(self, state):
        frozen = lambda cfg: {key: value for key, value in cfg.items()
                              if key not in ("wm_batch_size", "policy_batch_size")}
        if frozen(state["config"]) != frozen(asdict(self.cfg)):
            raise ValueError("learner configuration differs from checkpoint")
        self.counts = dict(state["updates"])

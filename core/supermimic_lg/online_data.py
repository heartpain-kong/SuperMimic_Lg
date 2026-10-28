"""Motion-conditioned observations and an episode-aware online transition ring."""
from __future__ import annotations

import torch
import math

from .config import lgdog1_spec
from .data.motion import load_motion_npz
from .imagination import ReferenceBatch
from .state import quat_apply_inverse, quat_normalize
from .training import ReplayWindow


def observation(state, previous_action, context):
    default = state.new_tensor(lgdog1_spec().default_joint_pos)
    gravity = quat_apply_inverse(state[..., 3:7], state.new_tensor([0., 0., -1.]).expand(state.shape[:-1] + (3,)))
    return torch.cat((state[..., 7:19] - default + context[..., 1:13], state[..., 25:37],
                      state[..., 22:25], gravity, previous_action), dim=-1)


def advance_history(state, action, context, history):
    obs = observation(state, action, context)
    if history.ndim == obs.ndim + 1:
        if history.shape[-1] != obs.shape[-1]:
            raise ValueError("history frame width does not match observation")
        return torch.cat((history[..., 1:, :], obs.unsqueeze(-2)), dim=-2)
    return torch.cat((history[..., obs.shape[-1]:], obs), dim=-1)


class MotionSequence:
    def __init__(self, path, goal_horizon, device):
        clip = load_motion_npz(path, expected_joints=12, expected_bodies=13)
        if not math.isfinite(clip.fps) or abs(clip.fps - 50.) > 1e-4:
            raise ValueError("LgDog control runs at 50 Hz; resample motion to 50 fps before training")
        self.length, self.goal_horizon = clip.frames, goal_horizon
        self.pos, self.quat, self.lin, self.ang, self.q, self.qd = [
            torch.tensor(x, dtype=torch.float32, device=device) for x in
            (clip.body_pos_w, clip.body_quat_w, clip.body_lin_vel_w, clip.body_ang_vel_w, clip.joint_pos, clip.joint_vel)]
        if (self.quat.norm(dim=-1) < .5).any():
            raise ValueError("motion contains invalid wxyz quaternions")
        self.quat = quat_normalize(self.quat)
        limits = self.q.new_tensor(lgdog1_spec().joint_pos_limits)
        if ((self.q < limits[:, 0] - .05) | (self.q > limits[:, 1] + .05)).any():
            raise ValueError("reference joint positions exceed LgDog limits; check radians and joint order")
        gravity = quat_apply_inverse(self.quat[:, 0], self.q.new_tensor([0., 0., -1.]).expand(self.length, 3))
        w = quat_apply_inverse(self.quat[:, 0], self.ang[:, 0])
        v = quat_apply_inverse(self.quat[:, 0], self.lin[:, 0])
        self.features = torch.cat((self.q, self.qd, gravity, w, v, self.pos[:, 0, 2:3]), dim=-1)
        self.states = torch.cat((self.pos[:, 0], self.quat[:, 0], self.q, self.lin[:, 0], w, self.qd), dim=-1)

    def goal(self, frame):
        offsets = torch.arange(1, self.goal_horizon + 1, device=frame.device)
        indices = (frame[..., None] + offsets).clamp_max(self.length - 1)
        return self.features[indices].flatten(-2)

    def future(self, frame, horizon):
        indices = (frame[None] + torch.arange(horizon + 1, device=frame.device)[:, None]).clamp_max(self.length - 1)
        return self.goal(indices), ReferenceBatch(self.pos[indices], self.quat[indices], self.lin[indices],
                                                 self.ang[indices], self.pos[indices, 0])

    def validate_kinematics(self, fk):
        indices = torch.linspace(0, self.length - 1, min(32, self.length), device=self.q.device).long()
        with torch.no_grad():
            pos, quat, _, _ = fk(self.states[indices])
            error = (pos - self.pos[indices]).norm(dim=-1).max().item()
            rotation_error = (1. - (quat * self.quat[indices]).sum(-1).abs().clamp_max(1.)).max().item()
        if error > .02 or rotation_error > .005:
            raise ValueError(f"motion does not match LgDog XML link origins/order: position error={error:.4f}m, quaternion error={rotation_error:.4f}")


class OnlineReplay:
    def __init__(self, capacity, num_envs, history_dim, device):
        self.capacity, self.num_envs, self.count = capacity, num_envs, 0
        def zeros(*dims):
            return torch.zeros(capacity, num_envs, *dims, device=device)
        self.states, self.next_states = zeros(37), zeros(37)
        self.actions, self.previous_actions = zeros(12), zeros(12)
        history_shape = (history_dim,) if isinstance(history_dim, int) else tuple(history_dim)
        self.contexts, self.histories = zeros(16), zeros(*history_shape)
        self.terminal, self.reset = zeros(), zeros()
        self.frames = torch.zeros(capacity, num_envs, dtype=torch.long, device=device)

    @torch.no_grad()
    def add(self, state, next_state, executed_action, previous_action, context, history, frame, terminal, reset):
        i = self.count % self.capacity
        for dest, src in ((self.states, state), (self.next_states, next_state), (self.actions, executed_action),
                          (self.previous_actions, previous_action), (self.contexts, context), (self.histories, history),
                          (self.frames, frame), (self.terminal, terminal), (self.reset, reset)):
            dest[i].copy_(src)
        self.count += 1

    def sample_window(self, batch_size, horizon, motion):
        if min(self.capacity, self.count) < horizon:
            raise ValueError("not enough replay steps")
        device = self.states.device
        start = torch.randint(max(0, self.count - self.capacity), self.count - horizon + 1, (batch_size,), device=device)
        env = torch.randint(self.num_envs, (batch_size,), device=device)
        index = (start[None] + torch.arange(horizon, device=device)[:, None]) % self.capacity
        reset = self.reset[index, env]
        valid = torch.cat((torch.ones_like(reset[:1]), (1. - reset[:-1]).cumprod(0)), dim=0)
        states = torch.cat((self.states[index[:1], env], self.next_states[index, env]), dim=0)
        refs = (self.frames[index, env] + 1).clamp_max(motion.length - 1)
        return ReplayWindow(states, self.actions[index, env], self.contexts[index[0], env],
                            self.terminal[index, env], motion.pos[refs, 0], valid, motion.quat[refs, 0])

    def sample_starts(self, batch_size, horizon, motion):
        indices = torch.arange(max(0, self.count - self.capacity), self.count, device=self.states.device) % self.capacity
        frame = self.frames[indices]
        eligible = (frame + horizon < motion.length).nonzero()
        if eligible.numel() == 0:
            raise ValueError("no replay starts with enough reference frames for imagination")
        sampled = eligible[torch.randint(len(eligible), (batch_size,), device=indices.device)]
        i, e = indices[sampled[:, 0]], sampled[:, 1]
        return (self.states[i, e], self.contexts[i, e], self.histories[i, e], self.frames[i, e], self.previous_actions[i, e])

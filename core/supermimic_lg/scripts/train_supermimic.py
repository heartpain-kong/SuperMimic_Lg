"""LgDog online SuperMimic: physics collection -> WM fitting -> imagined BPTT."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import time

import torch

from supermimic_lg.actor_critic import ActorCriticConfig, ImaginedActorCritic
from supermimic_lg.config import WorldModelConfig, lgdog1_spec
from supermimic_lg.kinematics import LgDogKinematics
from supermimic_lg.online_data import MotionSequence, OnlineReplay, advance_history, observation
from supermimic_lg.policy import Critic, Policy, PolicyShape
from supermimic_lg.rewards import tracking_reward
from supermimic_lg.simulator import MjlabSimulator, MujocoSimulator, load_lgdog_model, resolve_robot_xml
from supermimic_lg.state import quat_apply_inverse
from supermimic_lg.training import WorldModelTrainer
from supermimic_lg.world_model import WorldModel


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Trainer:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)
        torch.set_num_threads(args.torch_threads)
        self.robot = lgdog1_spec()
        self.output = args.output_dir.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.xml = resolve_robot_xml(args.mjlab_root, args.robot_xml)
        self.motion = MotionSequence(args.motion_file, args.goal_horizon, args.device)
        if self.motion.length <= args.horizon + args.goal_horizon + 1:
            raise ValueError("motion is too short: need more than horizon + goal_horizon + 1 frames")
        self.fk = LgDogKinematics(load_lgdog_model(self.xml), args.device)
        self.motion.validate_kinematics(self.fk)
        self.shape = PolicyShape(12, 13, history=args.history, goal_horizon=args.goal_horizon)
        self.wm_cfg = WorldModelConfig(hidden=(args.network_size,) * 3)
        self.wm = WorldModel(self.robot, cfg=self.wm_cfg, device=args.device)
        self.policy = Policy(self.shape, (args.network_size,) * 2, (args.network_size,) * 2).to(args.device)
        self.critic = Critic(self.shape, 16, hidden=(args.network_size,) * 2).to(args.device)
        self.wm_trainer = WorldModelTrainer(self.wm)
        self.ac = ImaginedActorCritic(self.wm, self.policy, self.critic, ActorCriticConfig(horizon=args.horizon))
        self.replay = OnlineReplay(args.replay_capacity, args.num_envs, (self.shape.history, self.shape.obs_dim), args.device)
        self.iteration = self.wm_updates = self.policy_updates = 0
        self.metadata = {"motion_sha256": file_hash(args.motion_file), "xml_sha256": file_hash(self.xml),
                         "policy_shape": asdict(self.shape), "network_size": args.network_size,
                         "state_convention": "root_link_pos_w,quat_wxyz,q,lin_vel_w,ang_vel_b,qd",
                         "context_mode": "fixed_friction_0.6_zero_encoder_bias_zero_com_offset"}
        if args.resume:
            self.load(args.resume)
            if args.iterations <= self.iteration:
                raise ValueError(f"--iterations is a total target and must exceed checkpoint iteration {self.iteration}")
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        (self.output / "config.json").write_text(json.dumps({"args": config, **self.metadata}, indent=2), encoding="utf-8")
        self.sim = MjlabSimulator(args.mjlab_root, args.num_envs, args.device) if args.backend == "mjlab" else MujocoSimulator(self.xml, args.num_envs, args.device)
        self.frames = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
        self.previous_action = torch.zeros(args.num_envs, 12, device=args.device)
        self.history = torch.zeros(args.num_envs, args.history, self.shape.obs_dim, device=args.device)
        self.episode_steps = torch.zeros_like(self.frames)
        self.episode_return = torch.zeros(args.num_envs, device=args.device)
        self.reset(torch.arange(args.num_envs, device=args.device))

    @torch.no_grad()
    def reset(self, ids):
        if not len(ids):
            return
        frames = torch.randint(self.motion.length - self.args.horizon - self.args.goal_horizon, (len(ids),), device=ids.device)
        self.sim.reset(ids, self.motion.states[frames])
        self.frames[ids] = frames
        self.previous_action[ids] = 0.
        state = self.sim.state()[ids]
        first_frame = observation(state, self.previous_action[ids], self.sim.context[ids])
        self.history[ids] = first_frame[:, None].expand(-1, self.args.history, -1)
        self.episode_steps[ids] = 0
        self.episode_return[ids] = 0.

    @torch.no_grad()
    def collect(self):
        rewards, failures, finished, lengths, returns = [], 0, 0, [], []
        for _ in range(self.args.collect_steps):
            state = self.sim.state()
            clean = self.policy(self.history, self.motion.goal(self.frames))
            executed = (clean + self.args.action_noise * torch.randn_like(clean)).clamp(-1., 1.)
            next_state = self.sim.step(executed)
            if not torch.isfinite(next_state).all():
                raise FloatingPointError("physics generated non-finite states; inspect model/control configuration")
            next_frame = self.frames + 1
            ref_gravity = quat_apply_inverse(self.motion.quat[next_frame, 0], state.new_tensor([0., 0., -1.]).expand(len(state), 3))
            gravity = quat_apply_inverse(next_state[:, 3:7], state.new_tensor([0., 0., -1.]).expand(len(state), 3))
            terminal = ((next_state[:, 2] - self.motion.pos[next_frame, 0, 2]).abs() > self.robot.height_threshold) | ((gravity - ref_gravity).norm(dim=-1) > self.robot.tilt_threshold)
            self.episode_steps += 1
            reset = terminal | (next_frame >= self.motion.length - 1) | (self.episode_steps >= self.args.episode_steps)
            reward, _ = tracking_reward(*self.fk(next_state), self.motion.pos[next_frame], self.motion.quat[next_frame],
                                        self.motion.lin[next_frame], self.motion.ang[next_frame], next_state[:, 7:19],
                                        state.new_tensor(self.robot.joint_pos_limits), clean, self.previous_action)
            # Save the pre-reset terminal state and the actually executed noisy action.
            self.replay.add(state, next_state, executed, self.previous_action, self.sim.context,
                            self.history, self.frames, terminal, reset)
            self.history = advance_history(next_state, clean, self.sim.context, self.history)
            self.previous_action = clean
            self.frames = next_frame
            self.episode_return += reward
            ids = reset.nonzero().flatten()
            if len(ids):
                lengths.extend(self.episode_steps[ids].tolist())
                returns.extend(self.episode_return[ids].tolist())
            finished += len(ids)
            failures += int(terminal.sum())
            rewards.append(float(reward.mean()))
            self.reset(ids)
        return {"real_reward": sum(rewards) / len(rewards), "failures": failures, "episodes": finished,
                "episode_length": sum(lengths) / max(1, len(lengths)), "episode_return": sum(returns) / max(1, len(returns))}

    def save(self):
        path = self.output / "checkpoint.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save({"algorithm": "supermimic_bptt", "iteration": self.iteration,
                    "wm_updates": self.wm_updates, "policy_updates": self.policy_updates,
                    "wm": self.wm.state_dict(), "policy": self.policy.state_dict(), "critic": self.critic.state_dict(),
                    "wm_opt": self.wm_trainer.optimizer.state_dict(), "policy_opt": self.ac.policy_opt.state_dict(),
                    "critic_opt": self.ac.critic_opt.state_dict(), "metadata": self.metadata,
                    "world_model_config": asdict(self.wm_cfg), "rng_state": torch.get_rng_state()}, tmp)
        tmp.replace(path)
        print(json.dumps({"checkpoint": str(path), "iteration": self.iteration, "policy_updates": self.policy_updates}), flush=True)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.args.device, weights_only=True)
        if checkpoint.get("algorithm") != "supermimic_bptt" or checkpoint.get("metadata") != self.metadata:
            saved_history = checkpoint.get("metadata", {}).get("policy_shape", {}).get("history", "unknown")
            raise ValueError("checkpoint algorithm, robot XML, motion, or policy architecture does not match this run; "
                             f"checkpoint history={saved_history}, current history={self.args.history}. "
                             "Resume with the checkpoint's history length, or start a new run for a different input shape.")
        for module, key in ((self.wm, "wm"), (self.policy, "policy"), (self.critic, "critic")):
            module.load_state_dict(checkpoint[key])
        for optimizer, key in ((self.wm_trainer.optimizer, "wm_opt"), (self.ac.policy_opt, "policy_opt"), (self.ac.critic_opt, "critic_opt")):
            optimizer.load_state_dict(checkpoint[key])
        self.iteration, self.wm_updates, self.policy_updates = checkpoint["iteration"], checkpoint["wm_updates"], checkpoint["policy_updates"]
        torch.set_rng_state(checkpoint["rng_state"].cpu())

    def run(self):
        started = time.monotonic()
        first = self.iteration
        print(json.dumps({"algorithm": "SuperMimic BPTT", "backend": self.args.backend, "device": self.args.device,
                          "robot": "lgdog1", "state_dim": 37, "action_dim": 12, "num_envs": self.args.num_envs,
                          "observation_history_shape": [self.args.num_envs, self.args.history, self.shape.obs_dim],
                          "resume_iteration": first, "context_mode": self.metadata["context_mode"]}), flush=True)
        try:
            for iteration in range(first + 1, self.args.iterations + 1):
                metrics = self.collect()
                if self.replay.count >= self.args.wm_horizon:
                    for _ in range(self.args.wm_updates):
                        window = self.replay.sample_window(self.args.batch_size, self.args.wm_horizon, self.motion)
                        metrics.update({"wm/" + k: v for k, v in self.wm_trainer.update(window).items()})
                        self.wm_updates += 1
                    if iteration - first >= self.args.warmup_iterations:
                        for _ in range(self.args.policy_updates):
                            state, context, history, frame, previous = self.replay.sample_starts(self.args.batch_size, self.args.horizon, self.motion)
                            goals, refs = self.motion.future(frame, self.args.horizon)
                            result = self.ac.update(state, context, history, goals, refs, previous, self.fk, advance_history)
                            metrics.update({"policy/" + k: v for k, v in result.items()})
                            self.policy_updates += 1
                        action = previous.detach().requires_grad_(True)
                        prediction = self.wm.step(state.detach(), action, context.detach())
                        gradient, = torch.autograd.grad(prediction.square().mean(), action)
                        metrics["dynamics_action_grad_norm"] = float(gradient.norm())
                self.iteration = iteration
                metrics.update(iteration=iteration, wm_updates=self.wm_updates, policy_updates=self.policy_updates,
                               transitions=self.replay.count * self.args.num_envs, elapsed_s=time.monotonic() - started)
                if not all(torch.isfinite(torch.tensor(v)) for v in metrics.values()):
                    raise FloatingPointError("non-finite training metrics")
                with (self.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(metrics) + "\n")
                print(json.dumps(metrics), flush=True)
                if iteration % self.args.save_interval == 0:
                    self.save()
        except KeyboardInterrupt:
            print("SuperMimic stop requested; saving current models.", flush=True)
        finally:
            self.save()
            self.sim.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mjlab-root", type=Path, default=Path("../unitree_rl_mjlab_lgdog"))
    p.add_argument("--robot-xml", type=Path)
    p.add_argument("--motion-file", type=Path, required=True)
    p.add_argument("--backend", choices=("mjlab", "mujoco"), default="mjlab")
    p.add_argument("--device", default="cuda:0")
    for name, default in (("num-envs", 64), ("iterations", 10000), ("collect-steps", 16), ("wm-updates", 10),
                          ("policy-updates", 2), ("warmup-iterations", 5), ("batch-size", 128), ("wm-horizon", 4),
                          ("horizon", 8), ("history", 3), ("goal-horizon", 10), ("replay-capacity", 256),
                          ("network-size", 128), ("episode-steps", 500), ("save-interval", 10), ("torch-threads", 4)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--action-noise", type=float, default=.1)
    p.add_argument("--output-dir", type=Path, default=Path("logs/supermimic_lgdog"))
    p.add_argument("--resume", type=Path)
    return p


def main():
    p = parser()
    args = p.parse_args()
    positive = ("num_envs", "iterations", "collect_steps", "wm_updates", "policy_updates", "warmup_iterations", "batch_size",
                "wm_horizon", "horizon", "history", "goal_horizon", "replay_capacity", "network_size", "episode_steps", "save_interval", "torch_threads")
    if any(getattr(args, x) <= 0 for x in positive) or args.replay_capacity < args.wm_horizon or not 0 <= args.action_noise <= 1:
        p.error("sizes/counts must be positive, replay capacity >= wm horizon, action noise in [0,1]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA is unavailable; use --backend mujoco --device cpu for real CPU physics")
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    Trainer(args).run()


if __name__ == "__main__":
    main()

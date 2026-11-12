"""LgDog online SuperMimic: physics collection -> WM fitting -> imagined BPTT."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from supermimic_lg.checkpointing import atomic_save, file_hash, resume_contract
from supermimic_lg.learning.schedule import CollectionSchedule
from supermimic_lg.actor_critic import ActorCriticConfig, ImaginedActorCritic
from supermimic_lg.config import WorldModelConfig, lgdog1_spec
from supermimic_lg.kinematics import LgDogKinematics
from supermimic_lg.online_data import MotionSequence, OnlineReplay, advance_history, observation
from supermimic_lg.policy import Critic, Policy, PolicyShape
from supermimic_lg.rewards import tracking_reward
from supermimic_lg.simulator import MjlabSimulator, MujocoSimulator, load_lgdog_model, resolve_robot_xml
from supermimic_lg.state import quat_apply_inverse
from supermimic_lg.training import WorldModelTrainer
from supermimic_lg.terrain import curriculum_decisions
from supermimic_lg.world_model import WorldModel




class Trainer:
    def __init__(self, args):
        self.args = args
        if args.backend == "mjlab" and not args.device.startswith("cuda:"):
            raise ValueError("mjlab requires a CUDA device; use mujoco for CPU physics")
        if args.terrain == "rough" and args.backend != "mjlab":
            raise ValueError("rough terrain requires the mjlab backend")
        torch.manual_seed(args.seed)
        torch.set_num_threads(args.torch_threads)
        self.robot = lgdog1_spec()
        self.output = args.output_dir.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.xml = resolve_robot_xml(args.mjlab_root, args.robot_xml)
        self.motion = MotionSequence(args.motion_file, args.goal_horizon, args.device)
        if args.terrain == "rough":
            root_xy = self.motion.pos[:, 0, :2]
            if float((root_xy.amax(0) - root_xy.amin(0)).norm()) < .5:
                raise ValueError("rough terrain needs a moving reference motion; standing.npz cannot teach terrain traversal")
        if self.motion.length <= args.horizon + args.goal_horizon + 1:
            raise ValueError("motion is too short: need more than horizon + goal_horizon + 1 frames")
        self.fk = LgDogKinematics(load_lgdog_model(self.xml), args.device)
        self.motion.validate_kinematics(self.fk)
        self.shape = PolicyShape(12, 13, history=args.history, goal_horizon=args.goal_horizon)
        self.wm_cfg = WorldModelConfig(hidden=args.wm_hidden)
        self.wm = WorldModel(self.robot, cfg=self.wm_cfg, device=args.device)
        self.policy = Policy(self.shape, args.actor_hidden, args.actor_hidden).to(args.device)
        self.critic = Critic(self.shape, 16, hidden=args.critic_hidden).to(args.device)
        self.wm_trainer = WorldModelTrainer(self.wm, lr=getattr(args, "wm_lr", None) or 3e-4)
        ac_options = {"horizon": args.horizon}
        for key in ("gamma", "gae_lambda", "critic_epochs", "critic_minibatches", "normalize_actor_loss",
                    "actor_beta2", "actor_grad_value", "reward_limit", "value_target_limit"):
            if getattr(args, key, None) is not None:
                ac_options[key] = getattr(args, key)
        for key, name in (("actor_lr", "policy_lr"), ("critic_lr", "critic_lr")):
            if getattr(args, key, None) is not None:
                ac_options[name] = getattr(args, key)
        self.ac = ImaginedActorCritic(self.wm, self.policy, self.critic, ActorCriticConfig(**ac_options))
        self.schedule = CollectionSchedule(args.collect_steps, getattr(args, "update_to_data_ratio", None))
        self.wm_batch_size = getattr(args, "wm_batch_size", None) or args.batch_size
        self.policy_batch_size = getattr(args, "policy_batch_size", None) or args.batch_size
        self.replay = OnlineReplay(args.replay_capacity, args.num_envs, (self.shape.history, self.shape.obs_dim), args.device)
        self.iteration = self.wm_updates = self.policy_updates = 0
        self.collected_transitions = 0
        self.metadata = {"training_core": "modular_v5", "actor_critic": asdict(self.ac.cfg),
                         "collection_schedule": asdict(self.schedule), "wm_lr": self.wm_trainer.optimizer.param_groups[0]["lr"],
                         "wm_batch_size": self.wm_batch_size, "policy_batch_size": self.policy_batch_size,
                         "motion_sha256": file_hash(args.motion_file), "xml_sha256": file_hash(self.xml),
                         "policy_shape": asdict(self.shape),
                         "wm_hidden": list(args.wm_hidden), "actor_hidden": list(args.actor_hidden),
                         "critic_hidden": list(args.critic_hidden),
                         "terrain": args.terrain,
                         "state_convention": "root_link_pos_w,quat_wxyz,q,lin_vel_w,ang_vel_b,qd",
                         "context_mode": "fixed_friction_0.6_zero_encoder_bias_zero_com_offset"}
        if args.resume:
            self.load(args.resume)
            if args.iterations <= self.iteration:
                raise ValueError(f"--iterations is a total target and must exceed checkpoint iteration {self.iteration}")
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        (self.output / "config.json").write_text(json.dumps({"args": config, **self.metadata}, indent=2), encoding="utf-8")
        self.sim = (MjlabSimulator(args.mjlab_root, args.num_envs, args.device, terrain=args.terrain, seed=args.seed)
                    if args.backend == "mjlab" else MujocoSimulator(self.xml, args.num_envs, args.device))
        self.frames = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
        self.previous_action = torch.zeros(args.num_envs, 12, device=args.device)
        self.history = torch.zeros(args.num_envs, args.history, self.shape.obs_dim, device=args.device)
        self.episode_steps = torch.zeros_like(self.frames)
        self.episode_return = torch.zeros(args.num_envs, device=args.device)
        self.episode_start_xy = torch.zeros(args.num_envs, 2, device=args.device)
        self.episode_reference_xy = torch.zeros_like(self.episode_start_xy)
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
        self.episode_start_xy[ids] = state[:, :2]
        self.episode_reference_xy[ids] = self.motion.pos[frames, 0, :2]
        first_frame = observation(state, self.previous_action[ids], self.sim.context[ids])
        self.history[ids] = first_frame[:, None].expand(-1, self.args.history, -1)
        self.episode_steps[ids] = 0
        self.episode_return[ids] = 0.

    @torch.no_grad()
    def collect(self, collect_steps=None):
        collect_steps = self.args.collect_steps if collect_steps is None else collect_steps
        rewards, failures, finished, lengths, returns = [], 0, 0, [], []
        promotions = demotions = 0
        for _ in range(collect_steps):
            state = self.sim.state()
            clean = self.policy(self.history, self.motion.goal(self.frames))
            executed = (clean + self.args.action_noise * torch.randn_like(clean)).clamp(-1., 1.)
            next_state = self.sim.step(executed)
            self.collected_transitions += self.args.num_envs
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
                if self.args.terrain == "rough":
                    up, down, _ = curriculum_decisions(
                        next_state[ids, :2] - self.episode_start_xy[ids],
                        self.motion.pos[next_frame[ids], 0, :2] - self.episode_reference_xy[ids],
                        self.episode_steps[ids], terminal[ids])
                    self.sim.update_terrain_curriculum(ids, up, down)
                    promotions += int(up.sum())
                    demotions += int(down.sum())
            finished += len(ids)
            failures += int(terminal.sum())
            rewards.append(float(reward.mean()))
            self.reset(ids)
        metrics = {"real_reward": sum(rewards) / len(rewards), "failures": failures, "episodes": finished,
                   "episode_length": sum(lengths) / max(1, len(lengths)), "episode_return": sum(returns) / max(1, len(returns))}
        if self.args.terrain == "rough":
            metrics.update(self.sim.terrain_metrics())
            metrics.update({"terrain/promotions": promotions, "terrain/demotions": demotions})
        return metrics

    def save(self):
        path = self.output / "checkpoint.pt"
        atomic_save({"algorithm": "supermimic_bptt", "iteration": self.iteration,
                    "collected_transitions": self.collected_transitions,
                    "wm_updates": self.wm_updates, "policy_updates": self.policy_updates,
                    "wm": self.wm.state_dict(), "policy": self.policy.state_dict(), "critic": self.critic.state_dict(),
                    "wm_opt": self.wm_trainer.optimizer.state_dict(), "policy_opt": self.ac.policy_opt.state_dict(),
                    "critic_opt": self.ac.critic_opt.state_dict(), "metadata": self.metadata,
                    "world_model_config": asdict(self.wm_cfg), "rng_state": torch.get_rng_state()}, path)
        print(json.dumps({"checkpoint": str(path), "iteration": self.iteration, "policy_updates": self.policy_updates}), flush=True)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.args.device, weights_only=True)
        saved_metadata = dict(checkpoint.get("metadata", {}))
        # Checkpoints created before the terrain option existed are flat runs.
        saved_metadata.setdefault("terrain", "flat")
        if checkpoint.get("algorithm") != "supermimic_bptt" or resume_contract(saved_metadata) != resume_contract(self.metadata):
            saved_history = checkpoint.get("metadata", {}).get("policy_shape", {}).get("history", "unknown")
            raise ValueError("checkpoint algorithm, robot XML, motion, or policy architecture does not match this run; "
                             f"checkpoint history={saved_history}, current history={self.args.history}. "
                             "Resume with the checkpoint's history length, or start a new run for a different input shape.")
        for module, key in ((self.wm, "wm"), (self.policy, "policy"), (self.critic, "critic")):
            module.load_state_dict(checkpoint[key])
        for optimizer, key in ((self.wm_trainer.optimizer, "wm_opt"), (self.ac.policy_opt, "policy_opt"), (self.ac.critic_opt, "critic_opt")):
            optimizer.load_state_dict(checkpoint[key])
        self.iteration, self.wm_updates, self.policy_updates = checkpoint["iteration"], checkpoint["wm_updates"], checkpoint["policy_updates"]
        self.collected_transitions = checkpoint.get("collected_transitions", 0)
        torch.set_rng_state(checkpoint["rng_state"].cpu())

    def run(self):
        started = time.monotonic()
        first = self.iteration
        print(json.dumps({"algorithm": "SuperMimic BPTT", "backend": self.args.backend, "device": self.args.device,
                          "robot": "lgdog1", "state_dim": 37, "action_dim": 12, "num_envs": self.args.num_envs,
                          "terrain": self.args.terrain,
                          "observation_history_shape": [self.args.num_envs, self.args.history, self.shape.obs_dim],
                          "resume_iteration": first, "context_mode": self.metadata["context_mode"]}), flush=True)
        try:
            for iteration in range(first + 1, self.args.iterations + 1):
                collection_steps = self.schedule.steps(iteration-1)
                metrics = self.collect(collection_steps)
                if self.replay.count >= self.args.wm_horizon:
                    for _ in range(self.args.wm_updates):
                        window = self.replay.sample_window(self.wm_batch_size, self.args.wm_horizon, self.motion)
                        metrics.update({"wm/" + k: v for k, v in self.wm_trainer.update(window).items()})
                        self.wm_updates += 1
                    if iteration - first >= self.args.warmup_iterations:
                        for _ in range(self.args.policy_updates):
                            state, context, history, frame, previous = self.replay.sample_starts(self.policy_batch_size, self.args.horizon, self.motion)
                            goals, refs = self.motion.future(frame, self.args.horizon)
                            result = self.ac.update(state, context, history, goals, refs, previous, self.fk, advance_history)
                            metrics.update({"policy/" + k: v for k, v in result.items()})
                            self.policy_updates += 1
                        action = previous.detach().requires_grad_(True)
                        prediction = self.wm.step(state.detach(), action, context.detach())
                        gradient, = torch.autograd.grad(prediction.square().mean(), action)
                        metrics["dynamics_action_grad_norm"] = float(gradient.norm())
                self.iteration = iteration
                metrics.update(iteration=iteration, collection_steps=collection_steps, wm_updates=self.wm_updates, policy_updates=self.policy_updates,
                               transitions=self.collected_transitions,
                               replay_transitions=self.replay.count * self.args.num_envs, elapsed_s=time.monotonic() - started)
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

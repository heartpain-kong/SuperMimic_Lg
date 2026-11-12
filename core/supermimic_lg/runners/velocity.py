"""Online, reference-free velocity MBRL using terrain-conditioned BPTT."""
import json
from dataclasses import asdict
import math
import time
from pathlib import Path
import torch
from torch.nn import functional as F

from supermimic_lg.checkpointing import atomic_save, file_hash, resume_contract, policy_state, VELOCITY_ALGORITHMS
from supermimic_lg.learning.velocity import VelocityLearner, VelocityLearningConfig
from supermimic_lg.learning.schedule import CollectionSchedule
from supermimic_lg.learning.optim import optimizer_step
from supermimic_lg.tasks.velocity.task import VelocityTask
from supermimic_lg.velocity_objective import objective_from, command_settings, sample_commands
from supermimic_lg.simulator import resolve_robot_xml
from supermimic_lg.state import heading_quat, quat_apply
from supermimic_lg.depth import DepthEncoder, DepthLagBuffer, corrupt_depth, depth_spec
from supermimic_lg.velocity_physics import VelocitySimulator
from supermimic_lg.tasks.velocity.networks import VelocityWorldModel, VelocityPolicy, VelocityCritic
from supermimic_lg.tasks.velocity.replay import VelocityReplay
from supermimic_lg.tasks.velocity.mdp import FootKinematics, frame, phase_at, failure, velocity_error, reward
from supermimic_lg.tasks.velocity.schema import frame_dim, context_dim, CONTEXT_DIM


def oom_hint(error, args, free_gib=None):
    """Actionable sizes for the next attempt, halving the terms that scale memory."""
    message = str(error).strip().splitlines()[0] if str(error).strip() else "CUDA out of memory"
    hint = dict(error="cuda_out_of_memory", message=message,
        current=dict(num_envs=args.num_envs, replay_capacity=args.replay_capacity,
            batch_size=args.batch_size, horizon=args.horizon, history=args.history, terrain=args.terrain),
        suggested=dict(num_envs=max(1, args.num_envs//2),
            replay_capacity=max(args.wm_horizon, args.replay_capacity//2),
            batch_size=max(1, args.batch_size//2), horizon=max(1, args.horizon//2)),
        note=("回放缓存随 num_envs × replay_capacity 增长，BPTT 峰值随 batch_size × horizon 增长；"
              "先把并行环境数减半并降低 replay-capacity、批大小和预测步长，或先用平地地形验证。"))
    if free_gib is not None:
        hint["free_gib_on_failure"] = free_gib
    if any(getattr(args, key, None) is not None for key in ("wm_batch_size", "policy_batch_size")):
        for key in ("wm_batch_size", "policy_batch_size"):
            size = getattr(args, key, None) or args.batch_size
            hint["current"][key] = size
            hint["suggested"][key] = max(1, size//2)
    return hint


class VelocityTrainer:
    def __init__(self, args):
        self.started = time.monotonic()
        self.args = args
        self.algorithm = getattr(args, "algorithm", "supermimic")
        if self.algorithm == "xmimic" and (args.depth or args.task != "velocity"):
            raise ValueError("XMimic-Lg supports blind velocity training only")
        self.checkpoint_algorithm = "xmimic_velocity_bptt" if self.algorithm == "xmimic" else "supermimic_velocity_bptt"
        self.objective = objective_from(args)
        self.command_config = command_settings(args)
        from supermimic_lg.benchmark import hardware
        self.training_hardware = hardware()
        torch.manual_seed(args.seed)
        torch.set_num_threads(args.torch_threads)
        if args.backend == 'mjlab':
            from mjlab.utils.torch import configure_torch_backends
            configure_torch_backends()
        self.output = args.output_dir.resolve()
        if (self.output/'config.json').exists() and not args.resume:
            raise FileExistsError("choose a fresh output directory, or resume its checkpoint explicitly")
        self.output.mkdir(parents=True, exist_ok=True)
        self.xml = resolve_robot_xml(args.mjlab_root, args.robot_xml)
        self.depth_spec = (depth_spec(args.depth_size, args.depth_fov, args.depth_range, args.depth_feat,
                                      noise=args.depth_noise, dropout=args.depth_dropout,
                                      latency=args.depth_latency)
                           if args.depth else None)
        self.depth_feature_dim = 0 if self.depth_spec is None else self.depth_spec["feat"]
        encoder = None if self.depth_spec is None else DepthEncoder(self.depth_spec).to(args.device)
        self.wm = VelocityWorldModel(args.wm_hidden, encoder).to(args.device)
        self.policy = VelocityPolicy(args.history, args.actor_hidden, self.depth_feature_dim).to(args.device)
        self.critic = VelocityCritic(args.critic_hidden, self.depth_feature_dim).to(args.device)
        self.feet = FootKinematics(self.xml, args.device)
        self.task = VelocityTask(self.feet, self.objective)
        if self.algorithm == "xmimic":
            from supermimic_lg.xmimic_lg.learner import XMimicLearner, XMimicLearningConfig
            self.learning_cfg = XMimicLearningConfig.from_args(args)
            self.learner = XMimicLearner(self.wm, self.policy, self.critic, self.task, self.learning_cfg)
        else:
            self.learning_cfg = VelocityLearningConfig.from_args(args)
            self.learner = VelocityLearner(self.wm, self.policy, self.critic, self.task, self.learning_cfg)
        self.optimizers = self.learner.optimizers
        self.schedule = CollectionSchedule(args.collect_steps, getattr(args, "update_to_data_ratio", None))
        self.iteration = self.transitions = self.elapsed_before = 0
        from supermimic_lg.velocity_contract import shared_task_contract
        self.task_contract,self.task_contract_sha=shared_task_contract(self.xml,args.mjlab_root,args)
        self.metadata = dict(task="velocity", version=3, training_revision=5, learning=asdict(self.learning_cfg),
            collection_schedule=asdict(self.schedule),
            wm_loss=args.wm_loss_scale,
            action_bound_weight=args.action_bound_weight, xml_sha256=file_hash(self.xml),
            learning_precision=args.precision,
            task_contract_sha256=self.task_contract_sha,
            history=args.history, frame_dim=frame_dim(self.depth_feature_dim),
            wm_hidden=list(args.wm_hidden), actor_hidden=list(args.actor_hidden),
            critic_hidden=list(args.critic_hidden),
            terrain=args.terrain, context_dim=context_dim(self.depth_feature_dim), state_dim=42, dt=.02,
            physics_dt=.005, depth=self.depth_spec,
            actor_sensing=("proprioception_command_phase_depth" if self.depth_spec is not None
                           else "blind_proprioception_command_phase"),
            terrain_grid=[9, 9, .25],
            objective=self.objective.as_dict(), command_config=self.command_config,
            failure=dict(min_clearance=.14, max_clearance=.65, min_up=.5, trunk_contact=True))
        if self.algorithm == "xmimic":
            self.metadata.update(training_revision=6, algorithm_variant="xmimic_lg_torch",
                optimizer="muon_adam", state_drift_weight=.625,
                warmup_steps=args.warmup_steps,
                upstream_snapshot_sha256=file_hash(Path(__file__).parents[1]/"vendor/xmimic/SOURCE_MANIFEST.json"))
        if args.init_policy:
            if args.resume: raise ValueError("init-policy and resume are mutually exclusive")
            ck = torch.load(args.init_policy,map_location=args.device,weights_only=True)
            if ck.get("algorithm") not in VELOCITY_ALGORITHMS or any(
                    ck.get("metadata",{}).get(k) != self.metadata[k]
                    for k in ("version","xml_sha256","history","frame_dim","actor_hidden","wm_hidden","depth")):
                raise ValueError("initial velocity policy robot/architecture contract differs")
            self.policy.load_state_dict(policy_state(ck)[0])
            if self.wm.encoder is not None:
                self.wm.encoder.load_state_dict({key[len("encoder."):]: value for key, value in ck["wm"].items()
                                                if key.startswith("encoder.")})
        if args.resume:
            ck = torch.load(args.resume, map_location=args.device, weights_only=True)
            if ck.get("algorithm") != self.checkpoint_algorithm or resume_contract(ck.get("metadata", {})) != resume_contract(self.metadata):
                raise ValueError("velocity checkpoint task/terrain/XML/policy contract differs; tracking checkpoints are incompatible")
            for m, key in ((self.wm, "wm"), (self.policy, "policy"), (self.critic, "critic")):
                m.load_state_dict(ck[key])
            for opt, state in zip(self.optimizers, ck["optimizers"]):
                opt.load_state_dict(state)
            self.learner.load_state_dict(ck["learner"])
            self.iteration, self.transitions = ck["iteration"], ck["transitions"]
            self.elapsed_before = ck.get("training_seconds", 0.)
            torch.set_rng_state(ck["rng_state"].cpu())
            if args.iterations <= self.iteration:
                raise ValueError("iterations must exceed checkpoint iteration (total target)")
        self.start_iteration = self.iteration
        if not args.resume:
            torch.manual_seed(args.seed)  # Isolate terrain initialization from network parameter draws.
        self.sim = VelocitySimulator(self.xml, args.mjlab_root, args.num_envs, args.backend,
                                     args.device, args.terrain, args.seed, depth=self.depth_spec)
        self.replay = VelocityReplay(args.replay_capacity, args.num_envs, args.history, args.device,
                                     frame=frame_dim(self.depth_feature_dim),
                                     depth_pixels=0 if self.depth_spec is None else self.depth_spec["pixels"])
        if str(args.device).startswith("cuda"):
            free, total = torch.cuda.mem_get_info()
            self.device_memory = dict(device=str(args.device), free_gib=round(free/2**30, 2),
                total_gib=round(total/2**30, 2))
            print(json.dumps({"device_memory": self.device_memory}), flush=True)
        n, d = args.num_envs, args.device
        self.steps = torch.zeros(n, dtype=torch.long, device=d)
        self.command = torch.zeros(n, 3, device=d)
        self.previous = torch.zeros(n, 12, device=d)
        self.history = torch.zeros(n, args.history, frame_dim(self.depth_feature_dim), device=d)
        self.depth_image = None
        self.depth_feature = None
        self.depth_lag = None
        self.ep_error = torch.zeros(n, device=d)
        self.ep_progress = torch.zeros_like(self.ep_error)
        self.ep_target = torch.zeros_like(self.ep_error)
        self.reset(torch.arange(n, device=d), physics=False)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        (self.output/"config.json").write_text(json.dumps({"args": config, **self.metadata,
            "shared_task_contract":self.task_contract,"shared_task_contract_sha256":self.task_contract_sha}, indent=2), encoding="utf-8")

    def command_scale(self):
        """Command curriculum: start small so a stable gait can appear first."""
        a = self.args
        if a.command_curriculum_iterations <= 0:
            return 1.
        progress = min(1., self.iteration/max(1, a.command_curriculum_iterations))
        return a.command_curriculum_start + (1. - a.command_curriculum_start)*progress

    def sample_commands(self, ids):
        self.command[ids] = sample_commands(self.args, len(ids), self.args.device)*self.command_scale()

    @torch.no_grad()
    def refresh_depth(self, ids=None):
        """Sense depth, apply the sensor model, then replay it with the configured delay."""
        if self.depth_spec is None:
            return
        image = corrupt_depth(self.sim.depth(), self.depth_spec)
        if self.depth_lag is None:
            self.depth_lag = DepthLagBuffer(self.depth_spec["latency"], image)
        observed = self.depth_lag.push(image, ids)
        feature = self.wm.encode_depth(observed)
        if self.depth_image is None or ids is None:
            self.depth_image, self.depth_feature = observed, feature
        else:
            self.depth_image[ids] = observed[ids]
            self.depth_feature[ids] = feature[ids]

    @torch.no_grad()
    def reset(self, ids, physics=True):
        if not len(ids):
            return
        if physics:
            self.sim.reset(ids)
        self.sample_commands(ids)
        self.steps[ids], self.previous[ids] = 0, 0.
        self.ep_error[ids], self.ep_progress[ids], self.ep_target[ids] = 0., 0., 0.
        state, context = self.sim.observe()
        self.current_state, self.current_context = state, context
        self.refresh_depth(ids)
        obs = frame(state[ids], self.previous[ids], self.command[ids], phase_at(self.steps[ids]),
                    None if self.depth_feature is None else self.depth_feature[ids])
        obs[:, 9:11] *= (self.command[ids].norm(dim=-1) >= 1e-3)[:, None]
        self.history[ids] = obs[:, None].expand(-1, self.args.history, -1)

    def exploration_noise(self):
        """Decaying action noise: full-range exploration first, then fine tuning.

        The policy head starts near zero, so without this the replay never contains
        large actions and the world model cannot learn what they would do.
        """
        a = self.args
        if a.explore_iterations <= 0:
            return a.action_noise
        progress = min(1., self.iteration/max(1, a.explore_iterations))
        return a.explore_noise + (a.action_noise - a.explore_noise)*progress

    @torch.no_grad()
    def collect(self, collect_steps=None):
        collect_steps = self.args.collect_steps if collect_steps is None else collect_steps
        totals = torch.zeros(9, device=self.args.device)
        component_totals = {}
        completed = promotions = demotions = 0
        lengths = []
        noise = self.exploration_noise()
        for _ in range(collect_steps):
            # The previous step already produced this snapshot. Observing it
            # again repeated ray packing and device checks, and pushed depth
            # latency twice per control step.
            state, base_context = self.current_state, self.current_context
            input_depth = self.depth_image
            context = (base_context if self.depth_feature is None
                       else torch.cat((base_context, self.depth_feature), -1))
            action = self.policy(self.history)
            executed = (action+noise*torch.randn_like(action)).clamp(-1., 1.)
            phase = phase_at(self.steps)
            phase *= (self.command.norm(dim=-1) >= 1e-3)[:, None]
            self.sim.step(executed)
            next_state, next_context = self.sim.observe()
            self.refresh_depth()
            if self.depth_feature is not None:
                next_context = torch.cat((next_context, self.depth_feature), -1)
            predicted = self.wm.step(state, executed, context)
            prediction_error = self.wm.spec.normalized_sq_error(predicted[:, :37], next_state[:, :37])
            contact_error = F.binary_cross_entropy(predicted[:, 37:].clamp(1e-6,1-1e-6),
                                                   next_state[:, 37:], reduction="none").mean(-1)
            failed = failure(next_state, next_context)
            self.steps += 1
            reset = failed | (self.steps >= self.args.episode_steps)
            r, reward_terms = reward(next_state, executed, self.previous, self.command, phase,
                                    next_context, self.feet, self.objective, return_terms=True)
            for key, value in reward_terms.items():
                component_totals[key] = component_totals.get(key, 0.) + value.sum()
            xy, yaw = velocity_error(next_state, self.command)
            world_command = quat_apply(heading_quat(state[:, 3:7]),
                torch.cat((self.command[:, :2], torch.zeros_like(self.command[:, :1])), -1))[:, :2]
            speed = world_command.norm(dim=-1)
            self.ep_progress += ((next_state[:, :2]-state[:, :2])*world_command/speed.clamp_min(1e-6)[:, None]).sum(-1)
            self.ep_target += speed*.02
            self.ep_error += xy
            stored = dict(state=state, next=next_state, context=base_context, history=self.history,
                action=executed, previous=self.previous, command=self.command, phase=phase,
                terminal=failed.float(), reset=reset.float())
            if input_depth is not None:
                # Pair s_t with its own image, not the image sensed after a_t.
                stored["depth"] = input_depth.reshape(len(input_depth), -1)
            self.replay.add(**stored)
            self.transitions += self.args.num_envs
            self.previous = executed
            ids = reset.nonzero().flatten()
            if len(ids):
                eligible = (self.steps[ids] >= 50) & (self.ep_target[ids] >= 2.)
                up = (~failed[ids] & eligible & (self.ep_progress[ids] >= 2.)
                      & (self.ep_progress[ids] >= self.ep_target[ids]*.7)
                      & (self.ep_error[ids]/self.steps[ids] < .8))
                down = failed[ids] | (eligible & (self.ep_progress[ids] < self.ep_target[ids]*.4))
                self.sim.curriculum(ids, up, down)
                completed += len(ids)
                promotions += int(up.sum()); demotions += int(down.sum())
                lengths.extend(self.steps[ids].tolist())
            # Commands change on boundaries, and the new command is visible before
            # the next action. Per-command replay retains its actual input command.
            change = ((self.steps % 250 == 0) & ~reset).nonzero().flatten()
            self.sample_commands(change)
            new_phase = phase_at(self.steps)
            new_phase *= (self.command.norm(dim=-1) >= 1e-3)[:, None]
            obs = frame(next_state, self.previous, self.command, new_phase, self.depth_feature)
            self.policy.obs_norm.update(obs)
            self.history = torch.cat((self.history[:, 1:], obs[:, None]), 1)
            totals += torch.stack((r.sum(), xy.sum(), yaw.sum(), failed.sum(), prediction_error.sum(), contact_error.sum(),
                executed.std()*self.args.num_envs, (action.abs() >= 1.).float().sum()/12.,
                action.abs().sum()/12.))
            self.current_state, self.current_context = next_state, next_context[:, :CONTEXT_DIM]
            self.reset(ids)
        denominator = collect_steps*self.args.num_envs
        values = (totals/denominator).cpu().tolist()
        return dict(real_reward=values[0], velocity_error_xy_mps=values[1],
            yaw_error_rps=values[2], failures=round(values[3]*denominator), episodes=completed,
            wm_online_state_error=values[4], wm_online_contact_error=values[5],
            episode_length=sum(lengths)/max(1,len(lengths)), terrain_promotions=promotions, terrain_demotions=demotions,
            action_std=values[6], policy_action_saturation=values[7], policy_raw_action_abs=values[8],
            exploration_noise=float(noise), command_scale=self.command_scale(),
            **{'reward/'+k:float(v/denominator) for k,v in component_totals.items()})


    def update_wm(self):
        rows = self.replay.sample(self.learning_cfg.wm_batch_size, self.learning_cfg.wm_horizon,
                                  fields=self.learner.fields())
        return self.learner.update_world_model(rows)

    def update_policy(self):
        rows = self.replay.sample(self.learning_cfg.policy_batch_size, fields=self.learner.fields(policy=True))
        return self.learner.update_policy(rows)

    optimize = staticmethod(optimizer_step)

    def save(self):
        ck = dict(algorithm=self.checkpoint_algorithm, metadata=self.metadata, iteration=self.iteration,
                  transitions=self.transitions, training_seconds=self.elapsed_before+time.monotonic()-self.started,
                  wm=self.wm.state_dict(), policy=self.policy.state_dict(), critic=self.critic.state_dict(),
                  optimizers=[o.state_dict() for o in self.optimizers], learner=self.learner.state_dict(),
                  rng_state=torch.get_rng_state())
        ck["training_hardware"] = self.training_hardware
        if self.algorithm == "xmimic":
            ck.update(self.learner.export())
        path = self.output/"checkpoint.pt"
        atomic_save(ck, path)

    def run(self):
        title = "XMimic-Lg PyTorch BPTT" if self.algorithm == "xmimic" else "SuperMimic velocity BPTT"
        print(json.dumps(dict(algorithm=title, reference_motion=None, **self.metadata)), flush=True)
        failed_oom = False
        try:
            if self.algorithm == "xmimic" and self.args.warmup_steps:
                with self.autocast():
                    self.collect(self.args.warmup_steps)
                print(json.dumps(dict(initial_collection_steps=self.args.warmup_steps,
                    transitions=self.transitions, replay_reinitialized=True)), flush=True)
            for i in range(self.iteration+1, self.args.iterations+1):
                if self.algorithm == "xmimic":
                    self.learner.set_iteration(i-1)
                iteration_started = self.timestamp()
                collect_steps = self.schedule.steps(i-1)
                with self.autocast():
                    metrics = self.collect(collect_steps)
                collected_at = self.timestamp()
                fitted_at = collected_at
                if min(self.replay.capacity,self.replay.count) >= self.args.wm_horizon:
                    with self.autocast():
                        for _ in range(self.args.wm_updates): metrics.update(self.update_wm())
                    fitted_at = self.timestamp()
                    if i-self.start_iteration >= self.args.warmup_iterations:
                        with self.autocast():
                            for _ in range(self.args.policy_updates): metrics.update(self.update_policy())
                updated_at = self.timestamp()
                self.iteration = i
                metrics.update(iteration=i, transitions=self.transitions,
                    elapsed_s=self.elapsed_before+time.monotonic()-self.started,
                    **{"time/collect_s": collected_at-iteration_started,
                       "time/wm_s": fitted_at-collected_at, "time/policy_s": updated_at-fitted_at,
                       "throughput/transitions_per_s": self.args.num_envs*collect_steps/max(1e-6, updated_at-iteration_started)})
                metrics.update(collection_steps=collect_steps, **{"updates/"+k:v for k,v in self.learner.counts.items()},
                    **{"batch_sizes/wm": self.learning_cfg.wm_batch_size, "batch_sizes/policy": self.learning_cfg.policy_batch_size})
                metrics.update(self.memory_metrics())
                keys = [k for k, v in metrics.items() if isinstance(v, torch.Tensor)]
                if keys:
                    values = torch.stack([metrics[k].float() for k in keys]).cpu().tolist()
                    metrics.update(zip(keys, values))
                if not all(math.isfinite(v) for v in metrics.values()):
                    raise FloatingPointError("non-finite velocity metrics")
                with (self.output/"metrics.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(metrics)+"\n")
                print(json.dumps(metrics), flush=True)
                if i % self.args.save_interval == 0: self.save()
                if self.args.max_training_seconds and time.monotonic()-self.started >= self.args.max_training_seconds:
                    print("Wall-time budget reached; saving checkpoint", flush=True)
                    break
        except KeyboardInterrupt:
            print("Velocity training interrupted; saving checkpoint", flush=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if "out of memory" not in str(error).lower():
                raise
            free_gib = None
            try:
                free_gib = round(torch.cuda.mem_get_info()[0]/2**30, 2)
                torch.cuda.empty_cache()
            except Exception:
                pass
            print(json.dumps(oom_hint(error, self.args, free_gib), ensure_ascii=False), flush=True)
            failed_oom = True
        finally:
            if failed_oom:
                try:
                    self.save()
                    print("Checkpoint saved despite CUDA out-of-memory", flush=True)
                except Exception as save_error:
                    print(json.dumps(dict(error="checkpoint_save_failed", message=str(save_error)),
                        ensure_ascii=False), flush=True)
            else:
                self.save()
            self.sim.close()
        if failed_oom:
            raise SystemExit(3)

    def timestamp(self):
        if str(self.args.device).startswith("cuda"):
            torch.cuda.synchronize(self.args.device)
        return time.perf_counter()

    def autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                              enabled=self.args.precision == "bf16")

    def memory_metrics(self):
        metrics = {"memory/replay_gib": sum(t.numel()*t.element_size() for t in self.replay.rows.values())/2**30}
        if str(self.args.device).startswith("cuda"):
            free, total = torch.cuda.mem_get_info(self.args.device)
            metrics.update({"memory/torch_allocated_gib": torch.cuda.memory_allocated(self.args.device)/2**30,
                "memory/torch_reserved_gib": torch.cuda.memory_reserved(self.args.device)/2**30,
                "memory/torch_peak_gib": torch.cuda.max_memory_allocated(self.args.device)/2**30,
                "memory/device_used_gib": (total-free)/2**30})
        return metrics

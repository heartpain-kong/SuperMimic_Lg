"""Online, reference-free velocity MBRL using terrain-conditioned BPTT."""
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F

from supermimic_lg.world_model import mlp
from supermimic_lg.simulator import resolve_robot_xml
from supermimic_lg.state import heading_quat, quat_apply
from supermimic_lg.velocity_physics import VelocitySimulator
from supermimic_lg.velocity_task import (VelocityWorldModel, VelocityPolicy, VelocityReplay,
    FootKinematics, frame, phase_at, failure, velocity_error, reward, FRAME_DIM, CONTEXT_DIM)


class VelocityTrainer:
    def __init__(self, args):
        self.started = time.monotonic()
        from .train_supermimic import file_hash
        self.args = args
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
        self.wm = VelocityWorldModel(args.network_size).to(args.device)
        self.policy = VelocityPolicy(args.history, args.network_size).to(args.device)
        self.critic = mlp(37+5+1+81+5, (args.network_size, args.network_size), 1).to(args.device)
        self.optimizers = [torch.optim.AdamW(m.parameters(), lr=lr)
                           for m, lr in ((self.wm, 3e-4), (self.policy, 2e-4), (self.critic, 2e-4))]
        self.iteration = self.transitions = self.elapsed_before = 0
        from supermimic_lg.velocity_contract import shared_task_contract
        self.task_contract,self.task_contract_sha=shared_task_contract(self.xml,args.mjlab_root,args)
        self.metadata = dict(task="velocity", version=1, xml_sha256=file_hash(self.xml),
            learning_precision="tf32" if args.backend=='mjlab' else "torch_default",
            task_contract_sha256=self.task_contract_sha,
            history=args.history, frame_dim=FRAME_DIM, network_size=args.network_size,
            terrain=args.terrain, context_dim=CONTEXT_DIM, state_dim=42, dt=.02, physics_dt=.005,
            actor_sensing="blind_proprioception_command_phase", terrain_grid=[9, 9, .25],
            failure=dict(min_clearance=.14, max_clearance=.65, min_up=.5, trunk_contact=True))
        if args.init_policy:
            if args.resume: raise ValueError("init-policy and resume are mutually exclusive")
            ck = torch.load(args.init_policy,map_location=args.device,weights_only=True)
            if ck.get("algorithm") != "supermimic_velocity_bptt" or any(
                    ck.get("metadata",{}).get(k) != self.metadata[k]
                    for k in ("version","xml_sha256","history","frame_dim","network_size")):
                raise ValueError("initial velocity policy robot/architecture contract differs")
            self.policy.load_state_dict(ck["policy"])
        if args.resume:
            ck = torch.load(args.resume, map_location=args.device, weights_only=True)
            if ck.get("algorithm") != "supermimic_velocity_bptt" or ck.get("metadata") != self.metadata:
                raise ValueError("velocity checkpoint task/terrain/XML/policy contract differs; tracking checkpoints are incompatible")
            for m, key in ((self.wm, "wm"), (self.policy, "policy"), (self.critic, "critic")):
                m.load_state_dict(ck[key])
            for opt, state in zip(self.optimizers, ck["optimizers"]):
                opt.load_state_dict(state)
            self.iteration, self.transitions = ck["iteration"], ck["transitions"]
            self.elapsed_before = ck.get("training_seconds", 0.)
            torch.set_rng_state(ck["rng_state"].cpu())
            if args.iterations <= self.iteration:
                raise ValueError("iterations must exceed checkpoint iteration (total target)")
        self.start_iteration = self.iteration
        if not args.resume:
            torch.manual_seed(args.seed)  # Isolate terrain initialization from network parameter draws.
        self.sim = VelocitySimulator(self.xml, args.mjlab_root, args.num_envs, args.backend,
                                     args.device, args.terrain, args.seed)
        self.feet = FootKinematics(self.xml, args.device)
        self.replay = VelocityReplay(args.replay_capacity, args.num_envs, args.history, args.device)
        n, d = args.num_envs, args.device
        self.steps = torch.zeros(n, dtype=torch.long, device=d)
        self.command = torch.zeros(n, 3, device=d)
        self.previous = torch.zeros(n, 12, device=d)
        self.history = torch.zeros(n, args.history, 47, device=d)
        self.ep_error = torch.zeros(n, device=d)
        self.ep_progress = torch.zeros_like(self.ep_error)
        self.ep_target = torch.zeros_like(self.ep_error)
        self.reset(torch.arange(n, device=d), physics=False)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        (self.output/"config.json").write_text(json.dumps({"args": config, **self.metadata,
            "shared_task_contract":self.task_contract,"shared_task_contract_sha256":self.task_contract_sha}, indent=2), encoding="utf-8")

    def sample_commands(self, ids):
        a = self.args
        self.command[ids] = torch.rand(len(ids), 3, device=a.device)*2.-1.
        self.command[ids] *= self.command.new_tensor([a.command_vx, a.command_vy, a.command_yaw])
        self.command[ids[torch.rand(len(ids), device=a.device)<.1]] = 0.

    @torch.no_grad()
    def reset(self, ids, physics=True):
        if not len(ids):
            return
        if physics:
            self.sim.reset(ids)
        self.sample_commands(ids)
        self.steps[ids], self.previous[ids] = 0, 0.
        self.ep_error[ids], self.ep_progress[ids], self.ep_target[ids] = 0., 0., 0.
        state, _ = self.sim.observe()
        obs = frame(state[ids], self.previous[ids], self.command[ids], phase_at(self.steps[ids]))
        obs[:, 9:11] *= (self.command[ids].norm(dim=-1) >= .1)[:, None]
        self.history[ids] = obs[:, None].expand(-1, self.args.history, -1)

    @torch.no_grad()
    def collect(self):
        totals = torch.zeros(6, device=self.args.device)
        completed = promotions = demotions = 0
        lengths = []
        for _ in range(self.args.collect_steps):
            state, context = self.sim.observe()
            action = self.policy(self.history)
            executed = (action+self.args.action_noise*torch.randn_like(action)).clamp(-1., 1.)
            phase = phase_at(self.steps)
            phase *= (self.command.norm(dim=-1) >= .1)[:, None]
            self.sim.step(executed)
            next_state, next_context = self.sim.observe()
            predicted = self.wm.step(state, executed, context)
            prediction_error = self.wm.spec.normalized_sq_error(predicted[:, :37], next_state[:, :37])
            contact_error = F.binary_cross_entropy(predicted[:, 37:].clamp(1e-6,1-1e-6),
                                                   next_state[:, 37:], reduction="none").mean(-1)
            failed = failure(next_state, next_context)
            self.steps += 1
            reset = failed | (self.steps >= self.args.episode_steps)
            r = reward(next_state, executed, self.previous, self.command, phase, next_context, self.feet)
            xy, yaw = velocity_error(next_state, self.command)
            world_command = quat_apply(heading_quat(state[:, 3:7]),
                torch.cat((self.command[:, :2], torch.zeros_like(self.command[:, :1])), -1))[:, :2]
            speed = world_command.norm(dim=-1)
            self.ep_progress += ((next_state[:, :2]-state[:, :2])*world_command/speed.clamp_min(1e-6)[:, None]).sum(-1)
            self.ep_target += speed*.02
            self.ep_error += xy
            self.replay.add(state=state, next=next_state, context=context, history=self.history,
                action=executed, previous=self.previous, command=self.command, phase=phase,
                terminal=failed.float(), reset=reset.float())
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
            new_phase *= (self.command.norm(dim=-1) >= .1)[:, None]
            obs = frame(next_state, self.previous, self.command, new_phase)
            self.history = torch.cat((self.history[:, 1:], obs[:, None]), 1)
            totals += torch.stack((r.sum(), xy.sum(), yaw.sum(), failed.sum(), prediction_error.sum(), contact_error.sum()))
            self.reset(ids)
        denominator = self.args.collect_steps*self.args.num_envs
        return dict(real_reward=float(totals[0]/denominator), velocity_error_xy_mps=float(totals[1]/denominator),
            yaw_error_rps=float(totals[2]/denominator), failures=int(totals[3]), episodes=completed,
            wm_online_state_error=float(totals[4]/denominator), wm_online_contact_error=float(totals[5]/denominator),
            episode_length=sum(lengths)/max(1,len(lengths)), terrain_promotions=promotions, terrain_demotions=demotions)

    def update_wm(self):
        a = self.args
        rows = self.replay.sample(a.batch_size, a.wm_horizon)
        state, context = rows["state"][0], rows["context"][0]
        errors, contacts = [], []
        for t in range(a.wm_horizon):
            state = self.wm.step(state, rows["action"][t], context)
            state = torch.where(rows["valid"][t, :, None].bool(), state, rows["next"][t].detach())
            errors.append(self.wm.spec.normalized_sq_error(state[:, :37], rows["next"][t, :, :37]))
            contacts.append(F.binary_cross_entropy(state[:, 37:].clamp(1e-6, 1-1e-6),
                                                  rows["next"][t, :, 37:], reduction="none").mean(-1))
        valid = rows["valid"]
        loss_state = (torch.stack(errors)*valid).sum()/valid.sum().clamp_min(1.)
        loss_contact = (torch.stack(contacts)*valid).sum()/valid.sum().clamp_min(1.)
        self.optimize(loss_state+loss_contact, self.wm, self.optimizers[0])
        return {"wm/state_loss": float(loss_state.detach()), "wm/contact_loss": float(loss_contact.detach())}

    @staticmethod
    def optimize(loss, model, optimizer):
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite velocity training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        return float(grad)

    def critic_value(self, state, context, command, phase):
        return self.critic(torch.cat((self.wm.features(state, context), command, phase), -1)).squeeze(-1)

    def update_policy(self):
        rows = self.replay.sample(self.args.batch_size)
        state, context = rows["state"][0], rows["context"][0]
        history, previous, command, phase = [rows[k][0] for k in ("history", "previous", "command", "phase")]
        self.wm.requires_grad_(False); self.critic.requires_grad_(False)
        rewards, survivals, inputs = [], [torch.ones(len(state), device=state.device)], []
        for t in range(self.args.horizon):
            inputs.append((state, phase))
            action = self.policy(history)
            next_state = self.wm.step(state, action, context)
            rewards.append(reward(next_state, action, previous, command, phase, context, self.feet))
            survivals.append(survivals[-1]*(1-self.wm.termination(next_state, context)))
            angle = 2*torch.pi*.02/.6
            sine, cosine = phase.unbind(-1)
            phase = torch.stack((sine*torch.cos(state.new_tensor(angle))+cosine*torch.sin(state.new_tensor(angle)),
                                  cosine*torch.cos(state.new_tensor(angle))-sine*torch.sin(state.new_tensor(angle))), -1)
            phase = phase*(command.norm(dim=-1) >= .1)[:, None]
            obs = frame(next_state, action, command, phase)
            history = torch.cat((history[:, 1:], obs[:, None]), 1)
            state, previous = next_state, action
        boot = self.critic_value(state, context, command, phase)
        objective = sum(.99**t*survivals[t]*r for t,r in enumerate(rewards)) + .99**len(rewards)*survivals[-1]*boot
        grad = self.optimize(-objective.mean(), self.policy, self.optimizers[1])
        self.critic.requires_grad_(True)
        with torch.no_grad():
            target = boot.detach()
            targets = []
            for t in reversed(range(len(rewards))):
                ratio = survivals[t+1]/survivals[t].clamp_min(1e-8)
                target = rewards[t].detach()+.99*ratio.detach()*target
                targets.append(target)
            targets.reverse()
        predictions = torch.stack([self.critic_value(s.detach(), context, command, p.detach()) for s,p in inputs])
        loss = F.mse_loss(predictions, torch.stack(targets))
        self.optimize(loss, self.critic, self.optimizers[2])
        self.wm.requires_grad_(True)
        return {"policy/actor_objective": float(objective.detach().mean()), "policy/critic_loss": float(loss.detach()),
                "policy/policy_grad_norm": grad}

    def save(self):
        ck = dict(algorithm="supermimic_velocity_bptt", metadata=self.metadata, iteration=self.iteration,
                  transitions=self.transitions, training_seconds=self.elapsed_before+time.monotonic()-self.started,
                  wm=self.wm.state_dict(), policy=self.policy.state_dict(), critic=self.critic.state_dict(),
                  optimizers=[o.state_dict() for o in self.optimizers], rng_state=torch.get_rng_state())
        ck["training_hardware"] = self.training_hardware
        path = self.output/"checkpoint.pt"
        tmp = self.output/"checkpoint.tmp.pt"
        torch.save(ck, tmp)
        for attempt in range(30):
            try:
                tmp.replace(path); break
            except PermissionError:
                if attempt == 29: raise
                time.sleep(.1)

    def run(self):
        print(json.dumps(dict(algorithm="SuperMimic velocity BPTT", reference_motion=None, **self.metadata)), flush=True)
        try:
            for i in range(self.iteration+1, self.args.iterations+1):
                metrics = self.collect()
                if min(self.replay.capacity,self.replay.count) >= self.args.wm_horizon:
                    for _ in range(self.args.wm_updates): metrics.update(self.update_wm())
                    if i-self.start_iteration >= self.args.warmup_iterations:
                        for _ in range(self.args.policy_updates): metrics.update(self.update_policy())
                self.iteration = i
                metrics.update(iteration=i, transitions=self.transitions,
                    elapsed_s=self.elapsed_before+time.monotonic()-self.started)
                if not all(torch.isfinite(torch.tensor(v)) for v in metrics.values()):
                    raise FloatingPointError("non-finite velocity metrics")
                with (self.output/"metrics.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(metrics)+"\n")
                print(json.dumps(metrics), flush=True)
                if i % self.args.save_interval == 0: self.save()
        except KeyboardInterrupt:
            print("Velocity training interrupted; saving checkpoint", flush=True)
        finally:
            self.save(); self.sim.close()

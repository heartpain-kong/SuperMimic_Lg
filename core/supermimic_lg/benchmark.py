"""Shared velocity-task evaluation; never infer training speed from a demo."""
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import time
import numpy as np
import torch

from .velocity_task import VelocityPolicy, frame, phase_at, failure, velocity_error, geometry
from .velocity_physics import VelocitySimulator
from .state import heading_quat, quat_apply


def hardware():
    # Do not create a CUDA context in the timing-wrapper parent process.
    gpu=[]
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],
                              capture_output=True,text=True,timeout=3,check=False)
        if result.returncode==0:
            gpu=[line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (OSError,subprocess.TimeoutExpired):pass
    return dict(host=platform.node(), os=platform.platform(), cpu=platform.processor(),
                python=platform.python_version(), torch=str(torch.__version__),
                cuda=torch.version.cuda, gpu=gpu, gpu_identity_verified=bool(gpu),
                cuda_device_count=torch.cuda.device_count())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CheckpointActor:
    def __init__(self, path, device, xml):
        ck = torch.load(path, map_location=device, weights_only=True)
        if ck.get("algorithm") != "supermimic_velocity_bptt":
            raise ValueError("benchmark needs a velocity BPTT checkpoint")
        self.metadata = ck["metadata"]
        if self.metadata["xml_sha256"] != digest(xml):
            raise ValueError("benchmark robot XML differs from checkpoint")
        self.history = self.metadata["history"]
        self.policy = VelocityPolicy(self.history, self.metadata["network_size"]).to(device).eval()
        self.policy.load_state_dict(ck["policy"])
        self.training_seconds = ck.get("training_seconds")
        self.iteration = ck["iteration"]

    def __call__(self, obs):
        with torch.inference_mode():
            return self.policy(obs)


class HIMActor:
    def __init__(self, path, device):
        import onnxruntime as ort
        from .config import lgdog1_spec
        options = ort.SessionOptions(); options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(Path(path).read_bytes(), options, providers=["CPUExecutionProvider"])
        m = self.session.get_modelmeta().custom_metadata_map
        if m.get("policy_architecture") != "lgdog_him_v1" or m.get("frame_dim") != "47":
            raise ValueError("expected LgDog HIM ONNX with 47-feature frames")
        if m.get("history_order") != "oldest_to_newest" or m.get("joint_names", "").split(",") != list(lgdog1_spec().joint_names):
            raise ValueError("HIM history/joint order mismatch")
        for key, expected in (("control_dt", .02), ("action_scale", .25)):
            if abs(float(m[key])-expected) > 1e-6: raise ValueError(f"HIM {key} mismatch")
        for key, expected in (("joint_stiffness", [20,20,40]*4), ("joint_damping", [1,1,2]*4),
                              ("default_joint_pos", list(lgdog1_spec().default_joint_pos))):
            if not np.allclose(np.fromstring(m[key], sep=","), expected): raise ValueError(f"HIM {key} mismatch")
        if m.get("observation_names") != "base_ang_vel,projected_gravity,command,phase,joint_pos,joint_vel,actions":
            raise ValueError("HIM observations differ from benchmark")
        self.history, self.device = int(m["history_length"]), device
        self.training_seconds = None
        self.iteration = None

    def __call__(self, obs):
        values = self.session.run(None, {self.session.get_inputs()[0].name: obs.flatten(1).cpu().numpy()})[0]
        return torch.tensor(values, device=self.device).clamp(-1., 1.)


def evaluate_actor(actor, sim, seeds, seconds):
    episodes = []
    for seed in seeds:
        sim.rng = np.random.default_rng(seed)
        sim.reset(torch.tensor([0], device=sim.device))
        previous = torch.zeros(1, 12, device=sim.device)
        history = None
        error_xy = error_yaw = progress = target = 0.
        failed = False
        clearance_min = float("inf")
        command = torch.tensor([[.6, 0., 0.]], device=sim.device)
        started = time.monotonic()
        for step in range(round(seconds*50)):
            # The exact same deterministic command schedule is used for both.
            command[:] = command.new_tensor([.6, 0., 0.] if step < 200 else ([.4,.15,0.] if step < 350 else [.4,-.15,.2]))
            state, context = sim.observe()
            phase = phase_at(torch.tensor([step], device=sim.device))
            obs = frame(state, previous, command, phase)
            history = obs[:,None].repeat(1,actor.history,1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
            action = actor(history).clamp(-1.,1.)
            if action.shape != (1,12) or not torch.isfinite(action).all():
                raise FloatingPointError("benchmark actor emitted invalid actions")
            sim.step(action)
            after, after_context = sim.observe()
            xy, yaw = velocity_error(after, command)
            error_xy += float(xy[0]); error_yaw += float(yaw[0])
            desired_w = quat_apply(heading_quat(state[:,3:7]), torch.cat((command[:,:2],command[:,:1]*0),-1))[:,:2]
            speed = desired_w.norm(dim=-1)
            progress += float(((after[:,:2]-state[:,:2])*desired_w/speed.clamp_min(1e-6)[:,None]).sum())
            target += float(speed[0])*.02
            clearance_min = min(clearance_min, float(geometry(after,after_context)[0][0]))
            failed = bool(failure(after, after_context)[0])
            previous = action
            if failed: break
        steps = step+1
        success = (not failed and steps == round(seconds*50) and progress >= 2.
                   and progress >= .7*target and error_xy/steps < .8)
        episodes.append(dict(seed=seed, steps=steps, survival_seconds=steps*.02, failed=failed,
            terrain_passed=success, velocity_error_xy_mps=error_xy/steps, yaw_error_rps=error_yaw/steps,
            command_aligned_progress_m=progress, commanded_distance_m=target, min_clearance_m=clearance_min,
            evaluation_seconds=time.monotonic()-started))
    return episodes


def paired_benchmark(xml, root, checkpoint, him_onnx, output, *, terrain="flat", level=1,
                     seeds=(0,1,42), seconds=10., backend="mujoco", device="cpu", allow_history_mismatch=False):
    if seconds < 5 or not seeds:
        raise ValueError("benchmark needs >=5 seconds and at least one seed")
    if terrain == "rough":
        raise ValueError("evaluate each named terrain family separately; one mixed-terrain world would not cover all families")
    torch.set_num_threads(1)
    actors = {"supermimic": CheckpointActor(checkpoint,device,xml), "him_ppo": HIMActor(him_onnx,device)}
    matched = actors["supermimic"].history == actors["him_ppo"].history
    if not matched and not allow_history_mismatch:
        raise ValueError("history lengths differ: train velocity --history to match HIM, or explicitly allow/report mismatch")
    sim = VelocitySimulator(xml,root,1,backend,device,terrain,seed=0,level=level)
    try:
        physics_hash = hashlib.sha256()
        for key in ("body_mass","body_inertia","geom_type","geom_pos","geom_size","geom_friction",
                    "jnt_range","dof_damping","actuator_gainprm","actuator_biasprm","hfield_data"):
            physics_hash.update(np.asarray(getattr(sim.model,key)).tobytes())
        protocol = dict(version=1, task="lgdog_velocity_v1", terrain=terrain, level=level, seeds=list(seeds),
            seconds=seconds, backend=backend, device=device, control_dt=.02, physics_dt=.005,
            xml_sha256=digest(xml), physics_sha256=physics_hash.hexdigest(), actor_sensing="blind_47",
            spawn_origin=sim.origins[0].cpu().tolist(),
            command_schedule=[[0,.6,0.,0.],[200,.4,.15,0.],[350,.4,-.15,.2]],
            success="full episode, no failure, aligned progress >=2m and >=70% of commanded distance, XY error <0.8m/s",
            failure="relative clearance outside [0.14,0.65]m, up-vector z <0.5 or trunk contact",
            task_code_sha256=digest(Path(__file__).with_name("velocity_task.py")))
        report = dict(protocol=protocol, protocol_sha256=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest(),
            evaluation_hardware=hardware(), matched_actor_history=matched,
            checkpoint_sha256={"supermimic":digest(checkpoint),"him_ppo":digest(him_onnx)}, results={})
        for name, actor in actors.items():
            episodes = evaluate_actor(actor,sim,seeds,seconds)
            report["results"][name] = dict(episodes=episodes, history=actor.history, iteration=actor.iteration,
                training_seconds=actor.training_seconds,
                velocity_error_xy_mps=float(np.mean([e["velocity_error_xy_mps"] for e in episodes])),
                failure_rate=float(np.mean([e["failed"] for e in episodes])),
                terrain_pass_rate=float(np.mean([e["terrain_passed"] for e in episodes])))
        report["training_speedup"] = None
        report["limitations"] = ["Evaluation uses the same physics, seeds and metric definitions for both policies.",
            "Evaluation wall time is not an inference-speed comparison: HIM uses ONNX Runtime CPU; the BPTT actor uses PyTorch on the selected device.",
            "Existing HIM training rewards, training hardware and compute budget are not verified by an ONNX export.",
            "Final checkpoint scores alone do not establish training time to a common quality threshold.",
            "A short integration checkpoint is not evidence of a trained locomotion policy."]
        if not matched: report["limitations"].append("Actor history differs; this is a diagnostic comparison, not a controlled ablation.")
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        Path(output).write_text(json.dumps(report,indent=2),encoding="utf-8")
        return report
    finally:
        sim.close()

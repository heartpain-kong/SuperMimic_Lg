"""Record a metadata-checked LgDog HIM ONNX teacher in real CPU MuJoCo."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from supermimic_lg.config import lgdog1_spec
from supermimic_lg.data.motion import MotionClip, save_motion_npz
from supermimic_lg.kinematics import LgDogKinematics
from supermimic_lg.online_data import MotionSequence
from supermimic_lg.simulator import MujocoSimulator, resolve_robot_xml
from supermimic_lg.state import quat_apply_inverse


def record(policy_path, xml, output, *, speed=.4, seconds=12., warmup=2., seed=42):
    import mujoco
    import onnxruntime as ort
    torch.set_num_threads(1)
    if not .1 <= speed <= 1. or seconds < 6 or warmup < 0:
        raise ValueError("speed must be .1..1 m/s, seconds >= 6, warmup >= 0")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = Path(policy_path).read_bytes()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(payload, options, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    spec = lgdog1_spec()
    expected = dict(policy_architecture="lgdog_him_v1", frame_dim="47",
                    history_order="oldest_to_newest", history_reset="repeat_first_frame",
                    observation_names="base_ang_vel,projected_gravity,command,phase,joint_pos,joint_vel,actions")
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"incompatible teacher metadata {key}: {meta.get(key)}")
    if meta.get("joint_names", "").split(",") != list(spec.joint_names):
        raise ValueError("teacher joint order must match canonical LgDog order")
    if abs(float(meta["control_dt"]) - .02) > 1e-6 or abs(float(meta["action_scale"]) - .25) > 1e-6:
        raise ValueError("teacher control timing/action scale differs")
    default = np.array(spec.default_joint_pos, dtype=np.float32)
    for key, expected_values in (("default_joint_pos", default),
                                 ("joint_stiffness", [20, 20, 40] * 4),
                                 ("joint_damping", [1, 1, 2] * 4)):
        if not np.allclose(np.fromstring(meta[key], sep=","), expected_values):
            raise ValueError(f"teacher {key} does not match simulator")
    history_length = int(meta["history_length"])
    if not 1 <= history_length <= 128:
        raise ValueError("invalid teacher history length")
    sim = MujocoSimulator(xml, 1, "cpu")
    rng = np.random.default_rng(seed)
    initial = np.zeros((1, 37), dtype=np.float32)
    initial[0, 2:4] = [.31, 1.]
    initial[0, 7:19] = default + rng.uniform(-.003, .003, 12)
    sim.reset(torch.tensor([0]), torch.from_numpy(initial))
    data, model = sim.data[0], sim.model
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in spec.tracked_body_names]
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_calf1_collision")
                for leg in ("FR", "FL", "RR", "RL")]
    previous = np.zeros(12, dtype=np.float32)
    history = None
    fields = [[] for _ in range(6)]
    contacts, upright, trunk_contacts = [], [], []
    start_step = round(warmup * 50)
    nframes = round(seconds * 50)
    try:
        for step in range(start_step + nframes):
            rotation = data.xmat[body_ids[0]].reshape(3, 3)
            gravity = rotation.T @ np.array([0., 0., -1.])
            phase = step * .02 / .6 * 2 * np.pi
            obs = np.concatenate((data.qvel[3:6], gravity, [speed, 0., 0.],
                                  [np.sin(phase), np.cos(phase)],
                                  data.qpos[sim.qadr] - default, data.qvel[sim.vadr], previous)).astype(np.float32)
            history = (np.tile(obs, (history_length, 1)) if history is None
                       else np.concatenate((history[1:], obs[None])))
            if step >= start_step:
                values = [data.qpos[sim.qadr].copy(), data.qvel[sim.vadr].copy(),
                          data.xpos[body_ids].copy(), data.xquat[body_ids].copy()]
                velocities = np.empty((len(body_ids), 6))
                for i, body in enumerate(body_ids):
                    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocities[i], 0)
                values += [velocities[:, 3:].copy(), velocities[:, :3].copy()]
                for rows, value in zip(fields, values):
                    rows.append(value)
                touch = set()
                trunk_touch = False
                for c in data.contact:
                    if c.dist <= .002:
                        touch.update((int(c.geom1), int(c.geom2)))
                        for geom in (c.geom1, c.geom2):
                            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
                            trunk_touch |= name == "trunk_collision"
                contacts.append([g in touch for g in foot_ids])
                upright.append(float(rotation[2, 2]))
                trunk_contacts.append(trunk_touch)
            if not np.isfinite(data.qpos).all() or data.qpos[2] < .15 or rotation[2, 2] < .7:
                raise ValueError(f"teacher fell or became unstable at {step*.02:.2f}s; no reference saved")
            raw = session.run(None, {session.get_inputs()[0].name: history.reshape(1, -1)})[0]
            if raw.shape != (1, 12) or not np.isfinite(raw).all():
                raise ValueError("invalid teacher action")
            previous = np.clip(raw[0], -1., 1.)
            sim.step(torch.from_numpy(previous[None]))
    finally:
        sim.close()
    arrays = [np.asarray(x, dtype=np.float32) for x in fields]
    root = arrays[2][:, 0]
    displacement = root[-1] - root[0]
    ratio = float(displacement[0] / ((nframes - 1) * .02 * speed))
    contact_fraction = np.mean(contacts, axis=0)
    report = dict(source="HIM teacher rollout in native MuJoCo; no root teleportation",
                  teacher_sha256=hashlib.sha256(payload).hexdigest(),
                  xml_sha256=hashlib.sha256(Path(xml).read_bytes()).hexdigest(),
                  seed=seed, commanded_speed=speed, frames=nframes, fps=50,
                  displacement_m=displacement.tolist(), progress_ratio=ratio,
                  min_upright=float(min(upright)), trunk_contacts=int(sum(trunk_contacts)),
                  foot_contact_fraction=contact_fraction.tolist(), terrain="flat")
    limits = np.asarray(spec.joint_pos_limits)
    accepted = (displacement[0] >= 2 and .6 <= ratio <= 1.5 and abs(displacement[1]) < 1.
                and min(upright) > .85 and not any(trunk_contacts)
                and np.all((contact_fraction > .05) & (contact_fraction < .95))
                and np.all(arrays[0] >= limits[:, 0] - .05) and np.all(arrays[0] <= limits[:, 1] + .05))
    report["accepted"] = bool(accepted)
    if not accepted:
        output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise ValueError(f"teacher rollout failed movement/contact quality gates: {report}")
    # Translation changes neither physical velocities nor root orientation.
    origin = arrays[2][0, 0, :2].copy()
    arrays[2][:, :, :2] -= origin
    clip = MotionClip(*arrays, fps=50., name=output.stem)
    # Validate all recorded frames before publishing the file.
    fk = LgDogKinematics(model)
    omega = quat_apply_inverse(torch.from_numpy(arrays[3][:, 0]), torch.from_numpy(arrays[5][:, 0]))
    states = np.concatenate((arrays[2][:, 0], arrays[3][:, 0], arrays[0], arrays[4][:, 0],
                             omega.numpy(), arrays[1]), axis=1)
    with torch.no_grad():
        pos, quat, lin, ang = fk(torch.from_numpy(states.astype(np.float32)))
    if np.max(np.linalg.norm(pos.numpy() - arrays[2], axis=-1)) > 1e-4:
        raise ValueError("recorded body positions do not agree with XML FK")
    if np.min(np.abs((quat.numpy() * arrays[3]).sum(-1))) < .9999:
        raise ValueError("recorded quaternions do not agree with XML FK")
    if max(np.max(np.abs(lin.numpy() - arrays[4])), np.max(np.abs(ang.numpy() - arrays[5]))) > 1e-3:
        raise ValueError("recorded link velocities do not agree with XML FK")
    temporary = output.with_name(output.stem + ".tmp.npz")
    try:
        save_motion_npz(temporary, clip)
        MotionSequence(temporary, 10, "cpu").validate_kinematics(fk)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    report["npz_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    report["max_fk_position_error_m"] = float(np.max(np.linalg.norm(pos.numpy() - arrays[2], axis=-1)))
    report["joint_names"] = list(spec.joint_names)
    report["body_names"] = list(spec.tracked_body_names)
    report["teacher_history_frames"] = history_length
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--mjlab-root", type=Path, default=Path("../unitree_rl_mjlab_lgdog"))
    p.add_argument("--robot-xml", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--speed", type=float, default=.4)
    p.add_argument("--seconds", type=float, default=12.)
    p.add_argument("--warmup", type=float, default=2.)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    print(json.dumps(record(args.teacher, resolve_robot_xml(args.mjlab_root, args.robot_xml), args.output,
                            speed=args.speed, seconds=args.seconds, warmup=args.warmup, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()

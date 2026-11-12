"""Play a trained checkpoint on the CPU, without a GPU: MuJoCo's C simulator steps one G1, XMimic's own observation, goal and
policy code (JAX on the CPU) acts, under the evaluation protocol (no randomization, no noise, friction 1.0).

    python tools/rollout_cpu.py <ckpt_*.npz> --clip dance1_subject2 --start 60 --seconds 12 --out rollout.npz

Writes the robot's qpos and the reference frame at every 50 Hz control step (tools/render_gif.py renders them). The
training simulator is MuJoCo Warp; MuJoCo's C simulator integrates the same model, so this is a sim-to-sim check too.
"""

from __future__ import annotations

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xmimic import nets  # noqa: E402
from xmimic.evaluate import load_ckpt  # noqa: E402
from xmimic.kin import Kinematics  # noqa: E402
from xmimic.mdp import MDP  # noqa: E402
from xmimic.model import load_model  # noqa: E402
from xmimic.motion import GOAL_Q_OFFSET, load_motion  # noqa: E402
from xmimic.rot import mat_to_quat, mv3, quat_to_mat, tr3  # noqa: E402
from xmimic.sim import DECIMATION, XiLayout  # noqa: E402


def rollout(ckpt, clip, start_s, seconds, motion_dir="data/lafan1_g1"):
    """{qpos (T+1, nq), f (T+1,) reference frames, alive (T+1,)} of one episode from frame start_s * 50 of ``clip``."""
    jax.config.update("jax_platforms", "cpu")
    params, bufs, meta_ck = load_ckpt(ckpt)
    a = meta_ck["args"]
    m, meta = load_model()
    J = len(meta["joint_names"])
    kin = Kinematics(meta)
    feet = [meta["tracked_bodies"].index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
    motion = load_motion(motion_dir, kin, feet)
    layout = XiLayout(meta)
    mdp = MDP(kin, meta["default_joint_pos"], feet, layout.dim)
    ref = (GOAL_Q_OFFSET, jnp.asarray(meta["default_joint_pos"], jnp.float32), jnp.asarray(meta["action_scale"], jnp.float32))
    m.geom_friction[np.asarray(meta["robot_geom_ids"]), 0] = 1.0  # evaluation friction
    d = mujoco.MjData(m)
    qadr, dadr, act = np.asarray(meta["qpos_adr"]), np.asarray(meta["dof_adr"]), np.asarray(meta["actuator_of_joint"])
    fq, fd = meta["free_qpos_adr"], meta["free_dof_adr"]
    default_q, alpha = np.asarray(meta["default_joint_pos"], np.float64), np.asarray(meta["action_scale"], np.float64)
    xi = jnp.zeros((1, layout.dim))  # no randomization: zero joint offsets

    def state(qpos, qvel):  # as Sim.state, for one world
        R = quat_to_mat(qpos[None, fq + 3 : fq + 7])
        return (qpos[None, fq : fq + 3], R, qvel[None, fd : fd + 3], mv3(R, qvel[None, fd + 3 : fd + 6]), qpos[None, qadr], qvel[None, dadr])

    @jax.jit
    def observe(qpos, qvel, a_prev, f):
        S = state(qpos.astype(jnp.float32), qvel.astype(jnp.float32))
        return mdp.obs(motion, S, a_prev[None], xi, f[None])[0], mdp.termination(motion, S, f[None])[0]

    @jax.jit
    def policy(hist, f):
        return nets.pi_apply(params, bufs, hist[None], motion.goal(f[None], a["goal_frames"]), ref)[0]

    c = motion.names.index(clip)
    cs, ce = int(motion.clip_start[c]), int(motion.clip_end[c])
    f = min(cs + int(round(start_s * 50)), ce - 1)
    p, R, v, w, q, qd = (np.asarray(x[f], np.float64) for x in motion.ref_state(jnp.arange(motion.num_frames)))
    d.qpos[fq : fq + 3], d.qpos[fq + 3 : fq + 7], d.qpos[qadr] = p, np.asarray(mat_to_quat(jnp.asarray(R))), q
    d.qvel[fd : fd + 3], d.qvel[fd + 3 : fd + 6], d.qvel[dadr] = v, R.T @ w, qd
    d.ctrl[act] = q
    mujoco.mj_forward(m, d)
    a_prev = jnp.zeros(J)
    o, _ = observe(d.qpos, d.qvel, a_prev, jnp.int32(f))
    hist = jnp.broadcast_to(o, (a["hist"], o.shape[-1]))
    out = {"qpos": [d.qpos.copy()], "f": [f], "alive": [True]}
    alive = True
    for t in range(int(round(seconds * 50))):
        if t > 0:
            hist = jnp.concatenate((hist[1:], o[None]), 0)
        a_t = policy(hist, jnp.int32(f))
        d.ctrl[act] = default_q + alpha * np.asarray(a_t, np.float64)
        for _ in range(DECIMATION):
            mujoco.mj_step(m, d)
        f = min(f + 1, ce - 1)
        a_prev = a_t
        o, term = observe(d.qpos, d.qvel, a_prev, jnp.int32(f))
        alive = alive and not bool(term)
        out["qpos"].append(d.qpos.copy())
        out["f"].append(f)
        out["alive"].append(alive)
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--clip", default="dance1_subject2")
    ap.add_argument("--start", type=float, default=60.0, help="start time in the clip (s)")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--motion_dir", default="data/lafan1_g1")
    ap.add_argument("--out", default="rollout.npz")
    args = ap.parse_args()
    r = rollout(args.ckpt, args.clip, args.start, args.seconds, args.motion_dir)
    np.savez(args.out, **r, clip=args.clip)
    print(f"{args.clip} from {args.start:.1f} s: {len(r['f']) - 1} steps, alive to the end: {bool(r['alive'][-1])} -> {args.out}")


if __name__ == "__main__":
    main()

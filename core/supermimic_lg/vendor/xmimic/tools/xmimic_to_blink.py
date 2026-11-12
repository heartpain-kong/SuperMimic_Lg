"""Convert XMimic checkpoints (ckpt_*.npz: the averaged policy and its normalizers) to BlinkMimic's format, so that
BlinkMimic's own evaluators (tools/eval_batch.py, time_to_target.py, eval_metrics.py, blinkmimic.evaluate) can score XMimic
runs on mjlab. This is how XMimic's evaluator was cross-checked.

Runs in BlinkMimic's environment:  PYTHONPATH=<blinkmimic>/src python tools/xmimic_to_blink.py <xmimic_run_dir> <out_dir>
Each checkpoint is loaded into blinkmimic.networks.Policy (strict), and its action on the test inputs stored in the
checkpoint is compared with the action XMimic computed when it saved it (bf16 matmuls on both sides).
"""

import glob
import json
import os
import sys
import time

import numpy as np
import torch

import blinkmimic.networks as networks
from blinkmimic.mdp import obs_dim
from blinkmimic.motion import GOAL_Q_OFFSET
from blinkmimic.networks import Policy
from blinkmimic.robot import get_robot

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
robot = get_robot("g1")
J = robot.num_joints
networks.AMP_DTYPE = torch.bfloat16
worst = 0.0
files = sorted(glob.glob(os.path.join(src, "ckpt_*.npz")))
assert files, src
for path in files:
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    a = meta["args"]
    sd = {}
    for name in ("enc", "dec"):
        i = 0
        while f"{name}.{i}.w" in z:
            sd[f"{name}.{2 * i}.weight"] = torch.tensor(z[f"{name}.{i}.w"].T.copy())
            sd[f"{name}.{2 * i}.bias"] = torch.tensor(z[f"{name}.{i}.b"])
            i += 1
    for nn_ in ("obs_norm", "goal_norm"):
        for k in ("mean", "var", "count"):
            sd[f"{nn_}.{k}"] = torch.tensor(z[f"{nn_}.{k}"])
    sd["ref_q0"] = torch.tensor(robot.default_joint_pos_list(), dtype=torch.float32)
    sd["ref_alpha"] = torch.tensor(robot.action_scale_list(), dtype=torch.float32)
    K, H = a["goal_frames"], a["hist"]
    widths = lambda s: tuple(int(x) for x in s.split(","))  # noqa: E731
    pi = Policy(obs_dim(J), K * (15 + 2 * J), J, hist=H, enc_hidden=widths(a["enc_hidden"]), dec_hidden=widths(a["dec_hidden"]),
                ref_skip=(GOAL_Q_OFFSET, robot.default_joint_pos_list(), robot.action_scale_list()))  # fmt: skip
    pi.load_state_dict(sd, strict=True)
    with torch.no_grad():
        act = pi(torch.tensor(z["test_oh"]), torch.tensor(z["test_g"])).numpy()
    err = float(np.abs(act - z["test_a"]).max() / max(1.0, np.abs(z["test_a"]).max()))
    worst = max(worst, err)
    it = int(meta["iteration"])
    ck = {"pi": sd, "iteration": it, "wall_time": float(meta["wall_time"]), "sim_steps": float(meta["sim_steps"]),
          "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path))),
          "args": {"goal_frames": K, "hist": H, "enc_hidden": a["enc_hidden"], "dec_hidden": a["dec_hidden"], "pi_ref_skip": True,
                   "amp": "bf16", "xmimic": a}}  # fmt: skip
    torch.save(ck, os.path.join(out, f"ckpt_{it:06d}.pt"))
json.dump({"goal_frames": K, "hist": H, "enc_hidden": a["enc_hidden"], "dec_hidden": a["dec_hidden"], "pi_ref_skip": True, "amp": "bf16",
           "ckpt_every": a.get("ckpt_every", 100),
           "converted_from": os.path.abspath(src)}, open(os.path.join(out, "args.json"), "w"), indent=1)  # fmt: skip
print(f"converted {len(files)} checkpoints -> {out}; max relative action difference torch vs JAX: {worst:.2e}")
assert worst < 5e-2, "converted policy does not reproduce XMimic's actions"

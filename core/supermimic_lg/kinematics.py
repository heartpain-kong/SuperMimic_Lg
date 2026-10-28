"""Differentiable link-origin FK, built from the actual compiled robot model."""
from __future__ import annotations

import numpy as np
import torch

from .config import lgdog1_spec
from .state import QuadrupedStateSpec, quat_apply, quat_from_rotvec, quat_mul, quat_normalize


class LgDogKinematics:
    def __init__(self, model, device="cpu"):
        import mujoco
        self.spec = QuadrupedStateSpec(12)
        robot = lgdog1_spec()
        self.names = robot.tracked_body_names
        self.ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.names]
        if min(self.ids) < 0:
            raise ValueError("model is missing LgDog tracked bodies")
        self.links = []
        tensor = lambda x: torch.tensor(np.array(x), dtype=torch.float32, device=device)
        for i, b in enumerate(self.ids):
            parent = int(model.body_parentid[b])
            if i == 0:
                if parent != 0:
                    raise ValueError("trunk must be a free root")
                continue
            p = self.ids.index(parent)
            if p >= i or model.body_jntnum[b] != 1:
                raise ValueError("expected a topologically ordered single-hinge LgDog tree")
            j = int(model.body_jntadr[b])
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("expected hinge joint")
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            index = robot.joint_names.index(name)
            self.links.append((p, index, tensor(model.body_pos[b]), tensor(model.body_quat[b]),
                               tensor(model.jnt_pos[j]), tensor(model.jnt_axis[j]),
                               float(model.qpos0[model.jnt_qposadr[j]])))

    def __call__(self, state):
        root, rotation, joints, velocity, omega, rates = self.spec.split(state)
        rotation = quat_normalize(rotation)
        pos, quat = [root], [rotation]
        lin, ang = [velocity], [quat_apply(rotation, omega)]
        shape = state.shape[:-1]
        for p, j, offset, orient, anchor, axis, q0 in self.links:
            origin = pos[p] + quat_apply(quat[p], offset.expand(*shape, 3))
            q = quat_mul(quat[p], orient.expand(*shape, 4))
            pivot = origin + quat_apply(q, anchor.expand(*shape, 3))
            axis_w = quat_apply(q, axis.expand(*shape, 3))
            q = quat_mul(q, quat_from_rotvec((joints[..., j] - q0)[..., None] * axis))
            origin = pivot - quat_apply(q, anchor.expand(*shape, 3))
            joint_w = axis_w * rates[..., j, None]
            lin.append(lin[p] + torch.cross(ang[p], origin - pos[p], dim=-1)
                       + torch.cross(joint_w, origin - pivot, dim=-1))
            ang.append(ang[p] + joint_w)
            pos.append(origin)
            quat.append(q)
        return tuple(torch.stack(x, dim=-2) for x in (pos, quat, lin, ang))

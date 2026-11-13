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
        # A quadruped has four independent chains. Evaluate all links at the
        # same depth together: three batched stages instead of twelve scalar
        # link traversals, while preserving XML offsets/anchors/orientations.
        depths = [0]
        for parent, *_ in self.links:
            depths.append(depths[parent]+1)
        self.levels = []
        previous = {0: 0}
        body_order = [0]
        for depth in range(1, max(depths)+1):
            bodies = [i for i, d in enumerate(depths) if d == depth]
            links = [self.links[i-1] for i in bodies]
            index_tensor = lambda values: torch.tensor(values, dtype=torch.long, device=device)
            self.levels.append((index_tensor([previous[link[0]] for link in links]),
                index_tensor([link[1] for link in links]),
                *[torch.stack([link[k] for link in links]) for k in range(2, 6)],
                tensor([link[6] for link in links])))
            previous = {body: i for i, body in enumerate(bodies)}
            body_order.extend(bodies)
        self.body_order = torch.tensor([body_order.index(i) for i in range(len(self.ids))],
                                       dtype=torch.long, device=device)

    def __call__(self, state):
        root, rotation, joints, velocity, omega, rates = self.spec.split(state)
        rotation = quat_normalize(rotation)
        pos, quat = [root.unsqueeze(-2)], [rotation.unsqueeze(-2)]
        lin, ang = [velocity.unsqueeze(-2)], [quat_apply(rotation, omega).unsqueeze(-2)]
        for parents, indices, offset, orient, anchor, axis, q0 in self.levels:
            parent_pos, parent_quat, parent_vel, parent_ang = (
                values[-1].index_select(-2, parents) for values in (pos, quat, lin, ang))
            origin = parent_pos + quat_apply(parent_quat, offset)
            q = quat_mul(parent_quat, orient)
            pivot = origin + quat_apply(q, anchor)
            axis_w = quat_apply(q, axis)
            q = quat_mul(q, quat_from_rotvec((joints.index_select(-1, indices)-q0)[..., None]*axis))
            origin = pivot - quat_apply(q, anchor)
            joint_w = axis_w*rates.index_select(-1, indices)[..., None]
            lin.append(parent_vel + torch.cross(parent_ang, origin-parent_pos, dim=-1)
                       + torch.cross(joint_w, origin - pivot, dim=-1))
            ang.append(parent_ang + joint_w)
            pos.append(origin)
            quat.append(q)
        return tuple(torch.cat(x, dim=-2).index_select(-2, self.body_order) for x in (pos, quat, lin, ang))

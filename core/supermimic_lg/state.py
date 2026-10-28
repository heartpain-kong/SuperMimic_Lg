from __future__ import annotations

from dataclasses import dataclass

import torch


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (aw * bw - ax * bx - ay * by - az * bz,
         aw * bx + ax * bw + ay * bz - az * by,
         aw * by - ax * bz + ay * bw + az * bx,
         aw * bz + ax * by - ay * bx + az * bw), dim=-1
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
    return quat_mul(quat_mul(q, qv), q * q.new_tensor([1.0, -1.0, -1.0, -1.0]))[..., 1:]


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qi = q * q.new_tensor([1.0, -1.0, -1.0, -1.0])
    return quat_apply(qi, v)


def quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
    angle = v.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    # Clamp the denominator before evaluating both branches.  ``torch.where``
    # selects the right value but still evaluates the other branch, so a raw
    # sin(x)/x would create NaNs in the backward graph at x=0.
    safe_angle = angle.clamp_min(1e-6)
    exact = torch.sin(half) / safe_angle
    approx = 0.5 - angle.square() / 48.0
    scale = torch.where(angle > 1e-6, exact, approx)
    return quat_normalize(torch.cat((torch.cos(half), v * scale), dim=-1))


def quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    return torch.stack((r00, r01, r10, r11, r20, r21), dim=-1)


def heading_quat(q: torch.Tensor) -> torch.Tensor:
    """Return the yaw-only quaternion of a wxyz quaternion."""

    w, x, y, z = quat_normalize(q).unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    half = 0.5 * yaw
    return torch.stack((torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)), dim=-1)


@dataclass
class CanonicalState:
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    joint_pos: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    joint_vel: torch.Tensor

    def flat(self) -> torch.Tensor:
        return torch.cat((self.root_pos, self.root_quat, self.joint_pos, self.root_lin_vel,
                          self.root_ang_vel, self.joint_vel), dim=-1)


class QuadrupedStateSpec:
    """Generic simulator state layout and differentiable integrator.

    State is ``[root_pos(3), root_quat(wxyz)(4), q(n), v(3), w(3), qd(n)]``.
    The learned residual is ``[dv(3), dw(3), dqd(n), dp(3), dtheta(3), dq(n)]``.
    This is the same constant velocity prior used by the reference reproduction,
    with the humanoid-specific 29 DoF constants removed.
    """

    def __init__(self, n_joints: int, dt: float = 0.02, explicit_integration: bool = True,
                 device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        if n_joints <= 0:
            raise ValueError("n_joints must be positive")
        self.n_joints = n_joints
        self.dt = float(dt)
        self.explicit_integration = bool(explicit_integration)
        self.device = device
        self.dtype = dtype
        self.state_dim = 13 + 2 * n_joints
        self.delta_dim = 12 + 2 * n_joints

    def split(self, state: torch.Tensor):
        n = self.n_joints
        return (state[..., :3], state[..., 3:7], state[..., 7:7+n],
                state[..., 7+n:10+n], state[..., 10+n:13+n], state[..., 13+n:13+2*n])

    def join(self, root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel):
        return torch.cat((root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel), dim=-1)

    def canonical(self, state: torch.Tensor) -> torch.Tensor:
        p, q, qj, v, w, qd = self.split(state)
        q = quat_normalize(q)
        hq = heading_quat(q)
        return torch.cat((p[..., 2:3], quat_to_rot6d(quat_mul(hq * hq.new_tensor([1., -1., -1., -1.]), q)),
                          quat_apply_inverse(hq, v), w, qj, qd), dim=-1)

    def integrate(self, state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        n = self.n_joints
        dv, dw, dqd = delta[..., :3], delta[..., 3:6], delta[..., 6:6+n]
        dp, dtheta, dq = delta[..., 6+n:9+n], delta[..., 9+n:12+n], delta[..., 12+n:12+2*n]
        p, q, qj, v, w, qd = self.split(state)
        q = quat_normalize(q)
        hq = heading_quat(q)
        v_new, w_new, qd_new = v + quat_apply(hq, dv), w + dw, qd + dqd
        dt = self.dt if self.explicit_integration else 0.0
        p_new = p + dt * v_new + quat_apply(hq, dp)
        q_new = quat_normalize(quat_mul(q, quat_from_rotvec(dt * w_new + dtheta)))
        qj_new = qj + dt * qd_new + dq
        return self.join(p_new, q_new, qj_new, v_new, w_new, qd_new)

    def error_vector(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n = self.n_joints
        p1, q1, qj1, v1, w1, qd1 = self.split(predicted)
        p2, q2, qj2, v2, w2, qd2 = self.split(target)
        # Quaternion error is represented by the vector part, which is smooth
        # around the identity and sufficient for the small tracking residuals.
        qerr = quat_mul(q2 * q2.new_tensor([1., -1., -1., -1.]), q1)
        rot = 2.0 * qerr[..., 1:]
        scale = predicted.new_tensor([0.05] * 3 + [0.1] * 3 + [0.5] * 3 + [1.0] * 3 + [0.1] * n + [2.0] * n)
        return torch.cat((p1-p2, rot, v1-v2, w1-w2, qj1-qj2, qd1-qd2), dim=-1) / scale

    def normalized_sq_error(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        e = self.error_vector(predicted, target)
        return e.square().mean(dim=-1)

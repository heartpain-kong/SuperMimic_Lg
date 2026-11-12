"""Structured 4-substep drift, adapted to LgDog's heading/body state convention."""
from ..state import QuadrupedStateSpec, quat_normalize, heading_quat, quat_apply, quat_apply_inverse, quat_mul, quat_from_rotvec, quat_to_rotvec


class XMimicStateSpec(QuadrupedStateSpec):
    drift_weight = .625  # (n+1)/(2*n), n=4 semi-implicit physics substeps.

    def integrate(self, state, delta):
        n = self.n_joints
        dv, dw, dqd = delta[..., :3], delta[..., 3:6], delta[..., 6:6+n]
        dp, rotation, dq = delta[..., 6+n:9+n], delta[..., 9+n:12+n], delta[..., 12+n:12+2*n]
        p, q, joints, v, w, qd = self.split(state)
        q = quat_normalize(q); heading = heading_quat(q)
        v1, w1, qd1 = v+quat_apply(heading, dv), w+dw, qd+dqd
        dt = self.dt if self.explicit_integration else 0.
        mix = lambda old, new: dt*((1.-self.drift_weight)*old+self.drift_weight*new)
        return self.join(p+mix(v, v1)+quat_apply(heading, dp),
            quat_normalize(quat_mul(q, quat_from_rotvec(mix(w, w1)+rotation))),
            joints+mix(qd, qd1)+dq, v1, w1, qd1)

    def delta_between(self, state, next_state):
        import torch
        p, q, joints, v, w, qd = self.split(state)
        p1, q1, joints1, v1, w1, qd1 = self.split(next_state)
        q = quat_normalize(q); heading = heading_quat(q)
        dt = self.dt if self.explicit_integration else 0.
        mix = lambda old, new: dt*((1.-self.drift_weight)*old+self.drift_weight*new)
        dp = quat_apply_inverse(heading, p1-p-mix(v, v1))
        rotation = quat_to_rotvec(quat_mul(q*q.new_tensor([1., -1., -1., -1.]), quat_normalize(q1)))-mix(w, w1)
        return torch.cat((quat_apply_inverse(heading, v1-v), w1-w, qd1-qd,
                          dp, rotation, joints1-joints-mix(qd, qd1)), -1)

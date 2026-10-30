"""Reference-free velocity rewards and terrain-conditioned differentiable dynamics."""
import torch
from torch import nn
from torch.nn import functional as F

from .config import lgdog1_spec
from .state import QuadrupedStateSpec, heading_quat, quat_apply, quat_apply_inverse
from .world_model import mlp
from .velocity_physics import GRID, offsets

STATE_DIM = 42
CONTEXT_DIM = 4 + GRID*GRID
FRAME_DIM = 47


def sample_ground(context, xy):
    """Bilinear patch lookup with gradients in the predicted world position.

    Patch rows are local X, columns local Y, exactly matching GridPatternCfg.
    Heights are world Z; anchor yaw remains fixed during an imagined rollout.
    """
    point_shape = xy.shape[1:-1]
    xy = xy.reshape(len(context), -1, 2) - context[:, None, :2]
    c, s = context[:, 3].cos()[:, None], context[:, 3].sin()[:, None]
    x, y = xy.unbind(-1)
    local = torch.stack((c*x+s*y, -s*x+c*y), -1)
    # grid_sample expects columns then rows, thus swap Y/X.
    grid = local[..., [1, 0]].unsqueeze(1)
    value = F.grid_sample(context[:, 4:].reshape(-1, 1, GRID, GRID), grid,
                          mode="bilinear", padding_mode="border", align_corners=True)
    return value[:, 0, 0].reshape((len(context),) + point_shape)


def ground_clearance(state, context):
    return state[:, 2] - sample_ground(context, state[:, None, :2]).squeeze(-1)


def geometry(state, context):
    q = heading_quat(state[:, 3:7])
    # Reserve the outer half of the sensed patch for predicted translation.
    probes = (offsets(state.device)*.5)[None].expand(len(state), -1, -1)
    local = torch.cat((probes, torch.zeros_like(probes[..., :1])), -1)
    world = quat_apply(q[:, None].expand(-1, GRID*GRID, -1), local)[..., :2] + state[:, None, :2]
    clearance = ground_clearance(state, context)
    scan = state[:, 2:3] - sample_ground(context, world)
    return clearance, scan


def frame(state, previous_action, command, phase):
    gravity = quat_apply_inverse(state[:, 3:7], state.new_tensor([0., 0., -1.]).expand(len(state), 3))
    return torch.cat((state[:, 22:25], gravity, command, phase,
                      state[:, 7:19] - state.new_tensor(lgdog1_spec().default_joint_pos),
                      state[:, 25:37], previous_action), -1)


def phase_at(steps):
    angle = steps*.02/.6*(2*torch.pi)
    return torch.stack((angle.sin(), angle.cos()), -1)


def failure(state, context):
    height = ground_clearance(state, context)
    gravity = quat_apply_inverse(state[:, 3:7], state.new_tensor([0., 0., -1.]).expand(len(state), 3))
    return (height < .14) | (height > .65) | (gravity[:, 2] > -.5) | (state[:, 41] > .5)


def velocity_error(state, command):
    velocity = quat_apply_inverse(heading_quat(state[:, 3:7]), state[:, 19:22])
    yaw_rate = quat_apply(state[:, 3:7], state[:, 22:25])[:, 2]
    return (velocity[:, :2] - command[:, :2]).norm(dim=-1), (yaw_rate-command[:, 2]).abs()


class VelocityPolicy(nn.Module):
    def __init__(self, history=3, width=128):
        super().__init__()
        self.history, self.width = history, width
        self.net = mlp(history*FRAME_DIM, (width, width), 12)

    def forward(self, history):
        return self.net(history.flatten(1)).tanh()


class VelocityWorldModel(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.spec = QuadrupedStateSpec(12)
        self.dynamics = mlp(37+5+1+GRID*GRID+12, (width,)*3, self.spec.delta_dim, zero_last=True)
        self.contacts = mlp(37+5+1+GRID*GRID+12, (width, width), 5)
        nn.init.constant_(self.contacts[-1].bias, -3.)

    def features(self, state, context):
        return state_features(state, context, self.spec)

    def step(self, state, action, context):
        features = torch.cat((self.features(state, context), action), -1)
        physical = self.spec.integrate(state[:, :37], self.dynamics(features))
        return torch.cat((physical, self.contacts(features).sigmoid()), -1)

    def termination(self, state, context):
        height = ground_clearance(state, context)
        gravity = quat_apply_inverse(state[:, 3:7], state.new_tensor([0., 0., -1.]).expand(len(state), 3))
        risks = torch.stack((torch.sigmoid((.14-height)/.015), torch.sigmoid((height-.65)/.025),
                             torch.sigmoid((gravity[:, 2]+.5)/.04), state[:, 41]), -1)
        # Stop imagining beyond the sensed patch rather than extrapolating terrain.
        dxy = state[:, :2]-context[:, :2]
        c, s = context[:, 3].cos(), context[:, 3].sin()
        delta = torch.stack((c*dxy[:, 0]+s*dxy[:, 1], -s*dxy[:, 0]+c*dxy[:, 1]), -1).abs().amax(-1)
        boundary = torch.sigmoid((delta-.4)/.03)
        boundary = torch.where(delta >= .5, torch.ones_like(boundary), boundary)
        return 1.-(1.-risks).prod(-1)*(1.-boundary)


def state_features(state, context, spec=None):
    height, scan = geometry(state, context)
    canonical = (spec or QuadrupedStateSpec(12)).canonical(state[:, :37])
    canonical = torch.cat((height[:, None], canonical[:, 1:]), -1)
    return torch.cat((canonical, state[:, 37:], height[:, None], scan.clamp(-2., 2.)), -1)


class FootKinematics:
    def __init__(self, xml, device):
        import mujoco
        from .simulator import load_lgdog_model
        from .kinematics import LgDogKinematics
        model = load_lgdog_model(xml)
        self.fk = LgDogKinematics(model, device)
        self.ids = [3, 6, 9, 12]
        geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_calf1_collision")
                    for leg in ("FR", "FL", "RR", "RL")]
        self.offsets = torch.tensor(model.geom_pos[geom_ids], dtype=torch.float32, device=device)

    def __call__(self, state):
        pos, quat, vel, ang = self.fk(state[:, :37])
        offset = quat_apply(quat[:, self.ids], self.offsets[None].expand(len(state), -1, -1))
        return pos[:, self.ids]+offset, vel[:, self.ids]+torch.cross(ang[:, self.ids], offset, dim=-1)


def reward(state, action, previous, command, phase, context, feet):
    xy, yaw = velocity_error(state, command)
    height = ground_clearance(state, context)
    gravity = quat_apply_inverse(state[:, 3:7], state.new_tensor([0., 0., -1.]).expand(len(state), 3))
    foot_pos, foot_vel = feet(state)
    clearance = foot_pos[..., 2] - sample_ground(context, foot_pos[..., :2]) - .012
    # Diagonal trot contact prior, driven only by phase, not expert trajectories.
    wave = phase[:, :1] * state.new_tensor([1., -1., -1., 1.])[None]
    swing = torch.sigmoid(wave*10.)
    contact = state[:, 37:41]
    moving = (command.norm(dim=-1) > .1).float()
    gait = (1.-(contact-(1.-swing)).square().mean(-1))*moving
    lift = torch.exp(-((clearance-.06*swing).square()/.03**2).mean(-1))*moving
    slip = (contact*foot_vel[..., :2].square().sum(-1)).mean(-1)
    q, qd = state[:, 7:19], state[:, 25:37]
    default = state.new_tensor(lgdog1_spec().default_joint_pos)
    kp, kd = state.new_tensor([20., 20., 40.]*4), state.new_tensor([1., 1., 2.]*4)
    limits = state.new_tensor([10., 10., 20.]*4)
    torque = ((default+.25*action-q)*kp-qd*kd).clamp(-limits, limits)
    energy = (torque*qd).abs().mean(-1)
    qlimits = state.new_tensor(lgdog1_spec().joint_pos_limits)
    joint_limit = (torch.relu(qlimits[:, 0]-q)+torch.relu(q-qlimits[:, 1])).square().mean(-1)
    return (2.*torch.exp(-xy.square()/.25) + torch.exp(-yaw.square()/.25)
            + .5*(-gravity[:, 2]).clamp(-1., 1.) + .5*torch.exp(-(height-.30).square()/.01)
            + .3*gait + .3*lift - .05*(action-previous).square().mean(-1)
            - .05*slip - .002*energy - 2.*joint_limit - 2.*state[:, 41])


class VelocityReplay:
    def __init__(self, capacity, n, history, device):
        self.capacity, self.n, self.count = capacity, n, 0
        dims = dict(state=(42,), next=(42,), context=(CONTEXT_DIM,), history=(history, 47),
                    action=(12,), previous=(12,), command=(3,), phase=(2,), terminal=(), reset=())
        self.rows = {k: torch.zeros(capacity, n, *shape, device=device) for k, shape in dims.items()}

    def add(self, **data):
        for key, value in data.items():
            self.rows[key][self.count % self.capacity].copy_(value.detach())
        self.count += 1

    def sample(self, batch, horizon=1):
        if min(self.capacity, self.count) < horizon:
            raise ValueError("not enough replay")
        device = self.rows["state"].device
        start = torch.randint(max(0, self.count-self.capacity), self.count-horizon+1, (batch,), device=device)
        env = torch.randint(self.n, (batch,), device=device)
        index = (start[None]+torch.arange(horizon, device=device)[:, None]) % self.capacity
        rows = {k: v[index, env] for k, v in self.rows.items()}
        rows["valid"] = torch.cat((torch.ones_like(rows["reset"][:1]),
                                   (1.-rows["reset"][:-1]).cumprod(0)), 0)
        return rows

"""Velocity-task geometry, observations, rewards and physical failure rules."""
import torch
from torch.nn import functional as F
from ...config import lgdog1_spec
from ...state import QuadrupedStateSpec, heading_quat, quat_apply, quat_apply_inverse
from ...velocity_physics import GRID, offsets
from ...velocity_objective import objective_from


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
    value = F.grid_sample(context[:, 4:4+GRID*GRID].reshape(-1, 1, GRID, GRID), grid,
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


def frame(state, previous_action, command, phase, depth=None):
    gravity = quat_apply_inverse(state[:, 3:7], state.new_tensor([0., 0., -1.]).expand(len(state), 3))
    features = torch.cat((state[:, 22:25], gravity, command, phase,
                      state[:, 7:19] - state.new_tensor(lgdog1_spec().default_joint_pos),
                      state[:, 25:37], previous_action), -1)
    return features if depth is None else torch.cat((features, depth), -1)


def phase_at(steps):
    angle = steps*.02/.6*(2*torch.pi)
    return torch.stack((angle.sin(), angle.cos()), -1)


def failure(state, context):
    height = ground_clearance(state, context)
    gravity = quat_apply_inverse(state[:, 3:7], state.new_tensor([0., 0., -1.]).expand(len(state), 3))
    return (height < .14) | (height > .65) | (gravity[:, 2] > -.5) | (state[:, 41] > .5)


def command_velocity(state):
    """Heading-frame planar velocity and world yaw rate in physical units."""
    velocity = quat_apply_inverse(heading_quat(state[:, 3:7]), state[:, 19:22])
    yaw_rate = quat_apply(state[:, 3:7], state[:, 22:25])[:, 2]
    return velocity[:, :2], yaw_rate


def velocity_error(state, command):
    velocity, yaw_rate = command_velocity(state)
    return (velocity[:, :2] - command[:, :2]).norm(dim=-1), (yaw_rate-command[:, 2]).abs()


def command_progress(velocity, yaw_rate, command):
    """Signed achievement of active planar/turn commands, averaged when both exist.

    Includes pure turns and reverse/sideways commands. The linear derivative at
    rest is retained; there is no hard minimum-speed threshold on the robot.
    """
    speed = command[:, :2].norm(dim=-1)
    planar = (speed > 1e-3).float()
    turning = (command[:, 2].abs() > 1e-3).float()
    direction = command[:, :2]/speed.clamp_min(1e-6)[:, None]
    linear = (velocity*direction).sum(-1)/speed.clamp_min(.1)
    angular = yaw_rate*command[:, 2].sign()/command[:, 2].abs().clamp_min(.1)
    return ((linear*planar+angular*turning)/(planar+turning).clamp_min(1.)).clamp(-1., 1.)


def state_features(state, context, spec=None):
    height, scan = geometry(state, context)
    canonical = (spec or QuadrupedStateSpec(12)).canonical(state[:, :37])
    canonical = torch.cat((height[:, None], canonical[:, 1:]), -1)
    return torch.cat((canonical, state[:, 37:], height[:, None], scan.clamp(-2., 2.)), -1)


class FootKinematics:
    def __init__(self, xml, device):
        import mujoco
        from ...simulator import load_lgdog_model
        from ...kinematics import LgDogKinematics
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


def reward(state, action, previous, command, phase, context, feet, objective=None, return_terms=False):
    cfg = objective_from(objective)
    local_velocity, yaw_rate = command_velocity(state)
    xy, yaw = (local_velocity-command[:, :2]).norm(dim=-1), (yaw_rate-command[:, 2]).abs()
    height = ground_clearance(state, context)
    gravity = quat_apply_inverse(state[:, 3:7], state.new_tensor([0., 0., -1.]).expand(len(state), 3))
    foot_pos, foot_vel = feet(state)
    clearance = foot_pos[..., 2] - sample_ground(context, foot_pos[..., :2]) - .012
    # Diagonal trot contact prior, driven only by phase, not expert trajectories.
    wave = phase[:, :1] * state.new_tensor([1., -1., -1., 1.])[None]
    swing = torch.sigmoid(wave*10.)
    contact = state[:, 37:41]
    moving = (command.norm(dim=-1) > 1e-3).float()
    # Signed progress along the commanded direction. Unlike exp(-err^2) this is linear,
    # so it keeps a constant gradient while the robot is standing still — that is the
    # signal that pulls the policy out of the standing local optimum.
    speed = command[:, :2].norm(dim=-1)
    direction = command[:, :2]/speed.clamp_min(1e-6)[:, None]
    progress = ((local_velocity*direction).sum(-1)/speed.clamp_min(.1)).clamp(-1., 1.)*moving
    gait = (1.-(contact-(1.-swing)).square().mean(-1))*moving
    lift = torch.exp(-((clearance-cfg.swing_height*swing).square()/.03**2).mean(-1))*moving
    slip = (contact*foot_vel[..., :2].square().sum(-1)).mean(-1)
    q, qd = state[:, 7:19], state[:, 25:37]
    default = state.new_tensor(lgdog1_spec().default_joint_pos)
    kp, kd = state.new_tensor([20., 20., 40.]*4), state.new_tensor([1., 1., 2.]*4)
    limits = state.new_tensor([10., 10., 20.]*4)
    torque = ((default+.25*action-q)*kp-qd*kd).clamp(-limits, limits)
    energy = (torque*qd).abs().mean(-1)
    qlimits = state.new_tensor(lgdog1_spec().joint_pos_limits)
    joint_limit = (torch.relu(qlimits[:, 0]-q)+torch.relu(q-qlimits[:, 1])).square().mean(-1)
    raw = dict(velocity=torch.exp(-xy.square()/cfg.velocity_tolerance**2), progress=progress,
        yaw=torch.exp(-yaw.square()/cfg.yaw_tolerance**2), upright=(-gravity[:, 2]).clamp(-1., 1.),
        height=torch.exp(-(height-cfg.target_height).square()/.01), gait=gait, lift=lift,
        action_rate=(action-previous).square().mean(-1), slip=slip, energy=energy,
        joint_limit=joint_limit, trunk_contact=state[:, 41])
    if cfg.motion_reward_mode == 'progress':
        progress = command_progress(local_velocity, yaw_rate, command)*moving
        raw['progress'] = progress
        # Subtract the tracking score of a stationary robot for movement
        # commands. Zero yaw and modest forward targets no longer pay for rest.
        raw['velocity'] = raw['velocity']-moving*torch.exp(-speed.square()/cfg.velocity_tolerance**2)
        raw['yaw'] = raw['yaw']-moving*torch.exp(-command[:, 2].square()/cfg.yaw_tolerance**2)
        # Keep explicit standing commands unchanged. During locomotion, posture
        # and gait income requires progress; retain gradients through the gate
        # so even a tiny correctly directed velocity is better than standing.
        gate = (1.-moving)+progress.clamp(0., 1.)*moving
        # Gate positive income only: an inverted body must retain its penalty.
        raw['upright'] = raw['upright'].clamp_min(0.)*gate+raw['upright'].clamp_max(0.)
        for key in ('height', 'gait', 'lift'):
            raw[key] = raw[key]*gate
    terms = {key: getattr(cfg, 'reward_'+key)*value for key, value in raw.items()}
    total = sum(terms.values())
    return (total, terms) if return_terms else total


def soft_termination(state, context):
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

"""Serializable task settings shared by BPTT, HIM/PPO and the workbench."""
from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class VelocityObjective:
    reward_velocity: float = 2.
    reward_progress: float = .5
    reward_yaw: float = 1.
    reward_upright: float = .5
    reward_height: float = .5
    reward_gait: float = .3
    reward_lift: float = .3
    reward_action_rate: float = -.05
    reward_slip: float = -.05
    reward_energy: float = -.002
    reward_joint_limit: float = -2.
    reward_trunk_contact: float = -2.
    target_height: float = .30
    swing_height: float = .06
    velocity_tolerance: float = .5
    yaw_tolerance: float = .5

    def as_dict(self):
        return asdict(self)


OBJECTIVE_BOUNDS = {
    **{f.name: ((-100., 0.) if f.default < 0 else (0., 100.))
       for f in fields(VelocityObjective) if f.name.startswith('reward_')},
    'target_height': (.15, .60), 'swing_height': (0., .35),
    'velocity_tolerance': (.05, 2.), 'yaw_tolerance': (.05, 2.),
}


def objective_from(source=None):
    if isinstance(source, VelocityObjective):
        return source
    source = source or {}
    values = {}
    for f in fields(VelocityObjective):
        raw = source.get(f.name, f.default) if isinstance(source, dict) else getattr(source, f.name, f.default)
        value = float(raw)
        low, high = OBJECTIVE_BOUNDS[f.name]
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{f.name} must be finite and in [{low}, {high}]')
        values[f.name] = value
    return VelocityObjective(**values)


def command_settings(source):
    def get(key, default):
        return source.get(key, default) if isinstance(source, dict) else getattr(source, key, default)
    mode = get('command_mode', 'random')
    if mode not in ('random', 'fixed'):
        raise ValueError('command_mode must be random or fixed')
    limits = [float(get(k, v)) for k, v in (('command_vx', .6), ('command_vy', .3), ('command_yaw', .6))]
    low = 0. if mode == 'random' else -3.
    if any(not math.isfinite(v) or not low <= v <= 3. for v in limits):
        raise ValueError('random command limits must be in [0,3]; fixed commands in [-3,3]')
    standing = float(get('standing_fraction', .1))
    if not math.isfinite(standing) or not 0 <= standing <= 1:
        raise ValueError('standing_fraction must be in [0,1]')
    return dict(mode=mode, values=limits, standing_fraction=standing, resampling_seconds=5.)


def sample_commands(source, n, device):
    import torch
    cfg = command_settings(source)
    values = torch.tensor(cfg['values'], device=device)
    command = ((torch.rand(n, 3, device=device)*2.-1.)*values if cfg['mode'] == 'random'
               else values.expand(n, -1).clone())
    command[torch.rand(n, device=device) < cfg['standing_fraction']] = 0.
    return command


def add_objective_arguments(parser):
    parser.add_argument('--command-mode', choices=('random', 'fixed'), default='random',
                        help='random symmetric ranges or exact fixed [vx,vy,yaw] targets')
    parser.add_argument('--standing-fraction', type=float, default=.1)
    for f in fields(VelocityObjective):
        parser.add_argument('--'+f.name.replace('_', '-'), type=float, default=f.default)

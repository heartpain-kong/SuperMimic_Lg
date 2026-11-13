"""Physical locomotion acceptance, independent of training rewards and losses."""
import math


def assess_locomotion(episodes, baseline, seconds):
    """Require motion in every paired planar-command episode, not just survival.

    This is a short evaluation gate, not evidence of generalization. Standing
    and pure-turn commands need a separate protocol and are never labelled as
    verified planar locomotion here.
    """
    if not episodes or len(episodes) != len(baseline) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('paired nonempty episodes and positive seconds required')
    checks = []
    for row, zero in zip(episodes, baseline):
        if row['seed'] != zero['seed']:
            raise ValueError('evaluation and zero-action seeds differ')
        # A fall stops sampling early. Use the full requested command distance,
        # otherwise an immediate fall could look like an absent travel command.
        target = row.get('planned_distance_m', row['commanded_distance_m'])
        if target < .1:
            return dict(status='not_applicable', motion_verified=False,
                        reason='planar travel command is too small; standing/pure turning needs a separate gate')
        speed = target/seconds
        fields = (row['command_aligned_progress_m'], row['velocity_error_xy_mps'], row['survival_seconds'],
                  zero['velocity_error_xy_mps'], target)
        finite = all(math.isfinite(v) for v in fields)
        gates = dict(finite=finite, survived=not row['failed'] and row['survival_seconds'] >= .99*seconds,
            commanded_progress=finite and row['command_aligned_progress_m'] >= .7*target,
            velocity_tracking=finite and row['velocity_error_xy_mps'] <= max(.05, .5*speed),
            beats_zero_action=finite and row['velocity_error_xy_mps'] <= zero['velocity_error_xy_mps']-max(.02, .1*speed))
        checks.append(dict(seed=row['seed'], passed=all(gates.values()), gates=gates,
                           progress_fraction=row['command_aligned_progress_m']/target if finite else None))
    passed = all(row['passed'] for row in checks)
    return dict(status='passed' if passed else 'needs_training', motion_verified=passed, episodes=checks,
        criteria=dict(min_progress_fraction=.7, max_error_fraction_of_command_speed=.5,
                      max_error_floor_mps=.05, min_error_improvement_fraction=.1,
                      min_error_improvement_floor_mps=.02, all_seeds_must_pass=True))

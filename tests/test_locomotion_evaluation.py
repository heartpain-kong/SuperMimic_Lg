import copy
import pytest
from supermimic_lg.evaluation import assess_locomotion


def episode(seed, distance=0., error=.4, failed=False):
    return dict(seed=seed, commanded_distance_m=4., command_aligned_progress_m=distance,
                velocity_error_xy_mps=error, survival_seconds=10., failed=failed)


def test_standing_is_rejected_even_when_it_survives_with_small_model_loss():
    rows = [episode(seed) for seed in (0,1,42)]
    report = assess_locomotion(rows, rows, 10.)
    assert report['status'] == 'needs_training' and not report['motion_verified']
    assert all(row['gates']['survived'] and not row['gates']['commanded_progress'] for row in report['episodes'])


def test_tracking_must_pass_on_every_seed_and_beat_zero_action():
    zero = [episode(seed) for seed in (0,1,42)]
    policy = [episode(seed, 3.2, .08) for seed in (0,1,42)]
    assert assess_locomotion(policy, zero, 10.)['motion_verified']
    bad = copy.deepcopy(policy); bad[-1]['failed'] = True
    assert not assess_locomotion(bad, zero, 10.)['motion_verified']
    assert not assess_locomotion(policy, policy, 10.)['motion_verified']


def test_nonfinite_or_incomplete_survival_cannot_pass():
    zero = [episode(0)]
    row = episode(0, 3.2, float('nan'))
    assert not assess_locomotion([row], zero, 10.)['motion_verified']
    row = episode(0, 3.2, .08); row['survival_seconds'] = 2.
    assert not assess_locomotion([row], zero, 10.)['motion_verified']
    row = episode(0, 0., .4, failed=True)
    row.update(commanded_distance_m=.008, planned_distance_m=4., survival_seconds=.02)
    assert assess_locomotion([row], zero, 10.)['status'] == 'needs_training'


def test_standing_and_turn_only_are_not_planar_locomotion_evidence():
    row = episode(0); row['commanded_distance_m'] = 0.
    assert assess_locomotion([row], [row], 10.)['status'] == 'not_applicable'
    with pytest.raises(ValueError, match='seeds differ'):
        assess_locomotion([episode(0)], [episode(1)], 10.)

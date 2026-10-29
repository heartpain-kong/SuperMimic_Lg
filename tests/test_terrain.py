import pytest
import torch

from supermimic_lg.terrain import curriculum_decisions


def test_curriculum_requires_aligned_travel_and_holds_short_or_standing_episodes():
    # standing, stalled, backward, short tail, sustained travel, sideways, fall
    actual = torch.tensor([[0., 0.], [0., 0.], [-3., 0.], [3., 0.],
                           [2.5, 0.], [0., 3.], [3., 0.]])
    reference = torch.tensor([[0., 0.]] + [[3., 0.]] * 6)
    up, down, progress = curriculum_decisions(
        actual, reference, torch.tensor([150, 150, 150, 20, 150, 150, 150]),
        torch.tensor([False] * 6 + [True]))
    assert up.tolist() == [False, False, False, False, True, False, False]
    assert down.tolist() == [False, True, True, False, False, True, True]
    assert torch.isfinite(progress).all()


def test_mjlab_config_accepts_contact_allocation_fields():
    pytest.importorskip("mjlab.sim")
    from supermimic_lg.simulator import mjlab_simulation_config
    rough = mjlab_simulation_config("rough")
    assert rough.nconmax == 35 and rough.contact_sensor_maxmatch == 500
    assert rough.mujoco.ccd_iterations == 500
    flat = mjlab_simulation_config("flat")
    assert flat.nconmax is None and flat.mujoco.ccd_iterations == 50

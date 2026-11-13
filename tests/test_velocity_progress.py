"""Movement commands must reward motion rather than stationary posture income."""
from dataclasses import replace
import os
from pathlib import Path
import pytest
import torch

from supermimic_lg.config import lgdog1_spec
from supermimic_lg.tasks.velocity.mdp import reward, phase_at
from supermimic_lg.velocity_objective import VelocityObjective, objective_from


def case(command=(.4, 0., 0.)):
    state = torch.zeros(1, 42)
    state[:, 2], state[:, 3] = .3, 1.
    state[:, 7:19] = torch.tensor(lgdog1_spec().default_joint_pos)
    state[:, 37:41] = 1.
    action, context = torch.zeros(1, 12), torch.zeros(1, 85)

    def feet(s):
        return (s[:, None, :3]+s.new_tensor([0., 0., -.288])).expand(-1, 4, -1), s.new_zeros(len(s), 4, 3)

    args = (action, action, torch.tensor([command]), phase_at(torch.zeros(1)), context, feet)
    return state, args


def test_movement_reward_has_no_static_income():
    state, args = case()
    total, terms = reward(state, *args, return_terms=True)
    torch.testing.assert_close(total, torch.zeros_like(total), atol=1e-7, rtol=0)
    for name in ('velocity', 'progress', 'yaw', 'upright', 'height', 'gait', 'lift'):
        assert terms[name].item() == 0., name


@pytest.mark.parametrize('command', [(.4, 0., 0.), (-.4, 0., 0.), (0., .4, 0.),
                                   (0., -.4, 0.), (0., 0., .4), (0., 0., -.4)])
def test_tracking_is_preferred_to_standing_and_wrong_direction(command):
    state, args = case(command)
    state.requires_grad_()
    stationary = reward(state, *args)
    grad = torch.autograd.grad(stationary.sum(), state)[0]
    desired = torch.tensor(command)
    assert float(grad[0, [19, 20, 24]] @ desired) > 0
    moving = state.detach().clone()
    moving[0, [19, 20, 24]] = desired
    backward = state.detach().clone()
    backward[0, [19, 20, 24]] = -desired
    assert reward(moving, *args).item() > stationary.item()+.5
    assert reward(backward, *args).item() < stationary.item()


def test_explicit_standing_commands_keep_stability_rewards():
    state, args = case((0., 0., 0.))
    current = reward(state, *args)
    legacy = reward(state, *args, objective=VelocityObjective(motion_reward_mode='legacy'))
    torch.testing.assert_close(current, legacy)
    assert current.item() == 4.


def test_legacy_formula_remains_available_and_mode_is_validated():
    state, args = case()
    legacy = VelocityObjective(motion_reward_mode='legacy')
    total, terms = reward(state, *args, objective=legacy, return_terms=True)
    torch.testing.assert_close(terms['velocity'], torch.tensor([2.])*torch.exp(torch.tensor([-.4**2/.5**2])))
    assert total.item() > 3.
    assert objective_from(legacy.as_dict()) == legacy
    assert VelocityObjective(3.).reward_velocity == 3.  # preserve numeric positional arguments
    with pytest.raises(ValueError, match='motion_reward_mode'):
        objective_from({'motion_reward_mode': 'invalid'})


def test_small_motion_is_rewarded_and_posture_still_matters():
    state, args = case()
    state[:, 19] = .02
    total, terms = reward(state, *args, return_terms=True)
    assert total.item() > 0 and terms['height'].item() > 0
    low = state.clone(); low[:, 2] = .15
    assert reward(low, *args).item() < total.item()
    cfg = replace(VelocityObjective(), reward_progress=1.)
    _, updated = reward(state, *args, objective=cfg, return_terms=True)
    torch.testing.assert_close(updated['progress'], 2*terms['progress'])


def test_inverted_body_penalty_is_not_removed_by_zero_progress():
    state, args = case()
    state[:, 3:7] = torch.tensor([0., 1., 0., 0.])
    total, terms = reward(state, *args, return_terms=True)
    assert terms['upright'].item() < 0 and total.item() < 0


def test_old_checkpoint_is_preserved_and_can_transfer_actor_but_not_old_value_objective(tmp_path):
    xml = os.environ.get('LGDOG_TEST_XML')
    if not xml or not Path(xml).is_file():
        pytest.skip('set LGDOG_TEST_XML for real checkpoint migration')
    from supermimic_lg.cli import parser
    from supermimic_lg.runners.velocity import VelocityTrainer
    from supermimic_lg.checkpointing import file_hash, policy_state
    args = parser().parse_args(['--algorithm','xmimic','--task','velocity','--robot-xml',xml,
        '--backend','mujoco','--device','cpu','--num-envs','2','--iterations','1',
        '--wm-hidden','16,16','--actor-hidden','16,16','--critic-hidden','16,16',
        '--batch-size','4','--horizon','2','--wm-horizon','2','--warmup-steps','0',
        '--wm-updates','1','--policy-updates','1','--critic-epochs','1','--critic-minibatches','1',
        '--output-dir',str(tmp_path/'source')])
    trainer = VelocityTrainer(args); trainer.run()
    ck = torch.load(tmp_path/'source/checkpoint.pt', weights_only=True)
    ck['metadata']['objective'].pop('motion_reward_mode')  # archived checkpoints had no mode field
    old = tmp_path/'archived.pt'; torch.save(ck, old); digest = file_hash(old)
    args.resume = old; args.iterations = 2
    with pytest.raises(ValueError, match='checkpoint motion reward differs'):
        VelocityTrainer(args)
    args.resume = None; args.init_policy = old; args.output_dir = tmp_path/'transfer'
    initialized = VelocityTrainer(args)
    try:
        assert initialized.iteration == 0 and initialized.learner.counts == dict(wm=0, actor=0, critic=0)
        for key, value in policy_state(ck)[0].items():
            torch.testing.assert_close(initialized.policy.state_dict()[key], value)
    finally:
        initialized.sim.close()
    assert file_hash(old) == digest

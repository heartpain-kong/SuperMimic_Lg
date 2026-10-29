from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")

from supermimic_lg.config import WorldModelConfig, lgdog1_spec
from supermimic_lg.data.motion import MotionClip, save_motion_npz
from supermimic_lg.kinematics import LgDogKinematics
from supermimic_lg.online_data import MotionSequence, OnlineReplay, advance_history, observation
from supermimic_lg.policy import Policy, PolicyShape
from supermimic_lg.scripts.train_supermimic import Trainer, parser
from supermimic_lg.simulator import joint_addresses, load_lgdog_model
from supermimic_lg.world_model import WorldModel


@pytest.fixture
def robot_xml(tmp_path):
    if os.environ.get("LGDOG_TEST_XML"):
        return Path(os.environ["LGDOG_TEST_XML"])
    # A portable physical fixture with the same 12-joint tree, not a fake simulator.
    legs = ""
    for leg, x, y in (("FR", .15, -.1), ("FL", .15, .1), ("RR", -.15, -.1), ("RL", -.15, .1)):
        legs += f'''<body name="{leg}_hip" pos="{x} {y} 0"><joint name="{leg}_hip_joint" axis="1 0 0"/>
        <geom type="sphere" size=".025" mass=".2"/><body name="{leg}_thigh" pos="0 0 -.05"><joint name="{leg}_thigh_joint" axis="0 1 0"/>
        <geom type="capsule" size=".02 .05" mass=".2"/><body name="{leg}_calf" pos="0 0 -.1"><joint name="{leg}_calf_joint" axis="0 1 0"/>
        <geom type="capsule" size=".02 .05" mass=".2"/></body></body></body>'''
    path = tmp_path / "robot.xml"
    path.write_text('<mujoco><worldbody><body name="trunk"><freejoint/><geom type="box" size=".2 .1 .05" mass="3"/>' + legs + '</body></worldbody></mujoco>')
    return path


def standing_motion(xml, path):
    model = load_lgdog_model(xml)
    states = torch.zeros(64, 37)
    states[:, 2], states[:, 3] = .31, 1.
    states[:, 7:19] = torch.tensor(lgdog1_spec().default_joint_pos)
    p, q, v, w = LgDogKinematics(model)(states)
    save_motion_npz(path, MotionClip(states[:, 7:19].numpy(), states[:, 25:37].numpy(),
                                   p.numpy(), q.numpy(), v.numpy(), w.numpy()))
    return path


def test_frame_history_preserves_time_order_and_policy_gradients():
    shape = PolicyShape(12, 13)
    state = torch.zeros(2, 37)
    state[:, 3] = 1.
    context, action = torch.zeros(2, 16), torch.zeros(2, 12)
    state[:, 7:19] = .1
    first = observation(state, action, context)
    history = first[:, None].repeat(1, shape.history, 1)
    state[:, 7:19] = .2
    second = observation(state, action, context)
    history = advance_history(state, action, context, history)
    state[:, 7:19] = .3
    third = observation(state, action, context)
    history = advance_history(state, action, context, history)
    assert history.shape == (2, 3, 42)
    torch.testing.assert_close(history, torch.stack((first, second, third), dim=1))
    policy = Policy(shape, encoder_hidden=(16,), decoder_hidden=(16,))
    goal = torch.zeros(2, shape.goal_dim)
    history.requires_grad_(True)
    output = policy(history, goal)
    assert output.shape == (2, 12)
    torch.testing.assert_close(output, policy(history.flatten(1), goal))
    output.square().sum().backward()
    assert torch.isfinite(history.grad).all()
    assert (history.grad.abs().sum(dim=(0, 2)) > 0).all()


def test_fk_matches_mujoco_link_poses_velocities_and_gradients(robot_xml):
    model = load_lgdog_model(robot_xml)
    fk = LgDogKinematics(model)
    rng = np.random.default_rng(7)
    states = torch.tensor(rng.normal(size=(8, 37)) * .2, dtype=torch.float32)
    states[:, 3:7] /= states[:, 3:7].norm(dim=-1, keepdim=True)
    states.requires_grad_(True)
    poses, quats, linear, angular = fk(states)
    qa, va = joint_addresses(model)
    for i, state in enumerate(states.detach().numpy()):
        data = mujoco.MjData(model)
        data.qpos[:7], data.qpos[qa] = state[:7], state[7:19]
        data.qvel[:6], data.qvel[va] = state[19:25], state[25:37]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(poses[i].detach().numpy(), data.xpos[fk.ids], atol=2e-5)
        assert np.min(np.abs((quats[i].detach().numpy() * data.xquat[fk.ids]).sum(-1))) > .99999
        velocity = np.zeros(6)
        for j, body in enumerate(fk.ids):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
            np.testing.assert_allclose(linear[i, j].detach().numpy(), velocity[3:], atol=2e-5)
            np.testing.assert_allclose(angular[i, j].detach().numpy(), velocity[:3], atol=2e-5)
    (poses.square().sum() + linear.square().sum() + angular.square().sum()).backward()
    assert torch.isfinite(states.grad).all() and states.grad.abs().sum() > 0


def test_replay_preserves_terminal_state_and_masks_reset(robot_xml, tmp_path):
    motion = MotionSequence(standing_motion(robot_xml, tmp_path / "motion.npz"), 2, "cpu")
    replay = OnlineReplay(4, 1, 84, "cpu")
    start = motion.states[:1].clone()
    end = start.clone()
    end[:, 2] += .1
    zeros = torch.zeros(1, 12)
    context = torch.zeros(1, 16)
    replay.add(start, end, zeros, zeros, context, torch.zeros(1, 84), torch.tensor([0]), torch.ones(1), torch.ones(1))
    replay.add(start + 10., start + 11., zeros, zeros, context, torch.zeros(1, 84), torch.tensor([0]), torch.zeros(1), torch.zeros(1))
    window = replay.sample_window(1, 2, motion)
    assert window.valid[:, 0].tolist() == [1., 0.]
    torch.testing.assert_close(window.states[1], end)
    model = WorldModel(lgdog1_spec(), cfg=WorldModelConfig(hidden=(8,), term_hidden=(8,)))
    loss1, _ = model.window_loss(window.as_batch(), window.reference_root)
    window.states[2] *= 100.
    loss2, _ = model.window_loss(window.as_batch(), window.reference_root)
    torch.testing.assert_close(loss1, loss2)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_real_physics_online_loop_and_resume(robot_xml, tmp_path, seed):
    motion = standing_motion(robot_xml, tmp_path / "motion.npz")
    args = parser().parse_args(["--robot-xml", str(robot_xml), "--motion-file", str(motion),
        "--backend", "mujoco", "--device", "cpu", "--num-envs", "2", "--iterations", "2",
        "--collect-steps", "4", "--wm-updates", "2", "--policy-updates", "1", "--warmup-iterations", "1",
        "--batch-size", "4", "--wm-horizon", "2", "--horizon", "2", "--goal-horizon", "2",
        "--replay-capacity", "8", "--network-size", "32", "--torch-threads", "2", "--seed", str(seed),
        "--output-dir", str(tmp_path / "run")])
    trainer = Trainer(args)
    assert trainer.shape.history == 3 and trainer.history.shape == (2, 3, 42)
    assert trainer.replay.histories.shape[-2:] == (3, 42)
    with torch.no_grad():
        actions = trainer.policy(trainer.history, trainer.motion.goal(trainer.frames))
        assert actions.shape == (2, 12)
    before = {k: v.clone() for k, v in trainer.policy.state_dict().items()}
    trainer.run()
    other_history = trainer.history[0].clone()
    trainer.reset(torch.tensor([1]))
    torch.testing.assert_close(trainer.history[0], other_history)
    torch.testing.assert_close(trainer.history[1], trainer.history[1, -1:].expand(3, -1))
    checkpoint = torch.load(tmp_path / "run/checkpoint.pt", weights_only=True)
    assert checkpoint["algorithm"] == "supermimic_bptt"
    assert checkpoint["metadata"]["policy_shape"]["history"] == 3
    assert checkpoint["policy_updates"] == 2 and checkpoint["wm_updates"] == 4
    assert any(not torch.equal(before[k], v) for k, v in checkpoint["policy"].items())
    metrics = [json.loads(x) for x in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["transitions"] == 16
    assert metrics[-1]["policy/policy_grad_norm"] > 0
    assert metrics[-1]["dynamics_action_grad_norm"] > 0
    args.resume, args.iterations = tmp_path / "run/checkpoint.pt", 3
    resumed = Trainer(args)
    for k, v in resumed.policy.state_dict().items():
        torch.testing.assert_close(v, checkpoint["policy"][k])
    resumed.run()
    assert resumed.iteration == 3 and resumed.policy_updates == 3
    assert resumed.collected_transitions == 24
    with pytest.raises(ValueError, match="must exceed checkpoint iteration"):
        Trainer(args)
    args.iterations, args.history = 4, 10
    with pytest.raises(ValueError, match="checkpoint history=3, current history=10"):
        Trainer(args)

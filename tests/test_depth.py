import os
from pathlib import Path
import numpy as np
import pytest
import torch

from supermimic_lg.depth import (DepthEncoder, DepthLagBuffer, corrupt_depth, depth_rays,
                                 depth_spec, parse_depth_size)
from supermimic_lg.velocity_task import (VelocityPolicy, VelocityWorldModel, VelocityReplay,
                                         frame, context_dim, frame_dim, CONTEXT_DIM)


def test_depth_rays_are_a_forward_down_pinhole_camera():
    origins, directions = depth_rays(6, 8, 60., pitch=30., device="cpu")
    assert origins.shape == directions.shape == (48, 3)
    torch.testing.assert_close(directions.norm(dim=-1), torch.ones(48))
    assert (directions[:, 0] > 0).all()          # camera looks forward
    assert (directions[:, 2] < 0).all()          # and downwards
    # Top rows look higher on the image column than bottom rows.
    top, bottom = directions[:8, 2].mean(), directions[-8:, 2].mean()
    assert top > bottom
    torch.testing.assert_close(origins, torch.tensor([.16, 0., .05]).expand(48, 3))
    flat_origins, flat = depth_rays(4, 4, 60., pitch=0., device="cpu")
    torch.testing.assert_close(flat.reshape(4, 4, 3)[1:3, :, 2].mean(), torch.tensor(0.), rtol=0., atol=1e-6)
    assert flat[:, 2].mean() > directions.reshape(6, 8, 3)[:, :, 2].mean()
    assert parse_depth_size("18x24") == (18, 24) and parse_depth_size("16") == (16, 16)
    with pytest.raises(ValueError):
        parse_depth_size("0x8")


def test_depth_encoder_maps_metres_to_bounded_features():
    spec = depth_spec("12x16", fov=60., range_m=3., feat=8)
    encoder = DepthEncoder(spec)
    image = torch.full((4, 12, 16), 3.)
    image[:, :3] = .5
    features = encoder(image)
    assert features.shape == (4, 8) and torch.isfinite(features).all()
    assert features.abs().max() <= 1.
    # Unclipped/zero input stays finite.
    assert torch.isfinite(encoder(torch.zeros(2, 12, 16))).all()


def test_depth_corruption_matches_a_stereo_sensor_model():
    spec = depth_spec("8x10", range_m=3., feat=4, noise=.02, dropout=.25)
    image = torch.full((2, 8, 10), 1.5)
    torch.manual_seed(0)
    noisy = corrupt_depth(image, spec)
    assert noisy.shape == image.shape
    assert (noisy >= 0).all() and (noisy <= 3).all()
    assert not torch.allclose(noisy, image)
    # Dropped pixels read as "no reading", i.e. the configured maximum range.
    assert (noisy == 3.).float().mean() > .05
    clean = depth_spec("8x10", feat=4)
    torch.testing.assert_close(corrupt_depth(image, clean), image)
    with pytest.raises(ValueError):
        depth_spec("8x10", noise=-1.)


def test_depth_lag_buffer_delays_by_configured_steps():
    lag = DepthLagBuffer(2, torch.zeros(1, 2, 2))
    seen = [float(lag.push(torch.full((1, 2, 2), float(step))).mean()) for step in range(5)]
    assert seen == [0., 0., 0., 1., 2.]
    immediate = DepthLagBuffer(0, torch.zeros(1, 2, 2))
    assert float(immediate.push(torch.full((1, 2, 2), 7.)).mean()) == 7.


def test_world_model_conditions_on_depth_and_trains_the_encoder():
    spec = depth_spec("10x12", feat=6)
    wm = VelocityWorldModel((16, 16), DepthEncoder(spec))
    assert wm.depth_feat == 6
    state = torch.zeros(2, 42); state[:, 2] = .3; state[:, 3] = 1.
    base = torch.zeros(2, CONTEXT_DIM)
    depth = torch.full((2, 10, 12), 1.5, requires_grad=False)
    depth[:, :2] = .4
    context = torch.cat((base, wm.encode_depth(depth)), -1)
    assert context.shape == (2, context_dim(6)) and wm.features(state, context).shape == (2, 124+6)
    predicted = wm.step(state, torch.zeros(2, 12), context)
    assert predicted.shape == (2, 42)
    predicted.sum().backward()
    assert wm.encoder.conv[0].weight.grad is not None
    assert torch.isfinite(wm.encoder.conv[0].weight.grad).all()


def test_depth_changes_actor_frame_and_replay_dtype():
    spec = depth_spec("8x8", feat=4)
    base_frame = frame(torch.cat((torch.zeros(1, 42), torch.tensor([[0.]])), -1)[:, :42],
                       torch.zeros(1, 12), torch.zeros(1, 3), torch.zeros(1, 2))
    assert base_frame.shape == (1, frame_dim(0))
    with_depth = frame(torch.cat((torch.zeros(1, 42), torch.tensor([[0.]])), -1)[:, :42],
                       torch.zeros(1, 12), torch.zeros(1, 3), torch.zeros(1, 2), torch.ones(1, 4))
    assert with_depth.shape == (1, frame_dim(4))
    policy = VelocityPolicy(history=2, hidden=(16, 16), depth_feat=4)
    assert policy(torch.zeros(3, 2, frame_dim(4))).shape == (3, 12)
    replay = VelocityReplay(4, 3, 2, "cpu", frame=frame_dim(4), depth_pixels=64)
    assert replay.rows["depth"].dtype == torch.float16
    assert replay.rows["history"].shape == (4, 3, 2, frame_dim(4))
    replay.add(state=torch.zeros(3, 42), next=torch.zeros(3, 42), context=torch.zeros(3, CONTEXT_DIM),
               history=torch.zeros(3, 2, frame_dim(4)), action=torch.zeros(3, 12),
               previous=torch.zeros(3, 12), command=torch.zeros(3, 3), phase=torch.zeros(3, 2),
               terminal=torch.zeros(3), reset=torch.zeros(3), depth=torch.full((3, 64), 2.))
    assert float(replay.rows["depth"][0].max()) == 2.


def test_reference_free_training_with_depth_on_real_physics(tmp_path):
    xml = os.environ.get("LGDOG_TEST_XML")
    if not xml or not Path(xml).is_file():
        pytest.skip("set LGDOG_TEST_XML for real LgDog integration")
    from supermimic_lg.scripts.train_supermimic import parser
    from supermimic_lg.scripts.train_velocity import VelocityTrainer
    common = ["--task", "velocity", "--robot-xml", xml, "--backend", "mujoco", "--device", "cpu",
              "--num-envs", "2", "--iterations", "2", "--collect-steps", "3", "--wm-updates", "1",
              "--policy-updates", "1", "--warmup-iterations", "1", "--batch-size", "4",
              "--wm-hidden", "16,16", "--actor-hidden", "16,16", "--critic-hidden", "16,16", "--horizon", "2", "--wm-horizon", "2", "--episode-steps", "4",
              "--command-mode", "fixed", "--standing-fraction", "0", "--depth",
              "--depth-size", "8x10", "--depth-feat", "6",
              "--depth-noise", "0.01", "--depth-dropout", "0.05", "--depth-latency", "1"]
    args = parser().parse_args(common + ["--output-dir", str(tmp_path/"depth")])
    trainer = VelocityTrainer(args)
    assert trainer.depth_spec["pixels"] == 80 and trainer.depth_feature_dim == 6
    assert trainer.depth_spec["noise"] == .01 and trainer.depth_spec["latency"] == 1
    image = trainer.sim.depth()
    assert image.shape == (2, 8, 10) and torch.isfinite(image).all()
    # A downward camera on flat ground must see the floor closer than max range.
    assert float(image.min()) < trainer.depth_spec["range"]
    trainer.run()
    checkpoint = torch.load(tmp_path/"depth/checkpoint.pt", weights_only=True)
    assert checkpoint["metadata"]["depth"]["feat"] == 6
    assert any(key.startswith("encoder.") for key in checkpoint["wm"])
    assert checkpoint["metadata"]["frame_dim"] == frame_dim(6)

    args.resume = tmp_path/"depth/checkpoint.pt"; args.iterations = 3
    args.depth_size = "8x12"
    with pytest.raises(ValueError, match="contract differs"):
        VelocityTrainer(args)
    args.depth_size = "8x10"
    resumed = VelocityTrainer(args)
    resumed.run()
    assert resumed.iteration == 3
    args.resume = None; args.init_policy = tmp_path/"depth/checkpoint.pt"; args.output_dir = tmp_path/"transfer"
    transferred = VelocityTrainer(args)
    try:
        assert transferred.iteration == 0
        for key, value in resumed.wm.encoder.state_dict().items():
            torch.testing.assert_close(transferred.wm.encoder.state_dict()[key], value)
    finally:
        transferred.sim.close()


def test_replay_depth_is_causal_and_sensed_once_per_step(tmp_path, monkeypatch):
    xml = os.environ.get("LGDOG_TEST_XML")
    if not xml or not Path(xml).is_file():
        pytest.skip("set LGDOG_TEST_XML for real LgDog integration")
    from supermimic_lg.scripts.train_supermimic import parser
    from supermimic_lg.scripts.train_velocity import VelocityTrainer
    args = parser().parse_args(["--task", "velocity", "--robot-xml", xml,
        "--backend", "mujoco", "--device", "cpu", "--num-envs", "2",
        "--collect-steps", "3", "--episode-steps", "100", "--depth", "--depth-size", "8x8",
        "--depth-feat", "4", "--wm-hidden", "16,16", "--actor-hidden", "16,16",
        "--critic-hidden", "16,16", "--output-dir", str(tmp_path/"causal")])
    trainer = VelocityTrainer(args)
    calls = {"step": 0, "observe": 0, "depth": 0}
    step, observe = trainer.sim.step, trainer.sim.observe

    def stepped(action):
        step(action)
        calls["step"] += 1

    def observed():
        calls["observe"] += 1
        return observe()

    def depth():
        calls["depth"] += 1
        return torch.full((2, 8, 8), float(calls["step"] + 1)*.5)

    monkeypatch.setattr(trainer.sim, "step", stepped)
    monkeypatch.setattr(trainer.sim, "observe", observed)
    monkeypatch.setattr(trainer.sim, "depth", depth)
    try:
        trainer.reset(torch.arange(2), physics=False)
        calls["observe"] = calls["depth"] = 0
        trainer.collect()
        torch.testing.assert_close(trainer.replay.rows["depth"][:3, 0, 0].float(),
                                   torch.tensor([.5, 1., 1.5]))
        assert calls == {"step": 3, "observe": 3, "depth": 3}
    finally:
        trainer.sim.close()

import json

import pytest
import torch

pytest.importorskip("mujoco")

from test_online_supermimic import robot_xml, standing_motion
from supermimic_lg.scripts.train_supermimic import Trainer, parser
from supermimic_lg.viewer import PolicyPlayback
from supermimic_lg.simulator import load_lgdog_model


def test_checkpoint_playback_is_deterministic_and_uses_saved_history(robot_xml, tmp_path):
    physical, visual = load_lgdog_model(robot_xml), load_lgdog_model(robot_xml, visuals=True)
    for key in ("body_mass", "body_inertia", "jnt_axis", "jnt_range", "dof_armature", "actuator_gainprm", "actuator_biasprm"):
        import numpy as np
        np.testing.assert_allclose(getattr(physical, key), getattr(visual, key))
    motion = standing_motion(robot_xml, tmp_path / "motion.npz")
    args = parser().parse_args(["--robot-xml", str(robot_xml), "--motion-file", str(motion),
        "--backend", "mujoco", "--device", "cpu", "--num-envs", "1", "--iterations", "1",
        "--collect-steps", "4", "--wm-updates", "1", "--policy-updates", "1", "--warmup-iterations", "1",
        "--batch-size", "2", "--wm-horizon", "1", "--horizon", "1", "--history", "2", "--goal-horizon", "2",
        "--replay-capacity", "8", "--network-size", "16", "--torch-threads", "1", "--output-dir", str(tmp_path / "run")])
    Trainer(args).run()
    checkpoint = tmp_path / "run/checkpoint.pt"
    a = PolicyPlayback(checkpoint, motion, robot_xml, visuals=False)
    b = PolicyPlayback(checkpoint, motion, robot_xml, visuals=False)
    try:
        assert a.history.shape == (1, 2, 42)
        for _ in range(6):
            a.step()
            b.step()
            torch.testing.assert_close(a.sim.state(), b.sim.state(), rtol=0, atol=0)
        assert a.telemetry()["steps"] == 6
        a.reset()
        torch.testing.assert_close(a.history[:, 0], a.history[:, 1])
        wrong = tmp_path / "wrong.npz"
        wrong.write_bytes(b"different motion")
        with pytest.raises(ValueError, match="检查点不一致"):
            PolicyPlayback(checkpoint, wrong, robot_xml, visuals=False)
        ck = torch.load(checkpoint, weights_only=True)
        ck["algorithm"] = "ppo"
        invalid = tmp_path / "ppo.pt"
        torch.save(ck, invalid)
        with pytest.raises(ValueError, match="SuperMimic"):
            PolicyPlayback(invalid, motion, robot_xml, visuals=False)
    finally:
        a.sim.close()
        b.sim.close()

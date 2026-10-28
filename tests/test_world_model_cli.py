from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def test_training_cli_saves_loadable_checkpoint(tmp_path):
    from supermimic_lg.config import lgdog1_spec
    from supermimic_lg.synthetic import make_replay_window

    window = make_replay_window(lgdog1_spec(), 2, 2)
    replay = tmp_path / "replay.npz"
    np.savez(replay, states=window.states.numpy(), actions=window.actions.numpy(),
             contexts=window.contexts.numpy(), done=window.done.numpy())
    output = tmp_path / "model.pt"
    result = subprocess.run([sys.executable, "-m", "supermimic_lg.scripts.train_world_model",
                             str(replay), "--device", "cpu", "--steps", "2", "--batch-size", "2",
                             "--horizon", "2", "--save-interval", "1", "--output", str(output)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    state = torch.load(output, weights_only=True)
    assert state["step"] == 2
    assert state["context_dim"] == 16
    assert state["model"] and state["optimizer"]
    assert np.isfinite(state["metrics"]["loss"])
    assert '"step": 2' in result.stdout
    assert not output.with_suffix(".pt.tmp").exists()

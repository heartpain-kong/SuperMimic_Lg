from __future__ import annotations


def test_bridge_fails_with_actionable_path_error(tmp_path):
    from supermimic_lg.adapters.mjlab_tracking import add_mjlab_project

    try:
        add_mjlab_project(tmp_path)
    except FileNotFoundError as exc:
        assert "unitree_rl_mjlab_lgdog" in str(exc)
    else:
        raise AssertionError("missing mjlab checkout must be rejected")

"""Run the existing mjlab train CLI after registering the LgDog task."""

from __future__ import annotations

import argparse
import sys

from supermimic_lg.adapters.mjlab_tracking import register_lgdog_tracking_task, add_mjlab_project


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mjlab-root", default=r"E:\mjlab")
    parser.add_argument("--task-id", default="LgDog1-SuperMimic-Tracking")
    args, rest = parser.parse_known_args()
    add_mjlab_project(args.mjlab_root)
    register_lgdog_tracking_task(args.mjlab_root, args.task_id)
    # The upstream CLI expects the task id as its first positional argument.
    sys.argv = [sys.argv[0], args.task_id, *rest]
    from scripts.train import main as mjlab_train_main
    mjlab_train_main()


if __name__ == "__main__":
    main()

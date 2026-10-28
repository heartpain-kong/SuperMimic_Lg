"""Register the LgDog1 tracking task in an existing mjlab checkout."""

from __future__ import annotations

import argparse

from supermimic_lg.adapters.mjlab_tracking import register_lgdog_tracking_task


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mjlab-root", default=r"E:\mjlab")
    p.add_argument("--task-id", default="LgDog1-SuperMimic-Tracking")
    args = p.parse_args()
    register_lgdog_tracking_task(args.mjlab_root, args.task_id)
    print(f"registered {args.task_id}")


if __name__ == "__main__":
    main()

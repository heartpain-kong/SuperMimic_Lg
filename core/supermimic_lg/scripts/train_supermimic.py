"""Backward-compatible training entry point."""
from supermimic_lg.cli import main, parser
from supermimic_lg.runners.tracking import Trainer, file_hash

__all__ = ["main", "parser", "Trainer", "file_hash"]

if __name__ == "__main__":
    main()

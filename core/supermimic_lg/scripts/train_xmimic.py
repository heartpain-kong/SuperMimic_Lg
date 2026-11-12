"""Train the PyTorch XMimic-Lg velocity adaptation through the common CLI."""
import sys
from supermimic_lg.cli import main as train


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(token.split("=", 1)[0] == "--algorithm" for token in argv):
        raise SystemExit("train_xmimic already selects XMimic; use the common CLI to choose another algorithm")
    return train(["--algorithm", "xmimic", "--task", "velocity", *argv])


if __name__ == "__main__":
    main()

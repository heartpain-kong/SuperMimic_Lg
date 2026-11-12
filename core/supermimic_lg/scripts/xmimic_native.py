"""Check or launch the bundled upstream JAX/G1 XMimic in a separate environment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

NATIVE_ROOT = Path(__file__).resolve().parents[1]/"vendor/xmimic"


def source_status(root):
    manifest = json.loads((root/"SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    invalid = [name for name, expected in manifest["files"].items()
               if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest() != expected]
    return dict(verified=not invalid, files=len(manifest["files"]), changed_or_missing=invalid)


def readiness(root, python, motion_dir):
    pinned = dict(line.strip().split("==", 1) for line in (root/"requirements.txt").read_text().splitlines()
                  if "==" in line and not line.lstrip().startswith("#"))
    probe = """import importlib.metadata as m,json,sys,platform
names=json.loads(sys.argv[1]);versions={}
for name in names:
    try: versions[name]=m.version(name)
    except m.PackageNotFoundError: versions[name]=None
print(json.dumps(dict(versions=versions,python=list(sys.version_info[:3]),system=platform.system())))
"""
    result = subprocess.run([str(python), "-c", probe, json.dumps(list(pinned))],
                            capture_output=True, text=True, timeout=20, check=False)
    if result.returncode:
        raise RuntimeError("cannot inspect native Python environment: "+result.stderr.strip())
    environment = json.loads(result.stdout)
    mismatches = {name: dict(expected=expected, installed=environment["versions"][name])
                  for name, expected in pinned.items() if environment["versions"][name] != expected}
    source = source_status(root)
    meshes = (root/"assets/g1/assets/meshes").is_dir()
    motion = motion_dir.is_dir() and any(motion_dir.glob("*.csv"))
    return dict(ready=(environment["system"] == "Linux" and not mismatches and source["verified"] and meshes and motion),
        environment=environment, pinned_version_mismatches=mismatches, source=source,
        g1_meshes_present=meshes, motion_data_present=motion, motion_dir=str(motion_dir),
        motion_format="upstream G1 CSV",
        gpu_runtime_validation="not performed; metadata/readiness check does not initialize JAX or test CUDA")


def build_command(root, python, motion_dir, forwarded):
    forwarded = forwarded[1:] if forwarded[:1] == ["--"] else list(forwarded)
    if not any(arg.split("=", 1)[0] == "--motion_dir" for arg in forwarded):
        forwarded += ["--motion_dir", str(motion_dir)]
    return [str(python), str(root/"train.py"), *forwarded]


def command_motion_dir(root, command):
    value = None
    for index, argument in enumerate(command[2:], 2):
        if argument == "--motion_dir":
            if index+1 >= len(command):
                raise ValueError("--motion_dir needs a path")
            value = command[index+1]
        elif argument.startswith("--motion_dir="):
            value = argument.split("=", 1)[1]
    path = Path(value).expanduser()
    return (root/path).resolve() if not path.is_absolute() else path.resolve()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=NATIVE_ROOT)
    p.add_argument("--python", default=sys.executable, help="Python executable in the separate upstream JAX environment")
    p.add_argument("--motion-dir", type=Path, help="upstream G1 CSV directory (not LgDog NPZ)")
    p.add_argument("--check", action="store_true", help="inspect source, versions and data without initializing JAX")
    p.add_argument("--print-command", action="store_true")
    p.add_argument("arguments", nargs=argparse.REMAINDER, help="upstream arguments after --, using their underscore names")
    args = p.parse_args()
    root = args.source_root.resolve()
    motion = (args.motion_dir or root/"data/lafan1_g1").resolve()
    command = build_command(root, args.python, motion, args.arguments)
    motion = command_motion_dir(root, command)
    if args.print_command:
        print(json.dumps(dict(command=command, cwd=str(root)), indent=2))
        return
    status = readiness(root, args.python, motion)
    print(json.dumps(status, indent=2), flush=True)
    if args.check:
        return
    if not status["ready"]:
        raise SystemExit("Native XMimic is not ready; prepare the separate Linux/JAX environment and G1 data shown above")
    raise SystemExit(subprocess.run(command, cwd=root, check=False).returncode)


if __name__ == "__main__":
    main()

"""Shared checkpoint writes; replacement is atomic for concurrent policy viewers."""
import hashlib
from pathlib import Path
import time
import torch

VELOCITY_ALGORITHMS = ("supermimic_velocity_bptt", "xmimic_velocity_bptt")


def policy_state(checkpoint, selection="auto"):
    if selection not in ("auto", "raw", "ema"):
        raise ValueError("policy selection must be auto, raw or ema")
    chosen = checkpoint.get("deployment_policy", "raw") if selection == "auto" else selection
    key = "policy_ema" if chosen == "ema" else "policy"
    if chosen not in ("raw", "ema") or key not in checkpoint:
        raise ValueError(f"checkpoint does not contain the requested {chosen} policy")
    return checkpoint[key], chosen


def resume_contract(metadata):
    """Batch sizes can change to fit device memory without changing weight semantics."""
    result = dict(metadata)
    for key in ("wm_batch_size", "policy_batch_size"):
        result.pop(key, None)
    if "learning" in result:
        result["learning"] = {key: value for key, value in result["learning"].items()
                              if key not in ("wm_batch_size", "policy_batch_size")}
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(checkpoint, path):
    path = Path(path)
    temporary = path.with_name(path.stem+".tmp"+path.suffix)
    torch.save(checkpoint, temporary)
    for attempt in range(30):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(.1)

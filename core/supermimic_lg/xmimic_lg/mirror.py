"""Optional sagittal WM augmentation for [FR,FL,RR,RL] LgDog states."""
import torch
from ..velocity_physics import GRID


def mirror_joints(value):
    perm = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    return value[..., perm]*value.new_tensor([-1., 1., 1.]*4)


def mirror_state(state):
    out = state.clone()
    out[..., :3] *= state.new_tensor([1., -1., 1.])
    out[..., 3:7] *= state.new_tensor([1., -1., 1., -1.])
    out[..., 7:19] = mirror_joints(state[..., 7:19])
    out[..., 19:22] *= state.new_tensor([1., -1., 1.])
    out[..., 22:25] *= state.new_tensor([-1., 1., -1.])
    out[..., 25:37] = mirror_joints(state[..., 25:37])
    out[..., 37:42] = state[..., [38, 37, 40, 39, 41]]
    return out


def mirror_context(context):
    if context.shape[-1] != 4+GRID*GRID:
        raise ValueError("XMimic WM mirroring supports blind terrain context only")
    out = context.clone()
    out[..., 1] *= -1.
    out[..., 3] *= -1.
    out[..., 4:] = context[..., 4:].reshape(*context.shape[:-1], GRID, GRID).flip(-1).flatten(-2)
    return out


def augment_windows(rows, fraction):
    count = int(rows["state"].shape[1]*fraction)
    if not count:
        return rows
    result = dict(rows)
    for key, transform in (("state", mirror_state), ("next", mirror_state),
                           ("action", mirror_joints), ("context", mirror_context)):
        value = rows[key]
        result[key] = torch.cat((transform(value[:, :count]), value[:, count:]), 1)
    return result

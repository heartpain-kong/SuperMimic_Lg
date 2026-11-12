"""LgDog velocity tensor layout, shared by physics, replay and networks."""
from ...velocity_physics import GRID
STATE_DIM = 42
CONTEXT_DIM = 4 + GRID*GRID
FRAME_DIM = 47

def context_dim(depth_feat=0):
    return CONTEXT_DIM + int(depth_feat)

def frame_dim(depth_feat=0):
    return FRAME_DIM + int(depth_feat)

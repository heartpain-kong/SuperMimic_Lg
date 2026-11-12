"""Episode-aware real-transition windows for the velocity learner."""
import torch
from .schema import CONTEXT_DIM, FRAME_DIM

class VelocityReplay:
    def __init__(self, capacity, n, history, device, frame=FRAME_DIM, depth_pixels=0):
        self.capacity, self.n, self.count = capacity, n, 0
        dims = dict(state=(42,), next=(42,), context=(CONTEXT_DIM,), history=(history, frame),
                    action=(12,), previous=(12,), command=(3,), phase=(2,), terminal=(), reset=())
        if depth_pixels:
            dims["depth"] = (depth_pixels,)
        self.rows = {k: torch.zeros(capacity, n, *shape,
                                    dtype=torch.float16 if k == "depth" else torch.float32,
                                    device=device) for k, shape in dims.items()}

    def add(self, **data):
        for key, value in data.items():
            self.rows[key][self.count % self.capacity].copy_(value.detach())
        self.count += 1

    def sample(self, batch, horizon=1, fields=None):
        if min(self.capacity, self.count) < horizon:
            raise ValueError("not enough replay")
        device = self.rows["state"].device
        start = torch.randint(max(0, self.count-self.capacity), self.count-horizon+1, (batch,), device=device)
        env = torch.randint(self.n, (batch,), device=device)
        index = (start[None]+torch.arange(horizon, device=device)[:, None]) % self.capacity
        keys = self.rows.keys() if fields is None else set(fields) | {"reset"}
        rows = {k: self.rows[k][index, env] for k in keys}
        rows["valid"] = torch.cat((torch.ones_like(rows["reset"][:1]),
                                   (1.-rows["reset"][:-1]).cumprod(0)), 0)
        return rows

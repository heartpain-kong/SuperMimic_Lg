"""Policy parameter/buffer EMA with first-update copy and resumable state."""
import copy
import torch


class PolicyEMA:
    def __init__(self, policy, decay=.995):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0,1)")
        self.model = copy.deepcopy(policy).eval().requires_grad_(False)
        self.decay, self.updates = float(decay), 0

    @torch.no_grad()
    def update(self, policy):
        live = policy.state_dict()
        groups = {}
        for key, averaged in self.model.state_dict().items():
            value = live[key]
            if not averaged.is_floating_point():
                averaged.copy_(value)
            else:
                groups.setdefault((averaged.device, averaged.dtype), []).append((averaged, value))
        for pairs in groups.values():
            averaged, values = zip(*pairs)
            if self.updates == 0:
                torch._foreach_copy_(averaged, values)
            else:
                torch._foreach_lerp_(averaged, values, 1.-self.decay)
        self.updates += 1

    def state_dict(self):
        return dict(decay=self.decay, updates=self.updates, model=self.model.state_dict())

    def load_state_dict(self, state):
        if state["decay"] != self.decay:
            raise ValueError("EMA decay differs from checkpoint")
        self.model.load_state_dict(state["model"])
        self.updates = state["updates"]

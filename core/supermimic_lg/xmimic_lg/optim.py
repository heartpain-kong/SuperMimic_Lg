"""PyTorch port of the local XMimic Muon/Adam equations.

Upstream: vendor/xmimic/xmimic/optim.py (MIT, Nico Bohlinger).
Hidden Linear matrices use Muon; boundary matrices, vectors and biases use Adam.
"""
import math
import torch


def newton_schulz(gradient, steps=5, eps=1e-7):
    dtype = torch.bfloat16 if gradient.device.type == "cuda" else torch.float32
    x = gradient.to(dtype)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x/x.float().norm().clamp_min(eps).to(dtype)
    for _ in range(steps):
        a = x @ x.T
        b = -4.775*a + 2.0315*(a @ a)
        x = 3.4445*x + b @ x
    return x.T if transposed else x


def hidden_matrices(model):
    """Select interior Linear layers within each MLP; exclude first/last layers."""
    selected = set()
    for module in model.modules():
        if isinstance(module, torch.nn.Sequential):
            layers = [child for child in module.children() if isinstance(child, torch.nn.Linear)]
            selected.update(id(layer.weight) for layer in layers[1:-1])
    return selected


class MuonAdam(torch.optim.Optimizer):
    def __init__(self, model, lr=4e-4, betas=(.9, .999), momentum=.95, ns_steps=5, eps=1e-8):
        super().__init__(model.parameters(), dict(lr=lr, betas=betas, momentum=momentum,
                                                ns_steps=ns_steps, eps=eps))
        self.muon_ids = hidden_matrices(model)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            adam_batches = {}
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise ValueError("MuonAdam requires dense gradients")
                state = self.state[parameter]
                if id(parameter) in self.muon_ids:
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    buffer = state["momentum_buffer"]
                    buffer.mul_(group["momentum"]).add_(gradient, alpha=1.-group["momentum"])
                    direction = gradient.lerp(buffer, group["momentum"])
                    update = newton_schulz(direction, group["ns_steps"]).to(parameter.dtype)
                    parameter.add_(update, alpha=-group["lr"]*.2*math.sqrt(max(parameter.shape)))
                else:
                    if not state:
                        state.update(step=0, exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
                    state["step"] += 1
                    # Keep the existing per-leaf state layout, including unequal
                    # step counts when a parameter has no gradient. Multi-tensor
                    # updates avoid launching each small Adam operation per leaf.
                    key = (parameter.device, parameter.dtype, state["step"])
                    adam_batches.setdefault(key, []).append((parameter, gradient, state))
            for (_, _, step), batch in adam_batches.items():
                params, gradients, states = zip(*batch)
                means = [s["exp_avg"] for s in states]
                variances = [s["exp_avg_sq"] for s in states]
                torch._foreach_mul_(means, beta1)
                torch._foreach_add_(means, gradients, alpha=1.-beta1)
                torch._foreach_mul_(variances, beta2)
                torch._foreach_addcmul_(variances, gradients, gradients, value=1.-beta2)
                denominator = torch._foreach_sqrt(variances)
                torch._foreach_div_(denominator, math.sqrt(1.-beta2**step))
                torch._foreach_add_(denominator, group["eps"])
                torch._foreach_addcdiv_(params, means, denominator, value=-group["lr"]/(1.-beta1**step))
        return loss

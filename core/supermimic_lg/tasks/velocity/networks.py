"""Velocity policy, critic and differentiable terrain-conditioned dynamics."""
import torch
from torch import nn
from ...state import QuadrupedStateSpec
from ...world_model import mlp
from ...norm import RunningNorm, normed_mlp
from ...velocity_physics import GRID
from .schema import CONTEXT_DIM, frame_dim
from .mdp import state_features, soft_termination


class VelocityPolicy(nn.Module):
    """Obs-normalised MLP with a small linear head, as the xmimic reference policy.

    The output is deliberately *not* squashed: a tanh head saturates and its
    gradient vanishes exactly where the actions matter, which is why the earlier
    version could not push the policy out of the initial joint pose.
    """

    def __init__(self, history=3, hidden=(512, 256, 128), depth_feat=0):
        super().__init__()
        self.history, self.hidden, self.depth_feat = history, tuple(hidden), int(depth_feat)
        self.obs_norm = RunningNorm(frame_dim(depth_feat))
        self.net = mlp(history*frame_dim(depth_feat), self.hidden, 12)
        with torch.no_grad():
            self.net[-1].weight.mul_(.01)
            self.net[-1].bias.zero_()

    def forward(self, history):
        return self.forward_normalized(self.obs_norm.apply(history))

    def forward_normalized(self, history):
        """Use a history whose frames were normalized when they were produced."""
        return self.net(history.flatten(1)).float()


class VelocityCritic(nn.Module):
    """Value head on normalised features (xmimic keeps a running norm on critic input)."""

    def __init__(self, hidden=(512, 256, 128), depth_feat=0):
        super().__init__()
        dim = 37+5+1+GRID*GRID+int(depth_feat)+5
        self.norm = RunningNorm(dim)
        self.net = mlp(dim, tuple(hidden), 1)

    def forward(self, features):
        return self.forward_normalized(self.norm.apply(features))

    def forward_normalized(self, features):
        return self.net(features).float().squeeze(-1)


class VelocityWorldModel(nn.Module):
    def __init__(self, hidden=(1024, 1024, 1024), depth_encoder=None):
        super().__init__()
        self.spec = QuadrupedStateSpec(12)
        self.encoder = depth_encoder
        self.hidden = tuple(hidden)
        self.depth_feat = 0 if depth_encoder is None else int(depth_encoder.spec["feat"])
        inputs = 37+5+1+GRID*GRID+self.depth_feat+12
        self.in_norm = RunningNorm(inputs)
        self.out_norm = RunningNorm(self.spec.delta_dim)
        self.error_norm = RunningNorm(self.spec.delta_dim, eps=1e-4)
        self.register_buffer("error_floor", torch.tensor(
            [.003]*3 + [.005]*3 + [.02]*3 + [.05]*3 + [.003]*12 + [.05]*12))
        self.dynamics = normed_mlp(inputs, self.hidden, self.spec.delta_dim, zero_last=True)
        self.contacts = mlp(inputs, (self.hidden[0],), 5)
        nn.init.constant_(self.contacts[-1].bias, -3.)

    def encode_depth(self, depth):
        """``[N, H, W]`` metres to ``[N, feat]``; None when depth is disabled."""
        return None if self.encoder is None else self.encoder(depth)

    def features(self, state, context):
        features = state_features(state, context, self.spec)
        if self.depth_feat:
            features = torch.cat((features, context[:, CONTEXT_DIM:]), -1)
        return features

    def raw_inputs(self, state, action, context):
        """Physical features before applying the running input statistics."""
        return torch.cat((self.features(state, context), action), -1)

    def condition(self, state, action, context):
        """Normalised network input for a state-action pair."""
        return self.in_norm.apply(self.raw_inputs(state, action, context))

    def step(self, state, action, context):
        return self.step_from_features(state, action, self.features(state, context))

    def step_from_features(self, state, action, features, normalizers=None):
        """Reuse the physical features for dynamics and later critic regression."""
        inputs = torch.cat((features, action), -1)
        if normalizers is None:
            x, output_std = self.in_norm.apply(inputs), self.out_norm.std()
        else:
            input_std, output_std = normalizers
            x = (inputs-self.in_norm.mean)/input_std
        physical = self.spec.integrate(state[:, :37], self.dynamics(x)*output_std + self.out_norm.mean)
        return torch.cat((physical, self.contacts(x).float().sigmoid()), -1)

    @torch.no_grad()
    def update_norms(self, state, action, context, next_state):
        """Fit the input and residual-output normalisers from real transitions."""
        self.in_norm.update(self.raw_inputs(state, action, context))
        self.out_norm.update(self.spec.delta_between(state, next_state))
        self.error_norm.update(self.spec.error_vector(next_state, state, normalized=False))

    def prediction_error(self, predicted, target):
        """Scale each error by real one-step increments, with physical floors.

        Fixed tolerances (5 cm position, 0.5 m/s velocity) made the loss small
        even when short-rollout velocity predictions were too poor for BPTT.
        """
        error = self.spec.error_vector(predicted[:, :37], target[:, :37], normalized=False)
        return (error/self.error_norm.std().clamp_min(self.error_floor)).square().mean(-1)

    def termination(self, state, context):
        return soft_termination(state, context)

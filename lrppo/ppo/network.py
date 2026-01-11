"""Actor-Critic network for continuous-action PPO.

This module implements a simple shared MLP trunk with separate actor and
critic heads. The actor outputs action means and a trainable log_std
parameter (diagonal Gaussian policy).
"""
from typing import Sequence, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(sizes: Sequence[int], activation: Callable = nn.Tanh) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Simple Actor-Critic network with Gaussian actions.

    Args:
        obs_dim: dimensionality of observation vector
        action_dim: dimensionality of action space
        hidden_sizes: sizes of hidden layers for the shared trunk
        activation: activation class to use between layers
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: Callable = nn.Tanh,
    ) -> None:
        super().__init__()
        sizes = (obs_dim, *hidden_sizes)
        self.trunk = _mlp(sizes, activation)

        # Actor head -> outputs mean for each action dim
        self.actor = nn.Linear(sizes[-1], action_dim)
        # Trainable log std parameter (initialized to small values)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

        # Critic head -> outputs scalar state value
        self.critic = nn.Linear(sizes[-1], 1)

    def forward(self, obs: torch.Tensor):
        """Forward pass returning action mean, log_std and state value.

        Args:
            obs: tensor of shape (B, obs_dim)
        Returns:
            mean: (B, action_dim)
            log_std: (action_dim,) (broadcastable)
            value: (B, 1)
        """
        x = self.trunk(obs)
        mean = self.actor(x)
        value = self.critic(x)
        return mean, self.log_std, value

    def get_dist(self, obs: torch.Tensor):
        """Return a Normal distribution for the policy at the given observations."""
        mean, log_std, _ = self.forward(obs)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean, std)

    def act(self, obs: torch.Tensor):
        """Sample action from policy and return (action, log_prob, value, entropy)."""
        mean, log_std, value = self.forward(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()  # reparameterized sample for backprop if needed
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return action, log_prob, value.squeeze(-1), entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate log_prob, entropy and state value for given actions."""
        mean, log_std, value = self.forward(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return log_prob, entropy, value.squeeze(-1)


__all__ = ["ActorCritic"]

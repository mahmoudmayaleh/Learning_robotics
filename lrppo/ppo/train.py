"""PPO trainer implementation (PyTorch, framework-agnostic).

This module implements a `PPOTrainer` that can be plugged into an environment
loop. It is independent of ROS and only depends on torch and numpy.
"""
from typing import Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import ActorCritic
from .buffer import RolloutBuffer


class PPOTrainer:
    """PPO trainer implementing clipped PPO with GAE.

    Args:
        obs_dim: observation dimensionality
        action_dim: action dimensionality (2 for linear and angular)
        action_low: per-dimension minimum action (scalar or array-like)
        action_high: per-dimension maximum action (scalar or array-like)
        device: torch device string (e.g., 'cpu' or 'cuda')
        lr: learning rate
        gamma: discount factor
        lam: GAE lambda
        clip_eps: PPO clip epsilon
        update_epochs: number of epochs per update
        minibatch_size: minibatch size
        value_coef: coefficient for value loss
        ent_coef: coefficient for entropy bonus
        max_grad_norm: gradient clipping norm
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low,
        action_high,
        device: str = "cpu",
        hidden_sizes: Tuple[int, ...] = (64, 64),
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        update_epochs: int = 10,
        minibatch_size: int = 64,
        value_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        rollout_capacity: int = 2048,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        self.action_low = np.array(action_low, dtype=np.float32)
        self.action_high = np.array(action_high, dtype=np.float32)

        self.gamma = float(gamma)
        self.lam = float(lam)
        self.clip_eps = float(clip_eps)
        self.update_epochs = int(update_epochs)
        self.minibatch_size = int(minibatch_size)
        self.value_coef = float(value_coef)
        self.ent_coef = float(ent_coef)
        self.max_grad_norm = float(max_grad_norm)

        self.model = ActorCritic(obs_dim, action_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.buffer = RolloutBuffer(obs_dim, action_dim, capacity=rollout_capacity, gamma=gamma, lam=lam)

        # Episode metrics storage (populated by TrainNode via record_episode)
        self.episode_metrics = []  # list of dicts with keys: episode_return, success, collisions, steps

    def record_episode(self, episode_return: float, success: bool, collisions: int, steps: int) -> None:
        """Record summary metrics for a finished episode.

        These are stored in-memory and can be exported by higher-level code or
        consumed by logging utilities (CSV, TensorBoard, etc.).
        """
        self.episode_metrics.append({
            'episode_return': float(episode_return),
            'success': bool(success),
            'collisions': int(collisions),
            'steps': int(steps),
        })

    def pop_episode_metrics(self):
        """Pop and return stored episode metrics and clear the buffer."""
        data = list(self.episode_metrics)
        self.episode_metrics.clear()
        return data
    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Select action given a single observation.

        Returns:
            action (clipped numpy array), log_prob (float), value (float)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_t, logp_t, value_t, _ = self.model.act(obs_t)
        action = action_t.cpu().numpy().squeeze(0)
        logp = logp_t.cpu().item()
        value = value_t.cpu().item()

        # Clip action to provided bounds for execution
        action_clipped = np.clip(action, self.action_low, self.action_high)
        return action_clipped, logp, value

    def store_transition(self, obs: np.ndarray, action: np.ndarray, logp: float, reward: float, done: bool, value: float) -> None:
        """Store a transition into the rollout buffer."""
        self.buffer.store(obs=obs, act=action, rew=reward, done=done, val=value, logp=logp)

    def finish_path(self, last_value: float = 0.0) -> None:
        """Signal the end of a trajectory (for GAE bootstrapping)."""
        self.buffer.finish_path(last_value=last_value)

    def update(self) -> Dict[str, float]:
        """Perform PPO update using data in the rollout buffer.

        Returns diagnostic info as a dict (losses, etc.).
        """
        data = self.buffer.get()
        obs = torch.as_tensor(data["obs"], dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(data["acts"], dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(data["logp"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(data["returns"], dtype=torch.float32, device=self.device)
        advs = torch.as_tensor(data["advs"], dtype=torch.float32, device=self.device)

        dataset_size = obs.shape[0]

        # Statistics for diagnostics
        epi_actor_loss = 0.0
        epi_value_loss = 0.0
        epi_entropy = 0.0
        n_updates = 0

        for epoch in range(self.update_epochs):
            # create random minibatches
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                mb_acts = acts[mb_idx]
                mb_old_logp = old_logp[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advs = advs[mb_idx]

                # Evaluate current policy
                new_logp, entropy, values_pred = self.model.evaluate_actions(mb_obs, mb_acts)

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advs
                actor_loss = -torch.mean(torch.min(surr1, surr2))

                value_loss = torch.mean((mb_returns - values_pred) ** 2)

                entropy_mean = torch.mean(entropy)

                loss = actor_loss + self.value_coef * value_loss - self.ent_coef * entropy_mean

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                epi_actor_loss += actor_loss.item()
                epi_value_loss += value_loss.item()
                epi_entropy += entropy_mean.item()
                n_updates += 1

        # Average metrics
        info = {
            "actor_loss": epi_actor_loss / max(1, n_updates),
            "value_loss": epi_value_loss / max(1, n_updates),
            "entropy": epi_entropy / max(1, n_updates),
        }
        return info


__all__ = ["PPOTrainer"]

"""A simple rollout buffer for PPO with GAE.

This buffer stores fixed-horizon trajectories and computes advantages using
Generalized Advantage Estimation (GAE-Lambda).
"""
from typing import Dict, Tuple

import numpy as np


class RolloutBuffer:
    """Fixed-size buffer used to store transitions for PPO updates.

    The buffer stores observations, actions, log probabilities, rewards, values
    and done flags. After a rollout (trajectory) finishes, call
    `finish_path(last_value)` to compute advantages and returns for that path.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        capacity: int,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.capacity = int(capacity)
        self.gamma = float(gamma)
        self.lam = float(lam)

        self.obs_buf = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(self.capacity, dtype=np.float32)
        self.done_buf = np.zeros(self.capacity, dtype=np.float32)
        self.val_buf = np.zeros(self.capacity, dtype=np.float32)
        self.logp_buf = np.zeros(self.capacity, dtype=np.float32)

        # buffers to be filled by finish_path
        self.adv_buf = np.zeros(self.capacity, dtype=np.float32)
        self.ret_buf = np.zeros(self.capacity, dtype=np.float32)

        self.ptr = 0
        self.path_start_idx = 0
        self.full = False

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        done: bool,
        val: float,
        logp: float,
    ) -> None:
        """Store one timestep of interaction data into the buffer."""
        if self.ptr >= self.capacity:
            raise IndexError("RolloutBuffer is full; call get() before storing more data")
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = float(rew)
        self.done_buf[self.ptr] = 1.0 if done else 0.0
        self.val_buf[self.ptr] = float(val)
        self.logp_buf[self.ptr] = float(logp)
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.full = True

    def finish_path(self, last_value: float = 0.0) -> None:
        """Compute GAE advantages and discounted returns for the current path.

        Should be called when an episode terminates or when a rollout is truncated
        (to bootstrap with `last_value`).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.rew_buf[path_slice]
        values = self.val_buf[path_slice]
        dones = self.done_buf[path_slice]

        # append last_value for bootstrapping
        values_ext = np.append(values, last_value)

        gae = 0.0
        adv = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values_ext[t + 1] * nonterminal - values_ext[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            adv[t] = gae

        self.adv_buf[path_slice] = adv
        self.ret_buf[path_slice] = adv + values

        self.path_start_idx = self.ptr

    def get(self) -> Dict[str, np.ndarray]:
        """Get all data from the buffer, normalize advantages, and reset buffer.
 
        Returns:
            A dict with keys: obs, acts, logp, returns, advs, vals
        """
        assert self.ptr > 0, "No data in buffer"

        # Only keep filled portion
        size = self.ptr
        obs = self.obs_buf[:size]
        acts = self.act_buf[:size]
        logp = self.logp_buf[:size]
        rets = self.ret_buf[:size]
        advs = self.adv_buf[:size]
        vals = self.val_buf[:size]

        # normalize advantages
        adv_mean = np.mean(advs)
        adv_std = np.std(advs) + 1e-8
        advs = (advs - adv_mean) / adv_std

        # Reset pointers
        self.ptr = 0
        self.path_start_idx = 0
        self.full = False

        return dict(obs=obs, acts=acts, logp=logp, returns=rets, advs=advs, vals=vals)


__all__ = ["RolloutBuffer"]

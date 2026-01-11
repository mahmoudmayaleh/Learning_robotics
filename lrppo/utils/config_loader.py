"""Config loader converting YAML into dataclasses for experiments."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    update_epochs: int = 10
    minibatch_size: int = 64
    value_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    rollout_capacity: int = 2048


@dataclass
class NetworkConfig:
    hidden_sizes: List[int] = (64, 64)


@dataclass
class RewardConfig:
    reward_goal: float = 100.0
    reward_collision: float = -50.0
    reward_timeout: float = -10.0
    progress_scale: float = 1.0


@dataclass
class EnvConfig:
    maze: str = 'default_maze'
    num_beams: int = 48
    lidar_max_range: float = 6.0
    control_rate: float = 10.0
    collision_distance: float = 0.15


@dataclass
class TrainingConfig:
    rollout_steps: int = 2048
    checkpoint_dir: str = 'checkpoints'
    checkpoint_interval: int = 1
    print_interval: int = 10


@dataclass
class ExperimentConfig:
    ppo: PPOConfig
    network: NetworkConfig
    reward: RewardConfig
    env: EnvConfig
    training: TrainingConfig


class ConfigLoader:
    """Loads YAML experiment config and converts to dataclass objects."""

    @staticmethod
    def load(path: str) -> ExperimentConfig:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'Config file not found: {p}')
        with p.open('r') as f:
            raw = yaml.safe_load(f) or {}

        ppo_raw = raw.get('ppo', {})
        for k in ['lr', 'gamma', 'lam', 'clip_eps', 'value_coef', 'ent_coef', 'max_grad_norm']:
            if k in ppo_raw:
                ppo_raw[k] = float(ppo_raw[k])
        for k in ['update_epochs', 'minibatch_size', 'rollout_capacity']:
            if k in ppo_raw:
                ppo_raw[k] = int(ppo_raw[k])

        ppo = PPOConfig(**ppo_raw)
        network = NetworkConfig(**(raw.get('network', {})))
        reward = RewardConfig(**(raw.get('reward', {})))
        env = EnvConfig(**(raw.get('env', {})))
        training = TrainingConfig(**(raw.get('training', {})))

        return ExperimentConfig(ppo=ppo, network=network, reward=reward, env=env, training=training)


__all__ = ["ConfigLoader", "ExperimentConfig", "PPOConfig", "NetworkConfig", "RewardConfig", "EnvConfig", "TrainingConfig"]
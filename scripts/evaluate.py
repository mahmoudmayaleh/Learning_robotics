#!/usr/bin/env python3
"""
Evaluate a trained PPO model on the navigation task.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lrppo.ppo.network import ActorCritic
from lrppo.envs.mock_maze import MockMazeEnv
from lrppo.utils.config_loader import ConfigLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='lrppo/config.yaml')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--deterministic', action='store_true', help='use mean action')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = ConfigLoader.load(args.config)

    env = MockMazeEnv(
        num_beams=cfg.env.num_beams,
        lidar_max_range=cfg.env.lidar_max_range,
        collision_distance=cfg.env.collision_distance,
        max_steps=200,
        seed=args.seed,
    )

    model = ActorCritic(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        hidden_sizes=tuple(cfg.network.hidden_sizes),
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.eval()

    returns = []
    successes = []
    collisions = []

    for ep in range(args.episodes):
        obs = env.reset()
        ep_return = 0.0

        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if args.deterministic:
                    mean, _, _ = model.forward(obs_t)
                    action = mean.squeeze(0).numpy()
                else:
                    action, _, _, _ = model.act(obs_t)
                    action = action.squeeze(0).numpy()

            action = np.clip(action, [-0.22, -2.84], [0.22, 2.84])
            obs, reward, done, info = env.step(action)
            ep_return += reward

            if done:
                break

        returns.append(ep_return)
        successes.append(int(info.get('reached_goal', False)))
        collisions.append(info.get('collisions', 0))

    print(f"\nEvaluation Results ({args.episodes} episodes)")
    print(f"  Success rate: {100*np.mean(successes):.1f}%")
    print(f"  Avg return:   {np.mean(returns):.2f} (+/- {np.std(returns):.2f})")
    print(f"  Avg collisions: {np.mean(collisions):.2f}")


if __name__ == '__main__':
    main()

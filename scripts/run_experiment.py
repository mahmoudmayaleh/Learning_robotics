#!/usr/bin/env python3
"""
PPO Training script for TurtleBot3 navigation.
Run with: python scripts/run_experiment.py --config lrppo/config.yaml
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lrppo.ppo.train import PPOTrainer
from lrppo.envs.mock_maze import MockMazeEnv
from lrppo.utils.config_loader import ConfigLoader
from lrppo.utils.logger import CSVLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='lrppo/config.yaml')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--render', action='store_true')
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

    trainer = PPOTrainer(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        action_low=np.array([-0.22, -2.84]),
        action_high=np.array([0.22, 2.84]),
        device='cpu',
        hidden_sizes=tuple(cfg.network.hidden_sizes),
        lr=cfg.ppo.lr,
        gamma=cfg.ppo.gamma,
        lam=cfg.ppo.lam,
        clip_eps=cfg.ppo.clip_eps,
        update_epochs=cfg.ppo.update_epochs,
        minibatch_size=cfg.ppo.minibatch_size,
        value_coef=cfg.ppo.value_coef,
        ent_coef=cfg.ppo.ent_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        rollout_capacity=cfg.ppo.rollout_capacity,
    )

    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    csv_path = os.path.join(cfg.training.checkpoint_dir, 'train_log.csv')
    logger = CSVLogger(csv_path, ['episode', 'return', 'success', 'collisions', 'steps'], 
                       print_interval=cfg.training.print_interval)

    episode = 0
    total_steps = 0
    update_count = 0
    start_time = time.time()

    print(f"Starting training for {args.episodes} episodes...")
    print(f"Checkpoints: {cfg.training.checkpoint_dir}")

    while episode < args.episodes:
        obs = env.reset()
        ep_return = 0.0
        ep_steps = 0

        while True:
            action, logp, value = trainer.select_action(obs)
            next_obs, reward, done, info = env.step(action)

            trainer.store_transition(obs, action, logp, reward, done, value)
            ep_return += reward
            ep_steps += 1
            total_steps += 1
            obs = next_obs

            if done:
                trainer.finish_path(last_value=0.0)
                break

            if trainer.buffer.full:
                _, _, last_val = trainer.select_action(obs)
                trainer.finish_path(last_value=last_val)
                
                info_update = trainer.update()
                update_count += 1
                
                if update_count % cfg.training.checkpoint_interval == 0:
                    ckpt_path = os.path.join(cfg.training.checkpoint_dir, f'model_{update_count}.pt')
                    torch.save(trainer.model.state_dict(), ckpt_path)

        logger.log_episode({
            'episode': episode,
            'return': ep_return,
            'success': int(info.get('reached_goal', False)),
            'collisions': info.get('collisions', 0),
            'steps': ep_steps,
        })
        episode += 1

        if trainer.buffer.ptr > 0 and trainer.buffer.full:
            info_update = trainer.update()
            update_count += 1

    # final checkpoint
    final_path = os.path.join(cfg.training.checkpoint_dir, 'model_final.pt')
    torch.save(trainer.model.state_dict(), final_path)

    elapsed = time.time() - start_time
    print(f"\nTraining complete: {episode} episodes, {total_steps} steps, {elapsed:.1f}s")
    print(f"Final model saved to {final_path}")
    logger.close()


if __name__ == '__main__':
    main()

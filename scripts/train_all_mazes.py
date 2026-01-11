#!/usr/bin/env python3
"""Train PPO on all 3 maze difficulties and generate logs/checkpoints."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from lrppo.ppo.train import PPOTrainer
from lrppo.envs.mock_maze import MockMazeEnv
from lrppo.utils.config_loader import ConfigLoader
from lrppo.utils.logger import CSVLogger


def train_maze(difficulty, episodes, cfg, output_dir):
    print(f"\n{'='*50}")
    print(f"Training on {difficulty} maze for {episodes} episodes")
    print(f"{'='*50}")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    env = MockMazeEnv(
        num_beams=cfg.env.num_beams,
        lidar_max_range=cfg.env.lidar_max_range,
        collision_distance=cfg.env.collision_distance,
        max_steps=200,
        seed=42,
        difficulty=difficulty,
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
        rollout_capacity=cfg.ppo.rollout_capacity,
    )

    maze_dir = os.path.join(output_dir, f'{difficulty}_maze')
    os.makedirs(maze_dir, exist_ok=True)
    
    csv_path = os.path.join(maze_dir, 'train_log.csv')
    logger = CSVLogger(csv_path, ['episode', 'return', 'success', 'collisions', 'steps'], print_interval=50)

    episode = 0
    while episode < episodes:
        obs = env.reset()
        ep_return = 0.0
        ep_steps = 0

        while True:
            action, logp, value = trainer.select_action(obs)
            next_obs, reward, done, info = env.step(action)

            trainer.store_transition(obs, action, logp, reward, done, value)
            ep_return += reward
            ep_steps += 1
            obs = next_obs

            if trainer.buffer.full:
                if not done:
                    _, _, last_val = trainer.select_action(obs)
                    trainer.finish_path(last_value=last_val)
                trainer.update()

            if done:
                trainer.finish_path(last_value=0.0)
                break

        logger.log_episode({
            'episode': episode,
            'return': ep_return,
            'success': int(info.get('reached_goal', False)),
            'collisions': info.get('collisions', 0),
            'steps': ep_steps,
        })
        episode += 1

        if episode % 100 == 0:
            ckpt = os.path.join(maze_dir, f'model_ep{episode}.pt')
            torch.save(trainer.model.state_dict(), ckpt)

    final_path = os.path.join(maze_dir, 'model_final.pt')
    torch.save(trainer.model.state_dict(), final_path)
    logger.close()
    
    print(f"Saved model to {final_path}")
    print(f"Saved logs to {csv_path}")
    return csv_path


def main():
    cfg = ConfigLoader.load('lrppo/config.yaml')
    output_dir = 'results'
    episodes = 300
    
    log_files = []
    for difficulty in ['simple', 'medium', 'complex']:
        log_path = train_maze(difficulty, episodes, cfg, output_dir)
        log_files.append(log_path)
    
    print("\n" + "="*50)
    print("Training complete! Generating plots...")
    print("="*50)
    
    # Generate plots
    import subprocess
    subprocess.run([
        sys.executable, 'scripts/plot_results.py',
        '--logs'] + log_files + [
        '--labels', 'Simple', 'Medium', 'Complex',
        '--output', os.path.join(output_dir, 'plots')
    ])


if __name__ == '__main__':
    main()

"""Integration tests for PPOTrainer using MockMazeEnv."""
import numpy as np

from lrppo.ppo.train import PPOTrainer
from lrppo.envs.mock_maze import MockMazeEnv


def test_trainer_rollout_and_update():
    """Test that PPOTrainer can collect a rollout and perform an update without errors."""
    env = MockMazeEnv(seed=42, max_steps=50)
    obs_dim = env.observation_dim
    action_dim = env.action_dim

    trainer = PPOTrainer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=np.array([-0.22, -2.84]),
        action_high=np.array([0.22, 2.84]),
        device='cpu',
        rollout_capacity=128,
        update_epochs=2,
        minibatch_size=32,
    )

    obs = env.reset()
    episode_return = 0.0
    steps = 0

    # Collect transitions until buffer is full or we finish some episodes
    while not trainer.buffer.full and steps < 200:
        action, logp, value = trainer.select_action(obs)
        next_obs, reward, done, info = env.step(action)

        trainer.store_transition(obs, action, logp, reward, done, value)
        episode_return += reward
        steps += 1
        obs = next_obs

        if done:
            trainer.finish_path(last_value=0.0)
            trainer.record_episode(episode_return, info['reached_goal'], info['collisions'], steps)
            obs = env.reset()
            episode_return = 0.0

    # Bootstrap if not terminal
    if not done:
        _, _, last_val = trainer.select_action(obs)
        trainer.finish_path(last_value=last_val)

    # Run PPO update
    info = trainer.update()

    # Verify update returns sensible diagnostics
    assert 'actor_loss' in info
    assert 'value_loss' in info
    assert 'entropy' in info
    assert np.isfinite(info['actor_loss'])
    assert np.isfinite(info['value_loss'])
    assert np.isfinite(info['entropy'])

    # Check that episode metrics were recorded
    metrics = trainer.pop_episode_metrics()
    assert len(metrics) >= 1  # at least one episode should have finished
    for m in metrics:
        assert 'episode_return' in m
        assert 'success' in m
        assert 'collisions' in m


def test_trainer_multiple_updates():
    """Test running multiple rollout+update cycles."""
    env = MockMazeEnv(seed=123, max_steps=30)
    trainer = PPOTrainer(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        action_low=np.array([-0.22, -2.84]),
        action_high=np.array([0.22, 2.84]),
        device='cpu',
        rollout_capacity=64,
        update_epochs=2,
        minibatch_size=16,
    )

    for _ in range(3):  # 3 update cycles
        obs = env.reset()
        while not trainer.buffer.full:
            action, logp, value = trainer.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            trainer.store_transition(obs, action, logp, reward, done, value)
            obs = next_obs
            if done:
                trainer.finish_path(last_value=0.0)
                obs = env.reset()

        if not done:
            _, _, last_val = trainer.select_action(obs)
            trainer.finish_path(last_value=last_val)

        info = trainer.update()
        assert np.isfinite(info['actor_loss'])

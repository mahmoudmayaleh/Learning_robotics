"""Mock maze environment for testing PPOTrainer without ROS/Gazebo.

This is a simple, deterministic environment that mimics the TurtleBot3 navigation
task interface: returns observations (fake LiDAR + velocity + goal), accepts
2D actions (linear_vel, angular_vel), and computes reward/done using MazeReward.
"""
import numpy as np

from lrppo.utils.observation import ObservationBuilder
from lrppo.utils.reward import MazeReward


class MockMazeEnv:
    """Minimal mock environment for PPO testing.

    The robot starts at a random position and must reach a goal. LiDAR is
    simulated as random ranges. Collision is triggered if any beam < threshold.
    """

    def __init__(
        self,
        num_beams: int = 48,
        lidar_max_range: float = 6.0,
        collision_distance: float = 0.15,
        goal_threshold: float = 0.2,
        max_steps: int = 200,
        seed: int = None,
        difficulty: str = 'simple',
    ):
        self.num_beams = num_beams
        self.lidar_max_range = lidar_max_range
        self.collision_distance = collision_distance
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps
        self.difficulty = difficulty

        self.obs_builder = ObservationBuilder(num_beams=num_beams, max_range=lidar_max_range)
        self.rewarder = MazeReward()

        self.rng = np.random.default_rng(seed)

        # difficulty affects obstacle density and goal distance
        self.obstacle_prob = {'simple': 0.05, 'medium': 0.12, 'complex': 0.20}.get(difficulty, 0.1)
        self.goal_range = {'simple': (1.0, 4.0), 'medium': (2.0, 6.0), 'complex': (3.0, 8.0)}.get(difficulty, (1.0, 5.0))

        self.goal_distance = 0.0
        self.goal_angle = 0.0
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.steps = 0
        self.collisions = 0

    @property
    def observation_dim(self) -> int:
        return self.obs_builder.observation_dim

    @property
    def action_dim(self) -> int:
        return 2

    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        self.goal_distance = self.rng.uniform(*self.goal_range)
        self.goal_angle = self.rng.uniform(-np.pi, np.pi)
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.steps = 0
        self.collisions = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        lidar = self.rng.uniform(0.5, self.lidar_max_range, size=360).astype(np.float32)
        if self.rng.random() < self.obstacle_prob:
            n_close = self.rng.integers(1, 5)
            idxs = self.rng.integers(0, 360, size=n_close)
            lidar[idxs] = self.rng.uniform(0.05, 0.3, size=n_close)
        return self.obs_builder.build_observation(
            lidar, self.lin_vel, self.ang_vel, self.goal_distance, self.goal_angle
        )

    def step(self, action: np.ndarray):
        """Take action and return (obs, reward, done, info)."""
        self.steps += 1
        lin_vel, ang_vel = float(action[0]), float(action[1])
        self.lin_vel = lin_vel
        self.ang_vel = ang_vel

        prev_dist = self.goal_distance

        # Simulate movement: reduce goal distance proportional to forward velocity
        self.goal_distance -= lin_vel * 0.1
        self.goal_distance = max(0.0, self.goal_distance)

        # Rotate goal angle based on angular velocity
        self.goal_angle -= ang_vel * 0.1
        self.goal_angle = (self.goal_angle + np.pi) % (2 * np.pi) - np.pi

        # Check termination conditions
        lidar = self.rng.uniform(0.5, self.lidar_max_range, size=360).astype(np.float32)
        collided = bool(np.any(lidar < self.collision_distance))
        if collided:
            self.collisions += 1

        reached_goal = self.goal_distance < self.goal_threshold
        timed_out = self.steps >= self.max_steps

        reward = self.rewarder.compute_reward(
            prev_dist, self.goal_distance, collided, reached_goal, timed_out, step_penalty=0.01
        )
        done = self.rewarder.is_done(collided, reached_goal, timed_out)

        obs = self._get_obs()
        info = {
            'collisions': self.collisions,
            'reached_goal': reached_goal,
            'timed_out': timed_out,
        }
        return obs, reward, done, info


__all__ = ["MockMazeEnv"]

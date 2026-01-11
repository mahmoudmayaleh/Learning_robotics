"""Reward function utilities for maze/TurtleBot3 navigation tasks.

Provides a simple, unit-testable `MazeReward` class that computes rewards based
on progress, collisions, timeouts, and reaching the goal.

All logic is framework-agnostic (no ROS imports).
"""
from typing import Final

REWARD_GOAL: Final[float] = 100.0
REWARD_COLLISION: Final[float] = -50.0
REWARD_TIMEOUT: Final[float] = -10.0
PROGRESS_REWARD_SCALE: Final[float] = 1.0


class MazeReward:
    """Compute rewards for maze navigation tasks.

    Reward rules:
      - If `reached_goal` is True -> `reward_goal` (big positive reward)
      - Else if `collided` is True -> `reward_collision` (large negative)
      - Else if `timed_out` is True -> `reward_timeout` (moderate negative)
      - Otherwise -> progress reward proportional to (prev_distance - curr_distance)
        and minus a small per-step penalty.

    The reward constants can be overridden via constructor parameters for easy
    tuning from config files.
    """

    def __init__(
        self,
        reward_goal: float = REWARD_GOAL,
        reward_collision: float = REWARD_COLLISION,
        reward_timeout: float = REWARD_TIMEOUT,
        progress_scale: float = PROGRESS_REWARD_SCALE,
    ) -> None:
        self.reward_goal = float(reward_goal)
        self.reward_collision = float(reward_collision)
        self.reward_timeout = float(reward_timeout)
        self.progress_scale = float(progress_scale)

    def compute_reward(
        self,
        prev_distance: float,
        curr_distance: float,
        collided: bool,
        reached_goal: bool,
        timed_out: bool,
        step_penalty: float = 0.0,
    ) -> float:
        """Return the reward for a single timestep.

        Args:
            prev_distance: Euclidean distance to the goal at previous timestep.
            curr_distance: Euclidean distance to the goal at current timestep.
            collided: Whether a collision occurred at this timestep.
            reached_goal: Whether the goal was reached at this timestep.
            timed_out: Whether the episode timed out at this timestep.
            step_penalty: Small (usually positive) value subtracted each step.

        Returns:
            A floating-point scalar reward.
        """
        if reached_goal:
            return float(self.reward_goal)

        if collided:
            return float(self.reward_collision)

        if timed_out:
            return float(self.reward_timeout)

        # Progress reward: positive if robot moved closer to the goal
        progress = prev_distance - curr_distance
        reward = self.progress_scale * progress

        # Subtract per-step penalty (commonly small positive number)
        reward -= float(step_penalty)

        return float(reward)

    def is_done(self, collided: bool, reached_goal: bool, timed_out: bool) -> bool:
        """Return True if the episode should terminate.

        Episodes end on collision, reaching the goal, or timeout.
        """
        return bool(collided or reached_goal or timed_out)

__all__ = ["MazeReward", "REWARD_GOAL", "REWARD_COLLISION", "REWARD_TIMEOUT", "PROGRESS_REWARD_SCALE"]

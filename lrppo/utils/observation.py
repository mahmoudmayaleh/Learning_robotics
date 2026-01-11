"""Observation builder for RL training (framework-agnostic).

Builds a fixed-size observation vector from raw LiDAR ranges and scalar info
(lin/ang velocities and relative goal info).
"""
from typing import Optional

import numpy as np


class ObservationBuilder:
    """Build a fixed-size observation vector.

    Args:
        num_beams: Number of LiDAR beams to downsample to (default: 48).
        max_range: Maximum LiDAR range used for clipping and normalization.
        min_range: Minimum valid LiDAR range (default: 0.0).
    """

    def __init__(self, num_beams: int = 48, max_range: float = 10.0, min_range: float = 0.0) -> None:
        if num_beams <= 0:
            raise ValueError("num_beams must be positive")
        if max_range <= min_range:
            raise ValueError("max_range must be greater than min_range")

        self.num_beams = int(num_beams)
        self.max_range = float(max_range)
        self.min_range = float(min_range)

    @property
    def observation_dim(self) -> int:
        """Return the dimensionality of the produced observation vector.

        It equals num_beams + 4 (lin_vel, ang_vel, goal_distance, goal_angle).
        """
        return self.num_beams + 4

    def build_observation(
        self,
        lidar_ranges: np.ndarray,
        lin_vel: float,
        ang_vel: float,
        goal_distance: float,
        goal_angle: float,
    ) -> np.ndarray:
        """Build and return a 1D float32 observation vector.

        Args:
            lidar_ranges: 1-D array of raw LiDAR distances (any length).
            lin_vel: Robot linear velocity (m/s).
            ang_vel: Robot angular velocity (rad/s).
            goal_distance: Relative distance to the goal (m).
            goal_angle: Relative angle to the goal (rad).

        Returns:
            1-D numpy array of dtype float32: [downsampled_lidar (normalized),
            lin_vel, ang_vel, goal_distance, goal_angle]

        Notes:
            - LiDAR is down-sampled using uniform indexing (np.linspace).
            - LiDAR ranges are clipped to [min_range, max_range] and normalized to [0, 1].
            - No framework-specific imports (e.g., ROS) are used here.
        """
        # Validate and coerce inputs
        lidar = np.asarray(lidar_ranges)
        if lidar.ndim != 1:
            raise ValueError("lidar_ranges must be a 1-D array")
        if lidar.size == 0:
            raise ValueError("lidar_ranges must not be empty")

        # Replace NaN/inf with max_range to indicate out-of-range / missing returns
        lidar = np.nan_to_num(lidar, nan=self.max_range, posinf=self.max_range, neginf=self.min_range)

        # Down-sample to fixed number of beams using uniform indices
        n = lidar.size
        indices = np.linspace(0, n - 1, num=self.num_beams)
        indices = np.round(indices).astype(int)
        sampled = lidar[indices]

        # Clip and normalize ranges to [0, 1]
        clipped = np.clip(sampled, self.min_range, self.max_range)
        denom = self.max_range - self.min_range
        if denom <= 0:
            # Should not happen because we validate in constructor
            raise ValueError("Invalid range normalization denominator")
        normalized = (clipped - self.min_range) / denom

        # Create final observation vector and cast to float32
        tail = np.array([lin_vel, ang_vel, goal_distance, goal_angle], dtype=np.float32)
        obs = np.concatenate([normalized.astype(np.float32), tail])

        return obs


# Module exports
__all__ = ["ObservationBuilder"]

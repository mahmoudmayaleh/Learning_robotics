import numpy as np
import pytest

from lrppo.utils.observation import ObservationBuilder


def test_downsample_shape_dtype():
    lidar = np.linspace(0.0, 10.0, 360)
    b = ObservationBuilder(num_beams=48, max_range=10.0)
    obs = b.build_observation(lidar, lin_vel=0.1, ang_vel=0.2, goal_distance=1.0, goal_angle=0.0)

    # Expect num_beams + 4 (lin_vel, ang_vel, goal_distance, goal_angle)
    assert obs.shape == (48 + 4,)
    assert obs.dtype == np.float32

    # Check that the lidar part matches the downsampled, clipped and normalized values
    indices = np.round(np.linspace(0, len(lidar) - 1, 48)).astype(int)
    expected = lidar[indices]
    expected = np.clip(expected, 0.0, 10.0)
    expected = (expected - 0.0) / 10.0

    np.testing.assert_allclose(obs[:48], expected.astype(np.float32), rtol=1e-6, atol=1e-6)

    # Tail should equal the passed scalars
    np.testing.assert_allclose(obs[48:], np.array([0.1, 0.2, 1.0, 0.0], dtype=np.float32))


def test_clipping_and_nan_inf():
    # Fill with large values; special values will be mapped to max/min
    lidar = np.full(360, 20.0, dtype=float)
    lidar[5] = np.nan
    lidar[10] = np.inf
    lidar[15] = -np.inf

    b = ObservationBuilder(num_beams=48, max_range=10.0, min_range=0.0)
    obs = b.build_observation(lidar, 0.0, 0.0, 0.0, 0.0)

    indices = np.round(np.linspace(0, len(lidar) - 1, 48)).astype(int)
    sampled = lidar[indices]
    sampled = np.nan_to_num(sampled, nan=10.0, posinf=10.0, neginf=0.0)
    clipped = np.clip(sampled, 0.0, 10.0)
    normalized = (clipped - 0.0) / 10.0

    np.testing.assert_allclose(obs[:48], normalized.astype(np.float32), rtol=1e-6, atol=1e-6)


def test_invalid_inputs():
    b = ObservationBuilder()
    # Empty lidar
    with pytest.raises(ValueError):
        b.build_observation(np.array([]), 0.0, 0.0, 0.0, 0.0)

    # Not 1-D lidar
    with pytest.raises(ValueError):
        b.build_observation(np.zeros((3, 3)), 0.0, 0.0, 0.0, 0.0)

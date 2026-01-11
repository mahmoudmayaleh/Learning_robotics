import pytest

from lrppo.utils.reward import (
    MazeReward,
    REWARD_GOAL,
    REWARD_COLLISION,
    REWARD_TIMEOUT,
)


def test_reached_goal_reward():
    r = MazeReward()
    val = r.compute_reward(5.0, 0.2, collided=False, reached_goal=True, timed_out=False)
    assert val == REWARD_GOAL


def test_collision_reward():
    r = MazeReward()
    val = r.compute_reward(2.0, 1.8, collided=True, reached_goal=False, timed_out=False)
    assert val == REWARD_COLLISION


def test_timeout_reward():
    r = MazeReward()
    val = r.compute_reward(1.0, 1.0, collided=False, reached_goal=False, timed_out=True)
    assert val == REWARD_TIMEOUT


def test_progress_and_step_penalty():
    r = MazeReward()
    prev = 5.0
    curr = 4.2
    # progress = 0.8, step_penalty = 0.1 -> reward = 0.7
    val = r.compute_reward(prev, curr, collided=False, reached_goal=False, timed_out=False, step_penalty=0.1)
    assert pytest.approx(val, rel=1e-6) == 0.7


def test_is_done():
    r = MazeReward()
    assert r.is_done(collided=True, reached_goal=False, timed_out=False)
    assert r.is_done(collided=False, reached_goal=True, timed_out=False)
    assert r.is_done(collided=False, reached_goal=False, timed_out=True)
    assert not r.is_done(collided=False, reached_goal=False, timed_out=False)

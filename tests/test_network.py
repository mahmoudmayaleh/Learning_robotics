import numpy as np
import torch

from lrppo.ppo.network import ActorCritic


def test_forward_shapes():
    obs_dim = 52
    action_dim = 2
    batch_size = 8

    model = ActorCritic(obs_dim, action_dim, hidden_sizes=(64, 64))
    obs = torch.randn(batch_size, obs_dim)

    mean, log_std, value = model.forward(obs)

    assert mean.shape == (batch_size, action_dim)
    assert log_std.shape == (action_dim,)
    assert value.shape == (batch_size, 1)


def test_act_returns_correct_shapes():
    obs_dim = 10
    action_dim = 2
    model = ActorCritic(obs_dim, action_dim)

    obs = torch.randn(1, obs_dim)
    action, log_prob, value, entropy = model.act(obs)

    assert action.shape == (1, action_dim)
    assert log_prob.shape == (1,)
    assert value.shape == (1,)
    assert entropy.shape == (1,)


def test_evaluate_actions():
    obs_dim = 10
    action_dim = 2
    batch_size = 4
    model = ActorCritic(obs_dim, action_dim)

    obs = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, action_dim)

    log_prob, entropy, values = model.evaluate_actions(obs, actions)

    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert values.shape == (batch_size,)

    # log_prob and entropy should be finite
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(values).all()


def test_get_dist():
    obs_dim = 5
    action_dim = 2
    model = ActorCritic(obs_dim, action_dim)

    obs = torch.randn(3, obs_dim)
    dist = model.get_dist(obs)

    # Should be a Normal distribution with correct batch shape
    assert dist.batch_shape == (3, action_dim)
    sample = dist.sample()
    assert sample.shape == (3, action_dim)

import numpy as np

from lrppo.ppo.buffer import RolloutBuffer


def compute_gae_numpy(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    # replicates the GAE loop from RolloutBuffer.finish_path
    values_ext = np.append(values, last_value)
    adv = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values_ext[t + 1] * nonterminal - values_ext[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


def test_rolloutbuffer_gae_and_get():
    obs_dim = 3
    act_dim = 2
    capacity = 10
    gamma = 0.99
    lam = 0.95

    buf = RolloutBuffer(obs_dim, act_dim, capacity=capacity, gamma=gamma, lam=lam)

    # Create short trajectory
    rewards = np.array([1.0, 0.0, -1.0, 2.0], dtype=np.float32)
    values = np.array([0.5, 0.6, 0.4, 0.2], dtype=np.float32)
    dones = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # last step terminates episode

    n = len(rewards)
    for i in range(n):
        obs = np.zeros(obs_dim, dtype=np.float32) + i
        act = np.zeros(act_dim, dtype=np.float32) + i
        rew = float(rewards[i])
        done = bool(dones[i])
        val = float(values[i])
        logp = -0.1 * i
        buf.store(obs, act, rew, done, val, logp)

    # Finish path (episode terminated so last_val=0)
    buf.finish_path(last_value=0.0)

    data = buf.get()

    # Compute expected raw advantages and returns
    raw_adv, expected_returns = compute_gae_numpy(rewards, values, dones, last_value=0.0, gamma=gamma, lam=lam)

    # Normalize expected advantages
    adv_mean = np.mean(raw_adv)
    adv_std = np.std(raw_adv) + 1e-8
    expected_advs = (raw_adv - adv_mean) / adv_std

    # Compare returned values
    np.testing.assert_allclose(data['advs'], expected_advs.astype(np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(data['returns'], expected_returns.astype(np.float32), rtol=1e-6, atol=1e-6)

    # Buffer should have been reset
    assert buf.ptr == 0
    assert buf.full is False


def test_rolloutbuffer_bootstrapping_with_last_value():
    # Test truncated rollout that should bootstrap using last_value
    obs_dim = 2
    act_dim = 1
    capacity = 10
    gamma = 0.9
    lam = 0.8

    buf = RolloutBuffer(obs_dim, act_dim, capacity=capacity, gamma=gamma, lam=lam)

    rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    dones = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # truncated

    for i in range(len(rewards)):
        buf.store(np.zeros(obs_dim), np.zeros(act_dim), float(rewards[i]), bool(dones[i]), float(values[i]), -0.1)

    last_value = 0.5
    buf.finish_path(last_value=last_value)

    data = buf.get()

    raw_adv, expected_returns = compute_gae_numpy(rewards, values, dones, last_value=last_value, gamma=gamma, lam=lam)
    adv_mean = np.mean(raw_adv)
    adv_std = np.std(raw_adv) + 1e-8
    expected_advs = (raw_adv - adv_mean) / adv_std

    np.testing.assert_allclose(data['advs'], expected_advs.astype(np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(data['returns'], expected_returns.astype(np.float32), rtol=1e-6, atol=1e-6)

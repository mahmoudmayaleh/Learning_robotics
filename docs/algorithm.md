# PPO Algorithm for Robot Navigation

## Overview

This project implements Proximal Policy Optimization (PPO) for training a TurtleBot3 robot to navigate through maze environments. PPO is a policy gradient method that provides stable and reliable training through a clipped surrogate objective.

## Algorithm Description

### Policy Gradient Foundation

PPO builds on the policy gradient theorem. The goal is to maximize expected cumulative reward:

```
J(θ) = E_τ~π_θ [Σ_t γ^t r_t]
```

The policy gradient is:

```
∇J(θ) = E [∇_θ log π_θ(a|s) · A(s,a)]
```

where A(s,a) is the advantage function estimating how much better action `a` is compared to the average action at state `s`.

### PPO Clipped Objective

Standard policy gradients can take large, destructive steps. PPO constrains updates using a clipped surrogate objective:

```
L^CLIP(θ) = E [min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)]
```

where:

- `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` is the probability ratio
- `ε` is the clip range (typically 0.2)

This prevents the policy from changing too drastically in a single update.

### Generalized Advantage Estimation (GAE)

We use GAE-λ for advantage estimation:

```
A_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
```

where `δ_t = r_t + γV(s_{t+1}) - V(s_t)` is the TD residual.

GAE interpolates between:

- λ=0: High bias, low variance (TD(0))
- λ=1: Low bias, high variance (Monte Carlo)

We use λ=0.95 for a balance between bias and variance.

### Actor-Critic Architecture

Our implementation uses a shared-trunk actor-critic network:

```
         Observation (52-dim)
              |
        [MLP Trunk: 64-64]
           /        \
      Actor Head    Critic Head
         |              |
   μ (action mean)    V(s)
         |
      log_std (learned)
```

The actor outputs parameters of a diagonal Gaussian distribution for continuous actions (linear velocity, angular velocity).

## Implementation Details

### Observation Space (52 dimensions)

- 48 downsampled LiDAR beams (normalized to [0,1])
- Linear velocity
- Angular velocity
- Goal distance
- Goal bearing angle

### Action Space (2 dimensions)

- Linear velocity: [-0.22, 0.22] m/s
- Angular velocity: [-2.84, 2.84] rad/s

### Reward Function

```python
if reached_goal:    return +100
if collided:        return -50
if timed_out:       return -10
else:               return (prev_dist - curr_dist) - step_penalty
```

### Hyperparameters

| Parameter      | Value | Description         |
| -------------- | ----- | ------------------- |
| γ (gamma)      | 0.99  | Discount factor     |
| λ (lambda)     | 0.95  | GAE parameter       |
| ε (clip)       | 0.2   | PPO clip range      |
| Learning rate  | 3e-4  | Adam optimizer      |
| Minibatch size | 64    | SGD batch size      |
| Update epochs  | 10    | Epochs per update   |
| Rollout steps  | 2048  | Steps before update |

## Training Loop

```
1. Collect rollout of 2048 steps using current policy
2. Compute advantages using GAE
3. For each of 10 epochs:
   - Shuffle data into minibatches of 64
   - Compute clipped PPO loss
   - Update network via Adam
4. Save checkpoint periodically
5. Repeat
```

## References

1. Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017)
2. Schulman, J., et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation." ICLR (2016)

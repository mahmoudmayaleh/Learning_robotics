# System Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         ROS 2 System                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │   Gazebo     │     │  TrainNode   │     │    PPO       │   │
│  │  Simulator   │────▶│   (ROS 2)    │────▶│   Trainer    │   │
│  │              │     │              │     │              │   │
│  │  - TurtleBot │◀────│  - Sensors   │◀────│  - Network   │   │
│  │  - Maze      │     │  - Control   │     │  - Buffer    │   │
│  │  - Physics   │     │  - Logging   │     │  - Update    │   │
│  └──────────────┘     └──────────────┘     └──────────────┘   │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │   /scan      │     │  Observation │     │   Model      │   │
│  │   /odom      │────▶│   Builder    │────▶│  Checkpoint  │   │
│  │   /cmd_vel   │     │              │     │   (.pt)      │   │
│  └──────────────┘     └──────────────┘     └──────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Gazebo Simulation

- Provides physics simulation for TurtleBot3
- Three maze environments: simple, medium, complex
- Publishes sensor data (LiDAR, odometry)
- Receives velocity commands

### 2. ROS 2 TrainNode

- Subscribes to `/scan` (LiDAR), `/odom` (odometry), `/goal` (target)
- Publishes to `/cmd_vel` (velocity commands)
- Manages episode lifecycle (reset, termination)
- Logs metrics to CSV

### 3. PPO Trainer

- Actor-Critic neural network (PyTorch)
- Rollout buffer with GAE computation
- Clipped PPO policy updates
- Model checkpointing

### 4. Observation Builder

- Downsamples 360 LiDAR beams to 48
- Normalizes ranges to [0, 1]
- Concatenates state vector: [lidar, vel, goal]

### 5. Reward Calculator

- Goal reached: +100
- Collision: -50
- Timeout: -10
- Progress: distance reduction bonus

## Data Flow

```
Sensors ──▶ ObservationBuilder ──▶ PPO Network ──▶ Action
   │                                                  │
   │              ┌──────────────────────────────────┘
   │              ▼
   │         cmd_vel ──▶ Gazebo ──▶ New State
   │              │
   │              ▼
   │         MazeReward ──▶ Reward, Done
   │              │
   │              ▼
   └─────── RolloutBuffer ──▶ PPO Update ──▶ Updated Policy
```

## File Structure

```
lrppo/
├── ros/
│   ├── train_node.py      # Training loop node
│   └── inference_node.py  # Evaluation node
├── ppo/
│   ├── network.py         # Actor-Critic network
│   ├── buffer.py          # Rollout buffer + GAE
│   └── train.py           # PPO trainer class
├── utils/
│   ├── observation.py     # Observation processing
│   ├── reward.py          # Reward calculation
│   ├── config_loader.py   # YAML config parser
│   └── logger.py          # CSV logging
├── envs/
│   └── mock_maze.py       # Mock env for testing
├── worlds/
│   ├── simple_maze.world
│   ├── medium_maze.world
│   └── complex_maze.world
├── launch/
│   ├── train_simple.launch.py
│   ├── train_medium.launch.py
│   ├── train_complex.launch.py
│   └── evaluate.launch.py
└── config.yaml
```

## Network Architecture

```
Input: 52-dim observation
    │
    ▼
┌─────────────┐
│ Linear(52→64)│
│    Tanh     │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Linear(64→64)│
│    Tanh     │
└─────────────┘
    │
    ├─────────────────┐
    ▼                 ▼
┌─────────────┐  ┌─────────────┐
│ Actor Head  │  │ Critic Head │
│ Linear(64→2)│  │ Linear(64→1)│
└─────────────┘  └─────────────┘
    │                 │
    ▼                 ▼
  μ (mean)         V(s)
    │
    ▼
┌─────────────┐
│ log_std     │  (learned parameter)
└─────────────┘
    │
    ▼
 N(μ, exp(log_std))  →  Action sample
```

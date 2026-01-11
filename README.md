# LRPPO - PPO Navigation for TurtleBot3

A ROS 2 package implementing Proximal Policy Optimization (PPO) for autonomous navigation of TurtleBot3 in maze environments.

## Authors

Group 4 - Learning Robotics Project

## Project Structure

```
lrppo/
├── ros/                 # ROS 2 nodes
│   ├── train_node.py    # Training node
│   └── inference_node.py# Evaluation node
├── ppo/                 # PPO implementation
│   ├── network.py       # Actor-Critic network
│   ├── buffer.py        # Rollout buffer + GAE
│   └── train.py         # PPO trainer
├── utils/               # Utilities
│   ├── observation.py   # Observation builder
│   ├── reward.py        # Reward function
│   ├── config_loader.py # Config parser
│   └── logger.py        # CSV logging
├── envs/                # Environments
│   └── mock_maze.py     # Mock env for testing
├── worlds/              # Gazebo worlds
│   ├── simple_maze.world
│   ├── medium_maze.world
│   └── complex_maze.world
├── launch/              # Launch files
└── config.yaml          # Hyperparameters
scripts/                 # Training & evaluation scripts
docs/                    # Documentation
results/                 # Training logs & plots
tests/                   # Unit tests
```

## Installation

### Prerequisites

- ROS 2 Humble
- Gazebo 11
- TurtleBot3 packages
- Python 3.10+

### Setup

```bash
# Clone into your ROS 2 workspace
cd ~/ros2_ws/src
git clone <repo_url> lrppo

# Install Python dependencies
pip install torch numpy pyyaml pandas matplotlib

# Build
cd ~/ros2_ws
colcon build --packages-select lrppo
source install/setup.bash

# Set TurtleBot3 model
export TURTLEBOT3_MODEL=burger
```

## Quick Start

### Run without ROS (Mock Environment)

```bash
# Train on all mazes
python scripts/train_all_mazes.py

# Train single experiment
python scripts/run_experiment.py --config lrppo/config.yaml --episodes 500

# Evaluate trained model
python scripts/evaluate.py --checkpoint results/simple_maze/model_final.pt --episodes 100
```

### Run with Gazebo + ROS 2

```bash
# Training on simple maze
ros2 launch lrppo train_simple.launch.py

# Training on medium maze
ros2 launch lrppo train_medium.launch.py

# Training on complex maze
ros2 launch lrppo train_complex.launch.py

# Evaluation
ros2 launch lrppo evaluate.launch.py world:=simple_maze checkpoint:=/path/to/model.pt
```

## Configuration

Edit `lrppo/config.yaml` to tune hyperparameters:

```yaml
ppo:
  lr: 3e-4 # Learning rate
  gamma: 0.99 # Discount factor
  clip_eps: 0.2 # PPO clip range
  update_epochs: 10 # Epochs per update

network:
  hidden_sizes: [64, 64]

reward:
  reward_goal: 100.0
  reward_collision: -50.0
```

## Testing

```bash
# Run all tests
pytest -v

# Run specific test
pytest tests/test_observation.py -v
```

## Results

Training curves and evaluation metrics are saved to `results/plots/`:

- `training_curves.png` - Episode return, success rate over training
- `maze_comparison.png` - Performance comparison across mazes

## Documentation

- [Algorithm Description](docs/algorithm.md) - PPO algorithm details
- [System Architecture](docs/architecture.md) - Component diagrams

## License

Apache-2.0

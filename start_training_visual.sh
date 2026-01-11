#!/bin/bash
# Start visual training with proper environment and paths

# Exit on error
set -e

# Navigate to workspace root
cd /mnt/c/pixi_ws/ros2_ws

# Source ROS2
source /opt/ros/jazzy/setup.bash

# Source workspace
source install/setup.bash

# Activate native Python venv
source ~/venv_ros2_native/bin/activate

# Navigate to package directory
cd src/learning_robotics

# Set config file absolute path
CONFIG_PATH=$(pwd)/lrppo/config.yaml

echo "========================================"
echo "Starting PPO Training with Visual Feedback"
echo "========================================"
echo "Config: $CONFIG_PATH"
echo "Checkpoint Dir: checkpoints"
echo "Rollout Steps: 128 (fast iteration)"
echo "========================================"
echo ""

# Run training node with absolute config path
python3 -m lrppo.ros.train_node \
    --ros-args \
    -p config_path:=$CONFIG_PATH \
    -p rollout_steps:=128 \
    -p checkpoint_interval:=5


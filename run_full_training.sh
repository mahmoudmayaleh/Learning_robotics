#!/bin/bash
# Complete training setup: Gazebo + TurtleBot3 + Goal Publisher + Training Node

# Exit on error
set -e

echo "========================================"
echo "Complete PPO Training Setup"
echo "========================================"

# Set TurtleBot3 model
export TURTLEBOT3_MODEL=burger

# Navigate to workspace
cd /mnt/c/pixi_ws/ros2_ws

# Source ROS2
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Launch Gazebo with TurtleBot3 in background
echo "1. Launching Gazebo with TurtleBot3..."
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py &
GAZEBO_PID=$!

# Wait for Gazebo to be ready
sleep 10

# Check if scan topic is publishing
echo "2. Verifying robot sensors..."
timeout 5 ros2 topic echo /scan --once > /dev/null 2>&1 && echo "   ✓ LiDAR working" || echo "   ✗ LiDAR not ready"

# Start goal publisher
echo "3. Starting goal publisher..."
cd src/learning_robotics
python3 publish_goal.py &
GOAL_PID=$!

sleep 2

# Start training node with native venv
echo "4. Starting PPO training node..."
source ~/venv_ros2_native/bin/activate
CONFIG_PATH=$(pwd)/lrppo/config.yaml

python3 -m lrppo.ros.train_node \
    --ros-args \
    -p config_path:=$CONFIG_PATH \
    -p rollout_steps:=128 \
    -p checkpoint_interval:=5 &
TRAIN_PID=$!

echo ""
echo "========================================"
echo "Training System Running"
echo "========================================"
echo "Gazebo PID: $GAZEBO_PID"
echo "Goal Publisher PID: $GOAL_PID"
echo "Training Node PID: $TRAIN_PID"
echo ""
echo "Press Ctrl+C to stop all processes"
echo "========================================"

# Wait and handle cleanup
trap "kill $GAZEBO_PID $GOAL_PID $TRAIN_PID 2>/dev/null; exit" INT TERM

# Keep script running
wait

#!/bin/bash
# Simplified training script that actually works

MAZE_TYPE=${1:-simple}
WORLD_FILE="/mnt/c/pixi_ws/ros2_ws/src/learning_robotics/lrppo/worlds/${MAZE_TYPE}_maze.world"
RESULTS_DIR="results/${MAZE_TYPE}_maze"

echo "=========================================="
echo "TRAINING: $MAZE_TYPE maze"
echo "=========================================="

# Kill old processes
pkill -9 gz
pkill -9 python3
pkill -9 ruby
sleep 2

# Create results directory
mkdir -p $RESULTS_DIR

# Launch Gazebo with custom world
cd /mnt/c/pixi_ws/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export TURTLEBOT3_MODEL=waffle

# Copy custom maze world over the standard world (temporary hack)
STANDARD_WORLD="/opt/ros/jazzy/share/turtlebot3_gazebo/worlds/turtlebot3_world.world"
sudo cp "$WORLD_FILE" "$STANDARD_WORLD"
echo "Loaded $MAZE_TYPE maze as standard world"

echo "[1/3] Launching Gazebo with $MAZE_TYPE maze..."
# Use the standard turtlebot3_world launch which handles everything
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py x_pose:=-2.0 y_pose:=-0.5 &
GAZEBO_PID=$!

echo "Waiting for Gazebo to fully initialize sensors..."
sleep 25

# Force sensor activation by echoing topics
source /opt/ros/jazzy/setup.bash
timeout 2 ros2 topic echo /scan --once > /dev/null 2>&1 &
timeout 2 ros2 topic echo /odom --once > /dev/null 2>&1 &
sleep 3

# Start goal publisher
echo "[2/3] Starting Goal Publisher..."
cd /mnt/c/pixi_ws/ros2_ws/src/learning_robotics
python3 publish_goal.py &
GOAL_PID=$!

sleep 3

# Start training node
echo "[3/3] Starting Training Node..."
source ~/venv_ros2_native/bin/activate
source /opt/ros/jazzy/setup.bash
export PYTHONPATH=/mnt/c/pixi_ws/ros2_ws/src/learning_robotics:$PYTHONPATH

python3 lrppo/ros/train_node.py \
    --ros-args \
    -p config_path:=/mnt/c/pixi_ws/ros2_ws/src/learning_robotics/lrppo/config.yaml \
    -p rollout_steps:=1200 \
    -p checkpoint_interval:=3 \
    -p checkpoint_dir:=$RESULTS_DIR

# Cleanup
kill $GAZEBO_PID $GOAL_PID 2>/dev/null

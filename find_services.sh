#!/bin/bash
cd /mnt/c/pixi_ws/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export TURTLEBOT3_MODEL=waffle

echo "Launching Gazebo..."
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py > /dev/null 2>&1 &
PID=$!

echo "Waiting 25 seconds for Gazebo to start..."
sleep 25

echo "--- AVAILABLE TOPICS ---"
ros2 topic list
echo "--- END TOPICS ---"

echo "Killing Gazebo..."
kill $PID
pkill -f gz

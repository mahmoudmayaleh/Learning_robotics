# Learning Robotics - TurtleBot3 PPO Training

## Quick Start

1. **Launch training:**

   ```bash
   cd /mnt/c/pixi_ws/ros2_ws/src/learning_robotics
   ./train_simple.sh simple
   ```

2. **What it does:**
   - Loads your custom maze (simple_maze.world)
   - Spawns TurtleBot3 robot
   - Starts goal publisher
   - Runs PPO training

## Current Status

- ✅ Custom maze loads
- ✅ Robot spawns
- ⚠️ Sensor timing issue (being debugged)
- Training starts after 25s wait

## Files Structure

- `train_simple.sh` - Main training script
- `lrppo/worlds/` - Custom maze worlds (simple/medium/complex)
- `lrppo/ros/train_node.py` - Training node
- `lrppo/ppo/` - PPO algorithm
- `lrppo/config.yaml` - Training parameters
- `publish_goal.py` - Goal publisher

## Training Parameters

- Episodes timeout: 100 steps
- Collision distance: 0.45m
- Arena bounds: -8 to +8
- Rollout steps: 1200
- Collision penalty: -10
- Goal reward: +200

## Known Issues

- Scan topic needs ~25s to activate (lazy bridge)
- No reset service in new Gazebo (uses soft reset)
- Flip detection disabled (causes infinite loops without reset)

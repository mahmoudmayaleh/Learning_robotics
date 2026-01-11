"""Train node for lrppo (ROS 2, rclpy).

High-level training loop stub that integrates sensors, the PPO trainer,
observation builder and reward calculator. This node is intentionally
light on environment-specific details and contains TODOs where you should
implement reset logic and goal conversion for your maze/simulator.
"""
from typing import Optional, Tuple
import os
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from geometry_msgs.msg import Twist, PoseStamped, TwistStamped
from std_srvs.srv import Empty

import numpy as np

import torch

from lrppo.utils.observation import ObservationBuilder
from lrppo.utils.reward import MazeReward
from lrppo.ppo.train import PPOTrainer


class TrainNode(Node):
    """ROS2 node that runs the training loop for TurtleBot3 + PPO.

    This implementation is a high-level template and contains TODOs for
    environment reset, goal handling, and sim-specific services.
    """

    def __init__(self):
        super().__init__('lrppo_train_node')

        # Parameters (tunable)
        self.declare_parameter('num_beams', 48)
        self.declare_parameter('lidar_max_range', 6.0)
        self.declare_parameter('control_rate', 10.0)  # Hz
        self.declare_parameter('collision_distance', 0.15)
        self.declare_parameter('rollout_steps', 2048)
        self.declare_parameter('checkpoint_dir', 'checkpoints')
        self.declare_parameter('checkpoint_interval', 1)  # updates
        self.declare_parameter('max_episode_steps', 100)  # Reduced timeout for faster recovery from flipped states
        
        # Arena boundaries (adjust based on your world)
        self.declare_parameter('arena_x_min', -8.0)
        self.declare_parameter('arena_x_max', 8.0)
        self.declare_parameter('arena_y_min', -8.0)
        self.declare_parameter('arena_y_max', 8.0)

        self.declare_parameter('load_checkpoint', '') # Path to .pt file to load
        
        num_beams = self.get_parameter('num_beams').get_parameter_value().integer_value
        lidar_max_range = self.get_parameter('lidar_max_range').get_parameter_value().double_value
        control_rate = self.get_parameter('control_rate').get_parameter_value().double_value
        self.collision_distance = self.get_parameter('collision_distance').get_parameter_value().double_value
        # Force larger collision distance for Waffle - increased for stability
        self.collision_distance = 0.45 
        
        self.rollout_steps = self.get_parameter('rollout_steps').get_parameter_value().integer_value
        self.checkpoint_dir = self.get_parameter('checkpoint_dir').get_parameter_value().string_value
        self.checkpoint_interval = self.get_parameter('checkpoint_interval').get_parameter_value().integer_value
        self.max_episode_steps = self.get_parameter('max_episode_steps').get_parameter_value().integer_value
        load_checkpoint_path = self.get_parameter('load_checkpoint').get_parameter_value().string_value
        
        # Arena boundaries
        self.arena_x_min = self.get_parameter('arena_x_min').get_parameter_value().double_value
        self.arena_x_max = self.get_parameter('arena_x_max').get_parameter_value().double_value
        self.arena_y_min = self.get_parameter('arena_y_min').get_parameter_value().double_value
        self.arena_y_max = self.get_parameter('arena_y_max').get_parameter_value().double_value

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Observation builder, reward, trainer
        # Load config if available (override params)
        config_path = self.declare_parameter('config_path', 'config.yaml').get_parameter_value().string_value
        try:
            from lrppo.utils.config_loader import ConfigLoader

            cfg = ConfigLoader.load(config_path)
            # Override values from config mostly, but respect CLI overrides for specific training params
            if self.rollout_steps == 2048: # Default value
                self.rollout_steps = cfg.training.rollout_steps
            
            # Allow CLI to override checkpoint dir
            if self.checkpoint_dir == 'checkpoints':
                self.checkpoint_dir = cfg.training.checkpoint_dir
                
            num_beams = cfg.env.num_beams
            lidar_max_range = cfg.env.lidar_max_range

            control_rate = cfg.env.control_rate
            self.collision_distance = cfg.env.collision_distance
            print_interval = cfg.training.print_interval
            hidden_sizes = tuple(cfg.network.hidden_sizes)
            ppo_kwargs = cfg.ppo
            reward_cfg = cfg.reward
            maze = cfg.env.maze
        except Exception as e:
            # fallback to parameters above
            self.get_logger().warn(f'Failed to load config: {e}; using node parameters')

            hidden_sizes = (64, 64)
            ppo_kwargs = None
            reward_cfg = None
            print_interval = 0
            maze = 'default_maze'

        # Instantiate components
        self.obs_builder = ObservationBuilder(num_beams=num_beams, max_range=lidar_max_range)
        # Use reward config if present
        if reward_cfg is not None:
            self.rewarder = MazeReward(
                reward_goal=reward_cfg.reward_goal,
                reward_collision=reward_cfg.reward_collision,
                reward_timeout=reward_cfg.reward_timeout,
                progress_scale=reward_cfg.progress_scale,
            )
        else:
            self.rewarder = MazeReward()

        # Action bounds (linear m/s, angular rad/s) - tweak for TurtleBot3
        # Waffle is more stable, allowing higher speeds
        action_low = np.array([-0.10, -1.5], dtype=np.float32)   # higher backward, standard turn
        action_high = np.array([0.20, 1.5], dtype=np.float32)    # standard forward, standard turn


        # Initialize PPO trainer (pass network hidden sizes and ppo hyperparams when available)
        trainer_kwargs = {}
        if ppo_kwargs is not None:
            trainer_kwargs = dict(
                lr=ppo_kwargs.lr,
                gamma=ppo_kwargs.gamma,
                lam=ppo_kwargs.lam,
                clip_eps=ppo_kwargs.clip_eps,
                update_epochs=ppo_kwargs.update_epochs,
                minibatch_size=ppo_kwargs.minibatch_size,
                value_coef=ppo_kwargs.value_coef,
                ent_coef=ppo_kwargs.ent_coef,
                max_grad_norm=ppo_kwargs.max_grad_norm,
                rollout_capacity=ppo_kwargs.rollout_capacity,
            )

        self.trainer = PPOTrainer(
            obs_dim=self.obs_builder.observation_dim,
            action_dim=2,
            action_low=action_low,
            action_high=action_high,
            device='cpu',
            hidden_sizes=hidden_sizes,
            **trainer_kwargs,
        )
        
        # Load checkpoint if provided
        if load_checkpoint_path and os.path.exists(load_checkpoint_path):
            self.get_logger().info(f'Loading model weights from {load_checkpoint_path}...')
            try:
                state_dict = torch.load(load_checkpoint_path, map_location=self.trainer.device)
                self.trainer.model.load_state_dict(state_dict)
                self.get_logger().info('Model weights loaded successfully.')
            except Exception as e:
                self.get_logger().error(f'Failed to load checkpoint: {e}')
        elif load_checkpoint_path:
             self.get_logger().warn(f'Checkpoint path {load_checkpoint_path} provided but not found. Starting from scratch.')


        # Set up CSV logger for episode metrics
        from lrppo.utils.logger import CSVLogger

        metrics_path = os.path.join(self.checkpoint_dir, 'metrics.csv')
        fieldnames = ['episode', 'episode_return', 'success', 'collisions', 'steps']
        self.csv_logger = CSVLogger(metrics_path, fieldnames=fieldnames, print_interval=print_interval)

        # ROS interfaces
        # Use Best Effort QoS for sensors to match Gazebo bridge defaults
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self._odom_cb, qos_profile_sensor_data)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal', self._goal_cb, 10)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        # Service for hard reset (try multiple known Gazebo service names)
        # New Gazebo (gz sim) doesn't have reset_simulation, use entity respawn instead
        self.reset_clients = []
        for srv_name in [
            '/reset_simulation',  # Classic Gazebo
            '/reset_world',  # Classic Gazebo
            '/gazebo/reset_simulation',  # Classic Gazebo with namespace
            '/gazebo/reset_world'  # Classic Gazebo with namespace
        ]:
            self.reset_clients.append(self.create_client(Empty, srv_name))
        
        # Fallback: If no reset service, we'll use entity delete + respawn
        # Import additional service types for new Gazebo
        try:
            from ros_gz_interfaces.srv import DeleteEntity, SpawnEntity
            self.delete_client = self.create_client(DeleteEntity, '/world/turtlebot3_world/remove')
            # Note: Spawn requires more parameters, so we'll skip it for now and just stop cmd_vel
            self.has_new_gazebo_services = True
        except ImportError:
            self.has_new_gazebo_services = False

        # Internal state
        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Odometry] = None
        self.latest_goal: Optional[PoseStamped] = None

        self.prev_dist: Optional[float] = None
        self.step_count = 0
        self.update_count = 0
        self.episode = 0

        # Episode accumulators
        self.episode_return = 0.0
        self.episode_collisions = 0
        self.episode_steps = 0
        self.reset_steps = 0  # Counter for recovery behavior
        self.episode_start_steps = 0  # Global step count at episode start
        self.in_recovery_mode = False  # Flag for flip recovery
        self.recovery_counter = 0  # Steps spent in recovery
        self.collision_grace_steps = 5  # Ignore collisions for first N steps

        # Timer for control loop

        period_s = 1.0 / float(control_rate)
        self.control_timer = self.create_timer(period_s, self._control_step)

        self.get_logger().info('lrppo train node started')

    # ----- Callbacks -----
    def _scan_cb(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def _odom_cb(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def _goal_cb(self, msg: PoseStamped) -> None:
        self.latest_goal = msg

    # ----- Helpers -----
    def _compute_goal_relative(self, odom: Odometry, goal: PoseStamped) -> Tuple[float, float]:
        """Compute relative goal distance and bearing from odometry and goal pose.

        Returns:
            (distance_m, angle_rad) where angle is relative bearing in robot frame.
        """
        # TODO: if you already have a helper that computes this, call it here
        px = odom.pose.pose.position.x
        py = odom.pose.pose.position.y
        q = odom.pose.pose.orientation
        # convert quaternion to yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        gx = goal.pose.position.x
        gy = goal.pose.position.y

        dx = gx - px
        dy = gy - py
        dist = math.hypot(dx, dy)

        goal_angle = math.atan2(dy, dx) - yaw
        # normalize to [-pi, pi]
        goal_angle = (goal_angle + math.pi) % (2 * math.pi) - math.pi
        return dist, goal_angle

    def _scan_to_ranges(self, scan: LaserScan) -> np.ndarray:
        """Convert LaserScan message to 1-D numpy ranges array with sanitization."""
        ranges = np.array(scan.ranges, dtype=np.float32)
        # Replace infinity/NaN with max range
        ranges = np.nan_to_num(ranges, posinf=scan.range_max, neginf=scan.range_min, nan=scan.range_max)
        # Filter out invalid close readings (Gazebo sometimes returns 0.0 for errors)
        # Only treat as processed if > range_min or reasonable threshold
        mask = ranges < 0.05
        ranges[mask] = scan.range_max
        return ranges

    def _is_flipped(self, odom: Odometry) -> bool:
        """Check if robot is flipped over (roll or pitch > threshold)."""
        q = odom.pose.pose.orientation
        
        # Roll
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Check thresholds - 45 de
    
    def _is_out_of_bounds(self, odom: Odometry) -> bool:
        """Check if robot is outside the arena boundaries."""
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        
        out_of_bounds = (x < self.arena_x_min or x > self.arena_x_max or 
                        y < self.arena_y_min or y > self.arena_y_max)
        
        if out_of_bounds:
            self.get_logger().warn(f'Robot out of bounds! Position: ({x:.2f}, {y:.2f})')
        
        return out_of_bounds

    def _call_reset_service(self, timeout_sec: float = 5.0) -> bool:
        """Attempt reset - but if unavailable, just do soft reset (stop robot).
        
        In new Gazebo without reset services, we rely on episode timeout
        and the robot learning to recover from flipped states.
        
        Returns:
            Always True (we handle reset gracefully either way).
        """
        # Try classic Gazebo reset services
        for client in self.reset_clients:
            if client.service_is_ready():
                req = Empty.Request()
                future = client.call_async(req)
                
                start_time = time.time()
                while not future.done() and (time.time() - start_time) < timeout_sec:
                    rclpy.spin_once(self, timeout_sec=0.1)
                
                if future.done():
                    self.get_logger().info(f'Reset service called: {client.srv_name}')
                    time.sleep(0.5)
                    return True
        
        # No reset service - do soft reset (stop velocities only)
        # Robot will timeout and start new episode from current position
        # This is acceptable for RL - robot learns to recover
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = 'base_link'
        twist.twist.linear.x = 0.0
        twist.twist.angular.z = 0.0
        for _ in range(3):
            self.cmd_pub.publish(twist)
            time.sleep(0.05)
        
        return True  # Soft reset is acceptable

    # ----- Main control step -----
    def _control_step(self) -> None:
        """One control timestep: create observation, ask trainer for action, publish and store transition."""
        
        # DISABLED: Flip detection causes issues without proper reset service
        # Check for catastrophic failure (flipping) - enter recovery mode first
        # if self.latest_odom is not None and self._is_flipped(self.latest_odom):
        #     ... recovery code disabled ...
        
        # Skip flip detection - let episode timeout handle it instead
        if self.in_recovery_mode:
            self.in_recovery_mode = False
            self.recovery_counter = 0

        # Require sensor and goal data
        if self.latest_scan is None or self.latest_odom is None or self.latest_goal is None:
            # waiting for data - log every 2 seconds roughly (control_rate is 10hz)
            if self.step_count % 20 == 0:
                self.get_logger().warn(
                    f'Waiting for data... Scan: {self.latest_scan is not None}, '
                    f'Odom: {self.latest_odom is not None}, Goal: {self.latest_goal is not None}'
                )
            self.step_count += 1
            return


        # Build observation
        lidar = self._scan_to_ranges(self.latest_scan)

        lin_vel = self.latest_odom.twist.twist.linear.x
        ang_vel = self.latest_odom.twist.twist.angular.z

        goal_dist, goal_angle = self._compute_goal_relative(self.latest_odom, self.latest_goal)

        # SPAM PROTECTION: If we "spawn" at the goal (because reset failed), retry reset and skip
        if self.episode_steps < 2 and goal_dist < 0.5:
            self.get_logger().warn('Spawned at goal (Reset failed?). Retrying reset...')
            self._call_reset_service()
            return
            
        obs = self.obs_builder.build_observation(lidar, lin_vel, ang_vel, goal_dist, goal_angle)

        # Select action
        action, logp, value = self.trainer.select_action(obs)

        
        # Check if robot is out of bounds (treat as collision)
        out_of_bounds = self._is_out_of_bounds(self.latest_odom)
        if out_of_bounds:
            collided = True
        # Publish cmd_vel Use TwistStamped as required by bridge
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = 'base_link'
        twist.twist.linear.x = float(action[0])
        twist.twist.angular.z = float(action[1])
        self.cmd_pub.publish(twist)

        # Basic collision detection from lidar with grace period
        # Ignore collisions for first few steps after reset to avoid immediate failure
        steps_since_episode_start = self.step_count - self.episode_start_steps
        collided_raw = np.any(lidar < self.collision_distance)
        collided = collided_raw and (steps_since_episode_start > self.collision_grace_steps)

        # Compute timed_out and reached_goal using environment/goal criteria
        reached_goal = bool(goal_dist < 0.2)
        timed_out = self.episode_steps >= self.max_episode_steps

        # Reward calculation
        prev_d = self.prev_dist if self.prev_dist is not None else goal_dist
        reward = self.rewarder.compute_reward(prev_d, goal_dist, collided, reached_goal, timed_out, step_penalty=0.01)
        done = self.rewarder.is_done(collided, reached_goal, timed_out)

        # Episode accumulators
        self.episode_return += float(reward)
        if collided:
            self.episode_collisions += 1
        self.episode_steps += 1

        # Store transition
        self.trainer.store_transition(obs, action, logp, reward, done, value)

        self.prev_dist = goal_dist
        self.step_count += 1

        # If episode done, finish path and reset env
        if done:
            # Bootstrapping with value=0 for terminal states
            self.trainer.finish_path(last_value=0.0)

            # Record episode metrics to trainer and CSV logger
            success = bool(reached_goal)
            self.trainer.record_episode(self.episode_return, success, self.episode_collisions, self.episode_steps)
            row = {
                'episode': int(self.episode),
                'episode_return': float(self.episode_return),
                'success': int(success),
                'collisions': int(self.episode_collisions),
                'steps': int(self.episode_steps),
            }
            try:
                self.csv_logger.log_episode(row)
                self.csv_logger.flush()
            except Exception as e:
                self.get_logger().warn(f'Failed to log episode metrics: {e}')

            self._reset_episode()
            # Do NOT return here, so we can check if update is needed
            # return

        # If rollout buffer full or enough steps collected, finish path and update

        if self.trainer.buffer.full or self.step_count >= self.rollout_steps:
            # Bootstrapping using value estimate of current obs
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.trainer.device).unsqueeze(0)
            with torch.no_grad():
                _, _, val, _ = self.trainer.model.act(obs_t)
            last_val = float(val.cpu().item())

            self.trainer.finish_path(last_value=last_val)

            info = self.trainer.update()
            self.update_count += 1

            # Save checkpoint periodically
            if self.update_count % max(1, self.checkpoint_interval) == 0:
                self._save_checkpoint(self.update_count)

            # Reset counters for next rollout
            self.step_count = 0

            self.get_logger().info(f'PPO update performed: {info}')

    def _reset_episode(self) -> None:
        """Reset environment and episode-internal state."""
        # Save metrics already recorded to trainer and CSV logger were done at episode end
        # Reset episode accumulators
        self.episode += 1
        self.prev_dist = None
        self.episode_return = 0.0
        self.episode_collisions = 0
        self.episode_steps = 0
        self.episode_start_steps = self.step_count
        
        # Call simulation reset to return robot to start pose
        # In new Gazebo, this may just stop the robot (soft reset)
        self._call_reset_service()
        self.get_logger().info(f'Episode {self.episode - 1} finished (steps={self.episode_steps}).')
        
        # Clear sensor data to force fresh readings after reset
        self.latest_scan = None
        self.latest_odom = None
        # Don't clear goal, it should persist


    def _save_checkpoint(self, update_idx: int) -> None:
        """Save model checkpoint to disk."""
        path = os.path.join(self.checkpoint_dir, f'ppo_update_{update_idx}.pt')
        try:
            torch.save(self.trainer.model.state_dict(), path)
            self.get_logger().info(f'Checkpoint saved: {path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save checkpoint: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = TrainNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

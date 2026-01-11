#!/usr/bin/env python3
"""Inference node - runs trained PPO model for evaluation/demo."""
import os
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped

import numpy as np
import torch

from lrppo.ppo.network import ActorCritic
from lrppo.utils.observation import ObservationBuilder


class InferenceNode(Node):
    def __init__(self):
        super().__init__('lrppo_inference_node')

        self.declare_parameter('checkpoint', '')
        self.declare_parameter('num_beams', 48)
        self.declare_parameter('lidar_max_range', 6.0)
        self.declare_parameter('control_rate', 10.0)
        self.declare_parameter('deterministic', True)

        ckpt = self.get_parameter('checkpoint').get_parameter_value().string_value
        num_beams = self.get_parameter('num_beams').get_parameter_value().integer_value
        lidar_max = self.get_parameter('lidar_max_range').get_parameter_value().double_value
        rate = self.get_parameter('control_rate').get_parameter_value().double_value
        self.deterministic = self.get_parameter('deterministic').get_parameter_value().bool_value

        self.obs_builder = ObservationBuilder(num_beams=num_beams, max_range=lidar_max)

        self.model = ActorCritic(self.obs_builder.observation_dim, 2, hidden_sizes=(64, 64))
        if ckpt and os.path.exists(ckpt):
            self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
            self.get_logger().info(f'Loaded checkpoint: {ckpt}')
        else:
            self.get_logger().warn('No checkpoint loaded, using random policy')
        self.model.eval()

        self.action_low = np.array([-0.22, -2.84])
        self.action_high = np.array([0.22, 2.84])

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal', self._goal_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Odometry] = None
        self.latest_goal: Optional[PoseStamped] = None

        self.timer = self.create_timer(1.0 / rate, self._control_step)
        self.get_logger().info('Inference node started')

    def _scan_cb(self, msg): self.latest_scan = msg
    def _odom_cb(self, msg): self.latest_odom = msg
    def _goal_cb(self, msg): self.latest_goal = msg

    def _control_step(self):
        if self.latest_scan is None or self.latest_odom is None or self.latest_goal is None:
            return

        lidar = np.array(self.latest_scan.ranges, dtype=np.float32)
        lin_vel = self.latest_odom.twist.twist.linear.x
        ang_vel = self.latest_odom.twist.twist.angular.z

        px = self.latest_odom.pose.pose.position.x
        py = self.latest_odom.pose.pose.position.y
        q = self.latest_odom.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

        gx = self.latest_goal.pose.position.x
        gy = self.latest_goal.pose.position.y
        dx, dy = gx - px, gy - py
        goal_dist = math.hypot(dx, dy)
        goal_angle = math.atan2(dy, dx) - yaw
        goal_angle = (goal_angle + math.pi) % (2*math.pi) - math.pi

        obs = self.obs_builder.build_observation(lidar, lin_vel, ang_vel, goal_dist, goal_angle)
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            if self.deterministic:
                mean, _, _ = self.model.forward(obs_t)
                action = mean.squeeze(0).numpy()
            else:
                action, _, _, _ = self.model.act(obs_t)
                action = action.squeeze(0).numpy()

        action = np.clip(action, self.action_low, self.action_high)

        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

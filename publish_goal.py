#!/usr/bin/env python3
"""Publish random goals for training."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import random
import time


class GoalPublisher(Node):
    def __init__(self):
        super().__init__('goal_publisher')
        self.publisher = self.create_publisher(PoseStamped, '/goal', 10)
        self.timer = self.create_timer(30.0, self.publish_goal)  # New goal every 30 seconds
        self.publish_goal()  # Publish first goal immediately
        self.get_logger().info('Goal publisher started')

    def publish_goal(self):
        """Publish a random goal pose within the environment bounds."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        
        # Random goal within reasonable bounds (adjust for your environment)
        msg.pose.position.x = random.uniform(-2.0, 2.0)
        msg.pose.position.y = random.uniform(-2.0, 2.0)
        msg.pose.position.z = 0.0
        
        # Orientation doesn't matter for goal reaching
        msg.pose.orientation.w = 1.0
        
        self.publisher.publish(msg)
        self.get_logger().info(
            f'Published goal: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}'
        )


def main():
    rclpy.init()
    node = GoalPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Simple test to move the robot"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class SimpleMove(Node):
    def __init__(self):
        super().__init__('simple_move')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.move)
        self.counter = 0
        
    def move(self):
        msg = Twist()
        if self.counter < 50:
            msg.linear.x = 0.2  # Forward
            self.get_logger().info('Moving forward')
        elif self.counter < 100:
            msg.angular.z = 0.5  # Turn
            self.get_logger().info('Turning')
        else:
            self.counter = 0
        
        self.pub.publish(msg)
        self.counter += 1

def main():
    rclpy.init()
    node = SimpleMove()
    print("Robot should start moving now!")
    rclpy.spin(node)

if __name__ == '__main__':
    main()

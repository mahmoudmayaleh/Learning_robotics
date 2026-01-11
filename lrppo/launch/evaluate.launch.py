from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('lrppo')

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='simple_maze'),
        DeclareLaunchArgument('checkpoint', default_value=''),
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('gazebo_ros'), '/launch/gazebo.launch.py'
            ]),
            launch_arguments={
                'world': [pkg_dir, '/worlds/', LaunchConfiguration('world'), '.world']
            }.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('turtlebot3_gazebo'), '/launch/spawn_turtlebot3.launch.py'
            ]),
            launch_arguments={'x_pose': '-4.0', 'y_pose': '-4.0'}.items(),
        ),

        Node(
            package='lrppo',
            executable='inference_node',
            name='inference_node',
            parameters=[{
                'checkpoint': LaunchConfiguration('checkpoint'),
                'deterministic': True,
                'use_sim_time': True,
            }],
            output='screen',
        ),
    ])

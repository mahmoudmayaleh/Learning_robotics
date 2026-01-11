from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('lrppo')
    world_file = os.path.join(pkg_dir, 'worlds', 'simple_maze.world')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('gazebo_ros'), '/launch/gazebo.launch.py'
            ]),
            launch_arguments={'world': world_file}.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('turtlebot3_gazebo'), '/launch/spawn_turtlebot3.launch.py'
            ]),
            launch_arguments={'x_pose': '-4.0', 'y_pose': '-4.0'}.items(),
        ),

        Node(
            package='lrppo',
            executable='train_node',
            name='train_node',
            parameters=[{
                'config_path': os.path.join(pkg_dir, 'config.yaml'),
                'use_sim_time': True,
            }],
            output='screen',
        ),
    ])

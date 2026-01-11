from setuptools import setup
import os
from glob import glob

package_name = 'lrppo'

setup(
    name=package_name,
    version='1.0.0',
    packages=['lrppo', 'lrppo.ros', 'lrppo.ppo', 'lrppo.utils', 'lrppo.envs'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['lrppo/config.yaml']),
        ('share/' + package_name + '/worlds', glob('lrppo/worlds/*.world')),
        ('share/' + package_name + '/launch', glob('lrppo/launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Group 4',
    maintainer_email='group4@example.com',
    description='PPO-based navigation training for TurtleBot3 in Gazebo.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_node = lrppo.ros.train_node:main',
            'inference_node = lrppo.ros.inference_node:main',
        ],
    },
)

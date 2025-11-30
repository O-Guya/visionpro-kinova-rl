#!/usr/bin/env python3
"""Launch file for Kinova Gen3 teleoperation system.

This launch file starts:
1. Kinova Gen3 robot controller (ros2_kortex)
2. Teleoperation system with Vision Pro
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate launch description."""

    # Declare arguments
    vision_pro_ip_arg = DeclareLaunchArgument(
        'vision_pro_ip',
        default_value='192.168.1.100',
        description='Vision Pro IP address'
    )

    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='my_gen3',
        description='Robot namespace'
    )

    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.1.10',
        description='Kinova Gen3 IP address'
    )

    sim_mode_arg = DeclareLaunchArgument(
        'sim',
        default_value='false',
        description='Run in simulation mode'
    )

    config_path_arg = DeclareLaunchArgument(
        'config',
        default_value='',
        description='Path to safety configuration YAML'
    )

    urdf_path_arg = DeclareLaunchArgument(
        'urdf',
        default_value='',
        description='Path to robot URDF file'
    )

    # Get package share directory
    # Note: This assumes the package is installed
    # For development, you may need to adjust paths

    # Launch Kinova robot controller (if not in sim mode)
    # This would typically include the kortex_bringup launch file
    # Uncomment and adjust path when ready:
    #
    # kortex_bringup = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([
    #         PathJoinSubstitution([
    #             FindPackageShare('kortex_bringup'),
    #             'launch',
    #             'gen3.launch.py'
    #         ])
    #     ]),
    #     launch_arguments={
    #         'robot_ip': LaunchConfiguration('robot_ip'),
    #         'use_fake_hardware': 'false',
    #     }.items()
    # )

    # Teleoperation system node
    # Note: This is a placeholder - actual execution will be via Python script
    # You would typically create a ROS2 node wrapper for the main_loop.py

    return LaunchDescription([
        vision_pro_ip_arg,
        robot_name_arg,
        robot_ip_arg,
        sim_mode_arg,
        config_path_arg,
        urdf_path_arg,

        # Add kortex_bringup when ready:
        # kortex_bringup,

        # Note: For now, users should run the teleoperation system directly:
        # python3 -m kinova_teleoperation.main_loop --vision-pro-ip IP --config config/safety_params.yaml
    ])


if __name__ == '__main__':
    generate_launch_description()

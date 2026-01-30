import os
import sys
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    rviz_guys_publisher = Node(
        package="eyes-on-the-guys",
        executable="rviz_publisher",
        name="rviz_guys_publisher",
        output="screen",
    )

    eyes_on_guys_package_share_location = get_package_share_directory(
        "eyes-on-the-guys"
    )
    simulator_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("rosflight_sim"),
                    "launch/fixedwing_standalone.launch.py",
                )
            ]
        ),
        launch_arguments={
            "rviz2_config_file": os.path.join(
                eyes_on_guys_package_share_location,
                "resource",
                "eyes-on-guys-rviz-config.rviz",
            )
        }.items(),
    )

    return LaunchDescription([rviz_guys_publisher, simulator_launch_include])

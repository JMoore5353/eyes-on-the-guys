import os
import sys
import launch.actions
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directories
    rosplane_dir = get_package_share_directory('rosplane')
    eyes_on_guys_dir = get_package_share_directory('eyes_on_the_guys')
    aircraft = "anaconda"  # Default aircraft

    autopilot_params = os.path.join(
        eyes_on_guys_dir,
        'params',
        aircraft + '_autopilot_params.yaml'
    )
    
    estimator_params = os.path.join(
        rosplane_dir,
        'params',
        'estimator.yaml'
    )

    # eyes_on_the_guys nodes
    rviz_guys_publisher = Node(
        package="eyes_on_the_guys",
        executable="rviz_publisher",
        name="rviz_guys_publisher",
        output="screen",
    )

    guys_sim = Node(
        package="eyes_on_the_guys",
        executable="guys",
        name="guy_sim",
        output="screen",
        parameters=[
            {'use_sim_time': launch.substitutions.LaunchConfiguration('use_sim_time')},
        ],
    )

    planner = Node(
        package="eyes_on_the_guys",
        executable="planner",
        name="planner",
        output="screen",
        parameters=[
            autopilot_params,
            {'use_sim_time': launch.substitutions.LaunchConfiguration('use_sim_time')},
        ]
    )

    # Rosplane nodes (without path_planner)
    controller = Node(
        package='rosplane',
        executable='controller',
        name='controller',
        parameters=[
            autopilot_params,
            {'use_sim_time': launch.substitutions.LaunchConfiguration('use_sim_time')},
        ],
        output='screen'
    )

    path_follower = Node(
        package='rosplane',
        executable='path_follower',
        name='path_follower',
        parameters=[
            autopilot_params,
            {'use_sim_time': launch.substitutions.LaunchConfiguration('use_sim_time')},
        ]
    )

    path_manager = Node(
        package='rosplane',
        executable='path_manager',
        name='path_manager',
        parameters=[
            autopilot_params,
            {'use_sim_time': launch.substitutions.LaunchConfiguration('use_sim_time')},
        ]
    )

    estimator = Node(
        package='rosplane',
        executable='estimator',
        name='estimator',
        output='screen',
        parameters=[
            estimator_params,
            {'use_sim_time': launch.substitutions.LaunchConfiguration('use_sim_time')},
        ],
    )

    eyes_on_guys_package_share_location = get_package_share_directory(
        "eyes_on_the_guys"
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

    rviz_waypoint_publisher = Node(
        package='eyes_on_the_guys',
        executable='rviz_waypoint_only_publisher',
        output='screen',
    )

    return LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Whether or not to use the /clock topic in simulation to run timers."
        ),
        rviz_guys_publisher,
        guys_sim,
        planner,
        controller,
        path_follower,
        path_manager,
        estimator,
        simulator_launch_include,
        rviz_waypoint_publisher,
    ])

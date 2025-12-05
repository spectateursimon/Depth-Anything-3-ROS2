from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for Depth Anything 3 ROS2 node."""

    # Get package directory
    pkg_dir = get_package_share_directory("depth_anything_3_ros2")
    config_file = os.path.join(pkg_dir, "config", "default.yaml")

    # Declare launch arguments
    # image_topic_arg = DeclareLaunchArgument(
    #     "image_topic",
    #     default_value="/camera/color/image_raw",
    #     description="Input RGB image topic",
    # )

    # depth_image_topic_arg = DeclareLaunchArgument(
    #     "depth_image_topic",
    #     default_value="/depth",
    #     description="Output depth image topic",
    # )

    # device_arg = DeclareLaunchArgument(
    #     "device",
    #     default_value="cuda:0",
    #     description="Device to run inference on (cuda:0, cpu, etc.)",
    # )

    # model_name_arg = DeclareLaunchArgument(
    #     "model_name",
    #     default_value="depth-anything/DA3-Large",
    #     description="Depth Anything 3 model name",
    # )

    # Create node
    depth_anything_node = Node(
        package="depth_anything_3_ros2",
        executable="depth_anything_node",
        name="depth_anything_3",
        output="screen",
        parameters=[config_file],
    )

    return LaunchDescription(
        [
            depth_anything_node,
        ]
    )

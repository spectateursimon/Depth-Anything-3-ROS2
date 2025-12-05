#!/usr/bin/env python3

from lib import utils
import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image


class DepthAnything3Node(Node):
    """ROS2 Node for Depth Anything 3 monocular depth estimation."""

    def __init__(self):
        super().__init__("depth_anything_3_node")

        # Declare parameters
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_image_topic", "/depth")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("model_name", "depth-anything/DA3-Large")

        # Get parameters
        self.image_topic = self.get_parameter("image_topic").value
        self.depth_image_topic = self.get_parameter("depth_image_topic").value
        self.device = self.get_parameter("device").value
        self.model_name = self.get_parameter("model_name").value

        self.get_logger().info("Initializing Depth Anything 3 Node...")
        self.get_logger().info(f"Input topic: {self.image_topic}")
        self.get_logger().info(f"Output topic: {self.depth_image_topic}")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(f"Model: {self.model_name}")

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # set colormap used for testing
        self.colormap = "turbo"

        # Load Depth Anything 3 model
        try:
            from depth_anything_3.api import DepthAnything3

            self.get_logger().info("Loading Depth Anything 3 model...")
            self.model = DepthAnything3.from_pretrained(self.model_name)

            # Set device
            if "cuda" in self.device and torch.cuda.is_available():
                self.model = self.model.to(self.device)
                self.get_logger().info(f"Model loaded on {self.device}")
            else:
                self.model = self.model.to("cpu")
                self.device = "cpu"
                self.get_logger().info("CUDA not available, using CPU")

        except ImportError:
            self.get_logger().error(
                ("Could not import depth_anything_3.", " Please install it first.")
            )
            self.get_logger().error(
                (
                    "pip install git",
                    "https://github.com/ByteDance-Seed/Depth-Anything-3.git",
                )
            )
            raise
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {str(e)}")
            raise

        # Create subscriber and publisher
        self.rgb_sub_ = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10
        )

        self.depth_pub_ = self.create_publisher(Image, self.depth_image_topic, 10)
        self.colored_pub_ = self.create_publisher(
            Image, self.depth_image_topic + "_colored", 10
        )

        self.get_logger().info("Depth Anything 3 Node initialized.")

    def image_callback(self, msg):
        """Callback function for processing incoming images."""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image (if needed by the API)
            pil_image = PILImage.fromarray(rgb_image)

            # Perform inference
            with torch.no_grad():
                prediction = self.model.inference([pil_image])

            # Extract depth map
            depth = prediction.depth[0]  # Get first image depth [H, W]

            # Normalize depth for visualization (convert to uint16)
            # DA3 outputs metric or relative depth depending on model variant
            depth_normalized = depth.astype(np.float32)

            # Convert depth to uint16 for ROS2 (commonly used for depth images)
            # Scale to use full range of uint16
            depth_min = depth_normalized.min()
            depth_max = depth_normalized.max()

            if depth_max > depth_min:
                depth_scaled = (
                    (depth_normalized - depth_min) / (depth_max - depth_min) * 65535.0
                ).astype(np.uint16)
            else:
                depth_scaled = np.zeros_like(depth_normalized, dtype=np.uint16)

            # Convert to ROS Image message
            depth_msg = self.bridge.cv2_to_imgmsg(depth_scaled, encoding="16UC1")
            depth_msg.header = msg.header  # Preserve timestamp and frame_id

            colored_depth_normalized = utils.colorize_depth(
                depth_normalized, colormap=self.colormap
            )
            depth_colored_msg = self.bridge.cv2_to_imgmsg(
                colored_depth_normalized, encoding="bgr8"
            )
            depth_colored_msg.header = msg.header

            # Publish depth image
            self.depth_pub_.publish(depth_msg)
            self.colored_pub_.publish(depth_colored_msg)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")


def main(args=None):
    rclpy.init(args=args)

    try:
        node = DepthAnything3Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

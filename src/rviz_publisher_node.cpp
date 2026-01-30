#include <rclcpp/rclcpp.hpp>

#include "rviz_publisher.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<eyes_on_guys::RvizPublisher>();
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}

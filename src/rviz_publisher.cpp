#include <rclcpp/rclcpp.hpp>

#include "rviz_publisher.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  rclcpp::shutdown();
  return 0;
}

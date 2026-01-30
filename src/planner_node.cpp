#include <rclcpp/rclcpp.hpp>

#include "planner.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<eyes_on_guys::Planner>();
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}

#include <rclcpp/rclcpp.hpp>

#include "guys.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<eyes_on_guys::Guys>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

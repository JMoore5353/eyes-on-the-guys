#ifndef EYES_ON_THE_GUYS_GUYS_HPP
#define EYES_ON_THE_GUYS_GUYS_HPP

#include <deque>
#include <random>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

namespace eyes_on_guys
{

class Guys : public rclcpp::Node
{
public:
  Guys();

private:
  void declare_parameters();
  void update_positions();
  void initialize_guys();
  void log_names();
  static geometry_msgs::msg::Quaternion yaw_to_quaternion(double yaw);
  static double quaternion_to_yaw(const geometry_msgs::msg::Quaternion & quat);

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::vector<std::string> names_;
  std::mt19937 rng_;
  std::vector<geometry_msgs::msg::PoseStamped> guys_;
};

} // namespace eyes_on_guys

#endif

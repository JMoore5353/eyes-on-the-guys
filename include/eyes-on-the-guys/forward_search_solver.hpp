#ifndef FORWARD_SEARCH_SOLVER_HPP
#define FOWARD_SEARCH_SOLVER_HPP

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace eyes_on_guys
{

class ForwardSearchSolver : public rclcpp::Node
{
public:
  ForwardSearchSolver();

private:
  void poseCallback(const geometry_msgs::msg::PoseStamped & msg);
  void searchCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr forward_search_srv_;

  std::unordered_map<std::string, geometry_msgs::msg::PoseStamped> poses_by_id_;
  std::vector<std::string> ids_;
};

} // namespace eyes_on_guys

#endif

#include "forward_search_solver.hpp"

namespace eyes_on_guys
{

ForwardSearchSolver::ForwardSearchSolver()
: Node("forward_search_solver")
{
  this->declare_parameter("depth", 5);

  pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    "guy_poses",
    10,
    std::bind(&ForwardSearchSolver::poseCallback, this, std::placeholders::_1));

  forward_search_srv_ = this->create_service<std_srvs::srv::Trigger>(
    "forward_search",
    std::bind(&ForwardSearchSolver::searchCallback, this, std::placeholders::_1, std::placeholders::_2));
}

void ForwardSearchSolver::poseCallback(const geometry_msgs::msg::PoseStamped & msg)
{
  const std::string & id = msg.header.frame_id;
  auto [it, inserted] = poses_by_id_.insert_or_assign(id, msg);
  if (inserted) {
    ids_.push_back(id);
  }
}

void ForwardSearchSolver::searchCallback(
  const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
  int size = ids_.size();
  int steps = this->get_parameter("depth").as_int();

  RCLCPP_INFO_STREAM(this->get_logger(), "Beginning Search...");

  response->success = true;
}

} // namespace eyes_on_guys

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<eyes_on_guys::ForwardSearchSolver>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

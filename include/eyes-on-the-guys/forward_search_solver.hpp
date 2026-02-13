#ifndef FORWARD_SEARCH_SOLVER_HPP
#define FORWARD_SEARCH_SOLVER_HPP

#include <cstddef>
#include <string>
#include <map>
#include <vector>
#include <limits>

#include <eigen3/Eigen/Dense>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <eyes_on_the_guys/msg/bit.hpp>

namespace eyes_on_guys
{

class ForwardSearchSolver : public rclcpp::Node
{
public:
  ForwardSearchSolver();

private:
  struct GuyState
  {
    geometry_msgs::msg::PoseStamped pose;
    float bits = 0.0f;

    GuyState(geometry_msgs::msg::PoseStamped msg) : pose(msg), bits(0.0f) {}
    GuyState() : pose(geometry_msgs::msg::PoseStamped()), bits(0.0f) {}
  };
  
  struct Action
  {
    std::string who_to_go_to;
    double value;
  };
  
  struct ActionSequence 
  {
    std::vector<Action> sequence;

    double total_value;
  };

  struct SharedInfo
  {
    std::map<std::string, GuyState> guy_states_by_id_;
  };
  
  struct State 
  {
    std::string current_guy;
    Eigen::MatrixXd shared_info_matrix;
    Eigen::VectorXd relay_info;
    Eigen::VectorXd relay_time_since_visit;
    double value;
  };

  void pose_callback(const geometry_msgs::msg::PoseStamped & msg);
  void bits_callback(const eyes_on_the_guys::msg::Bit & msg);
  void search_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  ActionSequence forward_search(int depth, State state);
  double reward_function(State state, Action action);
  double roll_out(State State);
  double compute_dubins_path_length(float start_n, float start_e, float start_chi,
                                           float end_n, float end_e, float end_chi);
  double calculate_path_length(GuyState current_guy_state, GuyState next_guy_state);
  int index_for_id(const std::string & id) const;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<eyes_on_the_guys::msg::Bit>::SharedPtr bits_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr forward_search_srv_;

  std::map<std::string, GuyState> guy_states_by_id_;
  std::vector<std::string> ids_;
  std::vector<float> current_bits_;
  ActionSequence best_;
  State current_state_;
};

} // namespace eyes_on_guys

#endif

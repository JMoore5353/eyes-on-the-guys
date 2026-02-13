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
    float bits_rate = 0.0f;
    double last_bits_update_sec = -1.0;

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

  struct State 
  {
    std::string current_guy;
    Eigen::MatrixXd shared_info_matrix;
    Eigen::VectorXd relay_info;
    Eigen::VectorXd relay_time_since_visit;
    Eigen::VectorXd guy_bits;
    Eigen::VectorXd guy_bits_rate;
    double value;
  };

  void pose_callback(const geometry_msgs::msg::PoseStamped & msg);
  void bits_callback(const eyes_on_the_guys::msg::Bit & msg);
  void search_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  ActionSequence forward_search(int depth, State state);
  double reward_function(const State & state, const Action & action) const;
  double roll_out(const State & state);
  double compute_dubins_path_length(float start_n, float start_e, float start_chi,
                                           float end_n, float end_e, float end_chi);
  double calculate_path_length(const GuyState & current_guy_state, const GuyState & next_guy_state) const;
  int index_for_id(const std::string & id) const;
  std::string format_sequence_with_start(const std::string & start_id, const ActionSequence & sequence) const;
  std::vector<std::string> action_candidates(const std::string & current_guy) const;
  State make_initial_state(const std::string & starting_id, int starting_index, int size) const;
  bool apply_action_transition(
    const State & state,
    const std::string & next_id,
    double gamma,
    double vel,
    State & next_state,
    double & reward,
    bool emit_debug_logs) const;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<eyes_on_the_guys::msg::Bit>::SharedPtr bits_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr forward_search_srv_;

  std::map<std::string, GuyState> guy_states_by_id_;
  std::vector<std::string> ids_;
  State current_state_;
};

} // namespace eyes_on_guys

#endif

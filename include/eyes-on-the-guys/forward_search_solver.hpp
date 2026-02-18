#ifndef FORWARD_SEARCH_SOLVER_HPP
#define FORWARD_SEARCH_SOLVER_HPP

#include <cstddef>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <limits>

#include <eigen3/Eigen/Dense>

#include <geometry_msgs/msg/pose_stamped.hpp>

namespace eyes_on_guys
{

class ForwardSearchSolver
{
public:
  struct GuyState
  {
    geometry_msgs::msg::PoseStamped pose;
    float bits = 0.0f;
    float bits_rate = 1.0f;
    double last_bits_update_sec = -1.0;

    GuyState(geometry_msgs::msg::PoseStamped msg) : pose(msg), bits(0.0f) {}
    GuyState() : pose(geometry_msgs::msg::PoseStamped()), bits(0.0f) {}
  };

  struct ForwardSearchConfig
  {
    int depth = 6;
    int num_rollouts = 20;
    int roll_out_depth = 5;
    double discount_factor = 0.5;
    double agent_velocity = 17.0;
    double info_shared_weight = 10.0;
    double path_length_weight = 1.0;
    double time_since_visit_weight = 10.0;
    int starting_guy = 0;
    Eigen::MatrixXd shared_info_matrix;
  };

  struct ForwardSearchInput
  {
    std::map<std::string, GuyState> guy_states_by_id;
    std::vector<std::string> ids;
    Eigen::MatrixXd shared_info_matrix;
  };

  struct ForwardSearchResult
  {
    bool success = false;
    std::string message;
    std::vector<std::string> sequence_ids;
    double total_value = -std::numeric_limits<double>::infinity();
  };

  ForwardSearchSolver();
  ForwardSearchResult solve(const ForwardSearchInput & input, const ForwardSearchConfig & config);

private:
  struct Action
  {
    int next_index = -1;
    double value;
  };
  
  struct ActionSequence 
  {
    std::vector<Action> sequence;
    double total_value = -std::numeric_limits<double>::infinity();
  };

  struct State 
  {
    int current_index = -1;
    Eigen::MatrixXd shared_info_matrix;
    Eigen::VectorXd relay_info;
    Eigen::VectorXd relay_time_since_visit;
    Eigen::VectorXd guy_bits;
    Eigen::VectorXd guy_bits_rate;
    double value = 0.0;
  };

  ActionSequence forward_search(int depth, State & state);
  double reward_function(
    const State & state,
    int current_index,
    int next_index,
    double path_length) const;
  double roll_out(const State & state);
  double compute_dubins_path_length(float start_n, float start_e, float start_chi,
                                           float end_n, float end_e, float end_chi);
  static double calculate_path_length(const GuyState & current_guy_state, const GuyState & next_guy_state);
  State make_initial_state(
    int starting_index,
    int size,
    const Eigen::MatrixXd & initial_shared_info_matrix) const;
  bool apply_action_transition(
    State & state,
    int next_index,
    double gamma,
    double vel,
    double & reward,
    double & transition_time,
    bool emit_debug_logs) const;

  std::map<std::string, GuyState> guy_states_by_id_;
  std::vector<std::string> ids_;
  std::unordered_map<std::string, int> id_to_index_;
  std::vector<std::vector<int>> candidates_by_index_;
  Eigen::MatrixXd cached_path_lengths_;
  State current_state_;
  ForwardSearchConfig current_config_;
};

} // namespace eyes_on_guys

#endif

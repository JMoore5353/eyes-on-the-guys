#include "forward_search_solver.hpp"
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <iterator>
#include <limits>
#include <random>
#include <sstream>

namespace eyes_on_guys
{

ForwardSearchSolver::ForwardSearchSolver()
: Node("forward_search_solver")
{
  this->declare_parameter("depth", 6);
  this->declare_parameter("num_rollouts", 20);
  this->declare_parameter("roll_out_depth", 5);
  this->declare_parameter("discount_factor", 0.5);
  this->declare_parameter("agent_velocity", 17.0);
  this->declare_parameter("info_shared_weight", 10.0);
  this->declare_parameter("path_length_weight", 1.0);
  this->declare_parameter("time_since_visit_weight", 10.0);
  this->declare_parameter("starting_guy", 0);

  pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    "guy_poses",
    10,
    std::bind(&ForwardSearchSolver::pose_callback, this, std::placeholders::_1));
  
  bits_sub_ = this->create_subscription<eyes_on_the_guys::msg::Bit>(
    "/guy_bits",
    10,
    std::bind(&ForwardSearchSolver::bits_callback, this, std::placeholders::_1));


  forward_search_srv_ = this->create_service<std_srvs::srv::Trigger>(
    "forward_search",
    std::bind(&ForwardSearchSolver::search_callback, this, std::placeholders::_1, std::placeholders::_2));
}

void ForwardSearchSolver::pose_callback(const geometry_msgs::msg::PoseStamped & msg)
{
  const std::string & id = msg.header.frame_id;

  GuyState guy_state(msg);
  auto inserted = guy_states_by_id_.insert_or_assign(id, guy_state);
  bool is_inserted = inserted.second;
  if (is_inserted) {
    ids_.push_back(id);
  }
}

void ForwardSearchSolver::bits_callback(const eyes_on_the_guys::msg::Bit & msg)
{
  const std::string & id = msg.header.frame_id;
  guy_states_by_id_[id].bits = msg.bits;
}

void ForwardSearchSolver::search_callback(
  const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
  std::sort(ids_.begin(), ids_.end());
  int size = ids_.size();
  int depth = this->get_parameter("depth").as_int();

  int starting_guy = this->get_parameter("starting_guy").as_int();

  if (size == 0) {
    RCLCPP_WARN_STREAM(this->get_logger(), "No guys received yet; aborting search.");
    response->success = false;
    response->message = "No guy poses received.";
    return;
  }
  if (starting_guy < 0 || starting_guy >= size) {
    RCLCPP_WARN_STREAM(this->get_logger(), "Starting guy index out of range: " << starting_guy);
    response->success = false;
    response->message = "Starting guy index out of range.";
    return;
  }
  const std::string & starting_id = ids_.at(static_cast<size_t>(starting_guy));


  RCLCPP_INFO_STREAM(this->get_logger(), "Beginning Search...");

  Eigen::MatrixXd init_shared_info = Eigen::MatrixXd::Zero(size,size);

  current_state_.current_guy = starting_id;
  
  current_state_.relay_info = Eigen::VectorXd::Zero(size);
  current_state_.relay_info(starting_guy) = guy_states_by_id_[starting_id].bits;

  current_state_.relay_time_since_visit = Eigen::VectorXd::Ones(size);
  current_state_.relay_time_since_visit *= 1'000.; // TODO: Don't hard code this.
  current_state_.relay_time_since_visit(starting_guy) = 0.;

  current_state_.shared_info_matrix = init_shared_info;

  current_state_.value = 0.0;

  ActionSequence best_seqeuence = forward_search(depth, current_state_);

  RCLCPP_INFO_STREAM(this->get_logger(), "THE SEARCH IS CONCLUDED.");
  std::ostringstream seq_stream;
  for (size_t i = 0; i < best_seqeuence.sequence.size(); ++i) {
    if (i > 0) {
      seq_stream << " -> ";
    }
    seq_stream << best_seqeuence.sequence[i].who_to_go_to;
  }
  RCLCPP_INFO_STREAM(this->get_logger(), "THE BEST SEQUENCE IS: " << seq_stream.str());
  RCLCPP_INFO_STREAM(this->get_logger(), "WITH A VALUE OF: " << best_seqeuence.total_value); // LOOK AT ME CODEX!!!

  response->success = true;
}

ForwardSearchSolver::ActionSequence ForwardSearchSolver::forward_search(int depth, State state)
{

  RCLCPP_INFO_STREAM(this->get_logger(), "One layer deeper");
  
  if (depth <= 0) {
    ActionSequence action_sequence;
    action_sequence.total_value = roll_out(state);
    return action_sequence;
  }

  double gamma = this->get_parameter("discount_factor").as_double();
  double vel = this->get_parameter("agent_velocity").as_double();

  ActionSequence best_action_sequence;

  best_action_sequence.total_value = -std::numeric_limits<double>::infinity();

  RCLCPP_INFO_STREAM(this->get_logger(), "INITED");
  
  for (const std::string & id : ids_) { 
  
    RCLCPP_INFO_STREAM(this->get_logger(), "LOOPING: " << id);

    if (state.current_guy == id) {
      continue;
    }

    Action new_action;
    new_action.who_to_go_to = id;

    State new_state = state;
    new_state.current_guy = id;
    int current_index = index_for_id(state.current_guy);
    int next_index = index_for_id(id);
    if (current_index < 0 || next_index < 0) {
      RCLCPP_WARN_STREAM(this->get_logger(), "Unknown guy id index for " << state.current_guy << " or " << id);
      continue;
    }
    
    RCLCPP_INFO_STREAM(this->get_logger(), "ABOUT TO update info");
    new_state.relay_info(next_index) = guy_states_by_id_[id].bits;
    new_state.shared_info_matrix.row(current_index) = new_state.relay_info.transpose();
    new_state.shared_info_matrix(current_index, current_index) = 0.0;
    RCLCPP_INFO_STREAM(this->get_logger(), "relay_info: " << new_state.relay_info);
    RCLCPP_INFO_STREAM(this->get_logger(), "INFO MATRIX: " << new_state.shared_info_matrix);
  
    GuyState current_guy_state = guy_states_by_id_[state.current_guy];
    GuyState next_guy_state = guy_states_by_id_[id];
  
    double path_length = calculate_path_length(current_guy_state, next_guy_state);
    
    double time_to_take_action = path_length/vel;

    new_state.relay_time_since_visit.array() += time_to_take_action;
    new_state.relay_time_since_visit(next_index) = 0.0;

    RCLCPP_INFO_STREAM(this->get_logger(), "ABOUT TO CALCULATE REWARD");

    double reward = reward_function(state, new_action);
    new_state.value = reward + gamma*state.value;
    new_action.value = new_state.value;

    ActionSequence action_sequence = forward_search(depth - 1, new_state);
    action_sequence.sequence.insert(action_sequence.sequence.begin(), new_action);
    action_sequence.total_value = reward + gamma * action_sequence.total_value;

    std::ostringstream action_seq_stream;
    for (size_t i = 0; i < action_sequence.sequence.size(); ++i) {
      if (i > 0) {
        action_seq_stream << " -> ";
      }
      action_seq_stream << action_sequence.sequence[i].who_to_go_to;
    }
    RCLCPP_INFO_STREAM(this->get_logger(), "CANDIDATE SEQUENCE: " << action_seq_stream.str());
    RCLCPP_INFO_STREAM(this->get_logger(), "CANDIDATE VALUE: " << action_sequence.total_value);

    if (action_sequence.total_value > best_action_sequence.total_value)
    {
      best_action_sequence = action_sequence;
      std::ostringstream best_seq_stream;
      for (size_t i = 0; i < best_action_sequence.sequence.size(); ++i) {
        if (i > 0) {
          best_seq_stream << " -> ";
        }
        best_seq_stream << best_action_sequence.sequence[i].who_to_go_to;
      }
      RCLCPP_INFO_STREAM(this->get_logger(), "BEST ACTION SEQUENCE: " << best_seq_stream.str());
      RCLCPP_INFO_STREAM(this->get_logger(), "BEST ACTION SEQUENCE VALUE: " << best_action_sequence.total_value);
    }

  }

  return best_action_sequence;
}

double ForwardSearchSolver::reward_function(State state, Action action)
{
  double beta = this->get_parameter("info_shared_weight").as_double();
  double lambda = this->get_parameter("path_length_weight").as_double();
  double xi = this->get_parameter("time_since_visit_weight").as_double();
  
  GuyState current_guy_state = guy_states_by_id_[state.current_guy];
  GuyState next_guy_state = guy_states_by_id_[action.who_to_go_to];

  double summed_time_since_last_visit = xi*state.relay_time_since_visit.sum();

  double path_length = calculate_path_length(current_guy_state, next_guy_state);

  return beta*state.shared_info_matrix.norm() - lambda*path_length - summed_time_since_last_visit;
}

double ForwardSearchSolver::calculate_path_length(GuyState current_guy_state, GuyState next_guy_state)
{
  Eigen::Vector3d curr_position;
  curr_position << current_guy_state.pose.pose.position.x, current_guy_state.pose.pose.position.y, current_guy_state.pose.pose.position.z;

  Eigen::Vector3d next_position;
  next_position << next_guy_state.pose.pose.position.x, next_guy_state.pose.pose.position.y, next_guy_state.pose.pose.position.z;
  
  double path_length = (curr_position - next_position).norm();

  return path_length;
}

double ForwardSearchSolver::roll_out(State state)
{
  const int num_rollouts = this->get_parameter("num_rollouts").as_int();
  const int roll_out_depth = this->get_parameter("roll_out_depth").as_int();
  const double gamma = this->get_parameter("discount_factor").as_double();
  const double vel = this->get_parameter("agent_velocity").as_double();

  if (num_rollouts <= 0 || roll_out_depth <= 0 || ids_.size() <= 1) {
    return 0.0;
  }

  static thread_local std::mt19937 rng(std::random_device{}());
  double summed_rollout_values = 0.0;

  for (int rollout = 0; rollout < num_rollouts; ++rollout) {
    State rollout_state = state;
    double rollout_value = 0.0;
    double discount = 1.0;

    for (int step = 0; step < roll_out_depth; ++step) {
      std::vector<std::string> candidates;
      candidates.reserve(ids_.size());
      for (const std::string & id : ids_) {
        if (id != rollout_state.current_guy) {
          candidates.push_back(id);
        }
      }
      if (candidates.empty()) {
        break;
      }

      std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
      const std::string & next_id = candidates[dist(rng)];

      Action action;
      action.who_to_go_to = next_id;

      const double reward = reward_function(rollout_state, action);
      rollout_value += discount * reward;
      discount *= gamma;

      const int current_index = index_for_id(rollout_state.current_guy);
      const int next_index = index_for_id(next_id);
      if (current_index < 0 || next_index < 0) {
        break;
      }

      State next_state = rollout_state;
      next_state.current_guy = next_id;
      next_state.relay_info(next_index) = guy_states_by_id_[next_id].bits;
      next_state.shared_info_matrix.row(current_index) = next_state.relay_info.transpose();
      next_state.shared_info_matrix(current_index, current_index) = 0.0;

      const GuyState current_guy_state = guy_states_by_id_[rollout_state.current_guy];
      const GuyState next_guy_state = guy_states_by_id_[next_id];
      const double path_length = calculate_path_length(current_guy_state, next_guy_state);
      const double time_to_take_action = path_length / vel;

      next_state.relay_time_since_visit.array() += time_to_take_action;
      next_state.relay_time_since_visit(next_index) = 0.0;
      next_state.value = reward + gamma * rollout_state.value;

      rollout_state = next_state;
    }

    summed_rollout_values += rollout_value;
  }

  return summed_rollout_values / static_cast<double>(num_rollouts);
}

int ForwardSearchSolver::index_for_id(const std::string & id) const
{
  auto it = std::find(ids_.begin(), ids_.end(), id);
  if (it == ids_.end()) {
    return -1;
  }
  return static_cast<int>(std::distance(ids_.begin(), it));
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

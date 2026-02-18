#include "forward_search_solver.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <random>
#include <sstream>

namespace eyes_on_guys
{

ForwardSearchSolver::ForwardSearchSolver()
: current_config_{}
{}

ForwardSearchSolver::ForwardSearchResult ForwardSearchSolver::solve(
  const ForwardSearchInput & input,
  const ForwardSearchConfig & config)
{
  current_config_ = config;
  guy_states_by_id_ = input.guy_states_by_id;
  ids_ = input.ids;
  if (ids_.empty()) {
    ids_.reserve(guy_states_by_id_.size());
    for (const auto & [id, _] : guy_states_by_id_) {
      (void)_;
      ids_.push_back(id);
    }
  }
  std::sort(ids_.begin(), ids_.end());

  ForwardSearchResult result;
  const int size = static_cast<int>(ids_.size());
  if (size == 0) {
    result.success = false;
    result.message = "No guy poses received.";
    return result;
  }
  if (current_config_.starting_guy < 0 || current_config_.starting_guy >= size) {
    result.success = false;
    result.message = "Starting guy index out of range.";
    return result;
  }

  Eigen::MatrixXd initial_shared_info_matrix = input.shared_info_matrix;
  if (current_config_.shared_info_matrix.rows() == size &&
      current_config_.shared_info_matrix.cols() == size)
  {
    initial_shared_info_matrix = current_config_.shared_info_matrix;
  }

  const std::string & starting_id = ids_.at(static_cast<size_t>(current_config_.starting_guy));
  current_state_ = make_initial_state(
    starting_id,
    current_config_.starting_guy,
    size,
    initial_shared_info_matrix);
  const ActionSequence best_sequence = forward_search(current_config_.depth, current_state_);

  result.success = true;
  result.message = "Search completed.";
  result.total_value = best_sequence.total_value;
  result.sequence_ids.reserve(best_sequence.sequence.size());
  for (const auto & action : best_sequence.sequence) {
    result.sequence_ids.push_back(action.who_to_go_to);
  }
  while (!result.sequence_ids.empty() && result.sequence_ids.front() == starting_id) {
    result.sequence_ids.erase(result.sequence_ids.begin());
  }
  return result;
}

ForwardSearchSolver::ActionSequence ForwardSearchSolver::forward_search(int depth, State state)
{
  if (depth <= 0) {
    ActionSequence action_sequence;
    action_sequence.total_value = roll_out(state);
    return action_sequence;
  }

  const double gamma = current_config_.discount_factor;
  const double vel = current_config_.agent_velocity;

  ActionSequence best_action_sequence;
  best_action_sequence.total_value = -std::numeric_limits<double>::infinity();

  for (const std::string & id : action_candidates(state.current_guy)) {
    Action new_action;
    new_action.who_to_go_to = id;

    double reward = 0.0;
    State new_state;
    if (!apply_action_transition(state, id, gamma, vel, new_state, reward, false)) {
      continue;
    }

    new_action.value = new_state.value;

    ActionSequence action_sequence = forward_search(depth - 1, new_state);
    action_sequence.sequence.insert(action_sequence.sequence.begin(), new_action);
    action_sequence.total_value = reward + gamma * action_sequence.total_value;

    if (action_sequence.total_value > best_action_sequence.total_value) {
      best_action_sequence = action_sequence;
    }
  }

  return best_action_sequence;
}

double ForwardSearchSolver::reward_function(const State & state, const Action & action) const
{
  const double beta = current_config_.info_shared_weight;
  const double lambda = current_config_.path_length_weight;
  const double xi = current_config_.time_since_visit_weight;

  const GuyState & current_guy_state = guy_states_by_id_.at(state.current_guy);
  const GuyState & next_guy_state = guy_states_by_id_.at(action.who_to_go_to);

  const double summed_time_since_last_visit = xi * state.relay_time_since_visit.sum();
  const double path_length = calculate_path_length(current_guy_state, next_guy_state);

  return beta * state.shared_info_matrix.norm() - lambda * path_length - summed_time_since_last_visit;
}

double ForwardSearchSolver::calculate_path_length(
  const GuyState & current_guy_state,
  const GuyState & next_guy_state) const
{
  Eigen::Vector3d curr_position;
  curr_position << current_guy_state.pose.pose.position.x,
    current_guy_state.pose.pose.position.y,
    current_guy_state.pose.pose.position.z;

  Eigen::Vector3d next_position;
  next_position << next_guy_state.pose.pose.position.x,
    next_guy_state.pose.pose.position.y,
    next_guy_state.pose.pose.position.z;

  const double path_length = (curr_position - next_position).norm();

  return path_length;
}

double ForwardSearchSolver::roll_out(const State & state)
{
  const int num_rollouts = current_config_.num_rollouts;
  const int roll_out_depth = current_config_.roll_out_depth;
  const double gamma = current_config_.discount_factor;
  const double vel = current_config_.agent_velocity;

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
      const std::vector<std::string> candidates = action_candidates(rollout_state.current_guy);
      if (candidates.empty()) {
        break;
      }

      std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
      const std::string & next_id = candidates[dist(rng)];

      double reward = 0.0;
      State next_state;
      if (!apply_action_transition(rollout_state, next_id, gamma, vel, next_state, reward, false)) {
        break;
      }
      rollout_value += discount * reward;
      discount *= gamma;

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

std::string ForwardSearchSolver::format_sequence_with_start(
  const std::string & start_id,
  const ActionSequence & sequence) const
{
  std::ostringstream seq_stream;
  seq_stream << start_id;
  for (const auto & action : sequence.sequence) {
    seq_stream << " -> " << action.who_to_go_to;
  }
  return seq_stream.str();
}

std::vector<std::string> ForwardSearchSolver::action_candidates(const std::string & current_guy) const
{
  std::vector<std::string> candidates;
  candidates.reserve(ids_.size());
  for (const std::string & id : ids_) {
    if (id != current_guy) {
      candidates.push_back(id);
    }
  }
  return candidates;
}

ForwardSearchSolver::State ForwardSearchSolver::make_initial_state(
  const std::string & starting_id,
  int starting_index,
  int size,
  const Eigen::MatrixXd & initial_shared_info_matrix) const
{
  State state;
  state.current_guy = starting_id;
  if (initial_shared_info_matrix.rows() == size && initial_shared_info_matrix.cols() == size) {
    state.shared_info_matrix = initial_shared_info_matrix;
  } else {
    state.shared_info_matrix = Eigen::MatrixXd::Zero(size, size);
  }
  state.relay_info = Eigen::VectorXd::Zero(size);
  state.relay_info(starting_index) = guy_states_by_id_.at(starting_id).bits;

  state.relay_time_since_visit = Eigen::VectorXd::Constant(size, 1'000.0);
  state.relay_time_since_visit(starting_index) = 0.0;

  state.guy_bits = Eigen::VectorXd::Zero(size);
  state.guy_bits_rate = Eigen::VectorXd::Ones(size);
  for (int i = 0; i < size; ++i) {
    const std::string & id = ids_.at(static_cast<size_t>(i));
    const GuyState & guy_state = guy_states_by_id_.at(id);
    state.guy_bits(i) = guy_state.bits;
    state.guy_bits_rate(i) = guy_state.bits_rate;
  }

  state.value = 0.0;
  return state;
}

bool ForwardSearchSolver::apply_action_transition(
  const State & state,
  const std::string & next_id,
  double gamma,
  double vel,
  State & next_state,
  double & reward,
  bool emit_debug_logs) const
{
  (void)emit_debug_logs;

  const int current_index = index_for_id(state.current_guy);
  const int next_index = index_for_id(next_id);
  if (current_index < 0 || next_index < 0) {
    return false;
  }

  next_state = state;
  next_state.current_guy = next_id;

  const GuyState & current_guy_state = guy_states_by_id_.at(state.current_guy);
  const GuyState & next_guy_state = guy_states_by_id_.at(next_id);
  const double path_length = calculate_path_length(current_guy_state, next_guy_state);
  const double time_to_take_action = path_length / vel;

  next_state.relay_time_since_visit.array() += time_to_take_action;
  next_state.relay_time_since_visit(next_index) = 0.0;
  next_state.guy_bits.array() += next_state.guy_bits_rate.array() * time_to_take_action;

  next_state.relay_info(next_index) = next_state.guy_bits(next_index);
  next_state.shared_info_matrix.row(current_index) = next_state.relay_info.transpose();
  next_state.shared_info_matrix(current_index, current_index) = 0.0;

  const Action action{next_id, 0.0};
  reward = reward_function(state, action);
  next_state.value = reward + gamma * state.value;
  return true;
}

} // namespace eyes_on_guys

#include "forward_search_solver.hpp"

#include <algorithm>
#include <limits>
#include <random>
#include <utility>

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

  id_to_index_.clear();
  id_to_index_.reserve(static_cast<size_t>(size));
  for (int i = 0; i < size; ++i) {
    id_to_index_[ids_.at(static_cast<size_t>(i))] = i;
  }

  candidates_by_index_.assign(static_cast<size_t>(size), {});
  for (int i = 0; i < size; ++i) {
    std::vector<int> candidates;
    candidates.reserve(static_cast<size_t>(size - 1));
    for (int j = 0; j < size; ++j) {
      if (j != i) {
        candidates.push_back(j);
      }
    }
    candidates_by_index_.at(static_cast<size_t>(i)) = std::move(candidates);
  }

  cached_path_lengths_ = Eigen::MatrixXd::Zero(size, size);
  for (int i = 0; i < size; ++i) {
    const GuyState & from = guy_states_by_id_.at(ids_.at(static_cast<size_t>(i)));
    for (int j = i + 1; j < size; ++j) {
      const GuyState & to = guy_states_by_id_.at(ids_.at(static_cast<size_t>(j)));
      const double distance = calculate_path_length(from, to);
      cached_path_lengths_(i, j) = distance;
      cached_path_lengths_(j, i) = distance;
    }
  }

  Eigen::MatrixXd initial_shared_info_matrix = input.shared_info_matrix;
  if (current_config_.shared_info_matrix.rows() == size &&
      current_config_.shared_info_matrix.cols() == size)
  {
    initial_shared_info_matrix = current_config_.shared_info_matrix;
  }

  current_state_ = make_initial_state(
    current_config_.starting_guy,
    size,
    initial_shared_info_matrix);
  ActionSequence best_sequence = forward_search(current_config_.depth, current_state_);

  result.success = true;
  result.message = "Search completed.";
  result.total_value = best_sequence.total_value;
  result.sequence_ids.reserve(best_sequence.sequence.size());
  for (const auto & action : best_sequence.sequence) {
    result.sequence_ids.push_back(ids_.at(static_cast<size_t>(action.next_index)));
  }
  return result;
}

ForwardSearchSolver::ActionSequence ForwardSearchSolver::forward_search(int depth, State & state)
{
  if (depth <= 0) {
    ActionSequence action_sequence;
    action_sequence.total_value = roll_out(state);
    return action_sequence;
  }

  const double gamma = current_config_.discount_factor;
  const double vel = current_config_.agent_velocity;

  ActionSequence best_action_sequence;
  const int current_index = state.current_index;
  const std::vector<int> & candidates = candidates_by_index_.at(static_cast<size_t>(current_index));
  Eigen::RowVectorXd previous_shared_row(state.shared_info_matrix.cols());

  for (const int next_index : candidates) {
    const int previous_current_index = state.current_index;
    const double previous_value = state.value;
    const double previous_next_time = state.relay_time_since_visit(next_index);
    const double previous_relay_info_next = state.relay_info(next_index);
    previous_shared_row = state.shared_info_matrix.row(previous_current_index);

    double reward = 0.0;
    double transition_time = 0.0;
    if (!apply_action_transition(state, next_index, gamma, vel, reward, transition_time, false)) {
      continue;
    }

    ActionSequence child_sequence = forward_search(depth - 1, state);
    ActionSequence candidate_sequence;
    candidate_sequence.total_value = reward + gamma * child_sequence.total_value;
    candidate_sequence.sequence.reserve(child_sequence.sequence.size() + 1);
    candidate_sequence.sequence.push_back(Action{next_index, state.value});
    candidate_sequence.sequence.insert(
      candidate_sequence.sequence.end(),
      child_sequence.sequence.begin(),
      child_sequence.sequence.end());

    if (candidate_sequence.total_value > best_action_sequence.total_value) {
      best_action_sequence = std::move(candidate_sequence);
    }

    state.current_index = previous_current_index;
    state.value = previous_value;
    state.relay_time_since_visit.array() -= transition_time;
    state.relay_time_since_visit(next_index) = previous_next_time;
    state.guy_bits.array() -= state.guy_bits_rate.array() * transition_time;
    state.relay_info(next_index) = previous_relay_info_next;
    state.shared_info_matrix.row(previous_current_index) = previous_shared_row;
  }

  return best_action_sequence;
}

double ForwardSearchSolver::reward_function(
  const State & state,
  int current_index,
  int next_index,
  double path_length) const
{
  (void)current_index;
  (void)next_index;
  const double beta = current_config_.info_shared_weight;
  const double lambda = current_config_.path_length_weight;
  const double xi = current_config_.time_since_visit_weight;

  double info_matrix_reward = (state.relay_info.transpose() - state.shared_info_matrix.row(next_index)).norm();

  const double summed_time_since_last_visit = xi * state.relay_time_since_visit.sum();
  return beta * info_matrix_reward - lambda * path_length - summed_time_since_last_visit;
}

double ForwardSearchSolver::calculate_path_length(
  const GuyState & current_guy_state,
  const GuyState & next_guy_state)
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
      const std::vector<int> & candidates =
        candidates_by_index_.at(static_cast<size_t>(rollout_state.current_index));
      if (candidates.empty()) {
        break;
      }

      std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
      const int next_index = candidates[dist(rng)];

      double reward = 0.0;
      double transition_time = 0.0;
      if (!apply_action_transition(
          rollout_state, next_index, gamma, vel, reward, transition_time, false))
      {
        break;
      }
      rollout_value += discount * reward;
      discount *= gamma;
    }

    summed_rollout_values += rollout_value;
  }

  return summed_rollout_values / static_cast<double>(num_rollouts);
}

ForwardSearchSolver::State ForwardSearchSolver::make_initial_state(
  int starting_index,
  int size,
  const Eigen::MatrixXd & initial_shared_info_matrix) const
{
  State state;
  state.current_index = starting_index;
  if (initial_shared_info_matrix.rows() == size && initial_shared_info_matrix.cols() == size) {
    state.shared_info_matrix = initial_shared_info_matrix;
  } else {
    state.shared_info_matrix = Eigen::MatrixXd::Zero(size, size);
  }
  state.relay_info = Eigen::VectorXd::Zero(size);
  state.relay_info(starting_index) =
    guy_states_by_id_.at(ids_.at(static_cast<size_t>(starting_index))).bits;

  state.relay_time_since_visit = Eigen::VectorXd::Constant(size, 0.0);
  state.relay_time_since_visit(starting_index) = 0.0;

  state.guy_bits = Eigen::VectorXd::Zero(size);
  state.guy_bits_rate = Eigen::VectorXd::Ones(size);
  for (int i = 0; i < size; ++i) {
    const std::string & id = ids_.at(static_cast<size_t>(i));
    const GuyState & guy_state = guy_states_by_id_.at(id);
    state.guy_bits(i) = guy_state.bits;
    state.guy_bits_rate(i) = guy_state.bits_rate;
  }
  return state;
}

bool ForwardSearchSolver::apply_action_transition(
  State & state,
  int next_index,
  double gamma,
  double vel,
  double & reward,
  double & transition_time,
  bool emit_debug_logs) const
{
  (void)emit_debug_logs;

  const int current_index = state.current_index;
  if (current_index < 0 || next_index < 0 ||
      current_index >= cached_path_lengths_.rows() ||
      next_index >= cached_path_lengths_.cols() ||
      current_index == next_index)
  {
    return false;
  }

  const double path_length = cached_path_lengths_(current_index, next_index);
  transition_time = path_length / vel;
  reward = reward_function(state, current_index, next_index, path_length);

  state.relay_time_since_visit.array() += transition_time;
  state.relay_time_since_visit(next_index) = 0.0;
  state.guy_bits.array() += state.guy_bits_rate.array() * transition_time;

  state.relay_info(next_index) = state.guy_bits(next_index);
  state.shared_info_matrix.row(current_index) = state.relay_info.transpose();
  state.shared_info_matrix(current_index, current_index) = 0.0;
  state.value = reward + gamma * state.value;
  state.current_index = next_index;
  return true;
}

} // namespace eyes_on_guys

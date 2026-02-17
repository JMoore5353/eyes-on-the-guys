#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "branch_and_bound_solver.hpp"
#include "eyes_on_guys_problem.hpp"

namespace eyes_on_guys
{

BranchAndBoundSolver::BranchAndBoundSolver(int num_states, int max_depth, int max_iterations,
                                           float discount_factor, bool debug_mode)
    : num_states_{num_states}
    , max_depth_{max_depth}
    , max_iterations_{max_iterations}
    , discount_factor_{std::max(discount_factor, 0.0F)}
    , debug_mode_{debug_mode}
    , next_node_id_{0U}
    , explored_nodes_count_{0U}
    , completed_paths_count_{0U}
    , total_pruned_nodes_{0U}
    , best_reward_{std::numeric_limits<float>::lowest()}
{}

// Descending u_max ordering: begin() is the most promising unexplored node.
bool BranchAndBoundSolver::UMaxComparator::operator()(const NodePtr & lhs,
                                                      const NodePtr & rhs) const
{
  if (lhs->u_max != rhs->u_max) {
    return lhs->u_max > rhs->u_max;
  }
  return lhs->id < rhs->id;
}

// Lower-bound estimate from a state.
float BranchAndBoundSolver::u_min(int curr_state, const EyesOnGuysProblem & problem_state,
                                  float path_reward, int depth) const
{
  return path_reward;
}

// Upper-bound estimate from a state.
float BranchAndBoundSolver::u_max(int curr_state, const EyesOnGuysProblem & problem_state,
                                  float path_reward, int depth) const
{
  return 10000000.0F;
}

// Create a node with computed bounds and assign a unique id used for tie-breaking.
BranchAndBoundSolver::NodePtr BranchAndBoundSolver::make_node(
  int curr_state, int depth, const std::vector<int> & path, float reward,
  const EyesOnGuysProblem & problem_state)
{
  NodePtr out = std::make_shared<Node>(
    u_min(curr_state, problem_state, reward, depth),
    u_max(curr_state, problem_state, reward, depth), path, reward, curr_state, depth,
    problem_state, next_node_id_);

  ++next_node_id_;

  return out;
}

// Insert node into the unexplored u_max index.
void BranchAndBoundSolver::add_unexplored_node(const NodePtr & node)
{
  if (node == nullptr) {
    return;
  }

  unexplored_nodes_by_umax_.insert(node);
}

// Remove node from the unexplored u_max index.
void BranchAndBoundSolver::erase_unexplored_node(const NodePtr & node)
{
  if (node == nullptr) {
    return;
  }

  auto u_max_it = unexplored_nodes_by_umax_.find(node);
  if (u_max_it != unexplored_nodes_by_umax_.end()) {
    unexplored_nodes_by_umax_.erase(u_max_it);
  }
}

// Prune unexplored nodes that cannot beat the incumbent threshold.
std::size_t BranchAndBoundSolver::prune_nodes_with_u_max_below(float threshold)
{
  std::size_t pruned_count{0U};

  while (!unexplored_nodes_by_umax_.empty()) {
    auto worst_it = std::prev(unexplored_nodes_by_umax_.end());
    if ((*worst_it)->u_max > threshold) {
      break;
    }

    unexplored_nodes_by_umax_.erase(worst_it);
    ++pruned_count;
  }

  total_pruned_nodes_ += pruned_count;
  return pruned_count;
}

// Return and remove the node with highest optimistic bound.
BranchAndBoundSolver::NodePtr BranchAndBoundSolver::pop_node_with_highest_u_max()
{
  if (unexplored_nodes_by_umax_.empty()) {
    return nullptr;
  }

  auto best_it = unexplored_nodes_by_umax_.begin();
  NodePtr best_node = *best_it;
  erase_unexplored_node(best_node);

  return best_node;
}

// Validate that the problem tensors match the configured number of states.
bool BranchAndBoundSolver::problem_dimensions_are_valid(const EyesOnGuysProblem & problem_info) const
{
  const int state_size = num_states_;

  return problem_info.relays_current_info.size() == state_size
    && problem_info.time_since_last_relay_contact_with_agent.size() == state_size
    && problem_info.shared_info_matrix.rows() == state_size
    && problem_info.shared_info_matrix.cols() == state_size
    && problem_info.distance_between_agents.rows() == state_size
    && problem_info.distance_between_agents.cols() == state_size;
}

// Update incumbent best solution if this node improves total reward.
void BranchAndBoundSolver::maybe_update_best_solution(const NodePtr & node)
{
  if (node == nullptr || node->path.empty()) {
    return;
  }

  if (node->reward > best_reward_) {
    best_reward_ = node->reward;
    best_path_ = node->path;
  }
}

// Optional per-iteration diagnostics to understand search progress.
void BranchAndBoundSolver::maybe_print_debug_info(int iteration,
                                                  std::size_t pruned_this_iteration) const
{
  if (!debug_mode_) {
    return;
  }

  std::cout << "[BranchAndBoundSolver] iter=" << iteration
            << " pruned_nodes=" << pruned_this_iteration
            << " completed_paths=" << completed_paths_count_ << std::endl;
}

// Main branch-and-bound loop.
std::vector<int> BranchAndBoundSolver::solve(int initial_state, const EyesOnGuysProblem & problem_info)
{
  add_unexplored_node(make_node(initial_state, 0, {initial_state}, 0.0F, problem_info));

  while (!unexplored_nodes_by_umax_.empty()) {
    NodePtr max_node = pop_node_with_highest_u_max();

    if (max_node->depth == max_depth_) {
      completed_paths_count_++;
      maybe_update_best_solution(max_node);

    } else {
      for (int next_state = 0; next_state < num_states_; ++next_state) {
        auto new_path = max_node->path;
        new_path.push_back(next_state);
        const auto new_problem = max_node->problem.create_child_eyes_on_guys_state(max_node->state, next_state);
        float new_reward = max_node->reward + compute_reward_model(max_node->state, next_state, max_node->problem, new_problem);

        add_unexplored_node(make_node(next_state, max_node->depth + 1, new_path, new_reward, new_problem));
      }
    }

    prune_nodes_with_u_max_below(best_reward_);
  }

  return best_path_;
}

} // namespace eyes_on_guys

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "branch_and_bound_solver.hpp"
#include "eyes_on_guys_problem.hpp"

namespace eyes_on_guys
{

BranchAndBoundSolver::BranchAndBoundSolver(int num_states, int max_depth, int max_iterations,
                                           double discount_factor, bool debug_mode,
                                           bool enable_pruning)
    : num_states_{num_states}
    , max_depth_{max_depth}
    , max_iterations_{max_iterations}
    , discount_factor_{std::max(discount_factor, 0.0)}
    , debug_mode_{debug_mode}
    , enable_pruning_{enable_pruning}
    , next_node_id_{0U}
    , explored_nodes_count_{0U}
    , completed_paths_count_{0U}
    , total_pruned_nodes_{0U}
    , best_reward_{std::numeric_limits<double>::lowest()}
{}

// Descending q_max ordering: begin() is the most promising unexplored node.
bool BranchAndBoundSolver::QMaxComparator::operator()(const NodePtr & lhs,
                                                      const NodePtr & rhs) const
{
  if (lhs->q_max != rhs->q_max) {
    return lhs->q_max > rhs->q_max;
  }
  return lhs->id < rhs->id;
}

// Upper-bound estimate from a state.
// Currently finds upper bound by finding the longest path from the current state without revising
// nodes (unless required) and finds reward of path without time/dist penalty.
double BranchAndBoundSolver::q_max(int state, std::vector<int> path,
                                   const EyesOnGuysProblem & problem_state,
                                   double path_reward, int depth) const
{
  int curr_state = state;

  // Get anti-greedy path
  auto ag_path = anti_greedy_path(curr_state, path, problem_state, depth);

  // Calculate reward of anti-greedy path, without time/dist penalty
  double reward = 0.0;
  EyesOnGuysProblem current_problem_state = problem_state;
  int curr_depth = depth;
  for (int next_state : ag_path) {
    EyesOnGuysProblem next_problem_state =
      current_problem_state.create_child_eyes_on_guys_state(curr_state, next_state);
    reward += std::pow(discount_factor_, curr_depth) *
      compute_reward_model(curr_state, next_state, current_problem_state, next_problem_state, false);

    curr_state = next_state;
    current_problem_state = next_problem_state;
    curr_depth++;
  }

  // Add to current path reward
  return path_reward + reward;
}

// Determines the longest path from the current node till search depth
// Avoid repeating nodes until necessary
std::vector<int> BranchAndBoundSolver::anti_greedy_path(
  int curr_state,
  std::vector<int> current_path,
  const EyesOnGuysProblem & problem_state,
  int depth) const
{
  // Get current distance matrix and path
  auto dist_matrix = problem_state.distance_between_agents;
  int curr_depth = depth;
  std::vector<int> result_path;

  // Zero out rows of visited states so they won't be selected when searching columns
  for (int state : current_path) {
    dist_matrix.row(state).setZero();
  }

  while (curr_depth < max_depth_) {
    // If matrix is all zeros, reset back to original
    if (dist_matrix.sum() == 0.0) {
      dist_matrix = problem_state.distance_between_agents;
    }
    
    // Find the index of the largest value in the column of the current state
    int max_idx;
    dist_matrix.col(curr_state).maxCoeff(&max_idx);

    // Move to this state and add it to the path
    result_path.push_back(max_idx);
    curr_state = max_idx;
    curr_depth++;

    // Zero out row of new current state to prevent revisiting
    dist_matrix.row(curr_state).setZero();
  }
  
  return result_path;
}

// Create a node with computed bounds and assign a unique id used for tie-breaking.
BranchAndBoundSolver::NodePtr BranchAndBoundSolver::make_node(
  int depth, std::vector<int> path, double reward,
  EyesOnGuysProblem problem_state)
{
  int curr_state = path.back();
  double upper = q_max(curr_state, path, problem_state, reward, depth);

  NodePtr out = std::make_shared<Node>(
    upper, std::move(path), reward, depth,
    std::move(problem_state), next_node_id_);

  ++next_node_id_;

  return out;
}

// Insert node into the unexplored q_max index.
void BranchAndBoundSolver::add_unexplored_node(const NodePtr & node)
{
  if (node == nullptr) {
    return;
  }

  unexplored_nodes_.insert(node);
}

// Remove node from the unexplored q_max index.
void BranchAndBoundSolver::erase_unexplored_node(const NodePtr & node)
{
  if (node == nullptr) {
    return;
  }

  auto q_max_it = unexplored_nodes_.find(node);
  if (q_max_it != unexplored_nodes_.end()) {
    unexplored_nodes_.erase(q_max_it);
  }
}

// Prune all unexplored nodes whose q_max cannot beat the incumbent best reward.
std::size_t BranchAndBoundSolver::prune_nodes()
{
  std::size_t pruned_count{0U};

  auto first_prunable = std::find_if(unexplored_nodes_.begin(), unexplored_nodes_.end(),
                                     [this](const NodePtr& node) { return node->q_max <= best_reward_; });

  pruned_count = std::distance(first_prunable, unexplored_nodes_.end());
  unexplored_nodes_.erase(first_prunable, unexplored_nodes_.end());

  return pruned_count;
}

// Return and remove the node with highest optimistic bound.
BranchAndBoundSolver::NodePtr BranchAndBoundSolver::pop_node_with_highest_q_max()
{
  if (unexplored_nodes_.empty()) {
    return nullptr;
  }

  auto best_it = unexplored_nodes_.begin();
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
void BranchAndBoundSolver::maybe_print_debug_info() const
{
  if (!debug_mode_) {
    return;
  }

  std::cout << "[BranchAndBoundSolver] explored_nodes=" << explored_nodes_count_
            << " pruned_nodes=" << total_pruned_nodes_
            << " completed_paths=" << completed_paths_count_ << std::endl;
}

// Reset all mutable search state so the solver can be reused across calls.
void BranchAndBoundSolver::reset()
{
  unexplored_nodes_.clear();
  next_node_id_ = 0U;
  explored_nodes_count_ = 0U;
  completed_paths_count_ = 0U;
  total_pruned_nodes_ = 0U;
  best_reward_ = std::numeric_limits<double>::lowest();
  best_path_.clear();
}

// Main branch-and-bound loop.
std::vector<int> BranchAndBoundSolver::solve(int initial_state, const EyesOnGuysProblem & problem_info)
{
  // Clear any previous search state
  reset();

  // Validate that the problem is properly configured
  if (!problem_dimensions_are_valid(problem_info)) {
    return {};
  }

  // Validate that initial_state is within bounds
  if (initial_state < 0 || initial_state >= num_states_) {
    return {};
  }

  // Initialize the search with the initial state
  add_unexplored_node(make_node(0, {initial_state}, 0.0, problem_info));

  int iteration = 0;
  while (!unexplored_nodes_.empty() && iteration < max_iterations_) {
    // Explore the current best option
    NodePtr max_node = pop_node_with_highest_q_max();
    ++explored_nodes_count_;

    // Check if this is a complete path
    if (max_node->depth == max_depth_) {
      ++completed_paths_count_;
      maybe_update_best_solution(max_node);

    } else {
      // Generate child nodes
      for (int next_state = 0; next_state < num_states_; ++next_state) {
        // Avoid revisiting current state
        if (next_state == max_node->path.back()) {
          continue;
        }

        auto new_path = max_node->path;
        new_path.push_back(next_state);
        int curr_state = max_node->path.back();
        auto new_problem = max_node->problem.create_child_eyes_on_guys_state(curr_state, next_state);

        // Compute the reward accumulated thus far with the path
        double step_reward =
          compute_reward_model(curr_state, next_state, max_node->problem, new_problem);
        double new_reward = max_node->reward + std::pow(discount_factor_, max_node->depth) * step_reward;

        // Add the new node to the unexplored set
        add_unexplored_node(
          make_node(max_node->depth + 1, std::move(new_path), new_reward,
                    std::move(new_problem)));
      }
    }

    // Prune nodes that cannot improve the best solution
    if (enable_pruning_) {
      total_pruned_nodes_ += prune_nodes();
    }
    maybe_print_debug_info();
    ++iteration;
  }

  return best_path_;
}

} // namespace eyes_on_guys

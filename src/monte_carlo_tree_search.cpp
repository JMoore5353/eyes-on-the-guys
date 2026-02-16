#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "monte_carlo_node.hpp"
#include "monte_carlo_tree_search.hpp"

namespace eyes_on_guys
{

MonteCarloTreeSearch::MonteCarloTreeSearch(const int num_states)
    : num_states_{num_states}
    , initial_node_{nullptr}
{}

int MonteCarloTreeSearch::search_for_best_action(const int initial_state, const int num_iter,
                                                 const int depth, const double discount_factor,
                                                 const double exploration_bonus,
                                                 const EyesOnGuysProblem & problem_info,
                                                 const int lookahead_depth,
                                                 const int lookahead_iters)
{
  if (initial_state >= num_states_ || initial_state < 0) {
    return -1;
  }

  initial_node_ =
    std::make_shared<MCTSNode>(initial_state, num_states_, exploration_bonus, problem_info);
  for (int i = 0; i < num_iter; ++i) {
    simulate(initial_node_, depth, discount_factor, lookahead_depth, lookahead_iters);
  }
  return find_greedy_action(initial_node_);
}

double MonteCarloTreeSearch::simulate(std::shared_ptr<MCTSNode> curr_state, const int depth,
                                      const double discount_factor, const int lookahead_depth,
                                      const int lookahead_iters)
{
  if (depth <= 0) {
    return lookahead_value_function_estimate(curr_state, num_states_, lookahead_depth,
                                             discount_factor, lookahead_iters);
  }

  if (!curr_state->has_been_visited()) {
    curr_state->visit_node();
    return lookahead_value_function_estimate(curr_state, num_states_, lookahead_depth,
                                             discount_factor, lookahead_iters);
  }

  int best_action = curr_state->explore_best_action();

  std::shared_ptr<MCTSNode> next_state = transition_from_state(curr_state, best_action);
  double r = compute_reward_from_transitioning(curr_state, next_state);

  double q = r
    + discount_factor
      * simulate(next_state, depth - 1, discount_factor, lookahead_depth, lookahead_iters);

  curr_state->update_count_and_action_value_function(best_action, q);

  return q;
}

std::vector<int> MonteCarloTreeSearch::get_greedy_sequence(bool print_info_matrices) const
{
  if (initial_node_ == nullptr) {
    return std::vector<int>();
  }

  std::vector<int> out;
  std::shared_ptr<const MCTSNode> curr_node = initial_node_;
  while (curr_node != nullptr) {
    int best_action = find_greedy_action(curr_node);
    curr_node = curr_node->get_child(best_action);
    if (curr_node != nullptr) {
      out.push_back(best_action);

      if (print_info_matrices) {
        std::cout << "Action: " << best_action << std::endl;
        std::cout << curr_node->get_ref_to_problem_info().shared_info_matrix << std::endl;
      }
    }
  }

  return out;
}

double lookahead_value_function_estimate(const std::shared_ptr<const MCTSNode> curr_state,
                                         const int num_states, const int depth,
                                         const double discount, const int num_iters)
{
  if (num_iters == 0) {
    return 0;
  }

  double average_reward{0.0};
  for (int j = 0; j < num_iters; ++j) {
    auto start_state = std::make_shared<MCTSNode>(*curr_state);

    for (int i = 0; i < depth; ++i) {
      int action = lookahead_get_random_action(start_state->get_id(), num_states);
      std::shared_ptr<MCTSNode> new_state = transition_from_state(start_state, action);
      average_reward +=
        std::pow(discount, i) * compute_reward_from_transitioning(start_state, new_state);
      start_state = new_state;
    }
  }
  return average_reward / num_iters;
}

int lookahead_get_random_action(const int curr_state, const int num_states)
{
  if (num_states <= 1) {
    return curr_state;
  }

  static std::random_device rd;
  static std::mt19937 generator(rd());
  std::uniform_int_distribution<> distr(0, num_states - 1);
  int action{curr_state};
  while (action == curr_state) {
    action = distr(generator);
  }
  return action;
}

std::shared_ptr<MCTSNode> transition_from_state(std::shared_ptr<MCTSNode> curr_node,
                                                const int action)
{
  return curr_node->take_action(action);
}

double compute_reward_from_transitioning(const std::shared_ptr<const MCTSNode> curr_state,
                                         const std::shared_ptr<const MCTSNode> next_state)
{
  int curr_state_id{curr_state->get_id()};
  int next_state_id{next_state->get_id()};
  const EyesOnGuysProblem & curr_state_problem_info{curr_state->get_ref_to_problem_info()};
  const EyesOnGuysProblem & next_state_problem_info{next_state->get_ref_to_problem_info()};
  return compute_reward_model(curr_state_id, next_state_id, curr_state_problem_info,
                              next_state_problem_info);
}

int find_greedy_action(const std::shared_ptr<const MCTSNode> curr_state)
{
  // TODO: Replace this with a call to get_greedy_sequence...
  return curr_state->get_greedy_action();
}

} // namespace eyes_on_guys

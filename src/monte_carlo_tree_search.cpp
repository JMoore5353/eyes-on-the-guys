#include <Eigen/Core>
#include <memory>

#include "monte_carlo_node.hpp"
#include "monte_carlo_tree_search.hpp"

namespace eyes_on_guys
{

MonteCarloTreeSearch::MonteCarloTreeSearch(const int num_states, const int num_actions,
                                           const double discount_factor)
    : num_states_{num_states}
    , num_actions_{num_actions}
    , discount_factor_{discount_factor}
{}

int MonteCarloTreeSearch::search_for_best_action(const int initial_state, const uint num_iter,
                                                 const uint depth, const double exploration_bonus,
                                                 const EyesOnGuysProblem & problem_info)
{
  if (initial_state >= num_states_) {
    return -1;
  }

  auto initial_node =
    std::make_shared<MTCSNode>(initial_state, num_actions_, exploration_bonus, problem_info);
  for (uint i = 0; i < num_iter; ++i) {
    simulate(initial_node, depth);
  }
  return find_greedy_action(initial_node);
}

double MonteCarloTreeSearch::simulate(std::shared_ptr<MTCSNode> curr_state, const uint & depth)
{
  if (depth <= 0) {
    return lookahead_value_function_estimate(curr_state);
  }

  if (!curr_state->has_been_visited()) {
    return lookahead_value_function_estimate(curr_state);
  }

  int best_action = curr_state->explore_best_action();

  std::shared_ptr<MTCSNode> next_state = transition_from_state(curr_state, best_action);
  double r = compute_reward_from_transitioning(curr_state, next_state);

  double q = r + discount_factor_ * simulate(next_state, depth - 1);

  curr_state->update_count_and_action_value_function(best_action, q);

  return q;
}

double lookahead_value_function_estimate(const std::shared_ptr<MTCSNode> curr_state)
{
  // TODO:
  return 0;
}

std::shared_ptr<MTCSNode> transition_from_state(std::shared_ptr<MTCSNode> curr_node,
                                                const int action)
{
  return curr_node->take_action(action);
}

double compute_reward_from_transitioning(const std::shared_ptr<MTCSNode> curr_state,
                                         const std::shared_ptr<MTCSNode> next_state)
{
  // TODO: Continue here... This function should return R = ||I||_F ...
  return 0;
}

int find_greedy_action(const std::shared_ptr<MTCSNode> curr_state)
{
  return curr_state->explore_best_action();
}

} // namespace eyes_on_guys

#ifndef MONTE_CARLO_TREE_SEARCH_HPP
#define MONTE_CARLO_TREE_SEARCH_HPP

#include <Eigen/Core>
#include <memory>

#include "eyes_on_guys_problem.hpp"
#include "monte_carlo_node.hpp"

namespace eyes_on_guys
{

class MonteCarloTreeSearch
{
public:
  MonteCarloTreeSearch(const int num_states, const int num_actions, const double discount_factor_);

  int search_for_best_action(const int initial_state, const uint num_iter, const uint depth,
                             const double exploration_bonus,
                             const EyesOnGuysProblem & problem_info);

private:
  int num_states_;
  int num_actions_;
  double discount_factor_;

  double simulate(std::shared_ptr<MTCSNode> curr_state, const uint & depth);
  // TODO: Might be worth it to get the greedy sequence... Would be interesting to plot how it changes over time.
};

double lookahead_value_function_estimate(const std::shared_ptr<MTCSNode> curr_state);
std::shared_ptr<MTCSNode> transition_from_state(std::shared_ptr<MTCSNode> curr_node,
                                                const int action);
double compute_reward_from_transitioning(const std::shared_ptr<MTCSNode> curr_state,
                                         const std::shared_ptr<MTCSNode> next_state);
int find_greedy_action(const std::shared_ptr<MTCSNode> curr_state);

} // namespace eyes_on_guys

#endif

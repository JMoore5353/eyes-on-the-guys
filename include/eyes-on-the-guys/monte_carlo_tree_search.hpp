#ifndef MONTE_CARLO_TREE_SEARCH_HPP
#define MONTE_CARLO_TREE_SEARCH_HPP

#include <Eigen/Core>
#include <memory>

#include "monte_carlo_node.hpp"

namespace eyes_on_guys
{

class MonteCarloTreeSearch
{
public:
  MonteCarloTreeSearch(const int num_states, const int num_actions, const double discount_factor_);

  int monte_carlo_tree_search(const int initial_state, const uint num_iter, const uint depth,
                              const double exploration_bonus);

private:
  int num_states_;
  int num_actions_;
  double discount_factor_;

  double simulate(std::shared_ptr<Node> curr_state, const uint & depth);
};

double compute_value_function_estimate(const std::shared_ptr<Node> curr_state);
std::shared_ptr<Node> transition_from_state(std::shared_ptr<Node> curr_node, const int action);
double compute_reward_from_transitioning(const std::shared_ptr<Node> curr_state,
                                         const std::shared_ptr<Node> next_state);
int find_greedy_action(const std::shared_ptr<Node> curr_state);

} // namespace eyes_on_guys

#endif

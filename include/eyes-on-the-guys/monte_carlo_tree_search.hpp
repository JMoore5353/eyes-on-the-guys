#ifndef MONTE_CARLO_TREE_SEARCH_HPP
#define MONTE_CARLO_TREE_SEARCH_HPP

#include <Eigen/Core>
#include <memory>
#include <random>

#include "eyes_on_guys_problem.hpp"
#include "monte_carlo_node.hpp"

namespace eyes_on_guys
{

class MonteCarloTreeSearch
{
public:
  MonteCarloTreeSearch(const int num_states);

  int search_for_best_action(const int initial_state, const int num_iter, const int depth,
                             const double discount_factor, const double exploration_bonus,
                             const EyesOnGuysProblem & problem_info, const int lookahead_depth = 5,
                             const int lookahead_iters = 10);

private:
  int num_states_;

  double simulate(std::shared_ptr<MCTSNode> curr_state, const int depth,
                  const double discount_factor, const int lookahead_depth,
                  const int lookahead_iters);
  // TODO: Might be worth it to get the greedy sequence... Would be interesting to plot how it changes over time.
};

double lookahead_value_function_estimate(const std::shared_ptr<MCTSNode> curr_state,
                                         const int num_states, const int depth,
                                         const double discount, const int num_iters);
int lookahead_get_random_action(const int curr_state, const int num_states);
std::shared_ptr<MCTSNode> transition_from_state(std::shared_ptr<MCTSNode> curr_node,
                                                const int action);
double compute_reward_from_transitioning(const std::shared_ptr<MCTSNode> curr_state,
                                         const std::shared_ptr<MCTSNode> next_state);
int find_greedy_action(const std::shared_ptr<MCTSNode> curr_state);

} // namespace eyes_on_guys

#endif

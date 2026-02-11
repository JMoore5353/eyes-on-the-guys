#ifndef MONTE_CARLO_TREE_SEARCH_HPP
#define MONTE_CARLO_TREE_SEARCH_HPP

#include <Eigen/Core>
#include <memory>
#include <vector>

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

  // TODO: Could you use this greedy sequence to get a bunch of the top options and compare to make sure that the
  // algorithm is returning the best option?
  std::vector<int> get_greedy_sequence(bool print_info_matrices = false) const;

private:
  int num_states_;
  std::shared_ptr<MCTSNode> initial_node_;

  double simulate(std::shared_ptr<MCTSNode> curr_state, const int depth,
                  const double discount_factor, const int lookahead_depth,
                  const int lookahead_iters);
};

double lookahead_value_function_estimate(const std::shared_ptr<const MCTSNode> curr_state,
                                         const int num_states, const int depth,
                                         const double discount, const int num_iters);
int lookahead_get_random_action(const int curr_state, const int num_states);
std::shared_ptr<MCTSNode> transition_from_state(std::shared_ptr<MCTSNode> curr_node,
                                                const int action);
double compute_reward_from_transitioning(const std::shared_ptr<const MCTSNode> curr_state,
                                         const std::shared_ptr<const MCTSNode> next_state);
int find_greedy_action(const std::shared_ptr<const MCTSNode> curr_state);

} // namespace eyes_on_guys

#endif

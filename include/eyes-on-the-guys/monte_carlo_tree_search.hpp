#ifndef MONTE_CARLO_TREE_SEARCH_HPP
#define MONTE_CARLO_TREE_SEARCH_HPP

#include <Eigen/Core>
#include <memory>
#include <vector>

namespace eyes_on_guys
{

class Node
{
public:
  Node(int branching_factor, double exploration_bonus);

  int get_best_action() const;

private:
  int branching_factor_;
  double exploration_bonus_;
  Eigen::VectorXi N_s_a_;
  Eigen::VectorXd Q_s_a_;
  std::vector<std::shared_ptr<Node>> children_;
};
double get_ucb1_bound(const int & action, const double & exploration_bonus,
                      const Eigen::VectorXi & N_s_a, const Eigen::VectorXd & Q_s_a);

class MonteCarloTreeSearch
{
public:
  MonteCarloTreeSearch(int num_states, int num_actions);

  int monte_carlo_tree_search(const int curr_state, const uint num_iter, const uint depth);

private:
  int num_states_;
  int num_actions_;

  void simulate();
  int get_greedy_action(const int curr_state);
};

} // namespace eyes_on_guys

#endif

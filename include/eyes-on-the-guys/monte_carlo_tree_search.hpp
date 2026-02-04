#ifndef MONTE_CARLO_TREE_SEARCH_HPP
#define MONTE_CARLO_TREE_SEARCH_HPP

#include <Eigen/Core>
#include <memory>
#include <vector>

namespace eyes_on_guys
{

class Node : public std::enable_shared_from_this<Node>
{
public:
  Node(const int id, const int branching_factor, const double exploration_bonus);

  int explore_best_action() const;
  bool has_been_visited();
  std::shared_ptr<Node> take_action(const int action);
  void update_count_and_action_value_function(const int best_action, const double q);

private:
  int id_;
  int branching_factor_;
  double exploration_bonus_;
  bool node_has_been_visited_;
  Eigen::VectorXi N_s_a_;
  Eigen::VectorXd Q_s_a_;
  std::vector<std::shared_ptr<Node>> children_;
};
double get_ucb1_bound(const int & action, const double & exploration_bonus,
                      const Eigen::VectorXi & N_s_a, const Eigen::VectorXd & Q_s_a);
double compute_running_average(const int new_val, const int old_val, const int new_count);

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
  int find_greedy_action(const std::shared_ptr<Node> curr_state) const;
};

double compute_value_function_estimate(const std::shared_ptr<Node> curr_state);
std::shared_ptr<Node> transition_from_state(std::shared_ptr<Node> curr_node, const int action);
double compute_reward_from_transitioning(const std::shared_ptr<Node> curr_state,
                                         const std::shared_ptr<Node> next_state);

} // namespace eyes_on_guys

#endif

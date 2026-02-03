#include <Eigen/Core>
#include <memory>
#include <vector>

#include "monte_carlo_tree_search.hpp"

namespace eyes_on_guys
{

Node::Node(int branching_factor, double exploration_bonus)
    : branching_factor_{branching_factor}
    , exploration_bonus_{exploration_bonus}
{
  N_s_a_ = Eigen::VectorXi::Zero(branching_factor_);
  Q_s_a_ = Eigen::VectorXd::Zero(branching_factor_);
  for (int i{0}; i < branching_factor_; ++i) {
    children_.push_back(nullptr);
  }
}

int Node::get_best_action() const
{
  int best_action{0};
  double best_action_ucb1_value = get_ucb1_bound(best_action, exploration_bonus_, N_s_a_, Q_s_a_);
  for (int i = 0; i < branching_factor_; ++i) {
    double ith_ucb1_value = get_ucb1_bound(i, exploration_bonus_, N_s_a_, Q_s_a_);
    if (ith_ucb1_value > best_action_ucb1_value) {
      best_action = i;
      best_action_ucb1_value = ith_ucb1_value;
    }
  }
  return best_action;
}

double get_ucb1_bound(const int & action, const double & exploration_bonus,
                      const Eigen::VectorXi & N_s_a, const Eigen::VectorXd & Q_s_a)
{
  if (action >= N_s_a.size() || action >= Q_s_a.size()) {
    return std::numeric_limits<double>::lowest();
  }

  if (N_s_a[action] == 0) {
    return std::numeric_limits<double>::infinity();
  }

  int N = N_s_a.sum();
  return Q_s_a[action] + exploration_bonus * std::sqrt(std::log(N) / N_s_a[action]);
}

MonteCarloTreeSearch::MonteCarloTreeSearch(int num_states, int num_actions)
    : num_states_{num_states}
    , num_actions_{num_actions}
{}

int MonteCarloTreeSearch::monte_carlo_tree_search(const int curr_state, const uint num_iter,
                                                  const uint depth)
{
  for (uint i = 0; i < num_iter; ++i) {
    simulate();
  }

  return get_greedy_action(curr_state);
}

void MonteCarloTreeSearch::simulate() { return; }
int MonteCarloTreeSearch::get_greedy_action(const int curr_state) { return 0; }

} // namespace eyes_on_guys

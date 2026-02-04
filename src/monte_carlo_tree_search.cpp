#include <Eigen/Core>
#include <memory>
#include <vector>

#include "monte_carlo_tree_search.hpp"

namespace eyes_on_guys
{

Node::Node(const int id, const int branching_factor, const double exploration_bonus)
    : id_{id}
    , branching_factor_{branching_factor}
    , exploration_bonus_{exploration_bonus}
    , node_has_been_visited_{false}
{
  N_s_a_ = Eigen::VectorXi::Zero(branching_factor_);
  Q_s_a_ = Eigen::VectorXd::Zero(branching_factor_);
  for (int i{0}; i < branching_factor_ + 1; ++i) {
    children_.push_back(nullptr);
  }
}

int Node::explore_best_action() const
{
  int best_action{0};
  double best_action_ucb1_value = get_ucb1_bound(best_action, exploration_bonus_, N_s_a_, Q_s_a_);
  for (int i = 0; i < branching_factor_; ++i) {
    if (i == id_) {
      continue;
    }

    double ith_ucb1_value = get_ucb1_bound(i, exploration_bonus_, N_s_a_, Q_s_a_);
    if (ith_ucb1_value > best_action_ucb1_value) {
      best_action = i;
      best_action_ucb1_value = ith_ucb1_value;
    }
  }
  return best_action;
}

bool Node::has_been_visited()
{
  if (!node_has_been_visited_) {
    node_has_been_visited_ = true;
    return false;
  }
  return true;
}

std::shared_ptr<Node> Node::take_action(const int action)
{
  if (action >= branching_factor_ || action < 0 || action == id_) {
    return shared_from_this();
  }

  if (children_.at(action) == nullptr) {
    children_[action] = std::make_shared<Node>(action, branching_factor_, exploration_bonus_);
  }
  return children_.at(action);
}

void Node::update_count_and_action_value_function(const int best_action, const double q)
{
  if (best_action >= branching_factor_ || best_action < 0) {
    return;
  }

  N_s_a_[best_action] += 1;
  Q_s_a_[best_action] = compute_running_average(q, Q_s_a_[best_action], N_s_a_[best_action]);
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

double compute_running_average(const int new_val, const int old_val, const int new_count)
{
  return old_val + (new_val - old_val) / new_count;
}

MonteCarloTreeSearch::MonteCarloTreeSearch(const int num_states, const int num_actions,
                                           const double discount_factor)
    : num_states_{num_states}
    , num_actions_{num_actions}
    , discount_factor_{discount_factor}
{}

int MonteCarloTreeSearch::monte_carlo_tree_search(const int initial_state, const uint num_iter,
                                                  const uint depth, const double exploration_bonus)
{
  if (initial_state >= num_states_) {
    return -1;
  }

  auto initial_node = std::make_shared<Node>(initial_state, num_actions_, exploration_bonus);
  for (uint i = 0; i < num_iter; ++i) {
    simulate(initial_node, depth);
  }
  return find_greedy_action(initial_node);
}

double MonteCarloTreeSearch::simulate(std::shared_ptr<Node> curr_state, const uint & depth)
{
  if (depth <= 0) {
    return compute_value_function_estimate(curr_state);
  }

  if (!curr_state->has_been_visited()) {
    return compute_value_function_estimate(curr_state);
  }

  int best_action = curr_state->explore_best_action();

  std::shared_ptr<Node> next_state = transition_from_state(curr_state, best_action);
  double r = compute_reward_from_transitioning(curr_state, next_state);

  double q = r + discount_factor_ * simulate(next_state, depth - 1);

  curr_state->update_count_and_action_value_function(best_action, q);

  return q;
}

int MonteCarloTreeSearch::find_greedy_action(const std::shared_ptr<Node> curr_state) const
{
  // TODO:
  return 0;
}

double compute_value_function_estimate(const std::shared_ptr<Node> curr_state)
{
  // TODO: optional lookahead with rollout
  return 0;
}

std::shared_ptr<Node> transition_from_state(std::shared_ptr<Node> curr_node, const int action)
{
  return curr_node->take_action(action);
}

double compute_reward_from_transitioning(const std::shared_ptr<Node> curr_state,
                                         const std::shared_ptr<Node> next_state)
{
  // TODO:
  return 0;
}

} // namespace eyes_on_guys

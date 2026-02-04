#include <Eigen/Core>
#include <memory>

#include "monte_carlo_node.hpp"

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
  return find_best_action(id_, branching_factor_, exploration_bonus_, N_s_a_, Q_s_a_);
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

void Node::update_count_and_action_value_function(const int action, const double q)
{
  if (action >= branching_factor_ || action < 0) {
    return;
  }

  N_s_a_[action] += 1;
  Q_s_a_[action] = compute_running_average(q, Q_s_a_[action], N_s_a_[action]);
}

int find_best_action(const int & id, const int & branching_factor, const double & exploration_bonus,
                     const Eigen::VectorXi & N_s_a, const Eigen::VectorXd & Q_s_a)
{
  int best_action{0};
  double best_action_ucb1_value = std::numeric_limits<double>::lowest();
  for (int i = 0; i < branching_factor + 1; ++i) {
    if (i == id) {
      continue;
    }

    double ith_ucb1_value = get_ucb1_bound(i, exploration_bonus, N_s_a, Q_s_a);
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

double compute_running_average(const double new_val, const double old_val, const int new_count)
{
  return old_val + (new_val - old_val) / new_count;
}

} // namespace eyes_on_guys

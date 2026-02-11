#include <Eigen/Core>
#include <memory>
#include <vector>

#include "eyes_on_guys_problem.hpp"
#include "monte_carlo_node.hpp"

namespace eyes_on_guys
{

MCTSNode::MCTSNode(const int id, const int num_agents, const double exploration_bonus)
    : id_{id}
    , num_agents_{num_agents}
    , exploration_bonus_{exploration_bonus}
    , node_has_been_visited_{false}
    , problem_info_{num_agents, 1.0, Eigen::MatrixXd::Zero(num_agents, num_agents)}
{
  N_s_a_ = Eigen::VectorXi::Zero(num_agents);
  Q_s_a_ = Eigen::VectorXd::Zero(num_agents);
  for (int i{0}; i < num_agents; ++i) {
    children_.push_back(nullptr);
  }
}

MCTSNode::MCTSNode(const int id, const int num_agents, const double exploration_bonus,
                   const EyesOnGuysProblem & problem_info)
    : id_{id}
    , num_agents_{num_agents}
    , exploration_bonus_{exploration_bonus}
    , node_has_been_visited_{false}
    , problem_info_{problem_info}
{
  N_s_a_ = Eigen::VectorXi::Zero(num_agents);
  Q_s_a_ = Eigen::VectorXd::Zero(num_agents);
  for (int i{0}; i < num_agents; ++i) {
    children_.push_back(nullptr);
  }
}

int MCTSNode::explore_best_action() const
{
  return find_best_action(id_, num_agents_, exploration_bonus_, N_s_a_, Q_s_a_);
}

bool MCTSNode::has_been_visited() { return node_has_been_visited_; }
void MCTSNode::visit_node() { node_has_been_visited_ = true; }

std::shared_ptr<MCTSNode> MCTSNode::take_action(const int action)
{
  if (action >= num_agents_ || action < 0 || action == id_) {
    return shared_from_this();
  }

  if (children_.at(action) == nullptr) {
    EyesOnGuysProblem child_problem_info =
      problem_info_.create_child_eyes_on_guys_state(id_, action);
    children_[action] =
      std::make_shared<MCTSNode>(action, num_agents_, exploration_bonus_, child_problem_info);
  }
  return children_.at(action);
}

void MCTSNode::update_count_and_action_value_function(const int action, const double q)
{
  if (action >= num_agents_ || action < 0) {
    return;
  }

  N_s_a_[action] += 1;
  Q_s_a_[action] = compute_running_average(q, Q_s_a_[action], N_s_a_[action]);
}

int find_best_action(const int & id, const int & num_agents, const double & exploration_bonus,
                     const Eigen::VectorXi & N_s_a, const Eigen::VectorXd & Q_s_a)
{
  int best_action{0};
  double best_action_ucb1_value = std::numeric_limits<double>::lowest();
  for (int i = 0; i < num_agents; ++i) {
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

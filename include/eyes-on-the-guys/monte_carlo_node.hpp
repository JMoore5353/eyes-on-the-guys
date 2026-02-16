#ifndef MONTE_CARLO_NODE_HPP
#define MONTE_CARLO_NODE_HPP

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "eyes_on_guys_problem.hpp"

namespace eyes_on_guys
{

class MCTSNode : public std::enable_shared_from_this<MCTSNode>
{
public:
  MCTSNode(const int id, const int num_agents, const double exploration_bonus);
  MCTSNode(const int id, const int num_agents, const double exploration_bonus,
           const EyesOnGuysProblem & problem_info);

  inline int get_id() const { return id_; }
  inline const EyesOnGuysProblem & get_ref_to_problem_info() const { return problem_info_; }
  inline const Eigen::VectorXd & get_action_value_function() const { return Q_s_a_; }
  inline const Eigen::VectorXi & get_action_state_count() const { return N_s_a_; }
  inline const std::vector<std::shared_ptr<MCTSNode>> & get_children_vector() const
  {
    return children_;
  }
  inline const std::shared_ptr<MCTSNode> get_child(const int child_id) const
  {
    return children_.at(child_id);
  }

  int explore_best_action() const;
  int get_greedy_action() const;
  bool has_been_visited() const;
  void visit_node();
  std::shared_ptr<MCTSNode> take_action(const int action);
  void update_count_and_action_value_function(const int action, const double q);

private:
  int id_;
  int num_agents_;
  double exploration_bonus_;
  bool node_has_been_visited_;
  Eigen::VectorXi N_s_a_;
  Eigen::VectorXd Q_s_a_;
  std::vector<std::shared_ptr<MCTSNode>> children_;

  EyesOnGuysProblem problem_info_;
};
int find_best_action(const int & id, const int & num_agents, const double & exploration_bonus,
                     const Eigen::VectorXi & N_s_a, const Eigen::VectorXd & Q_s_a);
int find_max_q_value(const int node_id, const Eigen::VectorXd & Q_s_a);
double get_ucb1_bound(const int & action, const double & exploration_bonus,
                      const Eigen::VectorXi & N_s_a, const Eigen::VectorXd & Q_s_a);
double compute_running_average(const double new_val, const double old_val, const int new_count);

} // namespace eyes_on_guys

#endif

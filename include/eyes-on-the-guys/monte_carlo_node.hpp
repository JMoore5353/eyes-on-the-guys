#ifndef MONTE_CARLO_NODE_HPP
#define MONTE_CARLO_NODE_HPP

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "eyes_on_guys_problem.hpp"

namespace eyes_on_guys
{

class MTCSNode : public std::enable_shared_from_this<MTCSNode>
{
public:
  MTCSNode(const int id, const int branching_factor, const double exploration_bonus);
  MTCSNode(const int id, const int branching_factor, const double exploration_bonus,
           const EyesOnGuysProblem & problem_info);

  inline int get_id() const { return id_; }
  inline const EyesOnGuysProblem & get_ref_to_problem_info() const { return problem_info_; }
  inline const Eigen::VectorXd & get_action_value_function() const { return Q_s_a_; }
  inline const Eigen::VectorXi & get_action_state_count() const { return N_s_a_; }
  inline const std::vector<std::shared_ptr<MTCSNode>> & get_children_vector() const
  {
    return children_;
  }

  int explore_best_action() const;
  bool has_been_visited();
  void visit_node();
  std::shared_ptr<MTCSNode> take_action(const int action);
  void update_count_and_action_value_function(const int action, const double q);

private:
  int id_;
  int branching_factor_;
  double exploration_bonus_;
  bool node_has_been_visited_;
  Eigen::VectorXi N_s_a_;
  Eigen::VectorXd Q_s_a_;
  std::vector<std::shared_ptr<MTCSNode>> children_;

  EyesOnGuysProblem problem_info_;
};
int find_best_action(const int & id, const int & branching_factor, const double & exploration_bonus,
                     const Eigen::VectorXi & N_s_a, const Eigen::VectorXd & Q_s_a);
double get_ucb1_bound(const int & action, const double & exploration_bonus,
                      const Eigen::VectorXi & N_s_a, const Eigen::VectorXd & Q_s_a);
double compute_running_average(const double new_val, const double old_val, const int new_count);

} // namespace eyes_on_guys

#endif

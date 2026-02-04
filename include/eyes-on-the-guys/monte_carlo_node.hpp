#ifndef MONTE_CARLO_NODE_HPP
#define MONTE_CARLO_NODE_HPP

#include <Eigen/Core>
#include <memory>

namespace eyes_on_guys
{

class Node : public std::enable_shared_from_this<Node>
{
public:
  Node(const int id, const int branching_factor, const double exploration_bonus);

  inline int get_id() const { return id_; }

  int explore_best_action() const;
  bool has_been_visited();
  std::shared_ptr<Node> take_action(const int action);
  void update_count_and_action_value_function(const int action, const double q);

private:
  int id_;
  int branching_factor_;
  double exploration_bonus_;
  bool node_has_been_visited_;
  Eigen::VectorXi N_s_a_;
  Eigen::VectorXd Q_s_a_;
  std::vector<std::shared_ptr<Node>> children_;
};
int find_best_action(const int & id, const int & branching_factor, const double & exploration_bonus,
                     const Eigen::VectorXi & N_s_a, const Eigen::VectorXd & Q_s_a);
double get_ucb1_bound(const int & action, const double & exploration_bonus,
                      const Eigen::VectorXi & N_s_a, const Eigen::VectorXd & Q_s_a);
double compute_running_average(const double new_val, const double old_val, const int new_count);

} // namespace eyes_on_guys

#endif

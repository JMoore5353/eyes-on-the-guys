#include <chrono>
#include <Eigen/Core>
#include <matplot/matplot.h>

#include "branch_and_bound_solver.hpp"
#include "eyes_on_guys_problem.hpp"
#include "forward_search_solver.hpp"
#include "monte_carlo_tree_search.hpp"

namespace eyes_on_guys
{

std::pair<double, double> run_mcts(const int num_agents,
                                              const int initial_state,
                                              const int mcts_num_iter,
                                              const int mcts_depth,
                                              const double mcts_discount_factor,
                                              const double mcts_exploration_bonus,
                                              const EyesOnGuysProblem& problem_info,
                                              const int mcts_lookahead_depth,
                                              const int mcts_lookahead_iters)
{
  MonteCarloTreeSearch tree_searcher{num_agents};

  auto start_time = std::chrono::high_resolution_clock::now();
  tree_searcher.search_for_best_action(initial_state,
                                       mcts_num_iter,
                                       mcts_depth,
                                       mcts_discount_factor,
                                       mcts_exploration_bonus,
                                       problem_info,
                                       mcts_lookahead_depth,
                                       mcts_lookahead_iters);
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  double solve_time = duration.count();
  double reward = tree_searcher.get_reward_of_greedy_action();

  return {solve_time, reward};
}

// std::pair<std::vector<int>, double> run_bnb()
// {
//   int num_agents = static_cast<int>(guy_names.size());
//   std::vector<std::string>::const_iterator it = std::find(guy_names.begin(), guy_names.end(), current_target_guy_);
//   int initial_state = std::distance(guy_names.begin(), it);
//
//   BranchAndBoundSolver bnb_solver{num_agents,
//                                   (int)this->get_parameter("bnb_max_depth").as_int(),
//                                   (int)this->get_parameter("bnb_max_iterations").as_int(),
//                                   this->get_parameter("bnb_discount_factor").as_double(),
//                                   this->get_parameter("bnb_debug_mode").as_bool(),
//                                   this->get_parameter("bnb_enable_pruning").as_bool()};
//   std::vector<int> optimal_sequence = bnb_solver.solve(initial_state, problem_info_);
//
//   current_target_guy_ = guy_names.at(optimal_sequence.at(0));
//   current_target_sequence_.clear();
//   int64_t depth = std::min(this->get_parameter("bnb_plan_depth").as_int(), (int64_t)optimal_sequence.size());
//   for (int64_t i=0; i<depth; ++i) {
//     int idx = optimal_sequence.at(i);
//     current_target_sequence_.push_back(guy_names.at(idx));
//   }
// }
//
// std::pair<std::vector<int>, double> run_forward_search()
// {
//   const int starting_guy = find_starting_guy_index(guy_names, current_target_guy_);
//   const ForwardSearchSolver::ForwardSearchInput input = make_forward_search_input(
//     guy_names,
//     guy_poses_,
//     problem_info_);
//   const ForwardSearchSolver::ForwardSearchConfig config = make_forward_search_config(
//     *this,
//     starting_guy,
//     problem_info_.shared_info_matrix);
//
//   RCLCPP_INFO_STREAM(
//     this->get_logger(),
//     "Forward search input guy_names.size(): " << guy_names.size());
//   RCLCPP_INFO_STREAM(
//     this->get_logger(),
//     "Forward search config depth passed to solver: " << config.depth);
//
//   ForwardSearchSolver solver;
//   const auto solve_start = std::chrono::steady_clock::now();
//   const ForwardSearchSolver::ForwardSearchResult result = solver.solve(input, config);
//   const auto solve_end = std::chrono::steady_clock::now();
//   const double solve_elapsed_sec =
//     std::chrono::duration<double>(solve_end - solve_start).count();
//   RCLCPP_INFO_STREAM(
//     this->get_logger(),
//     "Forward search solve() elapsed seconds: " << solve_elapsed_sec);
//   if (!result.success || result.sequence_ids.empty()) {
//     throw std::runtime_error("Forward search failed to produce a sequence: " + result.message);
//   }
//
//   std::ostringstream sequence_stream;
//   for (size_t i = 0; i < result.sequence_ids.size(); ++i) {
//     if (i > 0) {
//       sequence_stream << " -> ";
//     }
//     sequence_stream << result.sequence_ids.at(i);
//   }
//   RCLCPP_INFO_STREAM(
//     this->get_logger(),
//     "Forward search optimal sequence (len=" << result.sequence_ids.size() << "): "
//       << sequence_stream.str());
//
//   current_target_guy_ = result.sequence_ids.at(0);
//   current_target_sequence_ = result.sequence_ids;
//
//   RCLCPP_INFO_STREAM(this->get_logger(), "FORWARD NEXT: " << current_target_guy_);
// }

} // namespace eyes_on_guys

int main(int argc, char ** argv)
{
  const int num_agents{6};
  int initial_state{0};
  int mcts_num_iter{125};
  int mcts_depth{20};
  double mcts_discount_factor{0.9};
  double mcts_exploration_bonus{100};
  double relay_speed{10};
  auto dist_between_guys = 100 * Eigen::Matrix<double, num_agents, num_agents>::Random();
  eyes_on_guys::EyesOnGuysProblem problem_info{num_agents, relay_speed, dist_between_guys};
  int mcts_lookahead_depth{5};
  int mcts_lookahead_iters{30};
  std::pair<double, double> mcts_result = eyes_on_guys::run_mcts(num_agents,
                                                    initial_state,
                                                    mcts_num_iter,
                                                    mcts_depth,
                                                    mcts_discount_factor,
                                                    mcts_exploration_bonus,
                                                    problem_info,
                                                    mcts_lookahead_depth,
                                                    mcts_lookahead_iters);

  std::cout << "MCTS: R: " << mcts_result.second << " T: " << mcts_result.first << std::endl;
  return 0;
}

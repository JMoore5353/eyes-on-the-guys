#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <Eigen/Core>

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

// BnB uses max_iterations as a safety cap; with pruning enabled it often completes well
// before the limit for moderate depths.  discount_factor is matched to MCTS.
std::pair<double, double> run_bnb(const int num_agents,
                                  const int initial_state,
                                  const int depth,
                                  const double discount_factor,
                                  const EyesOnGuysProblem & problem_info,
                                  const int max_iterations = 100000)
{
  BranchAndBoundSolver solver{num_agents, depth, max_iterations, discount_factor};

  auto start_time = std::chrono::high_resolution_clock::now();
  solver.solve(initial_state, problem_info);
  auto end_time = std::chrono::high_resolution_clock::now();

  double solve_time =
    std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  double reward = solver.get_best_reward();

  return {solve_time, reward};
}

// Forward search uses its own reward function (shared_info_matrix norm, not delta norm), so
// rewards are not directly comparable to MCTS/BnB.  Structural parameters are matched:
//   depth, discount_factor, agent_velocity == relay_speed, rollout_depth, rollout_iters.
std::pair<double, double> run_fs(const int initial_state,
                                 const int depth,
                                 const double discount_factor,
                                 const double agent_velocity,
                                 const int rollout_depth,
                                 const int rollout_iters,
                                 const ForwardSearchSolver::ForwardSearchInput & input)
{
  ForwardSearchSolver solver;
  ForwardSearchSolver::ForwardSearchConfig config;
  config.depth = depth;
  config.discount_factor = discount_factor;
  config.agent_velocity = agent_velocity;
  config.roll_out_depth = rollout_depth;
  config.num_rollouts = rollout_iters;
  config.starting_guy = initial_state;
  config.shared_info_matrix = input.shared_info_matrix;

  auto start_time = std::chrono::high_resolution_clock::now();
  const ForwardSearchSolver::ForwardSearchResult result = solver.solve(input, config);
  auto end_time = std::chrono::high_resolution_clock::now();

  double solve_time =
    std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  double reward =
    result.success ? result.total_value : std::numeric_limits<double>::quiet_NaN();

  return {solve_time, reward};
}

} // namespace eyes_on_guys

int main(int argc, char ** argv)
{
  // ── Shared experiment parameters ──────────────────────────────────────────
  const int num_agents{6};
  const int initial_state{0};
  const double discount_factor{0.9};
  const double relay_speed{10.0};    // m/s — used as agent_velocity for ForwardSearch too
  const int rollout_depth{5};        // MCTS lookahead_depth == FS roll_out_depth
  const int rollout_iters{30};       // MCTS lookahead_iters  == FS num_rollouts
  const double mcts_exploration_bonus{100.0};
  const int mcts_num_iter{500};
  const int bnb_max_iterations{100000};
  const int num_trials{5};
  const int depth_min{4};
  const int depth_max{15};

  // ── Consistent geometry: generate 2D positions, derive distance matrix ────
  // All three solvers will see the same inter-agent distances.
  // ForwardSearch computes distances internally from poses; MCTS/BnB receive the
  // matrix directly.  Using srand(42) ensures reproducibility.
  srand(42);
  Eigen::MatrixXd positions = 500.0 * Eigen::MatrixXd::Random(num_agents, 2);

  Eigen::MatrixXd dist_between_guys = Eigen::MatrixXd::Zero(num_agents, num_agents);
  for (int i = 0; i < num_agents; ++i) {
    for (int j = 0; j < num_agents; ++j) {
      if (i != j) {
        dist_between_guys(i, j) = (positions.row(i) - positions.row(j)).norm();
        dist_between_guys(j, i) = dist_between_guys(i, j);
      }
    }
  }

  eyes_on_guys::EyesOnGuysProblem problem_info{num_agents, relay_speed, dist_between_guys};

  // Build ForwardSearch input from the same positions
  eyes_on_guys::ForwardSearchSolver::ForwardSearchInput fs_input;
  fs_input.ids.reserve(num_agents);
  fs_input.shared_info_matrix = Eigen::MatrixXd::Zero(num_agents, num_agents);
  for (int i = 0; i < num_agents; ++i) {
    const std::string id = std::to_string(i);
    fs_input.ids.push_back(id);
    eyes_on_guys::ForwardSearchSolver::GuyState state;
    state.pose.pose.position.x = positions(i, 0);
    state.pose.pose.position.y = positions(i, 1);
    state.pose.pose.position.z = 0.0;
    fs_input.guy_states_by_id[id] = state;
  }

  // ── Output CSV ─────────────────────────────────────────────────────────────
  // Note: MCTS/BnB rewards use compute_reward_model (delta shared_info norm);
  //       ForwardSearch uses its own reward function (full shared_info norm).
  //       Rewards across algorithms are not directly comparable.
  const std::string output_path = (argc > 1) ? argv[1] : "timing_and_reward_results.csv";
  std::ofstream csv(output_path);
  if (!csv.is_open()) {
    std::cerr << "Failed to open output file: " << output_path << std::endl;
    return 1;
  }
  csv << "algorithm,depth,trial,time_ms,reward\n";

  for (int depth = depth_min; depth <= depth_max; ++depth) {
    for (int trial = 0; trial < num_trials; ++trial) {

      auto [mcts_t, mcts_r] = eyes_on_guys::run_mcts(
        num_agents, initial_state, mcts_num_iter, depth,
        discount_factor, mcts_exploration_bonus, problem_info,
        rollout_depth, rollout_iters);
      csv << "mcts," << depth << "," << trial << "," << mcts_t << "," << mcts_r << "\n";

      auto [bnb_t, bnb_r] = eyes_on_guys::run_bnb(
        num_agents, initial_state, depth, discount_factor, problem_info, bnb_max_iterations);
      csv << "bnb," << depth << "," << trial << "," << bnb_t << "," << bnb_r << "\n";

      std::cout << "depth=" << depth << " trial=" << trial
                << "  mcts: t=" << mcts_t << "ms r=" << mcts_r
                << "  bnb:  t=" << bnb_t  << "ms r=" << bnb_r;

      if (depth <= 6) {
        auto [fs_t, fs_r] = eyes_on_guys::run_fs(
          initial_state, depth, discount_factor, relay_speed,
          rollout_depth, rollout_iters, fs_input);
        csv << "fs," << depth << "," << trial << "," << fs_t << "," << fs_r << "\n";

        std::cout << "  fs:   t=" << fs_t   << "ms r=" << fs_r << std::endl;
      } else {
        std::cout << std::endl;
      }

    }
  }

  csv.close();
  std::cout << "Results saved to: " << output_path << std::endl;
  return 0;
}

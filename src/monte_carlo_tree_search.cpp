#include <Eigen/Core>
#include <limits>
#include <memory>
#include <random>

#include "monte_carlo_node.hpp"
#include "monte_carlo_tree_search.hpp"

namespace eyes_on_guys
{

MonteCarloTreeSearch::MonteCarloTreeSearch(const int num_states)
    : num_states_{num_states}
{}

int MonteCarloTreeSearch::search_for_best_action(const int initial_state, const int num_iter,
                                                 const int depth, const double discount_factor,
                                                 const double exploration_bonus,
                                                 const EyesOnGuysProblem & problem_info,
                                                 const int lookahead_depth,
                                                 const int lookahead_iters)
{
  if (initial_state >= num_states_) {
    return -1;
  }

  auto initial_node =
    std::make_shared<MTCSNode>(initial_state, num_states_ - 1, exploration_bonus, problem_info);
  for (int i = 0; i < num_iter; ++i) {
    simulate(initial_node, depth, discount_factor, lookahead_depth, lookahead_iters);
  }
  return find_greedy_action(initial_node);
}

double MonteCarloTreeSearch::simulate(std::shared_ptr<MTCSNode> curr_state, const int depth,
                                      const double discount_factor, const int lookahead_depth,
                                      const int lookahead_iters)
{
  if (depth <= 0) {
    return lookahead_value_function_estimate(curr_state, num_states_, lookahead_depth,
                                             discount_factor, lookahead_iters);
  }

  if (!curr_state->has_been_visited()) {
    curr_state->visit_node();
    return lookahead_value_function_estimate(curr_state, num_states_, lookahead_depth,
                                             discount_factor, lookahead_iters);
  }

  int best_action = curr_state->explore_best_action();

  std::shared_ptr<MTCSNode> next_state = transition_from_state(curr_state, best_action);
  double r = compute_reward_from_transitioning(curr_state, next_state);

  double q = r
    + discount_factor
      * simulate(next_state, depth - 1, discount_factor, lookahead_depth, lookahead_iters);

  curr_state->update_count_and_action_value_function(best_action, q);

  return q;
}

double lookahead_value_function_estimate(const std::shared_ptr<MTCSNode> curr_state,
                                         const int num_states, const int depth,
                                         const double discount, const int num_iters)
{
  if (num_iters == 0) {
    return std::numeric_limits<double>::infinity();
  }

  double average_reward{0.0};
  for (int j = 0; j < num_iters; ++j) {
    auto start_state = std::make_shared<MTCSNode>(*curr_state);

    for (int i = 0; i < depth; ++i) {
      int action = lookahead_get_random_action(curr_state->get_id(), num_states);
      std::shared_ptr<MTCSNode> new_state = transition_from_state(curr_state, action);
      average_reward +=
        std::pow(discount, i) * compute_reward_from_transitioning(curr_state, new_state);
    }
  }
  return average_reward / num_iters;
}

int lookahead_get_random_action(const int curr_state, const int num_states)
{
  static std::random_device rd;
  static std::mt19937 generator(rd());
  std::uniform_int_distribution<> distr(0, num_states);
  int action{curr_state};
  while (action == curr_state) {
    action = distr(generator);
  }
  return action;
}

std::shared_ptr<MTCSNode> transition_from_state(std::shared_ptr<MTCSNode> curr_node,
                                                const int action)
{
  return curr_node->take_action(action);
}

double compute_reward_from_transitioning(const std::shared_ptr<MTCSNode> curr_state,
                                         const std::shared_ptr<MTCSNode> next_state)
{
  int curr_state_id{curr_state->get_id()};
  int next_state_id{next_state->get_id()};
  const EyesOnGuysProblem & curr_state_problem_info{curr_state->get_ref_to_problem_info()};
  const EyesOnGuysProblem & next_state_problem_info{next_state->get_ref_to_problem_info()};
  return compute_reward_model(curr_state_id, next_state_id, curr_state_problem_info,
                              next_state_problem_info);
}

int find_greedy_action(const std::shared_ptr<MTCSNode> curr_state)
{
  return curr_state->explore_best_action();
}

} // namespace eyes_on_guys

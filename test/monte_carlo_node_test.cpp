#include <Eigen/Core>
#include <gtest/gtest.h>
#include <memory>

#include "monte_carlo_node.hpp"

TEST(ucb1, GivenZeroN_ExpectInfinity)
{
  int action{0};
  double exploration_bonus{1.0};
  Eigen::Vector3i N_s_a{Eigen::Vector3i::Zero()};
  Eigen::Vector3d Q_s_a{Eigen::Vector3d::Zero()};

  double result = eyes_on_guys::get_ucb1_bound(action, exploration_bonus, N_s_a, Q_s_a);

  EXPECT_EQ(result, std::numeric_limits<double>::infinity());
}

TEST(ucb1, GivenOutOfBoundsIndex_ExpectLowest)
{
  int action{4};
  double exploration_bonus{1.0};
  Eigen::Vector3i N_s_a{Eigen::Vector3i::Zero()};
  Eigen::Vector3d Q_s_a{Eigen::Vector3d::Zero()};

  double result = eyes_on_guys::get_ucb1_bound(action, exploration_bonus, N_s_a, Q_s_a);

  EXPECT_EQ(result, std::numeric_limits<double>::lowest());
}

TEST(compute_running_average, GivenValues_ExpectCorrectAverage)
{
  double old_val{6.0};
  double new_val{3.0};
  int new_count{4};
  double expected_result{5.25};

  double result = eyes_on_guys::compute_running_average(new_val, old_val, new_count);

  EXPECT_EQ(result, expected_result);
}

TEST(find_best_action, ExpectFindBestActionDoesNotReturnSelf)
{
  int id{0};
  Eigen::Vector2i N_s_a = Eigen::Vector2i::Zero();
  Eigen::Vector2d Q_s_a = Eigen::Vector2d::Zero();

  int result = eyes_on_guys::find_best_action(id, 1, 1.0, N_s_a, Q_s_a);

  EXPECT_NE(result, id);

  id = 1;
  result = eyes_on_guys::find_best_action(id, 1, 1.0, N_s_a, Q_s_a);

  EXPECT_NE(result, id);
}

TEST(find_best_action, GivenNegativeValues_ExpectFindBestActionDoesNotReturnSelf)
{
  int id{0};
  Eigen::Vector2i N_s_a = Eigen::Vector2i::Zero();
  Eigen::Vector2d Q_s_a = Eigen::Vector2d::Zero();
  N_s_a[1] = 2;
  Q_s_a[1] = -1000;

  int result = eyes_on_guys::find_best_action(id, 1, 1.0, N_s_a, Q_s_a);

  EXPECT_NE(result, id);

  id = 1;
  result = eyes_on_guys::find_best_action(id, 1, 1.0, N_s_a, Q_s_a);

  EXPECT_NE(result, id);
}

TEST(find_best_action, ExpectFindBestActionReturnsFirstValue)
{
  int id{2};
  Eigen::Vector3i N_s_a = Eigen::Vector3i::Zero();
  Eigen::Vector3d Q_s_a = Eigen::Vector3d::Zero();

  int result = eyes_on_guys::find_best_action(id, 2, 1.0, N_s_a, Q_s_a);

  EXPECT_EQ(result, 0);
}

TEST(find_best_action, GivenQWithOneZero_ExpectReturnsCorrectAction)
{
  int id{0};
  Eigen::Matrix<int, 12, 1> N_s_a = Eigen::Matrix<int, 12, 1>::Ones();
  Eigen::Matrix<double, 12, 1> Q_s_a = Eigen::Matrix<double, 12, 1>::Ones();
  int correct_action{4};
  N_s_a(correct_action) = 0;

  int result = eyes_on_guys::find_best_action(id, 11, 1.0, N_s_a, Q_s_a);

  EXPECT_EQ(result, correct_action);
}

TEST(node, WhenTakingAction_ExpectCorrectMTCSNodeReturned)
{
  int id{0};
  auto curr_node = std::make_shared<eyes_on_guys::MTCSNode>(id, 5, 1.0);
  int action{2};

  std::shared_ptr<eyes_on_guys::MTCSNode> next_node = curr_node->take_action(action);

  EXPECT_EQ(next_node->get_id(), action);
}

TEST(node, GivenChildMTCSNodeAlreadyCreated_WhenTakingAction_ExpectSameMTCSNodeReturned)
{
  int id{0};
  auto curr_node = std::make_shared<eyes_on_guys::MTCSNode>(id, 5, 1.0);
  int action{2};

  std::shared_ptr<eyes_on_guys::MTCSNode> next_node = curr_node->take_action(action);
  std::shared_ptr<eyes_on_guys::MTCSNode> next_node2 = curr_node->take_action(action);

  EXPECT_EQ(next_node->get_id(), next_node2->get_id());
  EXPECT_EQ(next_node, next_node2);
}

TEST(node, WhenTakingSelfAction_ExpectSelfReturned)
{
  int id{0};
  auto curr_node = std::make_shared<eyes_on_guys::MTCSNode>(id, 5, 1.0);
  int action{0};

  std::shared_ptr<eyes_on_guys::MTCSNode> next_node = curr_node->take_action(action);

  EXPECT_EQ(next_node->get_id(), curr_node->get_id());
  EXPECT_EQ(next_node, curr_node);
}

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "monte_carlo_tree_search.hpp"

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

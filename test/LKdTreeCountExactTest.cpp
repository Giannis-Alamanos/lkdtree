// LKdTree - A fast incremental KdTree for real-time applications
// Copyright (C) 2025 Ioannis Alamanos. All rights reserved.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// Author: ioannesalamanos@gmail.com (Ioannis Alamanos)

// -----------------------------------------------------------------------------
// LKdTreeCountExactTest.cpp — unit tests for countExact
// -----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include "lkdtree/LKdTree.h"
#include <pcl/point_types.h>

using Point = pcl::PointXYZI;
using Tree  = lkdtree::LKdTree<Point, 3>;

class LKdTreeCountExactTest : public ::testing::Test {
protected:
  Tree tree_{16};

  Point makePoint(float x, float y, float z, float intensity = 1.0f) {
    return Point{x, y, z, intensity};
  }

  void insertPoints(const std::vector<Point>& points) {
    for (const auto& p : points) {
      tree_.insert(p);
    }
  }
};

// -----------------------------------------------------------------------------
// In a known point cloud, the countExact function should match the known count
// -----------------------------------------------------------------------------
TEST_F(LKdTreeCountExactTest, CountsExactMatchesCorrectly) {
    Point target = makePoint(1.0f, 2.0f, 3.0f);

    // Insert the target point 5 times, and a few others
    std::vector<Point> cloud = {
        target, target, target, target, target,
        makePoint(9.0f, 9.0f, 9.0f),
        makePoint(-1.0f, -2.0f, -3.0f)
    };
    insertPoints(cloud);

    EXPECT_EQ(tree_.countExact(target), 5);
}

// -----------------------------------------------------------------------------
// A point that does not exist in the cloud should return 0
// -----------------------------------------------------------------------------
TEST_F(LKdTreeCountExactTest, ReturnsZeroForNonExistingPoint) {
    std::vector<Point> cloud = {
        makePoint(0.0f, 0.0f, 0.0f),
        makePoint(1.0f, 2.0f, 3.0f),
        makePoint(4.0f, 5.0f, 6.0f)
    };
    insertPoints(cloud);

    Point query = makePoint(42.0f, 42.0f, 42.0f);  // Not present
    EXPECT_EQ(tree_.countExact(query), 0);
}

// -----------------------------------------------------------------------------
// Invalid points (e.g. with NaN) should trigger a warning and return -1
// -----------------------------------------------------------------------------
TEST_F(LKdTreeCountExactTest, InvalidPointReturnsZero) {
    std::vector<Point> cloud = {
        makePoint(0.0f, 0.0f, 0.0f),
        makePoint(1.0f, 2.0f, 3.0f),
        makePoint(4.0f, 5.0f, 6.0f)
    };
    insertPoints(cloud);

    Point invalid_nan = makePoint(NAN, 1.0f, 2.0f);
    Point invalid_inf = makePoint(1.0f, INFINITY, 2.0f);

    EXPECT_EQ(tree_.countExact(invalid_nan), 0u);
    EXPECT_EQ(tree_.countExact(invalid_inf), 0u);
}

// -----------------------------------------------------------------------------
// countExact should return 0 when the tree is empty
// -----------------------------------------------------------------------------
TEST_F(LKdTreeCountExactTest, QueryOnEmptyTreeReturnsZero) {
    Point query = makePoint(1.0f, 2.0f, 3.0f);
    EXPECT_EQ(tree_.countExact(query), 0u);
}

// -----------------------------------------------------------------------------
// countExact should return N when the tree contains N identical copies of the query
// -----------------------------------------------------------------------------
TEST_F(LKdTreeCountExactTest, AllPointsAreExactMatches) {
    constexpr std::size_t N = 100;
    Point p = makePoint(1.0f, 2.0f, 3.0f);

    std::vector<Point> duplicates(N, p);
    insertPoints(duplicates);

    EXPECT_EQ(tree_.countExact(p), N);
}
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
// LKdTreeCountExactTest.cpp — unit tests for insert
// -----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <random>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#define private   public
#define protected public
#include "lkdtree/LKdTree.h"
#undef private
#undef protected

using Point3D = pcl::PointXYZI;
using Point2D = pcl::PointXY;
using Tree3D = lkdtree::LKdTree<Point3D, 3>;   
using Tree2D = lkdtree::LKdTree<Point2D, 2>;   

class LKdTreeInsertTest3D : public ::testing::Test {
protected:
  std::unique_ptr<Tree3D> tree_;
  std::vector<Point3D> cloud_;

  void SetUp() override {
    tree_ = std::make_unique<Tree3D>(32); 
    generateRandomPoints(1000);
    for (const auto& pt : cloud_) {
      tree_->insert(pt);
    }
  }

  void generateRandomPoints(std::size_t count) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10000.0f, 10000.0f);

    cloud_.resize(count);
    for (auto& pt : cloud_) {
      pt.x = dist(gen);
      pt.y = dist(gen);
      pt.z = dist(gen);
    }
  }

  bool containsPoint(const std::vector<const Point3D*>& neighbors, const Point3D& query, float eps = 1e-6f) {
    for (const Point3D* p : neighbors) {
      if (std::abs(p->x - query.x) < eps &&
          std::abs(p->y - query.y) < eps &&
          std::abs(p->z - query.z) < eps) {
        return true;
      }
    }
    return false;
  }
};

class LKdTreeInsertTest2D : public ::testing::Test {
protected:
  std::unique_ptr<Tree2D> tree_;
  std::vector<Point2D> cloud_;

  void SetUp() override {
    tree_ = std::make_unique<Tree2D>(32); 
    generateRandomPoints(1000);
    for (const auto& pt : cloud_) {
      tree_->insert(pt);
    }
  }

  void generateRandomPoints(std::size_t count) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10000.0f, 10000.0f);

    cloud_.resize(count);
    for (auto& pt : cloud_) {
      pt.x = dist(gen);
      pt.y = dist(gen);
    }
  }

  bool containsPoint(const std::vector<const Point2D*>& neighbors, const Point2D& query, float eps = 1e-6f) {
    for (const Point2D* p : neighbors) {
      if (std::abs(p->x - query.x) < eps &&
          std::abs(p->y - query.y) < eps) {
        return true;
      }
    }
    return false;
  }
};


namespace {

// Helpers to compare points exactly (within eps)
inline bool equal2D(const Point2D& a, const Point2D& b, float eps=1e-6f) {
  return std::abs(a.x - b.x) <= eps && std::abs(a.y - b.y) <= eps;
}
inline bool equal3D(const Point3D& a, const Point3D& b, float eps=1e-6f) {
  return std::abs(a.x - b.x) <= eps && std::abs(a.y - b.y) <= eps && std::abs(a.z - b.z) <= eps;
}

// Find the ID for an exact-matching point by scanning leaves.
// Returns std::nullopt if not found.
std::optional<uint64_t> findIdForPoint(const Tree3D& tree, const Point3D& q, float eps=1e-6f) {
  using Node = Tree3D::Node;
  const Node* root = tree.root_.get();
  if (!root) return std::nullopt;

  std::vector<const Node*> stack;
  stack.push_back(root);

  while (!stack.empty()) {
    const Node* n = stack.back();
    stack.pop_back();

    if (n->is_leaf) {
      for (std::size_t i = 0; i < n->points.size(); ++i) {
        if (equal3D(n->points[i], q, eps)) {
          // points and ids must be aligned
          if (i < n->ids.size()) return n->ids[i];
          return std::nullopt;
        }
      }
      continue;
    }
    if (n->left)  stack.push_back(n->left.get());
    if (n->right) stack.push_back(n->right.get());
  }
  return std::nullopt;
}

std::optional<uint64_t> findIdForPoint(const Tree2D& tree, const Point2D& q, float eps=1e-6f) {
  using Node = Tree2D::Node;
  const Node* root = tree.root_.get();
  if (!root) return std::nullopt;

  std::vector<const Node*> stack;
  stack.push_back(root);

  while (!stack.empty()) {
    const Node* n = stack.back();
    stack.pop_back();

    if (n->is_leaf) {
      for (std::size_t i = 0; i < n->points.size(); ++i) {
        if (equal2D(n->points[i], q, eps)) {
          if (i < n->ids.size()) return n->ids[i];
          return std::nullopt;
        }
      }
      continue;
    }
    if (n->left)  stack.push_back(n->left.get());
    if (n->right) stack.push_back(n->right.get());
  }
  return std::nullopt;
}

// Collect all ids from all leaves into a set and check alignment
template <class Tree>
std::unordered_set<uint64_t> collectAllIdsAndAssertAlignment(const Tree& tree) {
  using Node = typename Tree::Node;
  std::unordered_set<uint64_t> ids;

  const Node* root = tree.root_.get();
  if (!root) return ids;

  std::vector<const Node*> st;
  st.push_back(root);

  while (!st.empty()) {
    const Node* n = st.back();
    st.pop_back();

    if (n->is_leaf) {
      // Non-fatal so this function can still return a value.
      EXPECT_EQ(n->points.size(), n->ids.size()) << "Leaf points/ids size mismatch";
      ids.insert(n->ids.begin(), n->ids.end());
    } else {
      if (n->left)  st.push_back(n->left.get());
      if (n->right) st.push_back(n->right.get());
    }
  }
  return ids;
}

} // namespace

template <class Tree>
static ::testing::AssertionResult CheckSubtreeSizes(const Tree& tree) {
  using Node = typename Tree::Node;
  const Node* root = tree.root_.get();
  if (!root) return ::testing::AssertionFailure() << "root is null";

  std::function<std::size_t(const Node*)> dfs = [&](const Node* n) -> std::size_t {
    if (!n) return 0;
    if (n->is_leaf) {
      EXPECT_EQ(n->points.size(), n->ids.size()) << "Leaf points/ids mismatch";
      EXPECT_EQ(n->subtree_size, n->points.size()) << "Leaf subtree_size mismatch";
      return n->points.size();
    }
    std::size_t L = dfs(n->left.get());
    std::size_t R = dfs(n->right.get());
    EXPECT_EQ(n->subtree_size, L + R) << "Internal subtree_size mismatch";
    return L + R;
  };

  std::size_t counted = dfs(root);
  EXPECT_EQ(counted, tree.size()) << "Tree size() disagrees with summed leaves";
  return ::testing::AssertionSuccess();
}

// -----------------------------------------------------------------------------
// Construction with invalid leaf size
// -----------------------------------------------------------------------------
TEST(LKdTreeInsertInvalidTest3D, ConstructWithZeroLeafSizeThrows) {
  EXPECT_THROW(Tree3D tree(0), std::invalid_argument);
}

TEST(LKdTreeInsertInvalidTest3D, ConstructWithNegativeLeafSizeThrows) {
  EXPECT_THROW(Tree3D tree(static_cast<std::size_t>(-1)), std::invalid_argument);
}

TEST(LKdTreeInsertInvalidTest2D, ConstructWithZeroLeafSizeThrows) {
  EXPECT_THROW(Tree2D tree(0), std::invalid_argument);
}

TEST(LKdTreeInsertInvalidTest2D, ConstructWithNegativeLeafSizeThrows) {
  EXPECT_THROW(Tree2D tree(static_cast<std::size_t>(-1)), std::invalid_argument);
}

// -----------------------------------------------------------------------------
// Valid insertion should not throw
// -----------------------------------------------------------------------------
TEST(LKdTreeInsertInvalidTest3D, InsertValidPointDoesNotThrow) {
  Tree3D tree(8);
  Point3D pt{1.0f, 2.0f, 3.0f};
  EXPECT_NO_THROW(tree.insert(pt));
}

TEST(LKdTreeInsertInvalidTest2D, InsertValidPointDoesNotThrow) {
  Tree2D tree(8);
  Point2D pt{1.0f, 2.0f};
  EXPECT_NO_THROW(tree.insert(pt));
}

// -----------------------------------------------------------------------------
// Insertion of points with NaN or Inf values
// -----------------------------------------------------------------------------
TEST(LKdTreeInsertInvalidTest3D, InsertNaNAndInfIgnored) {
  Tree3D tree(32);

  Point3D nan_pt_x = {std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f};
  Point3D nan_pt_y = {1.0f, std::numeric_limits<float>::quiet_NaN(), 0.f};
  Point3D nan_pt_z = {2.0f, 1.f, std::numeric_limits<float>::quiet_NaN()};
  Point3D inf_pt_x = {std::numeric_limits<float>::infinity(), 4.0f, 0.f};
  Point3D inf_pt_y = {1.f, std::numeric_limits<float>::infinity(), 0.f};
  Point3D inf_pt_z = {1.f, -8.0f, std::numeric_limits<float>::infinity()};

  tree.insert(nan_pt_x);
  tree.insert(nan_pt_y);
  tree.insert(nan_pt_z);
  tree.insert(inf_pt_x);
  tree.insert(inf_pt_y);
  tree.insert(inf_pt_z);

  EXPECT_EQ(tree.size(), 0u);
}

TEST(LKdTreeInsertInvalidTest2D, InsertNaNAndInfIgnored) {
  Tree2D tree(32);

  Point2D nan_pt_x = {std::numeric_limits<float>::quiet_NaN(), 0.f};
  Point2D nan_pt_y = {1.0f, std::numeric_limits<float>::quiet_NaN()};
  Point2D inf_pt_x = {std::numeric_limits<float>::infinity(), 4.0f};
  Point2D inf_pt_y = {1.f, std::numeric_limits<float>::infinity()};

  tree.insert(nan_pt_x);
  tree.insert(nan_pt_y);
  tree.insert(inf_pt_x);
  tree.insert(inf_pt_y);

  EXPECT_EQ(tree.size(), 0u);
}

// -----------------------------------------------------------------------------
// Exact match search in small point cloud
// -----------------------------------------------------------------------------
TEST(LKdTreeInsertBasicTest3D, ExactMatchInSmallCloud) {
  Tree3D tree(4);

  std::vector<Point3D> pts = {
      {1.0f, 2.0f, 3.0f},
      {-5.0f, 0.0f, 0.5f},
      {0.0f, 0.0f, 0.0f},
      {10.0f, 10.0f, 10.0f},
      {3.3f, -1.1f, 2.2f},
      {7.5f, 8.5f, 9.5f},
      {-2.0f, -3.0f, -1.0f},
      {4.0f, 4.0f, 4.0f}
  };

  for (const auto& pt : pts) {
    tree.insert(pt);
  }

  for (const auto& pt : pts) {
    auto results = tree.radiusSearch(pt, 1e-5f);
    ASSERT_EQ(results.size(), 1u);
    const auto* found = results[0];
    EXPECT_NE(found, nullptr);
    EXPECT_FLOAT_EQ(found->x, pt.x);
    EXPECT_FLOAT_EQ(found->y, pt.y);
    EXPECT_FLOAT_EQ(found->z, pt.z);
  }

  EXPECT_EQ(tree.size(), pts.size());
}

TEST(LKdTreeInsertBasicTest2D, ExactMatchInSmallCloud) {
  Tree2D tree(4);

  std::vector<Point2D> pts = {
      {0.0f, 0.0f},
      {1.0f, 1.0f},
      {-1.0f, -1.0f},
      {2.5f, 3.5f},
      {-4.0f, 2.0f},
      {3.0f, -2.0f},
      {5.0f, 5.0f},
      {-3.3f, -4.4f}
  };

  for (const auto& pt : pts) {
    tree.insert(pt);
  }

  for (const auto& pt : pts) {
    auto results = tree.radiusSearch(pt, 1e-5f);
    ASSERT_EQ(results.size(), 1u);
    const auto* found = results[0];
    EXPECT_NE(found, nullptr);
    EXPECT_FLOAT_EQ(found->x, pt.x);
    EXPECT_FLOAT_EQ(found->y, pt.y);
  }

  EXPECT_EQ(tree.size(), pts.size());
}

// -----------------------------------------------------------------------------
// Insertion and retrieval of points using radiusSearch
// -----------------------------------------------------------------------------
TEST_F(LKdTreeInsertTest3D, InsertAndFindAllPoints) {
  for (const auto& pt : cloud_) {
    auto result = tree_->radiusSearch(pt, 1e-5f);
    EXPECT_TRUE(containsPoint(result, pt)) << "Point not found: ("
                                           << pt.x << ", " << pt.y << ", " << pt.z << ")";
  }

  EXPECT_EQ(tree_->size(), cloud_.size());
}

TEST_F(LKdTreeInsertTest2D, InsertAndFindAllPoints) {
  for (const auto& pt : cloud_) {
    auto result = tree_->radiusSearch(pt, 1e-5f);
    EXPECT_TRUE(containsPoint(result, pt)) << "Point not found: ("
                                           << pt.x << ", " << pt.y << ")";
  }

  EXPECT_EQ(tree_->size(), cloud_.size());
}

// -----------------------------------------------------------------------------
// Insert mix of valid and invalid points, only valid should be stored
// -----------------------------------------------------------------------------
TEST_F(LKdTreeInsertTest3D, InsertMixedValidAndInvalidPoints) {
  // Inject some NaN and Inf values into the cloud
  std::vector<Point3D> invalid_pts = {
      {std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f},
      {1.f, std::numeric_limits<float>::infinity(), 0.f},
      {1.f, 2.f, std::numeric_limits<float>::quiet_NaN()},
      {std::numeric_limits<float>::infinity(), 5.f, -3.f},
  };

  for (const auto& pt : invalid_pts) {
    cloud_.emplace_back(pt);  // corrupt the cloud
    tree_->insert(pt);        // insert invalid — should be ignored
  }

  // Count only valid points
  std::size_t expected_valid = cloud_.size() - invalid_pts.size();

  for (std::size_t i = 0; i < expected_valid; ++i) {
    const auto& pt = cloud_[i];
    auto result = tree_->radiusSearch(pt, 1e-5f);
    EXPECT_TRUE(containsPoint(result, pt)) << "Point not found: ("
                                           << pt.x << ", " << pt.y << ", " << pt.z << ")";
  }

  EXPECT_EQ(tree_->size(), expected_valid);
}

TEST_F(LKdTreeInsertTest2D, InsertMixedValidAndInvalidPoints) {
  std::vector<Point2D> invalid_pts = {
      {std::numeric_limits<float>::quiet_NaN(), 0.f},
      {1.f, std::numeric_limits<float>::quiet_NaN()},
      {std::numeric_limits<float>::infinity(), -3.f},
      {0.f, std::numeric_limits<float>::infinity()}
  };

  for (const auto& pt : invalid_pts) {
    cloud_.emplace_back(pt);  // corrupt the cloud
    tree_->insert(pt);        // insert invalid — should be ignored
  }

  std::size_t expected_valid = cloud_.size() - invalid_pts.size();

  for (std::size_t i = 0; i < expected_valid; ++i) {
    const auto& pt = cloud_[i];
    auto result = tree_->radiusSearch(pt, 1e-5f);
    EXPECT_TRUE(containsPoint(result, pt)) << "Point not found: ("
                                           << pt.x << ", " << pt.y << ")";
  }

  EXPECT_EQ(tree_->size(), expected_valid);
}

// -----------------------------------------------------------------------------
// Clearing the tree removes all points and resets size/depth
// -----------------------------------------------------------------------------
TEST_F(LKdTreeInsertTest3D, ClearRemovesAllPoints) {

  ASSERT_EQ(tree_->size(), 1000u);  // Initial size after setup
  ASSERT_GT(tree_->maxDepth(), 0u);

  // Now clear the tree
  tree_->clear();

  ASSERT_EQ(tree_->size(), 0u);
  ASSERT_EQ(tree_->maxDepth(), 0u);

  // Make sure it accepts new insertions after clear
  tree_->insert(Point3D{1.0f, 2.0f, 3.0f});
  EXPECT_EQ(tree_->size(), 1u);
}

// -----------------------------------------------------------------------------
// Insert unique point multiple times and verify countExact matches
// -----------------------------------------------------------------------------
TEST_F(LKdTreeInsertTest3D, CountExactReflectsDuplicateInsertions) {
  Point3D p{3.14f, 2.71f, 1.41f};

  constexpr std::size_t num_duplicates = 20;
  for (std::size_t i = 0; i < num_duplicates; ++i) {
    tree_->insert(p);
  }

  EXPECT_EQ(tree_->countExact(p), num_duplicates);
}

// -----------------------------------------------------------------------------
// Insert mix of duplicates and unique points, verify countExact only for one
// -----------------------------------------------------------------------------
TEST_F(LKdTreeInsertTest3D, CountExactOnlyMatchesIdenticalDuplicates) {
  Point3D common{9.9f, -4.4f, 7.7f};

  // Insert 10 duplicates of `common`
  for (int i = 0; i < 10; ++i)
    tree_->insert(common);

  // Insert a bunch of unrelated points
  for (int i = 0; i < 100; ++i) {
    Point3D p{static_cast<float>(i), static_cast<float>(i * 2), static_cast<float>(-i)};
    tree_->insert(p);
  }

  EXPECT_EQ(tree_->countExact(common), 10u);
}

// -----------------------------------------------------------------------------
// Compile-time errors for invalid PointCloud inputs
// -----------------------------------------------------------------------------

// The following code will NOT COMPILE because std::vector does not have a `.points` field.
/*
TEST(LKdTreeInsertConceptFailTest, InsertVectorInsteadOfPointCloudTriggersConceptError) {
  Tree3D tree(8);
  std::vector<Point3D> vec = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  (void)tree.insertPointCloud(vec);  // ❌ should trigger compile-time error
}
*/

// The following code will NOT COMPILE because pcl::PointCloud<PointXY> has points of the wrong type for a Tree3D (expects PointXYZI).
/*
TEST(LKdTreeInsertConceptFailTest, InsertWrongPointTypeTriggersConceptError) {
  Tree3D tree(8);
  pcl::PointCloud<Point2D> pc;
  pc.points.emplace_back(Point2D{1.0f, 2.0f});
  (void)tree.insertPointCloud(pc);  // ❌ should trigger compile-time error
}
*/

// -----------------------------------------------------------------------------
// Inserting a full valid point cloud should store all points
// -----------------------------------------------------------------------------
TEST_F(LKdTreeInsertTest3D, InsertPointCloudAddsAllPoints)
{
  pcl::PointCloud<Point3D> pc;
  for (const auto& pt : cloud_) {
    pc.points.push_back(pt);
  }

  Tree3D fresh_tree(32);
  std::size_t inserted = fresh_tree.insertPointCloud(pc);

  EXPECT_EQ(inserted, cloud_.size());   
  EXPECT_EQ(fresh_tree.size(), cloud_.size());
}
  
// -----------------------------------------------------------------------------
// Inserting an empty cloud should result in no-op
// -----------------------------------------------------------------------------
TEST_F(LKdTreeInsertTest3D, InsertEmptyPointCloudIsNoOp)
{
  pcl::PointCloud<Point3D> pc;  // empty
  Tree3D fresh_tree(32);

  std::size_t inserted = fresh_tree.insertPointCloud(pc);

  EXPECT_EQ(inserted, 0u);
  EXPECT_EQ(fresh_tree.size(), 0u);
}

// -----------------------------------------------------------------------------
// Skips invalid points (NaN or Inf) and inserts only valid
// -----------------------------------------------------------------------------
TEST_F(LKdTreeInsertTest3D, InsertPointCloudSkipsInvalidPoints)
{
  pcl::PointCloud<Point3D> pc;

  // Add 10 valid points from the cloud
  for (std::size_t i = 0; i < 10; ++i)
    pc.points.push_back(cloud_[i]);

  // Add invalid points
  pc.points.emplace_back(Point3D{std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f});
  pc.points.emplace_back(Point3D{1.f, std::numeric_limits<float>::infinity(), 0.f});   
  pc.points.emplace_back(Point3D{1.f, 2.f, std::numeric_limits<float>::quiet_NaN()});

  Tree3D fresh_tree(32);
  std::size_t inserted = fresh_tree.insertPointCloud(pc);

  EXPECT_EQ(inserted, 10u);              // Only valid points should be counted
  EXPECT_EQ(fresh_tree.size(), 10u);     // Internal state should match
}

// -----------------------------------------------------------------------------
// buildBalanced() — from point cloud (3D)
// -----------------------------------------------------------------------------
TEST(LKdTreeBuildBalanced3D, PerfectFillBuildHasExpectedSizeAndDepth)
{
  const std::size_t leaf = 32;
  // Choose N so N = leaf * 2^d for predictable depth
  // Let d = 6  →  N = 32 * 64 = 2048
  const std::size_t N = leaf * (1u << 6);

  pcl::PointCloud<Point3D> pc;
  pc.points.resize(N);
  // Deterministic coordinates; spread them a bit so median splits are stable
  for (std::size_t i = 0; i < N; ++i) {
    pc.points[i].x = static_cast<float>(i);
    pc.points[i].y = static_cast<float>(i * 0.5);
    pc.points[i].z = static_cast<float>(-static_cast<long>(i));
  }

  Tree3D tree(leaf);
  std::size_t built = tree.buildBalancedFromPointCloud(pc, /*fill_ratio=*/1.0f);

  EXPECT_EQ(built, N);
  EXPECT_EQ(tree.size(), N);

  // For perfect fill, expected depth = log2(N / leaf)
  const std::size_t expected_depth = 6u;
  EXPECT_EQ(tree.maxDepth(), expected_depth);

  // Spot-check content: a few exact points should be found by tight radius
  const std::array<std::size_t, 7> probe = {0, 1, 17, 101, 999, 1500, N - 1};
  for (std::size_t i : probe) {
    auto res = tree.radiusSearch(pc.points[i], 1e-5f);
    ASSERT_FALSE(res.empty());
    const auto* p = res.front();
    EXPECT_FLOAT_EQ(p->x, pc.points[i].x);
    EXPECT_FLOAT_EQ(p->y, pc.points[i].y);
    EXPECT_FLOAT_EQ(p->z, pc.points[i].z);
  }
}

// -----------------------------------------------------------------------------
// buildBalanced() — slack leaves (3D)
// -----------------------------------------------------------------------------
TEST(LKdTreeBuildBalanced3D, SlackLeavesIncreaseDepthAsExpected)
{
  const std::size_t leaf = 32;
  const std::size_t N = leaf * (1u << 6); // 2048
  const float fill_ratio = 0.5f;          // threshold = ceil(32 * 0.5) = 16

  pcl::PointCloud<Point3D> pc;
  pc.points.resize(N);
  for (std::size_t i = 0; i < N; ++i) {
    pc.points[i].x = static_cast<float>(i);
    pc.points[i].y = static_cast<float>(i * 0.25);
    pc.points[i].z = static_cast<float>(-static_cast<long>(i));
  }

  Tree3D tree(leaf);
  std::size_t built = tree.buildBalancedFromPointCloud(pc, fill_ratio);

  EXPECT_EQ(built, N);
  EXPECT_EQ(tree.size(), N);

  // With threshold=16, expected depth = log2(2048 / 16) = 7
  const std::size_t expected_depth = 7u;
  EXPECT_EQ(tree.maxDepth(), expected_depth);
}

// -----------------------------------------------------------------------------
// buildBalanced() — invalid fill_ratio is rejected and tree unchanged (3D)
// -----------------------------------------------------------------------------
TEST(LKdTreeBuildBalanced3D, InvalidFillRatioDoesNotModifyTree)
{
  Tree3D tree(32);

  // Preload with a couple of inserts so we can detect unintended changes
  tree.insert(Point3D{1.f, 2.f, 3.f});
  tree.insert(Point3D{-1.f, -2.f, 0.5f});
  const auto size_before = tree.size();
  const auto depth_before = tree.maxDepth();

  pcl::PointCloud<Point3D> pc;
  pc.points.push_back(Point3D{10.f, 20.f, 30.f});

  // fill_ratio <= 0
  std::size_t built0 = tree.buildBalancedFromPointCloud(pc, 0.0f);
  EXPECT_EQ(built0, 0u);
  EXPECT_EQ(tree.size(), size_before);
  EXPECT_EQ(tree.maxDepth(), depth_before);

  // fill_ratio > 1
  std::size_t built2 = tree.buildBalancedFromPointCloud(pc, 1.5f);
  EXPECT_EQ(built2, 0u);
  EXPECT_EQ(tree.size(), size_before);
  EXPECT_EQ(tree.maxDepth(), depth_before);
}

// -----------------------------------------------------------------------------
// buildBalanced() — empty cloud → empty tree (3D)
// -----------------------------------------------------------------------------
TEST(LKdTreeBuildBalanced3D, EmptyCloudProducesEmptyTree)
{
  Tree3D tree(32);
  pcl::PointCloud<Point3D> empty;
  std::size_t built = tree.buildBalancedFromPointCloud(empty, 1.0f);

  EXPECT_EQ(built, 0u);
  EXPECT_EQ(tree.size(), 0u);
  EXPECT_EQ(tree.maxDepth(), 0u);
}

// -----------------------------------------------------------------------------
// buildBalanced() — skips NaN/Inf (3D)
// -----------------------------------------------------------------------------
TEST(LKdTreeBuildBalanced3D, SkipsInvalidPoints)
{
  Tree3D tree(32);
  pcl::PointCloud<Point3D> pc;

  // 8 valid + 3 invalid
  for (int i = 0; i < 8; ++i)
    pc.points.emplace_back(Point3D{static_cast<float>(i), static_cast<float>(i*i), 1.f});

  pc.points.emplace_back(Point3D{std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f});
  pc.points.emplace_back(Point3D{0.f, std::numeric_limits<float>::infinity(), 0.f});
  pc.points.emplace_back(Point3D{0.f, 0.f, -std::numeric_limits<float>::infinity()});

  std::size_t built = tree.buildBalancedFromPointCloud(pc, 1.0f);
  EXPECT_EQ(built, 8u);
  EXPECT_EQ(tree.size(), 8u);

  // Verify a couple of exact matches
  for (int i : {0, 3, 7}) {
    auto res = tree.radiusSearch(pc.points[static_cast<std::size_t>(i)], 1e-5f);
    ASSERT_FALSE(res.empty());
    EXPECT_FLOAT_EQ(res.front()->x, pc.points[static_cast<std::size_t>(i)].x);
    EXPECT_FLOAT_EQ(res.front()->y, pc.points[static_cast<std::size_t>(i)].y);
    EXPECT_FLOAT_EQ(res.front()->z, pc.points[static_cast<std::size_t>(i)].z);
  }
}

// -----------------------------------------------------------------------------
// buildBalanced() — 2D variant basic correctness
// -----------------------------------------------------------------------------
TEST(LKdTreeBuildBalanced2D, PerfectFillBuild2DHasExpectedDepthAndContent)
{
  const std::size_t leaf = 16;
  // Let d = 5 -> N = 16 * 32 = 512
  const std::size_t N = leaf * (1u << 5);

  pcl::PointCloud<Point2D> pc;
  pc.points.resize(N);
  for (std::size_t i = 0; i < N; ++i) {
    pc.points[i].x = static_cast<float>(i * 0.1);
    pc.points[i].y = static_cast<float>(-static_cast<long>(i));
  }

  Tree2D tree(leaf);
  std::size_t built = tree.buildBalancedFromPointCloud(pc, 1.0f);

  EXPECT_EQ(built, N);
  EXPECT_EQ(tree.size(), N);

  const std::size_t expected_depth = 5u; // log2(512 / 16)
  EXPECT_EQ(tree.maxDepth(), expected_depth);

  // Spot-check content
  const std::array<std::size_t, 4> probe2D = {0, 5, 123, N - 1};
  for (std::size_t i : probe2D) {
    auto res = tree.radiusSearch(pc.points[i], 1e-5f);
    ASSERT_FALSE(res.empty());
    const auto* p = res.front();
    EXPECT_FLOAT_EQ(p->x, pc.points[i].x);
    EXPECT_FLOAT_EQ(p->y, pc.points[i].y);
  }
}

// -----------------------------------------------------------------------------
// IDs with buildBalancedFromPointCloud (3D)
// -----------------------------------------------------------------------------
TEST(LKdTreeIds3D, BuildBalancedFromPointCloud_AssignsSequentialIdsAndKeepsAlignment) {
  const std::size_t N = 1500;
  pcl::PointCloud<Point3D> pc;
  pc.points.resize(N);
  for (std::size_t i = 0; i < N; ++i) {
    pc.points[i].x = static_cast<float>(i);
    pc.points[i].y = static_cast<float>(-static_cast<long>(i));
    pc.points[i].z = static_cast<float>(i) * 0.25f;
  }

  Tree3D tree(32);
  // Build; internally this increments next_insert_id_ N times.
  const std::size_t built = tree.buildBalancedFromPointCloud(pc, /*fill_ratio=*/1.0f);
  ASSERT_EQ(built, N);
  ASSERT_EQ(tree.size(), N);

  // Base ID is current next_insert_id_ - N
  const uint64_t base = tree.next_insert_id_ - N;

  // Spot-check a handful of points map to expected IDs (base + index)
  for (std::size_t i : {0ul, 1ul, 17ul, 233ul, 999ul, N - 1}) {
    auto id = findIdForPoint(tree, pc.points[i]);
    ASSERT_TRUE(id.has_value()) << "ID not found for point " << i;
    EXPECT_EQ(*id, base + i) << "Wrong id for point index " << i;
  }

  // All IDs are unique and the count matches tree size
  auto ids_set = collectAllIdsAndAssertAlignment(tree);
  EXPECT_EQ(ids_set.size(), N);
}

// -----------------------------------------------------------------------------
// IDs with buildBalancedFromPointCloud (2D, slack leaves)
// -----------------------------------------------------------------------------
TEST(LKdTreeIds2D, BuildBalancedWithSlack_AssignsSequentialIdsAndKeepsAlignment) {
  const std::size_t N = 1024;
  pcl::PointCloud<Point2D> pc;
  pc.points.resize(N);
  for (std::size_t i = 0; i < N; ++i) {
    pc.points[i].x = static_cast<float>(i) * 0.1f;
    pc.points[i].y = static_cast<float>(-static_cast<long>(i));
  }

  Tree2D tree(32);
  const float fill_ratio = 0.6f; // under-filled leaves
  const std::size_t built = tree.buildBalancedFromPointCloud(pc, fill_ratio);
  ASSERT_EQ(built, N);
  ASSERT_EQ(tree.size(), N);

  const uint64_t base = tree.next_insert_id_ - N;

  for (std::size_t i : {0ul, 3ul, 57ul, 321ul, N - 2, N - 1}) {
    auto id = findIdForPoint(tree, pc.points[i]);
    ASSERT_TRUE(id.has_value()) << "ID not found for point " << i;
    EXPECT_EQ(*id, base + i);
  }

  auto ids_set = collectAllIdsAndAssertAlignment(tree);
  EXPECT_EQ(ids_set.size(), N);
}

// -----------------------------------------------------------------------------
// IDs with incremental insert (3D)
// -----------------------------------------------------------------------------
TEST(LKdTreeIds3D, Insert_AssignsIncreasingIdsAndKeepsAlignment) {
  Tree3D tree(16);
  const std::size_t M = 500;

  // Record starting base (may be nonzero if reused in a longer suite)
  const uint64_t base_start = tree.next_insert_id_;

  std::vector<Point3D> pts;
  pts.resize(M);
  for (std::size_t i = 0; i < M; ++i) {
    pts[i].x = static_cast<float>(i) * 0.3f + 1.0f;
    pts[i].y = static_cast<float>(i) * -0.2f + 2.0f;
    pts[i].z = static_cast<float>(i) * 0.1f - 3.0f;
    tree.insert(pts[i]);
  }

  ASSERT_EQ(tree.size(), M);

  // Each inserted point should have id = base_start + i
  for (std::size_t i = 0; i < M; i += 53) { // sample a few to keep runtime light
    auto id = findIdForPoint(tree, pts[i]);
    ASSERT_TRUE(id.has_value()) << "ID not found for inserted point " << i;
    EXPECT_EQ(*id, base_start + i) << "Wrong id for inserted point index " << i;
  }

  // Verify global alignment and uniqueness
  auto ids_set = collectAllIdsAndAssertAlignment(tree);
  EXPECT_EQ(ids_set.size(), M);
}

// -----------------------------------------------------------------------------
// IDs continue after buildBalancedFromPointCloud + then insert (3D)
// -----------------------------------------------------------------------------
TEST(LKdTreeIds3D, BuildThenInsert_IdsContinueIncreasing) {
  const std::size_t N = 256;
  pcl::PointCloud<Point3D> pc;
  pc.points.resize(N);
  for (std::size_t i = 0; i < N; ++i) {
    pc.points[i].x = static_cast<float>(i);
    pc.points[i].y = static_cast<float>(i * 2);
    pc.points[i].z = static_cast<float>(-static_cast<long>(i));
  }

  Tree3D tree(32);
  ASSERT_EQ(tree.buildBalancedFromPointCloud(pc, 1.0f), N);

  const uint64_t expected_next = tree.next_insert_id_;

  // Insert one more point and check its ID is exactly expected_next
  Point3D pnew{1234.5f, -987.6f, 1.0f};
  tree.insert(pnew);

  auto id = findIdForPoint(tree, pnew);
  ASSERT_TRUE(id.has_value());
  EXPECT_EQ(*id, expected_next);

  // Alignment check
  collectAllIdsAndAssertAlignment(tree); // asserts inside
}

// -----------------------------------------------------------------------------
// Invalid arguments: must early-return and not modify state
// -----------------------------------------------------------------------------
TEST(LKdTreeInsertMaintain3D, InvalidArgumentsDoNotModifyTree) {
  Tree3D tree(8);

  // Seed with a couple points so we can detect unwanted changes
  tree.insert(Point3D{0.f,0.f,0.f});
  tree.insert(Point3D{1.f,1.f,1.f});
  const auto size_before = tree.size();
  const auto next_id_before = tree.next_insert_id_;

  auto try_call = [&](auto make_bad) {
    // baseline "good" settings
    float local_rebuild_ratio = 1.5f;
    std::size_t min_subtree   = 1;
    std::size_t max_subtree   = 1'000'000;
    float fill_ratio          = 1.0f;
    std::size_t keep_recent   = 8;
    std::size_t max_allowed   = 1'000'000;
    float high_imbalance      = 10.f;
    std::size_t full_keep     = 64;
    std::size_t par_thr       = 1;
    std::size_t task_thr      = 1;
    int num_threads           = 1;

    make_bad(local_rebuild_ratio, min_subtree, max_subtree, fill_ratio,
             keep_recent, max_allowed, high_imbalance, full_keep, par_thr, task_thr);

    tree.insertMaintain(Point3D{2.f,3.f,4.f},
                        local_rebuild_ratio, min_subtree, max_subtree,
                        fill_ratio, keep_recent, max_allowed, high_imbalance,
                        full_keep, par_thr, task_thr, num_threads);

    EXPECT_EQ(tree.size(), size_before)        << "size changed on invalid args";
    EXPECT_EQ(tree.next_insert_id_, next_id_before) << "next_insert_id_ changed on invalid args";
  };

  // Each case flips exactly one argument into an invalid value
  try_call([](auto& r, auto& min, auto& max, auto&, auto&, auto&, auto&, auto&, auto&, auto&){
    r = 1.0f;  // local_rebuild_ratio <= 1
  });
  try_call([](auto&, auto& min, auto& max, auto&, auto&, auto&, auto&, auto&, auto&, auto&){
    min = 0;   // min_subtree_for_local_rebuild <= 0
  });
  try_call([](auto&, auto& min, auto& max, auto&, auto&, auto&, auto&, auto&, auto&, auto&){
    max = 0;   // max_subtree_for_local_rebuild <= 0
  });
  try_call([](auto&, auto& min, auto& max, auto&, auto&, auto&, auto&, auto&, auto&, auto&){
    min = 10; max = 5;  // max_subtree <= min_subtree
  });
  try_call([](auto&, auto&, auto&, auto&, auto& keep, auto&, auto&, auto&, auto&, auto&){
    keep = 0;  // keep_recent_limit <= 0
  });
  try_call([](auto&, auto&, auto&, auto&, auto&, auto& max_allowed, auto&, auto&, auto&, auto&){
    max_allowed = 0;  // max_allowed_size <= 0
  });
  try_call([](auto&, auto&, auto&, auto&, auto&, auto&, auto& hi, auto&, auto&, auto&){
    hi = 1.0f;  // high_imbalance_limit <= 1
  });
  try_call([](auto&, auto&, auto&, auto&, auto&, auto&, auto&, auto& full, auto&, auto&){
    full = 0;  // full_rebuild_size <= 0
  });
  try_call([](auto&, auto&, auto&, auto&, auto&, auto&, auto&, auto&, auto& par, auto&){
    par = 0;  // point_num_parallel_thr <= 0
  });
  try_call([](auto&, auto&, auto&, auto&, auto&, auto&, auto&, auto&, auto&, auto& task){
    task = 0; // point_num_task_thr <= 0
  });
}

// -----------------------------------------------------------------------------
// No-rebuild path: behaves like a normal insert (size +1, point is findable)
// -----------------------------------------------------------------------------
TEST(LKdTreeInsertMaintain3D, NoRebuildActsLikeInsert) {
  Tree3D tree(8);
  // Make a small base so root may or may not split; we force "no rebuild" anyway
  for (int i = 0; i < 12; ++i)
    tree.insert(Point3D{static_cast<float>(i), 0.f, 0.f});

  const auto size_before = tree.size();
  Point3D p{123.4f, -5.6f, 7.8f};

  tree.insertMaintain(
      p,
      /*local_rebuild_ratio=*/1000.f,     // so local never triggers
      /*min_subtree=*/1'000'000,          // ensure window excludes everything
      /*max_subtree=*/2'000'000,
      /*fill_ratio=*/0.7f,
      /*keep_recent_limit=*/64,
      /*max_allowed_size=*/1'000'000,     // no global by size
      /*high_imbalance_limit=*/1e9f,      // no global by imbalance
      /*full_rebuild_size=*/64,
      /*point_num_parallel_thr=*/1,
      /*point_num_task_thr=*/1,
      /*num_threads=*/1);

  EXPECT_EQ(tree.size(), size_before + 1);
  auto res = tree.radiusSearch(p, 1e-5f);
  ASSERT_FALSE(res.empty());
  EXPECT_FLOAT_EQ(res.front()->x, p.x);
  EXPECT_TRUE(CheckSubtreeSizes(tree));
}

// -----------------------------------------------------------------------------
// Local rebuild at root: keeps only most recent ids (keep_recent_limit) + new pt
// -----------------------------------------------------------------------------
TEST(LKdTreeInsertMaintain3D, LocalRebuildAtRootKeepsOnlyRecentIds) {
  const std::size_t leaf = 4;
  Tree3D tree(leaf);

  // Create a tree with a split at root and some depth
  const std::size_t N = 40;
  for (std::size_t i = 0; i < N; ++i)
    tree.insert(Point3D{static_cast<float>(i), 0.f, 0.f});

  ASSERT_GT(tree.root_->subtree_size, leaf);
  ASSERT_FALSE(tree.root_->is_leaf);
  const auto* root = tree.root_.get();
  const std::size_t L0 = root->left  ? root->left->subtree_size  : 0;
  const std::size_t R0 = root->right ? root->right->subtree_size : 0;
  ASSERT_GT(L0 + R0, 0u);
  ASSERT_TRUE(root->left && root->right) << "Need both children for imbalance prediction";

  // Route to the larger side to maximize predicted ratio.
  const bool goes_left = (L0 >= R0);
  const float split = root->split_value;
  Point3D p = goes_left ? Point3D{split - 10.f, 0.f, 0.f}
                        : Point3D{split + 10.f, 0.f, 0.f};

  // Predicted ratio after insertion on chosen side
  const double lo = static_cast<double>(std::min(L0 + (goes_left ? 1u : 0u),
                                                 R0 + (goes_left ? 0u : 1u)));
  const double hi = static_cast<double>(std::max(L0 + (goes_left ? 1u : 0u),
                                                 R0 + (goes_left ? 0u : 1u)));
  const float predicted_ratio = static_cast<float>(hi / std::max(1.0, lo));

  const uint64_t next_before = tree.next_insert_id_;
  const std::size_t keep_recent = 10;  // we expect to keep exactly 10 old + 1 new
  const uint64_t cutoff_id = (next_before > keep_recent) ? (next_before - keep_recent) : 0;

  tree.insertMaintain(
      p,
      /*local_rebuild_ratio=*/predicted_ratio - 1e-6f, // guarantee trigger at root
      /*min_subtree=*/1,                               // include root
      /*max_subtree=*/1'000'000,
      /*fill_ratio=*/0.7f,
      /*keep_recent_limit=*/keep_recent,
      /*max_allowed_size=*/1'000'000,                  // no global-by-size
      /*high_imbalance_limit=*/1e9f,                   // no global-by-imbalance
      /*full_rebuild_size=*/64,
      /*point_num_parallel_thr=*/1,
      /*point_num_task_thr=*/1,
      /*num_threads=*/1);

  // Expect: kept exactly last 10 ids from before, plus the newly inserted id
  EXPECT_EQ(tree.size(), std::min<std::size_t>(keep_recent, N) + 1);

  // Collect IDs and check exactly the window [cutoff_id, next_before-1] plus next_before
  auto ids = collectAllIdsAndAssertAlignment(tree);
  ASSERT_EQ(ids.size(), std::min<std::size_t>(keep_recent, N) + 1);

  // Old recent ids
  for (uint64_t id = cutoff_id; id < next_before; ++id) {
    if (id + keep_recent <= next_before && id < next_before) {
      // Only the last 'keep_recent' should remain
      if (id < next_before - keep_recent) {
        EXPECT_EQ(ids.count(id), 0u) << "Old id should have been dropped: " << id;
      } else {
        EXPECT_EQ(ids.count(id), 1u) << "Recent id missing: " << id;
      }
    }
  }
  // New id
  EXPECT_EQ(ids.count(next_before), 1u) << "Newly inserted id missing";

  EXPECT_TRUE(CheckSubtreeSizes(tree));
}

// -----------------------------------------------------------------------------
// Global rebuild by max_allowed_size: keep last full_rebuild_size + new point
// -----------------------------------------------------------------------------
TEST(LKdTreeInsertMaintain3D, GlobalRebuildTriggeredByMaxAllowedSize) {
  Tree3D tree(8);

  // Build up a decent-size tree
  const std::size_t N = 50;
  for (std::size_t i = 0; i < N; ++i)
    tree.insert(Point3D{static_cast<float>(i), 0.1f * static_cast<float>(i), 0.f});

  const uint64_t next_before = tree.next_insert_id_;
  const std::size_t full_keep = 7;

  // Force global rebuild via size cap (note: wrapper ++total_points_ before internal check)
  tree.insertMaintain(
      Point3D{-999.f, 3.14f, 2.71f},
      /*local_rebuild_ratio=*/1000.f,
      /*min_subtree=*/1, /*max_subtree=*/1'000'000,
      /*fill_ratio=*/0.7f,
      /*keep_recent_limit=*/64,
      /*max_allowed_size=*/N - 3,          // ensure (N+1) > cap
      /*high_imbalance_limit=*/1e9f,
      /*full_rebuild_size=*/full_keep,
      /*point_num_parallel_thr=*/2500,
      /*point_num_task_thr=*/5000,
      /*num_threads=*/2);  

  // Expect to keep exactly last 'full_keep' ids plus the new one
  EXPECT_EQ(tree.size(), full_keep + 1);

  auto ids = collectAllIdsAndAssertAlignment(tree);
  ASSERT_EQ(ids.size(), full_keep + 1);

  // Kept window: [next_before - full_keep, next_before) + new id 'next_before'
  for (uint64_t id = next_before - full_keep; id < next_before; ++id)
    EXPECT_EQ(ids.count(id), 1u) << "Missing kept id: " << id;
  EXPECT_EQ(ids.count(next_before), 1u) << "Missing new id after global rebuild";

  EXPECT_TRUE(CheckSubtreeSizes(tree));
}

// -----------------------------------------------------------------------------
// allow_global_rebuild = false: size cap does not rebuild, nothing is dropped
// -----------------------------------------------------------------------------
TEST(LKdTreeInsertMaintain3D, GlobalRebuildBySizeDisabled) {
  Tree3D tree(8);

  const std::size_t N = 50;
  for (std::size_t i = 0; i < N; ++i)
    tree.insert(Point3D{static_cast<float>(i), 0.1f * static_cast<float>(i), 0.f});

  const uint64_t next_before = tree.nextInsertId();

  tree.insertMaintain(
      Point3D{-999.f, 3.14f, 2.71f},
      /*local_rebuild_ratio=*/1000.f,
      /*min_subtree=*/1, /*max_subtree=*/1'000'000,
      /*fill_ratio=*/0.7f,
      /*keep_recent_limit=*/64,
      /*max_allowed_size=*/N - 3,          // would trigger if allowed
      /*high_imbalance_limit=*/1e9f,
      /*full_rebuild_size=*/7,
      /*point_num_parallel_thr=*/2500,
      /*point_num_task_thr=*/5000,
      /*num_threads=*/2,
      /*allow_global_rebuild=*/false);

  EXPECT_EQ(tree.size(), N + 1);
  EXPECT_EQ(tree.nextInsertId(), next_before + 1);

  auto ids = collectAllIdsAndAssertAlignment(tree);
  ASSERT_EQ(ids.size(), N + 1);
  for (uint64_t id = 0; id <= next_before; ++id)
    EXPECT_EQ(ids.count(id), 1u) << "Missing id with global rebuild disabled: " << id;

  EXPECT_TRUE(CheckSubtreeSizes(tree));
}

// -----------------------------------------------------------------------------
// allow_global_rebuild: severe-imbalance trigger fires when allowed, not otherwise
// -----------------------------------------------------------------------------
TEST(LKdTreeInsertMaintain3D, GlobalRebuildByImbalanceRespectsFlag) {
  // Monotonic inserts leave the root heavily right-skewed.
  const std::size_t N = 50;
  const std::size_t full_keep = 7;
  auto make_tree = [&] {
    auto t = std::make_unique<Tree3D>(8);
    for (std::size_t i = 0; i < N; ++i)
      t->insert(Point3D{static_cast<float>(i), 0.f, 0.f});
    return t;
  };
  auto insert_right = [&](Tree3D& t, bool allow) {
    t.insertMaintain(
        Point3D{999.f, 0.f, 0.f},              // routes to the larger side
        /*local_rebuild_ratio=*/1000.f,
        /*min_subtree=*/1'000'000, /*max_subtree=*/2'000'000,  // no local rebuild
        /*fill_ratio=*/0.7f,
        /*keep_recent_limit=*/64,
        /*max_allowed_size=*/1'000'000,        // no global-by-size
        /*high_imbalance_limit=*/1.5f,         // global-by-imbalance if allowed
        /*full_rebuild_size=*/full_keep,
        /*point_num_parallel_thr=*/1,
        /*point_num_task_thr=*/1,
        /*num_threads=*/1,
        allow);
  };

  auto allowed = make_tree();
  insert_right(*allowed, true);
  EXPECT_EQ(allowed->size(), full_keep + 1) << "Control: imbalance should force a global rebuild";
  EXPECT_TRUE(CheckSubtreeSizes(*allowed));

  auto disabled = make_tree();
  insert_right(*disabled, false);
  EXPECT_EQ(disabled->size(), N + 1);
  EXPECT_EQ(collectAllIdsAndAssertAlignment(*disabled).size(), N + 1);
  EXPECT_EQ(disabled->nextInsertId(), N + 1);
  EXPECT_TRUE(CheckSubtreeSizes(*disabled));
}

// -----------------------------------------------------------------------------
// insertPointCloudMaintain honours allow_global_rebuild = false
// -----------------------------------------------------------------------------
TEST(LKdTreeInsertPointCloudMaintain3D, GlobalRebuildDisabled) {
  Tree3D tree(8);

  pcl::PointCloud<Point3D> pc;
  const std::size_t count = 200;
  for (std::size_t i = 0; i < count; ++i)
    pc.points.emplace_back(Point3D{static_cast<float>(i), 0.f, 0.f});

  tree.insertPointCloudMaintain(
      pc,
      /*local_rebuild_ratio=*/1000.0f,
      /*min_subtree_for_local_rebuild=*/1'000'000,
      /*max_subtree_for_local_rebuild=*/2'000'000,
      /*fill_ratio=*/0.7f,
      /*keep_recent_limit=*/64,
      /*max_allowed_size=*/20,               // would cap the tree if allowed
      /*high_imbalance_limit=*/1.5f,         // would fire on monotonic input
      /*full_rebuild_size=*/10,
      /*point_num_parallel_thr=*/1,
      /*point_num_task_thr=*/1,
      /*num_threads=*/1,
      /*allow_global_rebuild=*/false);

  EXPECT_EQ(tree.size(), count);
  EXPECT_EQ(tree.nextInsertId(), count);
  EXPECT_TRUE(CheckSubtreeSizes(tree));
}

// -----------------------------------------------------------------------------
// nextInsertId(): counts accepted points, skips invalid ones, survives clear()
// -----------------------------------------------------------------------------
TEST(LKdTreeInsert3D, NextInsertIdTracksAcceptedPoints) {
  Tree3D tree(8);
  EXPECT_EQ(tree.nextInsertId(), 0u);

  tree.insert(Point3D{1.f, 2.f, 3.f});
  tree.insert(Point3D{4.f, 5.f, 6.f});
  tree.insert(Point3D{7.f, 8.f, 9.f});
  EXPECT_EQ(tree.nextInsertId(), 3u);
  EXPECT_EQ(tree.nextInsertId(), tree.next_insert_id_);

  tree.insert(Point3D{std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f});
  EXPECT_EQ(tree.nextInsertId(), 3u) << "Invalid point must not consume an ID";

  auto ids = collectAllIdsAndAssertAlignment(tree);
  EXPECT_EQ(ids.count(tree.nextInsertId() - 1), 1u) << "Last inserted ID is nextInsertId() - 1";

  tree.clear();
  EXPECT_EQ(tree.nextInsertId(), 3u) << "clear() does not reset the ID counter";
  tree.insert(Point3D{0.f, 0.f, 0.f});
  EXPECT_EQ(tree.nextInsertId(), 4u);
}

// -----------------------------------------------------------------------------
// insertPointCloudMaintain — inserts all valid points, no rebuild path (3D)
// -----------------------------------------------------------------------------
TEST(LKdTreeInsertPointCloudMaintain3D, InsertsOnlyValidPoints)
{
  Tree3D tree(32);

  // Build a small point cloud: 20 valid + 3 invalid
  pcl::PointCloud<Point3D> pc;
  const std::size_t valid_count = 20;
  for (std::size_t i = 0; i < valid_count; ++i) {
    pc.points.emplace_back(Point3D{
        static_cast<float>(i) * 0.3f + 1.0f,
        static_cast<float>(i) * -0.2f + 2.0f,
        static_cast<float>(i) * 0.1f - 3.0f});
  }
  // Invalid points (should be skipped)
  pc.points.emplace_back(Point3D{std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f});
  pc.points.emplace_back(Point3D{0.f, std::numeric_limits<float>::infinity(), 0.f});
  pc.points.emplace_back(Point3D{0.f, 0.f, -std::numeric_limits<float>::infinity()});

  // Choose parameters that prevent both local and global rebuilds:
  //  - huge imbalance thresholds and subtree window out of range
  //  - large max_allowed_size so we won't trigger global-by-size
  tree.insertPointCloudMaintain(
      pc,
      /*local_rebuild_ratio=*/1000.0f,
      /*min_subtree_for_local_rebuild=*/1'000'000,
      /*max_subtree_for_local_rebuild=*/2'000'000,
      /*fill_ratio=*/0.7f,
      /*keep_recent_limit=*/64,
      /*max_allowed_size=*/1'000'000,
      /*high_imbalance_limit=*/1e9f,
      /*full_rebuild_size=*/64,
      /*point_num_parallel_thr=*/1,
      /*point_num_task_thr=*/1,
      /*num_threads=*/1);

  // Expect only the 20 valid points to be present
  EXPECT_EQ(tree.size(), valid_count);

  // Spot-check a few of the inserted points are retrievable exactly
  for (std::size_t i : {0ul, 3ul, 7ul, 12ul, 19ul}) {
    const auto& q = pc.points[i];
    auto res = tree.radiusSearch(q, 1e-5f);
    ASSERT_FALSE(res.empty()) << "Inserted point not found at index " << i;
    const Point3D* p = res.front();
    ASSERT_NE(p, nullptr);
    EXPECT_FLOAT_EQ(p->x, q.x);
    EXPECT_FLOAT_EQ(p->y, q.y);
    EXPECT_FLOAT_EQ(p->z, q.z);
  }

  // Basic structural sanity: subtree sizes should add up to size()
  EXPECT_TRUE(CheckSubtreeSizes(tree));
}


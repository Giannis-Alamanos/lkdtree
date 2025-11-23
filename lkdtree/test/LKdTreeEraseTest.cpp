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
// LKdTreeEraseTest.cpp — unit tests for erase, eraseSphere, and eraseBox
// -----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <random>
#include <pcl/point_types.h>
#define private public
#define protected public
#include "lkdtree/LKdTree.h"
#undef private
#undef protected

using Point = pcl::PointXYZI;
using Tree  = lkdtree::LKdTree<Point,3>;

namespace {

// ------------------------------------------------------------------
// Small helpers
// ------------------------------------------------------------------
Point makePoint(float x, float y, float z)        { return {x, y, z}; }

Point randomPoint(std::mt19937& g, float lo=-50.f, float hi=50.f)
{
  std::uniform_real_distribution<float> d(lo,hi);
  return { d(g), d(g), d(g) };
}

float distSq(const Point& a, const Point& b)
{
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  const float dz = a.z - b.z;
  return dx*dx + dy*dy + dz*dz;
}

// Recompute subtree size by traversing and counting points
std::size_t recomputeSize(const Tree::Node* n) {
  if (!n) return 0;
  if (n->is_leaf) return n->points.size();
  return recomputeSize(n->left.get()) + recomputeSize(n->right.get());
}

// Recompute tight AABB from contained points (∞/−∞ if empty)
void recomputeAABB(const Tree::Node* n, float mn[3], float mx[3]) {
  mn[0]=mn[1]=mn[2]= std::numeric_limits<float>::infinity();
  mx[0]=mx[1]=mx[2]=-std::numeric_limits<float>::infinity();

  std::function<void(const Tree::Node*)> dfs = [&](const Tree::Node* u){
    if (!u) return;
    if (u->is_leaf) {
      for (const auto& p : u->points) {
        const float v[3] = {p.x,p.y,p.z};
        for (int d=0; d<3; ++d) {
          mn[d] = std::min(mn[d], v[d]);
          mx[d] = std::max(mx[d], v[d]);
        }
      }
      return;
    }
    dfs(u->left.get());
    dfs(u->right.get());
  };
  dfs(n);
}

// Validate invariants recursively:
//  - subtree_size equals sum of contained points
//  - leaf ids size == points size
//  - node AABB equals tight AABB of contained points
void validateNode(const Tree::Node* n) {
  if (!n) return;

  // subtree_size invariant
  const std::size_t want_size = recomputeSize(n);
  EXPECT_EQ(n->subtree_size, want_size) << "subtree_size mismatch";

  // AABB invariant
  float mn[3], mx[3]; recomputeAABB(n, mn, mx);
  if (want_size == 0) {
    for (int d=0; d<3; ++d) {
      EXPECT_TRUE(std::isinf(n->bb_min[d]) && n->bb_min[d] > 0) << "empty-box min not +inf";
      EXPECT_TRUE(std::isinf(n->bb_max[d]) && n->bb_max[d] < 0) << "empty-box max not -inf";
    }
  } else {
    for (int d=0; d<3; ++d) {
      EXPECT_FLOAT_EQ(n->bb_min[d], mn[d]);
      EXPECT_FLOAT_EQ(n->bb_max[d], mx[d]);
    }
  }

  // leaf ids/points alignment
  if (n->is_leaf) {
    EXPECT_EQ(n->points.size(), n->ids.size()) << "ids/points length mismatch in leaf";
  } else {
    // internal: recurse
    validateNode(n->left.get());
    validateNode(n->right.get());
  }
}

} // anonymous namespace


// ============================================================================
// Test fixture — sets up a tree with a *single* leaf bucket (leaf_max_size=8)
// ============================================================================
class EraseFixture : public ::testing::Test
{
protected:
  Tree tree_{8};                 // force early splits for deterministic layout
  std::vector<Point> cloud_;     // keep originals for sanity checks
  std::mt19937 gen_{123};

  void SetUp() override
  {
    // Insert exactly 8 points so they all live in *one* leaf
    cloud_.reserve(8);
    for (int i=0;i<8;++i)
    {
      Point p = randomPoint(gen_);
      cloud_.push_back(p);
      tree_.insert(p);
    }
  }

  // helper: ensures point is absent
  void expectNotFound(const Point& q) const
  {
    auto res = tree_.radiusSearch(q,1e-6f);
    EXPECT_TRUE(res.empty());
    EXPECT_EQ(tree_.countExact(q),0u);
  }
};

// -----------------------------------------------------------------------------
// Erasing a point not present → returns 0, tree unchanged
// -----------------------------------------------------------------------------
TEST_F(EraseFixture, EraseNonExistingReturnsZero)
{
  Point bogus = makePoint(999.f,999.f,999.f);
  EXPECT_EQ(tree_.erase(bogus), 0u);
  EXPECT_EQ(tree_.size(), cloud_.size());
}

// -----------------------------------------------------------------------------
// Erase every element from the single leaf — search now empty, size==0
// -----------------------------------------------------------------------------
TEST_F(EraseFixture, LeafBecomesEmptyButTreeSafe)
{
  std::size_t removed = 0;
  for (const auto& p : cloud_)
    removed += tree_.erase(p);

  EXPECT_EQ(removed, cloud_.size());
  EXPECT_EQ(tree_.size(), 0u);
  EXPECT_EQ(tree_.countExact(cloud_.front()), 0u);

  for (const auto& p : cloud_) expectNotFound(p);
}

// -----------------------------------------------------------------------------
// After emptying a leaf, new insertions still work and bucket splits again
// -----------------------------------------------------------------------------
TEST_F(EraseFixture, InsertAfterEmptyLeafWorks)
{
  /* 1-a) remove everyone */
  for (const auto& p : cloud_) tree_.erase(p);
  ASSERT_EQ(tree_.size(),0u);

  /* 1-b) insert 16 new points ⇒ one split (leaf_max_size=8) */
  for (int i=0;i<16;++i) tree_.insert(randomPoint(gen_));

  EXPECT_EQ(tree_.size(),16u);
  EXPECT_GE(tree_.maxDepth(),1u);          // split happened
}

// -----------------------------------------------------------------------------
// Rebalancing after mass deletions still succeeds (no dangling data)
// -----------------------------------------------------------------------------
TEST_F(EraseFixture, RebalanceAfterMassErase)
{
  /* remove everything */
  for (const auto& p : cloud_) tree_.erase(p);
  ASSERT_EQ(tree_.size(),0u);

  /* should be a no-op but must not crash */
  EXPECT_FALSE(tree_.balanceTree());       // tooSmallToRebuild → false
  EXPECT_EQ(tree_.size(),0u);
  EXPECT_EQ(tree_.maxDepth(),0u);
}

// -----------------------------------------------------------------------------
// Mixed scenario: erase half the points, check size and search correctness
// -----------------------------------------------------------------------------
TEST_F(EraseFixture, PartialEraseMaintainsIntegrity)
{
  const std::size_t half = cloud_.size()/2;

  // erase first half
  std::size_t removed=0;
  for (std::size_t i=0;i<half;++i) removed += tree_.erase(cloud_[i]);

  EXPECT_EQ(removed, half);
  EXPECT_EQ(tree_.size(), cloud_.size() - half);

  // first half gone …
  for (std::size_t i=0;i<half;++i) expectNotFound(cloud_[i]);

  // … second half still present
  for (std::size_t i=half;i<cloud_.size();++i)
    EXPECT_EQ(tree_.countExact(cloud_[i]),1u);
}

// -----------------------------------------------------------------------------
// Gracefully ignores invalid input (NaN or Inf)
// -----------------------------------------------------------------------------
TEST_F(EraseFixture, InvalidPointIsIgnored)
{
  Point nan_pt = makePoint(NAN, 0.f, 0.f);
  Point inf_pt = makePoint(1.0f, -INFINITY, 2.0f);
  Point posinf  = makePoint(INFINITY, INFINITY, INFINITY);

  EXPECT_EQ(tree_.erase(nan_pt),   0u);
  EXPECT_EQ(tree_.erase(inf_pt),   0u);
  EXPECT_EQ(tree_.erase(posinf),   0u);

  // Tree should be unchanged
  EXPECT_EQ(tree_.size(), cloud_.size());

  for (const auto& pt : cloud_)
    EXPECT_EQ(tree_.countExact(pt), 1u);
}

// -----------------------------------------------------------------------------
// Erase on an empty tree should return 0 and not crash
// -----------------------------------------------------------------------------
TEST_F(EraseFixture, EraseOnEmptyTreeReturnsZero)
{
  tree_.clear();
  ASSERT_EQ(tree_.size(), 0u);

  Point query = makePoint(1.0f, 2.0f, 3.0f);
  EXPECT_EQ(tree_.erase(query), 0u);
}


// -----------------------------------------------------------------------------
// EraseSphere: removing a subset of points in a spherical region
// -----------------------------------------------------------------------------
TEST_F(EraseFixture, EraseSphereRemovesExpectedPoints)
{
  // Choose a center close to the first point
  Point center = cloud_[0];
  float radius = 1.0f;
  float radius_sq = radius * radius;

  std::size_t expected_removed = 0;
  for (const auto& pt : cloud_)
    if (distSq(pt, center) <= radius_sq)
      ++expected_removed;

  std::size_t removed = tree_.eraseSphere(center, radius);

  EXPECT_EQ(removed, expected_removed);
  EXPECT_EQ(tree_.size(), cloud_.size() - expected_removed);

  for (const auto& pt : cloud_)
  {
    if (distSq(pt, center) <= radius_sq)
      expectNotFound(pt);
    else
      EXPECT_EQ(tree_.countExact(pt), 1u);
  }
}

// -----------------------------------------------------------------------------
// EraseSphere: nothing to remove should return 0 and not alter tree
// -----------------------------------------------------------------------------
TEST_F(EraseFixture, EraseSphereNoEffectIfNothingInside)
{
  Point center = makePoint(999.f, 999.f, 999.f);
  float radius = 0.1f;

  std::size_t removed = tree_.eraseSphere(center, radius);
  EXPECT_EQ(removed, 0u);
  EXPECT_EQ(tree_.size(), cloud_.size());
}

// -----------------------------------------------------------------------------
// EraseSphere: removes everything if radius is very large
// -----------------------------------------------------------------------------
TEST_F(EraseFixture, EraseSphereWholeTree)
{
  Point center = makePoint(0.f, 0.f, 0.f);
  float radius = 1e6f;

  std::size_t removed = tree_.eraseSphere(center, radius);
  EXPECT_EQ(removed, cloud_.size());
  EXPECT_EQ(tree_.size(), 0u);

  for (const auto& pt : cloud_) expectNotFound(pt);
}

// -----------------------------------------------------------------------------
// EraseSphere: invalid query point or radius is ignored
// -----------------------------------------------------------------------------
TEST_F(EraseFixture, EraseSphereInvalidInputsIgnored)
{
  float radius = 5.0f;
  EXPECT_EQ(tree_.eraseSphere(makePoint(NAN, 0.f, 0.f), radius), 0u);
  EXPECT_EQ(tree_.eraseSphere(makePoint(0.f, 0.f, 0.f), -1.f),   0u);
  EXPECT_EQ(tree_.eraseSphere(makePoint(INFINITY, 0.f, 0.f), radius), 0u);

  // Tree should be unchanged
  EXPECT_EQ(tree_.size(), cloud_.size());
  for (const auto& pt : cloud_)
    EXPECT_EQ(tree_.countExact(pt), 1u);
}

// -----------------------------------------------------------------------------
// Erase and EraseSphere validated with brute-force on a large cloud
// -----------------------------------------------------------------------------
TEST(LargeEraseTest, EraseAgainstBruteForce)
{
  constexpr std::size_t N = 10'000;
  Tree tree;
  std::vector<Point> cloud;
  std::mt19937 gen(321);

  for (std::size_t i = 0; i < N; ++i) {
    cloud.push_back(randomPoint(gen));
    tree.insert(cloud.back());
  }

  std::vector<bool> erased(cloud.size(), false);
  std::size_t removed = 0;

  for (std::size_t i = 0; i < cloud.size(); i += 10) {
    removed += tree.erase(cloud[i]);
    erased[i] = true;
  }

  EXPECT_EQ(tree.size(), N - removed);

  for (std::size_t i = 0; i < cloud.size(); ++i) {
    if (erased[i])
      EXPECT_EQ(tree.countExact(cloud[i]), 0u);
    else
      EXPECT_EQ(tree.countExact(cloud[i]), 1u);
  }
}

TEST(LargeEraseTest, EraseSphereAgainstBruteForce)
{
  constexpr std::size_t N = 10'000;
  Tree tree;
  std::vector<Point> cloud;
  std::mt19937 gen(555);

  for (std::size_t i = 0; i < N; ++i) {
    cloud.push_back(randomPoint(gen));
    tree.insert(cloud.back());
  }

  Point center = makePoint(0.f, 0.f, 0.f);
  float radius = 25.f;
  float radius_sq = radius * radius;

  std::size_t expected = 0;
  for (const auto& p : cloud)
    if (distSq(p, center) <= radius_sq)
      ++expected;

  std::size_t removed = tree.eraseSphere(center, radius);
  EXPECT_EQ(removed, expected);
  EXPECT_EQ(tree.size(), N - expected);

  for (const auto& p : cloud) {
    if (distSq(p, center) <= radius_sq)
      EXPECT_EQ(tree.countExact(p), 0u);
    else
      EXPECT_EQ(tree.countExact(p), 1u);
  }
}

// -----------------------------------------------------------------------------
// EraseBox: removing points inside an axis-aligned bounding box
// -----------------------------------------------------------------------------
TEST_F(EraseFixture, EraseBoxRemovesExpectedPoints)
{
  Point min_pt = makePoint(-10.f, -10.f, -10.f);
  Point max_pt = makePoint(10.f, 10.f, 10.f);

  std::size_t expected_removed = 0;
  for (const auto& pt : cloud_)
    if (pt.x >= min_pt.x && pt.x <= max_pt.x &&
        pt.y >= min_pt.y && pt.y <= max_pt.y &&
        pt.z >= min_pt.z && pt.z <= max_pt.z)
      ++expected_removed;

  std::size_t removed = tree_.eraseBox(min_pt, max_pt);

  EXPECT_EQ(removed, expected_removed);
  EXPECT_EQ(tree_.size(), cloud_.size() - expected_removed);

  for (const auto& pt : cloud_)
  {
    if (pt.x >= min_pt.x && pt.x <= max_pt.x &&
        pt.y >= min_pt.y && pt.y <= max_pt.y &&
        pt.z >= min_pt.z && pt.z <= max_pt.z)
      expectNotFound(pt);
    else
      EXPECT_EQ(tree_.countExact(pt), 1u);
  }
}

// -----------------------------------------------------------------------------
// EraseBox: nothing to remove should return 0 and not alter tree
// -----------------------------------------------------------------------------
TEST_F(EraseFixture, EraseBoxNoEffectIfNothingInside)
{
  Point min_pt = makePoint(999.f, 999.f, 999.f);
  Point max_pt = makePoint(1000.f, 1000.f, 1000.f);

  std::size_t removed = tree_.eraseBox(min_pt, max_pt);
  EXPECT_EQ(removed, 0u);
  EXPECT_EQ(tree_.size(), cloud_.size());
}

// -----------------------------------------------------------------------------
// EraseBox: removes points inside box validated against brute-force
// -----------------------------------------------------------------------------
TEST(LargeEraseTest, EraseBoxAgainstBruteForce)
{
  constexpr std::size_t N = 100'000;
  Tree tree;
  std::vector<Point> cloud;
  std::mt19937 gen(888);

  for (std::size_t i = 0; i < N; ++i) {
    cloud.push_back(randomPoint(gen));
    tree.insert(cloud.back());
  }

  Point min_pt = makePoint(-15.f, -15.f, -15.f);
  Point max_pt = makePoint(15.f, 15.f, 15.f);

  std::size_t expected = 0;
  for (const auto& p : cloud)
    if (p.x >= min_pt.x && p.x <= max_pt.x &&
        p.y >= min_pt.y && p.y <= max_pt.y &&
        p.z >= min_pt.z && p.z <= max_pt.z)
      ++expected;

  std::size_t removed = tree.eraseBox(min_pt, max_pt);
  EXPECT_EQ(removed, expected);  
  EXPECT_EQ(tree.size(), N - expected);

  for (const auto& p : cloud) {
    if (p.x >= min_pt.x && p.x <= max_pt.x &&
        p.y >= min_pt.y && p.y <= max_pt.y &&
        p.z >= min_pt.z && p.z <= max_pt.z)
      EXPECT_EQ(tree.countExact(p), 0u);
    else
      EXPECT_EQ(tree.countExact(p), 1u);
  }
}

// -----------------------------------------------------------------------------
// EraseBox: invalid inputs (NaN, Inf, or min > max) are ignored
// -----------------------------------------------------------------------------
TEST_F(EraseFixture, EraseBoxInvalidInputsAreIgnored)
{
  std::size_t original_size = tree_.size();

  // NaN in min point
  Point nan_min = makePoint(NAN, -10.f, -10.f);
  Point valid_max = makePoint(10.f, 10.f, 10.f);
  EXPECT_EQ(tree_.eraseBox(nan_min, valid_max), 0u);

  // Inf in max point
  Point valid_min = makePoint(-10.f, -10.f, -10.f);
  Point inf_max = makePoint(10.f, INFINITY, 10.f);
  EXPECT_EQ(tree_.eraseBox(valid_min, inf_max), 0u);

  // min > max on x
  Point flipped_min = makePoint(5.f, -10.f, -10.f);
  Point flipped_max = makePoint(0.f, 10.f, 10.f);
  EXPECT_EQ(tree_.eraseBox(flipped_min, flipped_max), 0u);

  // Tree should be unchanged
  EXPECT_EQ(tree_.size(), original_size);
  for (const auto& pt : cloud_)
    EXPECT_EQ(tree_.countExact(pt), 1u);
}

// -----------------------------------------------------------------------------
// Internals: after various erase operations, subtree sizes and AABBs are correct
// -----------------------------------------------------------------------------
TEST(LKdTreeInternals, SizesAndAABBsAfterErase)
{
  Tree t(/*leaf_max_size=*/8);
  // Make a moderately branched tree
  for (int i=0;i<200;++i)
    t.insert(makePoint(i*0.1f, std::sin(i*0.1f)*5.f, std::cos(i*0.07f)*7.f));

  // Erase a pattern of points that hits many leaves/branches
  for (int i=0;i<200;i+=3)
    t.erase(makePoint(i*0.1f, std::sin(i*0.1f)*5.f, std::cos(i*0.07f)*7.f));

  // Validate invariants starting at root
  validateNode(t.root_.get());
  // And the public size must match root bookkeeping
  EXPECT_EQ(t.size(), t.root_->subtree_size);
}

// -----------------------------------------------------------------------------
// Internals: after eraseSphere and eraseBox, subtree sizes and AABBs are correct
// -----------------------------------------------------------------------------
TEST(LKdTreeInternals, SizesAndAABBsAfterEraseSphereAndBox)
{
  Tree t(8);
  for (int x=-10;x<=10;++x)
    for (int y=-10;y<=10;++y)
      for (int z=-2; z<=2; ++z)
        t.insert(makePoint(float(x), float(y), float(z)));

  // First a sphere cut
  t.eraseSphere(makePoint(0,0,0), /*radius=*/5.f);
  validateNode(t.root_.get());
  EXPECT_EQ(t.size(), t.root_->subtree_size);

  // Then a box cut that wipes one side of a split
  t.eraseBox(makePoint(-10,-10,-2), makePoint(-6,10,2));
  validateNode(t.root_.get());
  EXPECT_EQ(t.size(), t.root_->subtree_size);
}

// -----------------------------------------------------------------------------
// Internals: erasing all duplicates of a point keeps internal ids aligned
// -----------------------------------------------------------------------------
TEST(LKdTreeInternals, DuplicateCoordinatesEraseKeepsIdsAligned)
{
  Tree t(8);
  // Insert duplicates into a single leaf on purpose
  for (int i=0;i<16;++i) t.insert(makePoint(1.f,2.f,3.f)); // 16 identical points
  // Erase exact match should remove *all* matches
  const std::size_t removed = t.erase(makePoint(1.f,2.f,3.f));
  EXPECT_EQ(removed, 16u);
  validateNode(t.root_.get());
  EXPECT_EQ(t.size(), 0u);
}

// -----------------------------------------------------------------------------
// Internals: after erasing everything, root AABB is reset
// -----------------------------------------------------------------------------
TEST(LKdTreeInternals, EmptySubtreesHaveResetAABB)
{
  Tree t(4);
  // Fill a small grid
  for (int x=-3;x<=3;++x)
    for (int y=-3;y<=3;++y)
      t.insert(makePoint(float(x), float(y), 0.f));
  
  // Remove everything
  t.eraseBox(makePoint(-10,-10,-1), makePoint(10,10,1));

  // Root exists but should be "empty"
  ASSERT_TRUE(t.root_);
  EXPECT_EQ(t.root_->subtree_size, 0u);

  // Root AABB should be reset ( +inf / -inf as per resetBox )
  for (int d=0; d<3; ++d) {
    EXPECT_TRUE(std::isinf(t.root_->bb_min[d]) && t.root_->bb_min[d] > 0);
    EXPECT_TRUE(std::isinf(t.root_->bb_max[d]) && t.root_->bb_max[d] < 0);
  }
}

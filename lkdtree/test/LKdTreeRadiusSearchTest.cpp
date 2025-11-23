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
// LKdTreeCountExactTest.cpp — unit tests for radiusSearch
// -----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <random>
#include <chrono>
#include <unordered_set>
#include "lkdtree/LKdTree.h"  
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

using Point = pcl::PointXYZI;
using Tree  = lkdtree::LKdTree<Point, 3>;

namespace {

inline float Dist(const Point& a, const Point& b) {
  float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

Point RandInCube(std::mt19937& g, float min = -50.f, float max = 50.f) {
  std::uniform_real_distribution<float> d(min, max);
  return {d(g), d(g), d(g)};
}

}  // namespace

// =============================================================================
// Fixture
// =============================================================================
class RadiusSearchFixture : public ::testing::Test {
 protected:
  Tree tree_{32};                   // realistic leaf size
  std::mt19937 gen_{2025};
  std::vector<Point> cloud_;        // keep a copy for brute-force checks

  void InsertRandomCloud(int n) {
    for (int i = 0; i < n; ++i) {
      Point point = RandInCube(gen_);
      cloud_.push_back(point);
      tree_.insert(point);
    }
  }
};

// -----------------------------------------------------------------------------
// 1. Basic single-point behaviour
// -----------------------------------------------------------------------------
TEST_F(RadiusSearchFixture, SinglePointBehaviour) {
  Point p{0,0,0};
  tree_.insert(p);

  EXPECT_EQ(tree_.radiusSearch(p, 1e-4f).size(), 1u);
  EXPECT_TRUE(tree_.radiusSearch({10,10,10}, 0.5f).empty());
}

// -----------------------------------------------------------------------------
// 2. Duplicate coordinates return duplicates
// -----------------------------------------------------------------------------
TEST_F(RadiusSearchFixture, DuplicateCoords) {
  Point p{3,4,5};
  for (int i = 0; i < 10; ++i) tree_.insert(p);
  EXPECT_EQ(tree_.radiusSearch(p, 1e-4f).size(), 10u);
}

// -----------------------------------------------------------------------------
// 3. Radius search matches brute-force result on 20 k pts
// -----------------------------------------------------------------------------
TEST_F(RadiusSearchFixture, MatchesBruteForce20k) {
  const int N = 20'000;
  InsertRandomCloud(N);

  tree_.clear();
  pcl::PointCloud<Point> pc;
  for (const auto& p : cloud_)
    pc.push_back(p);

  ASSERT_EQ(tree_.buildBalancedFromPointCloud(pc, 1.0f), cloud_.size());
  ASSERT_EQ(tree_.size(), cloud_.size());

  Point query = RandInCube(gen_);
  float R = 2.5f;

  std::size_t brute = 0;
  for (const auto& p : cloud_)
    if (Dist(p, query) <= R) ++brute;

  auto res = tree_.radiusSearch(query, R);
  EXPECT_EQ(res.size(), brute);
}

// -----------------------------------------------------------------------------
// 4. Radius search matches brute-force result on 200 k pts
// -----------------------------------------------------------------------------
TEST_F(RadiusSearchFixture, MatchesBruteForce200k) {
  const int N = 200'000;
  InsertRandomCloud(N);

  tree_.clear();
  pcl::PointCloud<Point> pc;
  for (const auto& p : cloud_)
    pc.push_back(p);

  ASSERT_EQ(tree_.buildBalancedFromPointCloud(pc, 0.6f), cloud_.size());
  ASSERT_EQ(tree_.size(), cloud_.size());

  Point query = RandInCube(gen_);
  float R = 2.5f;

  std::size_t brute = 0;
  for (const auto& p : cloud_)
    if (Dist(p, query) <= R) ++brute;

  auto res = tree_.radiusSearch(query, R);
  EXPECT_EQ(res.size(), brute);
}

// -----------------------------------------------------------------------------
// 5. Whole-cloud query returns all points
// -----------------------------------------------------------------------------
TEST_F(RadiusSearchFixture, WholeCloudQuery) {
  InsertRandomCloud(5000);
  auto res = tree_.radiusSearch({0,0,0}, 1e5f);
  EXPECT_EQ(res.size(), 5000u);
}

// -----------------------------------------------------------------------------
// 6. Empty-tree query returns empty
// -----------------------------------------------------------------------------
TEST_F(RadiusSearchFixture, EmptyTreeQuery) {
  Tree empty(32);
  EXPECT_TRUE(empty.radiusSearch({0,0,0}, 5.f).empty());
} 

// -----------------------------------------------------------------------------
// 7. Negative radius returns no results
// -----------------------------------------------------------------------------
TEST_F(RadiusSearchFixture, NegativeRadiusReturnsEmpty) {
  InsertRandomCloud(100);  // insert some points just in case

  Point query = {0.f, 0.f, 0.f};
  float negative_radius = -1.0f;

  auto res = tree_.radiusSearch(query, negative_radius);
  EXPECT_TRUE(res.empty()) << "Negative radius should return empty result.";
}

// -----------------------------------------------------------------------------
// 8. Exact match test on small known point cloud
// -----------------------------------------------------------------------------
TEST_F(RadiusSearchFixture, ExactMatchesInSmallCloud) {
  std::vector<Point> known_points = {
    {0.f, 0.f, 0.f},
    {1.f, 0.f, 0.f},
    {0.f, 1.f, 0.f},
    {0.f, 0.f, 1.f},
    {1.f, 1.f, 1.f},
    {2.f, 2.f, 2.f},
    {3.f, 3.f, 3.f},
    {-1.f, -1.f, -1.f}
  };

  for (const auto& pt : known_points) {
    tree_.insert(pt);
  }

  for (const auto& pt : known_points) {
    auto res = tree_.radiusSearch(pt, 1e-5f);
    ASSERT_EQ(res.size(), 1u) << "Expected exactly one match for point: ("
                              << pt.x << ", " << pt.y << ", " << pt.z << ")";

    const auto* match = res[0];
    EXPECT_FLOAT_EQ(match->x, pt.x);
    EXPECT_FLOAT_EQ(match->y, pt.y);
    EXPECT_FLOAT_EQ(match->z, pt.z);
  }

  EXPECT_EQ(tree_.size(), known_points.size());
}

using Point2D = pcl::PointXY;
using Tree2D = lkdtree::LKdTree<Point2D, 2>;

namespace {

inline float Dist(const Point2D& a, const Point2D& b) {
  float dx = a.x - b.x, dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

Point2D RandInSquare(std::mt19937& g, float min = -50.f, float max = 50.f) {
  std::uniform_real_distribution<float> d(min, max);
  return {d(g), d(g)};
}

} // namespace

class RadiusSearchFixture2D : public ::testing::Test {
protected:
  Tree2D tree_{32};
  std::mt19937 gen_{2025};
  std::vector<Point2D> cloud_;

  void InsertRandomCloud(int n) {
    for (int i = 0; i < n; ++i) {
      Point2D p = RandInSquare(gen_);
      cloud_.push_back(p);
      tree_.insert(p);
    }
  }
};

// -----------------------------------------------------------------------------
// 1. Basic single-point behaviour
// -----------------------------------------------------------------------------
TEST_F(RadiusSearchFixture2D, SinglePointBehaviour) {
  Point2D p{0, 0};
  tree_.insert(p);

  EXPECT_EQ(tree_.radiusSearch(p, 1e-4f).size(), 1u);
  EXPECT_TRUE(tree_.radiusSearch({10, 10}, 0.5f).empty());
}

// -----------------------------------------------------------------------------
// 2. Duplicate coordinates return duplicates
// -----------------------------------------------------------------------------
TEST_F(RadiusSearchFixture2D, DuplicateCoords) {
  Point2D p{3, 4};
  for (int i = 0; i < 10; ++i) tree_.insert(p);
    EXPECT_EQ(tree_.radiusSearch(p, 1e-4f).size(), 10u);
}

// -----------------------------------------------------------------------------
// 3. Radius search matches brute-force result on 20 k pts
// -----------------------------------------------------------------------------
TEST_F(RadiusSearchFixture2D, MatchesBruteForce20k) {
  const int N = 20'000;
  InsertRandomCloud(N);

  Point2D query = RandInSquare(gen_);
  float R = 2.5f;

  std::size_t brute = 0;
  for (const auto& p : cloud_)
    if (Dist(p, query) <= R) ++brute;

  auto res = tree_.radiusSearch(query, R);
  EXPECT_EQ(res.size(), brute);
}

// -----------------------------------------------------------------------------
// 4. Radius search matches brute-force result on 200 k pts
// -----------------------------------------------------------------------------
TEST_F(RadiusSearchFixture2D, MatchesBruteForce200k) {
  const int N = 200'000;
  InsertRandomCloud(N);

  Point2D query = RandInSquare(gen_);
  float R = 2.5f;

  std::size_t brute = 0;
  for (const auto& p : cloud_)
    if (Dist(p, query) <= R) ++brute;

  auto res = tree_.radiusSearch(query, R);
  EXPECT_EQ(res.size(), brute);
}

// -----------------------------------------------------------------------------
// 5. Whole-cloud query returns all points
// -----------------------------------------------------------------------------
TEST_F(RadiusSearchFixture2D, WholeCloudQuery) {
  InsertRandomCloud(5000);
  auto res = tree_.radiusSearch({0, 0}, 1e5f);
  EXPECT_EQ(res.size(), 5000u);
}

// -----------------------------------------------------------------------------
// 6. Empty-tree query returns empty
// -----------------------------------------------------------------------------
TEST_F(RadiusSearchFixture2D, EmptyTreeQuery) {
  Tree2D empty(32);
  EXPECT_TRUE(empty.radiusSearch({0, 0}, 5.f).empty());
}

// -----------------------------------------------------------------------------
// 7. Negative radius returns no results
// -----------------------------------------------------------------------------
TEST_F(RadiusSearchFixture2D, NegativeRadiusReturnsEmpty) {
  InsertRandomCloud(100);

  Point2D query = {0.f, 0.f};
  float negative_radius = -1.0f;

  auto res = tree_.radiusSearch(query, negative_radius);
  EXPECT_TRUE(res.empty()) << "Negative radius should return empty result.";
}

// -----------------------------------------------------------------------------
// 8. Exact match test on small known point cloud
// -----------------------------------------------------------------------------
TEST_F(RadiusSearchFixture2D, ExactMatchesInSmallCloud) {
  std::vector<Point2D> known_points = {
    {0.f, 0.f},
    {1.f, 0.f},
    {0.f, 1.f},
    {1.f, 1.f},
    {2.f, 2.f},
    {3.f, 3.f},
    {-1.f, -1.f}
  };

  for (const auto& pt : known_points) {
    tree_.insert(pt);
  }

  for (const auto& pt : known_points) {
    auto res = tree_.radiusSearch(pt, 1e-5f);
    ASSERT_EQ(res.size(), 1u) << "Expected exactly one match for point: ("
    << pt.x << ", " << pt.y << ")";
    const auto* match = res[0];
    EXPECT_FLOAT_EQ(match->x, pt.x);
    EXPECT_FLOAT_EQ(match->y, pt.y);
  }

  EXPECT_EQ(tree_.size(), known_points.size());
}

// ───────────────────────────────────────────────────────────────────────── //
namespace {

inline Point makePt(float x,float y,float z){ return {x,y,z,0.f}; }

float sqDist(const Point& a,const Point& b)
{
  const float dx=a.x-b.x,dy=a.y-b.y,dz=a.z-b.z;return dx*dx+dy*dy+dz*dz;
}

// Compare two result sets (vector of ptrs) element-wise.
void expectEqualPtrLists(const std::vector<const Point*>& A,
                         const std::vector<const Point*>& B,
                         const Point&                      q,
                         float                             tol=1e-6f)
{
  ASSERT_EQ(A.size(),B.size());
  for(std::size_t i=0;i<A.size();++i)
    EXPECT_NEAR(sqDist(*A[i],q),sqDist(*B[i],q),tol)
      << "Mismatch at rank "<<i;
}

} // namespace

// =============================================================================
// Fixture supplying a fully-populated tree + raw cloud for reference tests
// =============================================================================
class BatchFixture : public ::testing::Test {
protected:
  static constexpr std::size_t N_POINTS  = 100'000;
  static constexpr std::size_t LEAF_MAX  = 64;

  Tree               tree_{LEAF_MAX};
  std::vector<Point> cloud_;

  void SetUp() override {
    std::mt19937 rng(2025);
    std::uniform_real_distribution<float> unif(-1'000.f,1'000.f);

    cloud_.reserve(N_POINTS);
    for(std::size_t i=0;i<N_POINTS;++i){
      cloud_.push_back(makePt(unif(rng),unif(rng),unif(rng)));
      tree_.insert(cloud_.back());
    }
  }
};

// -----------------------------------------------------------------------------
// 9. radiusSearchBatch: Correctness vs. individual radiusSearch()
// -----------------------------------------------------------------------------
TEST_F(BatchFixture, MatchesIndividualRadiusSearch)
{
  constexpr std::size_t Q_N = 1'000;
  constexpr float R = 2.5f;
  std::vector<Point> queries;
  queries.reserve(Q_N);
  std::mt19937 rng(17);
  std::uniform_real_distribution<float> unif(-500.f,500.f);
  for(std::size_t i=0; i<Q_N; ++i)
    queries.push_back(makePt(unif(rng),unif(rng),unif(rng)));

  auto batch = tree_.radiusSearchBatch(queries, R, 0);
  ASSERT_EQ(batch.size(), Q_N);
  for(std::size_t qi=0; qi<Q_N; ++qi){
    auto ref = tree_.radiusSearch(queries[qi], R);
    const auto& batch_res = batch[qi];

    ASSERT_EQ(batch_res.size(), ref.size()) << "Mismatch at query " << qi;

    for (const Point* p : batch_res) {
      const auto it = std::find(ref.begin(), ref.end(), p);
      EXPECT_TRUE(it != ref.end()) << "Point not found in reference at query " << qi;
    }
  }

  auto par = tree_.radiusSearchBatch(queries, R);
  ASSERT_EQ(par.size(), Q_N);
  for(std::size_t qi=0; qi<Q_N; ++qi){
    ASSERT_EQ(par[qi].size(), batch[qi].size()) << "Mismatch in parallel result at query " << qi;
    for (const Point* p : par[qi]) {
      const auto it = std::find(batch[qi].begin(), batch[qi].end(), p);
      EXPECT_TRUE(it != batch[qi].end()) << "Parallel result mismatch at query " << qi;
    }
  }
}

// -----------------------------------------------------------------------------
// 10. radiusSearchBatch: Performance micro-benchmark — parallel faster
// -----------------------------------------------------------------------------
TEST_F(BatchFixture, RadiusBatchParallelFasterThanSequential)
{
#if defined(_OPENMP)
  constexpr std::size_t Q_N = 20'000;
  constexpr float R = 3.0f;

  std::vector<Point> queries;
  queries.reserve(Q_N);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> unif(-800.f,800.f);
  for(std::size_t i=0;i<Q_N;++i)
    queries.emplace_back(makePt(unif(rng),unif(rng),unif(rng)));

  auto t0 = std::chrono::steady_clock::now();
  auto seq = tree_.radiusSearchBatch(queries, R, 0);
  auto t1 = std::chrono::steady_clock::now();

  auto t2 = std::chrono::steady_clock::now();
  auto par = tree_.radiusSearchBatch(queries, R);
  auto t3 = std::chrono::steady_clock::now();

  auto seq_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  auto par_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

  std::cout << "\n=== radiusSearchBatch Timing Comparison ===" << '\n';
  std::cout << "Total queries: " << queries.size() << ", Radius = " << R << '\n';
  std::cout << "Sequential time: " << seq_ms << " ms" << '\n';
  std::cout << "Parallel time:   " << par_ms  << " ms" << '\n';

  if (par_ms > 0) {
    double speedup = static_cast<double>(seq_ms) / par_ms;
    std::cout << "Speedup (seq / par): " << speedup << "x" << '\n';
  }
  std::cout << "============================================" << '\n';

  ASSERT_EQ(seq.size(), par.size());
  for(std::size_t i=0;i<seq.size();++i){
    ASSERT_EQ(seq[i].size(), par[i].size()) << "Mismatch in result size at query " << i;
    for (const Point* p : par[i]) {
      const auto it = std::find(seq[i].begin(), seq[i].end(), p);
      EXPECT_TRUE(it != seq[i].end()) << "Parallel result mismatch at query " << i;
    }
  }

  EXPECT_LT(par_ms, seq_ms)
      << "Parallel("<<par_ms<<" ms) not faster than sequential("<<seq_ms<<" ms)";
#else
  GTEST_SKIP() << "OpenMP disabled — performance test skipped.";
#endif
}

// -----------------------------------------------------------------------------
// 11. radiusSearchBatch: Negative radius returns empty result vectors
// -----------------------------------------------------------------------------
TEST_F(BatchFixture, RadiusBatchNegativeRadius)
{
  std::vector<Point> queries = {
    makePt(0.f, 0.f, 0.f),
    makePt(1.f, 1.f, 1.f),
    makePt(-5.f, -2.f, 3.f)
  };

  const float R = -3.0f;
  auto res = tree_.radiusSearchBatch(queries, R);

  EXPECT_EQ(res.size(), 0u);
}   





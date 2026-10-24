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

#include <gtest/gtest.h>
#include <random>
#include <chrono>
#include <limits>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "lkdtree/LKdTree.h"

using Point  = pcl::PointXYZI;
using Tree   = lkdtree::LKdTree<Point,3>;

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
class BoxSearchFixture : public ::testing::Test {
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

// =============================================================================
// Box‑search tests
// =============================================================================
namespace {
// helper: brute‑force points inside box
std::vector<const Point*>
bruteBox(const std::vector<Point>& cloud,
         const Point&              min_pt,
         const Point&              max_pt)
{
  std::vector<const Point*> out;
  for (const Point& p : cloud) {
    bool inside = (p.x >= min_pt.x && p.x <= max_pt.x) &&
                  (p.y >= min_pt.y && p.y <= max_pt.y) &&
                  (p.z >= min_pt.z && p.z <= max_pt.z);
    if (inside) out.push_back(&p);
  }
  return out;
}
// compare two unsorted pointer vectors as multisets
// compare two unsorted result vectors by coordinates (not by pointer)
void expectSamePoints(const std::vector<const Point*>& A,
                      const std::vector<const Point*>& B,
                      float tol = 1e-6f)
{
  ASSERT_EQ(A.size(), B.size());
  auto key = [&](const Point* p){ return std::tuple<float,float,float>{p->x,p->y,p->z}; };
  std::multiset<std::tuple<float,float,float>> setA, setB;
  for(const Point* p:A) setA.insert(key(p));
  for(const Point* p:B) setB.insert(key(p));
  EXPECT_EQ(setA, setB);
}
} // anon

// -----------------------------------------------------------------------------
// 1. boxSearch: invalid inputs (min > max, NaN)
// -----------------------------------------------------------------------------
TEST_F(BoxSearchFixture, BoxSearchInvalidInputs)
{
  // min > max in x → expect empty
  Point min_pt = makePt(10, -1, -1);
  Point max_pt = makePt( 5,  1,  1);
  EXPECT_TRUE(tree_.boxSearch(min_pt, max_pt).empty());

  // NaN coordinate in query box → expect empty
  min_pt = makePt(NAN, -100, -100);
  max_pt = makePt(100, 100, 100);
  EXPECT_TRUE(tree_.boxSearch(min_pt, max_pt).empty());
}

// -----------------------------------------------------------------------------
// 2. boxSearch: small deterministic cloud
// -----------------------------------------------------------------------------
TEST(BoxStandalone, BoxSearchSmallCloud)
{
  Tree t(8);
  const std::vector<Point> pts = {
    makePt(0,0,0), makePt(1,1,1), makePt(-1,-1,-1), makePt(2,2,2)
  };
  for (auto p:pts) t.insert(p);

  auto res = t.boxSearch( makePt(-0.5f,-0.5f,-0.5f), makePt(1.5f,1.5f,1.5f) );
  ASSERT_EQ(res.size(), 2u);
  ASSERT_EQ(res.size(), 2u);
  // Check coordinates instead of pointer identity
  auto hasPoint = [&](const Point& target){
    return std::any_of(res.begin(), res.end(), [&](const Point* p){
      return std::abs(p->x - target.x) < 1e-6f &&
             std::abs(p->y - target.y) < 1e-6f &&
             std::abs(p->z - target.z) < 1e-6f;
    });
  };
  EXPECT_TRUE(hasPoint(pts[0]));
  EXPECT_TRUE(hasPoint(pts[1]));
}

// -----------------------------------------------------------------------------
// 3. boxSearch vs. brute‑force on 100 k random points
// -----------------------------------------------------------------------------
TEST_F(BoxSearchFixture, BoxSearchMatchesBruteForce)
{
  std::mt19937 rng(77);
  std::uniform_real_distribution<float> unif(-300.f,300.f);

  Point min_pt = makePt(unif(rng), unif(rng), unif(rng));
  Point max_pt = makePt(min_pt.x + 50.f, min_pt.y + 60.f, min_pt.z + 70.f);

  auto fast  = tree_.boxSearch(min_pt, max_pt);
  auto brute = bruteBox(cloud_, min_pt, max_pt);
  expectSamePoints(fast, brute);
}

// -----------------------------------------------------------------------------
// 4. boxSearch vs. brute‑force on 100 k random points (second)
// -----------------------------------------------------------------------------
TEST_F(BoxSearchFixture, BoxSearchMatchesBruteForceSecond)
{
  std::mt19937 rng(77);
  std::uniform_real_distribution<float> unif(-500.f,500.f);

  Point min_pt = makePt(unif(rng), unif(rng), unif(rng));
  Point max_pt = makePt(min_pt.x + 100.f, min_pt.y + 100.f, min_pt.z + 100.f);

  auto fast  = tree_.boxSearch(min_pt, max_pt);
  auto brute = bruteBox(cloud_, min_pt, max_pt);
  expectSamePoints(fast, brute);
}

// -----------------------------------------------------------------------------
// 5. boxSearch: whole‑cloud box returns everything
// -----------------------------------------------------------------------------
TEST_F(BoxSearchFixture, BoxSearchWholeCloud)
{
  Point min_pt = makePt(-2000.f, -2000.f, -2000.f);
  Point max_pt = makePt( 2000.f,  2000.f,  2000.f);
  auto res = tree_.boxSearch(min_pt, max_pt);
  ASSERT_EQ(res.size(), cloud_.size());
}

// -----------------------------------------------------------------------------
// 6. boxSearch: point exactly on the box boundaries is included
// -----------------------------------------------------------------------------
TEST(BoxStandalone, BoxSearchIncludesBoundaryPoints)
{
  Tree t(8);
  const Point min_pt = makePt(-1.0f, -1.0f, -1.0f);
  const Point max_pt = makePt( 1.0f,  1.0f,  1.0f);

  const std::vector<Point> pts = {
    makePt(-1.0f, -1.0f, -1.0f),                           // on lower bound
    max_pt,                           // on upper bound
    makePt(0.0f, 0.0f, 0.0f),         // inside
    makePt(2.0f, 0.0f, 0.0f),         // outside
    makePt(-2.0f, 0.0f, 0.0f)         // outside
  };

  for (const auto& p : pts)
    t.insert(p);

  auto res = t.boxSearch(min_pt, max_pt);
  ASSERT_EQ(res.size(), 3u);  // only first 3 are in box

  auto hasPoint = [&](const Point& target){
    return std::any_of(res.begin(), res.end(), [&](const Point* p){
      return std::abs(p->x - target.x) < 1e-6f &&
             std::abs(p->y - target.y) < 1e-6f &&
             std::abs(p->z - target.z) < 1e-6f;
    });
  };

  EXPECT_TRUE(hasPoint(min_pt));                 // lower bound
  EXPECT_TRUE(hasPoint(max_pt));                 // upper bound
  EXPECT_TRUE(hasPoint(pts[2]));                 // interior point
}

// -----------------------------------------------------------------------------
// 7. boxSearchBatch: matches brute force results
// -----------------------------------------------------------------------------
TEST_F(BoxSearchFixture, BoxSearchBatchMatchesBrute)
{
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> unif(-400.f, 400.f);

  std::vector<std::pair<Point,Point>> queries;
  for (int i = 0; i < 10; ++i) {
    Point min_pt = makePt(unif(rng), unif(rng), unif(rng));
    Point max_pt = makePt(min_pt.x + 60.f, min_pt.y + 60.f, min_pt.z + 60.f);
    queries.emplace_back(min_pt, max_pt);
  }

  auto fast = tree_.boxSearchBatch(queries);

  for (std::size_t i = 0; i < queries.size(); ++i) {
    auto brute = bruteBox(cloud_, queries[i].first, queries[i].second);
    expectSamePoints(fast[i], brute);
  }
}

// -----------------------------------------------------------------------------
// 8. boxSearchBatch: empty input gives empty output
// -----------------------------------------------------------------------------
TEST_F(BoxSearchFixture, BoxSearchBatchEmptyInput)
{
  std::vector<std::pair<Point,Point>> empty;
  auto res = tree_.boxSearchBatch(empty);
  EXPECT_TRUE(res.empty());
}

// -----------------------------------------------------------------------------
// 9. boxSearchBatch: includes boundary points
// -----------------------------------------------------------------------------
TEST(BoxStandalone, BoxSearchBatchIncludesBoundary)
{
  Tree t(8);
  std::vector<Point> pts = {
    makePt(-1.0f,-1.0f,-1.0f),
    makePt( 1.0f, 1.0f, 1.0f),
    makePt( 0.0f, 0.0f, 0.0f),
    makePt( 5.0f, 5.0f, 5.0f)
  };
  for (auto p : pts) t.insert(p);

  std::vector<std::pair<Point,Point>> boxes = {
    { makePt(-1.0f,-1.0f,-1.0f), makePt(1.0f,1.0f,1.0f) }
  };

  auto res = t.boxSearchBatch(boxes);
  ASSERT_EQ(res.size(), 1u);
  const auto& hits = res[0];

  auto hasPoint = [&](const Point& target){
    return std::any_of(hits.begin(), hits.end(), [&](const Point* p){
      return std::abs(p->x - target.x) < 1e-6f &&
             std::abs(p->y - target.y) < 1e-6f &&
             std::abs(p->z - target.z) < 1e-6f;
    });
  };

  EXPECT_TRUE(hasPoint(pts[0]));  // min bound
  EXPECT_TRUE(hasPoint(pts[1]));  // max bound
  EXPECT_TRUE(hasPoint(pts[2]));  // inside
}

// -----------------------------------------------------------------------------
// 10. boxSearchBatch: parallel version is faster than sequential
// -----------------------------------------------------------------------------
TEST_F(BoxSearchFixture, BoxSearchBatchParallelFasterThanSequential)
{
#if defined(_OPENMP)
  constexpr std::size_t Q_N = 15'000;

  std::vector<std::pair<Point, Point>> boxes;
  boxes.reserve(Q_N);
  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> unif(-600.f, 600.f);

  for (std::size_t i = 0; i < Q_N; ++i) {
    Point min_pt = makePt(unif(rng), unif(rng), unif(rng));
    Point max_pt = makePt(min_pt.x + 40.f, min_pt.y + 40.f, min_pt.z + 40.f);
    boxes.emplace_back(min_pt, max_pt);
  }

  auto t0 = std::chrono::steady_clock::now();
  auto seq = tree_.boxSearchBatch(boxes, 0);
  auto t1 = std::chrono::steady_clock::now();

  auto t2 = std::chrono::steady_clock::now();
  auto par = tree_.boxSearchBatch(boxes); 
  auto t3 = std::chrono::steady_clock::now();       

  auto seq_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  auto par_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

  std::cout << "=== boxSearchBatch Timing Comparison ===" << '\n';
  std::cout << "Total queries: " << boxes.size() << '\n';
  std::cout << "Sequential time: " << seq_ms << " ms" << '\n';     
  std::cout << "Parallel time:   " << par_ms  << " ms" << '\n';
  if (par_ms > 0)
    std::cout << "Speedup (seq / par): " << static_cast<double>(seq_ms) / par_ms << "x" << '\n';
  std::cout << "=========================================" << '\n';

  ASSERT_EQ(seq.size(), par.size());
  for (std::size_t i = 0; i < boxes.size(); ++i)
    expectSamePoints(seq[i], par[i]);

  EXPECT_LT(par_ms, seq_ms)
      << "Parallel(" << par_ms << " ms) not faster than sequential(" << seq_ms << " ms)";
#else
  GTEST_SKIP() << "OpenMP disabled — performance test skipped.";
#endif
}

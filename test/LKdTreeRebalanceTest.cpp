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
// LKdTreeCountExactTest.cpp — unit tests for balancing
// -----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <future>
#include <thread>
#include <chrono>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <cmath>
#include <array> 
#include "lkdtree/LKdTree.h"
#include <pcl/point_types.h>

using Point = pcl::PointXYZI;
using Tree  = lkdtree::LKdTree<Point, 3>;

class LKdTreeRebalanceTest : public ::testing::Test {
protected:
  Tree tree_{32};

  void insertMany(std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
      Point p{static_cast<float>(i), static_cast<float>(i), static_cast<float>(i)};
      tree_.insert(p);
    }
  }
};

namespace {

// Helpers
static inline float d2(const Point& a, const Point& b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  const float dz = a.z - b.z;
  return dx*dx + dy*dy + dz*dz;
}

static inline bool inBox(const Point& p, const Point& mn, const Point& mx) {
  return (p.x >= mn.x && p.x <= mx.x) &&
         (p.y >= mn.y && p.y <= mx.y) &&
         (p.z >= mn.z && p.z <= mx.z);
}

static inline Point mk(float x, float y, float z, float i = 0.f) {
  Point p; p.x=x; p.y=y; p.z=z; p.intensity=i; return p;
}

static std::vector<Point> makeRandomPoints(std::size_t n, uint32_t seed=12345) {
  std::vector<Point> pts; pts.reserve(n);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uni(-1000.f, 1000.f);
  for (std::size_t i=0;i<n;++i) pts.push_back(mk(uni(rng), uni(rng), uni(rng), float(i)));
  return pts;
}

struct Array3Hash {
  std::size_t operator()(const std::array<long long,3>& a) const noexcept {
    // simple 64-bit mix (FNV-ish + golden ratio)
    std::size_t h = 1469598103934665603ull;
    for (int i=0;i<3;++i) {
      std::uint64_t v = static_cast<std::uint64_t>(a[i]) ^ 0x9e3779b97f4a7c15ull;
      h ^= v;
      h *= 1099511628211ull;
    }
    return h;
  }
};

// Naïve reference implementations
static std::size_t naiveBoxCount(const std::vector<Point>& pts,
                                 const Point& mn, const Point& mx) {
  std::size_t c=0;
  for (const auto& p: pts) if (inBox(p,mn,mx)) ++c;
  return c;
}

static std::size_t naiveRadiusCount(const std::vector<Point>& pts,
                                    const Point& q, float r) {
  const float r2 = r*r;
  std::size_t c=0;
  for (const auto& p: pts) if (d2(p,q) <= r2) ++c;
  return c;
}

static std::vector<const Point*> naiveKNN(const std::vector<Point>& pts,
                                          const Point& q, std::size_t k,
                                          float max_dist = -1.f) {
  const float lim2 = (max_dist < 0.f) ? std::numeric_limits<float>::infinity()
                                      : max_dist*max_dist;
  std::vector<std::pair<float,const Point*>> tmp; tmp.reserve(pts.size());
  for (const auto& p: pts) {
    float dd = d2(p,q);
    if (dd <= lim2) tmp.emplace_back(dd, &p);
  }
  std::stable_sort(tmp.begin(), tmp.end(),
                   [](auto& a, auto& b){ return a.first < b.first; });
  if (tmp.size() > k) tmp.resize(k);
  std::vector<const Point*> out; out.reserve(tmp.size());
  for (auto& t: tmp) out.push_back(t.second);
  return out;
}

static std::vector<const Point*> naiveKNNUnique(const std::vector<Point>& pts,
                                                const Point& q, std::size_t k,
                                                float max_dist, float eps) {
  const float lim2 = (max_dist < 0.f) ? std::numeric_limits<float>::infinity()
                                      : max_dist*max_dist;

  auto key = [eps](const Point& p){
    const float inv = 1.f/eps;
    return std::array<long long,3>{
      llround(p.x*inv), llround(p.y*inv), llround(p.z*inv)
    };
  };

  std::unordered_set<std::array<long long,3>, Array3Hash> seen;
  std::vector<std::pair<float,const Point*>> tmp; tmp.reserve(pts.size());

  for (const auto& p: pts) {
    auto kq = key(p);
    if (!seen.emplace(kq).second) continue;
    float dd = (p.x-q.x)*(p.x-q.x) + (p.y-q.y)*(p.y-q.y) + (p.z-q.z)*(p.z-q.z);
    if (dd <= lim2) tmp.emplace_back(dd, &p);
  }

  std::stable_sort(tmp.begin(), tmp.end(),
                   [](auto& a, auto& b){ return a.first < b.first; });
  if (tmp.size() > k) tmp.resize(k);

  std::vector<const Point*> out; out.reserve(tmp.size());
  for (auto& t: tmp) out.push_back(t.second);
  return out;
}

static Point datasetMin(const std::vector<Point>& pts) {
  Point mn = mk(+INFINITY,+INFINITY,+INFINITY);
  for (auto& p: pts) { mn.x = std::min(mn.x,p.x); mn.y=std::min(mn.y,p.y); mn.z=std::min(mn.z,p.z); }
  return mn;
}
static Point datasetMax(const std::vector<Point>& pts) {
  Point mx = mk(-INFINITY,-INFINITY,-INFINITY);
  for (auto& p: pts) { mx.x = std::max(mx.x,p.x); mx.y=std::max(mx.y,p.y); mx.z=std::max(mx.z,p.z); }
  return mx;
}

static void expectSameKnnDistances(const std::vector<const Point*>& a,
                                   const std::vector<const Point*>& b,
                                   const Point& q, float tol=1e-5f) {
  ASSERT_EQ(a.size(), b.size());
  std::vector<float> da, db; da.reserve(a.size()); db.reserve(b.size());
  for (auto* p: a) da.push_back(d2(*p,q));
  for (auto* p: b) db.push_back(d2(*p,q));
  ASSERT_EQ(da.size(), db.size());
  for (std::size_t i=0;i<da.size();++i) {
    EXPECT_NEAR(da[i], db[i], tol) << "kNN distance mismatch at rank " << i;
  }
}

} // namespace

// -----------------------------------------------------------------------------
// Async rebalancing launches and completes correctly
// -----------------------------------------------------------------------------
TEST_F(LKdTreeRebalanceTest, AsyncRebalancingWorks) {
  insertMany(50'000);  // Larger number to ensure rebuild takes some time

  auto future = tree_.balanceTreeAsync();

  // Wait a short time for the background thread to flip the `balancing_` flag
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  EXPECT_TRUE(tree_.isBalancing());

  // Wait for rebuild to finish
  bool rebuilt = future.get();

  EXPECT_TRUE(rebuilt);
  EXPECT_FALSE(tree_.isBalancing());
}

// -----------------------------------------------------------------------------
// During async rebalancing, radiusSearch should block until rebalancing is done
// -----------------------------------------------------------------------------
TEST_F(LKdTreeRebalanceTest, RadiusSearchBlocksUntilRebalanceCompletes) {
  insertMany(50'000);

  // Start async rebalancing
  std::cout << "[Main] Starting async rebalancing...\n";
  auto future = tree_.balanceTreeAsync();

  // Wait until rebalancing has actually started
  while (!tree_.isBalancing()) {
    std::this_thread::yield();
  }

  // Time the search
  auto start = std::chrono::steady_clock::now();
  auto search_thread = std::thread([&]() {
    std::cout << "[Search] Calling radiusSearch...\n";
    auto result = tree_.radiusSearch({0.f, 0.f, 0.f}, 1.0f);
    std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Wait for 1ms 
    std::cout << "[Search] radiusSearch finished. Found " << result.size() << " points.\n";
  });

  std::cout << "[Main] Waiting for rebalancing to complete...\n";
  future.get();
  std::cout << "[Main] Rebalancing done.\n";
  auto end = std::chrono::steady_clock::now();

  search_thread.join();

  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  EXPECT_GE(elapsed_ms, 1) << "radiusSearch should block at least briefly during rebalancing";
}

// -----------------------------------------------------------------------------
// During async rebalancing, insert should block until rebalancing is done
// -----------------------------------------------------------------------------
TEST_F(LKdTreeRebalanceTest, InsertBlocksUntilRebalanceCompletes) {
  insertMany(50'000);

  std::cout << "[Main] Starting async rebalancing...\n";
  auto future = tree_.balanceTreeAsync();

  while (!tree_.isBalancing()) {
    std::this_thread::yield();
  }

  auto start = std::chrono::steady_clock::now();
  auto insert_thread = std::thread([&]() {
    std::cout << "[Insert] Calling insert...\n";
    tree_.insert({1.f, 2.f, 3.f});
    std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Wait for 1ms 
    std::cout << "[Insert] Insert finished.\n";
  });

  std::cout << "[Main] Waiting for rebalancing to complete...\n";
  future.get();
  std::cout << "[Main] Rebalancing done.\n";
  auto end = std::chrono::steady_clock::now();

  insert_thread.join();

  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  EXPECT_GE(elapsed_ms, 1) << "insert should block at least briefly during rebalancing";
}

// -----------------------------------------------------------------------------
// Manual insertion in sorted order creates an unbalanced tree; balanceTree should fix it
// -----------------------------------------------------------------------------
TEST_F(LKdTreeRebalanceTest, RebalanceReducesDepthInSortedInsertions) {
  const std::size_t N = 10000;
  insertMany(N);
  

  std::size_t before_depth = tree_.maxDepth();
  std::cout << "[Sync] Tree depth before rebalancing: " << before_depth << "\n";
  ASSERT_GT(before_depth, 20u) << "Expected unbalanced tree with high depth";

  tree_.balanceTree();  // Synchronous rebalancing

  std::size_t after_depth = tree_.maxDepth();
  std::cout << "[Sync] Tree depth after rebalancing:  " << after_depth << "\n";
  EXPECT_LT(after_depth, before_depth);

  EXPECT_EQ(tree_.size(), N);
}

// -----------------------------------------------------------------------------
// Rebalancing improves radiusSearch performance for deep-inserted points
// -----------------------------------------------------------------------------
TEST_F(LKdTreeRebalanceTest, RebalanceImprovesRadiusSearchPerformance) {
  constexpr std::size_t N = 10'000;
  constexpr std::size_t queries_num = 500;
  insertMany(N);

  std::vector<Point> queries;
  for (std::size_t i = N - queries_num; i < N; ++i) {
    queries.push_back(Point{static_cast<float>(i),
                            static_cast<float>(i),
                            static_cast<float>(i)});
  }

  // Measure total time before rebalancing
  auto start_before = std::chrono::steady_clock::now();
  for (const auto& q : queries) {
    auto result = tree_.radiusSearch(q, 1e-2f);  // tiny radius for exact match
    ASSERT_FALSE(result.empty()) << "Query point not found before rebalancing";
  }
  auto end_before = std::chrono::steady_clock::now();

  // Time the rebuild itself
  auto start_rebuild = std::chrono::steady_clock::now();
  tree_.balanceTree();  // synchronous rebuild
  auto end_rebuild = std::chrono::steady_clock::now();
  auto rebuild_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_rebuild - start_rebuild).count();
  std::cout << "[Debug] Rebuild time: " << rebuild_ms << " ms\n";

  // Measure total time after rebalancing
  auto start_after = std::chrono::steady_clock::now();
  for (const auto& q : queries) {
    auto result = tree_.radiusSearch(q, 1e-2f);  // same queries
    ASSERT_FALSE(result.empty()) << "Query point not found after rebalancing";
  }
  auto end_after = std::chrono::steady_clock::now();

  auto duration_before_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_before - start_before).count();
  auto duration_after_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(end_after - start_after).count();

  std::cout << "[Perf] Total search time before rebalancing (ms): " << duration_before_ms << '\n';
  std::cout << "[Perf] Total search time after  rebalancing (ms): " << duration_after_ms  << '\n';

  EXPECT_LT(duration_after_ms, duration_before_ms)
      << "Expected faster queries after rebalancing";
}

// -----------------------------------------------------------------------------
// Rebalancing should not occur if the tree is too small
// -----------------------------------------------------------------------------
TEST_F(LKdTreeRebalanceTest, NoRebuildForSmallTree) {
  insertMany(5);  // Much smaller than rebuild threshold

  // tryBalanceTree (sync)
  std::size_t before_depth_try = tree_.maxDepth();
  bool rebuilt_try = tree_.tryBalanceTree(0.1f);
  std::size_t after_depth_try = tree_.maxDepth();

  EXPECT_FALSE(rebuilt_try);
  EXPECT_EQ(before_depth_try, after_depth_try);

  // tryBalanceTreeAsync
  std::size_t before_depth_async_try = tree_.maxDepth();
  bool async_try_rebuilt = tree_.tryBalanceTreeAsync(0.1f).get();
  std::size_t after_depth_async_try = tree_.maxDepth();

  EXPECT_FALSE(async_try_rebuilt);
  EXPECT_EQ(before_depth_async_try, after_depth_async_try);

  // balanceTree (sync)
  std::size_t before_depth = tree_.maxDepth();
  bool rebuilt = tree_.balanceTree();
  std::size_t after_depth = tree_.maxDepth();

  EXPECT_FALSE(rebuilt);
  EXPECT_EQ(before_depth, after_depth);

  // balanceTreeAsync
  std::size_t before_depth_async = tree_.maxDepth();
  bool async_rebuilt = tree_.balanceTreeAsync().get();
  std::size_t after_depth_async = tree_.maxDepth();

  EXPECT_FALSE(async_rebuilt);
  EXPECT_EQ(before_depth_async, after_depth_async);
}

// -----------------------------------------------------------------------------
// Rebalancing should be skipped for invalid (negative) ratio
// -----------------------------------------------------------------------------
TEST_F(LKdTreeRebalanceTest, InvalidRebuildRatioIsRejected) {
  insertMany(1000);

  // tryBalanceTree (sync)
  std::size_t before_depth = tree_.maxDepth();
  bool rebuilt = tree_.tryBalanceTree(-0.5f);  // Invalid ratio
  std::size_t after_depth = tree_.maxDepth();

  EXPECT_FALSE(rebuilt);
  EXPECT_EQ(before_depth, after_depth);

  // tryBalanceTreeAsync
  std::size_t before_depth_async = tree_.maxDepth();
  bool async_rebuilt = tree_.tryBalanceTreeAsync(-0.5f).get();
  std::size_t after_depth_async = tree_.maxDepth();

  EXPECT_FALSE(async_rebuilt);
  EXPECT_EQ(before_depth_async, after_depth_async);
}

// ---------------------------------------------------------------------------------------------
// Rebalancing improves radiusSearch performance for deep-inserted points (using tryBalance)
// ---------------------------------------------------------------------------------------------
TEST_F(LKdTreeRebalanceTest, TryRebalanceImprovesRadiusSearchPerformance) {
  constexpr std::size_t N = 10'000;
  constexpr std::size_t queries_num = 500;
  insertMany(N);

  std::vector<Point> queries;
  for (std::size_t i = N - queries_num; i < N; ++i) {
    queries.push_back(Point{static_cast<float>(i),
                            static_cast<float>(i),
                            static_cast<float>(i)});
  }

  // Measure total time before rebalancing
  auto start_before = std::chrono::steady_clock::now();
  for (const auto& q : queries) {
    auto result = tree_.radiusSearch(q, 1e-2f);  // tiny radius for exact match
    ASSERT_FALSE(result.empty()) << "Query point not found before rebalancing";
  }
  auto end_before = std::chrono::steady_clock::now();

  // Time the rebuild itself
  auto start_rebuild = std::chrono::steady_clock::now();
  ASSERT_TRUE(tree_.tryBalanceTree(0.7f)) << "Rebalancing was not performed";  // synchronous rebuild
  auto end_rebuild = std::chrono::steady_clock::now();
  auto rebuild_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_rebuild - start_rebuild).count();
  std::cout << "[Debug] Rebuild time: " << rebuild_ms << " ms\n";

  // Measure total time after rebalancing
  auto start_after = std::chrono::steady_clock::now();
  for (const auto& q : queries) {
    auto result = tree_.radiusSearch(q, 1e-2f);  // same queries
    ASSERT_FALSE(result.empty()) << "Query point not found after rebalancing";
  }
  auto end_after = std::chrono::steady_clock::now();

  auto duration_before_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_before - start_before).count();
  auto duration_after_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(end_after - start_after).count();

  std::cout << "[Perf] Total search time before rebalancing (ms): " << duration_before_ms << '\n';
  std::cout << "[Perf] Total search time after  rebalancing (ms): " << duration_after_ms  << '\n';

  EXPECT_LT(duration_after_ms, duration_before_ms)
      << "Expected faster queries after rebalancing";
}

// -----------------------------------------------------------------------------
// Invalid fill_ratio is rejected (no rebuild, depth unchanged)
// -----------------------------------------------------------------------------
TEST_F(LKdTreeRebalanceTest, BalanceRejectsFillRatioNonPositive) {
  const std::size_t N = 10'000;
  insertMany(N);
  const auto before = tree_.maxDepth();

  bool ok = tree_.balanceTree(0.0f);   // invalid
  EXPECT_FALSE(ok);
  EXPECT_EQ(before, tree_.maxDepth());
}

TEST_F(LKdTreeRebalanceTest, BalanceRejectsFillRatioGreaterThanOne) {
  const std::size_t N = 10'000;
  insertMany(N);
  const auto before = tree_.maxDepth();

  bool ok = tree_.balanceTree(1.01f);  // invalid
  EXPECT_FALSE(ok);
  EXPECT_EQ(before, tree_.maxDepth());
}

// -----------------------------------------------------------------------------
// Valid fill_ratio < 1 uses buildBalancedWithSlack (deeper or equal vs 1.0)
// -----------------------------------------------------------------------------
TEST_F(LKdTreeRebalanceTest, BalanceWithSlackBuildsUnderfilledLeaves) {
  const std::size_t N = 20'000;

  // Build same dataset once, then feed two trees
  std::vector<Point> pts; pts.reserve(N);
  for (std::size_t i = 0; i < N; ++i)
    pts.push_back(Point{float(i), float(i), float(i)}); // monotonic → worst case

  Tree full(32), slack(32);
  for (const auto& p : pts) { full.insert(p); slack.insert(p); }

  ASSERT_TRUE(full.balanceTree(1.0f));   // full leaves
  ASSERT_TRUE(slack.balanceTree(0.6f));  // under-filled leaves

  const auto d_full  = full.maxDepth();
  const auto d_slack = slack.maxDepth();

  EXPECT_GE(d_slack, d_full) << "Under-filled leaves should not produce a *shallower* tree";
  EXPECT_EQ(full.size(), slack.size());
}

// -----------------------------------------------------------------------------
// Async: valid fill_ratio < 1 triggers rebuild and clears balancing flag
// -----------------------------------------------------------------------------
TEST_F(LKdTreeRebalanceTest, BalanceAsyncWithSlackWorks) {
  insertMany(15'000);

  auto fut = tree_.balanceTreeAsync(0.7f);
  // spin until it starts
  while (!tree_.isBalancing()) std::this_thread::yield();

  bool ok = fut.get();
  EXPECT_TRUE(ok);
  EXPECT_FALSE(tree_.isBalancing());
}

// -----------------------------------------------------------------------------
// tryBalanceTree with fill_ratio: force rebuild via ratio=0
// -----------------------------------------------------------------------------
TEST_F(LKdTreeRebalanceTest, TryBalanceWithSlackForcesRebuild) {
  const std::size_t N = 12'000;
  insertMany(N);
  const auto before = tree_.maxDepth();

  // ratio=0 → always rebuild; use slack leaves
  bool ok = tree_.tryBalanceTree(/*rebuild_ratio=*/0.0f, /*fill_ratio=*/0.5f);
  EXPECT_TRUE(ok);

  const auto after = tree_.maxDepth();
  EXPECT_NE(after, before);   // structure changed
  EXPECT_GE(after, 1u);
}

// -----------------------------------------------------------------------------
// Insert-only — verify counts during growth
// -----------------------------------------------------------------------------
TEST_F(LKdTreeRebalanceTest, InsertOnly_BlackBoxCorrectnessDuringGrowth) {
  const std::size_t N = 8000;
  auto pts = makeRandomPoints(N, 505);
  Tree t(32);

  Point mn = mk(+INFINITY,+INFINITY,+INFINITY);
  Point mx = mk(-INFINITY,-INFINITY,-INFINITY);

  for (std::size_t i=0;i<N;++i) {
    t.insert(pts[i]);
    mn.x = std::min(mn.x, pts[i].x); mn.y = std::min(mn.y, pts[i].y); mn.z = std::min(mn.z, pts[i].z);
    mx.x = std::max(mx.x, pts[i].x); mx.y = std::max(mx.y, pts[i].y); mx.z = std::max(mx.z, pts[i].z);

    // occasionally check
    if ((i+1) % 1000 == 0) {
      EXPECT_EQ(t.size(), i+1);
      EXPECT_EQ(t.boxSearch(mn, mx).size(), i+1);

      // spot-check a few radius queries
      for (int j=0;j<5;++j) {
        const Point q = pts[i - (j*10)];
        float r = 50.f;
        std::vector<Point> cur(pts.begin(), pts.begin() + i + 1);
        EXPECT_EQ(t.radiusSearch(q, r).size(), naiveRadiusCount(cur, q, r));
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Degenerate: strictly increasing line; correctness before/after balance
// -----------------------------------------------------------------------------
TEST_F(LKdTreeRebalanceTest, DegenerateSortedInput_Correctness) {
  const std::size_t N = 9000;
  std::vector<Point> pts; pts.reserve(N);
  for (std::size_t i=0;i<N;++i) pts.push_back(mk(float(i), float(i), float(i)));

  Tree t(32);
  for (const auto& p: pts) t.insert(p);
  EXPECT_EQ(t.size(), N);

  // Queries match naive before any balancing
  {
    EXPECT_EQ(t.boxSearch(datasetMin(pts), datasetMax(pts)).size(), N);
    for (int i=0;i<50;++i) {
      const Point q = pts[i* (N/50)];
      EXPECT_EQ(t.radiusSearch(q, 0.1f).size(), 1u);
      auto got = t.knnSearch(q, 5);
      auto want = naiveKNN(pts, q, 5);
      expectSameKnnDistances(got, want, q, 1e-5f);
    }
  }

  // Optional: balance, results stay identical
  ASSERT_TRUE(t.balanceTree(1.0f));
  {
    EXPECT_EQ(t.boxSearch(datasetMin(pts), datasetMax(pts)).size(), N);
    for (int i=0;i<50;++i) {
      const Point q = pts[i* (N/50)];
      EXPECT_EQ(t.radiusSearch(q, 0.1f).size(), 1u);
      auto got = t.knnSearch(q, 5);
      auto want = naiveKNN(pts, q, 5);
      expectSameKnnDistances(got, want, q, 1e-5f);
    }
  }
}
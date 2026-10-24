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
// LKdTreeKnnSearchTest.cpp — unit tests for k-nearest-neighbour search
// -----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <random>
#include <limits>
#include <algorithm>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "lkdtree/LKdTree.h"

using Point  = pcl::PointXYZI;
using Tree   = lkdtree::LKdTree<Point, 3>;

// ──────────────────────────────────────────────────────────────────────────
// Helper utilities
// ──────────────────────────────────────────────────────────────────────────
namespace {

float sqDist(const Point& a, const Point& b)
{
  const float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
  return dx*dx + dy*dy + dz*dz;
}

Point makePt(float x, float y, float z) { return {x,y,z,0.f}; }

/* Brute-force k-NN for verification */
std::vector<const Point*> bruteKnn(const std::vector<Point>& cloud,
                                   const Point& query,
                                   std::size_t k)
{
  std::vector<const Point*> out;
  out.reserve(cloud.size());
  for (const auto& p : cloud) out.push_back(&p);

  std::partial_sort(out.begin(), out.begin()+k, out.end(),
    [&](const Point* a, const Point* b)
    { return sqDist(*a,query) < sqDist(*b,query); });

  out.resize(k);
  return out;
}

using CoordKey = std::array<int64_t, 3>;

struct CoordHash {
  std::size_t operator()(const CoordKey& a) const noexcept {
    std::size_t h = 0;
    for (auto v : a)
      h = h * 31u + std::hash<int64_t>{}(v);
    return h;
  }
};

std::vector<const Point*> bruteKnnUnique(const std::vector<Point>& cloud,
                                         const Point& query,
                                         std::size_t k,
                                         float eps = 1e-5f)
{
  std::unordered_set<CoordKey, CoordHash> seen;
  std::vector<const Point*> out;

  auto coordKey = [&](const Point& p) -> CoordKey {
    return { static_cast<int64_t>(std::floor(p.x / eps)),
             static_cast<int64_t>(std::floor(p.y / eps)),
             static_cast<int64_t>(std::floor(p.z / eps)) };
  };

  std::vector<const Point*> candidates;
  for (const auto& pt : cloud)
    candidates.push_back(&pt);

  std::sort(candidates.begin(), candidates.end(),
    [&](const Point* a, const Point* b) {
      return sqDist(*a, query) < sqDist(*b, query);
    });

  for (const Point* pt : candidates)
  {
    CoordKey key = coordKey(*pt);
    if (seen.count(key)) continue;
    seen.insert(key);
    out.push_back(pt);
    if (out.size() >= k) break;
  }

  return out;
}

// ---------------------------------------------------------------------------
// helper – squared distance given a pointer
// ---------------------------------------------------------------------------
inline float sqDistPtr(const Point* p, const Point& q)
{
  const float dx = p->x - q.x,
              dy = p->y - q.y,
              dz = p->z - q.z;
  return dx*dx + dy*dy + dz*dz;
}

// Compare two result sets (vector of ptrs) element‑wise.
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
// Fixture with helpers
// =============================================================================
class KnnFixture : public ::testing::Test
{
protected:
  Tree               tree_{32};
  std::vector<Point> cloud_;

  void addRandom(std::size_t n, int seed=2025)
  {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> d(-1000.f, 1000.f);
    for (std::size_t i=0;i<n;++i)
    {
      Point p{d(gen), d(gen), d(gen), 0.f};
      cloud_.push_back(p);
      tree_.insert(p);
    }
  }
};

// =============================================================================
// Fixture supplying a fully‑populated tree + raw cloud for reference tests
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
// 1. Invalid-input behaviour
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, KZeroOrTooLargeReturnEmpty)
{
  addRandom(10);

  Point q = makePt(0,0,0);

  EXPECT_TRUE(tree_.knnSearch(q, 0).empty());        // k == 0
  EXPECT_TRUE(tree_.knnSearch(q, 50).empty());       // k  > size()
}

TEST_F(KnnFixture, QueryWithNaNOrInfReturnsEmpty)
{
  addRandom(10);

  Point nan_q = makePt(std::numeric_limits<float>::quiet_NaN(), 0, 0);
  Point inf_q = makePt(0, 0, std::numeric_limits<float>::infinity());

  EXPECT_TRUE(tree_.knnSearch(nan_q, 3).empty());
  EXPECT_TRUE(tree_.knnSearch(inf_q , 3).empty());
}

TEST(KnnStandalone, EmptyTreeReturnsEmpty)
{
  Tree empty(16);
  Point q = makePt(1,2,3);
  EXPECT_TRUE(empty.knnSearch(q, 5).empty());
}

// -----------------------------------------------------------------------------
// 2. Deterministic small cloud – exact answers & order
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, KnnOnSmallKnownCloud)
{
  const std::vector<Point> pts = {
      makePt(0,0,0),   // d²=0
      makePt(1,0,0),   // d²=1
      makePt(0,1,0),   // d²=1
      makePt(0,0,1),   // d²=1
      makePt(2,0,0)    // d²=4
  };
  for (auto p : pts) { cloud_.push_back(p); tree_.insert(p); }

  Point q = makePt(0,0,0);
  auto knn = tree_.knnSearch(q, 3);
  ASSERT_EQ(knn.size(), 3u);

  // first neighbour must be the query itself (distance 0)
  EXPECT_NEAR(sqDistPtr(knn[0], q), 0.0f, 1e-6f);

  // the next two must have distance-squared exactly 1.0
  EXPECT_NEAR(sqDistPtr(knn[1], q), 1.0f, 1e-6f);
  EXPECT_NEAR(sqDistPtr(knn[2], q), 1.0f, 1e-6f);
}

// -----------------------------------------------------------------------------
// 3. Large random cloud – result matches brute force
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, MatchesBruteForceOnRandomCloud)
{
  constexpr std::size_t N = 5'000, K = 8;
  addRandom(N);
  Point query = makePt(10, -5, 3.3f);

  auto fast  = tree_.knnSearch(query, K);
  auto brute = bruteKnn(cloud_, query, K);

  ASSERT_EQ(fast.size(), K);
  for (std::size_t i = 0; i < K; ++i)
    EXPECT_NEAR( sqDistPtr(fast[i],  query),
                 sqDistPtr(brute[i], query), 1e-6f )
        << "Mismatch at rank " << i;
}

// -----------------------------------------------------------------------------
// 4. Very Large random cloud – result matches brute force
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, MatchesBruteForceOnRandomLargeCloud)
{
  constexpr std::size_t N = 50'000, K = 15;
  addRandom(N);
  Point query = makePt(120.0f, -55.0f, 33.3f);

  auto fast  = tree_.knnSearch(query, K);
  auto brute = bruteKnn(cloud_, query, K);

  ASSERT_EQ(fast.size(), K);
  for (std::size_t i = 0; i < K; ++i)
    EXPECT_NEAR( sqDistPtr(fast[i],  query),
                 sqDistPtr(brute[i], query), 1e-6f )
        << "Mismatch at rank " << i;
}

// -----------------------------------------------------------------------------
// 5. Multiple exact duplicates — query matches many identical points
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, QueryWithExactDuplicates)
{
  Point p = makePt(1.0f, 2.0f, 3.0f);
  constexpr std::size_t N = 100;
  for (std::size_t i = 0; i < N; ++i)
    tree_.insert(p);

  Point q = makePt(1.0f, 2.0f, 3.0f);
  auto result = tree_.knnSearch(q, 5);

  ASSERT_EQ(result.size(), 5u);
  for (auto* pt : result)
    EXPECT_NEAR(sqDist(*pt, q), 0.0f, 1e-6f);
}

// -----------------------------------------------------------------------------
// 6. All points are far — nearest results must be sorted by distance
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, AllPointsFarButSortedByDistance)
{
  const std::vector<Point> pts = {
      makePt(100,0,0), makePt(90,0,0), makePt(80,0,0),
      makePt(70,0,0),  makePt(60,0,0), makePt(50,0,0),
  };
  for (auto& p : pts) tree_.insert(p);
  Point q = makePt(0,0,0);
  auto knn = tree_.knnSearch(q, 4);

  ASSERT_EQ(knn.size(), 4u);
  for (std::size_t i = 1; i < knn.size(); ++i)
    EXPECT_LE(sqDist(*knn[i-1], q), sqDist(*knn[i], q));
}

// -----------------------------------------------------------------------------
// 7. Points are numerically very close — must still sort correctly
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, NumericallyClosePoints)
{
  for (std::size_t i = 0; i < 100; ++i)
    tree_.insert(makePt(1.0f + i * 1e-7f, 2.0f, 3.0f));

  Point q = makePt(1.0f, 2.0f, 3.0f);
  auto knn = tree_.knnSearch(q, 5);

  ASSERT_EQ(knn.size(), 5u);
  for (auto* pt : knn)
    EXPECT_NEAR(pt->y, 2.0f, 1e-6f);
}

// -----------------------------------------------------------------------------
// 8. Degenerate cloud on a line — search must still be correct
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, DegenerateLineCloud)
{
  for (int i = 0; i < 1000; ++i)
    tree_.insert(makePt(static_cast<float>(i), 0.0f, 0.0f));

  Point q = makePt(500.0f, 0.0f, 0.0f);
  auto knn = tree_.knnSearch(q, 10);

  ASSERT_EQ(knn.size(), 10u);
  for (auto* pt : knn)
    EXPECT_NEAR(pt->y, 0.0f, 1e-6f);
}

// -----------------------------------------------------------------------------
// 9. Non-power-of-two number of points — structural edge case
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, NonPowerOfTwoSize)
{
  for (int i = 0; i < 37; ++i)
    tree_.insert(makePt(i * 0.5f, i * 0.2f, i * -0.3f));

  Point q = makePt(5.0f, 2.0f, -3.0f);
  auto knn = tree_.knnSearch(q, 7);

  ASSERT_EQ(knn.size(), 7u);
  for (auto* pt : knn)
    EXPECT_TRUE(pt != nullptr);
}

// -----------------------------------------------------------------------------
// 10. Match PCL KdTree on small random cloud
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, MatchesPCLKdTreeOnSmallCloud)
{
  constexpr std::size_t N = 500, K = 5;
  addRandom(N);

  pcl::KdTreeFLANN<Point> pcl_tree;
  pcl::PointCloud<Point>::Ptr pcl_cloud(new pcl::PointCloud<Point>);
  for (const auto& p :cloud_) {
    pcl_cloud->points.push_back(p);
  }
  pcl_tree.setInputCloud(pcl_cloud);

  Point query = makePt(10.0f, -20.0f, 5.0f);
  std::vector<int> indices(K);
  std::vector<float> dists(K);
  int found = pcl_tree.nearestKSearch(query, K, indices, dists);
  ASSERT_EQ(found, K);

  auto knn = tree_.knnSearch(query, K);
  ASSERT_EQ(knn.size(), K);

  for (std::size_t i = 0; i < K; ++i)
    EXPECT_NEAR(sqDistPtr(knn[i], query), dists[i], 1e-6f)
      << "Mismatch at rank " << i;
}

// -----------------------------------------------------------------------------
// 11. Compare with PCL KdTree — large random cloud
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, MatchesPclKdTreeOnLargeCloud)
{
  constexpr std::size_t N = 100'000, K = 10;
  addRandom(N);
  Point query = makePt(-40.0f, 80.0f, -40.0f);

  tree_.balanceTree();
  auto our_knn = tree_.knnSearch(query, K);

  pcl::KdTreeFLANN<Point> pcl_tree;
  pcl::PointCloud<Point>::Ptr pcl_cloud(new pcl::PointCloud<Point>());
  for (const auto& p :cloud_) {
    pcl_cloud->points.push_back(p);
  }
  pcl_tree.setInputCloud(pcl_cloud);

  std::vector<int>   indices(K);
  std::vector<float> sq_dists(K);
  pcl_tree.nearestKSearch(query, K, indices, sq_dists);

  ASSERT_EQ(our_knn.size(), K);
  for (std::size_t i = 0; i < K; ++i)
    EXPECT_NEAR(sqDist(*our_knn[i], query), sq_dists[i], 1e-6f)
        << "Mismatch at rank " << i;
}

// -----------------------------------------------------------------------------
// 12. Unique-k-NN should return distinct coordinates even with duplicates
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, UniqueKnnSkipsExactDuplicates)
{
  Point p = makePt(0,0,0);
  for (int i=0;i<20;++i) tree_.insert(p);        // 20 duplicates

  tree_.insert(makePt(1,0,0));                   // distinct point

  auto uniq = tree_.knnSearchUnique(makePt(0,0,0), 2);
  ASSERT_EQ(uniq.size(), 2u);
  EXPECT_NEAR(sqDistPtr(uniq[0], p), 0.f, 1e-6f);        // p itself
  EXPECT_NEAR(sqDistPtr(uniq[1], p), 1.f, 1e-6f);        // (1,0,0)
}

// -----------------------------------------------------------------------------
// 13. Unique-k-NN: returns empty on invalid inputs (k=0, k>size, NaN query)
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, UniqueKnnRejectsInvalidInputs)
{
  addRandom(10);
  Point q = makePt(0,0,0);
  EXPECT_TRUE(tree_.knnSearchUnique(q, 0).empty());          // k = 0
  EXPECT_TRUE(tree_.knnSearchUnique(q, 50).empty());         // k > size()
  EXPECT_TRUE(tree_.knnSearchUnique(makePt(NAN,0,0), 3).empty());
  EXPECT_TRUE(tree_.knnSearchUnique(makePt(0,std::numeric_limits<float>::infinity(),0), 3).empty());
}

// -----------------------------------------------------------------------------
// 14. Unique-k-NN: query inside large random cloud
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, UniqueKnnOnLargeRandomCloud)
{
  constexpr std::size_t N = 20'000;
  addRandom(N);

  Point q = makePt(12.5f, -3.0f, 9.9f);
  auto uniq = tree_.knnSearchUnique(q, 10);
  ASSERT_EQ(uniq.size(), 10u);

  // Distances must increase monotonically
  for (std::size_t i = 1; i < uniq.size(); ++i)
    EXPECT_LE(sqDistPtr(uniq[i-1], q), sqDistPtr(uniq[i], q));
}

// -----------------------------------------------------------------------------
// 15. Unique-k-NN: multiple near-duplicates with small spacing
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, UniqueKnnRejectsNearbyDuplicates)
{
  for (int i = 0; i < 20; ++i)
    tree_.insert(makePt(0.0f + i * 1e-7f, 0.0f, 0.0f));

  Point query = makePt(0.0f, 0.0f, 0.0f);

  auto uniq = tree_.knnSearchUnique(query, 10, -1.0f, /* eps */ 1e-5f);

  // They are very close, so should be deduplicated
  EXPECT_LT(uniq.size(), 10u);

  for (std::size_t i = 0; i + 1 < uniq.size(); ++i)
    EXPECT_GT(sqDist(*uniq[i], *uniq[i+1]), 1e-5f);
}

// -----------------------------------------------------------------------------
// 16. Unique-k-NN: works on degenerate structure (e.g. line)
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, UniqueKnnOnLineStructure)
{
  for (int i = 0; i < 1000; ++i)
    tree_.insert(makePt(static_cast<float>(i), 0.0f, 0.0f));

  Point q = makePt(500.f, 0.f, 0.f);
  auto uniq = tree_.knnSearchUnique(q, 20);

  ASSERT_EQ(uniq.size(), 20u);
  for (std::size_t i = 1; i < uniq.size(); ++i)
    EXPECT_GT(sqDist(*uniq[i-1], *uniq[i]), 0.0f);
}

// -----------------------------------------------------------------------------
// 17. Unique-k-NN: edge case with fewer distinct points than k
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, UniqueKnnWhenFewerDistinctPointsThanK)
{
  for (int i = 0; i < 10; ++i)
    tree_.insert(makePt(0,0,0));
  for (int i = 0; i < 3; ++i)
    tree_.insert(makePt(i+1,0,0));

  auto uniq = tree_.knnSearchUnique(makePt(0,0,0), 10);
  EXPECT_EQ(uniq.size(), 4u); // only 4 distinct points total
}

// -----------------------------------------------------------------------------
// 18. Unique-k-NN: small random cloud – matches brute force results
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, UniqueKnnMatchesBruteForceSmallCloud)
{
  constexpr std::size_t N = 500, K = 5;
  addRandom(N);
  Point q = makePt(10.0f, -20.0f, 5.0f);
  float eps = 1e-5f;

  auto expected = bruteKnnUnique(cloud_, q, K, eps);
  auto actual   = tree_.knnSearchUnique(q, K, -1.0f, eps);

  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_NEAR(sqDistPtr(actual[i], q), sqDistPtr(expected[i], q), 1e-6f)
      << "Mismatch at rank " << i;
}

// -----------------------------------------------------------------------------
// 19. Unique-k-NN: large random cloud – matches brute force results
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, UniqueKnnMatchesBruteForceLargeCloud)
{
  constexpr std::size_t N = 10000, K = 10;
  addRandom(N);
  Point q = makePt(-50.0f, 42.0f, -30.0f);
  float eps = 1e-4f;

  auto expected = bruteKnnUnique(cloud_, q, K, eps);
  auto actual   = tree_.knnSearchUnique(q, K, -1.0f, eps);

  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_NEAR(sqDistPtr(actual[i], q), sqDistPtr(expected[i], q), 1e-5f)
      << "Mismatch at rank " << i;
}

// -----------------------------------------------------------------------------
// 20. knnSearchBatch: Correctness: empty query range, k==0, k>size
// -----------------------------------------------------------------------------
TEST_F(BatchFixture, HandlesInvalidInputsGracefully)
{
  // empty query list → empty result list
  std::vector<Point> queries;   // empty
  auto res = tree_.knnSearchBatch(queries, 5);
  EXPECT_TRUE(res.empty());

  // k > cloud size → empty as spec says
  res = tree_.knnSearchBatch(queries, N_POINTS+1);
  EXPECT_TRUE(res.empty());

  // NaN query returns empty sub‑vector
  queries = { makePt(NAN,0,0) };
  res = tree_.knnSearchBatch(queries, 3);
  EXPECT_TRUE(res[0].empty());
}

// -----------------------------------------------------------------------------
// 21. knnSearchBatch: Correctness vs. individual knnSearch()
// -----------------------------------------------------------------------------
TEST_F(BatchFixture, MatchesIndividualKnnSearch)
{
  constexpr std::size_t K   = 10;
  constexpr std::size_t Q_N = 1'000;
  // random queries
  std::vector<Point> queries;
  queries.reserve(Q_N);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> unif(-500.f,500.f);
  for(std::size_t i=0;i<Q_N;++i)
    queries.push_back(makePt(unif(rng),unif(rng),unif(rng)));

  // batch sequential path → num_threads=0
  tree_.balanceTree();  // ensure tree is balanced
  auto seqRes = tree_.knnSearchBatch(queries,K, -1.f, 0);
  ASSERT_EQ(seqRes.size(), Q_N);

  // compute reference by calling knnSearch one‑by‑one
  for(std::size_t qi=0; qi<Q_N; ++qi){
    auto ref = tree_.knnSearch(queries[qi],K);
    expectEqualPtrLists(ref, seqRes[qi], queries[qi]);
  }

  // parallel path: let OpenMP decide thread count
  auto parRes = tree_.knnSearchBatch(queries,K);
  ASSERT_EQ(parRes.size(), Q_N);
  for(std::size_t qi=0; qi<Q_N; ++qi)
    expectEqualPtrLists(seqRes[qi], parRes[qi], queries[qi]);
}

// -----------------------------------------------------------------------------
// 22. knnSearchBatch: Performance micro‑benchmark — parallel faster than sequential
// -----------------------------------------------------------------------------
TEST_F(BatchFixture, ParallelFasterThanSequential)
{
#if defined(_OPENMP)
  constexpr std::size_t K   = 8;
  constexpr std::size_t Q_N = 20'000;   // big enough to show difference

  // generate queries
  std::vector<Point> queries;
  queries.reserve(Q_N);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> unif(-800.f,800.f);
  for(std::size_t i=0;i<Q_N;++i)
    queries.emplace_back(makePt(unif(rng),unif(rng),unif(rng)));

  // sequential
  auto t0 = std::chrono::steady_clock::now();
  auto seq = tree_.knnSearchBatch(queries, K, -1.f, 0);   // force single‑thread
  auto t1 = std::chrono::steady_clock::now();

  // parallel (use all cores)
  auto t2 = std::chrono::steady_clock::now();
  auto par = tree_.knnSearchBatch(queries, K);      // default threads
  auto t3 = std::chrono::steady_clock::now();

  // duration
  auto seq_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  auto par_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

  std::cout << "=== knnSearchBatch Timing Comparison ===" << '\n';
  std::cout << "Total queries: " << queries.size() << ", K = " << K << '\n';
  std::cout << "Sequential time: " << seq_ms << " ms" << '\n';
  std::cout << "Parallel time:   " << par_ms  << " ms" << '\n';

  if (par_ms > 0) {
    double speedup = static_cast<double>(seq_ms) / par_ms;
    std::cout << "Speedup (seq / par): " << speedup << "x" << '\n';
  }
  std::cout << "========================================" << '\n';


  // correctness sanity
  ASSERT_EQ(seq.size(), par.size());
  for(std::size_t i=0;i<seq.size();++i)
    expectEqualPtrLists(seq[i], par[i], queries[i]);

  // performance assertion: parallel must be strictly faster
  EXPECT_LT(par_ms, seq_ms)
      << "Parallel("<<par_ms<<" ms) not faster than sequential("<<seq_ms<<" ms)";
#else
  GTEST_SKIP() << "OpenMP disabled — performance test skipped.";
#endif
}

// -----------------------------------------------------------------------------
// 23. knnSearchBatch vs. PCL KdTreeFLANN — correctness + speed
// -----------------------------------------------------------------------------
TEST_F(BatchFixture, ParallelBatchVsPCLKdTreePerformance)
{
#if defined(_OPENMP)
  constexpr std::size_t K   = 10;
  constexpr std::size_t Q_N = 2000;   // balance runtime

  // generate queries
  std::vector<Point> queries;
  queries.reserve(Q_N);
  std::mt19937 rng(9);
  std::uniform_real_distribution<float> unif(-900.f,900.f);
  for(std::size_t i=0;i<Q_N;++i)
    queries.emplace_back(makePt(unif(rng),unif(rng),unif(rng)));

  // -------------------- our parallel batch --------------------
  auto t0 = std::chrono::steady_clock::now();
  auto lkd_results = tree_.knnSearchBatch(queries, K);              // full parallel
  auto t1 = std::chrono::steady_clock::now();

  // -------------------- PCL FLANN -----------------------------
  pcl::KdTreeFLANN<Point> pcl_tree;
  pcl::PointCloud<Point>::Ptr pcl_cloud(new pcl::PointCloud<Point>());
  for (const auto& p :cloud_) {
    pcl_cloud->points.push_back(p);
  }
  pcl_tree.setInputCloud(pcl_cloud);

  std::vector<std::vector<const Point*>> pcl_results(Q_N);
  std::vector<int>   indices(K);
  std::vector<float> sq_dists(K);

  auto t2 = std::chrono::steady_clock::now();
  for(std::size_t qi=0; qi<Q_N; ++qi)
  {
    int found = pcl_tree.nearestKSearch(queries[qi], K, indices, sq_dists);
    ASSERT_EQ(found,K);
    pcl_results[qi].resize(K);
    for(std::size_t j=0;j<K;++j)
      pcl_results[qi][j] = &cloud_[indices[j]];
  }
  auto t3 = std::chrono::steady_clock::now();

  // durations
  auto lkd_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  auto pcl_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

  std::cout << "\n=== knnSearchBatch vs. PCL::KdTreeFLANN ===" << '\n';
  std::cout << "Queries="<<Q_N<<", K="<<K<<"\n"
            << "Our parallel batch: "<<lkd_ms<<" ms\n"
            << "PCL FLANN:         "<<pcl_ms <<" ms"<< '\n';
  if (pcl_ms>0)
    std::cout << "Speedup (lkd/pcl)="<< static_cast<double>(pcl_ms)/lkd_ms << "x" << '\n';
  std::cout << "===========================================" << '\n';

  // ---- correctness: distance equality ---------------------------------
  ASSERT_EQ(lkd_results.size(), pcl_results.size());
  for(std::size_t qi=0; qi<Q_N; ++qi)
    expectEqualPtrLists(lkd_results[qi], pcl_results[qi], queries[qi]);

#else
  GTEST_SKIP() << "OpenMP disabled — performance test skipped.";
#endif
}

// -----------------------------------------------------------------------------
// 24. Wrong query range type (should cause compilation error) — commented out
// -----------------------------------------------------------------------------
/*
TEST(WrongTemplateUsage, CompilationFailsForWrongType)
{
  Tree tree;
  int not_a_range = 42;
  auto result = tree.knnSearchBatch(not_a_range, 3); // invalid query type
  (void)result;
}
*/


// -----------------------------------------------------------------------------
// 25. knnSearchUniqueBatch: Correctness: empty query range, k==0, k>size
// -----------------------------------------------------------------------------
TEST_F(BatchFixture, UniqueBatchHandlesInvalidInputs)
{
  std::vector<Point> queries;
  auto res = tree_.knnSearchUniqueBatch(queries, 5);
  EXPECT_TRUE(res.empty());

  res = tree_.knnSearchUniqueBatch(queries, N_POINTS+1);
  EXPECT_TRUE(res.empty());

  queries = { makePt(NAN, 0, 0) };
  res = tree_.knnSearchUniqueBatch(queries, 3);
  EXPECT_TRUE(res[0].empty());
}

// -----------------------------------------------------------------------------
// 26. knnSearchUniqueBatch: Correctness vs. individual knnSearchUnique()
// -----------------------------------------------------------------------------
TEST_F(BatchFixture, UniqueBatchMatchesIndividual)
{
  constexpr std::size_t K   = 10;
  constexpr std::size_t Q_N = 1'000;

  std::vector<Point> queries;
  queries.reserve(Q_N);
  std::mt19937 rng(99);
  std::uniform_real_distribution<float> unif(-500.f, 500.f);
  for (std::size_t i = 0; i < Q_N; ++i)
    queries.push_back(makePt(unif(rng), unif(rng), unif(rng)));

  auto seqRes = tree_.knnSearchUniqueBatch(queries, K, -1.0f, 1e-5f, 0);
  ASSERT_EQ(seqRes.size(), Q_N);

  for (std::size_t qi = 0; qi < Q_N; ++qi) {
    auto ref = tree_.knnSearchUnique(queries[qi], K, -1.0f, 1e-5f);
    expectEqualPtrLists(ref, seqRes[qi], queries[qi]);
  }

  auto parRes = tree_.knnSearchUniqueBatch(queries, K, -1.0f, 1e-5f);
  ASSERT_EQ(parRes.size(), Q_N);
  for (std::size_t qi = 0; qi < Q_N; ++qi)
    expectEqualPtrLists(seqRes[qi], parRes[qi], queries[qi]);
}

// -----------------------------------------------------------------------------
// 27. knnSearchUniqueBatch: Performance micro-benchmark — parallel faster than sequential
// -----------------------------------------------------------------------------
TEST_F(BatchFixture, UniqueBatchParallelFasterThanSequential)
{
#if defined(_OPENMP)
  constexpr std::size_t K   = 6;
  constexpr std::size_t Q_N = 15'000;

  std::vector<Point> queries;
  queries.reserve(Q_N);
  std::mt19937 rng(111);
  std::uniform_real_distribution<float> unif(-600.f, 600.f);
  for (std::size_t i = 0; i < Q_N; ++i)
    queries.push_back(makePt(unif(rng), unif(rng), unif(rng)));

  auto t0 = std::chrono::steady_clock::now();
  auto seq = tree_.knnSearchUniqueBatch(queries, K, -1.0f, 1e-5f, 0);
  auto t1 = std::chrono::steady_clock::now();

  auto t2 = std::chrono::steady_clock::now();
  auto par = tree_.knnSearchUniqueBatch(queries, K, -1.0f, 1e-5f);
  auto t3 = std::chrono::steady_clock::now();
  
  auto seq_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  auto par_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

  std::cout << "=== knnSearchUniqueBatch Timing Comparison ===" << '\n';
  std::cout << "Total queries: " << queries.size() << ", K = " << K << '\n';
  std::cout << "Sequential time: " << seq_ms << " ms" << '\n';
  std::cout << "Parallel time:   " << par_ms  << " ms" << '\n';
  if (par_ms > 0)
    std::cout << "Speedup (seq / par): " << static_cast<double>(seq_ms) / par_ms << "x" << '\n';
  std::cout << "=============================================" << '\n';

  ASSERT_EQ(seq.size(), par.size());
  for (std::size_t i = 0; i < seq.size(); ++i)
    expectEqualPtrLists(seq[i], par[i], queries[i]);

  EXPECT_LT(par_ms, seq_ms)
      << "Parallel(" << par_ms << " ms) not faster than sequential(" << seq_ms << " ms)";
#else
  GTEST_SKIP() << "OpenMP disabled — performance test skipped.";
#endif
}

// -----------------------------------------------------------------------------
// 28 bis. max_dist : only neighbours ≤ max_dist are returned
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, RespectsMaxDistCutOff)
{
  /* Build a small synthetic cloud ------------------------------------------ */
  // ⋅⋅ q  = (0,0,0)
  // ⋅⋅ p0 = (0,0,0)    d = 0
  // ⋅⋅ p1 = (0.5,0,0)  d = 0.5
  // ⋅⋅ p2 = (1,0,0)    d = 1
  // ⋅⋅ p3 = (2,0,0)    d = 2
  tree_.insert( makePt(0,0,0) );
  tree_.insert( makePt(0.5f,0,0) );
  tree_.insert( makePt(1,0,0) );
  tree_.insert( makePt(2,0,0) );

  Point  q            = makePt(0,0,0);
  float  max_dist     = 0.75f;                   // includes p0 & p1 only
  size_t k_requested  = 4;

  auto res = tree_.knnSearch(q, k_requested, max_dist);

  ASSERT_EQ(res.size(), 2u);                     // only 2 points in range
  for (auto* p : res)
    EXPECT_LE( std::sqrt(sqDist(*p,q)), max_dist + 1e-6f );
}

// -----------------------------------------------------------------------------
// 29. max_dist : returns empty when nothing is within range
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, ReturnsEmptyWhenNoPointWithinRange)
{
  addRandom(50);                                 // cloud around origin ~1 000 × cube
  Point q = makePt(10'000, 10'000, 10'000);      // far away
  auto res = tree_.knnSearch(q, 5, /*max_dist=*/5.0f);
  EXPECT_TRUE(res.empty());
}

// -----------------------------------------------------------------------------
// 30. max_dist < 0 ⇒ “no range-limit” (back-compat behaviour)
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, NegativeMaxDistMeansUnlimited)
{
  addRandom(1'000);
  Point q = makePt(12, -7, 3);
  auto limited   = tree_.knnSearch(q, 7, /*max_dist=*/-1.0f);   // unlimited
  auto reference = tree_.knnSearch(q, 7);                       // old overload
  expectEqualPtrLists(limited, reference, q);
}

// -----------------------------------------------------------------------------
// 31. max_dist with k larger than points within range
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, KGreaterThanCandidatesWithinRange)
{
  /* Two points inside the radius .......................................... */
  tree_.insert( makePt(0,0,0) );
  tree_.insert( makePt(0.20f,0,0) );

  /* Plus some extra points **outside** the radius so that size() ≥ k ...... */
  for (int i = 0; i < 20; ++i)            // now tree.size() == 22
    tree_.insert( makePt(10.0f + i, 0, 0) );

  Point  q          = makePt(0,0,0);
  float  max_dist   = 0.25f;              // only the first 2 points qualify
  size_t k_requested = 10;                // ≤ size(), but > #in-range

  auto res = tree_.knnSearch(q, k_requested, max_dist);

  EXPECT_EQ(res.size(), 2u);              // exactly the two inside the shell
  for (auto* p : res)
    EXPECT_LE( std::sqrt(sqDist(*p,q)), max_dist + 1e-6f );
}


// -----------------------------------------------------------------------------
// 32 unique max_dist : only neighbours ≤ max_dist are returned
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, RespectsMaxDistCutOffUnique)
{
  /* Build a small synthetic cloud ------------------------------------------ */
  // ⋅⋅ q  = (0,0,0)
  // ⋅⋅ p0 = (0,0,0)    d = 0
  // ⋅⋅ p1 = (0.5,0,0)  d = 0.5
  // ⋅⋅ p2 = (1,0,0)    d = 1
  // ⋅⋅ p3 = (2,0,0)    d = 2
  tree_.insert( makePt(0,0,0) );
  tree_.insert( makePt(0.5f,0,0) );
  tree_.insert( makePt(1,0,0) );
  tree_.insert( makePt(2,0,0) );

  Point  q            = makePt(0,0,0);
  float  max_dist     = 0.75f;                   // includes p0 & p1 only
  size_t k_requested  = 4;

  auto res = tree_.knnSearchUnique(q, k_requested, max_dist, 1e-6f);

  ASSERT_EQ(res.size(), 2u);                     // only 2 points in range
  for (auto* p : res)
    EXPECT_LE( std::sqrt(sqDist(*p,q)), max_dist + 1e-6f );
}

// -----------------------------------------------------------------------------
// 33. max_dist : returns empty when nothing is within range
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, ReturnsEmptyWhenNoPointWithinRangeUnique)
{
  addRandom(50);                                 // cloud around origin ~1 000 × cube
  Point q = makePt(10'000, 10'000, 10'000);      // far away
  auto res = tree_.knnSearchUnique(q, 5, /*max_dist=*/5.0f, 1e-6f);
  EXPECT_TRUE(res.empty());
}

// -----------------------------------------------------------------------------
// 34. max_dist < 0 ⇒ “no range-limit” (back-compat behaviour)
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, NegativeMaxDistMeansUnlimitedUnique)
{
  addRandom(1'000);
  Point q = makePt(12, -7, 3);
  auto limited   = tree_.knnSearchUnique(q, 7, /*max_dist=*/-1.0f, 1e-6f);   // unlimited
  auto reference = tree_.knnSearchUnique(q, 7);                       
  expectEqualPtrLists(limited, reference, q);
}

// -----------------------------------------------------------------------------
// 35. max_dist with k larger than points within range
// -----------------------------------------------------------------------------
TEST_F(KnnFixture, KGreaterThanCandidatesWithinRangeUnique)
{
  /* Two points inside the radius .......................................... */
  tree_.insert( makePt(0,0,0) );
  tree_.insert( makePt(0.20f,0,0) );

  /* Plus some extra points **outside** the radius so that size() ≥ k ...... */
  for (int i = 0; i < 20; ++i)            // now tree.size() == 22
    tree_.insert( makePt(10.0f + i, 0, 0) );

  Point  q          = makePt(0,0,0);
  float  max_dist   = 0.25f;              // only the first 2 points qualify
  size_t k_requested = 10;                // ≤ size(), but > #in-range

  auto res = tree_.knnSearchUnique(q, k_requested, max_dist, 1e-6f);

  EXPECT_EQ(res.size(), 2u);              // exactly the two inside the shell
  for (auto* p : res)
    EXPECT_LE( std::sqrt(sqDist(*p,q)), max_dist + 1e-6f );
}
 
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

#include <benchmark/benchmark.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <random>
#include "lkdtree/LKdTree.h"

// ---------- Monotonic (pathological) point cloud ----------
static std::vector<pcl::PointXYZ>
generateMonotonicCloud(std::size_t n)
{
    std::vector<pcl::PointXYZ> cloud;
    cloud.reserve(n);
    // strictly increasing in all axes
    // (use small deltas to avoid excessive magnitudes for huge n)
    const float dx = 1.0f, dy = 1.0f, dz = 1.0f;
    float x = 0.f, y = 0.f, z = 0.f;
    for (std::size_t i = 0; i < n; ++i) {
        cloud.emplace_back(x, y, z);
        x += dx; y += dy; z += dz;
    }
    return cloud;
}

// Generate synthetic point cloud
static std::vector<pcl::PointXYZ>
generatePointCloud(std::size_t n)
{
    std::vector<pcl::PointXYZ> cloud;
    cloud.reserve(n);

    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(-1000.f, 1000.f);

    for (std::size_t i = 0; i < n; ++i)
        cloud.emplace_back(dist(rng), dist(rng), dist(rng));

    return cloud;
}


// ---- LKdTree batch KNN ----
static void BM_LKdTree_KnnSearch(benchmark::State& state) {
    const size_t N = state.range(0); // tree size
    const size_t Q = 500; // number of queries
    const int    K = 5; // K
    const int    numThreads = state.range(1);
    std::vector<pcl::PointXYZ> cloud;
    std::vector<pcl::PointXYZ> queries;
    lkdtree::LKdTree<pcl::PointXYZ,3> tree{32};

    // build + generate queries out of timed region
    cloud    = generatePointCloud(N);
    queries  = generatePointCloud(Q);
    pcl::PointCloud<pcl::PointXYZ> pc;
    for (const auto& p : cloud)
        pc.push_back(p);

    tree.buildBalancedFromPointCloud(pc, 0.6f);
    for (auto _ : state) {
        auto result = tree.knnSearchBatch(queries, K, -1.f, numThreads);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_LKdTree_KnnSearch)
    ->ArgsProduct({
        {1000,10000,100000,1000000,10000000}, // tree sizes
        {1,2,4,8,16}
    })
    ->Unit(benchmark::kMillisecond);

// ---- PCL KdTree per‐query KNN ----
static void BM_PCL_KnnSearch(benchmark::State& state) {
    const size_t N = state.range(0);
    const size_t Q = 500;
    const int    K = 5;
    std::vector<pcl::PointXYZ> cloud, queries;
    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<int>    indices(K);
    std::vector<float>  dists(K);

    cloud   = generatePointCloud(N);
    queries = generatePointCloud(Q);
    for (const auto& pt : cloud)
        pcl_cloud->points.push_back(pt);
    tree.setInputCloud(pcl_cloud);

    for (auto _ : state) {
        for (auto& q : queries) {
            tree.nearestKSearch(q, K, indices, dists);
            benchmark::DoNotOptimize(indices);
            benchmark::DoNotOptimize(dists);
        }
    }
}
BENCHMARK(BM_PCL_KnnSearch)
    ->ArgsProduct({
        {1000,10000,100000,1000000,10000000}
    })
    ->Unit(benchmark::kMillisecond);

// Entry point
BENCHMARK_MAIN();

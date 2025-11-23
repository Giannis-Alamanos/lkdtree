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

// Generate synthetic point cloud
static std::vector<pcl::PointXYZ>
generatePointCloud(std::size_t n)
{
    std::vector<pcl::PointXYZ> cloud;
    cloud.reserve(n);
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(-100000.f, 100000.f);
    for (std::size_t i = 0; i < n; ++i)
        cloud.emplace_back(dist(rng), dist(rng), dist(rng));
    return cloud;
}

// ---- LKdTree batch Radius Search ----
static void BM_LKdTree_RadiusSearch(benchmark::State& state) {
    const size_t N = state.range(0);      // tree size
    const size_t Q = 500;                 // fixed number of queries
    const float  radius = 10.0f;          // your chosen search radius
    const int    numThreads = state.range(1);
    std::vector<pcl::PointXYZ> cloud, queries;
    lkdtree::LKdTree<pcl::PointXYZ,3> tree;     
    
    // build + generate queries out of timed region
    cloud   = generatePointCloud(N);
    queries = generatePointCloud(Q);
    for (auto& pt : cloud)
        tree.insert(pt);
    tree.balanceTree(); // ensure tree is balanced before searching

    for (auto _ : state) {
        auto results = tree.radiusSearchBatch(queries, radius, numThreads);
        benchmark::DoNotOptimize(results);
    }
}
BENCHMARK(BM_LKdTree_RadiusSearch)
    ->ArgsProduct({
        {1000,10000,100000,1000000,10000000},
        {1,2,4,8,16}
    })
    ->Unit(benchmark::kMillisecond);

// ---- PCL KdTree per‐query Radius Search ----
static void BM_PCL_RadiusSearch(benchmark::State& state) {
    const size_t N = state.range(0);
    const size_t Q = 500;
    const float  radius = 10.0f;
    std::vector<pcl::PointXYZ> cloud, queries;
    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<int>    indices;
    std::vector<float>  dists;

    cloud   = generatePointCloud(N);
    queries = generatePointCloud(Q);
    for (auto& pt : cloud)
        pcl_cloud->points.push_back(pt);
    tree.setInputCloud(pcl_cloud);

    for (auto _ : state) {
        for (auto& q : queries) {
            // find all neighbors within radius
            indices.clear();
            dists.clear();
            tree.radiusSearch(q, radius, indices, dists, /* max_nn= */ std::numeric_limits<int>::max());
            benchmark::DoNotOptimize(indices);
            benchmark::DoNotOptimize(dists);
        }
    }
}
BENCHMARK(BM_PCL_RadiusSearch)
    ->ArgsProduct({
        {1000,10000,100000,1000000,10000000}
    })
    ->Unit(benchmark::kMillisecond);

// Entry point
BENCHMARK_MAIN();

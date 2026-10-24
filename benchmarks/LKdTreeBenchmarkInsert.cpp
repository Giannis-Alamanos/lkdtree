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
    std::uniform_real_distribution<float> dist(-1000.f, 1000.f);

    for (std::size_t i = 0; i < n; ++i)
        cloud.emplace_back(dist(rng), dist(rng), dist(rng));

    return cloud;
}

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

// LKdTree Build benchmark
static void BM_LKdTree_Insert(benchmark::State& state) {
    auto cloud = generatePointCloud(state.range(0));
    lkdtree::LKdTree<pcl::PointXYZ,3> tree{32};
    pcl::PointCloud<pcl::PointXYZ> pc;
    for (const auto& p : cloud)
        pc.push_back(p);
    for (auto _ : state) {
        tree.buildBalancedFromPointCloud(pc, 0.6f);
    }
}
BENCHMARK(BM_LKdTree_Insert)
->Arg(1000)
->Arg(10000)
->Arg(100000)
->Arg(1000000)
// ->Arg(10000000)
->Unit(benchmark::kMillisecond)
->Iterations(1)
->Repetitions(1);


// PCL KdTree Build benchmark
static void BM_PCL_KdTree_Insert(benchmark::State& state) {
    auto cloud = generatePointCloud(state.range(0));
    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& pt : cloud)
        pcl_cloud->points.push_back(pt);
    for (auto _ : state) {
        tree.setInputCloud(pcl_cloud);
    }
}
BENCHMARK(BM_PCL_KdTree_Insert)
->Arg(1000)
->Arg(10000)
->Arg(100000)
->Arg(1000000)
// ->Arg(10000000)
->Unit(benchmark::kMillisecond)
->Iterations(1)
->Repetitions(1);

// Entry point
BENCHMARK_MAIN();

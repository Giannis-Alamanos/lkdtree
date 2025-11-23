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
#include <pcl/filters/crop_box.h>
#include <pcl/point_types.h>

#include <random>
#include <limits>
#include <memory>

#include "lkdtree/LKdTree.h"

// ──────────────────────────────────────────────
// Synthetic points & random AABB generator
// ──────────────────────────────────────────────
static std::vector<pcl::PointXYZ>
generatePointCloud(std::size_t n)
{
    std::vector<pcl::PointXYZ> cloud;
    cloud.reserve(n);

    std::mt19937                     rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(-1'000.f, 1'000.f);

    for (std::size_t i = 0; i < n; ++i)
        cloud.emplace_back(dist(rng), dist(rng), dist(rng));
    return cloud;
}

static std::vector<std::pair<pcl::PointXYZ,pcl::PointXYZ>>
generateQueryBoxes(std::size_t q, float half_len = 5.f)
{
    std::vector<std::pair<pcl::PointXYZ,pcl::PointXYZ>> boxes;
    boxes.reserve(q);

    std::mt19937                     rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(-1'000.f, 1'000.f);

    for (std::size_t i = 0; i < q; ++i)
    {
        pcl::PointXYZ c(dist(rng), dist(rng), dist(rng));
        pcl::PointXYZ mn(c.x - half_len, c.y - half_len, c.z - half_len);
        pcl::PointXYZ mx(c.x + half_len, c.y + half_len, c.z + half_len);
        boxes.emplace_back(mn, mx);
    }
    return boxes;
}

// ──────────────────────────────────────────────
// LKdTree   – batch boxSearch
// ──────────────────────────────────────────────
static void BM_LKdTree_BoxSearch(benchmark::State& state)
{
    const std::size_t N = state.range(0);    // point count
    const std::size_t Q = 500;               // query count
    const int         threads = state.range(1);
    const float       box_half = 5.f;

    std::vector<pcl::PointXYZ>                     cloud = generatePointCloud(N);
    std::vector<std::pair<pcl::PointXYZ,pcl::PointXYZ>> boxes = generateQueryBoxes(Q, box_half);

    lkdtree::LKdTree<pcl::PointXYZ,3> tree{32};
    pcl::PointCloud<pcl::PointXYZ> pc;
    for (const auto& p : cloud)
        pc.push_back(p);
    tree.buildBalancedFromPointCloud(pc, 0.6f);

    for (auto _ : state)
    {
        auto res = tree.boxSearchBatch(boxes, threads);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_LKdTree_BoxSearch)
    ->ArgsProduct({
        {1000, 10000, 100000, 1000000, 10000000},   // N
        {1,2,4,8,16}                            // threads
    })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

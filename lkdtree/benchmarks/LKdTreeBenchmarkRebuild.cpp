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
#include <pcl/point_types.h>
#include <random>
#include "lkdtree/LKdTree.h"

#include <vector>

// Monotonic (pathological) cloud: strictly increasing x,y,z → very unbalanced.
static std::vector<pcl::PointXYZ>
generateMonotonicCloud(std::size_t n)
{
    std::vector<pcl::PointXYZ> cloud;
    cloud.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        float t = static_cast<float>(i);
        cloud.emplace_back(t, t, t);
    }
    return cloud;
}

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

/* ------------------------------------------------------------------------- */
/* LKdTree: Rebalance time from a deliberately unbalanced tree               */
/* ------------------------------------------------------------------------- */
static void BM_LKdTree_Rebalance_From_Unbalanced(benchmark::State& state)
{
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    if (N < 5) { state.SkipWithError("N must be >= 5"); return; }

    // Generate once; reused each iteration (not timed).
    const auto cloud = generatePointCloud(N);

    for (auto _ : state)
    {
        state.PauseTiming();
        {
            lkdtree::LKdTree<pcl::PointXYZ,3> tree(/*leaf_max_size=*/32);

            // Build an unbalanced tree by monotonic insertion (not timed).
            for (const auto& p : cloud) tree.insert(p);

            // Record pre‑rebuild metrics (not timed).
            state.counters["depth_pre"] = static_cast<double>(tree.maxDepth());
            state.counters["size"]      = static_cast<double>(tree.size());

            state.ResumeTiming();

            // ---- Timed section: the actual rebuild ----------------------
            bool did = tree.balanceTree();
            benchmark::DoNotOptimize(did);

            state.PauseTiming();

            // Post‑rebuild metric; still outside timed region.
            state.counters["depth_post"] = static_cast<double>(tree.maxDepth());
            if (state.counters["depth_post"] > 0.0) {
                state.counters["depth_ratio"] =
                    state.counters["depth_pre"] / state.counters["depth_post"];
            }

            // Tree is destroyed here while timing is paused.
        }
        state.ResumeTiming(); // Prepare for next iteration
    }
}

BENCHMARK(BM_LKdTree_Rebalance_From_Unbalanced)
    ->Arg(1'000)
    ->Arg(10'000)
    ->Arg(100'000)
    ->Arg(1'000'000)
    ->Arg(10'000'000) // uncomment only if your machine can handle it
    ->Unit(benchmark::kMillisecond);

// Entry point
BENCHMARK_MAIN();
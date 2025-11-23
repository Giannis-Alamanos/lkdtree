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
 
#pragma once

/**
 * @file RadiusSearch.hpp
 * @brief Template definitions for lkdtree::LKdTree radius search operations.
 */

namespace lkdtree {

// -----------------------------------------------------------------------------
// Public radiusSearch wrapper
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
std::vector<const PointT *>
LKdTree<PointT, k>::radiusSearch(const PointT &query, float radius) const 
{
  if (!isFinitePoint(query)) {
    std::cerr << "[LKdTree] radiusSearch(): Skipping invalid query point with NaN or Inf" << '\n';
    return {};
  }

  if (radius < 0.0f) {
    std::cerr << "[LKdTree] radiusSearch(): radius must be non-negative, got: "
              << radius << '\n';  
    return {};
  }

  // multiple concurrent readers are fine
  std::shared_lock lock(tree_mtx_);

  std::vector<const PointT *> result;
  radiusSearch(root_, query, radius, result, /*depth=*/0);
  return result;
}

// -----------------------------------------------------------------------------
// Recursive radius search
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
void LKdTree<PointT, k>::radiusSearch(const NodePtr &node, const PointT &query, float radius,
                                      std::vector<const PointT *> &result, std::size_t depth) const {
  if (!node) return;

  float radius_sq = radius * radius;

  if (node->is_leaf) {
    // Brute-force within the leaf bucket.
    for (const PointT &p : node->points) {
      if (distanceSquared(p, query) <= radius_sq) {
        result.emplace_back(&p);
      }
    }
    return;
  }

  // Choose axis and decide which side(s) to search.
  std::size_t axis = node->axis;
  float query_coord = PointTraits<PointT, k>::get(query, axis);

  // Always search the child that contains the query point first.
  if (query_coord < node->split_value) {
    radiusSearch(node->left, query, radius, result, depth + 1);
    // Check overlap with hypersphere; if the other side could contain points, search it too.
    if ((node->split_value - query_coord) <= radius) {
      radiusSearch(node->right, query, radius, result, depth + 1);
    }
  } 
  else {
    radiusSearch(node->right, query, radius, result, depth + 1);
    if ((query_coord - node->split_value) <= radius) {
      radiusSearch(node->left, query, radius, result, depth + 1);
    }
  }
}

// -----------------------------------------------------------------------------
// Batch radius search (thread-parallel)
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
template <PointRange<PointT> QueryRange>
std::vector<std::vector<const PointT*>>
LKdTree<PointT,k>::radiusSearchBatch(const QueryRange& queries,
                                     float             radius,
                                     int               num_threads) const
{
  // Basic argument validation 
  if (queries.empty()) {
    std::cerr << "[LKdTree] Ignoring radiusSearchBatch() with empty query set"
              << '\n';
    return {};
  }

  if (radius < 0.0f) {
    std::cerr << "[LKdTree] radiusSearchBatch(): radius must be non-negative, got: "
              << radius << '\n';
    return {};
  }

  // pre-allocate result container
  std::vector<std::vector<const PointT*>> results(queries.size());

  // sequential fallback
  if (num_threads == 0) {                             // user said “no threads”
    for (std::size_t i = 0; i < queries.size(); ++i)
      results[i] = radiusSearch(queries[i], radius);
    return results;
  }

  // OpenMP parallel loop
#if defined(_OPENMP)
  int threads_to_use = (num_threads > 0)                // explicit request
                       ? num_threads
                       : omp_get_max_threads();         // default

  #pragma omp parallel for default(none)                         \
                           shared(queries, results)              \
                           firstprivate(radius)                  \
                           schedule(static)                      \
                           if(threads_to_use > 1)                \
                           num_threads(threads_to_use)
  for (std::size_t i = 0; i < queries.size(); ++i)
    results[i] = radiusSearch(queries[i], radius);
#else
  // No OpenMP available – always sequential 
  (void) num_threads;   // suppress “unused” warning
  for (std::size_t i = 0; i < queries.size(); ++i)
    results[i] = radiusSearch(queries[i], radius);
#endif

  return results;
}

}  // namespace lkdtree
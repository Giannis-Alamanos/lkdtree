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
 * @file BoxSearch.hpp
 * @brief Template definitions for lkdtree::LKdTree box search operations.
 */

namespace lkdtree {

// -----------------------------------------------------------------------------
// Public boxSearch wrapper
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
std::vector<const PointT *>
LKdTree<PointT, k>::boxSearch(const PointT &min_pt, 
                              const PointT &max_pt) const {

  if (!isFinitePoint(min_pt) ||
      !isFinitePoint(max_pt)) {
    std::cerr << "[LKdTree] boxSearch(): Skipping invalid point given with NaN or Inf" << '\n';
    return {};
  }

  for (std::size_t i = 0; i < k; ++i)
  {
    if (PointTraits<PointT,k>::get(min_pt,i) >
        PointTraits<PointT,k>::get(max_pt,i))
    {
      std::cerr << "[LKdTree] boxSearch(): min > max on axis " << i << " – ignored" << '\n';
      return {};
    }
  }

  // multiple concurrent readers are fine
  std::shared_lock lock(tree_mtx_);

  std::vector<const PointT *> result;
  boxSearch(root_, min_pt, max_pt, result);
  return result;
}

// -----------------------------------------------------------------------------
// Recursive box search
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
void LKdTree<PointT, k>::boxSearch(const NodePtr& node,
                                   const PointT& min_pt, const PointT& max_pt,
                                   std::vector<const PointT*>& result) const
{
  if (!node) return;

  // Precompute query bounds once
  const float qmin0 = PointTraits<PointT,k>::get(min_pt,0);
  const float qmax0 = PointTraits<PointT,k>::get(max_pt,0);
  const float qmin1 = PointTraits<PointT,k>::get(min_pt,1);
  const float qmax1 = PointTraits<PointT,k>::get(max_pt,1);
  float qmin2 = 0.f, qmax2 = 0.f;
  if constexpr (k >= 3) {
    qmin2 = PointTraits<PointT,k>::get(min_pt,2);
    qmax2 = PointTraits<PointT,k>::get(max_pt,2);
  }

  // Dynamic stack to avoid overflow/dropped branches
  std::vector<const Node*> stack;
  stack.reserve(std::max<std::size_t>(32, max_depth_ * 2 + 32));
  stack.push_back(node.get());

  while (!stack.empty()) {
    const Node* n = stack.back();
    stack.pop_back();

    // AABB rejection
    bool reject = (n->bb_max[0] < qmin0) || (n->bb_min[0] > qmax0)
               || (n->bb_max[1] < qmin1) || (n->bb_min[1] > qmax1);
    if constexpr (k >= 3) {
      reject |= (n->bb_max[2] < qmin2) || (n->bb_min[2] > qmax2);
    }
    if (reject) continue;

    if (n->is_leaf) {
      const auto* p   = n->points.data();
      const auto* end = p + n->points.size();

      if constexpr (k == 2) {
        for (; p != end; ++p) {
          const float x = PointTraits<PointT,2>::get(*p,0);
          if (x < qmin0 || x > qmax0) continue;
          const float y = PointTraits<PointT,2>::get(*p,1);
          if (y >= qmin1 && y <= qmax1) result.emplace_back(p);
        }
      } else { // k == 3
        for (; p != end; ++p) {
          const float x = PointTraits<PointT,3>::get(*p,0);
          if (x < qmin0 || x > qmax0) continue;
          const float y = PointTraits<PointT,3>::get(*p,1);
          if (y < qmin1 || y > qmax1) continue;
          const float z = PointTraits<PointT,3>::get(*p,2);
          if (z >= qmin2 && z <= qmax2) result.emplace_back(p);
        }
      }
      continue;
    }

    // Internal node – choose sides
    const Node* L = n->left .get();
    const Node* R = n->right.get();

    const std::size_t axis = n->axis;
    float bmin, bmax;
    if      (axis == 0) { bmin = qmin0; bmax = qmax0; }
    else if (axis == 1) { bmin = qmin1; bmax = qmax1; }
    else /* axis==2, k>=3 */ { bmin = qmin2; bmax = qmax2; }

    // If the box lies entirely on one side of the split, go only there
    if (bmax < n->split_value) { if (L) stack.push_back(L); continue; }
    if (bmin > n->split_value){ if (R) stack.push_back(R); continue; }

    // Otherwise, both sides might intersect — push both with quick AABB checks
    if (R) {
      bool rejR = (R->bb_max[0] < qmin0) || (R->bb_min[0] > qmax0)
               || (R->bb_max[1] < qmin1) || (R->bb_min[1] > qmax1);
      if constexpr (k >= 3) rejR |= (R->bb_max[2] < qmin2) || (R->bb_min[2] > qmax2);
      if (!rejR) stack.push_back(R);
    }
    if (L) {
      bool rejL = (L->bb_max[0] < qmin0) || (L->bb_min[0] > qmax0)
               || (L->bb_max[1] < qmin1) || (L->bb_min[1] > qmax1);
      if constexpr (k >= 3) rejL |= (L->bb_max[2] < qmin2) || (L->bb_min[2] > qmax2);
      if (!rejL) stack.push_back(L);
    }
  }
}

// -----------------------------------------------------------------------------
// Batch box search (thread-parallel)
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
template <BoxRange<PointT> QueryRange>
std::vector<std::vector<const PointT*>>   
LKdTree<PointT,k>::boxSearchBatch(const QueryRange& boxes,
                                  int               num_threads) const
{
  // Basic argument validation
  if (boxes.empty()) {
    std::cerr << "[LKdTree] Ignoring boxSearchBatch() with empty box query set"
              << '\n';
    return {};
  }

  // pre-allocate result container 
  std::vector<std::vector<const PointT*>> results(boxes.size());

  // sequential fallback 
  if (num_threads == 0) {                             // user said “no threads”
    for (std::size_t i = 0; i < boxes.size(); ++i)
      results[i] = boxSearch(boxes[i].first, boxes[i].second);
    return results;
  }

  // OpenMP parallel loop 
#if defined(_OPENMP)
  int threads_to_use = (num_threads > 0)                // explicit request
                       ? num_threads
                       : omp_get_max_threads();         // default

  #pragma omp parallel for default(none)                         \
                           shared(boxes, results)                \
                           schedule(static)                      \
                           if(threads_to_use > 1)                \
                           num_threads(threads_to_use)
  for (std::size_t i = 0; i < boxes.size(); ++i)
    results[i] = boxSearch(boxes[i].first, boxes[i].second);    
#else
  // No OpenMP available – always sequential
  (void)num_threads;   // suppress “unused” warning
  for (std::size_t i = 0; i < boxes.size(); ++i)
    results[i] = boxSearch(boxes[i].first, boxes[i].second);
#endif

  return results;
}

} // namespace lkdtree
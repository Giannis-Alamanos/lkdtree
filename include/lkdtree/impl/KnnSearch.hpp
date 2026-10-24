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
 * @file KnnSearch.hpp
 * @brief Template definitions for lkdtree::LKdTree k-nearest neighbor search operations.
 */

namespace lkdtree {

// -----------------------------------------------------------------------------
// k-nearest-neighbour search (public wrapper)
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
std::vector<const PointT*>
LKdTree<PointT, k>::knnSearch(const PointT& query, 
                              std::size_t k_neigh,
                              float max_dist) const
{
  // Basic argument validation 
  if (k_neigh == 0 || k_neigh > size()) {
    std::cerr << "[LKdTree] knnSearch(): invalid k=" << k_neigh
              << "; must be in [1, " << size() << "]" << '\n';
    return {};
  }
  if (!isFinitePoint(query)) {
    std::cerr << "[LKdTree] Ignoring knnSearch() with invalid (NaN/Inf) point"
              << '\n';
    return {};
  }

  // Search 
  std::shared_lock lock(tree_mtx_);  // many readers – one writer

  using Pair = std::pair<float,const PointT*>;
  std::vector<Pair> buf;                     
  buf.reserve(k_neigh + 1);                 

  std::priority_queue<Pair,std::vector<Pair>,std::less<>> heap(std::less<>(),
                                                              std::move(buf));
  knnSearch(root_, query, k_neigh, max_dist, heap);

  // Pop from max-heap → reverse into ascending-distance order
  std::vector<const PointT*> out(heap.size());
  for (std::size_t i = heap.size(); i-- > 0; ) {
    out[i] = heap.top().second;
    heap.pop();
  }
  return out;
}

// -----------------------------------------------------------------------------
// k-NN recursive helper
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
void
LKdTree<PointT,k>::knnSearch(const NodePtr& node,
                             const PointT&  query,
                             std::size_t    k_neigh,
                             float          max_dist,
                             std::priority_queue<
                               std::pair<float,const PointT*>,
                               std::vector<std::pair<float,const PointT*>>,
                               std::less<>>& heap) const
{
  if (!node) return;

  // Preload query coords
  float q0=0.f,q1=0.f,q2=0.f;
  if constexpr (k >= 1) q0 = PointTraits<PointT,k>::get(query,0);
  if constexpr (k >= 2) q1 = PointTraits<PointT,k>::get(query,1);
  if constexpr (k >= 3) q2 = PointTraits<PointT,k>::get(query,2);

  const bool  has_limit = (max_dist >= 0.f);
  const float max_d2    = has_limit ? max_dist * max_dist
                                    : std::numeric_limits<float>::infinity();

  // Small‑k fast buffer 
  constexpr std::size_t KBUF = 64;    // Size threshold for small buffer
  const bool use_small = (k_neigh <= KBUF);

  struct NB { float d2; const PointT* p; };
  NB sbuf[KBUF]; std::size_t sbuf_size = 0;
  float tau = std::numeric_limits<float>::infinity();
  tau = std::min(tau, max_d2);
  std::size_t worst_idx = 0; // index of current max (only valid when full)

  auto small_insert = [&](float d2, const PointT* p) {
    if (sbuf_size < k_neigh) {
      sbuf[sbuf_size++] = {d2, p};
      if (sbuf_size == k_neigh) {
        // compute initial worst (max d2) and tau
        worst_idx = 0;
        float w = sbuf[0].d2;
        for (std::size_t i=1;i<k_neigh;++i)
          if (sbuf[i].d2 > w) { w = sbuf[i].d2; worst_idx = i; }
        tau = w;
        tau = std::min(tau, max_d2);
      }
      return;
    }
    if (d2 >= tau) return;            // not better than current worst
    // replace worst and recompute worst
    sbuf[worst_idx] = {d2, p};
    worst_idx = 0;
    float w = sbuf[0].d2;
    for (std::size_t i=1;i<k_neigh;++i)
      if (sbuf[i].d2 > w) { w = sbuf[i].d2; worst_idx = i; }
    tau = w;
    tau = std::min(tau, max_d2);
  };

  // When not using small buffer, we mirror τ from the heap.
  auto refresh_tau_from_heap = [&]() {
    tau = (use_small)
        ? tau
        : ((heap.size() == k_neigh) ? heap.top().first
                                    : std::numeric_limits<float>::infinity());
    tau = std::min(tau, max_d2);
  };

  // Local helpers
  auto aabb_dist2 = [&](const Node* n)->float {
    float d2 = 0.f;
    if constexpr (k >= 1) {
      if (q0 < n->bb_min[0]) { float t = q0 - n->bb_min[0]; d2 += t*t; }
      else if (q0 > n->bb_max[0]) { float t = q0 - n->bb_max[0]; d2 += t*t; }
    }
    if constexpr (k >= 2) {
      if (q1 < n->bb_min[1]) { float t = q1 - n->bb_min[1]; d2 += t*t; }
      else if (q1 > n->bb_max[1]) { float t = q1 - n->bb_max[1]; d2 += t*t; }
    }
    if constexpr (k >= 3) {
      if (q2 < n->bb_min[2]) { float t = q2 - n->bb_min[2]; d2 += t*t; }
      else if (q2 > n->bb_max[2]) { float t = q2 - n->bb_max[2]; d2 += t*t; }
    }
    if constexpr (k > 3) {
      for (std::size_t i=3;i<k;++i) {
        float qi = PointTraits<PointT,k>::get(query,i);
        if      (qi < n->bb_min[i]) { float t = qi - n->bb_min[i]; d2 += t*t; }
        else if (qi > n->bb_max[i]) { float t = qi - n->bb_max[i]; d2 += t*t; }
      }
    }
    return d2;
  };

  auto leaf_scan_smallk = [&](const Node* n) {
    if constexpr (k == 2) {
      for (const PointT& p : n->points) {
        float dx = PointTraits<PointT,2>::get(p,0) - q0;
        float dy = PointTraits<PointT,2>::get(p,1) - q1;
        float d2 = dx*dx + dy*dy;
        if (d2 > max_d2) continue; 
        if (sbuf_size < k_neigh || d2 < tau) small_insert(d2, &p);
      }
    } 
    else if constexpr (k == 3) {
      for (const PointT& p : n->points) {
        float dx = PointTraits<PointT,3>::get(p,0) - q0;
        float dy = PointTraits<PointT,3>::get(p,1) - q1;
        float d2 = dx*dx + dy*dy; 
        if (d2 > max_d2) continue;
        if (d2 >= tau && sbuf_size==k_neigh) continue;
        float dz = PointTraits<PointT,3>::get(p,2) - q2;
        d2 += dz*dz;
        if (d2 > max_d2) continue;
        if (sbuf_size < k_neigh || d2 < tau) small_insert(d2, &p);
      }
    } 
    else {
      for (const PointT& p : n->points) {
        float d2 = 0.f;
        for (std::size_t i=0;i<k;++i) {
          float d = PointTraits<PointT,k>::get(p,i) - PointTraits<PointT,k>::get(query,i);
          d2 += d*d;
          if (d2 > max_d2) goto next_point;
          if (sbuf_size==k_neigh && d2 >= tau) goto next_point;
        } 
        if (sbuf_size < k_neigh || d2 < tau) small_insert(d2, &p);
        next_point: ;
      }
    }
  };

  auto leaf_scan_heap = [&](const Node* n) {
    refresh_tau_from_heap();
    if constexpr (k == 2) {
      for (const PointT& p : n->points) {
        float dx = PointTraits<PointT,2>::get(p,0) - q0;
        float dy = PointTraits<PointT,2>::get(p,1) - q1;
        float d2 = dx*dx + dy*dy;
        if (d2 > max_d2) continue; 
        if (heap.size() < k_neigh) {
          heap.emplace(d2, &p);
          if (heap.size()==k_neigh) tau = heap.top().first;
        } 
        else if (d2 < tau) {
          heap.pop(); heap.emplace(d2, &p); tau = heap.top().first;
        }
      }
    } 
    else if constexpr (k == 3) {
      for (const PointT& p : n->points) {
        float dx = PointTraits<PointT,3>::get(p,0) - q0;
        float dy = PointTraits<PointT,3>::get(p,1) - q1;
        float d2 = dx*dx + dy*dy;
        if (d2 > max_d2) continue;
        if (d2 >= tau && heap.size()==k_neigh) continue;
        float dz = PointTraits<PointT,3>::get(p,2) - q2;
        d2 += dz*dz;
        if (d2 > max_d2) continue;
        if (heap.size() < k_neigh) {
          heap.emplace(d2, &p);
          if (heap.size()==k_neigh) tau = heap.top().first;
        } 
        else if (d2 < tau) {
          heap.pop(); heap.emplace(d2, &p); tau = heap.top().first;
        }
      }
    } 
    else {
      for (const PointT& p : n->points) {
        float d2 = 0.f;
        for (std::size_t i=0;i<k;++i) {
          float d = PointTraits<PointT,k>::get(p,i) - PointTraits<PointT,k>::get(query,i);
          d2 += d*d;
          if (d2 > max_d2) goto next_point;
          if (heap.size()==k_neigh && d2 >= tau) goto next_point;
        }
        if (heap.size() < k_neigh) {
          heap.emplace(d2, &p);
          if (heap.size()==k_neigh) tau = heap.top().first;
        } 
        else if (d2 < tau) {
          heap.pop(); heap.emplace(d2, &p); tau = heap.top().first;
        }
        next_point: ;
      }
    }
  };

  // Iterative traversal (near-first)
  struct StackItem { const Node* n; };
  std::vector<StackItem> istack;
  istack.resize(max_depth_ + 1);       
  std::size_t sp = 0;
  istack[sp++] = { node.get() };

  while (sp) {
    const Node* n = istack[--sp].n;

    // AABB prune of the whole subtree if we already have k
    if ( (use_small && sbuf_size == k_neigh) || (!use_small && heap.size()==k_neigh) ) {
      float d2b = aabb_dist2(n);
      if (d2b >= tau) continue;
    }

    if (n->is_leaf) {
      if (use_small) leaf_scan_smallk(n);
      else           leaf_scan_heap(n);
      continue;
    }

    // Internal node: choose visit order by child AABB distances
    const Node* left  = n->left.get();
    const Node* right = n->right.get();
    if (!left && !right) continue;  // should not happen, but be safe

    refresh_tau_from_heap();

    // Compute child AABB d2 and decide order (nearer first)
    float d2L = left  ? aabb_dist2(left)  : std::numeric_limits<float>::infinity();
    float d2R = right ? aabb_dist2(right) : std::numeric_limits<float>::infinity();

    // Visit both sides that can beat tau; push farther first so nearer is popped next
    if (d2L <= d2R) {
      if (right && d2R < tau) istack[sp++] = { right };
      if (left  && d2L < tau) istack[sp++] = { left  };
    } 
    else {
      if (left  && d2L < tau) istack[sp++] = { left  };
      if (right && d2R < tau) istack[sp++] = { right };
    }
  }

  // If we used the small‑k buffer, move results into the heap (caller expects it)
  if (use_small) {
    for (std::size_t i=0;i<sbuf_size;++i)
      heap.emplace(sbuf[i].d2, sbuf[i].p);
  }
}

// -----------------------------------------------------------------------------
// Batch k-nearest-neighbour search (thread-parallel)
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
template <PointRange<PointT> QueryRange>
std::vector<std::vector<const PointT*>>
LKdTree<PointT,k>::knnSearchBatch(const QueryRange& queries,
                                  std::size_t       k_neigh,
                                  float             max_dist,
                                  int               num_threads) const
{
  // Basic argument validation
  if (queries.empty()) {
    std::cerr << "[LKdTree] Ignoring knnSearchBatch() with empty query set"
              << '\n';
    return {};
  }

  if (k_neigh == 0 || k_neigh > size()) {
    std::cerr << "[LKdTree] knnSearchBatch(): invalid k=" << k_neigh
              << "; must be in [1, " << size() << "]" << '\n';
    return {};
  }

  // pre-allocate result container
  std::vector<std::vector<const PointT*>> results(queries.size());

  // sequential fallback 
  if (num_threads == 0) {                             // no threads
    for (std::size_t i = 0; i < queries.size(); ++i)
      results[i] = knnSearch(queries[i], k_neigh, max_dist);
    return results;
  }

// OpenMP parallel loop 
#if defined(_OPENMP)
  int threads_to_use = (num_threads > 0)                // explicit request
                       ? num_threads
                       : omp_get_max_threads();         // default

  #pragma omp parallel for default(none)                         \
                           shared(queries, results)              \
                           firstprivate(k_neigh , max_dist)      \
                           schedule(static)                      \
                           if(threads_to_use > 1)                \
                           num_threads(threads_to_use)
  for (std::size_t i = 0; i < queries.size(); ++i)
    results[i] = knnSearch(queries[i], k_neigh, max_dist);
#else
  // No OpenMP available – always sequential 
  (void) num_threads;   // suppress “unused” warning
  for (std::size_t i = 0; i < queries.size(); ++i)
    results[i] = knnSearch(queries[i], k_neigh, max_dist);
#endif

  return results;
}

// -----------------------------------------------------------------------------
// k-nearest-neighbour search *without duplicate coordinates* (public wrapper)
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
std::vector<const PointT*>
LKdTree<PointT,k>::knnSearchUnique(const PointT& query,
                                   std::size_t   k_neigh,
                                   float         max_dist,
                                   float         eps) const
{
  // Basic argument validation
  if (k_neigh == 0 || k_neigh > size()) {
    std::cerr << "[LKdTree] knnSearch(): invalid k=" << k_neigh
              << "; must be in [1, " << size() << "]" << '\n';
    return {};
  }
  if (!isFinitePoint(query)) {
    std::cerr << "[LKdTree] Ignoring knnSearch() with invalid (NaN/Inf) point"
              << '\n';
    return {};
  }

  if (eps <= 0.0f) {
    std::cerr << "[LKdTree] knnSearchUnique(): eps must be positive, got: "
              << eps << '\n';
    return {};
  }

  // Search 
  std::shared_lock lock(tree_mtx_);           // parallel readers OK

  using Pair = std::pair<float,const PointT*>;
  std::vector<Pair> buf;                     
  buf.reserve(k_neigh + 1);  

  std::priority_queue<Pair,std::vector<Pair>,std::less<>> heap(std::less<>(),
                                                              std::move(buf));
  std::unordered_set<std::array<int64_t, k>, CoordHash> seen;

  knnSearchUnique(root_, query, k_neigh, max_dist, eps, heap, seen);

  // Pop from max-heap → reverse into ascending-distance order
  std::vector<const PointT*> out(heap.size());
  for (std::size_t i = heap.size(); i-- > 0; ) {
    out[i] = heap.top().second;
    heap.pop();
  }
  return out;
}

// -----------------------------------------------------------------------------
// Recursive helper for unique-k-NN
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
void LKdTree<PointT,k>::knnSearchUnique(
        const NodePtr&  node,
        const PointT&   query,
        std::size_t     k_neigh,
        float           max_dist,
        float           eps,
        std::priority_queue<
          std::pair<float,const PointT*>,
          std::vector<std::pair<float,const PointT*>>,
          std::less<>>& heap,
        std::unordered_set<std::array<int64_t,k>,
                           CoordHash>& seen) const
{
  if (!node) return;

  // --- cache query coords
  float q0=0.f, q1=0.f, q2=0.f;
  if constexpr (k >= 1) q0 = PointTraits<PointT,k>::get(query,0);
  if constexpr (k >= 2) q1 = PointTraits<PointT,k>::get(query,1);
  if constexpr (k >= 3) q2 = PointTraits<PointT,k>::get(query,2);

  const bool  has_limit = (max_dist >= 0.f);
  const float max_d2    = has_limit ? max_dist * max_dist
                                    : std::numeric_limits<float>::infinity();

  // --- small-k fast buffer
  constexpr std::size_t KBUF = 64;
  const bool use_small = (k_neigh <= KBUF);

  struct NB { float d2; const PointT* p; };
  NB   sbuf[KBUF];
  std::size_t sbuf_size = 0;
  float tau = std::min(std::numeric_limits<float>::infinity(), max_d2);
  std::size_t worst_idx = 0;

  // quantize point -> uniqueness key
  auto make_key = [&](const PointT& p) {
    std::array<int64_t,k> key{};
    const float inv = 1.0f / eps; 
    for (std::size_t i=0;i<k;++i) {
      const float v = PointTraits<PointT,k>::get(p,i) * inv;
      key[i] = static_cast<int64_t>(std::llround(v));
    }
    return key;
  };

  auto small_insert_raw = [&](float d2, const PointT* p) {
    if (sbuf_size < k_neigh) {
      sbuf[sbuf_size++] = {d2, p};
      if (sbuf_size == k_neigh) {
        worst_idx = 0;
        float w = sbuf[0].d2;
        for (std::size_t i=1;i<k_neigh;++i)
          if (sbuf[i].d2 > w) { w = sbuf[i].d2; worst_idx = i; }
        tau = std::min(w, max_d2);
      }
      return;
    }
    if (d2 >= tau) return; // not better
    sbuf[worst_idx] = {d2, p};
    worst_idx = 0;
    float w = sbuf[0].d2;
    for (std::size_t i=1;i<k_neigh;++i)
      if (sbuf[i].d2 > w) { w = sbuf[i].d2; worst_idx = i; }
    tau = std::min(w, max_d2);
  };

  // wrappers that enforce uniqueness only when we are going to insert
  auto small_try_unique = [&](float d2, const PointT* p) {
    if (sbuf_size < k_neigh || d2 < tau) {
      auto key = make_key(*p);
      if (seen.insert(key).second) small_insert_raw(d2, p);
    }
  };

  auto refresh_tau_from_heap = [&]() {
    tau = use_small ? tau
                    : ((heap.size() == k_neigh) ? heap.top().first
                                                : std::numeric_limits<float>::infinity());
    tau = std::min(tau, max_d2);
  };

  auto heap_try_unique_insert = [&](float d2, const PointT* p) {
    refresh_tau_from_heap();
    if (d2 > tau) return;
    auto key = make_key(*p);
    if (!seen.insert(key).second) return; // duplicate → skip
    if (heap.size() < k_neigh) {
      heap.emplace(d2, p);
      if (heap.size() == k_neigh) tau = heap.top().first;
    } else { // d2 < tau
      heap.pop();
      heap.emplace(d2, p);
      tau = heap.top().first;
    }
  };

  // AABB distance (with cached q0/q1/q2 fast path)
  auto aabb_dist2 = [&](const Node* n)->float {
    float d2 = 0.f;
    if constexpr (k >= 1) {
      if (q0 < n->bb_min[0]) { float t = q0 - n->bb_min[0]; d2 += t*t; }
      else if (q0 > n->bb_max[0]) { float t = q0 - n->bb_max[0]; d2 += t*t; }
    }
    if constexpr (k >= 2) {
      if (q1 < n->bb_min[1]) { float t = q1 - n->bb_min[1]; d2 += t*t; }
      else if (q1 > n->bb_max[1]) { float t = q1 - n->bb_max[1]; d2 += t*t; }
    }
    if constexpr (k >= 3) {
      if (q2 < n->bb_min[2]) { float t = q2 - n->bb_min[2]; d2 += t*t; }
      else if (q2 > n->bb_max[2]) { float t = q2 - n->bb_max[2]; d2 += t*t; }
    }
    if constexpr (k > 3) {
      for (std::size_t i=3;i<k;++i) {
        float qi = PointTraits<PointT,k>::get(query,i);
        if      (qi < n->bb_min[i]) { float t = qi - n->bb_min[i]; d2 += t*t; }
        else if (qi > n->bb_max[i]) { float t = qi - n->bb_max[i]; d2 += t*t; }
      }
    }
    return d2;
  };

  // leaf scans (compute d2, prune by max_d2/tau, then try unique insert)
  auto leaf_scan_smallk = [&](const Node* n) {
    if constexpr (k == 2) {
      for (const PointT& p : n->points) {
        float dx = PointTraits<PointT,2>::get(p,0) - q0;
        float dy = PointTraits<PointT,2>::get(p,1) - q1;
        float d2 = dx*dx + dy*dy;
        if (d2 > max_d2) continue;
        small_try_unique(d2, &p);
      }
    } else if constexpr (k == 3) {
      for (const PointT& p : n->points) {
        float dx = PointTraits<PointT,3>::get(p,0) - q0;
        float dy = PointTraits<PointT,3>::get(p,1) - q1;
        float d2 = dx*dx + dy*dy;
        if (d2 > max_d2) continue;
        if (sbuf_size==k_neigh && d2 >= tau) continue;
        float dz = PointTraits<PointT,3>::get(p,2) - q2;
        d2 += dz*dz;
        if (d2 > max_d2) continue;
        small_try_unique(d2, &p);
      }
    } else {
      for (const PointT& p : n->points) {
        float d2 = 0.f;
        for (std::size_t i=0;i<k;++i) {
          float d = PointTraits<PointT,k>::get(p,i) - PointTraits<PointT,k>::get(query,i);
          d2 += d*d;
          if (d2 > max_d2) goto next_point;
          if (sbuf_size==k_neigh && d2 >= tau) goto next_point;
        }
        small_try_unique(d2, &p);
        next_point: ;
      }
    }
  };

  auto leaf_scan_heap = [&](const Node* n) {
    refresh_tau_from_heap();
    if constexpr (k == 2) {
      for (const PointT& p : n->points) {
        float dx = PointTraits<PointT,2>::get(p,0) - q0;
        float dy = PointTraits<PointT,2>::get(p,1) - q1;
        float d2 = dx*dx + dy*dy;
        if (d2 > max_d2) continue;
        if (heap.size() < k_neigh || d2 < tau)
          heap_try_unique_insert(d2, &p);
      }
    } else if constexpr (k == 3) {
      for (const PointT& p : n->points) {
        float dx = PointTraits<PointT,3>::get(p,0) - q0;
        float dy = PointTraits<PointT,3>::get(p,1) - q1;
        float d2 = dx*dx + dy*dy;
        if (d2 > max_d2) continue;
        if (heap.size()==k_neigh && d2 >= tau) continue;
        float dz = PointTraits<PointT,3>::get(p,2) - q2;
        d2 += dz*dz;
        if (d2 > max_d2) continue;
        if (heap.size() < k_neigh || d2 < tau)
          heap_try_unique_insert(d2, &p);
      }
    } else {
      for (const PointT& p : n->points) {
        float d2 = 0.f;
        for (std::size_t i=0;i<k;++i) {
          float d = PointTraits<PointT,k>::get(p,i) - PointTraits<PointT,k>::get(query,i);
          d2 += d*d;
          if (d2 > max_d2) goto next_point;
          if (heap.size()==k_neigh && d2 >= tau) goto next_point;
        }
        heap_try_unique_insert(d2, &p);
        next_point: ;
      }
    }
  };

  // --- iterative near-first traversal
  struct StackItem { const Node* n; };
  std::vector<StackItem> istack;
  istack.resize(max_depth_ + 1);
  std::size_t sp = 0;
  istack[sp++] = { node.get() };

  while (sp) {
    const Node* n = istack[--sp].n;

    // AABB prune if we already have k
    if ( (use_small && sbuf_size == k_neigh) || (!use_small && heap.size()==k_neigh) ) {
      float d2b = aabb_dist2(n);
      if (d2b >= tau) continue;
    }

    if (n->is_leaf) {
      if (use_small) leaf_scan_smallk(n);
      else           leaf_scan_heap(n);
      continue;
    }

    const Node* left  = n->left.get();
    const Node* right = n->right.get();
    if (!left && !right) continue;

    refresh_tau_from_heap();

    float d2L = left  ? aabb_dist2(left)  : std::numeric_limits<float>::infinity();
    float d2R = right ? aabb_dist2(right) : std::numeric_limits<float>::infinity();

    if (d2L <= d2R) {
      if (right && d2R < tau) istack[sp++] = { right };
      if (left  && d2L < tau) istack[sp++] = { left  };
    } else {
      if (left  && d2L < tau) istack[sp++] = { left  };
      if (right && d2R < tau) istack[sp++] = { right };
    }
  }

  // move small buffer to heap if used
  if (use_small) {
    for (std::size_t i=0;i<sbuf_size;++i)
      heap.emplace(sbuf[i].d2, sbuf[i].p);
  }
}

// -----------------------------------------------------------------------------
// Batch unique k-nearest-neighbour search (thread-parallel)
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
template <PointRange<PointT> QueryRange>
std::vector<std::vector<const PointT*>>
LKdTree<PointT,k>::knnSearchUniqueBatch(const QueryRange& queries,
                                        std::size_t       k_neigh,
                                        float             max_dist,
                                        float             eps,
                                        int               num_threads) const
{
  // Basic argument validation
  if (queries.empty()) {
    std::cerr << "[LKdTree] Ignoring knnSearchUniqueBatch() with empty query set"
              << '\n';
    return {};
  }

  if (k_neigh == 0 || k_neigh > size()) {
    std::cerr << "[LKdTree] knnSearchUniqueBatch(): invalid k=" << k_neigh
              << "; must be in [1, " << size() << "]" << '\n';
    return {};
  }

  if (eps <= 0.0f) {
    std::cerr << "[LKdTree] knnSearchUniqueBatch(): eps must be positive, got: "
              << eps << '\n';
    return {};
  }

  // pre-allocate result container 
  std::vector<std::vector<const PointT*>> results(queries.size());

  // sequential fallback 
  if (num_threads == 0) {                             // user said “no threads”
    for (std::size_t i = 0; i < queries.size(); ++i)
      results[i] = knnSearchUnique(queries[i], k_neigh, max_dist, eps);
    return results;
  }

  // OpenMP parallel loop 
#if defined(_OPENMP)
  int threads_to_use = (num_threads > 0)                // explicit request
                       ? num_threads
                       : omp_get_max_threads();         // default

  #pragma omp parallel for default(none)                         \
                           shared(queries, results)              \
                           firstprivate(k_neigh, max_dist, eps)  \
                           schedule(static)                      \
                           if(threads_to_use > 1)                \
                           num_threads(threads_to_use)
  for (std::size_t i = 0; i < queries.size(); ++i)
    results[i] = knnSearchUnique(queries[i], k_neigh, max_dist, eps);
#else
  // No OpenMP available – always sequential
  (void)num_threads;   // suppress “unused” warning
  for (std::size_t i = 0; i < queries.size(); ++i)
    results[i] = knnSearchUnique(queries[i], k_neigh, max_dist, eps);
#endif

  return results;
}

} // namespace lkdtree
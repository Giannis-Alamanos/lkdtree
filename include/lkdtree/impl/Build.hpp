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
 * @file Build.hpp
 * @brief Template definitions for lkdtree::LKdTree build operations.
 */

namespace lkdtree {

// -----------------------------------------------------------------------------
// Build balanced tree from point cloud
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
template <PointCloudLike<PointT> PointCloud>
std::size_t
LKdTree<PointT,k>::buildBalancedFromPointCloud(const PointCloud& point_cloud, float fill_ratio)
{
  if (invalidFillRatio(fill_ratio)) return 0;
  std::unique_lock lock(tree_mtx_);

  std::vector<PointT> pts;
  std::vector<uint64_t> ids;
  pts.reserve(point_cloud.points.size());
  ids.reserve(point_cloud.points.size());

  std::size_t skipped = 0;
  for (const auto& p : point_cloud.points) {
    if (isFinitePoint(p)) 
    {
      pts.emplace_back(p);
      ids.emplace_back(next_insert_id_++);
    }
    else ++skipped;
  }
  if (skipped) {
    std::cerr << "[LKdTree] buildBalanced(): skipped " << skipped
              << " invalid points (NaN/Inf)\n";
  }

  max_depth_ = 0;
  if (pts.empty()) {
    root_ = std::make_unique<Node>();
    total_points_ = 0;
    points_at_last_rebuild_ = 0;
    return 0;
  }

  const std::size_t n_valid = pts.size();

  if (fill_ratio >= 0.999f) {
    root_ = buildBalanced(pts, ids, 0);
  } 
  else {
    root_ = buildBalancedWithSlack(pts, ids, 0, fill_ratio);
  }

  total_points_ = n_valid;
  points_at_last_rebuild_ = n_valid;
  return n_valid;
}

// -----------------------------------------------------------------------------
// Build a perfectly balanced tree via median splitting
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
auto LKdTree<PointT, k>::buildBalanced(std::vector<PointT>& pts,
                                       std::vector<uint64_t>& ids,
                                       std::size_t depth) -> NodePtr
{
  if (pts.empty()) return nullptr;

  // Index view over [0..pts.size())
  std::vector<std::size_t> order(pts.size());
  std::iota(order.begin(), order.end(), 0);

  using Traits = PointTraits<PointT, k>;

  // recurse over order[lo,hi)
  auto build_range = [&](auto&& self, std::size_t lo, std::size_t hi, std::size_t d) -> NodePtr
  {
    const std::size_t n = hi - lo;
    if (n == 0) return nullptr;
    max_depth_ = std::max(max_depth_, d);

    // --- Leaf
    if (n <= leaf_max_size_) {
      auto leaf = std::make_unique<Node>();
      leaf->is_leaf = true;
      leaf->points.reserve(n);
      leaf->ids.reserve(n);

      // move slice into leaf and build bbox on the fly
      for (std::size_t i = lo; i < hi; ++i) {
        const std::size_t idx = order[i];
        leaf->points.emplace_back(std::move(pts[idx]));
        leaf->ids.emplace_back(std::move(ids[idx]));
        expandBox(*leaf, leaf->points.back());
      }
      leaf->subtree_size = n;
      return leaf;
    }

    // --- Internal
    const std::size_t axis = d % k;
    const std::size_t mid  = lo + n/2;

    auto idx_comp = [&](std::size_t i, std::size_t j){
      return Traits::get(pts[i], axis) < Traits::get(pts[j], axis);
    };
    std::nth_element(order.begin()+lo, order.begin()+mid, order.begin()+hi, idx_comp);

    const float split = Traits::get(pts[ order[mid] ], axis);

    auto node = std::make_unique<Node>();
    node->is_leaf     = false;
    node->axis        = axis;
    node->split_value = split;

    node->left  = self(self, lo,  mid, d+1);
    node->right = self(self, mid, hi,  d+1);

    setBoxFromChildren(*node, node->left.get(), node->right.get());
    node->subtree_size =
        (node->left  ? node->left->subtree_size  : 0) +
        (node->right ? node->right->subtree_size : 0);
    return node;
  };

  return build_range(build_range, 0, pts.size(), depth);
}

// -----------------------------------------------------------------------------
// Build a balanced-enough tree with under-filled leaves
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
auto LKdTree<PointT, k>::buildBalancedWithSlack(std::vector<PointT>& pts,
                                                std::vector<uint64_t>& ids,
                                                std::size_t depth,
                                                float fill_ratio) -> NodePtr
{
  if (pts.empty()) return nullptr;

  const std::size_t leaf_fill_threshold =
      std::max<std::size_t>(1,
        static_cast<std::size_t>(std::ceil(leaf_max_size_ * fill_ratio)));

  // pack point+id together to keep them in sync while partitioning
  struct BuildEntry { PointT point; uint64_t id; };

  std::vector<BuildEntry> build_data;
  build_data.reserve(pts.size());
  for (std::size_t i = 0; i < pts.size(); ++i)
    build_data.push_back(BuildEntry{std::move(pts[i]), ids[i]});

  // free incoming vectors
  std::vector<PointT>().swap(pts);
  std::vector<uint64_t>().swap(ids);

  using Coord = PointTraits<PointT, k>;

  auto make_leaf = [&](std::size_t first, std::size_t last_excl) -> NodePtr {
    const std::size_t n = last_excl - first;

    auto leaf      = std::make_unique<Node>();
    leaf->is_leaf  = true;
    leaf->points.reserve(leaf_max_size_);
    leaf->ids.reserve(leaf_max_size_);

    float bb_min[k], bb_max[k];
    for (std::size_t dim = 0; dim < k; ++dim) {
      bb_min[dim] =  std::numeric_limits<float>::infinity();
      bb_max[dim] = -std::numeric_limits<float>::infinity();
    }

    for (std::size_t i = first; i < last_excl; ++i) {
      leaf->points.emplace_back(std::move(build_data[i].point));
      leaf->ids.emplace_back(build_data[i].id);
      const PointT& p = leaf->points.back();
      for (std::size_t dim = 0; dim < k; ++dim) {
        const float v = Coord::get(p, dim);
        if (v < bb_min[dim]) bb_min[dim] = v;
        if (v > bb_max[dim]) bb_max[dim] = v;
      }
    }

    for (std::size_t dim = 0; dim < k; ++dim) {
      leaf->bb_min[dim] = bb_min[dim];
      leaf->bb_max[dim] = bb_max[dim];
    }
    leaf->subtree_size = n;
    return leaf;
  };

  std::function<NodePtr(std::size_t,std::size_t,std::size_t)> build =
    [&](std::size_t first, std::size_t last_excl, std::size_t depth_local) -> NodePtr
  {
    const std::size_t n = last_excl - first;
    if (n == 0) return nullptr;
    if (depth_local > max_depth_) max_depth_ = depth_local;

    if (n <= leaf_fill_threshold) {
      return make_leaf(first, last_excl);
    }

    const std::size_t axis = depth_local % k;
    const std::size_t mid  = first + (n >> 1);

    auto compare_by_axis = [&](const BuildEntry& A, const BuildEntry& B) {
      return Coord::get(A.point, axis) < Coord::get(B.point, axis);
    };
    std::nth_element(build_data.begin() + first,
                     build_data.begin() + mid,
                     build_data.begin() + last_excl,
                     compare_by_axis);

    const float split = Coord::get(build_data[mid].point, axis);

    auto node          = std::make_unique<Node>();
    node->is_leaf      = false;
    node->axis         = axis;
    node->split_value  = split;

    node->left  = build(first, mid,      depth_local + 1);
    node->right = build(mid,   last_excl, depth_local + 1);

    setBoxFromChildren(*node, node->left.get(), node->right.get());
    node->subtree_size =
      (node->left  ? node->left->subtree_size  : 0) +
      (node->right ? node->right->subtree_size : 0);
    return node;
  };

  NodePtr root;

  root = build(0, build_data.size(), depth);


  return root;
}

// -----------------------------------------------------------------------------
// Build a balanced-enough tree with under-filled leaves (thread-parallel)
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
auto LKdTree<PointT, k>::buildBalancedWithSlackParallel(std::vector<PointT>& pts,
                                                std::vector<uint64_t>& ids,
                                                std::size_t depth,
                                                float fill_ratio,
                                                std::size_t point_num_parallel_thr,
                                                std::size_t point_num_task_thr,
                                                int num_threads) -> NodePtr
{
  if (pts.empty()) return nullptr;

  const std::size_t leaf_fill_threshold =
      std::max<std::size_t>(1,
        static_cast<std::size_t>(std::ceil(leaf_max_size_ * fill_ratio)));

  // pack point+id together to keep them in sync while partitioning
  struct BuildEntry { PointT point; uint64_t id; };

  std::vector<BuildEntry> build_data;
  build_data.reserve(pts.size());
  for (std::size_t i = 0; i < pts.size(); ++i)
    build_data.push_back(BuildEntry{std::move(pts[i]), ids[i]});

  // free incoming vectors early (we moved from them)
  std::vector<PointT>().swap(pts);
  std::vector<uint64_t>().swap(ids);

  using Coord = PointTraits<PointT, k>;

  auto make_leaf = [&](std::size_t first, std::size_t last_excl) -> NodePtr {
    const std::size_t n = last_excl - first;

    auto leaf      = std::make_unique<Node>();
    leaf->is_leaf  = true;
    leaf->points.reserve(leaf_max_size_);
    leaf->ids.reserve(leaf_max_size_);

    float bb_min[k], bb_max[k];
    for (std::size_t dim = 0; dim < k; ++dim) {
      bb_min[dim] =  std::numeric_limits<float>::infinity();
      bb_max[dim] = -std::numeric_limits<float>::infinity();
    }

    for (std::size_t i = first; i < last_excl; ++i) {
      leaf->points.emplace_back(std::move(build_data[i].point));
      leaf->ids.emplace_back(build_data[i].id);
      const PointT& p = leaf->points.back();
      for (std::size_t dim = 0; dim < k; ++dim) {
        const float v = Coord::get(p, dim);
        if (v < bb_min[dim]) bb_min[dim] = v;
        if (v > bb_max[dim]) bb_max[dim] = v;
      }
    }

    for (std::size_t dim = 0; dim < k; ++dim) {
      leaf->bb_min[dim] = bb_min[dim];
      leaf->bb_max[dim] = bb_max[dim];
    }
    leaf->subtree_size = n;
    return leaf;
  };

  std::function<NodePtr(std::size_t,std::size_t,std::size_t)> build =
    [&](std::size_t first, std::size_t last_excl, std::size_t depth_local) -> NodePtr
  {
    const std::size_t n = last_excl - first;
    if (n == 0) return nullptr;

    #if defined(_OPENMP)
      #pragma omp critical(lkdtree_update_max_depth)
    #endif
    {
      if (depth_local > max_depth_) max_depth_ = depth_local;
    }

    if (n <= leaf_fill_threshold) {
      return make_leaf(first, last_excl);
    }

    const std::size_t axis = depth_local % k;
    const std::size_t mid  = first + (n >> 1);

    auto compare_by_axis = [&](const BuildEntry& A, const BuildEntry& B) {
      return Coord::get(A.point, axis) < Coord::get(B.point, axis);
    };
    std::nth_element(build_data.begin() + first,
                     build_data.begin() + mid,
                     build_data.begin() + last_excl,
                     compare_by_axis);

    const float split = Coord::get(build_data[mid].point, axis);

    auto node          = std::make_unique<Node>();
    node->is_leaf      = false;
    node->axis         = axis;
    node->split_value  = split;

    #if defined(_OPENMP)
      if (n >= point_num_task_thr) 
      {
        NodePtr left_subtree, right_subtree;

        #pragma omp task shared(left_subtree) firstprivate(first, mid, depth_local) \
                        if(n >= (point_num_task_thr<<1)) mergeable
        { left_subtree  = build(first, mid, depth_local + 1); }

        #pragma omp task shared(right_subtree) firstprivate(mid, last_excl, depth_local) \
                        if(n >= (point_num_task_thr<<1)) mergeable
        { right_subtree = build(mid, last_excl, depth_local + 1); }

        #pragma omp taskwait
        node->left  = std::move(left_subtree);
        node->right = std::move(right_subtree);
      } 
      else
    #endif
      {
        node->left  = build(first, mid,      depth_local + 1);
        node->right = build(mid,   last_excl, depth_local + 1);
      }

    setBoxFromChildren(*node, node->left.get(), node->right.get());
    node->subtree_size =
      (node->left  ? node->left->subtree_size  : 0) +
      (node->right ? node->right->subtree_size : 0);
    return node;
  };

  NodePtr root;

  #if defined(_OPENMP)
  {
    const int threads_to_use = (num_threads > 0 && num_threads <= omp_get_max_threads()) 
      ? num_threads : omp_get_max_threads();
    const bool create_team   = (threads_to_use > 1) && (build_data.size() >= point_num_parallel_thr);

    if (create_team) 
    {
      #pragma omp parallel num_threads(threads_to_use)
      {
        #pragma omp single nowait
        { root = build(0, build_data.size(), depth); }
      }
    } 
    else 
    {
      root = build(0, build_data.size(), depth);
    }
  }
  #else
    (void)num_threads; (void)point_num_parallel_thr; (void)point_num_task_thr;
    root = build(0, build_data.size(), depth);
  #endif

  return root;
}

// -----------------------------------------------------------------------------
// Full tree rebuild (forced balancing)
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
bool LKdTree<PointT, k>::balanceTree(float fill_ratio) {
  if (tooSmallToRebuild()) return false;
  if (invalidFillRatio(fill_ratio)) return false;

  performRebuild(fill_ratio);
  return true;
}

// -----------------------------------------------------------------------------
// Asynchronous full tree rebuild (forced balancing)
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
std::future<bool> LKdTree<PointT, k>::balanceTreeAsync(float fill_ratio) {
  return std::async(std::launch::async, [this, fill_ratio]() {
    return this->balanceTree(fill_ratio);  // synchronous inside async
  });
}

// -----------------------------------------------------------------------------
// Conditional rebuild (based on imbalance or growth)
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
bool LKdTree<PointT, k>::tryBalanceTree(float rebuild_ratio, float fill_ratio) {
  if (tooSmallToRebuild()) return false;
  if (invalidRebuildRatio(rebuild_ratio)) return false;
  if (invalidFillRatio(fill_ratio)) return false;

  if(tryRebuild(rebuild_ratio, fill_ratio)) return true;

  return false;
}

// -----------------------------------------------------------------------------
// Asynchronous conditional rebuild (based on imbalance or growth)
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
std::future<bool> LKdTree<PointT, k>::tryBalanceTreeAsync(float rebuild_ratio, float fill_ratio) {
  return std::async(std::launch::async, [this, rebuild_ratio, fill_ratio]() {
    return this->tryBalanceTree(rebuild_ratio, fill_ratio);  // synchronous logic
  });
}

// -----------------------------------------------------------------------------
// Flatten all tree points into a vector (in-order traversal)
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
void LKdTree<PointT, k>::flatten(const NodePtr &node,
                                 std::vector<PointT> &out_pts,
                                 std::vector<uint64_t> &out_ids) const
{
  if (!node) return;
  if (node->is_leaf) {
    out_pts.insert(out_pts.end(), node->points.begin(), node->points.end());
    out_ids.insert(out_ids.end(), node->ids.begin(), node->ids.end());
  } 
  else {
    flatten(node->left , out_pts, out_ids);
    flatten(node->right, out_pts, out_ids);
  }
}

// -----------------------------------------------------------------------------
// Flatten all tree points into a vector and keep only those with id >= cutoff_id
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
void LKdTree<PointT,k>::flattenSinceId(const NodePtr& node,
                                       uint64_t       cutoff_id,
                                       std::vector<PointT>&   out_pts,
                                       std::vector<uint64_t>& out_ids) const
{
  if (!node) return;

  if (node->is_leaf) {
    const auto& ids = node->ids;
    const auto& pts = node->points;
    const std::size_t n = ids.size();
    for (std::size_t i = 0; i < n; ++i) {
      if (ids[i] >= cutoff_id) {
        out_pts.emplace_back(pts[i]);
        out_ids.emplace_back(ids[i]);
      }
    }
    return;
  }

  flattenSinceId(node->left , cutoff_id, out_pts, out_ids);
  flattenSinceId(node->right, cutoff_id, out_pts, out_ids);
}

// -----------------------------------------------------------------------------
// Attempt rebuild if tree is unbalanced or grew significantly
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
bool LKdTree<PointT, k>::tryRebuild(float rebuild_ratio, float fill_ratio) {
  if (!needsRebuild(rebuild_ratio)) return false;

  performRebuild(fill_ratio);
  return true;
}

// -----------------------------------------------------------------------------
// Perform the actual rebuild of the LKD-tree
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
void LKdTree<PointT, k>::performRebuild(float fill_ratio) {
  std::unique_lock lock(tree_mtx_);  // Ensure exclusive access during rebuild
  balancing_.store(true, std::memory_order_release);

  // Balance the tree by rebuilding it from scratch.
  std::vector<PointT>   pts;
  std::vector<uint64_t> ids;
  pts.reserve(total_points_);
  ids.reserve(total_points_);
  max_depth_ = 0;  // Reset max depth for the new tree
  flatten(root_, pts, ids);

  if (fill_ratio >= 0.999f) root_ = buildBalanced(pts, ids, 0);
  else                      root_ = buildBalancedWithSlack(pts, ids, 0, fill_ratio);

  points_at_last_rebuild_ = total_points_;

  balancing_.store(false, std::memory_order_release);
}

} // namespace lkdtree
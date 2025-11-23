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
 * @file Insert.hpp
 * @brief Template definitions for lkdtree::LKdTree insertion operations.
 */

namespace lkdtree {

// -----------------------------------------------------------------------------
// Public insert wrapper
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
void LKdTree<PointT, k>::insert(const PointT &pt) {
  if (!isFinitePoint(pt)) {
    std::cerr << "[LKdTree] Skipping invalid point with NaN or Inf (insert)" << '\n';
    return;
  }
  std::unique_lock lock(tree_mtx_);     // <── exclusive
  insert(root_, pt, /*depth=*/0);
  ++total_points_;
}

// -----------------------------------------------------------------------------
// Public insert maintain wrapper
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
void LKdTree<PointT,k>::insertMaintain(const PointT& pt, float local_rebuild_ratio,
                               std::size_t min_subtree_for_local_rebuild,
                               std::size_t max_subtree_for_local_rebuild,
                               float fill_ratio, std::size_t keep_recent_limit,
                               std::size_t max_allowed_size, float high_imbalance_limit,
                               std::size_t full_rebuild_size, std::size_t point_num_parallel_thr,
                               std::size_t point_num_task_thr, int num_threads) 
{
  if (!isFinitePoint(pt)) {
    std::cerr << "[LKdTree] Skipping invalid point with NaN or Inf (insertMaintain)" << '\n';
    return;
  }  
  if (local_rebuild_ratio <= 1) {
    std::cerr << "[LKdTree] Local rebuild ratio must be greater than 1 (insertMaintain)" << '\n';
    return;   
  }
  if (min_subtree_for_local_rebuild <= 0) {
    std::cerr << "[LKdTree] Minimum subtree size for local rebuild must be positive (insertMaintain)" << '\n';
    return;   
  }
  if (max_subtree_for_local_rebuild <= 0) {
    std::cerr << "[LKdTree] Maximum subtree size for local rebuild must be positive (insertMaintain)" << '\n';
    return;   
  }
  if (max_subtree_for_local_rebuild <= min_subtree_for_local_rebuild) {
    std::cerr << "[LKdTree] Maximum subtree size for local rebuild must be greater than minimum " 
              << "subtree size for local rebuild (insertMaintain)" << '\n';
    return;   
  }
  if (fill_ratio <= 0 || fill_ratio > 1) {
    std::cerr << "[LKdTree] Fill ratio must be between 0 exclusive and 1 inclusive (insertMaintain)" << '\n';
    return;   
  }
  if (keep_recent_limit <= 0) {
    std::cerr << "[LKdTree] Keep recent limit must be positive (insertMaintain)" << '\n';
    return;   
  }
  if (max_allowed_size <= 0) {
    std::cerr << "[LKdTree] Maximum allowed tree size must be positive (insertMaintain)" << '\n';
    return;   
  }
  if (high_imbalance_limit <= 1) {
    std::cerr << "[LKdTree] High imbalance limit must be greater than 1 (insertMaintain)" << '\n';
    return;   
  }
  if (full_rebuild_size <= 0) {
    std::cerr << "[LKdTree] Full rebuild size must be positive (insertMaintain)" << '\n';
    return;   
  }
  if (point_num_parallel_thr <= 0) {
    std::cerr << "[LKdTree] Minimum tree size (point_num_parallel_thr) for multithreaded "
              << "insertion must be positive. (insertMaintain)" << '\n';
    return;   
  } 
  if (point_num_task_thr <= 0) {
    std::cerr << "[LKdTree] Minimum subtree size (point_num_task_thr) "
              << "for task separation must be positive. (insertMaintain)" << '\n';
    return;   
  } 

  std::unique_lock lock(tree_mtx_);
  ++total_points_;
  insertMaintain(root_, pt, /*depth=*/0,
                 local_rebuild_ratio,
                 min_subtree_for_local_rebuild,
                 max_subtree_for_local_rebuild,
                 fill_ratio,
                 keep_recent_limit,
                 max_allowed_size,
                 high_imbalance_limit,
                 full_rebuild_size, point_num_parallel_thr,
                 point_num_task_thr, num_threads);
}

// -----------------------------------------------------------------------------
// Public insertPointCloud wrapper 
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
template <PointCloudLike<PointT> PointCloud>
std::size_t
LKdTree<PointT, k>::insertPointCloud(const PointCloud& point_cloud)
{
  std::unique_lock lock(tree_mtx_);  // one mutex for the whole batch

  std::size_t added = 0;
  for (const auto& point : point_cloud.points)
  {
    if (!isFinitePoint(point))
    {
      std::cerr << "[LKdTree] Skipping invalid point with NaN or Inf (insertPointCloud)" << '\n';
      continue;
    }

    insert(root_, point, /*depth=*/0); 
    ++added;
  }

  total_points_ += added;
  return added;
}

// -----------------------------------------------------------------------------
// Public insertPointCloudMaintain wrapper 
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
template <PointCloudLike<PointT> PointCloud>
void LKdTree<PointT,k>::insertPointCloudMaintain(const PointCloud& point_cloud, float local_rebuild_ratio,
                               std::size_t min_subtree_for_local_rebuild,
                               std::size_t max_subtree_for_local_rebuild,
                               float fill_ratio, std::size_t keep_recent_limit,
                               std::size_t max_allowed_size, float high_imbalance_limit,
                               std::size_t full_rebuild_size, std::size_t point_num_parallel_thr,
                               std::size_t point_num_task_thr, int num_threads) 
{
  if (local_rebuild_ratio <= 1) {
    std::cerr << "[LKdTree] Local rebuild ratio must be greater than 1 (insertPointCloudMaintain)" << '\n';
    return;   
  }
  if (min_subtree_for_local_rebuild <= 0) {
    std::cerr << "[LKdTree] Minimum subtree size for local rebuild must be positive (insertPointCloudMaintain)" << '\n';
    return;   
  }
  if (max_subtree_for_local_rebuild <= 0) {
    std::cerr << "[LKdTree] Maximum subtree size for local rebuild must be positive (insertPointCloudMaintain)" << '\n';
    return;   
  }
  if (max_subtree_for_local_rebuild <= min_subtree_for_local_rebuild) {
    std::cerr << "[LKdTree] Maximum subtree size for local rebuild must be greater than minimum " 
              << "subtree size for local rebuild (insertPointCloudMaintain)" << '\n';
    return;   
  }
  if (fill_ratio <= 0 || fill_ratio > 1) {
    std::cerr << "[LKdTree] Fill ratio must be between 0 exclusive and 1 inclusive (insertPointCloudMaintain)" << '\n';
    return;
  }
  if (keep_recent_limit <= 0) {
    std::cerr << "[LKdTree] Keep recent limit must be positive (insertPointCloudMaintain)" << '\n';
    return;
  }
  if (max_allowed_size <= 0) {
    std::cerr << "[LKdTree] Maximum allowed tree size must be positive (insertPointCloudMaintain)" << '\n';
    return;
  }
  if (high_imbalance_limit <= 1) {
    std::cerr << "[LKdTree] High imbalance limit must be greater than 1 (insertPointCloudMaintain)" << '\n';
    return;
  }
  if (full_rebuild_size <= 0) {
    std::cerr << "[LKdTree] Full rebuild size must be positive (insertPointCloudMaintain)" << '\n';
    return;
  }
  if (point_num_parallel_thr <= 0) {
    std::cerr << "[LKdTree] Minimum tree size (point_num_parallel_thr) for multithreaded "
              << "insertion must be positive. (insertPointCloudMaintain)" << '\n';
    return;
  } 
  if (point_num_task_thr <= 0) {
    std::cerr << "[LKdTree] Minimum subtree size (point_num_task_thr) "
              << "for task separation must be positive. (insertPointCloudMaintain)" << '\n';
    return;
  }

  std::unique_lock lock(tree_mtx_);
  
  for (const auto& point : point_cloud.points)
  {
    if (!isFinitePoint(point)) {
      std::cerr << "[LKdTree] Skipping invalid point with NaN or Inf (insertPointCloudMaintain)" << '\n';
      continue;
    }
    
    ++total_points_;
    insertMaintain(root_, point, /*depth=*/0,
                  local_rebuild_ratio,
                  min_subtree_for_local_rebuild,
                  max_subtree_for_local_rebuild,
                  fill_ratio,
                  keep_recent_limit,
                  max_allowed_size,
                  high_imbalance_limit,
                  full_rebuild_size, point_num_parallel_thr,
                  point_num_task_thr, num_threads);
  }
}

// -----------------------------------------------------------------------------
// Recursive insertion
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
void LKdTree<PointT, k>::insert(NodePtr& node, const PointT& pt, std::size_t depth)
{
  Node* cur = node.get();
  std::size_t d = depth;

  while (true)
  {
    // maintain depth & bbox on the path
    if (d > max_depth_) max_depth_ = d;
    expandBox(*cur, pt);
    cur->subtree_size++;

    if (cur->is_leaf)
    {
      cur->points.emplace_back(pt);
      cur->ids.emplace_back(next_insert_id_++);

      if (cur->points.size() <= leaf_max_size_) return;

      // -------- split the leaf --------

      std::size_t axis = d % k;
      auto& pts = cur->points;
      auto& ids = cur->ids;
      const std::size_t n   = pts.size();
      const std::size_t mid = n >> 1;

      // Build index view and nth_element it by the chosen axis
      std::vector<std::size_t> order(n);
      std::iota(order.begin(), order.end(), 0);
      auto idx_comp = [&](std::size_t i, std::size_t j) {
        return PointTraits<PointT,k>::get(pts[i], axis) <
               PointTraits<PointT,k>::get(pts[j], axis);
      };
      std::nth_element(order.begin(),
                       order.begin() + static_cast<std::ptrdiff_t>(mid),
                       order.end(),
                       idx_comp);

      // pivot value becomes split plane
      const float split = PointTraits<PointT,k>::get(pts[order[mid]], axis);

      // create children and move points/ids by index (no extra comparisons)
      cur->is_leaf     = false;
      cur->axis        = axis;
      cur->split_value = split;

      cur->left  = std::make_unique<Node>();
      cur->right = std::make_unique<Node>();
      Node* L = cur->left.get();
      Node* R = cur->right.get();

      L->points.reserve(mid);        L->ids.reserve(mid);
      R->points.reserve(n - mid);    R->ids.reserve(n - mid);

      // move & expand child AABBs
      for (std::size_t i = 0; i < mid; ++i) {
        const std::size_t idx = order[i];
        L->points.emplace_back(std::move(pts[idx]));
        L->ids.emplace_back(std::move(ids[idx]));
        expandBox(*L, L->points.back());
      }
      for (std::size_t i = mid; i < n; ++i) {
        const std::size_t idx = order[i];
        R->points.emplace_back(std::move(pts[idx]));
        R->ids.emplace_back(std::move(ids[idx]));
        expandBox(*R, R->points.back());
      }

      // exact subtree sizes for children
      L->subtree_size = L->points.size();
      R->subtree_size = R->points.size();

      // free parent leaf storage
      std::vector<PointT>().swap(pts);
      std::vector<uint64_t>().swap(ids);

      if (d + 1 > max_depth_) max_depth_ = d + 1;

      return;
    }

    // internal node: descend
    const float v = PointTraits<PointT,k>::get(pt, cur->axis);
    cur = (v < cur->split_value) ? cur->left.get() : cur->right.get();
    d++;
  }
}

// -----------------------------------------------------------------------------
// Insert + maintain balance with predictive local rebuild
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
void LKdTree<PointT,k>::insertMaintain(NodePtr& slot, const PointT& pt,
                                       std::size_t depth,
                                       float local_imbalance,
                                       std::size_t min_subtree,
                                       std::size_t max_subtree,
                                       float fill_ratio,
                                       std::size_t keep_recent_limit,
                                       std::size_t max_allowed_size,
                                       float high_imbalance_limit,
                                       std::size_t full_rebuild_size,
                                       std::size_t point_num_parallel_thr,
                                       std::size_t point_num_task_thr, 
                                       int num_threads)
{
  // Build a new root from the most recent points.
  auto rebuild_root_keep_last = [&]()
  {
    std::vector<PointT>   pts_recent;
    std::vector<uint64_t> ids_recent;

    // Choose cutoff by ID so we keep the newest 'full_rebuild_size' points.
    uint64_t cutoff_id = 0;
    if (next_insert_id_ > full_rebuild_size)
      cutoff_id = next_insert_id_ - full_rebuild_size;

    pts_recent.reserve(full_rebuild_size + 1);
    ids_recent.reserve(full_rebuild_size + 1);

    flattenSinceId(root_, cutoff_id, pts_recent, ids_recent);

    pts_recent.emplace_back(pt);
    ids_recent.emplace_back(next_insert_id_++);

    max_depth_ = 0;
    NodePtr new_root =
        (fill_ratio >= 0.999f)
          ? buildBalanced(pts_recent, ids_recent, /*depth=*/0)
          : buildBalancedWithSlackParallel(
                pts_recent, ids_recent,
                /*depth=*/0, fill_ratio,
                point_num_parallel_thr,
                point_num_task_thr,
                num_threads);

    if (!new_root) new_root = std::make_unique<Node>();

    root_ = std::move(new_root);

    // Update counters.
    total_points_ = root_->subtree_size;
    points_at_last_rebuild_ = total_points_;
  };

  if (total_points_ > max_allowed_size)
  {
    rebuild_root_keep_last();
    return;
    // continue to insert the new point into the new root
  }

  NodePtr* cur_slot = &slot;
  if (!*cur_slot) *cur_slot = std::make_unique<Node>();
  Node* cur = cur_slot->get();
  std::size_t d = depth;

  std::vector<Node*> parents;
  parents.reserve(max_depth_ + 8);

  while (true)
  {
    // maintain depth & bbox on the path
    if (d > max_depth_) max_depth_ = d;
    expandBox(*cur, pt);
    cur->subtree_size++;

    if (cur->is_leaf)
    {
      cur->points.emplace_back(pt);
      cur->ids.emplace_back(next_insert_id_++);

      if (cur->points.size() <= leaf_max_size_) return;

      // -------- split the leaf (unchanged) --------
      std::size_t axis = d % k;
      auto& pts = cur->points;
      auto& ids = cur->ids;
      const std::size_t n   = pts.size();
      const std::size_t mid = n >> 1;

      std::vector<std::size_t> order(n);
      std::iota(order.begin(), order.end(), 0);
      auto idx_comp = [&](std::size_t i, std::size_t j) {
        return PointTraits<PointT,k>::get(pts[i], axis) <
               PointTraits<PointT,k>::get(pts[j], axis);
      };
      std::nth_element(order.begin(),
                       order.begin() + static_cast<std::ptrdiff_t>(mid),
                       order.end(),
                       idx_comp);

      const float split = PointTraits<PointT,k>::get(pts[order[mid]], axis);

      cur->is_leaf     = false;
      cur->axis        = axis;
      cur->split_value = split;

      cur->left  = std::make_unique<Node>();
      cur->right = std::make_unique<Node>();
      Node* L = cur->left.get();
      Node* R = cur->right.get();

      L->points.reserve(mid);        L->ids.reserve(mid);
      R->points.reserve(n - mid);    R->ids.reserve(n - mid);

      for (std::size_t i = 0; i < mid; ++i) {
        const std::size_t idx = order[i];
        L->points.emplace_back(std::move(pts[idx]));
        L->ids.emplace_back(std::move(ids[idx]));
        expandBox(*L, L->points.back());
      }
      for (std::size_t i = mid; i < n; ++i) {
        const std::size_t idx = order[i];
        R->points.emplace_back(std::move(pts[idx]));
        R->ids.emplace_back(std::move(ids[idx]));
        expandBox(*R, R->points.back());
      }

      L->subtree_size = L->points.size();
      R->subtree_size = R->points.size();

      // free parent leaf storage and recompute parent AABB/size
      std::vector<PointT>().swap(pts);
      std::vector<uint64_t>().swap(ids);
      setBoxFromChildren(*cur, L, R);
      cur->subtree_size = L->subtree_size + R->subtree_size;

      if (d + 1 > max_depth_) max_depth_ = d + 1;
      return;
    }

    // --- Internal node: decide child, then check predictive imbalance
    const float qv = PointTraits<PointT,k>::get(pt, cur->axis);
    const bool goes_left = (qv < cur->split_value);

    if (cur->left && cur->right &&
        isImbalancedAfterInsert(cur, goes_left, high_imbalance_limit))
    {  
      rebuild_root_keep_last();
      return; 
    }

    if (cur->left && cur->right &&
        cur->subtree_size >= min_subtree && cur->subtree_size <= max_subtree &&
        isImbalancedAfterInsert(cur, goes_left, local_imbalance))
    {
      // Collect all points under this subtree, add the new point, rebuild in place
      const std::size_t old_sz = (*cur_slot)->subtree_size;

      uint64_t cutoff_id = 0;
      if (keep_recent_limit > 0 && next_insert_id_ > keep_recent_limit)
        cutoff_id = next_insert_id_ - keep_recent_limit;

      std::vector<PointT>   pts;  pts.reserve(old_sz + 1);
      std::vector<uint64_t> ids;  ids.reserve(old_sz + 1);
      flattenSinceId(*cur_slot, cutoff_id, pts, ids);

      pts.emplace_back(pt);
      ids.emplace_back(next_insert_id_++);

      *cur_slot = (fill_ratio >= 0.999f)
                ? buildBalanced(pts, ids, d)
                : buildBalancedWithSlackParallel(pts, ids, /*depth*/d, fill_ratio,                 
                  point_num_parallel_thr, point_num_task_thr, num_threads);


      const std::size_t new_sz = (*cur_slot)->subtree_size; // set by builders
      const long delta = static_cast<long>(new_sz) - static_cast<long>(old_sz);
      for (Node* a : parents) {
        a->subtree_size = static_cast<std::size_t>(
            static_cast<long>(a->subtree_size) + delta);
      }
      total_points_ = static_cast<std::size_t>( static_cast<long>(total_points_) + delta );
      // AABBs will be tightened later on
      return; // subtree replaced; done
    }

    // descend
    parents.push_back(cur);
    cur_slot = goes_left ? &cur->left : &cur->right;
    if (!*cur_slot) *cur_slot = std::make_unique<Node>();
    cur = cur_slot->get();
    ++d;
  }  
}

} // namespace lkdtree
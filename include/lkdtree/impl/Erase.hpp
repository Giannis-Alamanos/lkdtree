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
 * @file Erase.hpp
 * @brief Template definitions for lkdtree::LKdTree erase operations.
 */

namespace lkdtree {

// -----------------------------------------------------------------------------
// Public erase wrapper
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
std::size_t
LKdTree<PointT,k>::erase(const PointT& pt)
{
  if (!isFinitePoint(pt))
  {
    std::cerr << "[LKdTree] Ignoring erase() with invalid (NaN/Inf) point (erase)" << '\n';
    return 0;
  }

  std::unique_lock lock(tree_mtx_);      // exclusive writer
  constexpr float eps = 1e-6f;

  std::size_t removed = erase(root_, pt, eps);
  if (removed == 0) return 0;

  total_points_ -= removed;

  // Depth may shrink only when entire branches become empty, which we
  // don't prune here, so max_depth_ is still valid.  The next rebuild
  // will tighten it if necessary.
  return removed;
}
  
// -----------------------------------------------------------------------------
// Recursive erasion of exact-matching points
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
std::size_t
LKdTree<PointT,k>::erase(NodePtr& node,
                         const PointT& query,
                         float eps)
{
  if (!node) return 0;

  if (node->is_leaf)
  {
    auto& pts = node->points;
    auto& ids = node->ids;

    // compact in place, keeping pts/ids aligned
    std::size_t write = 0;
    const std::size_t n = pts.size();
    auto matches = [&](const PointT& p){
      for (std::size_t i = 0; i < k; ++i)
        if (std::abs(PointTraits<PointT,k>::get(p,i) -
                     PointTraits<PointT,k>::get(query,i)) > eps)
          return false;
      return true;
    };

    for (std::size_t i = 0; i < n; ++i) {
      if (!matches(pts[i])) {
        if (write != i) {
          pts[write] = std::move(pts[i]);
          ids[write] = ids[i];
        }
        ++write;
      }
    }

    const std::size_t removed = n - write;
    if (!removed) return 0;

    pts.resize(write);
    ids.resize(write);

    // update leaf bookkeeping
    node->subtree_size = write;

    // recompute leaf AABB
    if (write == 0) {
      resetBox(*node);
    } 
    else {
      // tighten from remaining points
      resetBox(*node);
      for (const auto& p : pts) expandBox(*node, p);
    }
    return removed;
  }

  // internal node – descend on the side(s) that could match
  std::size_t axis = node->axis;
  float q          = PointTraits<PointT,k>::get(query, axis);
  std::size_t erased = 0;

  if (q < node->split_value - eps) {
    erased += erase(node->left , query, eps);
  } 
  else if (q > node->split_value + eps) {
    erased += erase(node->right, query, eps);
  } 
  else {
    erased += erase(node->left , query, eps);
    erased += erase(node->right, query, eps);
  }

  if (!erased) return 0;

  // update subtree size and AABB from children
  node->subtree_size =
      (node->left  ? node->left->subtree_size  : 0) +
      (node->right ? node->right->subtree_size : 0);

  if (node->subtree_size == 0) {
    resetBox(*node);
  }
  else {
    setBoxFromChildren(*node, node->left.get(), node->right.get());
  }
  return erased;
}

// -----------------------------------------------------------------------------
// Public eraseSphere wrapper
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
std::size_t
LKdTree<PointT,k>::eraseSphere(const PointT& center, float radius)
{
  if (!isFinitePoint(center) || radius < 0.0f) {
    std::cerr << "[LKdTree] eraseSphere(): invalid center or radius" << '\n';
    return 0;
  }

  std::unique_lock lock(tree_mtx_);          // exclusive writer
  float radius_sq = radius * radius;

  std::size_t removed = eraseSphere(root_, center, radius_sq);
  total_points_ -= removed;
  return removed;
}

// -----------------------------------------------------------------------------
// Recursive erase sphere
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
std::size_t
LKdTree<PointT,k>::eraseSphere(NodePtr&      node,
                               const PointT& center,
                               float         radius_sq)
{
  if (!node) return 0;

  if (node->is_leaf) {
    auto& pts = node->points;
    auto& ids = node->ids;

    const std::size_t n = pts.size();
    std::size_t write = 0;

    for (std::size_t i = 0; i < n; ++i) {
      const bool inside = (distanceSquared(pts[i], center) <= radius_sq);
      if (inside) continue;                       // drop it
      if (write != i) {
        pts[write] = std::move(pts[i]);
        ids[write] = ids[i];
      }
      ++write;
    }

    const std::size_t removed = n - write;
    pts.resize(write);
    ids.resize(write);

    // Recompute leaf AABB and size
    if (write == 0) {
      resetBox(*node);
    } else {
      resetBox(*node);
      for (const auto& p : pts) expandBox(*node, p);
    }
    node->subtree_size = write;
    return removed;
  }

  // Internal node – recurse, then update subtree_size and AABB
  std::size_t removed = 0;

  const std::size_t axis = node->axis;
  const float split      = node->split_value;
  const float c          = PointTraits<PointT,k>::get(center, axis);

  // Near side
  if (c < split)
    removed += eraseSphere(node->left , center, radius_sq);
  else
    removed += eraseSphere(node->right, center, radius_sq);

  // Far side if sphere crosses the split plane
  const float plane_d = std::abs(c - split);
  if (plane_d * plane_d <= radius_sq) {
    if (c < split)
      removed += eraseSphere(node->right, center, radius_sq);
    else
      removed += eraseSphere(node->left , center, radius_sq);
  }

  // Update bookkeeping from children
  node->subtree_size =
      (node->left  ? node->left ->subtree_size : 0) +
      (node->right ? node->right->subtree_size : 0);
  setBoxFromChildren(*node, node->left.get(), node->right.get());

  return removed;
}

// -----------------------------------------------------------------------------
// Public eraseBox wrapper
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
std::size_t
LKdTree<PointT,k>::eraseBox(const PointT& min_pt,
                            const PointT& max_pt)
{
  // basic validation 
  if (!isFinitePoint(min_pt) || !isFinitePoint(max_pt))
  {
    std::cerr << "[LKdTree] eraseBox(): corner contains NaN/Inf – ignored" << '\n';
    return 0;
  }

  for (std::size_t i = 0; i < k; ++i)
  {
    if (PointTraits<PointT,k>::get(min_pt,i) >
        PointTraits<PointT,k>::get(max_pt,i))
    {
      std::cerr << "[LKdTree] eraseBox(): min > max on axis " << i << " – ignored" << '\n';
      return 0;
    }
  }

  // erase 
  std::unique_lock lock(tree_mtx_);      // exclusive writer
  std::size_t removed = eraseBox(root_, min_pt, max_pt);
  total_points_      -= removed;
  return removed;
}

// -----------------------------------------------------------------------------
// Recursive helper for eraseBox
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
std::size_t
LKdTree<PointT,k>::eraseBox(NodePtr&      node,
                            const PointT& min_pt,
                            const PointT& max_pt)
{
  if (!node) return 0;

  if (node->is_leaf)
  {
    auto& pts = node->points;
    auto& ids = node->ids;

    const std::size_t n = pts.size();
    std::size_t write = 0;

    for (std::size_t i = 0; i < n; ++i) {
      const bool inside = inBox(pts[i], min_pt, max_pt);
      if (inside) continue;                      // drop it
      if (write != i) {
        pts[write] = std::move(pts[i]);
        ids[write] = ids[i];
      }
      ++write;
    }

    const std::size_t removed = n - write;
    pts.resize(write);
    ids.resize(write);

    // Recompute leaf AABB and size
    if (write == 0) {
      resetBox(*node);
    } 
    else {
      resetBox(*node);
      for (const auto& p : pts) expandBox(*node, p);
    }
    node->subtree_size = write;
    return removed;
  }

  // Internal node: descend where the box can intersect
  std::size_t removed = 0;

  const std::size_t axis = node->axis;
  const float split      = node->split_value;
  const float min_a      = PointTraits<PointT,k>::get(min_pt, axis);
  const float max_a      = PointTraits<PointT,k>::get(max_pt, axis);

  if (min_a <= split) removed += eraseBox(node->left , min_pt, max_pt);
  if (max_a >= split) removed += eraseBox(node->right, min_pt, max_pt);

  // Update bookkeeping from children
  node->subtree_size =
      (node->left  ? node->left ->subtree_size : 0) +
      (node->right ? node->right->subtree_size : 0);
  setBoxFromChildren(*node, node->left.get(), node->right.get());

  return removed;
}

} // namespace lkdtree
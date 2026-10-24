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
 * @file Helpers.hpp
 * @brief Template definitions for lkdtree::LKdTree helper functions.
 */

namespace lkdtree {

// -----------------------------------------------------------------------------
// Helper: check if subtree would be imbalanced after inserting to the chosen side
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
inline bool LKdTree<PointT, k>::isImbalancedAfterInsert(
    const Node* n,
    bool goes_left,
    float ratio_thresh)
{
  if (!n || n->is_leaf || !n->left || !n->right) return false;
  std::size_t L = n->left ->subtree_size + (goes_left ? 1 : 0);
  std::size_t R = n->right->subtree_size + (goes_left ? 0 : 1);
  const std::size_t lo = std::min(L, R);
  const std::size_t hi = std::max(L, R);
  return (lo == 0) || (static_cast<double>(hi) / static_cast<double>(lo) > ratio_thresh);
}

// -----------------------------------------------------------------------------
// Check if rebuild is needed (based on depth or growth)
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
bool LKdTree<PointT, k>::needsRebuild(float rebuild_ratio) const {
  if (tooSmallToRebuild()) return false;

  const double ideal_depth = std::log2(static_cast<double>(total_points_));
  const bool too_deep      = max_depth_ > ideal_depth * (1.0 + rebuild_ratio);
  const bool grew_enough   = total_points_ > points_at_last_rebuild_ * (1.0 + rebuild_ratio);

  return too_deep || grew_enough;
}

// -----------------------------------------------------------------------------
// Expand bounding box to include point
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
inline void LKdTree<PointT, k>::expandBox(Node& n, const PointT& p) noexcept
{
  #pragma unroll
  for (std::size_t i = 0; i < k; ++i)
  {
    float v = PointTraits<PointT,k>::get(p,i);
    if (v < n.bb_min[i])  n.bb_min[i] = v;   // only update when necessary
    if (v > n.bb_max[i])  n.bb_max[i] = v;
  }
}

// -----------------------------------------------------------------------------
// Reset bounding box to initial state
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
inline void LKdTree<PointT,k>::resetBox(Node& n) noexcept {
  for (std::size_t i = 0; i < k; ++i) {
    n.bb_min[i] =  std::numeric_limits<float>::infinity();
    n.bb_max[i] = -std::numeric_limits<float>::infinity();
  }
}

// -----------------------------------------------------------------------------
// Set bounding box from children
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
inline void LKdTree<PointT,k>::setBoxFromChildren(Node& n, const Node* L, const Node* R) noexcept {
  // start with an empty box
  for (std::size_t i = 0; i < k; ++i) {
    float mn =  std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    if (L) { mn = std::min(mn, L->bb_min[i]); mx = std::max(mx, L->bb_max[i]); }
    if (R) { mn = std::min(mn, R->bb_min[i]); mx = std::max(mx, R->bb_max[i]); }
    n.bb_min[i] = mn;
    n.bb_max[i] = mx;
  }
}

// -----------------------------------------------------------------------------
// Distance helper (squared Euclidean)
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
inline float LKdTree<PointT, k>::distanceSquared(const PointT &a, const PointT &b) const 
{
  float sum = 0.0f;
  for (std::size_t i = 0; i < k; ++i) {
    float diff = PointTraits<PointT, k>::get(a, i) - PointTraits<PointT, k>::get(b, i);
    sum += diff * diff;
  }
  return sum;
}

// -----------------------------------------------------------------------------
// Check whether a point is inside a bounding box
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
inline bool LKdTree<PointT,k>::inBox(const PointT& point,
                                     const PointT& min_pt,
                                     const PointT& max_pt) const
{
  for (std::size_t i = 0; i < k; ++i)
    if (PointTraits<PointT,k>::get(point,i) < PointTraits<PointT,k>::get(min_pt,i) ||
        PointTraits<PointT,k>::get(point,i) > PointTraits<PointT,k>::get(max_pt,i))
      return false;
  return true;
}

// -----------------------------------------------------------------------------
// Check whether the tree is too small to justify rebalancing
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>  
inline bool LKdTree<PointT, k>::tooSmallToRebuild() const {
  if (total_points_ < kMinPointsBeforeRebuild * leaf_max_size_) {
    std::cerr << "[LKdTree] Tree has very few points: " << total_points_ << '\n';
    return true;
  }
  return false;
}

// -----------------------------------------------------------------------------
// Check whether point is finite in all dimensions
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
inline bool LKdTree<PointT, k>::isFinitePoint(const PointT& pt) const {
  for (std::size_t i = 0; i < k; ++i) {
    float coord = PointTraits<PointT, k>::get(pt, i);
    if (!std::isfinite(coord)) return false;
  }
  return true;
} 

// -----------------------------------------------------------------------------
// Check if the user has provided a valid float rebuild ratio
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>  
inline bool LKdTree<PointT, k>::invalidRebuildRatio(float rebuild_ratio) const {
  if (rebuild_ratio < 0.0f) {
    std::cerr << "[LKdTree] Rebuild ratio must be non-negative, got: " << rebuild_ratio << '\n';
    return true;
  }
  return false;
}

// -----------------------------------------------------------------------------
// Check if the user has provided a valid float fill ratio -> (0,1]
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
inline bool LKdTree<PointT,k>::invalidFillRatio(float fill_ratio) const {
  if (!std::isfinite(fill_ratio) || fill_ratio <= 0.0f || fill_ratio > 1.0f) {
    std::cerr << "[LKdTree] fill_ratio must be in (0, 1], got: "
              << fill_ratio << '\n';
    return true;
  }
  return false;
}

}  // namespace lkdtree
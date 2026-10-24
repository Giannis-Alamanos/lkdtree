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
 * @file Accessors.hpp
 * @brief Template definitions for lkdtree::LKdTree access operations.
 */

namespace lkdtree {

// -----------------------------------------------------------------------------
// Getters for tree properties
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>  
inline std::size_t LKdTree<PointT, k>::size() const noexcept        { return total_points_; }

template <typename PointT, std::size_t k>  
inline std::size_t LKdTree<PointT, k>::maxLeafSize() const noexcept { return leaf_max_size_; }

template <typename PointT, std::size_t k>  
inline std::size_t LKdTree<PointT, k>::maxDepth() const noexcept { 
  std::shared_lock lock(tree_mtx_);
  return max_depth_; 
}  

template <typename PointT, std::size_t k>
inline uint64_t LKdTree<PointT, k>::nextInsertId() const noexcept {
  std::shared_lock lock(tree_mtx_);
  return next_insert_id_;
}

template <typename PointT, std::size_t k>
inline bool LKdTree<PointT, k>::isBalancing() const noexcept {
  return balancing_.load(std::memory_order_acquire);   
}

}  // namespace lkdtree
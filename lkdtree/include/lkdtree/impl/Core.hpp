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
 * @file Core.hpp
 * @brief Template definitions for lkdtree::LKdTree core operations.
 */

namespace lkdtree {

// -----------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
LKdTree<PointT, k>::LKdTree(std::size_t leaf_max_size)
    : root_(std::make_unique<Node>()),
      leaf_max_size_(leaf_max_size) {
  if (leaf_max_size <= 0 || leaf_max_size >= 1'000'000)
    throw std::invalid_argument("LKdTree: leaf_max_size must be greater than 0 and less than 1'000'000");

  static_assert(k == 2 || k == 3,
              "LKdTree dimension must be equal to 2 or 3");
}

// -----------------------------------------------------------------------------
// Clear all data from the tree
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
void LKdTree<PointT, k>::clear()
{
  std::unique_lock lock(tree_mtx_);          // exclusive access
  root_ = std::make_unique<Node>();          // fresh empty leaf
  total_points_            = 0;
  max_depth_               = 0;
  points_at_last_rebuild_  = 0;
}

}  // namespace lkdtree
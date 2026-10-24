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
 * @file ExactSearch.hpp
 * @brief Template definitions for lkdtree::LKdTree exact search operations.
 */

namespace lkdtree {

// -----------------------------------------------------------------------------
// Exact-match counter (public wrapper)
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
std::size_t LKdTree<PointT, k>::countExact(const PointT& query) const
{
  if (!isFinitePoint(query)) {
    std::cerr << "[LKdTree] Ignoring query point with invalid (NaN or Inf) coordinates" << '\n';
    return 0;
  }
  std::shared_lock lock(tree_mtx_);      // Many readers - one writer

  constexpr float eps = 1e-6f;
  std::size_t matches = 0;
  countExact(root_, query, eps, matches);
  return matches;
}

// -----------------------------------------------------------------------------
// Recursive exact-match counter
// -----------------------------------------------------------------------------
template <typename PointT, std::size_t k>
void LKdTree<PointT, k>::countExact(const NodePtr& node,
                                    const PointT& query,
                                    float eps,
                                    std::size_t& out) const
{
  if (!node) return;

  if (node->is_leaf)
  {
    for (const PointT& point : node->points)
    {
      bool equal = true;
      for (std::size_t i = 0; i < k && equal; ++i)
        equal &= std::abs(PointTraits<PointT, k>::get(point,i) -
                          PointTraits<PointT, k>::get(query,i)) <= eps;
      if (equal) ++out;
    }
    return;
  }

  std::size_t axis = node->axis;
  float q = PointTraits<PointT, k>::get(query, axis);

  // Go down the obvious side first …
  if (q < node->split_value - eps)
  {
    countExact(node->left , query, eps, out);
  }
  else if (q > node->split_value + eps)
  {
    countExact(node->right, query, eps, out);
  }
  else          // |q − split| <= eps    →  could be on *both* sides
  {
    countExact(node->left , query, eps, out);
    countExact(node->right, query, eps, out);
  }
}

} // namespace lkdtree
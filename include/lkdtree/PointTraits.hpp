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
 * @file PointTraits.hpp
 * @brief Concepts and coordinate access utilities for PointT types used in LKdTree.
 *
 * This internal header defines the `PointTraits<PointT, Dim>` structure used by the
 * `lkdtree::LKdTree` class to access point coordinates in a dimension-agnostic way.
 * It also defines several concepts for compile-time validation of point cloud-like
 * containers, such as `PointCloudLike`, `PointRange`, and `BoxRange`.
 *
 * @note This file is intended for internal use and is included automatically
 *       through `LKdTree.h`. You should not include it directly.
 */

#include <type_traits>
#include <utility>        
#include <cstddef>        
#include <stdexcept>   
#include <type_traits>
#include <ranges>  

namespace lkdtree {

// ────────────────────────────────────────────────────────────────────────────
//  helper: detect whether T has a member named M
// ────────────────────────────────────────────────────────────────────────────

/**
 * @brief Type trait to detect if a type has a public member `x`.
 *
 * Evaluates to `true` if `T::x` is a valid expression; otherwise `false`.
 * Used to validate that a point type supports access to its `x` coordinate.
 *
 * @tparam T Type to inspect.
 */
template <typename, typename = void> struct has_x : std::false_type {};

/**
 * @brief Specialization of `has_x` that evaluates to `true` when `T` has a public member `x`.
 */
template <typename T>
struct has_x<T, std::void_t<decltype(std::declval<T>().x)>> : std::true_type {};


/**
 * @brief Type trait to detect if a type has a public member `y`.
 *
 * Evaluates to `true` if `T::y` is a valid expression; otherwise `false`.
 * Used to validate that a point type supports access to its `y` coordinate.
 *
 * @tparam T Type to inspect.
 */
template <typename, typename = void> struct has_y : std::false_type {};

/**
 * @brief Specialization of `has_y` that evaluates to `true` when `T` has a public member `y`.
 */
template <typename T>
struct has_y<T, std::void_t<decltype(std::declval<T>().y)>> : std::true_type {};


/**
 * @brief Type trait to detect if a type has a public member `z`.
 *
 * Evaluates to `true` if `T::z` is a valid expression; otherwise `false`.
 * Used to validate that a point type supports access to its `z` coordinate.
 *
 * @tparam T Type to inspect.
 */
template <typename, typename = void> struct has_z : std::false_type {};

/**
 * @brief Specialization of `has_z` that evaluates to `true` when `T` has a public member `z`.
 */
template <typename T>
struct has_z<T, std::void_t<decltype(std::declval<T>().z)>> : std::true_type {};

/**
 * @brief Concept that checks whether a type behaves like a point cloud.
 *
 * This concept is satisfied if the type has a public `.points` member that is
 * a range (e.g. vector, list) whose elements are of type `PointT`.
 *
 * @tparam PointCloud The candidate point cloud type.
 * @tparam PointT The point type expected (e.g. pcl::PointXYZI).
 *
 * @note This concept is used to constrain functions like `insertPointCloud()`.
 * It provides a clear compile-time error if the input type does not match expectations.
 */
template <typename PointCloud, typename PointT>
concept PointCloudLike = requires(PointCloud pc) {
  { pc.points } -> std::ranges::range;
  requires std::is_same_v<std::remove_cvref_t<decltype(*pc.points.begin())>, PointT>;
};

/**
 * @brief Concept that represents **any** read‑only range whose value type is
 *        exactly `PointT` (ignoring cv‑qualification and references).
 *
 * A type `R` satisfies `PointRange<R, PointT>` when:
 *   1. `R` models `std::ranges::input_range` — it can be iterated once.
 *   2. `std::remove_cvref_t<std::ranges::range_value_t<R>>` is the same type
 *      as `PointT`.
 *
 * @tparam Range   Candidate range type (e.g. `std::vector<PointT>`,
 *                 `std::span<const PointT>`).
 * @tparam PointT  The expected point element type.
 */
template <typename Range, typename PointT>
concept PointRange =
  std::ranges::input_range<Range> &&
  std::same_as< std::remove_cvref_t<std::ranges::range_value_t<Range>>, PointT >;

/**
 * @brief Concept that checks if a range consists of bounding boxes (min,max).
 *
 * A BoxRange is any iterable range whose elements are `std::pair<PointT, PointT>`,
 * where each `PointT` satisfies the PointConcept and both members of the pair
 * are valid point types (e.g., pcl::PointXYZ, pcl::PointXYZI).
 *
 * This is typically used to pass multiple box queries to functions like
 * `boxSearchBatch()`.
 *
 * @tparam PointT  The type of the 3D point.
 */
template <typename Range, typename PointT>
concept BoxRange = requires(Range r) {
  typename std::ranges::range_value_t<Range>;
  requires std::ranges::input_range<Range>;
  requires std::same_as<
      typename std::ranges::range_value_t<Range>,
      std::pair<PointT, PointT>>;
};


// ────────────────────────────────────────────────────────────────────────────
//  Primary template
// ────────────────────────────────────────────────────────────────────────────
/**
 * @brief Traits class for extracting coordinate values from user-defined point types.
 *
 * @tparam PointT  User-defined point type (e.g., pcl::PointXYZ).
 * @tparam Dim     Spatial dimensionality (2 or 3).
 *
 * Primary template declaration only. This template must be explicitly specialized
 * for supported dimensions (currently 2 and 3). Attempting to use unsupported
 * dimensions results in a compilation error.
 *
 * @see PointTraits<PointT, 2>
 * @see PointTraits<PointT, 3>
 */
template <typename PointT, std::size_t Dim>
struct PointTraits;   // no definition



// ────────────────────────────────────────────────────────────────────────────
//  2-D specialisation
// ────────────────────────────────────────────────────────────────────────────
/**
 * Coordinate accessor for 2-D points.
 *
 * The point type must expose public data members `x` and `y`.
 * If either member is missing, compilation stops with a clear message.
 */
template <typename PointT>
struct PointTraits<PointT, 2>
{
  static_assert(has_x<PointT>::value && has_y<PointT>::value,
                "PointTraits<PointT,2>: PointT must have members x and y");

  static constexpr std::size_t dim = 2;

  static float get(const PointT& pt, std::size_t idx)
  {
    switch (idx) {
      case 0: return pt.x;
      case 1: return pt.y;
      default: throw std::out_of_range("PointTraits<…,2>::get: index > 1");
    }
  }
};


// ────────────────────────────────────────────────────────────────────────────
//  3-D specialisation
// ────────────────────────────────────────────────────────────────────────────
/**
 * Coordinate accessor for 3-D points.
 *
 * The point type must expose public data members `x`, `y`, and `z`.
 * If any member is missing, compilation stops with a clear message.
 */
template <typename PointT>
struct PointTraits<PointT, 3>
{
  static_assert(has_x<PointT>::value && has_y<PointT>::value && has_z<PointT>::value,
                "PointTraits<PointT,3>: PointT must have members x, y and z");

  static constexpr std::size_t dim = 3;

  static float get(const PointT& pt, std::size_t idx)
  {
    switch (idx) {
      case 0: return pt.x;
      case 1: return pt.y;
      case 2: return pt.z;
      default: throw std::out_of_range("PointTraits<…,3>::get: index > 2");
    }
  }
};

} // namespace lkdtree
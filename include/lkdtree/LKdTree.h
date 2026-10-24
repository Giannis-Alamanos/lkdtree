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
 * @file LKdTree.h
 * @brief Public interface for the dynamic k-d tree (LKdTree).
 *
 * This header defines the main template class `lkdtree::LKdTree`, which implements
 * an incremental k-dimensional tree supporting real-time insertion and radius-based
 * nearest neighbor search. It is designed for high-performance use in applications
 * such as robotics, SLAM, and point cloud processing.
 */

#include <memory>
#include <vector>
#include <cstddef>
#include <thread>
#include <algorithm> 
#include <cmath>   
#include <future>  
#include <shared_mutex>
#include <queue>
#include <unordered_set>
#include <array>
#include <iostream>
#include <numeric>

#if defined(_OPENMP)
  #include <omp.h>             
#endif  

#include "lkdtree/PointTraits.hpp"

/**
 * @namespace lkdtree
 * @brief Contains all classes and utilities for the incremental KD-tree.
 *
 * This namespace provides the main `LKdTree` class template and related
 * utilities for spatial queries, insertion, deletion, and batch operations.
 */
namespace lkdtree {

/**
 * @class LKdTree
 * @brief Incremental k-d tree for 2D/3D spatial indexing and search.
 *
 * Features
 * - Incremental insertion with leaf buckets and on-overflow splitting.
 * - Predictive local rebuild and capped global rebuild during insert.
 * - Deletions: exact point, spherical region, and axis-aligned box.
 * - Queries: radius/box, k-NN, and k-NN with uniqueness; batch variants with
 *   optional OpenMP parallelism.
 * - Full/conditional rebuild (sync/async). Rebuilders can perfectly fill
 *   leaves or leave slack (under-filled leaves) to reduce churn.
 *
 * Invariants
 * - Each node keeps an axis-aligned bounding box (AABB) of its subtree and a
 *   `subtree_size` equal to the number of points beneath it.
 * - Leaves store `points` and matching `ids` (kept in lock-step).
 * - Monotonic `ids` reflect insertion order and are used to keep “recent”
 *   points during rebuilds.
 *
 * @tparam PointT Point type with coordinates accessible via PointTraits.
 * @tparam k      Dimensionality. This implementation is restricted to k==2 or k==3.
 */
template <typename PointT, std::size_t k>
class LKdTree {
 public:
  /**
   * @brief Construct an empty LKdTree.
   *
   * @param leaf_max_size Maximum number of points per leaf before it splits.
   */
  explicit LKdTree(std::size_t leaf_max_size = 32);    

  /**
   * @brief Insert a new point into the KD-tree.  
   *
   * This may trigger a local split if the corresponding leaf exceeds its capacity.
   *
   * @param[in] pt Point to be inserted.
   */
  void insert(const PointT& pt);

  /**
   * @brief Insert a point and proactively maintain balance (local or global).
   *
   * Inserts @p pt like `insert()`, but also performs *predictive* rebalancing:
   * before descending at each internal node, it estimates whether inserting
   * to the chosen child would make the two children too imbalanced. Depending
   * on size/ratio thresholds, it either rebuilds the affected subtree in place
   * (local rebuild) or rebuilds the whole tree root (global rebuild).
   *
   * Rebuild strategy:
   * - **Local rebuild** (in-place): When the current subtree size is within
   *   [`min_subtree_for_local_rebuild`, `max_subtree_for_local_rebuild`] *and*
   *   the child-size ratio after this insertion would exceed
   *   `local_rebuild_ratio`. The subtree is flattened (optionally keeping only
   *   recent points; see @p keep_recent_limit), the new point is added, and the
   *   subtree is rebuilt starting at the same depth using either a perfectly
   *   balanced builder or a "slack" builder controlled by @p fill_ratio
   *   (and optionally parallelized; see threading params).
   * - **Global rebuild** (new root): Triggered if the tree size exceeds
   *   @p max_allowed_size or if a very high predicted imbalance exceeds
   *   @p high_imbalance_limit. The root is rebuilt from (up to)
   *   @p full_rebuild_size most recent points plus the new point.
   *
   * @param[in] pt  Point to insert.
   * @param[in] local_rebuild_ratio
   *   Imbalance threshold (>1). For a node with children L and R (after
   *   hypothetically inserting @p pt to the chosen side), if
   *   max(L,R)/min(L,R) > `local_rebuild_ratio` and the subtree size lies within
   *   the local size window, the subtree is locally rebuilt.
   * @param[in] min_subtree_for_local_rebuild
   *   Minimum subtree size to allow a local rebuild (must be > 0).
   * @param[in] max_subtree_for_local_rebuild
   *   Maximum subtree size to allow a local rebuild (must be > min value).
   * @param[in] fill_ratio
   *   Leaf fill target in (0,1]. If >= 0.999, uses the perfectly balanced
   *   median-split builder. Otherwise uses the “slack” builder that allows
   *   under-filled leaves (~`leaf_max_size_ * fill_ratio`) to reduce churn.
   * @param[in] keep_recent_limit
   *   If > 0, a local rebuild keeps only points in the affected subtree whose
   *   insertion IDs are within the most recent `keep_recent_limit`; older points
   *   are discarded from that subtree during the rebuild. (The new point is
   *   always included.)
   * @param[in] max_allowed_size
   *   Hard cap on total points. If `size() > max_allowed_size`, perform a global
   *   rebuild from the most recent @p full_rebuild_size points plus @p pt.
   * @param[in] high_imbalance_limit
   *   Strong imbalance threshold (>1). If the predicted post-insert child-size
   *   ratio at any internal node exceeds this value, perform a **global** rebuild
   *   (even if the local size window is not met).
   * @param[in] full_rebuild_size
   *   Number of most recent points to keep when performing a global rebuild.
   * @param[in] point_num_parallel_thr
   *   Minimum total points (in the current build workload) to create an OpenMP
   *   team for the parallel slack builder. Ignored if OpenMP is not available.
   * @param[in] point_num_task_thr
   *   Minimum subproblem size at which the parallel builder spawns tasks for
   *   left/right recursive builds. Ignored if OpenMP is not available.
   * @param[in] num_threads
   *   Number of threads for the parallel slack builder. If <= 0, uses
   *   `omp_get_max_threads()`. Ignored if OpenMP is not available.
   * @param[in] allow_global_rebuild
   *   If `false`, global rebuilds are disabled: neither exceeding
   *   @p max_allowed_size nor exceeding @p high_imbalance_limit rebuilds the
   *   root, so no points are discarded by the global keep-recent policy and the
   *   tree size is not capped. Local rebuilds are unaffected. The global
   *   parameters are still validated. Defaults to `true`.
   *
   * @note Thread-safety: this method acquires an exclusive lock; concurrent
   *       readers of query APIs are blocked during the short critical section.
   * @note Depth and bounding boxes are maintained along the descent; when a
   *       subtree is rebuilt its `subtree_size` and AABBs are recomputed by the
   *       builder. `max_depth_` is updated consistently (rebuilds start at the
   *       current node’s depth).
   * @warning Passing invalid arguments (e.g. ratios <= 1, non-positive sizes,
   *          fill_ratio out of (0,1]) causes the call to return early without
   *          modifying the tree (a diagnostic is printed to `std::cerr`).
   */
  void insertMaintain(const PointT& pt,
                      float local_rebuild_ratio,
                      std::size_t min_subtree_for_local_rebuild,
                      std::size_t max_subtree_for_local_rebuild,
                      float fill_ratio,
                      std::size_t keep_recent_limit,
                      std::size_t max_allowed_size,
                      float high_imbalance_limit,
                      std::size_t full_rebuild_size,
                      std::size_t point_num_parallel_thr,
                      std::size_t point_num_task_thr,
                      int num_threads = 1,
                      bool allow_global_rebuild = true);

  /**
   * @brief Inserts an entire point cloud into the KD-tree.
   *
   * This method accepts any point cloud-like structure that exposes a `.points`
   * member that is a range of `PointT` elements. All valid (finite) points are
   * inserted using the internal recursive logic. Invalid (NaN or Inf) points are
   * skipped with a warning.
   *
   * @tparam PointCloud A point cloud type satisfying the `PointCloudLike<PointT>` concept.
   * @param[in] point_cloud The input point cloud to insert.
   * @return The number of valid points successfully inserted.
   *
   * @note This function is thread-safe. The KD-tree will lock internally for the entire batch.
   * @see PointCloudLike
   */
  template <PointCloudLike<PointT> PointCloud>  
  std::size_t insertPointCloud(const PointCloud& point_cloud);

  /**
   * @brief Insert an entire point cloud and proactively maintain balance (local or global).
   *
   * Inserts all valid points from @p point_cloud like `insert()`, but also performs
   * *predictive* rebalancing: before descending at each internal node, it estimates
   * whether inserting to the chosen child would make the two children too imbalanced.
   * Depending on size/ratio thresholds, it either rebuilds the affected subtree in place
   * (local rebuild) or rebuilds the whole tree root (global rebuild).
   *
   * Rebuild strategy:
   * - **Local rebuild** (in-place): When the current subtree size is within
   *   [`min_subtree_for_local_rebuild`, `max_subtree_for_local_rebuild`] *and*
   *   the child-size ratio after this insertion would exceed
   *   `local_rebuild_ratio`. The subtree is flattened (optionally keeping only
   *   recent points; see @p keep_recent_limit), the new point is added, and the
   *   subtree is rebuilt starting at the same depth using either a perfectly
   *   balanced builder or a "slack" builder controlled by @p fill_ratio
   *   (and optionally parallelized; see threading params).
   * - **Global rebuild** (new root): Triggered if the tree size exceeds
   *   @p max_allowed_size or if a very high predicted imbalance exceeds
   *   @p high_imbalance_limit. The root is rebuilt from (up to)
   *   @p full_rebuild_size most recent points plus the new point.
   *
   * @param[in] point_cloud  Point cloud to insert.
   * @param[in] local_rebuild_ratio
   *   Imbalance threshold (>1). For a node with children L and R (after
   *   hypothetically inserting a point of @p point_cloud to the chosen side), if
   *   max(L,R)/min(L,R) > `local_rebuild_ratio` and the subtree size lies within
   *   the local size window, the subtree is locally rebuilt.
   * @param[in] min_subtree_for_local_rebuild
   *   Minimum subtree size to allow a local rebuild (must be > 0).
   * @param[in] max_subtree_for_local_rebuild
   *   Maximum subtree size to allow a local rebuild (must be > min value).
   * @param[in] fill_ratio
   *   Leaf fill target in (0,1]. If >= 0.999, uses the perfectly balanced
   *   median-split builder. Otherwise uses the “slack” builder that allows
   *   under-filled leaves (~`leaf_max_size_ * fill_ratio`) to reduce churn.
   * @param[in] keep_recent_limit
   *   If > 0, a local rebuild keeps only points in the affected subtree whose
   *   insertion IDs are within the most recent `keep_recent_limit`; older points
   *   are discarded from that subtree during the rebuild. (The new point is
   *   always included.)
   * @param[in] max_allowed_size
   *   Hard cap on total points. If `size() > max_allowed_size`, perform a global
   *   rebuild from the most recent @p full_rebuild_size points plus @p pt.
   * @param[in] high_imbalance_limit
   *   Strong imbalance threshold (>1). If the predicted post-insert child-size
   *   ratio at any internal node exceeds this value, perform a **global** rebuild
   *   (even if the local size window is not met).
   * @param[in] full_rebuild_size
   *   Number of most recent points to keep when performing a global rebuild.
   * @param[in] point_num_parallel_thr
   *   Minimum total points (in the current build workload) to create an OpenMP
   *   team for the parallel slack builder. Ignored if OpenMP is not available.
   * @param[in] point_num_task_thr
   *   Minimum subproblem size at which the parallel builder spawns tasks for
   *   left/right recursive builds. Ignored if OpenMP is not available.
   * @param[in] num_threads
   *   Number of threads for the parallel slack builder. If <= 0, uses
   *   `omp_get_max_threads()`. Ignored if OpenMP is not available.
   * @param[in] allow_global_rebuild
   *   If `false`, global rebuilds are disabled (see insertMaintain()).
   *   Defaults to `true`.
   *
   * @note Thread-safety: this method acquires an exclusive lock; concurrent
   *       readers of query APIs are blocked during the short critical section.
   * @note Depth and bounding boxes are maintained along the descent; when a
   *       subtree is rebuilt its `subtree_size` and AABBs are recomputed by the
   *       builder. `max_depth_` is updated consistently (rebuilds start at the
   *       current node’s depth).
   * @warning Passing invalid arguments (e.g. ratios <= 1, non-positive sizes,
   *          fill_ratio out of (0,1]) causes the call to return early without
   *          modifying the tree (a diagnostic is printed to `std::cerr`).
   */
  template <PointCloudLike<PointT> PointCloud>
  void insertPointCloudMaintain(const PointCloud& point_cloud,
                      float local_rebuild_ratio,
                      std::size_t min_subtree_for_local_rebuild,
                      std::size_t max_subtree_for_local_rebuild,
                      float fill_ratio,
                      std::size_t keep_recent_limit,
                      std::size_t max_allowed_size,
                      float high_imbalance_limit,
                      std::size_t full_rebuild_size,
                      std::size_t point_num_parallel_thr,
                      std::size_t point_num_task_thr,
                      int num_threads = 1,
                      bool allow_global_rebuild = true);

  /**
   * @brief Build a balanced tree directly from a point cloud.
   *
   * Replaces the current tree with a balanced build from @p point_cloud.
   * If @p fill_ratio < 1, leaves are under-filled to leave headroom
   * for future incremental insertions (same semantics as balanceTree()).
   *
   * @tparam PointCloud  A type satisfying PointCloudLike<PointT> (has .points)
   * @param[in]  point_cloud Input point cloud
   * @param[in]  fill_ratio  Leaf-fill threshold in (0,1]; 1.0 = perfectly filled
   * @return Number of points ingested into the rebuilt tree.
   */
  template <PointCloudLike<PointT> PointCloud>
  std::size_t buildBalancedFromPointCloud(const PointCloud& point_cloud, float fill_ratio = 1.0f);

  /**
   * @brief Erase every point exactly equal to @p pt.
   *
   * If the point contains NaN/Inf nothing happens and 0 is returned.
   *
   * Thread-safe: acquires an exclusive lock (tree is modified).
   *
   * @param[in]  pt   Query point to erase.
   * @return          Number of erased points (may be 0).
   */
  std::size_t erase(const PointT& pt);

  /**
   * @brief Erase every point within a defined sphere.
   *
   * If the center of the sphere contains NaN/Inf or radius is negative nothing happens and 0 is returned.
   *
   * Thread-safe: acquires an exclusive lock (tree is modified).
   *
   * @param[in] center  Center of the sphere.
   * @param[in] radius  Radius of the sphere.
   * @return            Number of erased points (may be 0).
   */
  std::size_t eraseSphere(const PointT& center, float radius);

  /**
   * @brief Erase all points whose coordinates are inside the inclusive
   *        axis-aligned box [min_pt … max_pt].
   * @param[in] min_pt  lower-corner of the box.
   * @param[in] max_pt  upper-corner of the box.
   * @return    number of points removed.
   */
  std::size_t eraseBox(const PointT& min_pt, const PointT& max_pt);

  /**  
   * @brief Perform a radius-based search for neighbors.
   *
   * Finds all points within a given radius of the query point.
   *
   * @param[in] query  The query point.
   * @param[in] radius Search radius (Euclidean-meters).
   * @return Vector of raw pointers to matching points (within radius) in the tree.
   */
  [[nodiscard]] std::vector<const PointT*> 
  radiusSearch(const PointT& query, float radius) const;

  /**
   * @brief   Run radius search for a whole set of queries in parallel.
   * @tparam  QueryRange   A container whose value_type is PointT
   *                       (e.g. std::vector<PointT>, std::span<const PointT>)
   * @param[in] queries      All query points
   * @param[in] radius       Search radius (Euclidean-meters).
   * @param[in] num_threads  Number of threads to use:
   *                     - `0` = sequential
   *                     - `<0` = auto (use all cores)
   *                     - `>0` = exact thread count
   * @return  results[i] holds the matching points (within radius) for queries[i].
   */
  template <PointRange<PointT> QueryRange>
  [[nodiscard]] std::vector<std::vector<const PointT*>>
  radiusSearchBatch(const QueryRange& queries,
                    float             radius,
                    int               num_threads = -1) const;

  /**  
   * @brief Perform a box-based search for neighbors.
   *
   * Finds all points within a given box defined by min_pt and max_pt.
   *
   * @param[in] min_pt The minimum corner of the search box.
   * @param[in] max_pt The maximum corner of the search box.
   * @return Vector of raw pointers to matching points (within box) in the tree.
   */
  [[nodiscard]] std::vector<const PointT*> 
  boxSearch(const PointT& min_pt, 
            const PointT& max_pt) const;

  /**
   * @brief   Run boxSearch for a whole set of box queries in parallel.
   * @tparam  QueryRange   A container whose value_type is std::pair<PointT, PointT>
   *                       (e.g. std::vector<std::pair<PointT, PointT>>, std::span<const std::pair<PointT, PointT>>)
   * @param[in] queries      All query boxes
   * @param[in] num_threads  Number of threads to use:
   *                     - `0` = sequential
   *                     - `<0` = auto (use all cores)
   *                     - `>0` = exact thread count
   * @return  results[i] holds the matching points for boxes[i].
   */
  template <BoxRange<PointT> QueryRange>
  [[nodiscard]] std::vector<std::vector<const PointT*>>
  boxSearchBatch(const QueryRange& boxes,
                 int               num_threads = -1) const;

  /**
   * @brief k-nearest-neighbour query.
   *
   * Finds the k closest points to the given query using Euclidean distance.
   *
   * @param[in] query    The query point.
   * @param[in] k        Number of nearest neighbours to return (k ≥ 1).
   * @param[in] max_dist  Maximum allowed distance of a neighbour.
   *                      • **< 0**  ⇒ no distance limit (default)  
   *                      • **≥ 0** ⇒ ignore points with distance > max_dist
   * @return Vector of raw pointers to the k closest points, sorted
   *         from nearest to farthest. Returns an empty vector if:
   *         - k == 0
   *         - k > total number of points
   *         - the query contains NaN or Inf coordinates
   *         - the tree is empty
   */
  [[nodiscard]] std::vector<const PointT*>
  knnSearch(const PointT& query,
            std::size_t   k_neigh,
            float         max_dist = -1.0f) const;

  /**
   * @brief   Run k-NN for a whole set of queries in parallel.
   * @tparam  QueryRange   A container whose value_type is PointT
   *                       (e.g. std::vector<PointT>, std::span<const PointT>)
   * @param[in] queries      All query points
   * @param[in] k_neigh      How many neighbours per query
   * @param[in] max_dist  Maximum allowed distance of a neighbour.
   *                      • **< 0**  ⇒ no distance limit (default)  
   *                      • **≥ 0** ⇒ ignore points with distance > max_dist
   * @param[in] num_threads  Number of threads to use:
   *                     - `0` = sequential
   *                     - `<0` = auto (use all cores)
   *                     - `>0` = exact thread count
   * @return  results[i] holds the neighbours for queries[i] in ascending
   *          distance order.
   */
  template <PointRange<PointT> QueryRange>
  [[nodiscard]] std::vector<std::vector<const PointT*>>
  knnSearchBatch(const QueryRange& queries,
                std::size_t       k_neigh,
                float             max_dist = -1.0f,
                int               num_threads = -1) const;

  /**
   * @brief k-nearest-neighbour query without duplicates.
   *
   * Similar to knnSearch() but guarantees that every neighbour has unique
   * coordinates ( |Δx|,|Δy|,|Δz| ≤ @p eps are considered identical ).  
   * If the cloud contains fewer than @p k geometrically-distinct points,
   * the returned vector will be smaller.
   *
   * @param[in] query    Query point.
   * @param[in] k_neigh  Maximum number of unique neighbours to return (k ≥ 1).
   * @param[in] max_dist Maximum allowed distance of a neighbour.
   * @param[in] eps      Tolerance used to decide whether two points are the “same”.
   *                     Default: 1 e-6 ≈ single-precision micrometre.
   * @return Raw-pointer vector sorted from nearest to farthest, or an empty
   *         vector when the arguments are invalid (k==0, k>size(), or
   *         query contains NaN/Inf).
   */
  [[nodiscard]]
  std::vector<const PointT*>
  knnSearchUnique(const PointT& query,
                  std::size_t   k_neigh,
                  float         max_dist = -1.0f,
                  float         eps = 1e-6f) const;

  /**
   * @brief   Run k-NN Unique for a whole set of queries in parallel.
   * @tparam  QueryRange   A container whose value_type is PointT
   *                       (e.g. std::vector<PointT>, std::span<const PointT>)
   * @param[in] queries      All query points
   * @param[in] k_neigh      How many neighbours per query
   * @param[in] max_dist     Maximum allowed distance of a neighbour.
   * @param[in] eps          Tolerance used to decide whether two points are the “same”.
   *                         Default: 1 e-6 ≈ single-precision micrometre.
   * @param[in] num_threads  Number of threads to use:
   *                         - `0` = sequential
   *                         - `<0` = auto (use all cores)
   *                         - `>0` = exact thread count
   * @return  results[i] holds the neighbours for queries[i] in ascending
   *          distance order.
   */
  template <PointRange<PointT> QueryRange>
  [[nodiscard]] std::vector<std::vector<const PointT*>>
  knnSearchUniqueBatch(const QueryRange& queries,
                       std::size_t       k_neigh,
                       float             max_dist = -1.0f,
                       float             eps = 1e-6f,
                       int               num_threads = -1) const;

  /**
   * @brief Count how many stored points are *identical* to @p query.
   *
   * The comparison is per-coordinate and inclusive:  
   * a coordinate is considered equal when  
   * `std::abs(a_i − b_i) <= eps`.
   *
   * @param[in] query  Point whose duplicates we want to count.
   * @return The number of occurrences of @p query found in the tree.
   */
  [[nodiscard]] std::size_t 
  countExact(const PointT& query) const;

  /**
   * @brief Synchronously rebuild and balance the tree in the current thread.
   *
   * This performs a full rebuild of the internal KD-tree structure and replaces the current tree.
   * 
   * @param[in] fill_ratio Leaf-fill threshold in (0, 1].  Values < 1 create
   *                       under-filled leaves; 1.0 degenerates to a perfectly
   *                       balanced build.
   * @return true if the tree was rebuilt; false otherwise.
   */
  bool balanceTree(float fill_ratio = 1.0f);

  /**
   * @brief Launch a background thread to rebuild and balance the tree.
   *
   * Returns immediately. The actual rebuild is performed in a separate thread.
   *
   * @note While a rebuild is in progress, other operations (insertion/search) may be blocked.
   *       Future: could add locking or shared access to allow concurrent reads.
   * 
   * @param[in] fill_ratio Leaf-fill threshold in (0, 1].  Values < 1 create
   *                       under-filled leaves; 1.0 degenerates to a perfectly
   *                       balanced build.   
   * @return std::future<bool> Future resolving to true if the tree was rebuilt.
   */
  std::future<bool> balanceTreeAsync(float fill_ratio = 1.0f);

  /**
   * @brief Conditionally rebalance the tree if it has grown or become unbalanced.
   *
   * Rebuilds the tree only if the number of points or tree depth has significantly increased
   * since the last rebuild.
   *
   * @param[in] rebuild_ratio Fractional threshold to trigger rebuild (e.g., 0.5 = 50% growth or depth increase).
   * @param[in] fill_ratio    Leaf-fill threshold in (0, 1].  Values < 1 create
   *                          under-filled leaves; 1.0 degenerates to a perfectly
   *                          balanced build. 
   * @return True if the tree was rebuilt, false otherwise.
   */
  bool tryBalanceTree(float rebuild_ratio = 0.5f, float fill_ratio = 1.0f);

  /**
   * @brief Launch an asynchronous task to conditionally rebalance the tree if it is unbalanced.
   *
   * If the tree meets the imbalance criteria (based on rebuild_ratio), it spawns a background
   * thread to rebuild the tree using `tryBalanceTree()`.
   *
   * @param[in] rebuild_ratio Fractional threshold to trigger rebuild (e.g., 0.5 = 50% growth or depth increase).
   * @param[in] fill_ratio    Leaf-fill threshold in (0, 1].  Values < 1 create
   *                          under-filled leaves; 1.0 degenerates to a perfectly
   *                          balanced build. 
   * @return std::future<bool> Future resolving to true if the tree was rebuilt.
   */
  std::future<bool> tryBalanceTreeAsync(float rebuild_ratio = 0.5f, float fill_ratio = 1.0f);

  /**
   * @brief Remove every point and reset the KD-tree to its initial state.
   *
   * Removes all nodes and points, resets size and depth to zero,
   * but preserves configuration like leaf_max_size.
   *
   */
  void clear();

  /**
   * @brief Get the number of points currently stored in the tree.
   * 
   * @return std::size_t Total number of inserted points.
   */
  [[nodiscard]] std::size_t size() const noexcept;
   
  /**
   * @brief Get the maximum number of points allowed in a leaf before it splits.
   * 
   * This is the threshold used during insertion to determine when a leaf node
   * should be subdivided.
   * 
   * @return std::size_t The maximum size of a leaf node.
   */
  [[nodiscard]] std::size_t maxLeafSize() const noexcept;

  /**
   * @brief Get the maximum depth observed in the current KD-tree structure.
   * 
   * This value represents the deepest level reached in the current tree,
   * which can be used as a proxy for tree balance.
   * 
   * @return std::size_t The maximum depth of the KD-tree.
   */
  [[nodiscard]] std::size_t maxDepth() const noexcept;

  /**
   * @brief Get the insertion ID that the next inserted point will receive.
   *
   * IDs increase monotonically and are never reused, so the last inserted point
   * has ID `nextInsertId() - 1`.
   *
   * @return uint64_t The ID of the next point to be inserted.
   */
  [[nodiscard]] uint64_t nextInsertId() const noexcept;

  /**
   * @brief Check if the tree is currently undergoing a rebalancing operation.
   *
   * This method returns true if the tree is being rebalanced asynchronously and false otherwise.
   *
   * @return true if the tree is currently rebalancing, false otherwise.
   */
  [[nodiscard]] bool isBalancing() const noexcept;

 private:
  static constexpr std::size_t kMinPointsBeforeRebuild = 2;  ///< Minimum multiple of leaf_max_size_ before allowing rebuild.
  mutable std::shared_mutex tree_mtx_;                       ///< Readers–writers lock
  std::atomic_bool balancing_{false};                        ///< True while a rebuild is running

  /**
   * @struct Node
   * @brief Internal node representation (leaf or split).
   */
  struct Node;
  using NodePtr = std::unique_ptr<Node>;

  /**
   * @brief Internal representation of a tree node in the LKdTree.
   *
   * A node is either:
   * - A **leaf node** (`is_leaf == true`) containing a list of points.
   * - Or a **split node** (`is_leaf == false`) with left/right children and a splitting rule.
   *
   * The split rule consists of:
   * - `axis`: the coordinate axis (0 = x, 1 = y, 2 = z, etc.)
   * - `split_value`: the coordinate value where the space is split
   */
  struct Node {
    bool is_leaf = true;                   ///< Whether this node is a leaf.
    std::vector<PointT> points;            ///< Points stored in the leaf (valid only if is_leaf is true).
    std::vector<uint64_t> ids;             ///< Monotonically increasing IDs for each point (valid only if is_leaf is true).

    // Split info (valid only if is_leaf == false)
    std::size_t axis {0};                  ///< Splitting axis (0 = x, 1 = y, 2 = z, etc.).
    float split_value {0.0f};              ///< Coordinate value along axis where the split occurs.
    NodePtr left {nullptr};                ///< Left child (points < split_value).
    NodePtr right {nullptr};               ///< Right child (points ≥ split_value).
    std::array<float,k> bb_min;            ///< per‑axis minimum
    std::array<float,k> bb_max;            ///< per‑axis maximum
    std::size_t subtree_size {0};          ///< Total number of points in this subtree.

    Node()
    {
        bb_min.fill(std::numeric_limits<float>::infinity());
        bb_max.fill(-std::numeric_limits<float>::infinity());
    }
  };

  NodePtr root_;                           ///< Root node of the KD-tree.
  std::size_t leaf_max_size_;              ///< Max number of points allowed in a leaf before splitting.
  std::size_t total_points_ {0};           ///< Total number of inserted points.
  std::size_t max_depth_ {0};              ///< Deepest level seen so far.
  std::size_t points_at_last_rebuild_ {0}; ///< Points count at last rebuild.
  uint64_t next_insert_id_{0};             ///< Monotonically increasing ID for each inserted point.

  /**
   * @brief Custom hash function for fixed-size coordinate arrays.
   *
   * Applies a simple polynomial rolling hash using a small prime multiplier.
   */
  struct CoordHash {
    std::size_t operator()(const std::array<int64_t,k>& a) const noexcept {
      std::size_t h = 0;
      for (auto v : a) h = h*17u + std::hash<int64_t>{}(v);
      return h;
    }
  };
  
  /**
   * @brief Candidate subtree for rebalancing.
   *
   * Used internally to track subtrees that may need rebalancing
   * based on their size and imbalance score.
   */
  struct Candidate {
    NodePtr*      slot;   ///< Pointer to the unique_ptr slot for replacement
    std::size_t   size;   ///< Subtree size at selection time
    double        score;  ///< Imbalance score (higher = worse)
  };

  /**
   * @brief Recursive insert routine.
   *
   * @param[in] node  Current node reference.
   * @param[in] pt    Point to insert.
   * @param[in] depth Current depth (used to select split axis).
   */
  void insert(NodePtr& node, const PointT& pt, std::size_t depth);

  /**
   * @brief Insert into a subtree and proactively maintain balance.
   *
   * Inserts @p pt into the subtree rooted at @p slot (at tree @p depth), updating
   * AABBs and subtree sizes on the path. If the target leaf overflows, it splits.
   * Before descending at each internal node, it predicts the post-insert child
   * ratio; if the ratio exceeds @p local_imbalance and the subtree size is within
   * [ @p min_subtree, @p max_subtree], it flattens that subtree (optionally
   * keeping only points with ID >= cutoff from @p keep_recent_limit), appends the
   * new point, and rebuilds the subtree in place using either the perfectly
   * balanced builder (when @p fill_ratio >= 0.999) or the slack builder
   * (optionally parallelized via @p point_num_parallel_thr / @p point_num_task_thr
   * and @p num_threads).
   *
   * A **global rebuild** is triggered instead when total size exceeds
   * @p max_allowed_size or when the predicted imbalance exceeds
   * @p high_imbalance_limit; in that case the root is rebuilt from the most
   * recent @p full_rebuild_size points plus @p pt.
   *
   * @param[in,out] slot  Reference to the unique_ptr of the current subtree; may be replaced.
   * @param[in]     pt    Point to insert.
   * @param[in]     depth Depth of @p slot (passed to builders unchanged).
   * @param[in]     local_imbalance   Local rebuild ratio threshold (>1).
   * @param[in]     min_subtree       Min subtree size to allow local rebuild.
   * @param[in]     max_subtree       Max subtree size to allow local rebuild.
   * @param[in]     fill_ratio        Leaf fill target in (0,1]; >=0.999 → perfect builder.
   * @param[in]     keep_recent_limit Keep only this many most-recent IDs on local rebuild (if >0).
   * @param[in]     max_allowed_size  Hard cap; exceed → global rebuild.
   * @param[in]     high_imbalance_limit Strong ratio (>1) that forces global rebuild.
   * @param[in]     full_rebuild_size Number of most-recent points kept on global rebuild.
   * @param[in]     point_num_parallel_thr Min workload to create an OpenMP team (slack builder).
   * @param[in]     point_num_task_thr     Min subproblem size to spawn tasks (slack builder).
   * @param[in]     num_threads       Thread count for parallel builder (ignored if no OpenMP).
   * @param[in]     allow_global_rebuild If false, neither global trigger rebuilds the root.
   */
  void insertMaintain(NodePtr& slot, const PointT& pt,
                      std::size_t depth, float local_imbalance,
                      std::size_t min_subtree, std::size_t max_subtree,
                      float fill_ratio, std::size_t keep_recent_limit,
                      std::size_t max_allowed_size, float high_imbalance_limit,
                      std::size_t full_rebuild_size, std::size_t point_num_parallel_thr,
                      std::size_t point_num_task_thr, int num_threads,
                      bool allow_global_rebuild);

  /**
   * @brief Recursive helper for erase().
   *
   * Traverses the subtree rooted at @p node, removes every point that matches
   * @p query within the tolerance @p eps, and returns the number of points
   * actually erased.
   *
   * @param[in,out] node  Current subtree root (its leaf vectors may be shrunk).
   * @param[in]     query Point to match and delete.
   * @param[in]     eps   Tolerance used for coordinate comparison.
   * @return              Count of points removed in this subtree.
   */
  std::size_t erase(NodePtr& node, const PointT& query, float eps);

  /**
   * @brief Recursively removes all points within a sphere.
   *
   * Traverses the subtree rooted at \p node and deletes every point lying within
   * a spherical region defined by \p center and squared radius \p radius_sq.
   *
   * @param[in,out] node       Current subtree (may be modified).
   * @param[in]     center     Center of the spherical erase region.
   * @param[in]     radius_sq  Radius squared (avoid square roots).
   * @return        Number of points removed from this subtree.
   */
  std::size_t eraseSphere(NodePtr& node, const PointT& center, float radius_sq);  

  /**
   * @brief Internal DFS that prunes sub-trees lying entirely outside the box.
   *
   * @param[in,out] node     current subtree (may be modified in-place).
   * @param[in]     min_pt   lower-corner of the axis-aligned box.
   * @param[in]     max_pt   upper-corner of the axis-aligned box.
   * @return        number of points removed from this subtree.
   *
   * Called only while the caller already holds the writer lock.
   */
  std::size_t eraseBox(NodePtr& node, const PointT& min_pt, const PointT& max_pt);

  /**
   * @brief Perform a recursive radius-based search (inclusive).
   *
   * Traverses the tree to find all points within or exactly on the
   * specified radius from the query point.
   *
   * @param[in] node    Current node in the traversal.
   * @param[in] query   The query point.  
   * @param[in] radius  The search radius (inclusive).
   * @param[out] result Output vector accumulating pointers to matching points.
   * @param[in] depth   Current depth in the KD-tree.
   */
  void radiusSearch(const NodePtr& node, const PointT& query, float radius,
                    std::vector<const PointT*>& result, std::size_t depth) const;

  /**
   * @brief Perform a recursive box-based search (inclusive).
   *
   * Traverses the tree to find all points within or exactly on the
   * specified box.
   *
   * @param[in]  node   Current node in the traversal.
   * @param[in]  min_pt Lower-left (or lower-front-left) corner of the query box.
   * @param[in]  max_pt Upper-right (or upper-back-right) corner of the query box.
   * @param[out] result Output vector accumulating pointers to matching points.
   */
  void boxSearch(const NodePtr&               node,
                 const PointT&                min_pt,
                 const PointT&                max_pt,
                 std::vector<const PointT*>&  result) const;

  /**
   * @brief Recursive helper for k-nearest-neighbour search.
   *
   * Traverses the KD-tree to maintain a max-heap of the current k
   * closest points. The search visits branches only if they could
   * contain a closer point than the current farthest in the heap.
   *
   * @param[in] node     Current node in the tree.
   * @param[in] query    Query point.
   * @param[in] k_neigh  Number of neighbors to keep.
   * @param[in] max_dist Maximum allowed distance of a neighbour.
   * @param[in,out] heap A max-heap of (distance², point) pairs storing the k closest matches so far.
   */
  void knnSearch(const NodePtr&          node,
                 const PointT&           query,
                 std::size_t             k_neigh,
                 float                   max_dist,
                 std::priority_queue<
                     std::pair<float, const PointT*>,
                     std::vector<std::pair<float,const PointT*>>,
                     std::less<>>&       heap) const;

  /**
   * @brief Recursive helper for k-NN search with uniqueness constraint.
   *
   * Traverses the KD-tree to find up to `k_neigh` nearest neighbours to the
   * given `query` point, while enforcing a uniqueness constraint: only one
   * point is retained from any group of neighbours that fall within a small
   * distance (`eps`) of each other, as determined by quantized coordinates.
   *
   * Uniqueness is checked via the `seen` hash set, which stores quantized
   * coordinate keys to reject spatial duplicates.
   *
   * @param[in]     node     Current tree node being visited.
   * @param[in]     query    The query point.
   * @param[in]     k_neigh  Maximum number of unique neighbours to return.
   * @param[in]     max_dist Maximum allowed distance of a neighbour.
   * @param[in]     eps      Squared distance threshold for uniqueness filtering.
   * @param[in,out] heap     Max-heap of accepted nearest neighbours.
   * @param[in,out] seen     Set of quantized coordinate keys used to enforce uniqueness.
   */
  void knnSearchUnique(const NodePtr& node,
                      const PointT&  query,
                      std::size_t    k_neigh,
                      float          max_dist,
                      float          eps,
                      std::priority_queue<
                          std::pair<float, const PointT*>,
                          std::vector<std::pair<float, const PointT*>>,
                          std::less<>>& heap,
                          std::unordered_set<std::array<int64_t,k>,
                           CoordHash>& seen) const;

  /**
   * @brief Recursive exact-match search inside the tree.
   *
   * Traverses the KD-tree using coordinate comparisons within @p eps
   * to locate all leaf buckets that may contain the query. For points
   * within the leaf, performs full coordinate comparison.
   *
   * @param[in] node   The current node being visited.
   * @param[in] query  The query point to match.
   * @param[in] eps    Tolerance for floating-point comparisons.
   * @param[out] count Accumulator for number of matches found.
   */
  void countExact(const NodePtr& node, const PointT& query,
                  float eps, std::size_t& out) const;

  /**
   * @brief Attempt to rebuild the tree if it is unbalanced or has grown significantly.
   *
   * Performs a full rebuild of the tree structure using a balanced median-split strategy,
   * but only if certain growth or imbalance conditions are met (see `needsRebuild()`).
   *
   * @param[in] rebuild_ratio The fractional threshold for triggering a rebuild.
   * @param[in] fill_ratio    Leaf-fill threshold in (0, 1].  Values < 1 create
   *                          under-filled leaves; 1.0 degenerates to a perfectly
   *                          balanced build.
   * @return true if the tree was rebuilt; false if no rebuild was necessary.
   */
  bool tryRebuild(float rebuild_ratio, float fill_ratio);

  /**
   * @brief Predict whether inserting on a side would make an internal node imbalanced.
   *
   * For an internal node @p n with two children, this computes the *hypothetical*
   * child sizes after routing the incoming point to the indicated side
   * ( @p goes_left). It then checks the imbalance ratio
   * max(L,R) / min(L,R) against @p ratio_thresh.
   *
   * @param[in] n            Internal node to test (must have both children).
   * @param[in] goes_left    @c true if the new point would be routed to the left child.
   * @param[in] ratio_thresh Allowed imbalance ratio (> 1.0).
   * @return @c true if the predicted insert would violate @p ratio_thresh (or degenerate),
   *         otherwise @c false.
   */
  bool isImbalancedAfterInsert(const Node* n, bool goes_left, float ratio_thresh);
    
  /**
   * @brief Determine whether the tree structure needs to be rebuilt.
   *
   * A rebuild is recommended if the tree is significantly deeper than log₂(n) or
   * if the number of points has increased substantially since the last rebuild.
   *
   * @param[in] rebuild_ratio The fractional threshold for evaluating imbalance or growth.
   * @return true if the tree should be rebuilt; false otherwise.
   */
  bool needsRebuild(float rebuild_ratio) const;

  /**
   * @brief Flatten the tree into a linear vector of points.
   *
   * Traverses the entire KD-tree and collects all stored points into the given vector.
   * This is used during tree rebuilding to extract all data for balanced reconstruction.
   *
   * @param[in]  node     Root of the subtree to flatten.
   * @param[out] out_pts  Vector to receive all points from the tree.
   * @param[out] out_ids  Vector to receive all point IDs from the tree.
   */
  void flatten(const NodePtr &node, std::vector<PointT> &out_pts,
               std::vector<uint64_t>& out_ids) const;

  /**
   * @brief Append all points in a subtree whose IDs are >= a cutoff.
   *
   * Performs a depth-first (left, then right) traversal of @p node and appends
   * to @p out_pts / @p out_ids every (point, id) pair stored in leaves for which
   * id >= @p cutoff_id. The function does **not** clear the output vectors;
   * it only appends. If @p node is null, it returns immediately.
   *
   * This is used to keep only the most recent points when locally or globally
   * rebuilding the tree (IDs are assigned monotonically at insertion time).
   *
   * @param[in]  node       Root of the subtree to flatten.
   * @param[in]  cutoff_id  Inclusive threshold; keep elements with id >= cutoff_id.
   * @param[out] out_pts    Destination vector for points (appended).
   * @param[out] out_ids    Destination vector for corresponding IDs (appended).
   *
   */
  void flattenSinceId(const NodePtr& node,
                      uint64_t       cutoff_id,
                      std::vector<PointT>&   out_pts,
                      std::vector<uint64_t>& out_ids) const;

  /**
   * @brief Build a perfectly balanced KD-tree from a point set.
   *
   * Recursively partitions the input vector using median splits to create a balanced
   * tree structure. Used internally for rebalancing.
   *
   * @param[in,out] pts   Point set to build the tree from (will be consumed).
   * @param[in,out] ids   Corresponding point IDs (will be consumed).
   * @param[in]     depth Current recursion depth (determines split axis).
   * @return Root node of the newly constructed balanced subtree.
   */
  NodePtr buildBalanced(std::vector<PointT> &pts, std::vector<uint64_t>& ids, std::size_t depth);

  /**
   * @brief Build a “balanced-enough” KD-tree that leaves headroom in its buckets.
   *
   * Similar to buildBalanced(), but allows each leaf to stop splitting once its
   * size drops below `leaf_max_size_ × fill_ratio`, leaving spare capacity so
   * that future incremental insertions are less likely to trigger immediate
   * re-splits.  The resulting tree is still logarithmic in height, yet its leaves
   * may be only partially filled (controlled by @p fill_ratio).
   *
   * @param[in,out] pts        Vector of points to consume and organise. The
   *                           contents are moved into leaf nodes; the vector
   *                           itself is left empty on return.
   * @param[in,out] ids        Corresponding point IDs.
   * @param[in]     depth      Current recursion depth, decides the split axis.
   * @param[in]     fill_ratio Leaf-fill threshold in (0, 1].  Values < 1 create
   *                           under-filled leaves; 1.0 degenerates to a perfectly
   *                           balanced build.
   * @return Root node of the newly built (slack-balanced) subtree.
   *
   * @complexity O(n log n) due to the repeated median partitioning.
   */
  NodePtr buildBalancedWithSlack(std::vector<PointT>& pts, std::vector<uint64_t>& ids,
                                 std::size_t depth, float fill_ratio);

  /**
   * @brief Build a “balanced-enough” KD-tree that leaves headroom in its buckets
   *        using parallelism (OpenMP). 
   *
   * Similar to buildBalanced(), but allows each leaf to stop splitting once its
   * size drops below `leaf_max_size_ × fill_ratio`, leaving spare capacity so
   * that future incremental insertions are less likely to trigger immediate
   * re-splits.  The resulting tree is still logarithmic in height, yet its leaves
   * may be only partially filled (controlled by @p fill_ratio).
   *
   * ### Parallel behavior (OpenMP):
   * - If compiled with OpenMP, a parallel team is created only when
   *   `pts.size() >= point_num_parallel_thr` **and** `num_threads > 1`.
   * - Inside the builder, OpenMP **tasks** are spawned for subproblems with
   *   `n >= point_num_task_thr`/.
   * - If OpenMP is not available, the build runs sequentially and the three
   *   parallel parameters are ignored.
   *
   * @param[in,out] pts                     Vector of points to consume and organise. The
   *                                        contents are moved into leaf nodes; the vector
   *                                        itself is left empty on return.
   * @param[in,out] ids                     Corresponding point IDs.
   * @param[in]     depth                   Current recursion depth, decides the split axis.
   * @param[in]     fill_ratio              Leaf-fill threshold in (0, 1].  Values < 1 create
   *                                        under-filled leaves; 1.0 degenerates to a perfectly
   *                                        balanced build.
   * @param[in]     point_num_parallel_thr  Minimum total point count to create an OpenMP team.
   * @param[in]     point_num_task_thr      Minimum slice size to spawn OpenMP tasks inside the build.
   * @param[in]     num_threads             Desired number of OpenMP threads (<=0 → use `omp_get_max_threads()`).
   *
   * @return Root node of the newly built (slack-balanced) subtree.
   */
  NodePtr buildBalancedWithSlackParallel(std::vector<PointT>& pts,
                                      std::vector<uint64_t>& ids,
                                      std::size_t depth,
                                      float fill_ratio,
                                      std::size_t point_num_parallel_thr,
                                      std::size_t point_num_task_thr,
                                      int num_threads);

  /**
   * @brief Perform the actual rebuild of the KD-tree.
   * 
   * This function assumes exclusive access to the tree (the caller must lock).
   * It flattens the tree, rebuilds it in balanced form, and updates internal metadata.
   * 
   * @param[in] fill_ratio Leaf-fill threshold in (0, 1].  Values < 1 create
   *                       under-filled leaves; 1.0 degenerates to a perfectly
   *                       balanced build.
   */
  void performRebuild(float fill_ratio);

  /**
   * @brief Enlarge a node’s bounding box so it contains a new point.
   *
   * Every coordinate of @p p is compared to the current minimum/maximum stored
   * in @p n.bb_min / @p n.bb_max.  Values that lie outside the existing range
   * overwrite the corresponding bound, guaranteeing that the node’s axis-aligned
   * bounding box (AABB) always encloses all of its points.
   *
   * @param[in,out] n Node whose AABB will be updated.
   * @param[in]     p Point to assimilate into the bounding box.
   *
   * @complexity O(k) where *k* is the dimensionality (2 or 3).
   */
  void expandBox(Node& n, const PointT& p) noexcept;

  /**
   * @brief Reset a node’s bounding box to an empty state.
   *
   * Sets every component of @p n.bb_min to +∞ and every component of
   * @p n.bb_max to −∞.  A subsequent call to expandBox() with real data will
   * then establish valid limits.
   *
   * @param[in,out] n Node whose AABB is being cleared.
   *
   * @complexity O(k).
   */
  void resetBox(Node& n) noexcept;

  /**
   * @brief Recompute a parent node’s AABB from its children.
   *
   * Writes each component of @p n.bb_min / @p n.bb_max as the minimum / maximum
   * of the corresponding components in the child nodes’ boxes.  Either child
   * pointer may be `nullptr`, in which case the parent adopts the bounds of the
   * non-null child.
   *
   * @param[in,out] n Parent node whose AABB will be refreshed.
   * @param[in]     L Pointer to left  child (may be `nullptr`).
   * @param[in]     R Pointer to right child (may be `nullptr`).
   *
   * @pre At least one of @p L or @p R must be non-null.
   * @complexity O(k).
   */
  void setBoxFromChildren(Node& n, const Node* L, const Node* R) noexcept;
  
  /**
   * @brief Compute squared Euclidean distance between two points.
   *
   * @param[in] a First point.
   * @param[in] b Second point.
   * @return Squared distance between a and b.
   */
  float distanceSquared(const PointT& a, const PointT& b) const;

  /**
   * @brief Check if a point is inside a bounding box defined by two corners.
   *
   * @param[in] point   Point to check.
   * @param[in] min_pt  Minimum point of the bounding box.
   * @param[in] max_pt  Maximum point of the bounding box.
   * @return True if the point is inside the box, false otherwise.
   */
  bool inBox(const PointT& point,
             const PointT& min_pt,
             const PointT& max_pt) const;

  /**
   * @brief Check if the tree has too few points to justify a rebuild.
   *
   * Determines whether the total number of inserted points is below a threshold
   * that would make rebuilding the tree inefficient or unnecessary.
   *
   * This check is useful for skipping unnecessary rebuilds when the tree is
   * still very small and inherently fast to search even without rebalancing.
   *
   * @return true if the tree is too small to rebuild; false otherwise.
   */
  bool tooSmallToRebuild() const;

  /**
   * @brief Check if all point coordinates are finite (no NaN or Inf).
   *
   * @param[in] pt The point to validate.
   * @return true if all coordinates are finite; false otherwise.
   */
  bool isFinitePoint(const PointT& pt) const;

  /**     
   * @brief Check whether the provided rebuild ratio is invalid.
   *
   * A valid rebuild ratio must be non-negative. Ratios >= 0.0f are allowed,
   * where a value of 0.0 forces unconditional rebuild.
   *
   * @param rebuild_ratio The rebuild threshold ratio.
   * @return true if the value is invalid (i.e., negative), false otherwise.
   */
  bool invalidRebuildRatio(float rebuild_ratio) const;

  /**     
   * @brief Check whether the provided fill ratio is invalid.
   *
   * A valid rebuild ratio must be (0,1].
   *
   * @param fill_ratio The fill threshold ratio.
   * @return true if the value is invalid, false otherwise.
   */
  bool invalidFillRatio(float fill_ratio) const;

  /**
   * @brief Deleted copy constructor.
   *
   * The LKdTree is not copyable to avoid unintended duplication of the internal tree structure.
   */
  LKdTree(const LKdTree&) = delete;

  /**
   * @brief Deleted copy assignment operator.
   *
   * Prevents accidental copying of the KD-tree. Use move semantics instead.
   */
  LKdTree& operator=(const LKdTree&) = delete;

  /**
   * @brief Defaulted move constructor.
   *
   * Allows efficient transfer of ownership of the internal KD-tree structure.
   */
  LKdTree(LKdTree&&) noexcept = default;

  /**
   * @brief Defaulted move assignment operator.
   *
   * Supports move assignment of the entire KD-tree.
   */
  LKdTree& operator=(LKdTree&&) noexcept = default;
};

}  // namespace lkdtree

#ifdef LKDTREE_HEADER_ONLY
  #include "impl/Core.hpp"
  #include "impl/Insert.hpp"
  #include "impl/Build.hpp"
  #include "impl/Erase.hpp"
  #include "impl/RadiusSearch.hpp"
  #include "impl/BoxSearch.hpp"
  #include "impl/KnnSearch.hpp"
  #include "impl/Helpers.hpp"
  #include "impl/Accessors.hpp"
  #include "impl/ExactSearch.hpp"
#endif

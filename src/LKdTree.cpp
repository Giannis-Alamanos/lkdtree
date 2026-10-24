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

#include "lkdtree/LKdTree.h"
#include "lkdtree/impl/Core.hpp"
#include "lkdtree/impl/Insert.hpp"
#include "lkdtree/impl/Build.hpp"
#include "lkdtree/impl/Erase.hpp"
#include "lkdtree/impl/RadiusSearch.hpp"
#include "lkdtree/impl/BoxSearch.hpp"
#include "lkdtree/impl/KnnSearch.hpp"
#include "lkdtree/impl/Helpers.hpp"
#include "lkdtree/impl/Accessors.hpp"
#include "lkdtree/impl/ExactSearch.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

// Explicit-instantiation helper macro
#define LKDTREE_INSTANTIATE_PCL(PointT, K)                                                \
  /* class template itself */                                                             \
  template class LKdTree<PointT, K>;                                                      \
  /* member templates */                                                                    \
  template std::size_t LKdTree<PointT,K>                                                  \
    ::buildBalancedFromPointCloud<pcl::PointCloud<PointT>>(                               \
        pcl::PointCloud<PointT> const&, float);                                           \
  template std::size_t LKdTree<PointT,K>                                                  \
    ::insertPointCloud<pcl::PointCloud<PointT>>(pcl::PointCloud<PointT> const&);          \
  template void LKdTree<PointT,K>                                                         \
    ::insertPointCloudMaintain<pcl::PointCloud<PointT>>(                                  \
        pcl::PointCloud<PointT> const&, float, std::size_t, std::size_t, float,           \
        std::size_t, std::size_t, float, std::size_t, std::size_t, std::size_t, int,     \
        bool);                                                                            \
  template std::vector<std::vector<PointT const*>>                                        \
    LKdTree<PointT,K>::knnSearchBatch<std::vector<PointT>>(                               \
        std::vector<PointT> const&, std::size_t, float, int) const;                       \
  template std::vector<std::vector<PointT const*>>                                        \
    LKdTree<PointT,K>::knnSearchUniqueBatch<std::vector<PointT>>(                         \
        std::vector<PointT> const&, std::size_t, float, float, int) const;                \
  template std::vector<std::vector<PointT const*>>                                        \
    LKdTree<PointT,K>::radiusSearchBatch<std::vector<PointT>>(                            \
        std::vector<PointT> const&, float, int) const;                                    \
  template std::vector<std::vector<PointT const*>>                                        \
    LKdTree<PointT,K>::boxSearchBatch<std::vector<std::pair<PointT,PointT>>>(             \
        std::vector<std::pair<PointT,PointT>> const&, int) const

namespace lkdtree {

// Explicit instantiations for common PCL point types
LKDTREE_INSTANTIATE_PCL(pcl::PointXYZ, 3);
LKDTREE_INSTANTIATE_PCL(pcl::PointXYZI, 3);
LKDTREE_INSTANTIATE_PCL(pcl::PointXYZL, 3);
LKDTREE_INSTANTIATE_PCL(pcl::PointXYZRGBA, 3);
LKDTREE_INSTANTIATE_PCL(pcl::PointXYZRGB, 3);
LKDTREE_INSTANTIATE_PCL(pcl::PointXYZRGBL, 3);
LKDTREE_INSTANTIATE_PCL(pcl::PointXYZLAB, 3);
LKDTREE_INSTANTIATE_PCL(pcl::PointXYZHSV, 3);
LKDTREE_INSTANTIATE_PCL(pcl::InterestPoint, 3);
LKDTREE_INSTANTIATE_PCL(pcl::PointNormal, 3);
LKDTREE_INSTANTIATE_PCL(pcl::PointXYZRGBNormal, 3);
LKDTREE_INSTANTIATE_PCL(pcl::PointXYZINormal, 3);
LKDTREE_INSTANTIATE_PCL(pcl::PointXYZLNormal, 3);
LKDTREE_INSTANTIATE_PCL(pcl::PointWithRange, 3);
LKDTREE_INSTANTIATE_PCL(pcl::PointWithViewpoint, 3);
LKDTREE_INSTANTIATE_PCL(pcl::PointWithScale, 3);
LKDTREE_INSTANTIATE_PCL(pcl::PointSurfel, 3);
LKDTREE_INSTANTIATE_PCL(pcl::PointXY, 2);

} // namespace lkdtree
# LKdTree Library

Description
-----------
**LKdTree** is a standalone, high-performance C++ library for real-time spatial indexing and nearest-neighbor search in multi-dimensional point data. It provides an incremental, dynamically balanced KD-Tree optimized for streaming applications such as SLAM, LiDAR processing, and robotics, offering faster insertions and queries than traditional KD-Tree implementations.

LKdTree is released under the **GNU Affero General Public License v3 (AGPL-3.0)**, ensuring openness and reciprocity. Users who modify or extend the library must share their changes under the same license.  
The project is developed and maintained by **Ioannis Alamanos**, with contributions from the open-source community.

Dependencies
------------
**Core:**
- **C++20** compliant compiler  
  (GCC ≥ 10, Clang ≥ 10, or MSVC ≥ 19.3x)

**Optional (controlled by CMake options):**
- **OpenMP** (`LKDTREE_USE_OPENMP=ON` by default): enables multi-threaded tree build and query operations  
- **PCL** (`pcl/common`) — *required only for shared-library builds* (`LKDTREE_HEADER_ONLY=OFF`)

### Header-only vs Shared Library

| Mode | Description | Dependencies |
|------|--------------|--------------|
| **Header-only (`LKDTREE_HEADER_ONLY=ON`)** | All code is contained in headers and compiled directly into your target. Ideal when integrating into your own codebase or when customizing point types. | No runtime dependencies. OpenMP optional. |
| **Shared (`LKDTREE_HEADER_ONLY=OFF`)** | Builds a precompiled shared library that provides ready-to-use template instantiations for the most common PCL point types. This reduces build times and ensures ABI stability for downstream projects. | Requires PCL common and optionally OpenMP. |

Compiling
---------
### CMake options
| Option | Default | Description |
|---------|----------|-------------|
| `LKDTREE_HEADER_ONLY` | `OFF` | Build as a standalone shared library or as header-only. |
| `LKDTREE_USE_OPENMP`  | `ON`  | Enable OpenMP parallelism. |
| `BUILD_TESTS`          | `OFF` | Build unit tests. |
| `BUILD_BENCHMARKS`     | `OFF` | Build benchmarks. |

### Linux
LKdTree is currently supported only on **Linux (tested on Ubuntu 24.04)**.  

To install **LKdTree** as a shared library:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
make
sudo make install
```

To install **LKdTree** as a header-only library:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DLKDTREE_HEADER_ONLY=ON
sudo make install
```

Using LKdTree
-------------
Once installed, **LKdTree** can be easily integrated into any CMake project.

To use LKdTree as a dependency in a standard CMake project, simply call `find_package()` and link against the exported target `lkdtree::lkdtree`. For example:

```
cmake_minimum_required(VERSION 3.14)
project(my_project LANGUAGES CXX)

find_package(lkdtree REQUIRED)

add_executable(example src/main.cpp)
target_link_libraries(example PRIVATE lkdtree::lkdtree)
```

Include the LKdTree header in your source file:

```
#include <lkdtree/LKdTree.h>
```

API Documentation
-----------------
Comprehensive class and function-level documentation is available in the generated Doxygen pages.  
It covers all template parameters, configuration options, and detailed descriptions of insertion, query, and rebalancing strategies.

#### ➜ **Full API Reference:** [View Here](#)

---

### Example — Real-Time LiDAR–Inertial Odometry
The following example demonstrates how **LKdTree** can be integrated into a simplified **LiDAR–Inertial Odometry (LIO)** pipeline to maintain a dynamic local map and perform nearest-neighbor matching in real time.

```
// Initialize the LKdTree with a desired leaf size (e.g. 32 points per leaf)
auto lkdtree = std::make_unique<lkdtree::LKdTree<PointType, 3>>(32);

// Build an initial balanced tree from accumulated scans
// fill_ratio = 0.6 leaves some headroom for incremental insertions
lkdtree->buildBalancedFromPointCloud(point_cloud, 0.6f);

// Perform a k-NN search for a feature point
// (knnSearchBatch() can also be used for parallel queries)
auto neighbours = lkdtree->knnSearch(point_, 5, 5.0f);

// Optionally check redundancy using a box search
// (boxSearchBatch() can also be used for batch processing)
auto result = lkdtree->boxSearch(point_down, point_up);
if (!result.empty())
    continue;  // Skip insertion if the point is too close to existing ones

// Insert the new point while maintaining balance and capping tree size
// This keeps the structure logarithmic and avoids unbounded growth
lkdtree->insertMaintain(point,
                        1.5f,      
                        8 * 32,    
                        70'000,   
                        0.7f,      
                        100'000,   
                        600'000,   
                        16.0f,     
                        100'000,   
                        5'000,     
                        2'500,    
                        2);        

// For inserting entire point clouds with the same parameters,
// use the batch variant:
lkdtree->insertPointCloudMaintain(point_cloud, ...);     
```

License
-------
This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.  
By using, modifying, or distributing this software, you agree to the terms of the AGPL-3.0 license.  

**Copyright © 2025 Ioannis Alamanos.**  
All rights reserved.

For full license details, see the [`LICENSE`](LICENSE) file.

Citation
--------
If you use **LKdTree** in your research, software, or publications, please cite it appropriately.  

```
To be added
```
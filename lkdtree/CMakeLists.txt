cmake_minimum_required(VERSION 3.14)
project(lkdtree VERSION 0.1.0 LANGUAGES C CXX)

# -------------------------------------------------------------
# Build Settings
# -------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)         
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# -------------------------------------------------------------
# Options
# -------------------------------------------------------------
option(LKDTREE_HEADER_ONLY   "Build as a header-only library"       OFF)
option(LKDTREE_USE_OPENMP    "Enable OpenMP parallelism in LKdTree" ON)
option(BUILD_TESTS           "Build unit tests"                     OFF)
option(BUILD_BENCHMARKS      "Build benchmarks"                     OFF)

# -------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------
if(LKDTREE_USE_OPENMP)
  find_package(OpenMP QUIET)
endif()

if(NOT LKDTREE_HEADER_ONLY)
  find_package(PCL REQUIRED COMPONENTS common)
endif()

# -------------------------------------------------------------
# Library target
# -------------------------------------------------------------
if(LKDTREE_HEADER_ONLY)
  add_library(lkdtree INTERFACE)
    
  target_compile_features(lkdtree INTERFACE cxx_std_20)

  target_include_directories(lkdtree INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )

  target_compile_definitions(lkdtree INTERFACE LKDTREE_HEADER_ONLY=1)

  if(LKDTREE_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(lkdtree INTERFACE OpenMP::OpenMP_CXX)
  endif()

else()
  add_library(lkdtree SHARED src/LKdTree.cpp)

  target_compile_features(lkdtree PUBLIC cxx_std_20)

  target_include_directories(lkdtree
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
      ${PCL_INCLUDE_DIRS}
  )

  if(LKDTREE_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(lkdtree PUBLIC OpenMP::OpenMP_CXX)
  endif()

  set_target_properties(lkdtree PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME lkdtree
  )
endif()

# -------------------------------------------------------------
# Install + CMake package export
# -------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/usr" CACHE PATH "Install prefix" FORCE)
  endif()
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_INSTALL_PREFIX MATCHES "^/usr/?$")
  if(NOT DEFINED CMAKE_LIBRARY_ARCHITECTURE OR CMAKE_LIBRARY_ARCHITECTURE STREQUAL "")
    find_program(DPKG_ARCH_EXECUTABLE dpkg-architecture)
    if(DPKG_ARCH_EXECUTABLE)
      execute_process(
        COMMAND ${DPKG_ARCH_EXECUTABLE} -qDEB_HOST_MULTIARCH
        OUTPUT_VARIABLE CMAKE_LIBRARY_ARCHITECTURE
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    endif()
  endif()

  if(CMAKE_LIBRARY_ARCHITECTURE AND CMAKE_INSTALL_LIBDIR STREQUAL "lib")
    set(CMAKE_INSTALL_LIBDIR "lib/${CMAKE_LIBRARY_ARCHITECTURE}" CACHE PATH
        "Output directory for libraries" FORCE)
  endif()
endif()

message(STATUS "Installing libraries to: ${CMAKE_INSTALL_FULL_LIBDIR}")

# Install headers
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install the library (works for INTERFACE and normal targets)
install(TARGETS lkdtree
  EXPORT lkdtreeTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets file
install(EXPORT lkdtreeTargets
  FILE lkdtreeTargets.cmake
  NAMESPACE lkdtree::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lkdtree
)

# Generate and install the package config files
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/lkdtreeConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# These variables are substituted into the .in template
set(LKDTREE_HEADER_ONLY ${LKDTREE_HEADER_ONLY})
set(LKDTREE_USE_OPENMP  ${LKDTREE_USE_OPENMP})

configure_package_config_file(
  cmake/lkdtreeConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lkdtreeConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lkdtree
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/lkdtreeConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/lkdtreeConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lkdtree
)

# Make build tree findable without install
export(EXPORT lkdtreeTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/lkdtreeTargets.cmake"
  NAMESPACE lkdtree::
)
export(PACKAGE lkdtree)

# -------------------------------------------------------------
# Unit Tests  - GTest & PCL & MPI
# -------------------------------------------------------------
if (BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  # Some distros drag MPI via VTK when PCL is present in tests
  find_package(MPI REQUIRED C)
  find_package(PCL REQUIRED COMPONENTS common io)
  find_package(PCL REQUIRED COMPONENTS kdtree)

  add_executable(lkdtree_insert_test            test/LKdTreeInsertTest.cpp)
  add_executable(lkdtree_radius_search_test     test/LKdTreeRadiusSearchTest.cpp)
  add_executable(lkdtree_rebalance_test         test/LKdTreeRebalanceTest.cpp)
  add_executable(lkdtree_count_exact_test       test/LKdTreeCountExactTest.cpp)
  add_executable(lkdtree_erase_test             test/LKdTreeEraseTest.cpp)
  add_executable(lkdtree_knn_search_test        test/LKdTreeKnnSearchTest.cpp)
  add_executable(lkdtree_box_search_test        test/LKdTreeBoxSearchTest.cpp)

  foreach(tgt IN ITEMS
      lkdtree_insert_test
      lkdtree_radius_search_test
      lkdtree_rebalance_test
      lkdtree_count_exact_test
      lkdtree_erase_test
      lkdtree_knn_search_test
      lkdtree_box_search_test
  )
    target_include_directories(${tgt} PRIVATE ${PCL_INCLUDE_DIRS})
    target_link_libraries(${tgt}
      PRIVATE
        lkdtree
        GTest::gtest
        GTest::gtest_main
        ${PCL_LIBRARIES}
        MPI::MPI_C
    )
    add_test(NAME ${tgt} COMMAND ${tgt})
  endforeach()
endif()

# -------------------------------------------------------------
# Benchmarks - Google Benchmark & PCL
# -------------------------------------------------------------
if (BUILD_BENCHMARKS)
  find_package(PCL REQUIRED COMPONENTS common io)
  find_package(PCL REQUIRED COMPONENTS kdtree)
  find_package(benchmark REQUIRED)

  add_executable(lkdtree_benchmark_insert         benchmarks/LKdTreeBenchmarkInsert.cpp)
  add_executable(lkdtree_benchmark_knn_search     benchmarks/LKdTreeBenchmarkKnnSearch.cpp)
  add_executable(lkdtree_benchmark_radius_search  benchmarks/LKdTreeBenchmarkRadiusSearch.cpp)
  add_executable(lkdtree_benchmark_box_search     benchmarks/LKdTreeBenchmarkBoxSearch.cpp)
  add_executable(lkdtree_benchmark_rebuild        benchmarks/LKdTreeBenchmarkRebuild.cpp)

  foreach(tgt IN ITEMS
      lkdtree_benchmark_insert
      lkdtree_benchmark_knn_search
      lkdtree_benchmark_radius_search
      lkdtree_benchmark_box_search
      lkdtree_benchmark_rebuild
  )
    target_link_libraries(${tgt}
      PRIVATE
        lkdtree
        ${PCL_LIBRARIES}
        benchmark::benchmark
    )
  endforeach()
endif()

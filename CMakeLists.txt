cmake_minimum_required(VERSION 3.14)
project(cuda_benchmark CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 80)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/googletest)
add_subdirectory(third_party/benchmark)

# Add benchmark
add_subdirectory(benchmark)

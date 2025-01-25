#!/bin/bash

# Get the absolute path of the directory of the script
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Set version
BENCHMARK_VERSION="v1.8.3"
GTEST_VERSION="v1.14.0"

# Create third_party directory
mkdir -p "${SCRIPT_DIR}/../third_party"
cd "${SCRIPT_DIR}/../third_party"

# Download Google Benchmark
echo "Downloading Google Benchmark ${BENCHMARK_VERSION}..."
curl -L "https://github.com/google/benchmark/archive/refs/tags/${BENCHMARK_VERSION}.tar.gz" | tar xz
mv "benchmark-${BENCHMARK_VERSION#v}" benchmark

# Download Google Test
echo "Downloading Google Test ${GTEST_VERSION}..."
curl -L "https://github.com/google/googletest/archive/refs/tags/${GTEST_VERSION}.tar.gz" | tar xz
mv "googletest-${GTEST_VERSION#v}" googletest

echo "Dependencies downloaded successfully!"

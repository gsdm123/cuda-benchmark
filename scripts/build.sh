#!/bin/bash

# Get the absolute path of the directory of the script
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Function to setup CUDA environment
setup_cuda_env() {
    local cuda_version=$1
    local cuda_path="/usr/local/cuda-${cuda_version}"
    
    if [ ! -d "$cuda_path" ]; then
        echo "CUDA ${cuda_version} not found at ${cuda_path}"
        return 1
    fi
    
    export PATH="${cuda_path}/bin:$PATH"
    export LD_LIBRARY_PATH="${cuda_path}/lib64:$LD_LIBRARY_PATH"
    export CUDA_HOME="${cuda_path}"
    export CUDA_PATH="${cuda_path}"
    
    echo "Using CUDA ${cuda_version} from ${cuda_path}"
    return 0
}

# Parse command line arguments
CUDA_VERSION=${1:-"12.8"}  # Default to CUDA 12.8 if not specified

# Setup CUDA environment
if ! setup_cuda_env "$CUDA_VERSION"; then
    echo "Failed to setup CUDA environment"
    exit 1
fi

# Create build directory with CUDA version
BUILD_DIR="build_cuda_${CUDA_VERSION}"
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

# Configure
cmake ..

# Build
cmake --build . -j$(nproc)

# Create timestamp for results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="${SCRIPT_DIR}/../result/${TIMESTAMP}"
mkdir -p "${RESULT_DIR}"

# Run benchmarks and save results
echo "Running benchmarks..."
if [ -f "benchmark/run_benchmark" ]; then
    # Run benchmark once and save JSON output
    ./benchmark/run_benchmark \
        --benchmark_format=json \
        --benchmark_out="${RESULT_DIR}/benchmark_results.json" \
        --benchmark_out_format=json \
        --benchmark_color=true \
        --benchmark_counters_tabular=true || exit 1

    # Process results and generate visualizations
    cd ..
    python3 scripts/process_benchmark.py "${RESULT_DIR}/benchmark_results.json"

    # Create summary
    cat > "${RESULT_DIR}/summary.md" << EOF
# 性能测试结果摘要

## 测试环境
- 测试时间: $(date)
- GPU: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)
- CUDA: $(nvcc --version | grep "release" | awk '{print $5}')
- 系统: $(uname -a)
- 总耗时: $(grep "Total Execution Time:" "${RESULT_DIR}/report.md" | awk '{print $4, $5}')

## 测试结果
- [性能图表](benchmark_plot.png)
- [HTML报告](report.html)
- [Markdown报告](report.md)
- [原始数据](benchmark_results.json)
EOF

    echo "Results saved in: ${RESULT_DIR}"
    echo "View summary at: ${RESULT_DIR}/summary.md"
else
    echo "Error: benchmark executable not found"
    exit 1
fi

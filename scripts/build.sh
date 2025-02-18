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

 # Create timestamp for results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Function to build and run with specific optimization level
build_and_run() {
    local opt_level=$1
    echo "Building with optimization level $opt_level..."

    # Create and enter build directory
    BUILD_DIR="build_cuda_${CUDA_VERSION}_O${opt_level}"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Configure with specific optimization level
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CUDA_FLAGS="-O${opt_level}" \
          ..

    # Build
    cmake --build . -j$(nproc)

    RESULT_DIR="${SCRIPT_DIR}/../result/O${opt_level}_${TIMESTAMP}"
    mkdir -p "${RESULT_DIR}"

    # Run benchmarks and save results
    if [ -f "benchmark/run_benchmark" ]; then
        echo "Running benchmarks with O${opt_level}..."
        ./benchmark/run_benchmark \
            --benchmark_format=json \
            --benchmark_out="${RESULT_DIR}/benchmark_results.json" \
            --benchmark_out_format=json \
            --benchmark_color=true \
            --benchmark_counters_tabular=true || exit 1

        # Process results and generate visualizations
        cd ..
        python3 scripts/process_benchmark.py \
            "${RESULT_DIR}/benchmark_results.json" \
            "O${opt_level}"

        # Create summary
        cat > "${RESULT_DIR}/summary.md" << EOF
# 性能测试结果摘要 (优化级别: O${opt_level})

## 测试环境
- 测试时间: $(date)
- GPU: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)
- CUDA: $(nvcc --version | grep "release" | awk '{print $5}')
- 系统: $(uname -a)
- 优化级别: O${opt_level}
- 总耗时: $(grep "Total Execution Time:" "${RESULT_DIR}/report.md" | awk '{print $4, $5}')

## 测试结果
- [性能图表](benchmark_plot.png)
- [HTML报告](report.html)
- [Markdown报告](report.md)
- [原始数据](benchmark_results.json)

查看完整的 HTML 报告请打开: ${RESULT_DIR}/report.html
EOF

        echo "Results for O${opt_level} saved in: ${RESULT_DIR}"
        echo "View summary at: ${RESULT_DIR}/summary.md"
        echo "View HTML report at: ${RESULT_DIR}/report.html"
    else
        echo "Error: benchmark executable not found"
        exit 1
    fi

    cd "${SCRIPT_DIR}/.."
}

# Build and run for each optimization level
for opt in 0 1 2; do
    build_and_run $opt
done

# Generate comparison report
# O0 VS O1
echo "Generating O0 VS O1 report..."
python3 "$SCRIPT_DIR/compare_optimizations.py" \
    --base "$SCRIPT_DIR/../result/O0_${TIMESTAMP}/benchmark_results.json" \
    --optimized "$SCRIPT_DIR/../result/O1_${TIMESTAMP}/benchmark_results.json" \
    --output "$SCRIPT_DIR/../result/optimization_comparison_O0_vs_O1.html"

# O0 VS O2
echo "Generating O0 VS O2 report..."
python3 "$SCRIPT_DIR/compare_optimizations.py" \
    --base "$SCRIPT_DIR/../result/O0_${TIMESTAMP}/benchmark_results.json" \
    --optimized "$SCRIPT_DIR/../result/O2_${TIMESTAMP}/benchmark_results.json" \
    --output "$SCRIPT_DIR/../result/optimization_comparison_O0_vs_O2.html"

echo "All benchmarks completed. Results are available in the result directory."

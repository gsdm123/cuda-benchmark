#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <string>

#include "benchmark.h"

// Unified memory allocation and free performance test
static void BM_UnifiedMemoryAllocFree(benchmark::State& state) {
    auto start_total = std::chrono::high_resolution_clock::now();

    const int N = state.range(0);
    std::cout << "\n[Starting] Unified Memory Alloc/Free benchmark [size: " << N << "]"
              << std::endl;

    // Initialize CUDA and metrics
    InitCUDA();
    KernelMetrics metrics;
    metrics.size_kb = N * sizeof(float) / 1024.0;

    // Benchmark iterations
    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    CUDAEventTimer timer;

    for (auto _ : state) {
        float* ptr = nullptr;
        timer.Start();
        CUDA_CHECK(cudaMallocManaged(&ptr, N * sizeof(float)));
        CUDA_CHECK(cudaFree(ptr));
        timer.Stop();

        metrics.kernel_time = timer.ElapsedMillis();
        metrics.bandwidth = (metrics.size_kb * 1024) / (metrics.kernel_time * 1e-3) / 1e9;

        state.SetIterationTime(metrics.kernel_time / 1000.0);
        state.counters["KernelTime_ms"] = metrics.kernel_time;
        state.counters["Bandwidth_GB/s"] = metrics.bandwidth;
        state.counters["GFLOPS"] = metrics.gflops;
        state.counters["Size_KB"] = metrics.size_kb;
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    metrics.total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    state.counters["TotalTime_ms"] = metrics.total_time;

    // Cleanup
    CleanupCUDA();
    std::cout << "[Completed] Unified Memory Alloc/Free benchmark" << std::endl;
}

// Unified memory access kernel
__global__ void accessUnifiedMemory(float* data, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] + 1.0f;
    }
}

// Unified memory access performance test
static void BM_UnifiedMemoryAccess(benchmark::State& state) {
    auto start_total = std::chrono::high_resolution_clock::now();

    const int N = state.range(0);
    std::cout << "\n[Starting] Unified Memory Access benchmark [size: " << N << "]" << std::endl;

    // Initialize CUDA and metrics
    InitCUDA();
    KernelMetrics metrics;
    metrics.size_kb = N * sizeof(float) / 1024.0;

    float* data = nullptr;

    // Allocate and initialize unified memory
    CUDA_CHECK(cudaMallocManaged(&data, N * sizeof(float)));
    std::fill_n(data, N, 1.0f);

    // Synchronize device
    CUDA_CHECK(cudaDeviceSynchronize());

    // Configure kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Warm up
    std::cout << "[Warmup] Running warmup iteration..." << std::endl;
    accessUnifiedMemory<<<numBlocks, blockSize>>>(data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark iterations
    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    CUDAEventTimer timer;

    for (auto _ : state) {
        timer.Start();
        accessUnifiedMemory<<<numBlocks, blockSize>>>(data, N);
        timer.Stop();

        metrics.kernel_time = timer.ElapsedMillis();
        metrics.bandwidth = (2.0 * metrics.size_kb * 1024) / (metrics.kernel_time * 1e-3) / 1e9;
        metrics.gflops = (1.0 * N) / (metrics.kernel_time * 1e-3) / 1e9;

        state.SetIterationTime(metrics.kernel_time / 1000.0);
        state.counters["KernelTime_ms"] = metrics.kernel_time;
        state.counters["Bandwidth_GB/s"] = metrics.bandwidth;
        state.counters["GFLOPS"] = metrics.gflops;
        state.counters["Size_KB"] = metrics.size_kb;
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    metrics.total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    state.counters["TotalTime_ms"] = metrics.total_time;

    // Cleanup
    CUDA_CHECK(cudaFree(data));
    CleanupCUDA();
    std::cout << "[Completed] Unified Memory Access benchmark" << std::endl;
}

// Page migration performance test
static void BM_UnifiedMemoryMigration(benchmark::State& state) {
    auto start_total = std::chrono::high_resolution_clock::now();

    const int N = state.range(0);
    std::cout << "\n[Starting] Unified Memory Migration benchmark [size: " << N << "]" << std::endl;

    // Initialize CUDA and metrics
    InitCUDA();
    KernelMetrics metrics;
    metrics.size_kb = N * sizeof(float) / 1024.0;

    float* data = nullptr;

    CUDA_CHECK(cudaMallocManaged(&data, N * sizeof(float)));
    std::fill_n(data, N, 1.0f);

    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    CUDAEventTimer timer;

    for (auto _ : state) {
        timer.Start();
        // Migrate to GPU
        CUDA_CHECK(cudaMemPrefetchAsync(data, N * sizeof(float), 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        // Migrate to CPU
        CUDA_CHECK(cudaMemPrefetchAsync(data, N * sizeof(float), cudaCpuDeviceId));
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.Stop();

        metrics.kernel_time = timer.ElapsedMillis();
        metrics.bandwidth = (2.0 * metrics.size_kb * 1024) / (metrics.kernel_time * 1e-3) / 1e9;

        state.SetIterationTime(metrics.kernel_time / 1000.0);
        state.counters["KernelTime_ms"] = metrics.kernel_time;
        state.counters["Bandwidth_GB/s"] = metrics.bandwidth;
        state.counters["GFLOPS"] = metrics.gflops;
        state.counters["Size_KB"] = metrics.size_kb;
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    metrics.total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    state.counters["TotalTime_ms"] = metrics.total_time;

    // Cleanup
    CUDA_CHECK(cudaFree(data));
    CleanupCUDA();
    std::cout << "[Completed] Unified Memory Migration benchmark" << std::endl;
}

// Register unified memory allocation and free benchmark
BENCHMARK(BM_UnifiedMemoryAllocFree)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register unified memory access benchmark
BENCHMARK(BM_UnifiedMemoryAccess)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// // Register unified memory migration benchmark
// BENCHMARK(BM_UnifiedMemoryMigration)
//     ->RangeMultiplier(2)
//     ->Range(1 << 8, 1 << 10)
//     ->UseManualTime()
//     ->Unit(benchmark::kMicrosecond)
//     ->Repetitions(2);

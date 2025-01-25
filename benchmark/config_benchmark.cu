#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <string>

#include "benchmark.h"

// Cache configuration test kernel
__global__ void cacheTestKernel(float* data, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * 2.0f;
    }
}

// Cache configuration performance test
static void BM_CacheConfig(benchmark::State& state) {
    try {
        const int N = state.range(0);
        const cudaFuncCache cacheConfig = (cudaFuncCache)state.range(1);
        std::cout << "\n[Starting] Cache Config benchmark [size: " << N << ", config: " 
                  << cacheConfig << "]" << std::endl;

        InitCUDA();
        KernelMetrics metrics;
        metrics.size_kb = N * sizeof(float) / 1024.0;

        float* d_data = nullptr;
        auto start_total = std::chrono::high_resolution_clock::now();

        try {
            CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
            CUDA_CHECK(cudaFuncSetCacheConfig(cacheTestKernel, cacheConfig));

            // Configure kernel
            int blockSize = 256;
            int numBlocks = (N + blockSize - 1) / blockSize;

            // Warm up
            std::cout << "[Warmup] Running warmup iteration..." << std::endl;
            cacheTestKernel<<<numBlocks, blockSize>>>(d_data, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Benchmark iterations
            std::cout << "[Running] Executing benchmark iterations..." << std::endl;
            CUDAEventTimer timer;

            for (auto _ : state) {
                timer.Start();
                cacheTestKernel<<<numBlocks, blockSize>>>(d_data, N);
                timer.Stop();

                metrics.kernel_time = timer.ElapsedMillis();
                metrics.bandwidth = (2.0 * N * sizeof(float)) / (metrics.kernel_time * 1e-3) / 1e9;
                metrics.gflops = (N * 1.0) / (metrics.kernel_time * 1e-3) / 1e9;

                state.SetIterationTime(metrics.kernel_time / 1000.0);
                state.counters["KernelTime_ms"] = metrics.kernel_time;
                state.counters["Bandwidth_GB/s"] = metrics.bandwidth;
                state.counters["GFLOPS"] = metrics.gflops;
                state.counters["Size_KB"] = metrics.size_kb;
            }

            CUDA_CHECK(cudaFree(d_data));

        } catch (...) {
            if (d_data) cudaFree(d_data);
            throw;
        }

        auto end_total = std::chrono::high_resolution_clock::now();
        metrics.total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        state.counters["TotalTime_ms"] = metrics.total_time;

        CleanupCUDA();
        std::cout << "[Completed] Cache Config benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Shared memory test kernel
__global__ void sharedMemTestKernel(float* data, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        sdata[tid] = data[i];
        __syncthreads();
        data[i] = sdata[tid] * 2.0f;
    }
}

// Shared memory configuration performance test
static void BM_SharedMemConfig(benchmark::State& state) {
    try {
        const int N = state.range(0);
        std::cout << "\n[Starting] Shared Memory Config benchmark [size: " << N << "]" << std::endl;

        InitCUDA();
        KernelMetrics metrics;
        metrics.size_kb = N * sizeof(float) / 1024.0;

        float* d_data = nullptr;
        auto start_total = std::chrono::high_resolution_clock::now();

        try {
            CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));

            // Configure kernel
            int blockSize = 256;
            int numBlocks = (N + blockSize - 1) / blockSize;

            // Warm up
            std::cout << "[Warmup] Running warmup iteration..." << std::endl;
            sharedMemTestKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_data, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Benchmark iterations
            std::cout << "[Running] Executing benchmark iterations..." << std::endl;
            CUDAEventTimer timer;

            for (auto _ : state) {
                timer.Start();
                sharedMemTestKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_data, N);
                timer.Stop();

                metrics.kernel_time = timer.ElapsedMillis();
                metrics.bandwidth = (2.0 * N * sizeof(float)) / (metrics.kernel_time * 1e-3) / 1e9;
                metrics.gflops = (N * 1.0) / (metrics.kernel_time * 1e-3) / 1e9;

                state.SetIterationTime(metrics.kernel_time / 1000.0);
                state.counters["KernelTime_ms"] = metrics.kernel_time;
                state.counters["Bandwidth_GB/s"] = metrics.bandwidth;
                state.counters["GFLOPS"] = metrics.gflops;
                state.counters["Size_KB"] = metrics.size_kb;
            }

            CUDA_CHECK(cudaFree(d_data));

        } catch (...) {
            if (d_data) cudaFree(d_data);
            throw;
        }

        auto end_total = std::chrono::high_resolution_clock::now();
        metrics.total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        state.counters["TotalTime_ms"] = metrics.total_time;

        CleanupCUDA();
        std::cout << "[Completed] Shared Memory Config benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Register benchmarks
BENCHMARK(BM_CacheConfig)
    ->RangeMultiplier(2)
    ->Ranges({{1 << 8, 1 << 10}, {cudaFuncCachePreferNone, cudaFuncCachePreferL1}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

BENCHMARK(BM_SharedMemConfig)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

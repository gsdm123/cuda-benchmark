#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <string>

#include "benchmark.h"

namespace cg = cooperative_groups;

// Warp shuffle performance test kernel
__global__ void warpShuffleKernel(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x % warpSize;

    if (tid < n) {
        float value = data[tid];
        value = __shfl_sync(0xffffffff, value, (lane + 1) % warpSize);
        data[tid] = value;
    }
}

// Warp ballot performance test kernel
__global__ void warpBallotKernel(float* data, int* results, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float value = data[tid];
        unsigned mask = __ballot_sync(0xffffffff, value > 0.0f);
        if (threadIdx.x % warpSize == 0) {
            results[tid / warpSize] = __popc(mask);
        }
    }
}

// Cooperative groups performance test kernel
__global__ void cooperativeKernel(float* data, int n) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int tid = grid.thread_rank();
    if (tid < n) {
        float value = data[tid];
        block.sync();
        value += warp.shfl_down(value, 1);
        data[tid] = value;
    }
}

// Warp shuffle performance test
static void BM_WarpShuffle(benchmark::State& state) {
    try {
        const int N = state.range(0);
        std::cout << "\n[Starting] Warp Shuffle benchmark [size: " << N << "]" << std::endl;

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
            warpShuffleKernel<<<numBlocks, blockSize>>>(d_data, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Benchmark iterations
            std::cout << "[Running] Executing benchmark iterations..." << std::endl;
            CUDAEventTimer timer;

            for (auto _ : state) {
                timer.Start();
                warpShuffleKernel<<<numBlocks, blockSize>>>(d_data, N);
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
        std::cout << "[Completed] Warp Shuffle benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Warp ballot performance test
static void BM_WarpBallot(benchmark::State& state) {
    try {
        const int N = state.range(0);
        std::cout << "\n[Starting] Warp Ballot benchmark [size: " << N << "]" << std::endl;

        InitCUDA();
        KernelMetrics metrics;
        metrics.size_kb = (N * sizeof(float) + (N / 32) * sizeof(int)) / 1024.0;

        float* d_data = nullptr;
        int* d_results = nullptr;
        auto start_total = std::chrono::high_resolution_clock::now();

        try {
            CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_results, (N / 32 + 1) * sizeof(int)));

            // Configure kernel
            int blockSize = 256;
            int numBlocks = (N + blockSize - 1) / blockSize;

            // Warm up
            std::cout << "[Warmup] Running warmup iteration..." << std::endl;
            warpBallotKernel<<<numBlocks, blockSize>>>(d_data, d_results, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Benchmark iterations
            std::cout << "[Running] Executing benchmark iterations..." << std::endl;
            CUDAEventTimer timer;

            for (auto _ : state) {
                timer.Start();
                warpBallotKernel<<<numBlocks, blockSize>>>(d_data, d_results, N);
                timer.Stop();

                metrics.kernel_time = timer.ElapsedMillis();
                metrics.bandwidth = (N * sizeof(float) + (N / 32) * sizeof(int)) / 
                                  (metrics.kernel_time * 1e-3) / 1e9;
                metrics.gflops = (N * 2.0) / (metrics.kernel_time * 1e-3) / 1e9;

                state.SetIterationTime(metrics.kernel_time / 1000.0);
                state.counters["KernelTime_ms"] = metrics.kernel_time;
                state.counters["Bandwidth_GB/s"] = metrics.bandwidth;
                state.counters["GFLOPS"] = metrics.gflops;
                state.counters["Size_KB"] = metrics.size_kb;
            }

            CUDA_CHECK(cudaFree(d_data));
            CUDA_CHECK(cudaFree(d_results));

        } catch (...) {
            if (d_data) cudaFree(d_data);
            if (d_results) cudaFree(d_results);
            throw;
        }

        auto end_total = std::chrono::high_resolution_clock::now();
        metrics.total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        state.counters["TotalTime_ms"] = metrics.total_time;

        CleanupCUDA();
        std::cout << "[Completed] Warp Ballot benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Cooperative groups performance test
static void BM_CooperativeGroups(benchmark::State& state) {
    try {
        const int N = state.range(0);
        std::cout << "\n[Starting] Cooperative Groups benchmark [size: " << N << "]" << std::endl;

        InitCUDA();
        KernelMetrics metrics;
        metrics.size_kb = N * sizeof(float) / 1024.0;

        float* d_data = nullptr;
        auto start_total = std::chrono::high_resolution_clock::now();

        try {
            CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));

            // Check cooperative launch support
            int deviceId;
            CUDA_CHECK(cudaGetDevice(&deviceId));
            int supportCooperativeLaunch;
            CUDA_CHECK(cudaDeviceGetAttribute(&supportCooperativeLaunch, 
                cudaDevAttrCooperativeLaunch, deviceId));

            if (!supportCooperativeLaunch) {
                state.SkipWithError("Device does not support cooperative launch");
                return;
            }

            // Configure kernel
            int blockSize = 256;
            int numBlocks = (N + blockSize - 1) / blockSize;

            // Warm up
            std::cout << "[Warmup] Running warmup iteration..." << std::endl;
            cooperativeKernel<<<numBlocks, blockSize>>>(d_data, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Benchmark iterations
            std::cout << "[Running] Executing benchmark iterations..." << std::endl;
            CUDAEventTimer timer;

            for (auto _ : state) {
                timer.Start();
                cooperativeKernel<<<numBlocks, blockSize>>>(d_data, N);
                timer.Stop();

                metrics.kernel_time = timer.ElapsedMillis();
                metrics.bandwidth = (2.0 * N * sizeof(float)) / (metrics.kernel_time * 1e-3) / 1e9;
                metrics.gflops = (N * 2.0) / (metrics.kernel_time * 1e-3) / 1e9;

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
        std::cout << "[Completed] Cooperative Groups benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Register warp shuffle benchmark
BENCHMARK(BM_WarpShuffle)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register warp ballot benchmark
BENCHMARK(BM_WarpBallot)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register cooperative groups benchmark
BENCHMARK(BM_CooperativeGroups)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

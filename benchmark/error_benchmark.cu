#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <string>

#include "benchmark.h"

// Error detection performance test
static void BM_GetLastError(benchmark::State& state) {
    try {
        std::cout << "\n[Starting] GetLastError benchmark" << std::endl;

        InitCUDA();
        KernelMetrics metrics;
        metrics.size_kb = 0;  // No data transfer

        auto start_total = std::chrono::high_resolution_clock::now();
        CUDAEventTimer timer;

        std::cout << "[Running] Executing benchmark iterations..." << std::endl;
        for (auto _ : state) {
            timer.Start();
            cudaGetLastError();
            timer.Stop();

            metrics.kernel_time = timer.ElapsedMillis();

            state.SetIterationTime(metrics.kernel_time / 1000.0);
            state.counters["KernelTime_ms"] = metrics.kernel_time;
            state.counters["Bandwidth_GB/s"] = metrics.bandwidth;
            state.counters["GFLOPS"] = metrics.gflops;
            state.counters["Size_KB"] = metrics.size_kb;
        }

        auto end_total = std::chrono::high_resolution_clock::now();
        metrics.total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        state.counters["TotalTime_ms"] = metrics.total_time;

        CleanupCUDA();
        std::cout << "[Completed] GetLastError benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Error checking performance test
static void BM_ErrorCheck(benchmark::State& state) {
    try {
        std::cout << "\n[Starting] Error Check benchmark" << std::endl;

        InitCUDA();
        KernelMetrics metrics;
        metrics.size_kb = 0;  // No data transfer

        auto start_total = std::chrono::high_resolution_clock::now();
        CUDAEventTimer timer;

        std::cout << "[Running] Executing benchmark iterations..." << std::endl;
        for (auto _ : state) {
            timer.Start();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                const char* errorStr = cudaGetErrorString(error);
                (void)errorStr;  // Avoid compiler warning
            }
            timer.Stop();

            metrics.kernel_time = timer.ElapsedMillis();

            state.SetIterationTime(metrics.kernel_time / 1000.0);
            state.counters["KernelTime_ms"] = metrics.kernel_time;
            state.counters["Bandwidth_GB/s"] = metrics.bandwidth;
            state.counters["GFLOPS"] = metrics.gflops;
            state.counters["Size_KB"] = metrics.size_kb;
        }

        auto end_total = std::chrono::high_resolution_clock::now();
        metrics.total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        state.counters["TotalTime_ms"] = metrics.total_time;

        CleanupCUDA();
        std::cout << "[Completed] Error Check benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Error recovery performance test
static void BM_ErrorRecovery(benchmark::State& state) {
    try {
        const int N = state.range(0);
        std::cout << "\n[Starting] Error Recovery benchmark [size: " << N << "]" << std::endl;

        InitCUDA();
        KernelMetrics metrics;
        metrics.size_kb = N * N * sizeof(float) / 1024.0;  // Large allocation to trigger error

        float* ptr = nullptr;
        auto start_total = std::chrono::high_resolution_clock::now();
        CUDAEventTimer timer;

        std::cout << "[Running] Executing benchmark iterations..." << std::endl;
        for (auto _ : state) {
            timer.Start();
            cudaError_t error = cudaMalloc(&ptr, size_t(N) * size_t(N) * sizeof(float));
            if (error != cudaSuccess) {
                cudaGetLastError();
                CUDA_CHECK(cudaDeviceReset());
            }
            timer.Stop();

            metrics.kernel_time = timer.ElapsedMillis();

            state.SetIterationTime(metrics.kernel_time / 1000.0);
            state.counters["KernelTime_ms"] = metrics.kernel_time;
            state.counters["Bandwidth_GB/s"] = metrics.bandwidth;
            state.counters["GFLOPS"] = metrics.gflops;
            state.counters["Size_KB"] = metrics.size_kb;
        }

        auto end_total = std::chrono::high_resolution_clock::now();
        metrics.total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        state.counters["TotalTime_ms"] = metrics.total_time;

        if (ptr) CUDA_CHECK(cudaFree(ptr));
        CleanupCUDA();
        std::cout << "[Completed] Error Recovery benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Register getLastError benchmark
BENCHMARK(BM_GetLastError)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register errorCheck benchmark
BENCHMARK(BM_ErrorCheck)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register errorRecovery benchmark
// BENCHMARK(BM_ErrorRecovery)
//     ->RangeMultiplier(2)
//     ->Range(1 << 12, 1 << 15)  // Large sizes to trigger errors
//     ->UseManualTime()
//     ->Unit(benchmark::kMicrosecond)
//     ->Repetitions(2);

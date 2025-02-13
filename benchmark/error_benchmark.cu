#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <string>

#include "benchmark.h"

// Error detection performance test
static void BM_GetLastError(benchmark::State& state) {
    auto start_total = std::chrono::high_resolution_clock::now();

    std::cout << "\n[Starting] GetLastError benchmark" << std::endl;

    // Initialize CUDA and metrics
    InitCUDA();
    KernelMetrics metrics;
    metrics.size_kb = 0;  // No data transfer

    // Benchmark iterations
    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    CUDAEventTimer timer;

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

    // Cleanup
    CleanupCUDA();
    std::cout << "[Completed] GetLastError benchmark" << std::endl;
}

// Error checking performance test
static void BM_ErrorCheck(benchmark::State& state) {
    auto start_total = std::chrono::high_resolution_clock::now();

    std::cout << "\n[Starting] Error Check benchmark" << std::endl;

    // Initialize CUDA and metrics
    InitCUDA();
    KernelMetrics metrics;
    metrics.size_kb = 0;  // No data transfer

    // Benchmark iterations
    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    CUDAEventTimer timer;

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

    // Cleanup
    CleanupCUDA();
    std::cout << "[Completed] Error Check benchmark" << std::endl;
}
// Register getLastError benchmark
BENCHMARK(BM_GetLastError)->UseManualTime()->Unit(benchmark::kMicrosecond)->Repetitions(2);

// Register errorCheck benchmark
BENCHMARK(BM_ErrorCheck)->UseManualTime()->Unit(benchmark::kMicrosecond)->Repetitions(2);

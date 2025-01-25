#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <string>

#include "benchmark.h"

// Device management performance test
static void BM_CudaSetDevice(benchmark::State& state) {
    try {
        std::cout << "\n[Starting] CudaSetDevice benchmark" << std::endl;
        InitCUDA();

        KernelMetrics metrics;
        metrics.size_kb = 0;  // No data transfer

        auto start_total = std::chrono::high_resolution_clock::now();
        CUDAEventTimer timer;

        std::cout << "[Running] Executing benchmark iterations..." << std::endl;
        for (auto _ : state) {
            timer.Start();
            CUDA_CHECK(cudaSetDevice(0));
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
        std::cout << "[Completed] CudaSetDevice benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Get device properties performance test
static void BM_CudaGetDeviceProperties(benchmark::State& state) {
    try {
        std::cout << "\n[Starting] CudaGetDeviceProperties benchmark" << std::endl;
        InitCUDA();

        KernelMetrics metrics;
        metrics.size_kb = sizeof(cudaDeviceProp) / 1024.0;

        auto start_total = std::chrono::high_resolution_clock::now();
        CUDAEventTimer timer;
        cudaDeviceProp prop;

        std::cout << "[Running] Executing benchmark iterations..." << std::endl;
        for (auto _ : state) {
            timer.Start();
            CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
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
        std::cout << "[Completed] CudaGetDeviceProperties benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Context synchronization performance test
static void BM_CudaDeviceSynchronize(benchmark::State& state) {
    try {
        std::cout << "\n[Starting] CudaDeviceSynchronize benchmark" << std::endl;
        InitCUDA();

        KernelMetrics metrics;
        metrics.size_kb = 0;  // No data transfer

        auto start_total = std::chrono::high_resolution_clock::now();
        CUDAEventTimer timer;

        std::cout << "[Running] Executing benchmark iterations..." << std::endl;
        for (auto _ : state) {
            timer.Start();
            CUDA_CHECK(cudaDeviceSynchronize());
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
        std::cout << "[Completed] CudaDeviceSynchronize benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Event management performance test
static void BM_CudaEventCreateDestroy(benchmark::State& state) {
    try {
        std::cout << "\n[Starting] CudaEventCreateDestroy benchmark" << std::endl;
        InitCUDA();

        KernelMetrics metrics;
        metrics.size_kb = 0;  // No data transfer

        auto start_total = std::chrono::high_resolution_clock::now();
        CUDAEventTimer timer;

        std::cout << "[Running] Executing benchmark iterations..." << std::endl;
        for (auto _ : state) {
            cudaEvent_t event;
            timer.Start();
            CUDA_CHECK(cudaEventCreate(&event));
            CUDA_CHECK(cudaEventDestroy(event));
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
        std::cout << "[Completed] CudaEventCreateDestroy benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Register setDevice benchmark test
BENCHMARK(BM_CudaSetDevice)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register getDeviceProperties benchmark test
BENCHMARK(BM_CudaGetDeviceProperties)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register deviceSynchronize benchmark test
BENCHMARK(BM_CudaDeviceSynchronize)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register eventCreateDestroy benchmark test
BENCHMARK(BM_CudaEventCreateDestroy)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

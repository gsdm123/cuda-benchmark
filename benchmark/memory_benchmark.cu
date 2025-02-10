#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <string>

#include "benchmark.h"

// Host memory allocation performance test
static void BM_CudaMallocHost(benchmark::State& state) {
    try {
        const int N = state.range(0);
        std::cout << "\n[Starting] CudaMallocHost benchmark [size: " << N << "]" << std::endl;

        InitCUDA();
        KernelMetrics metrics;
        metrics.size_kb = N * sizeof(float) / 1024.0;

        float* h_data = nullptr;
        auto start_total = std::chrono::high_resolution_clock::now();

        std::cout << "[Running] Executing benchmark iterations..." << std::endl;
        CUDAEventTimer timer;

        for (auto _ : state) {
            timer.Start();
            CUDA_CHECK(cudaMallocHost(&h_data, N * sizeof(float)));
            CUDA_CHECK(cudaFreeHost(h_data));
            timer.Stop();

            metrics.kernel_time = timer.ElapsedMillis();
            metrics.bandwidth = (N * sizeof(float)) / (metrics.kernel_time * 1e-3) / 1e9;

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
        std::cout << "[Completed] CudaMallocHost benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Device memory allocation performance test
static void BM_CudaMalloc(benchmark::State& state) {
    try {
        const int N = state.range(0);
        std::cout << "\n[Starting] CudaMalloc benchmark [size: " << N << "]" << std::endl;

        InitCUDA();
        KernelMetrics metrics;
        metrics.size_kb = N * sizeof(float) / 1024.0;

        float* d_data = nullptr;
        auto start_total = std::chrono::high_resolution_clock::now();

        std::cout << "[Running] Executing benchmark iterations..." << std::endl;
        CUDAEventTimer timer;

        for (auto _ : state) {
            timer.Start();
            CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
            CUDA_CHECK(cudaFree(d_data));
            timer.Stop();

            metrics.kernel_time = timer.ElapsedMillis();
            metrics.bandwidth = (N * sizeof(float)) / (metrics.kernel_time * 1e-3) / 1e9;

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
        std::cout << "[Completed] CudaMalloc benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Memory copy performance tests
static void BM_CudaMemcpy(benchmark::State& state, cudaMemcpyKind kind) {
    try {
        const int N = state.range(0);
        const char* kindStr = (kind == cudaMemcpyDeviceToHost) ? "D2H" : "H2D";
        std::cout << "\n[Starting] CudaMemcpy" << kindStr << " benchmark [size: " << N << "]" << std::endl;

        InitCUDA();
        KernelMetrics metrics;
        metrics.size_kb = N * sizeof(float) / 1024.0;

        float *h_data = nullptr, *d_data = nullptr;
        CUDA_CHECK(cudaMallocHost(&h_data, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));

        auto start_total = std::chrono::high_resolution_clock::now();
        std::cout << "[Running] Executing benchmark iterations..." << std::endl;
        CUDAEventTimer timer;

        for (auto _ : state) {
            timer.Start();
            CUDA_CHECK(cudaMemcpy(
                (kind == cudaMemcpyDeviceToHost) ? h_data : d_data,
                (kind == cudaMemcpyDeviceToHost) ? d_data : h_data,
                N * sizeof(float),
                kind));
            timer.Stop();

            metrics.kernel_time = timer.ElapsedMillis();
            metrics.bandwidth = (N * sizeof(float)) / (metrics.kernel_time * 1e-3) / 1e9;

            state.SetIterationTime(metrics.kernel_time / 1000.0);
            state.counters["KernelTime_ms"] = metrics.kernel_time;
            state.counters["Bandwidth_GB/s"] = metrics.bandwidth;
            state.counters["GFLOPS"] = metrics.gflops;
            state.counters["Size_KB"] = metrics.size_kb;
        }

        auto end_total = std::chrono::high_resolution_clock::now();
        metrics.total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        state.counters["TotalTime_ms"] = metrics.total_time;

        CUDA_CHECK(cudaFreeHost(h_data));
        CUDA_CHECK(cudaFree(d_data));

        CleanupCUDA();
        std::cout << "[Completed] CudaMemcpy" << kindStr << " benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

static void BM_CudaMemcpyD2H(benchmark::State& state) {
    BM_CudaMemcpy(state, cudaMemcpyDeviceToHost);
}

static void BM_CudaMemcpyH2D(benchmark::State& state) {
    BM_CudaMemcpy(state, cudaMemcpyHostToDevice);
}

// Register host memory allocation benchmark
BENCHMARK(BM_CudaMallocHost)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register device memory allocation benchmark
BENCHMARK(BM_CudaMalloc)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register device-to-host memory copy benchmark
BENCHMARK(BM_CudaMemcpyD2H)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register host-to-device memory copy benchmark
BENCHMARK(BM_CudaMemcpyH2D)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

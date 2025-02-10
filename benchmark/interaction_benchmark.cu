#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <string>

#include "benchmark.h"

// Host-side data preprocessing performance test
static void BM_HostPreprocess(benchmark::State& state) {
    try {
        const int N = state.range(0);
        std::cout << "\n[Starting] Host Preprocess benchmark [size: " << N << "]" << std::endl;

        InitCUDA();
        KernelMetrics metrics;
        metrics.size_kb = N * sizeof(float) / 1024.0;

        float* h_data = nullptr;
        auto start_total = std::chrono::high_resolution_clock::now();

        try {
            h_data = new float[N];

            std::cout << "[Running] Executing benchmark iterations..." << std::endl;
            CUDAEventTimer timer;

            for (auto _ : state) {
                timer.Start();
                for (int i = 0; i < N; i++) {
                    h_data[i] = float(i) * 0.5f;
                }
                timer.Stop();

                metrics.kernel_time = timer.ElapsedMillis();
                metrics.bandwidth = (N * sizeof(float)) / (metrics.kernel_time * 1e-3) / 1e9;

                state.SetIterationTime(metrics.kernel_time / 1000.0);
                state.counters["KernelTime_ms"] = metrics.kernel_time;
                state.counters["Bandwidth_GB/s"] = metrics.bandwidth;
                state.counters["GFLOPS"] = metrics.gflops;
                state.counters["Size_KB"] = metrics.size_kb;
            }

            delete[] h_data;

        } catch (...) {
            delete[] h_data;
            throw;
        }

        auto end_total = std::chrono::high_resolution_clock::now();
        metrics.total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        state.counters["TotalTime_ms"] = metrics.total_time;

        CleanupCUDA();
        std::cout << "[Completed] Host Preprocess benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Host-device data interaction performance test
static void BM_HostDeviceInteraction(benchmark::State& state) {
    try {
        const int N = state.range(0);
        std::cout << "\n[Starting] Host-Device Interaction benchmark [size: " << N << "]" << std::endl;

        InitCUDA();
        KernelMetrics metrics;
        metrics.size_kb = N * sizeof(float) * 2 / 1024.0;  // Input + output data

        float *h_data = nullptr, *d_data = nullptr;
        auto start_total = std::chrono::high_resolution_clock::now();

        try {
            CUDA_CHECK(cudaMallocHost(&h_data, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));

            std::cout << "[Running] Executing benchmark iterations..." << std::endl;
            CUDAEventTimer timer;

            for (auto _ : state) {
                timer.Start();

                // Host data preparation
                for (int i = 0; i < N; i++) {
                    h_data[i] = float(i);
                }

                // Transfer to device
                CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

                // Read back from device
                CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));

                // Host processing result
                float sum = 0.0f;
                for (int i = 0; i < N; i++) {
                    sum += h_data[i];
                }

                timer.Stop();

                metrics.kernel_time = timer.ElapsedMillis();
                metrics.bandwidth = (2.0 * N * sizeof(float)) / (metrics.kernel_time * 1e-3) / 1e9;

                state.SetIterationTime(metrics.kernel_time / 1000.0);
                state.counters["KernelTime_ms"] = metrics.kernel_time;
                state.counters["Bandwidth_GB/s"] = metrics.bandwidth;
                state.counters["GFLOPS"] = metrics.gflops;
                state.counters["Size_KB"] = metrics.size_kb;
            }

            CUDA_CHECK(cudaFreeHost(h_data));
            CUDA_CHECK(cudaFree(d_data));
        } catch (...) {
            if (h_data) cudaFreeHost(h_data);
            if (d_data) cudaFree(d_data);
            throw;
        }

        auto end_total = std::chrono::high_resolution_clock::now();
        metrics.total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        state.counters["TotalTime_ms"] = metrics.total_time;

        CleanupCUDA();
        std::cout << "[Completed] Host-Device Interaction benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Empty kernel for launch overhead test
__global__ void emptyKernel() {}

// Kernel launch overhead performance test
static void BM_KernelLaunchOverhead(benchmark::State& state) {
    try {
        std::cout << "\n[Starting] Kernel Launch Overhead benchmark" << std::endl;

        InitCUDA();
        KernelMetrics metrics;
        metrics.size_kb = 0;  // No data transfer

        auto start_total = std::chrono::high_resolution_clock::now();
        CUDAEventTimer timer;

        std::cout << "[Running] Executing benchmark iterations..." << std::endl;
        for (auto _ : state) {
            timer.Start();
            emptyKernel<<<1, 1>>>();
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
        std::cout << "[Completed] Kernel Launch Overhead benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Register host preprocess benchmark
BENCHMARK(BM_HostPreprocess)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register host-device interaction benchmark
BENCHMARK(BM_HostDeviceInteraction)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register kernel launch overhead benchmark
BENCHMARK(BM_KernelLaunchOverhead)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

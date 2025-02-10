#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <string>

#include "benchmark.h"

// Kernel vector add
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Vector add performance test
static void BM_VectorAdd(benchmark::State& state) {
    try {
        const int N = state.range(0);
        std::cout << "\n[Starting] Vector Add benchmark [size: " << N << "]" << std::endl;

        // Initialize CUDA and metrics
        InitCUDA();
        KernelMetrics metrics;
        metrics.size_kb = N * sizeof(float) * 3 / 1024.0;  // Input + output data size

        // Allocate and initialize data
        float *h_a = nullptr, *h_b = nullptr;
        float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

        auto start_total = std::chrono::high_resolution_clock::now();

        try {
            // Memory allocation and data initialization
            h_a = new float[N];
            h_b = new float[N];
            std::fill_n(h_a, N, 1.0f);
            std::fill_n(h_b, N, 2.0f);

            CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

            CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

            // Configure kernel
            int blockSize = 256;
            int numBlocks = (N + blockSize - 1) / blockSize;

            // Warm up
            std::cout << "[Warmup] Running warmup iteration..." << std::endl;
            vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Benchmark iterations
            std::cout << "[Running] Executing benchmark iterations..." << std::endl;
            CUDAEventTimer timer;

            for (auto _ : state) {
                timer.Start();
                vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
                timer.Stop();

                // Record metrics
                metrics.kernel_time = timer.ElapsedMillis();
                metrics.bandwidth = (3.0 * N * sizeof(float)) / (metrics.kernel_time * 1e-3) / 1e9;
                metrics.gflops = (N * 1.0) / (metrics.kernel_time * 1e-3) / 1e9;

                // Set benchmark metrics
                state.SetIterationTime(metrics.kernel_time / 1000.0);
                state.counters["KernelTime_ms"] = metrics.kernel_time;
                state.counters["Bandwidth_GB/s"] = metrics.bandwidth;
                state.counters["GFLOPS"] = metrics.gflops;
                state.counters["Size_KB"] = metrics.size_kb;
            }

            // Cleanup
            delete[] h_a;
            delete[] h_b;
            CUDA_CHECK(cudaFree(d_a));
            CUDA_CHECK(cudaFree(d_b));
            CUDA_CHECK(cudaFree(d_c));

        } catch (...) {
            delete[] h_a;
            delete[] h_b;
            if (d_a) cudaFree(d_a);
            if (d_b) cudaFree(d_b);
            if (d_c) cudaFree(d_c);
            throw;
        }

        // Record total time
        auto end_total = std::chrono::high_resolution_clock::now();
        metrics.total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        state.counters["TotalTime_ms"] = metrics.total_time;

        CleanupCUDA();
        std::cout << "[Completed] Vector Add benchmark" << std::endl;

    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    } catch (...) {
        state.SkipWithError("Unknown error occurred");
    }
}

// Register vector addition benchmark
BENCHMARK(BM_VectorAdd)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

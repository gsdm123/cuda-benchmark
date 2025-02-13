#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <string>

#include "benchmark.h"

// Kernel vector add
__global__ void kernelVectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// CPU reference implementation for vector addition
void VectorAddCPUReference(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Vector add performance test
static void BM_BasicVectorAdd(benchmark::State& state) {
    auto start_total = std::chrono::high_resolution_clock::now();

    const int N = state.range(0);
    std::cout << "\n[Starting] Vector Add benchmark [size: " << N << "]" << std::endl;

    // Initialize CUDA and metrics
    InitCUDA();
    KernelMetrics metrics;
    metrics.size_kb = N * sizeof(float) * 3 / 1024.0;

    // Allocate and initialize data
    float *h_a = nullptr, *h_b = nullptr, *h_c = nullptr;
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    // Memory allocation and data initialization
    h_a = new float[N];
    h_b = new float[N];
    h_c = new float[N];
    std::fill_n(h_a, N, 1.0f);
    std::fill_n(h_b, N, 2.0f);

#ifdef ENABLE_VERIFICATION
    // Initialize data for verification
    float* h_c_cpu = new float[N];
    std::cout << "[CPU] Calculating CPU reference results..." << std::endl;
    VectorAddCPUReference(h_a, h_b, h_c_cpu, N);
    std::cout << "[CPU] CPU reference results calculated" << std::endl;
#endif

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
    kernelVectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef ENABLE_VERIFICATION
    // Verify results
    std::cout << "[Verify] Verifying vector addition..." << std::endl;
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    Verify<float> verifier(h_c, h_c_cpu, N);
    if (!verifier.VerifyResults()) {
        state.SkipWithError("Result verification failed");
        delete[] h_a;
        delete[] h_b;
        delete[] h_c;
        delete[] h_c_cpu;
        return;
    }
    std::cout << "[Verify] Vector addition verified" << std::endl;
#endif
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
#ifdef ENABLE_VERIFICATION
    delete[] h_c_cpu;
#endif

    // Benchmark iterations
    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    CUDAEventTimer timer;

    for (auto _ : state) {
        timer.Start();
        kernelVectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
        timer.Stop();

        metrics.kernel_time = timer.ElapsedMillis();
        metrics.bandwidth = (metrics.size_kb * 1024) / (metrics.kernel_time * 1e-3) / 1e9;
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
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CleanupCUDA();
    std::cout << "[Completed] Vector Add benchmark" << std::endl;
}

// Register vector addition benchmark
BENCHMARK(BM_BasicVectorAdd)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

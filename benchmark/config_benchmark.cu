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

// CPU reference implementation for cache configuration
void CacheConfigCPUReference(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = data[i] * 2.0f;
    }
}

// Cache configuration performance test
static void BM_CacheConfig(benchmark::State& state) {
    auto start_total = std::chrono::high_resolution_clock::now();

    const int N = state.range(0);
    const cudaFuncCache cacheConfig = (cudaFuncCache)state.range(1);
    std::cout << "\n[Starting] Cache Config benchmark [size: " << N << ", config: " << cacheConfig
              << "]" << std::endl;

    // Initialize CUDA and metrics
    InitCUDA();
    KernelMetrics metrics;
    metrics.size_kb = N * sizeof(float) / 1024.0;

    // Allocate and initialize data
    float* h_data = nullptr;
    float* d_data = nullptr;
    // Memory allocation and data initialization
    h_data = new float[N];
    std::fill_n(h_data, N, 1.0f);

#ifdef ENABLE_VERIFICATION
    // Initialize data for verification
    float* h_data_cpu = new float[N];
    std::fill_n(h_data_cpu, N, 1.0f);
    std::cout << "[CPU] Calculating CPU reference results..." << std::endl;
    CacheConfigCPUReference(h_data_cpu, N);
    std::cout << "[CPU] CPU reference results calculated" << std::endl;
#endif

    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaFuncSetCacheConfig(cacheTestKernel, cacheConfig));

    // Configure kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Warm up
    std::cout << "[Warmup] Running warmup iteration..." << std::endl;
    cacheTestKernel<<<numBlocks, blockSize>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef ENABLE_VERIFICATION
    // Verify results
    std::cout << "[Verify] Verifying cache configuration..." << std::endl;
    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));
    Verify<float> verifier(h_data, h_data_cpu, N);
    if (!verifier.VerifyResults()) {
        state.SkipWithError("Result verification failed");
        delete[] h_data;
        delete[] h_data_cpu;
        return;
    }
    std::cout << "[Verify] Cache configuration verified" << std::endl;
#endif
    delete[] h_data;
#ifdef ENABLE_VERIFICATION
    delete[] h_data_cpu;
#endif
    // Benchmark iterations
    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    CUDAEventTimer timer;

    for (auto _ : state) {
        timer.Start();
        cacheTestKernel<<<numBlocks, blockSize>>>(d_data, N);
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
    CUDA_CHECK(cudaFree(d_data));
    CleanupCUDA();
    std::cout << "[Completed] Cache Config benchmark" << std::endl;
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

// CPU reference implementation for shared memory configuration
void SharedMemConfigCPUReference(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = data[i] * 2.0f;
    }
}

// Shared memory configuration performance test
static void BM_SharedMemConfig(benchmark::State& state) {
    auto start_total = std::chrono::high_resolution_clock::now();
    const int N = state.range(0);
    std::cout << "\n[Starting] Shared Memory Config benchmark [size: " << N << "]" << std::endl;

    // Initialize CUDA and metrics
    InitCUDA();
    KernelMetrics metrics;
    metrics.size_kb = N * sizeof(float) / 1024.0;

    // Allocate and initialize data
    float* h_data = nullptr;
    float* d_data = nullptr;
    // Memory allocation and data initialization
    h_data = new float[N];
    std::fill_n(h_data, N, 1.0f);

#ifdef ENABLE_VERIFICATION
    // Initialize data for verification
    float* h_data_cpu = new float[N];
    std::fill_n(h_data_cpu, N, 1.0f);
    std::cout << "[CPU] Calculating CPU reference results..." << std::endl;
    SharedMemConfigCPUReference(h_data_cpu, N);
    std::cout << "[CPU] CPU reference results calculated" << std::endl;
#endif

    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    // Configure kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Warm up
    std::cout << "[Warmup] Running warmup iteration..." << std::endl;
    sharedMemTestKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef ENABLE_VERIFICATION
    // Verify results
    std::cout << "[Verify] Verifying shared memory configuration..." << std::endl;
    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));
    Verify<float> verifier(h_data, h_data_cpu, N);
    if (!verifier.VerifyResults()) {
        state.SkipWithError("Result verification failed");
        delete[] h_data;
        delete[] h_data_cpu;
        return;
    }
    std::cout << "[Verify] Shared memory configuration verified" << std::endl;
#endif
    delete[] h_data;
#ifdef ENABLE_VERIFICATION
    delete[] h_data_cpu;
#endif

    // Benchmark iterations
    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    CUDAEventTimer timer;

    for (auto _ : state) {
        timer.Start();
        sharedMemTestKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_data, N);
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
    CUDA_CHECK(cudaFree(d_data));
    CleanupCUDA();
    std::cout << "[Completed] Shared Memory Config benchmark" << std::endl;
}

// Register cache configuration benchmark
BENCHMARK(BM_CacheConfig)
    ->RangeMultiplier(2)
    ->Ranges({{1 << 8, 1 << 10}, {cudaFuncCachePreferNone, cudaFuncCachePreferL1}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register shared memory configuration benchmark
BENCHMARK(BM_SharedMemConfig)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

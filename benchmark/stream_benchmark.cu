#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <string>

#include "benchmark.h"

// Simple vector addition kernel
__global__ void kernelStreamSample(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Stream creation and destruction performance test
static void BM_StreamCreateDestroy(benchmark::State& state) {
    auto start_total = std::chrono::high_resolution_clock::now();

    std::cout << "\n[Starting] Stream Create/Destroy benchmark" << std::endl;

    // Initialize CUDA and metrics
    InitCUDA();
    KernelMetrics metrics;
    metrics.size_kb = 0;  // No data transfer

    // Benchmark iterations
    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    CUDAEventTimer timer;

    for (auto _ : state) {
        cudaStream_t stream;
        timer.Start();
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
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
    std::cout << "[Completed] Stream Create/Destroy benchmark" << std::endl;
}

// Stream synchronization performance test
static void BM_StreamSynchronize(benchmark::State& state) {
    auto start_total = std::chrono::high_resolution_clock::now();
    std::cout << "\n[Starting] Stream Synchronize benchmark" << std::endl;

    // Initialize CUDA and metrics
    InitCUDA();
    KernelMetrics metrics;
    metrics.size_kb = 0;  // No data transfer

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Benchmark iterations
    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    CUDAEventTimer timer;

    for (auto _ : state) {
        timer.Start();
        CUDA_CHECK(cudaStreamSynchronize(stream));
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
    CUDA_CHECK(cudaStreamDestroy(stream));
    CleanupCUDA();
    std::cout << "[Completed] Stream Synchronize benchmark" << std::endl;
}

// Multi-stream concurrency execution performance test
static void BM_StreamConcurrency(benchmark::State& state) {
    auto start_total = std::chrono::high_resolution_clock::now();

    const int N = state.range(0);
    const int numStreams = 4;
    const int streamSize = N / numStreams;

    std::cout << "\n[Starting] Stream Concurrency benchmark [size: " << N << "]" << std::endl;

    // Initialize CUDA and metrics
    InitCUDA();
    KernelMetrics metrics;
    metrics.size_kb = N * sizeof(float) * 3 / 1024.0;  // Input + output data

    // Allocate and initialize data
    float *h_a = nullptr, *h_b = nullptr, *h_c = nullptr;
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    // Memory allocation and data initialization
    CUDA_CHECK(cudaMallocHost(&h_a, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_b, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    std::fill_n(h_a, N, 1.0f);
    std::fill_n(h_b, N, 2.0f);

    // Create streams
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Configure kernel
    int blockSize = 256;
    int streamBlocks = (streamSize + blockSize - 1) / blockSize;

    // Warm up
    std::cout << "[Warmup] Running warmup iteration..." << std::endl;
    for (int i = 0; i < numStreams; ++i) {
        int offset = i * streamSize;
        kernelStreamSample<<<streamBlocks, blockSize, 0, streams[i]>>>(d_a + offset, d_b + offset,
                                                                       d_c + offset, streamSize);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark iterations
    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    CUDAEventTimer timer;

    for (auto _ : state) {
        timer.Start();
        // Launch operations in multiple streams
        for (int i = 0; i < numStreams; ++i) {
            int offset = i * streamSize;
            CUDA_CHECK(cudaMemcpyAsync(d_a + offset, h_a + offset, streamSize * sizeof(float),
                                       cudaMemcpyHostToDevice, streams[i]));
            CUDA_CHECK(cudaMemcpyAsync(d_b + offset, h_b + offset, streamSize * sizeof(float),
                                       cudaMemcpyHostToDevice, streams[i]));
            kernelStreamSample<<<streamBlocks, blockSize, 0, streams[i]>>>(
                d_a + offset, d_b + offset, d_c + offset, streamSize);
            CUDA_CHECK(cudaMemcpyAsync(h_c + offset, d_c + offset, streamSize * sizeof(float),
                                       cudaMemcpyDeviceToHost, streams[i]));
        }

        // Wait for all streams
        for (int i = 0; i < numStreams; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
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
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    for (int i = 0; i < numStreams; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CleanupCUDA();
    std::cout << "[Completed] Stream Concurrency benchmark" << std::endl;
}

// Register stream creation and destruction benchmark
BENCHMARK(BM_StreamCreateDestroy)->UseManualTime()->Unit(benchmark::kMicrosecond)->Repetitions(2);

// Register stream synchronization benchmark
BENCHMARK(BM_StreamSynchronize)->UseManualTime()->Unit(benchmark::kMicrosecond)->Repetitions(2);

// Register multi-stream concurrency benchmark
BENCHMARK(BM_StreamConcurrency)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <string>

#include "benchmark.h"

// Vector addition kernel
__global__ void kernelVectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Matrix multiplication kernel
__global__ void kernelMatrixMul(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

// Reduction kernel
__global__ void kernelReduce(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

// Vector addition performance test
static void BM_KernelVectorAdd(benchmark::State& state) {
    try {
        const int N = state.range(0);
        std::cout << "\n[Starting] Vector Add kernel benchmark [size: " << N << "]" << std::endl;

        InitCUDA();
        KernelMetrics metrics;
        metrics.size_kb = N * sizeof(float) * 3 / 1024.0;  // Input + output data

        float *h_a = nullptr, *h_b = nullptr;
        float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
        auto start_total = std::chrono::high_resolution_clock::now();

        try {
            // Allocate and initialize memory
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
            kernelVectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Benchmark iterations
            std::cout << "[Running] Executing benchmark iterations..." << std::endl;
            CUDAEventTimer timer;

            for (auto _ : state) {
                timer.Start();
                kernelVectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
                timer.Stop();

                metrics.kernel_time = timer.ElapsedMillis();
                metrics.bandwidth = (3.0 * N * sizeof(float)) / (metrics.kernel_time * 1e-3) / 1e9;
                metrics.gflops = (N * 1.0) / (metrics.kernel_time * 1e-3) / 1e9;

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

        auto end_total = std::chrono::high_resolution_clock::now();
        metrics.total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        state.counters["TotalTime_ms"] = metrics.total_time;

        CleanupCUDA();
        std::cout << "[Completed] Vector Add kernel benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Matrix multiplication performance test
static void BM_KernelMatrixMul(benchmark::State& state) {
    try {
        const int N = state.range(0);
        std::cout << "\n[Starting] Matrix Multiplication kernel benchmark [size: " << N << "x" << N << "]" 
                  << std::endl;

        InitCUDA();
        KernelMetrics metrics;
        metrics.size_kb = N * N * sizeof(float) * 3 / 1024.0;  // Input + output matrices

        float *h_a = nullptr, *h_b = nullptr;
        float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
        auto start_total = std::chrono::high_resolution_clock::now();

        try {
            // Allocate and initialize memory
            size_t matrix_size = N * N * sizeof(float);
            h_a = new float[N * N];
            h_b = new float[N * N];
            std::fill_n(h_a, N * N, 1.0f);
            std::fill_n(h_b, N * N, 2.0f);

            CUDA_CHECK(cudaMalloc(&d_a, matrix_size));
            CUDA_CHECK(cudaMalloc(&d_b, matrix_size));
            CUDA_CHECK(cudaMalloc(&d_c, matrix_size));

            CUDA_CHECK(cudaMemcpy(d_a, h_a, matrix_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_b, h_b, matrix_size, cudaMemcpyHostToDevice));

            // Configure kernel
            dim3 blockSize(16, 16);
            dim3 numBlocks((N + blockSize.x - 1) / blockSize.x, 
                          (N + blockSize.y - 1) / blockSize.y);

            // Warm up
            std::cout << "[Warmup] Running warmup iteration..." << std::endl;
            kernelMatrixMul<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N, N, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Benchmark iterations
            std::cout << "[Running] Executing benchmark iterations..." << std::endl;
            CUDAEventTimer timer;

            for (auto _ : state) {
                timer.Start();
                kernelMatrixMul<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N, N, N);
                timer.Stop();

                metrics.kernel_time = timer.ElapsedMillis();
                metrics.bandwidth = (3.0 * matrix_size) / (metrics.kernel_time * 1e-3) / 1e9;
                metrics.gflops = (2.0 * N * N * N) / (metrics.kernel_time * 1e-3) / 1e9;

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

        auto end_total = std::chrono::high_resolution_clock::now();
        metrics.total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        state.counters["TotalTime_ms"] = metrics.total_time;

        CleanupCUDA();
        std::cout << "[Completed] Matrix Multiplication kernel benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Reduction performance test
static void BM_KernelReduce(benchmark::State& state) {
    try {
        const int N = state.range(0);
        std::cout << "\n[Starting] Reduction kernel benchmark [size: " << N << "]" << std::endl;

        InitCUDA();
        KernelMetrics metrics;
        metrics.size_kb = N * sizeof(float) * 2 / 1024.0;  // Input + output data

        float *h_input = nullptr, *h_output = nullptr;
        float *d_input = nullptr, *d_output = nullptr;
        auto start_total = std::chrono::high_resolution_clock::now();

        try {
            // Allocate and initialize memory
            h_input = new float[N];
            h_output = new float[N];
            std::fill_n(h_input, N, 1.0f);

            CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

            CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

            // Configure kernel
            int blockSize = 256;
            int numBlocks = (N + blockSize - 1) / blockSize;

            // Warm up
            std::cout << "[Warmup] Running warmup iteration..." << std::endl;
            kernelReduce<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Benchmark iterations
            std::cout << "[Running] Executing benchmark iterations..." << std::endl;
            CUDAEventTimer timer;

            for (auto _ : state) {
                timer.Start();
                kernelReduce<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N);
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

            // Cleanup
            delete[] h_input;
            delete[] h_output;
            CUDA_CHECK(cudaFree(d_input));
            CUDA_CHECK(cudaFree(d_output));

        } catch (...) {
            delete[] h_input;
            delete[] h_output;
            if (d_input) cudaFree(d_input);
            if (d_output) cudaFree(d_output);
            throw;
        }

        auto end_total = std::chrono::high_resolution_clock::now();
        metrics.total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        state.counters["TotalTime_ms"] = metrics.total_time;

        CleanupCUDA();
        std::cout << "[Completed] Reduction kernel benchmark" << std::endl;
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Register vector addition benchmark
BENCHMARK(BM_KernelVectorAdd)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register matrix multiplication benchmark
BENCHMARK(BM_KernelMatrixMul)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 6)  // Smaller range for matrix multiplication
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register reduction benchmark
BENCHMARK(BM_KernelReduce)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

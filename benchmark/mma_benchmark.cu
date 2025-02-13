#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <chrono>
#include <iostream>

#include "benchmark.h"

// Constants
constexpr int BLOCK_SIZE = 16;

// Basic matrix multiplication implementation
__global__ void MatrixMulBasicKernel(const float* A, const float* B, float* C, int M, int N,
                                     int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Matrix multiplication implementation using shared memory
__global__ void MatrixMulSharedMemKernel(const float* A, const float* B, float* C, int M, int N,
                                         int K) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    for (int i = 0; i < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
        // Load data to shared memory
        if (row < M && i * BLOCK_SIZE + tx < K)
            As[ty][tx] = A[row * K + i * BLOCK_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (i * BLOCK_SIZE + ty < K && col < N)
            Bs[ty][tx] = B[(i * BLOCK_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Calculate partial results
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// CPU reference implementation for matrix multiplication
void MatrixMultiplyCPUReference(const float* A, const float* B, float* C, int M, int N, int K) {
    // Initialize output matrix with zeros
    std::fill_n(C, M * N, 0.0f);

    // Compute matrix multiplication
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

// Benchmark class
class BM_MatrixMultiplication : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        M = state.range(0);
        N = state.range(0);
        K = state.range(0);

        // Calculate data size (KB)
        size_kb = (M * K + K * N + M * N) * sizeof(float) / 1024.0;
        metrics.size_kb = size_kb;

        std::cout << "\n[Starting] Matrix Multiplication benchmark [size: " << M << "x" << N << "x"
                  << K << "]" << std::endl;

        InitCUDA();
        AllocateMemory();
        InitializeData();

#ifdef ENABLE_VERIFICATION
        // Only generate CPU reference results when verification is enabled
        h_C_cpu = new float[M * N];
        std::cout << "[CPU] Calculating CPU reference results..." << std::endl;
        MatrixMultiplyCPUReference(h_A, h_B, h_C_cpu, M, N, K);
        std::cout << "[CPU] CPU reference results calculated" << std::endl;
#endif
    }

    void TearDown(const benchmark::State& state) override {
        FreeMemory();
        CleanupCUDA();
#ifdef ENABLE_VERIFICATION
        delete[] h_C_cpu;
#endif
        std::cout << "[Completed] Resources freed" << std::endl;
    }

protected:
    void AllocateMemory() {
        // Host memory
        h_A = new float[M * K];
        h_B = new float[K * N];
        h_C = new float[M * N];

        // Device memory
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    }

    void InitializeData() {
        // Initialize input matrices
        for (int i = 0; i < M * K; i++) {
            h_A[i] = rand() / (float)RAND_MAX;
        }
        for (int i = 0; i < K * N; i++) {
            h_B[i] = rand() / (float)RAND_MAX;
        }

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    }

    void FreeMemory() {
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    // Helper function to verify current implementation
    bool VerifyImplementation(const char* impl_name) {
#ifdef ENABLE_VERIFICATION
        // Copy GPU results back to host
        CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        // Verify results
        std::cout << "[Verify] Verifying " << impl_name << " implementation..." << std::endl;
        Verify<float> verify(h_C, h_C_cpu, M * N);
        bool passed = verify.VerifyResults(1e-2);
        std::cout << "[Verify] " << impl_name << " implementation "
                  << (passed ? "passed" : "failed") << std::endl;
        return passed;
#else
        return true;
#endif
    }

    int M, N, K;
    float size_kb;
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    CUDAEventTimer timer;
    KernelMetrics metrics;
#ifdef ENABLE_VERIFICATION
    float* h_C_cpu;
#endif
};

// Basic CUDA Core matrix multiplication benchmark
BENCHMARK_DEFINE_F(BM_MatrixMultiplication, BasicMMA)(benchmark::State& state) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    auto start_total = std::chrono::high_resolution_clock::now();

    // Warm up
    std::cout << "[Warmup] Running warmup iteration..." << std::endl;
    MatrixMulBasicKernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify results
    if (!VerifyImplementation("Basic Matrix Multiplication")) {
        state.SkipWithError("Result verification failed");
        return;
    }

    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    for (auto _ : state) {
        timer.Start();
        MatrixMulBasicKernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        timer.Stop();

        // Record metrics
        metrics.kernel_time = timer.ElapsedMillis();
        metrics.bandwidth = (metrics.size_kb * 1024) / (metrics.kernel_time * 1e-3) / 1e9;
        metrics.gflops = (2.0 * M * N * K) / (metrics.kernel_time * 1e-3) / 1e9;

        // Set benchmark metrics
        state.SetIterationTime(metrics.kernel_time / 1000.0);
        state.counters["KernelTime_ms"] = metrics.kernel_time;
        state.counters["Bandwidth_GB/s"] = metrics.bandwidth;
        state.counters["GFLOPS"] = metrics.gflops;
        state.counters["Size_KB"] = metrics.size_kb;
    }
    auto end_total = std::chrono::high_resolution_clock::now();
    metrics.total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    state.counters["TotalTime_ms"] = metrics.total_time;
}

// Shared Memory matrix multiplication benchmark
BENCHMARK_DEFINE_F(BM_MatrixMultiplication, SharedMemMMA)(benchmark::State& state) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    auto start_total = std::chrono::high_resolution_clock::now();

    // Warm up
    std::cout << "[Warmup] Running warmup iteration..." << std::endl;
    MatrixMulSharedMemKernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify results
    if (!VerifyImplementation("Shared Memory Matrix Multiplication")) {
        state.SkipWithError("Result verification failed");
        return;
    }

    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    for (auto _ : state) {
        timer.Start();
        MatrixMulSharedMemKernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        timer.Stop();

        // Record metrics
        metrics.kernel_time = timer.ElapsedMillis();
        metrics.bandwidth = (metrics.size_kb * 1024) / (metrics.kernel_time * 1e-3) / 1e9;
        metrics.gflops = (2.0 * M * N * K) / (metrics.kernel_time * 1e-3) / 1e9;

        // Set benchmark metrics
        state.SetIterationTime(metrics.kernel_time / 1000.0);
        state.counters["KernelTime_ms"] = metrics.kernel_time;
        state.counters["Bandwidth_GB/s"] = metrics.bandwidth;
        state.counters["GFLOPS"] = metrics.gflops;
        state.counters["Size_KB"] = metrics.size_kb;
    }
    auto end_total = std::chrono::high_resolution_clock::now();
    metrics.total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    state.counters["TotalTime_ms"] = metrics.total_time;
}

// Register Basic CUDA Core matrix multiplication benchmark
BENCHMARK_REGISTER_F(BM_MatrixMultiplication, BasicMMA)
    ->RangeMultiplier(2)
    ->Range(128, 2048)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->Repetitions(2);

// Register Shared Memory matrix multiplication benchmark
BENCHMARK_REGISTER_F(BM_MatrixMultiplication, SharedMemMMA)
    ->RangeMultiplier(2)
    ->Range(128, 2048)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->Repetitions(2);

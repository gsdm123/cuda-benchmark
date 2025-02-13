#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <string>

#include "benchmark.h"

namespace cg = cooperative_groups;

// Warp shuffle performance test kernel
__global__ void warpShuffleKernel(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x % warpSize;

    if (tid < n) {
        float value = data[tid];
        value = __shfl_sync(0xffffffff, value, (lane + 1) % warpSize);
        data[tid] = value;
    }
}

// CPU reference implementation for warp shuffle
void WarpShuffleCPUReference(const float* input, float* output, int N) {
    for (int i = 0; i < N; i += 32) {  // 32 is warp size
        for (int j = 0; j < 32 && (i + j) < N; j++) {
            output[i + j] = input[i + (j + 1) % 32];
        }
    }
}

// Warp ballot performance test kernel
__global__ void warpBallotKernel(float* input, int* output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float value = input[tid];
        unsigned mask = __ballot_sync(0xffffffff, value > 0.0f);
        if (tid % warpSize == 0) {
            output[tid / warpSize] = __popc(mask);
        }
    }
}

// CPU reference implementation for warp ballot
void WarpBallotCPUReference(const float* input, int* output, int N) {
    for (int i = 0; i < N; i += 32) {
        int count = 0;
        for (int j = 0; j < 32 && (i + j) < N; j++) {
            if (input[i + j] > 0.0f) {
                count++;
            }
        }
        output[i / 32] = count;
    }
}

// Cooperative groups performance test kernel
__global__ void cooperativeKernel(float* data, int n) {
    cg::thread_block block = cg::this_thread_block();
    int tid = block.thread_rank();

    int index = blockIdx.x * blockDim.x + tid;

    if (index < n) {
        data[index] += tid;
    }
}

// CPU reference implementation for cooperative groups
void CooperativeGroupsCPUReference(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] += i % 256;
    }
}

// Warp-level reduction kernel
__global__ void WarpReduceKernel(float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;
    int warp_id = tid / warpSize;

    if (tid < N) {
        float val = input[tid];

        // Warp reduce using shuffle
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        // First thread in warp writes the result
        if (lane == 0) {
            output[warp_id] = val;
        }
    }
}

// CPU reference implementation for warp reduce
void WarpReduceCPUReference(const float* input, float* output, int N) {
    int warp_size = 32;

    for (int i = 0; i < N; i += warp_size) {
        float sum = 0.0f;
        int end = min(i + warp_size, N);

        for (int j = i; j < end; j++) {
            sum += input[j];
        }

        output[i / warp_size] = sum;
    }
}

// Warp shuffle performance test
static void BM_WarpShuffle(benchmark::State& state) {
    auto start_total = std::chrono::high_resolution_clock::now();

    const int N = state.range(0);
    std::cout << "\n[Starting] Warp Shuffle benchmark [size: " << N << "]" << std::endl;

    // Initialize CUDA and metrics
    InitCUDA();
    KernelMetrics metrics;
    metrics.size_kb = N * sizeof(float) / 1024.0;

    // Allocate and initialize data
    float *h_input = nullptr, *h_output = nullptr;
    float* d_data = nullptr;
    // Memory allocation and data initialization
    h_input = new float[N];
    h_output = new float[N];
    std::fill_n(h_input, N, 1.0f);

#ifdef ENABLE_VERIFICATION
    // Initialize data for verification
    float* h_output_cpu = new float[N];
    std::cout << "[CPU] Calculating CPU reference results..." << std::endl;
    WarpShuffleCPUReference(h_input, h_output_cpu, N);
    std::cout << "[CPU] CPU reference results calculated" << std::endl;
#endif

    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Configure kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Warm up
    std::cout << "[Warmup] Running warmup iteration..." << std::endl;
    warpShuffleKernel<<<numBlocks, blockSize>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef ENABLE_VERIFICATION
    // Verify results
    std::cout << "[Verify] Verifying warp shuffle..." << std::endl;
    CUDA_CHECK(cudaMemcpy(h_output, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));
    Verify<float> verifier(h_output, h_output_cpu, N);
    if (!verifier.VerifyResults()) {
        state.SkipWithError("Result verification failed");
        delete[] h_input;
        delete[] h_output;
        delete[] h_output_cpu;
        return;
    }
    std::cout << "[Verify] Warp shuffle verified" << std::endl;
#endif
    delete[] h_input;
    delete[] h_output;
#ifdef ENABLE_VERIFICATION
    delete[] h_output_cpu;
#endif

    // Benchmark iterations
    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    CUDAEventTimer timer;

    for (auto _ : state) {
        timer.Start();
        warpShuffleKernel<<<numBlocks, blockSize>>>(d_data, N);
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
    std::cout << "[Completed] Warp Shuffle benchmark" << std::endl;
}

// Warp ballot performance test
static void BM_WarpBallot(benchmark::State& state) {
    auto start_total = std::chrono::high_resolution_clock::now();

    const int N = state.range(0);
    std::cout << "\n[Starting] Warp Ballot benchmark [size: " << N << "]" << std::endl;

    // Initialize CUDA and metrics
    InitCUDA();
    KernelMetrics metrics;
    metrics.size_kb = (N * sizeof(float) + N * sizeof(int)) / 1024.0;

    // Allocate and initialize data
    float* h_input = nullptr;
    int* h_output = nullptr;
    float* d_input = nullptr;
    int* d_output = nullptr;
    // Memory allocation and data initialization
    h_input = new float[N];
    h_output = new int[N];
    std::fill_n(h_input, N, 1.0f);

#ifdef ENABLE_VERIFICATION
    // Initialize data for verification
    int* h_output_cpu = new int[N];
    std::fill_n(h_output_cpu, N, 0);
    std::cout << "[CPU] Calculating CPU reference results..." << std::endl;
    WarpBallotCPUReference(h_input, h_output_cpu, N);
    std::cout << "[CPU] CPU reference results calculated" << std::endl;
#endif

    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, N * sizeof(int)));

    // Configure kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Warm up
    std::cout << "[Warmup] Running warmup iteration..." << std::endl;
    warpBallotKernel<<<numBlocks, blockSize>>>(d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef ENABLE_VERIFICATION
    // Verify results
    std::cout << "[Verify] Verifying warp ballot..." << std::endl;
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    Verify<int> verifier(h_output, h_output_cpu, N);
    if (!verifier.VerifyResults()) {
        state.SkipWithError("Result verification failed");
        delete[] h_input;
        delete[] h_output;
        delete[] h_output_cpu;
        return;
    }
    std::cout << "[Verify] Warp ballot verified" << std::endl;
#endif
    delete[] h_input;
    delete[] h_output;
#ifdef ENABLE_VERIFICATION
    delete[] h_output_cpu;
#endif

    // Benchmark iterations
    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    CUDAEventTimer timer;

    for (auto _ : state) {
        timer.Start();
        warpBallotKernel<<<numBlocks, blockSize>>>(d_input, d_output, N);
        timer.Stop();

        metrics.kernel_time = timer.ElapsedMillis();
        metrics.bandwidth = (metrics.size_kb * 1024) / (metrics.kernel_time * 1e-3) / 1e9;
        metrics.gflops = (2.0 * N) / (metrics.kernel_time * 1e-3) / 1e9;

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
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CleanupCUDA();
    std::cout << "[Completed] Warp Ballot benchmark" << std::endl;
}

// Cooperative groups performance test
static void BM_CooperativeGroups(benchmark::State& state) {
    auto start_total = std::chrono::high_resolution_clock::now();

    const int N = state.range(0);
    std::cout << "\n[Starting] Cooperative Groups benchmark [size: " << N << "]" << std::endl;

    // Initialize CUDA and metrics
    InitCUDA();
    KernelMetrics metrics;
    metrics.size_kb = N * sizeof(float) / 1024.0;

    // Allocate and initialize data
    float *h_input = nullptr, *h_output = nullptr;
    float* d_data = nullptr;
    // Memory allocation and data initialization
    h_input = new float[N];
    h_output = new float[N];
    std::fill_n(h_input, N, 1.0f);

#ifdef ENABLE_VERIFICATION
    // Initialize data for verification
    float* h_output_cpu = new float[N];
    std::fill_n(h_output_cpu, N, 1.0f);
    std::cout << "[CPU] Calculating CPU reference results..." << std::endl;
    CooperativeGroupsCPUReference(h_output_cpu, N);
    std::cout << "[CPU] CPU reference results calculated" << std::endl;
#endif

    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Check cooperative launch support
    int deviceId;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    int supportCooperativeLaunch;
    CUDA_CHECK(
        cudaDeviceGetAttribute(&supportCooperativeLaunch, cudaDevAttrCooperativeLaunch, deviceId));

    if (!supportCooperativeLaunch) {
        state.SkipWithError("Device does not support cooperative launch");
        return;
    }

    // Configure kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Warm up
    std::cout << "[Warmup] Running warmup iteration..." << std::endl;
    cooperativeKernel<<<numBlocks, blockSize>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef ENABLE_VERIFICATION
    // Verify results
    std::cout << "[Verify] Verifying cooperative groups..." << std::endl;
    CUDA_CHECK(cudaMemcpy(h_output, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));
    Verify<float> verifier(h_output, h_output_cpu, N);
    if (!verifier.VerifyResults()) {
        state.SkipWithError("Result verification failed");
        delete[] h_input;
        delete[] h_output;
        delete[] h_output_cpu;
        return;
    }
    std::cout << "[Verify] Cooperative groups verified" << std::endl;
#endif
    delete[] h_input;
    delete[] h_output;
#ifdef ENABLE_VERIFICATION
    delete[] h_output_cpu;
#endif

    // Benchmark iterations
    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    CUDAEventTimer timer;

    for (auto _ : state) {
        timer.Start();
        cooperativeKernel<<<numBlocks, blockSize>>>(d_data, N);
        timer.Stop();

        metrics.kernel_time = timer.ElapsedMillis();
        metrics.bandwidth = (metrics.size_kb * 1024) / (metrics.kernel_time * 1e-3) / 1e9;
        metrics.gflops = (2.0 * N) / (metrics.kernel_time * 1e-3) / 1e9;

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
    std::cout << "[Completed] Cooperative Groups benchmark" << std::endl;
}

// Warp reduce benchmark
static void BM_WarpReduce(benchmark::State& state) {
    auto start_total = std::chrono::high_resolution_clock::now();

    const int N = state.range(0);
    std::cout << "\n[Starting] Warp Reduce benchmark [size: " << N << "]" << std::endl;

    // Initialize CUDA and metrics
    InitCUDA();
    KernelMetrics metrics;
    metrics.size_kb = N * sizeof(float) * 2 / 1024.0;  // Input + output data

    // Allocate and initialize data
    float *h_input = nullptr, *h_output = nullptr;
    float *d_input = nullptr, *d_output = nullptr;

    // Memory allocation and data initialization
    h_input = new float[N];
    h_output = new float[N];
    std::fill_n(h_input, N, 1.0f);

#ifdef ENABLE_VERIFICATION
    // Initialize data for verification
    float* h_output_cpu = new float[N];
    memset(h_output_cpu, 0, N * sizeof(float));
    std::cout << "[CPU] Calculating CPU reference results..." << std::endl;
    WarpReduceCPUReference(h_input, h_output_cpu, N);
    std::cout << "[CPU] CPU reference results calculated" << std::endl;
#endif

    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, N * sizeof(float)));

    // Configure kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Warm up
    std::cout << "[Warmup] Running warmup iteration..." << std::endl;
    WarpReduceKernel<<<numBlocks, blockSize>>>(d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef ENABLE_VERIFICATION
    // Verify results
    std::cout << "[Verify] Verifying warp reduce..." << std::endl;
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    Verify<float> verifier(h_output, h_output_cpu, N);
    if (!verifier.VerifyResults()) {
        state.SkipWithError("Result verification failed");
        delete[] h_input;
        delete[] h_output;
        delete[] h_output_cpu;
        return;
    }
    std::cout << "[Verify] Warp reduce verified" << std::endl;
#endif
    delete[] h_input;
    delete[] h_output;
#ifdef ENABLE_VERIFICATION
    delete[] h_output_cpu;
#endif

    // Benchmark iterations
    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    CUDAEventTimer timer;

    for (auto _ : state) {
        timer.Start();
        WarpReduceKernel<<<numBlocks, blockSize>>>(d_input, d_output, N);
        timer.Stop();

        metrics.kernel_time = timer.ElapsedMillis();
        metrics.bandwidth = (metrics.size_kb * 1024) / (metrics.kernel_time * 1e-3) / 1e9;
        metrics.gflops = (N * log2(32)) / (metrics.kernel_time * 1e-3) / 1e9;

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
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CleanupCUDA();
    std::cout << "[Completed] Warp Reduce benchmark" << std::endl;
}

// Register warp shuffle benchmark
BENCHMARK(BM_WarpShuffle)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register warp ballot benchmark
BENCHMARK(BM_WarpBallot)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register cooperative groups benchmark
BENCHMARK(BM_CooperativeGroups)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

// Register warp reduce benchmark
BENCHMARK(BM_WarpReduce)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 20)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(2);

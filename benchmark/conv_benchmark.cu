#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <iostream>

#include "benchmark.h"

// Constants
constexpr int BLOCK_SIZE = 16;

// Basic direct convolution implementation
__global__ void ConvDirectKernel(const float* input, const float* kernel, float* output,
                                 int batch_size, int in_channels, int out_channels, int height,
                                 int width, int kernel_size, int padding, int stride) {
    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;

    int n = blockIdx.x;                                       // batch index
    int c = blockIdx.y;                                       // output channel
    int h = (blockDim.y * blockIdx.z + threadIdx.y) / out_w;  // output height
    int w = (blockDim.y * blockIdx.z + threadIdx.y) % out_w;  // output width

    if (n < batch_size && c < out_channels && h < out_h && w < out_w) {
        float sum = 0.0f;

        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int in_h = h * stride - padding + kh;
                    int in_w = w * stride - padding + kw;

                    if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                        float in_val =
                            input[((n * in_channels + ic) * height + in_h) * width + in_w];
                        float kernel_val =
                            kernel[((c * in_channels + ic) * kernel_size + kh) * kernel_size + kw];
                        sum += in_val * kernel_val;
                    }
                }
            }
        }

        output[((n * out_channels + c) * out_h + h) * out_w + w] = sum;
    }
}

// Convolution implementation using shared memory
__global__ void ConvSharedMemKernel(const float* input, const float* kernel, float* output,
                                    int batch_size, int in_channels, int out_channels, int height,
                                    int width, int kernel_size, int padding, int stride) {
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_kernel =
        shared_mem + (BLOCK_SIZE + kernel_size - 1) * (BLOCK_SIZE + kernel_size - 1);

    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;
    int batch_id = blockIdx.z / out_channels;
    int out_channel = blockIdx.z % out_channels;

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ic++) {
        for (int i = ty; i < BLOCK_SIZE + kernel_size - 1; i += BLOCK_SIZE) {
            for (int j = tx; j < BLOCK_SIZE + kernel_size - 1; j += BLOCK_SIZE) {
                int y = by + i - padding;
                int x = bx + j - padding;

                if (y >= 0 && y < height && x >= 0 && x < width) {
                    shared_input[i * (BLOCK_SIZE + kernel_size - 1) + j] =
                        input[((batch_id * in_channels + ic) * height + y) * width + x];
                } else {
                    shared_input[i * (BLOCK_SIZE + kernel_size - 1) + j] = 0.0f;
                }
            }
        }

        if (tx < kernel_size && ty < kernel_size) {
            shared_kernel[ty * kernel_size + tx] =
                kernel[((out_channel * in_channels + ic) * kernel_size + ty) * kernel_size + tx];
        }

        __syncthreads();

        if (tx < BLOCK_SIZE && ty < BLOCK_SIZE) {
            int out_x = bx + tx;
            int out_y = by + ty;

            if (out_x < out_w && out_y < out_h) {
                for (int i = 0; i < kernel_size; i++) {
                    for (int j = 0; j < kernel_size; j++) {
                        sum += shared_input[(ty + i) * (BLOCK_SIZE + kernel_size - 1) + (tx + j)] *
                               shared_kernel[i * kernel_size + j];
                    }
                }
            }
        }

        __syncthreads();
    }

    if (tx < BLOCK_SIZE && ty < BLOCK_SIZE) {
        int out_x = bx + tx;
        int out_y = by + ty;

        if (out_x < out_w && out_y < out_h) {
            output[((batch_id * out_channels + out_channel) * out_h + out_y) * out_w + out_x] = sum;
        }
    }
}

// CPU reference implementation for convolution
void ConvolutionCPUReference(const float* input, const float* kernel, float* output, int batch_size,
                             int in_channels, int out_channels, int height, int width,
                             int kernel_size, int padding, int stride) {
    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;

    // Initialize output to zero
    int out_size = batch_size * out_channels * out_h * out_w;
    std::fill_n(output, out_size, 0.0f);

    // Compute convolution
    for (int n = 0; n < batch_size; n++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    int in_idx =
                                        ((n * in_channels + ic) * height + ih) * width + iw;
                                    int kernel_idx =
                                        ((oc * in_channels + ic) * kernel_size + kh) * kernel_size +
                                        kw;
                                    sum += input[in_idx] * kernel[kernel_idx];
                                }
                            }
                        }
                    }
                    int out_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

// Benchmark class
class BM_Convolution : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        // Set convolution parameters
        batch_size = 1;
        in_channels = 3;
        out_channels = 64;
        height = state.range(0);
        width = state.range(0);
        kernel_size = state.range(1);
        padding = kernel_size / 2;
        stride = 1;

        // Calculate output size
        out_height = (height + 2 * padding - kernel_size) / stride + 1;
        out_width = (width + 2 * padding - kernel_size) / stride + 1;

        // Calculate data size
        size_kb = (batch_size * in_channels * height * width +
                   out_channels * in_channels * kernel_size * kernel_size +
                   batch_size * out_channels * out_height * out_width) *
                  sizeof(float) / 1024.0f;
        metrics.size_kb = size_kb;

        std::cout << "\n[Starting] Convolution benchmark "
                  << "[input: " << height << "x" << width << ", kernel: " << kernel_size << "x"
                  << kernel_size << "]" << std::endl;

        InitCUDA();
        AllocateMemory();
        InitializeData();

#ifdef ENABLE_VERIFICATION
        // Only generate CPU reference output when verification is enabled
        h_output_cpu = new float[batch_size * out_channels * out_height * out_width];
        std::cout << "[CPU] Calculating CPU reference results..." << std::endl;
        ConvolutionCPUReference(h_input, h_kernel, h_output_cpu, batch_size, in_channels,
                                out_channels, height, width, kernel_size, padding, stride);
        std::cout << "[CPU] CPU reference results calculated" << std::endl;
#endif
    }

    void TearDown(const benchmark::State& state) override {
        FreeMemory();
        CleanupCUDA();
#ifdef ENABLE_VERIFICATION
        delete[] h_output_cpu;
#endif
        std::cout << "[Completed] Resources freed" << std::endl;
    }

protected:
    void AllocateMemory() {
        // Host memory
        h_input = new float[batch_size * in_channels * height * width];
        h_kernel = new float[out_channels * in_channels * kernel_size * kernel_size];
        h_output = new float[batch_size * out_channels * out_height * out_width];

        // Device memory
        CUDA_CHECK(cudaMalloc(&d_input, batch_size * in_channels * height * width * sizeof(float)));
        CUDA_CHECK(cudaMalloc(
            &d_kernel, out_channels * in_channels * kernel_size * kernel_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output,
                              batch_size * out_channels * out_height * out_width * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_col, batch_size * in_channels * kernel_size * kernel_size *
                                          out_height * out_width * sizeof(float)));
    }

    void InitializeData() {
        // Initialize input data and convolution kernel
        for (int i = 0; i < batch_size * in_channels * height * width; i++) {
            h_input[i] = rand() / (float)RAND_MAX;
        }
        for (int i = 0; i < out_channels * in_channels * kernel_size * kernel_size; i++) {
            h_kernel[i] = rand() / (float)RAND_MAX;
        }

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_input, h_input,
                              batch_size * in_channels * height * width * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(d_kernel, h_kernel,
                       out_channels * in_channels * kernel_size * kernel_size * sizeof(float),
                       cudaMemcpyHostToDevice));
    }

    void FreeMemory() {
        delete[] h_input;
        delete[] h_kernel;
        delete[] h_output;

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_kernel));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_col));
    }

    // Helper function to verify current implementation
    bool VerifyImplementation(const char* impl_name) {
#ifdef ENABLE_VERIFICATION
        // Copy GPU results back to host
        CUDA_CHECK(cudaMemcpy(h_output, d_output,
                              batch_size * out_channels * out_height * out_width * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Verify results
        std::cout << "[Verify] Verifying " << impl_name << " implementation..." << std::endl;
        int output_size = batch_size * out_channels * out_height * out_width;
        Verify<float> verify(h_output, h_output_cpu, output_size);
        bool passed = verify.VerifyResults(1e-2);
        std::cout << "[Verify] " << impl_name << " implementation "
                  << (passed ? "passed" : "failed") << std::endl;
        return passed;
#else
        return true;
#endif
    }

    int batch_size, in_channels, out_channels;
    int height, width, kernel_size, padding, stride;
    int out_height, out_width;
    float size_kb;

    float *h_input, *h_kernel, *h_output;
    float *d_input, *d_kernel, *d_output, *d_col;
    CUDAEventTimer timer;
    KernelMetrics metrics;
#ifdef ENABLE_VERIFICATION
    float* h_output_cpu;
#endif
};

// Direct convolution benchmark
BENCHMARK_DEFINE_F(BM_Convolution, DirectConv)(benchmark::State& state) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(batch_size, out_channels, (out_height * out_width + BLOCK_SIZE - 1) / BLOCK_SIZE);

    auto start_total = std::chrono::high_resolution_clock::now();

    // Warm up
    std::cout << "[Warmup] Running warmup iteration..." << std::endl;
    ConvDirectKernel<<<grid, block>>>(d_input, d_kernel, d_output, batch_size, in_channels,
                                      out_channels, height, width, kernel_size, padding, stride);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (!VerifyImplementation("Direct Convolution")) {
        state.SkipWithError("Result verification failed");
        return;
    }

    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    for (auto _ : state) {
        timer.Start();
        ConvDirectKernel<<<grid, block>>>(d_input, d_kernel, d_output, batch_size, in_channels,
                                          out_channels, height, width, kernel_size, padding,
                                          stride);
        timer.Stop();

        metrics.kernel_time = timer.ElapsedMillis();
        metrics.bandwidth = (metrics.size_kb * 1024) / (metrics.kernel_time * 1e-3) / 1e9;
        metrics.gflops = (2.0 * batch_size * out_channels * out_height * out_width * in_channels *
                          kernel_size * kernel_size) /
                         (metrics.kernel_time * 1e-3) / 1e9;

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

// Shared memory convolution benchmark
BENCHMARK_DEFINE_F(BM_Convolution, SharedMemConv)(benchmark::State& state) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE,
              batch_size * out_channels);

    auto start_total = std::chrono::high_resolution_clock::now();

    int shared_mem_size = ((BLOCK_SIZE + kernel_size - 1) * (BLOCK_SIZE + kernel_size - 1) +
                           kernel_size * kernel_size) *
                          sizeof(float);

    // Warm up
    std::cout << "[Warmup] Running warmup iteration..." << std::endl;
    ConvSharedMemKernel<<<grid, block, shared_mem_size>>>(d_input, d_kernel, d_output, batch_size,
                                                          in_channels, out_channels, height, width,
                                                          kernel_size, padding, stride);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (!VerifyImplementation("Shared Memory Convolution")) {
        state.SkipWithError("Result verification failed");
        return;
    }

    std::cout << "[Running] Executing benchmark iterations..." << std::endl;
    for (auto _ : state) {
        timer.Start();
        ConvSharedMemKernel<<<grid, block, shared_mem_size>>>(
            d_input, d_kernel, d_output, batch_size, in_channels, out_channels, height, width,
            kernel_size, padding, stride);
        timer.Stop();

        metrics.kernel_time = timer.ElapsedMillis();
        metrics.bandwidth = (metrics.size_kb * 1024) / (metrics.kernel_time * 1e-3) / 1e9;
        metrics.gflops = (2.0 * batch_size * out_channels * out_height * out_width * in_channels *
                          kernel_size * kernel_size) /
                         (metrics.kernel_time * 1e-3) / 1e9;

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

// Register benchmarks
BENCHMARK_REGISTER_F(BM_Convolution, DirectConv)
    ->Args({224, 3})
    ->Args({224, 5})
    ->Args({224, 7})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->Repetitions(2);

// Register shared memory convolution benchmarks
BENCHMARK_REGISTER_F(BM_Convolution, SharedMemConv)
    ->Args({224, 3})
    ->Args({224, 5})
    ->Args({224, 7})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->Repetitions(2);

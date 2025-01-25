#pragma once

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <stdexcept>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t error = call;                                             \
        if (error != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error: ") + \
                                   cudaGetErrorString(error)); \
        }                                                                     \
    } while (0)

// CUDA initialization and cleanup functions
inline void InitCUDA() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA devices found");
    }
    CUDA_CHECK(cudaSetDevice(0));
}

// CUDA cleanup function
inline void CleanupCUDA() {
    // Don't call cudaDeviceReset() as it can cause problems with multiple benchmarks
    CUDA_CHECK(cudaDeviceSynchronize());
}

// CUDA event timer class
class CUDAEventTimer {
public:
    CUDAEventTimer() noexcept {
        if (cudaEventCreate(&start_) != cudaSuccess) {
            start_ = nullptr;
        }
        if (cudaEventCreate(&stop_) != cudaSuccess) {
            stop_ = nullptr;
        }
    }

    ~CUDAEventTimer() noexcept {
        if (start_) cudaEventDestroy(start_);
        if (stop_) cudaEventDestroy(stop_);
    }

    void Start() {
        if (!start_ || !stop_) {
            throw std::runtime_error("CUDA events not properly initialized");
        }
        CUDA_CHECK(cudaEventRecord(start_));
    }

    void Stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }

    float ElapsedMillis() {
        float elapsed;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_, stop_));
        return elapsed;
    }

private:
    cudaEvent_t start_{nullptr};
    cudaEvent_t stop_{nullptr};
};

// Performance metrics struct
struct KernelMetrics {
    double kernel_time = 0.0;    // kernel execution time in ms
    double bandwidth = 0.0;      // memory bandwidth in GB/s
    double gflops = 0.0;        // compute throughput in GFLOPS
    double size_kb = 0.0;       // data size in KB
    double total_time = 0.0;    // total execution time in ms
};

// Version information structure
struct CUDAVersionInfo {
    int driver_version;
    int runtime_version;
    cudaDeviceProp device_prop;

    static CUDAVersionInfo Get() {
        CUDAVersionInfo info;
        CUDA_CHECK(cudaDriverGetVersion(&info.driver_version));
        CUDA_CHECK(cudaRuntimeGetVersion(&info.runtime_version));

        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&info.device_prop, device));
        return info;
    }

    std::string ToString() const {
        char buf[1024];
        snprintf(buf, sizeof(buf),
                 "CUDA Driver Version: %d.%d\n"
                 "CUDA Runtime Version: %d.%d\n"
                 "GPU: %s (SM %d.%d)\n"
                 "Compute Capability: %d.%d\n"
                 "CUDA Cores: %d\n"
                 "Memory: %.1f GB\n"
                 "Memory Bus Width: %d-bit\n"
                 "Memory Clock: %.1f GHz\n"
                 "Memory Bandwidth: %.1f GB/s",
                 driver_version / 1000, (driver_version % 100) / 10, runtime_version / 1000,
                 (runtime_version % 100) / 10, device_prop.name, device_prop.major,
                 device_prop.minor, device_prop.major, device_prop.minor,
                 device_prop.multiProcessorCount * 64,  // Approximate for most architectures
                 device_prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
                 device_prop.memoryBusWidth, device_prop.memoryClockRate * 1e-6,
                 2.0 * device_prop.memoryClockRate * (device_prop.memoryBusWidth / 8) / 1.0e6);
        return std::string(buf);
    }
};

## 测试点

- KernelTime_ms: 内核执行时间
- Bandwidth_GB/s: 内存带宽
- GFLOPS: 浮点运算性能
- Size_KB: 数据大小
- TotalTime_ms: 总时间

---

### 1. **CUDA Runtime API 性能测试**
测试 CUDA Runtime API 的调用开销和性能。

#### 测试点：
- **设备管理**：
  - `cudaSetDevice`：切换设备的开销。
  - `cudaGetDeviceProperties`：获取设备属性的性能。
- **上下文管理**：
  - `cudaDeviceSynchronize`：同步设备的开销。
- **事件管理**：
  - `cudaEventCreate` 和 `cudaEventDestroy`：事件创建和销毁的开销。
  - `cudaEventRecord` 和 `cudaEventSynchronize`：事件记录和同步的开销。

---

### 2. **内存操作性能测试**
测试主机和设备内存的分配、释放、复制和置位操作的性能。

#### 测试点：
- **主机内存**：
  - `cudaMallocHost` 和 `cudaFreeHost`：主机端页锁定内存的分配和释放性能。
- **设备内存**：
  - `cudaMalloc` 和 `cudaFree`：设备内存的分配和释放性能。
- **数据传输**：
  - `cudaMemcpy`：主机到设备、设备到主机、设备到设备的数据传输性能。
  - `cudaMemcpyAsync`：异步数据传输的性能。
- **数据置位**：
  - `cudaMemset`：设备内存置位的性能。
  - `cudaMemsetAsync`：异步设备内存置位的性能。

---

### 3. **CUDA 内核性能测试**
测试 CUDA 内核的执行时间、并行效率和资源利用率。

#### 测试点：
- **简单内核**：
  - 向量加法：测试基本算术操作的性能。
  - 矩阵乘法：测试全局内存访问和计算密集型任务的性能。
  - 归约操作：测试并行规约算法的性能。
- **复杂内核**：
  - 卷积操作：测试共享内存和全局内存的协同性能。
  - 排序算法：测试并行排序算法的性能（如 Bitonic Sort）。
  - 图像处理：测试图像滤波、变换等操作的性能。
- **内核启动开销**：
  - 测试不同线程块大小和网格大小对内核启动开销的影响。
  - 测试动态并行（Dynamic Parallelism）的性能。

---

### 4. **CUDA 流性能测试**
测试 CUDA 流的并发执行和数据传输性能。

#### 测试点：
- **流管理**：
  - `cudaStreamCreate` 和 `cudaStreamDestroy`：流的创建和销毁开销。
  - `cudaStreamSynchronize`：流同步的开销。
- **并发执行**：
  - 测试多个流并发执行内核的性能。
  - 测试流之间的数据传输和计算重叠性能。
- **事件同步**：
  - 测试使用事件同步流的性能。

---

### 5. **统一内存性能测试**
测试 CUDA 统一内存（Unified Memory）的性能。

#### 测试点：
- **内存分配**：
  - 测试 `cudaMallocManaged` 的性能。
- **数据访问**：
  - 测试统一内存的访问性能（主机访问 vs 设备访问）。
  - 测试统一内存的页面迁移性能。
- **并发访问**：
  - 测试多 GPU 对统一内存的并发访问性能。

---

### 6. **CUDA 高级特性性能测试**
测试 CUDA 的高级特性（如 Cooperative Groups、Tensor Cores）的性能。

#### 测试点：
- **Cooperative Groups**：
  - 测试协作组（Cooperative Groups）的性能。
  - 测试协作组在多 GPU 环境下的性能。
- **Tensor Cores**：
  - 测试 Tensor Cores 在矩阵乘法中的性能。
  - 测试混合精度计算的性能。
- **Warp-Level Primitives**：
  - 测试 Warp 级原语（如 `__shfl`、`__ballot`）的性能。

---

### 7. **CUDA 错误处理性能测试**
测试 CUDA 错误处理的性能开销。

#### 测试点：
- **错误检测**：
  - 测试 `cudaGetLastError` 的性能。
- **错误恢复**：
  - 测试错误恢复机制的性能。

---

### 8. **CUDA 与主机交互性能测试**
测试主机端代码与 CUDA 设备交互的性能。

#### 测试点：
- **主机端代码**：
  - 测试主机端代码的性能（如数据预处理、后处理）。
- **设备端代码**：
  - 测试设备端代码的性能（如内核执行、内存访问）。

---

### 9. **CUDA 环境配置性能测试**
测试不同 CUDA 环境配置对性能的影响。

#### 测试点：
- **编译器优化**：
  - 测试不同编译器优化级别（如 `-O0`、`-O2`、`-O3`）对性能的影响。
- **CUDA 版本**：
  - 测试不同 CUDA 版本对性能的影响。
- **硬件配置**：
  - 测试不同 GPU 型号对性能的影响。

---

### 10. **CUDA 性能分析工具集成**
集成 CUDA 性能分析工具（如 NVIDIA Nsight Systems、nvprof）进行性能分析。

#### 测试点：
- **性能分析**：
  - 使用 Nsight Systems 分析内核执行时间、内存带宽等。
  - 使用 nvprof 分析内核性能瓶颈。

---

# CUDA 性能测试用例

## 文件说明

- `benchmark.h`: 基准测试框架的公共头文件
- `basic_benchmark.cu`: 基础性能测试
- `runtime_benchmark.cu`: CUDA Runtime API 性能测试
- `memory_benchmark.cu`: 内存操作性能测试
- `stream_benchmark.cu`: CUDA 流性能测试
- `unified_memory_benchmark.cu`: 统一内存性能测试
- `advanced_benchmark.cu`: 高级特性性能测试
- `error_benchmark.cu`: 错误处理性能测试
- `interaction_benchmark.cu`: 主机设备交互测试
- `config_benchmark.cu`: 环境配置性能测试
- `mm_benchmark.cu`: 矩阵乘法性能测试
- `conv_benchmark.cu`: 卷积性能测试

## 测试用例说明

每个测试文件包含一组相关的性能测试，使用 Google Benchmark 框架实现。

### 测试命名规范
- 前缀 `BM_` 表示基准测试函数
- 使用驼峰命名法
- 名称应反映被测试的功能

### 测试参数设置
- 使用 Range 设置输入大小范围
- 使用 Repetitions 设置重复次数
- 使用 TimeUnit 设置时间单位

### 性能指标
- 执行时间（ns/us/ms）
- 吞吐量（GB/s）
- 操作次数（ops/s）

### 测试用例说明
- 注释为英文
- 测试用例的入口和出口,有打印标志,可以让用户直观地了解当前测试的进度
- 测试用例的执行时间,有两个,一个是整个测试用例的执行时间,一个是测试用例中内核函数的执行时间
- 测试用例的性能参数主要有: KernelTime_ms，Bandwidth_GB/s，GFLOPS，Size_KB，TotalTime

## 添加新的测试

1. 创建新的 .cu 文件
2. 包含必要的头文件：
```cpp
#include "benchmark.h"
#include <cuda_runtime.h>
```

3. 实现测试函数：
```cpp
static void BM_NewTest(benchmark::State& state) {
    // 初始化
    for (auto _ : state) {
        // 测试代码
    }
    // 清理
}
```

4. 注册测试：
```cpp
BENCHMARK(BM_NewTest)
    ->RangeMultiplier(2)
    ->Range(1<<10, 1<<20)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);
```

5. 在 CMakeLists.txt 中添加新文件

## 运行测试

```bash
# 运行所有测试
./run_benchmark

# 显示所有测试
./run_benchmark --benchmark_list_tests

# 运行特定测试
./run_benchmark --benchmark_filter=BM_CudaMemcpy

# 生成 JSON 格式结果
./run_benchmark --benchmark_format=json --benchmark_out=results.json
```

---

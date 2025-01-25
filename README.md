# CUDA 基准测试套件

这个项目提供了一套全面的 CUDA 性能基准测试，用于评估 CUDA 运行时和内存操作的性能。

## 性能指标

每个测试用例都会报告以下指标：

- **KernelTime_ms**: CUDA kernel 执行时间（毫秒）
- **Bandwidth_GB/s**: 内存带宽（GB/s）
- **GFLOPS**: 浮点运算性能（GFLOPS）
- **Size_KB**: 数据大小（KB）
- **TotalTime**: 总执行时间（毫秒）

## 测试用例说明

### 运行时测试 (runtime_benchmark.cu)
- CudaSetDevice: 设备管理性能测试
- CudaGetDeviceProperties: 获取设备属性性能测试
- CudaDeviceSynchronize: 上下文管理性能测试
- CudaEventCreateDestroy: 事件管理性能测试

### 内存测试 (memory_benchmark.cu)
- CudaMallocHost: 主机内存分配/释放性能测试
- CudaMalloc: 设备内存分配/释放性能测试
- CudaMemcpyD2H: 设备到主机内存拷贝性能测试
- CudaMemcpyHtoD: 主机到设备内存拷贝性能测试
- CudaMemset: 设备内存设置性能测试

## 构建和运行

```bash
mkdir build && cd build
cmake ..
make
./cuda_benchmark
```

## 输出说明

每个测试用例都会输出详细的执行信息，包括：
- 测试开始和结束提示
- 详细的性能指标
- 如果出现错误，会显示具体的错误信息

## 注意事项

1. 所有测试用例使用统一的计时方式（CUDAEventTimer）
2. 对于没有特定指标的测试，相应的计数器会显示 0 或 "N/A"
3. 测试结果的可靠性依赖于系统负载状态

## 功能特点

- 基于 Google Benchmark 框架
- 支持多种 CUDA 功能的性能测试
- 自动生成性能报告和可视化图表
- 支持 JSON/CSV 格式的测试结果输出

## 测试内容

1. **Runtime API 测试**
   - 设备管理
   - 上下文管理
   - 事件管理

2. **内存操作测试**
   - 内存分配/释放
   - 内存传输
   - 内存置位

3. **内核执行测试**
   - 向量运算
   - 矩阵运算
   - 归约操作

4. **流和事件测试**
   - 流创建/销毁
   - 流同步
   - 多流并发

5. **统一内存测试**
   - 分配/释放
   - 访问性能
   - 页面迁移

6. **高级特性测试**
   - Warp 级原语
   - 协作组
   - 共享内存

7. **错误处理测试**
   - 错误检测
   - 错误恢复
   - 设备重置

8. **主机设备交互测试**
   - 数据预处理
   - 主机设备同步
   - 内核启动开销

## 环境要求

- CUDA Toolkit 11.0 或更高版本
- CMake 3.14 或更高版本
- C++17 兼容的编译器
- Python 3.6 或更高版本（用于生成报告）
- NVIDIA GPU（计算能力 7.0 或更高）

## 自定义测试

1. 在 `benchmark` 目录下创建新的 .cu 文件
2. 实现测试函数和内核
3. 在 `benchmark/CMakeLists.txt` 中添加新文件
4. 重新构建并运行

示例：
```cpp
static void BM_NewTest(benchmark::State& state) {
    const int N = state.range(0);
    // 初始化
    for (auto _ : state) {
        // 测试代码
    }
    // 清理
}

BENCHMARK(BM_NewTest)
    ->RangeMultiplier(2)
    ->Range(1<<10, 1<<20)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);
```

## 项目结构

```
cuda-benchmark/
├── benchmark/          # 测试源代码
│   ├── basic_benchmark.cu
│   ├── memory_benchmark.cu
│   └── ...
├── scripts/           # 构建和分析脚本
├── third_party/       # 第三方依赖
├── result/            # 测试结果
└── CMakeLists.txt    # CMake 配置
```

## 注意事项

1. 确保 GPU 驱动已正确安装
2. 运行测试前先清理其他 GPU 任务
3. 对于大规模测试，注意 GPU 温度
4. 测试结果可能因硬件配置而异

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

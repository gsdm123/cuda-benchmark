# CUDA 基准测试套件

这个项目提供了一套全面的 CUDA 性能基准测试，用于评估 CUDA 运行时和内存操作的性能。

## 参考项目
nvbench(https://github.com/NVIDIA/nvbench), 这个项目是 NVIDIA 官方的 CUDA 性能基准测试套件，主要用于调优，本项目比 nvbench 小，主要用于测试不同版本编译器对性能的影响。


## 性能指标

每个测试用例都会报告以下指标：

- **KernelTime_ms**: CUDA kernel 执行时间（毫秒）
- **Bandwidth_GB/s**: 内存带宽（GB/s）
- **GFLOPS**: 浮点运算性能（GFLOPS）
- **Size_KB**: 数据大小（KB）
- **TotalTime**: 总执行时间（毫秒）

该项目无法获取编译时间、可知文件大小、ELF 文件大小等编译器的性能参数。Kernel 函数执行时间并不是通过 nvprof 工具获取的，而是通过 CUDAEventTimer 获取的，所以存在一定的误差。
建议该项目与 llvm-test-suite 项目一起使用，在 llvm-test-suite 项目中，我实现了获取编译时间、可知文件大小、ELF 文件大小等编译器的性能参数，同时使用 nvprof 工具获取 Kernel 函数执行时间，获取的执行时间更加准确。两个项目的不同点是，这个项目会把所有源文件链接到一个可执行文件，而 llvm-test-suite 项目会为每个源文件生成一个可执行文件。另一个项目还在开发中，暂时无法使用。

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

### 下载 google test 和 google benchmark
```bash
./scripts/download_deps.sh

```

### 安装 python 依赖库
```bash
python3 -m pip install -r requirements.txt
```
### 运行测试

```bash
./scripts/build.sh
# 指定 cuda toolkit 版本，默认为 12.6
./scripts/build.sh 12.6
# 性能测试时，不会对结果进行验证，如果需要验证，请修改 build.sh 里面的 "cmake .." 为 "cmake .. -DENABLE_VERIFICATION=ON", 该选项默认为关闭状态
```

## 输出说明

测试完成会生成 result目录，里面包含所有测试的详细结果:
- summary.md 所有测试的汇总结果
- report.md 所有测试的详细结果
- report.html 所有测试的详细结果的可视化图表
- benchmark_result.json 所有测试的详细结果的 json 格式

每个测试用例在运行过程中都会打印详细的执行信息，包括：
- 测试开始和结束提示
- 详细的性能指标
- 如果出现错误，会显示具体的错误信息

在 result 目录下，会生成三个文件夹，分别对应 O0, O1, O2 的测试结果:
- O0_${TIMESTAMP}
- O1_${TIMESTAMP}
- O2_${TIMESTAMP}

在 result 目录下，会生成 O0 VS O1 和 O0 VS O2 的报告文件，报告文件的格式为 html 格式，内容包括：
- 所有测试的详细结果
- 目前仅展示 kernel time 和 total time 的对比

## 注意事项

1. 所有测试用例使用统一的计时方式（CUDAEventTimer）
2. 对于没有特定指标的测试，相应的计数器会显示 0 或 "N/A"
3. 测试结果的可靠性依赖于系统负载状态

## 功能特点

- 基于 Google Benchmark 框架
- 支持多种 CUDA 功能的性能测试
- 自动生成性能报告和可视化图表
- 支持 JSON/CSV 格式的测试结果输出
- 由于所有源文件最后会打包成一个可执行文件，所有准确得到编译时间，关于编译时间这类的性能指标，可以参考另一个项目

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

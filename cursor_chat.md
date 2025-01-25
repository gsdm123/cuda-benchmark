# CUDA Benchmark Project Chat History

## Project Overview
这是一个基于 Google Benchmark 的 CUDA 性能测试框架，用于测试和分析 CUDA 程序的各个方面的性能表现。

## Key Requirements
1. 使用 Google Benchmark 框架
2. 支持多种 CUDA 功能的性能测试
3. 自动生成性能报告和可视化图表
4. 支持 JSON 格式的测试结果输出
5. 每个测试用例运行两次
6. 使用 CUDA 事件计时器记录内核执行时间
7. 生成 HTML 和 Markdown 格式的性能报告

## Project Structure
```
.
├── CMakeLists.txt
├── benchmark/
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── benchmark.h
│   ├── basic_benchmark.cu
│   ├── runtime_benchmark.cu
│   ├── memory_benchmark.cu
│   ├── kernel_benchmark.cu
│   ├── stream_benchmark.cu
│   ├── unified_memory_benchmark.cu
│   ├── advanced_benchmark.cu
│   ├── error_benchmark.cu
│   ├── interaction_benchmark.cu
│   └── config_benchmark.cu
├── scripts/
│   ├── build.sh
│   ├── download_deps.sh
│   └── process_benchmark.py
├── third_party/
│   ├── benchmark/
│   └── googletest/
└── result/
    └── {timestamp}/
        ├── benchmark_results.json
        ├── benchmark_plot.png
        ├── report.html
        ├── report.md
        └── summary.md
```

## Key Files Description

### benchmark.h
- CUDA 错误检查宏
- CUDA 初始化和清理函数
- CUDA 事件计时器类
- 性能指标结构体
- 基准测试初始化类

### basic_benchmark.cu
- 向量加法性能测试
- 使用 CUDA 事件计时
- 记录内核执行时间、内存带宽和计算吞吐量

### build.sh
- 创建构建目录
- 编译项目
- 运行基准测试
- 生成性能报告

### process_benchmark.py
- 处理基准测试结果
- 生成性能图表
- 创建 HTML 和 Markdown 报告

## Performance Metrics
1. 内核执行时间 (ms)
2. 内存带宽 (GB/s)
3. 计算吞吐量 (GFLOPS)
4. 数据大小 (KB)
5. 总执行时间 (us)

## Test Configuration
- 测试数据范围：2^8 到 2^10
- 每个测试重复 2 次
- 使用手动计时
- 显示微秒级结果

## Error Handling
- CUDA 错误检查和处理
- 异常捕获和资源清理
- 测试跳过机制

## Report Generation
1. JSON 格式原始数据
2. 性能可视化图表
3. HTML 格式详细报告
4. Markdown 格式报告
5. 测试结果摘要

## Build and Run
```bash
# 下载依赖
./scripts/download_deps.sh

# 构建和运行
./scripts/build.sh
```

## Notes
1. 确保 GPU 驱动已正确安装
2. 运行测试前先清理其他 GPU 任务
3. 对于大规模测试，注意 GPU 温度
4. 测试结果可能因硬件配置而异

## Version Testing Guide

### Prerequisites
1. 安装多个版本的 CUDA Toolkit
2. 安装对应版本的 NVIDIA 驱动
3. 确保系统环境变量正确配置

### Testing Different CUDA Versions
```bash
# Test with CUDA 11.0
./scripts/build.sh 11.0

# Test with CUDA 11.8
./scripts/build.sh 11.8

# Test with CUDA 12.0
./scripts/build.sh 12.0
```

### Version Compatibility
- CUDA 11.0+ 需要 driver >= 450.36.06
- CUDA 11.8+ 需要 driver >= 520.61.05
- CUDA 12.0+ 需要 driver >= 525.60.13

### Driver Installation
```bash
# 下载指定版本驱动
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/525.60.13/NVIDIA-Linux-x86_64-525.60.13.run

# 安装驱动
sudo bash NVIDIA-Linux-x86_64-525.60.13.run

# 验证安装
nvidia-smi
```

### CUDA Toolkit Installation
```bash
# 下载指定版本 CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run

# 安装 CUDA Toolkit
sudo sh cuda_12.0.0_525.60.13_linux.run

# 验证安装
nvcc --version
```

### Test Results Location
测试结果将保存在以下目录结构中：
```
result/
└── {timestamp}_cuda_{version}/
    ├── benchmark_results.json
    ├── benchmark_plot.png
    ├── report.html
    ├── report.md
    └── summary.md
```

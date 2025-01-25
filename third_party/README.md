# 第三方库

本目录包含项目使用的第三方库。

## 库列表

1. Google Benchmark (v1.8.3)
   - 用于性能基准测试
   - 项目地址：https://github.com/google/benchmark

2. Google Test (v1.14.0)
   - 用于单元测试
   - 项目地址：https://github.com/google/googletest

## 更新依赖

使用 download_deps.sh 脚本更新依赖：

```bash
./scripts/download_deps.sh
```

## 版本说明

- Google Benchmark: v1.8.3
  - 改进的时间测量
  - 更好的 CUDA 支持
  - 完整的统计分析

- Google Test: v1.14.0
  - 改进的测试发现
  - 更好的错误报告
  - 现代 C++ 支持

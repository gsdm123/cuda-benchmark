# 代码规范指南

## 1. 通用规则

### 1.1 语言选择
- C++: 用于核心计算和 CUDA 内核实现
- Python: 用于工具脚本和结果分析
- Shell: 用于构建和自动化脚本

### 1.2 文件组织
- 源代码文件放置在 `src/` 目录
- 构建和工具脚本放置在 `scripts/` 目录
- 文档放置在 `doc/` 目录
- 测试和基准测试放置在 `benchmark/` 目录
- 结果输出到 `result/` 目录

### 1.3 命名规范
- 文件名使用小写字母，单词间用下划线分隔
- C++ 类名使用 PascalCase
- 函数和变量名使用 snake_case
- 常量使用大写字母，单词间用下划线分隔

## 2. C++ 代码规则

### 2.1 CUDA 相关
- 设备函数使用 `__device__` 前缀
- 全局函数使用 `__global__` 前缀
- 主机函数使用 `__host__` 前缀
- CUDA 内核函数名以 `Kernel` 结尾

### 2.2 错误处理
- 使用 `checkCudaErrors` 宏检查 CUDA 操作
- 使用异常处理机制报告错误
- 所有 CUDA 函数调用后必须检查错误

### 2.3 性能优化
- 使用 `constexpr` 声明编译时常量
- 优化内存访问模式，避免非合并访问
- 合理使用共享内存减少全局内存访问

## 3. Python 代码规则

### 3.1 代码风格
- 遵循 PEP 8 规范
- 使用 4 空格缩进
- 行长度限制在 120 字符以内

### 3.2 文档规范
- 所有函数都需要 docstring 说明
- 使用 Google 风格的文档字符串
- 关键算法需要添加详细注释

### 3.3 数据处理
- 使用 Pandas 进行数据分析
- 使用 Matplotlib 和 Seaborn 进行可视化
- JSON 格式用于数据存储和交换

## 4. Shell 脚本规则

### 4.1 基本规范
- 使用 `#!/bin/bash` 作为脚本头
- 使用 4 空格缩进
- 变量名使用大写字母

### 4.2 错误处理
- 使用 `set -e` 确保错误时退出
- 检查命令返回值
- 提供清晰的错误信息

### 4.3 路径处理
- 使用 `$(cd "$(dirname "$0")" && pwd)` 获取脚本目录
- 使用双引号包围路径变量
- 使用绝对路径避免相对路径问题

## 5. 版本控制

### 5.1 Git 规范
- 提交信息使用清晰的描述
- 每个提交专注于单一功能或修复
- 保持提交历史整洁

### 5.2 分支管理
- 主分支保持稳定
- 功能开发使用特性分支
- 版本发布使用标签管理

## 6. 测试规范

### 6.1 基准测试
- 使用 Google Benchmark 框架
- 测试结果需要包含性能指标
- 生成可视化报告和分析

### 6.2 结果报告
- 同时生成 Markdown 和 HTML 格式报告
- 包含详细的性能指标和图表
- 保存完整的测试环境信息

## 7. 文档要求

### 7.1 代码文档
- 所有公共 API 需要详细文档
- 复杂算法需要说明实现原理
- 包含使用示例和注意事项

### 7.2 项目文档
- README 文件说明项目概况
- 提供详细的构建和使用说明
- 维护更新日志和版本信息

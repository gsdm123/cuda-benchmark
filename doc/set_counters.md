## 设置统计指标

在 Google Benchmark 中，**自定义统计**功能允许你为测试结果添加自定义的统计信息。这些统计信息可以是任何与测试相关的指标，例如缓存命中率、内存使用量、算法复杂度等。通过自定义统计，你可以更全面地分析测试结果。

Google Benchmark 提供了 `state.counters` 来支持自定义统计。`state.counters` 是一个键值对容器，可以存储自定义的统计指标。

---

### **1. 基本用法**

#### **`state.counters["CounterName"]`**
- **作用**：添加或访问自定义统计指标。
- **类型**：`benchmark::Counter`。
- **支持的操作**：
  - 设置值：`state.counters["CounterName"] = value;`
  - 设置单位：`state.counters["CounterName"].SetUnit(unit);`
  - 设置统计类型：`state.counters["CounterName"].SetFlags(flags);`

#### **统计类型（Flags）**
- `benchmark::Counter::kIsRate`：表示该指标是一个速率（例如，操作数/秒）。
- `benchmark::Counter::kAvgThreads`：表示该指标是每个线程的平均值。
- `benchmark::Counter::kAvgIterations`：表示该指标是每次迭代的平均值。
- `benchmark::Counter::kIsIterationInvariant`：表示该指标在每次迭代中保持不变。

---

### **2. 示例**

#### **示例 1：添加简单的自定义统计**
以下示例展示了如何添加一个简单的自定义统计指标（例如，缓存命中率）：

```cpp
#include <benchmark/benchmark.h>

static void BM_CustomCounter(benchmark::State& state) {
    for (auto _ : state) {
        // 模拟一些工作
        int sum = 0;
        for (int i = 0; i < state.range(0); ++i) {
            sum += i;
        }
        benchmark::DoNotOptimize(sum);

        // 添加自定义统计
        state.counters["CacheHitRate"] = 0.95; // 假设缓存命中率为 95%
    }
}
BENCHMARK(BM_CustomCounter)->Range(1, 1 << 10);

BENCHMARK_MAIN();
```

#### **输出结果**
```
BM_CustomCounter/256        100 ns        100 ns     7000000 CacheHitRate=0.95
BM_CustomCounter/512        200 ns        200 ns     3500000 CacheHitRate=0.95
BM_CustomCounter/1024       400 ns        400 ns     1750000 CacheHitRate=0.95
```

---

#### **示例 2：添加带单位的自定义统计**
以下示例展示了如何添加一个带单位的自定义统计指标（例如，内存使用量，单位为 KB）：

```cpp
#include <benchmark/benchmark.h>

static void BM_CustomCounterWithUnit(benchmark::State& state) {
    for (auto _ : state) {
        // 模拟一些工作
        int sum = 0;
        for (int i = 0; i < state.range(0); ++i) {
            sum += i;
        }
        benchmark::DoNotOptimize(sum);

        // 添加带单位的自定义统计
        state.counters["MemoryUsage"] = 1024; // 假设内存使用量为 1024 KB
        state.counters["MemoryUsage"].SetUnit("KB");
    }
}
BENCHMARK(BM_CustomCounterWithUnit)->Range(1, 1 << 10);

BENCHMARK_MAIN();
```

#### **输出结果**
```
BM_CustomCounterWithUnit/256        100 ns        100 ns     7000000 MemoryUsage=1024 KB
BM_CustomCounterWithUnit/512        200 ns        200 ns     3500000 MemoryUsage=1024 KB
BM_CustomCounterWithUnit/1024       400 ns        400 ns     1750000 MemoryUsage=1024 KB
```

---

#### **示例 3：添加速率统计**
以下示例展示了如何添加一个速率统计指标（例如，操作数/秒）：

```cpp
#include <benchmark/benchmark.h>

static void BM_CustomRateCounter(benchmark::State& state) {
    for (auto _ : state) {
        // 模拟一些工作
        int sum = 0;
        for (int i = 0; i < state.range(0); ++i) {
            sum += i;
        }
        benchmark::DoNotOptimize(sum);

        // 添加速率统计
        state.counters["OpsPerSecond"] = state.range(0) / state.iterations();
        state.counters["OpsPerSecond"].SetUnit("ops/s");
        state.counters["OpsPerSecond"].SetFlags(benchmark::Counter::kIsRate);
    }
}
BENCHMARK(BM_CustomRateCounter)->Range(1, 1 << 10);

BENCHMARK_MAIN();
```

#### **输出结果**
```
BM_CustomRateCounter/256        100 ns        100 ns     7000000 OpsPerSecond=2560000 ops/s
BM_CustomRateCounter/512        200 ns        200 ns     3500000 OpsPerSecond=2560000 ops/s
BM_CustomRateCounter/1024       400 ns        400 ns     1750000 OpsPerSecond=2560000 ops/s
```

---

#### **示例 4：添加线程平均统计**
以下示例展示了如何添加一个线程平均统计指标（例如，每个线程的平均操作数）：

```cpp
#include <benchmark/benchmark.h>

static void BM_CustomThreadAverageCounter(benchmark::State& state) {
    for (auto _ : state) {
        // 模拟一些工作
        int sum = 0;
        for (int i = 0; i < state.range(0); ++i) {
            sum += i;
        }
        benchmark::DoNotOptimize(sum);

        // 添加线程平均统计
        state.counters["OpsPerThread"] = state.range(0) / state.threads();
        state.counters["OpsPerThread"].SetFlags(benchmark::Counter::kAvgThreads);
    }
}
BENCHMARK(BM_CustomThreadAverageCounter)->Range(1, 1 << 10)->Threads(4);

BENCHMARK_MAIN();
```

#### **输出结果**
```
BM_CustomThreadAverageCounter/256/threads:4        100 ns        100 ns     7000000 OpsPerThread=64
BM_CustomThreadAverageCounter/512/threads:4        200 ns        200 ns     3500000 OpsPerThread=128
BM_CustomThreadAverageCounter/1024/threads:4       400 ns        400 ns     1750000 OpsPerThread=256
```

---

### **3. 总结**
- **`state.counters`**：用于添加自定义统计指标。
- **`SetUnit`**：设置统计指标的单位。
- **`SetFlags`**：设置统计指标的类型（例如，速率、线程平均值等）。
- 通过自定义统计，你可以为测试结果添加更多维度的信息，从而更全面地分析性能。

以上示例展示了如何在 Google Benchmark 中使用自定义统计功能。根据你的需求，可以灵活地添加各种统计指标。

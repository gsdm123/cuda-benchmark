## 预运行（Preliminary Runs）

在 Google Benchmark 中，**预运行（Preliminary Runs）** 是测试执行过程中的一个重要步骤，用于动态调整测试参数（如迭代次数），以确保测试结果的准确性和稳定性。以下是对预运行的详细解释，以及一个测试用例会被执行几次的分析。

---

### **1. 预运行的作用**
预运行的主要目的是：
1. **估算执行时间**：通过少量迭代运行，估算测试用例的执行时间。
2. **动态调整迭代次数**：根据估算的执行时间，调整正式运行时的迭代次数，以确保测试结果具有统计意义。
3. **避免过长的测试时间**：如果测试用例执行时间较长，预运行可以减少正式运行的迭代次数，从而缩短总测试时间。

---

### **2. 预运行的执行流程**
Google Benchmark 的测试执行流程通常包括以下几个阶段：

#### **（1）预运行阶段**
- Google Benchmark 会先执行少量迭代（通常是 1 次或几次），以估算测试用例的执行时间。
- 根据预运行的结果，动态调整正式运行的迭代次数。

#### **（2）正式运行阶段**
- 根据预运行的结果，执行多次迭代，以获取稳定的性能数据。
- 正式运行的迭代次数由 Google Benchmark 自动调整，通常目标是使总运行时间达到一个合理的范围（例如几秒钟）。

#### **（3）统计结果**
- 对正式运行的迭代结果进行统计分析，计算平均值、中位数、标准差等性能指标。

---

### **3. 一个测试用例会被执行几次？**
一个测试用例的执行次数取决于以下几个因素：
1. **预运行的迭代次数**：
   - 预运行通常执行 1 次或几次迭代。
   - 预运行的迭代次数是固定的，不会根据测试用例的执行时间调整。

2. **正式运行的迭代次数**：
   - 正式运行的迭代次数由 Google Benchmark 动态调整。
   - 调整的目标是使总运行时间达到一个合理的范围（例如几秒钟）。
   - 如果测试用例的执行时间很短，Google Benchmark 会增加迭代次数；如果执行时间较长，则会减少迭代次数。

3. **重复运行次数**：
   - 如果使用了 `Repetitions` 方法，测试用例会重复运行多次。
   - 每次重复运行都会包括预运行和正式运行。

---

### **4. 示例分析**
假设有一个测试用例 `BM_Example`，其执行流程如下：

#### **（1）预运行**
- Google Benchmark 执行 1 次迭代，估算执行时间为 100 ns。

#### **（2）正式运行**
- Google Benchmark 的目标是使总运行时间达到 1 秒。
- 根据预运行的估算，正式运行的迭代次数为：
  \[
  \text{迭代次数} = \frac{\text{目标总时间}}{\text{每次迭代时间}} = \frac{1\, \text{秒}}{100\, \text{ns}} = 10^7\, \text{次}
  \]
- 因此，正式运行会执行 1000 万次迭代。

#### **（3）重复运行**
- 如果设置了 `Repetitions(3)`，则整个测试流程会重复 3 次。
- 每次重复运行都会包括预运行和正式运行。

#### **总执行次数**
- 预运行：1 次迭代 × 3 次重复 = 3 次迭代。
- 正式运行：1000 万次迭代 × 3 次重复 = 3000 万次迭代。
- 总执行次数：3 + 3000 万 = 3000 万零 3 次迭代。

---

### **5. 如何控制执行次数**
如果你希望手动控制测试用例的执行次数，可以使用以下方法：

#### **（1）固定迭代次数**
使用 `Iterations` 方法固定迭代次数：
```cpp
BENCHMARK(BM_Example)->Iterations(1000);
```

#### **（2）固定运行时间**
使用 `MinTime` 方法设置最小运行时间：
```cpp
BENCHMARK(BM_Example)->MinTime(2.0); // 至少运行 2 秒
```

#### **（3）固定重复次数**
使用 `Repetitions` 方法设置重复运行次数：
```cpp
BENCHMARK(BM_Example)->Repetitions(5); // 重复运行 5 次
```

---

### **6. 总结**
- **预运行**：用于估算执行时间和动态调整迭代次数。
- **正式运行**：根据预运行的结果执行多次迭代，以获取稳定的性能数据。
- **总执行次数**：取决于预运行、正式运行和重复运行的设置。
- **控制执行次数**：可以通过 `Iterations`、`MinTime` 和 `Repetitions` 方法手动控制。

通过理解预运行和测试执行流程，你可以更好地控制 Google Benchmark 的测试行为，并获取准确的性能数据。

# Google Benchmark 参数说明

1. `--benchmark_list_tests={true|false}`
   - **功能**: 列出所有可用的基准测试，而不执行任何基准测试。
   - **参数**: `true` 列出所有基准测试，`false` 不列出。
   - **用法**: 如果你只想查看可用的测试，而不是运行它们，可以使用这个选项。
   - **示例**:
     ```bash
     benchmark_executable --benchmark_list_tests=true
     ```

2. `--benchmark_filter=<regex>`
   - **功能**: 通过正则表达式过滤要运行的基准测试。只运行匹配的测试。
   - **参数**: 传入一个正则表达式来过滤基准测试。
   - **用法**: 如果你只希望运行某些特定名称的基准测试，可以使用此选项。
   - **示例**:
     ```bash
     benchmark_executable --benchmark_filter=MyTest.*
     ```

3. `--benchmark_min_time=<integer>x OR <float>s`
   - **功能**: 设置每个基准测试的最小运行时间。你可以使用整数（代表重复次数）或浮动时间（代表秒）。
   - **参数**: 例如 `10x` 或 `2.5s`。
   - **用法**: 这个选项允许你强制基准测试运行至少指定的时间，而不仅仅是迭代次数。
   - **示例**:
     ```bash
     benchmark_executable --benchmark_min_time=1.5s
     ```

4. `--benchmark_min_warmup_time=<min_warmup_time>`
   - **功能**: 设置基准测试的最小热身时间。热身时间用于准备基准测试环境，确保测试数据在首次运行时不受缓存或其他因素的影响。
   - **参数**: 输入时间，例如 `0.5s`。
   - **用法**: 这有助于减少因首次执行而产生的性能波动。
   - **示例**:
     ```bash
     benchmark_executable --benchmark_min_warmup_time=0.5s
     ```

5. `--benchmark_repetitions=<num_repetitions>`
   - **功能**: 设置基准测试的重复次数。可以帮助提高测试的准确性。
   - **参数**: 输入一个整数，指定重复次数。
   - **用法**: 如果你想确保结果的稳定性，可以增加重复次数。
   - **示例**:
     ```bash
     benchmark_executable --benchmark_repetitions=5
     ```

6. `--benchmark_enable_random_interleaving={true|false}`
   - **功能**: 启用或禁用随机交错执行测试。
   - **参数**: `true` 启用，`false` 禁用。
   - **用法**: 启用后，每个基准测试的执行顺序会随机打乱，这有助于测试不同执行顺序下的性能差异。
   - **示例**:
     ```bash
     benchmark_executable --benchmark_enable_random_interleaving=true
     ```

7. `--benchmark_report_aggregates_only={true|false}`
   - **功能**: 仅报告汇总数据，而不报告每个测试的详细数据。
   - **参数**: `true` 只报告汇总数据，`false` 报告详细数据。
   - **用法**: 如果你只关心整体性能趋势而不是每个单独的基准测试，可以使用此选项。
   - **示例**:
     ```bash
     benchmark_executable --benchmark_report_aggregates_only=true
     ```

8. `--benchmark_display_aggregates_only={true|false}`
   - **功能**: 控制是否仅显示汇总数据（类似于 `--benchmark_report_aggregates_only`），但影响的是显示输出。
   - **参数**: `true` 仅显示汇总结果，`false` 显示所有数据。
   - **用法**: 这个选项决定了显示结果时是否省略详细数据，仅展示汇总信息。
   - **示例**:
     ```bash
     benchmark_executable --benchmark_display_aggregates_only=true
     ```

9. `--benchmark_format=<console|json|csv>`
   - **功能**: 设置输出格式。可以选择 `console`、`json` 或 `csv` 格式。
   - **参数**: `console` 输出到控制台，`json` 输出为 JSON 格式，`csv` 输出为 CSV 格式。
   - **用法**: 根据输出需求选择适当的格式。`json` 或 `csv` 格式通常用于机器处理。
   - **示例**:
     ```bash
     benchmark_executable --benchmark_format=json
     ```

10. `--benchmark_out=<filename>`
    - **功能**: 将基准测试结果输出到指定的文件中。
    - **参数**: 输入文件路径，如 `result.json`。
    - **用法**: 如果你希望将结果保存到文件中，而不是直接显示，可以使用此选项。
    - **示例**:
      ```bash
      benchmark_executable --benchmark_out=result.csv
      ```

11. `--benchmark_out_format=<json|console|csv>`
    - **功能**: 设置输出结果的格式，支持 `json`、`console` 和 `csv` 格式。
    - **参数**: `json`、`console` 或 `csv`。
    - **用法**: 与 `--benchmark_format` 类似，用来控制文件的输出格式。
    - **示例**:
      ```bash
      benchmark_executable --benchmark_out_format=json
      ```

12. `--benchmark_color={auto|true|false}`
    - **功能**: 控制是否启用彩色输出。
    - **参数**: `auto` 自动启用（如果输出到终端），`true` 强制启用，`false` 禁用。
    - **用法**: 如果你的终端支持彩色，可以选择启用彩色输出，增加可读性。
    - **示例**:
      ```bash
      benchmark_executable --benchmark_color=true
      ```

13. `--benchmark_counters_tabular={true|false}`
    - **功能**: 使用表格形式显示计数器数据。
    - **参数**: `true` 启用表格格式，`false` 禁用。
    - **用法**: 如果你想以表格形式查看详细的计数器数据，可以使用此选项。
    - **示例**:
      ```bash
      benchmark_executable --benchmark_counters_tabular=true
      ```

14. `--benchmark_context=<key>=<value>,...`
    - **功能**: 传递额外的上下文信息，可以是多个键值对。可以在基准测试中使用这些上下文信息。
    - **参数**: 键值对，多个键值对用逗号分隔。
    - **用法**: 用于传递自定义的上下文数据，可能对某些特定测试有用。
    - **示例**:
      ```bash
      benchmark_executable --benchmark_context=env=production,platform=linux
      ```

15. `--benchmark_time_unit={ns|us|ms|s}`
    - **功能**: 设置输出结果的时间单位，支持 `ns`（纳秒）、`us`（微秒）、`ms`（毫秒）、`s`（秒）。
    - **参数**: 选择时间单位。
    - **用法**: 如果你需要以特定的时间单位查看结果，可以使用此选项。
    - **示例**:
      ```bash
      benchmark_executable --benchmark_time_unit=ms
      ```

16. `--v=<verbosity>`
    - **功能**: 设置输出的详细程度（日志级别），指定日志的冗余度。
    - **参数**: 输入一个整数，通常从 0 到 3，数字越大，输出越详细。
    - **用法**: 用于调试，调整日志的详细程度。
    - **示例**:
      ```bash
      benchmark_executable --v=2
      ```

这些选项为你提供了灵活的控制，可以根据需求调整基准测试的行为和输出方式。

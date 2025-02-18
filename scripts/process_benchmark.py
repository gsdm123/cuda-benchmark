#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import json
import os
from datetime import datetime
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_html_report(benchmarks, output_dir, total_time_ms, suffix=""):
    """Create HTML format report"""
    html_content = f"""
    <html>
    <head>
        <title>CUDA Benchmark Report{suffix}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .benchmark {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
            .metrics {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
            .metric {{ background: #f5f5f5; padding: 10px; }}
            h1, h2 {{ color: #333; }}
            .plot {{ margin: 20px 0; }}
            .timestamp {{ color: #666; }}
            .total-time {{ background: #e8f5e9; padding: 10px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>CUDA Benchmark Performance Report{suffix}</h1>
        <p class="timestamp">Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="total-time">
            <h2>Total Test Time: {total_time_ms:.2f} ms</h2>
        </div>

        <div class="plot">
            <h2>Performance Overview</h2>
            <img src="kernel_time_plot{suffix}.svg" alt="Kernel Time Plot" style="max-width: 100%;">
            <img src="bandwidth_plot{suffix}.svg" alt="Bandwidth Plot" style="max-width: 100%;">
        </div>

        <h2>Detailed Results</h2>
    """

    # Add benchmark results
    for benchmark in benchmarks:
        name = benchmark['name']
        if benchmark.get('run_type') == 'iteration':
            name += f" (Repetition {benchmark.get('repetition_index', 0)})"
        time = benchmark['real_time']
        size = benchmark.get('Size_KB', 0)
        kernel_time = benchmark.get('KernelTime_ms', 0)
        bandwidth = benchmark.get('Bandwidth_GB/s', 0)
        gflops = benchmark.get('GFLOPS', 0)
        iterations = benchmark.get('iterations', 0)
        total_test_time = time * iterations / 1000.0

        html_content += f"""
        <div class="benchmark">
            <h3>{name}</h3>
            <div class="metrics">
                <div class="metric">
                    <strong>Data Size:</strong> {size:.2f} KB
                </div>
                <div class="metric">
                    <strong>Kernel Time:</strong> {kernel_time:.3f} ms
                </div>
                <div class="metric">
                    <strong>Memory Bandwidth:</strong> {bandwidth:.2f} GB/s
                </div>
                <div class="metric">
                    <strong>Compute Throughput:</strong> {gflops:.2f} GFLOPS
                </div>
                <div class="metric">
                    <strong>Single Run Time:</strong> {time:.3f} us
                </div>
                <div class="metric">
                    <strong>Iterations:</strong> {iterations}
                </div>
                <div class="metric">
                    <strong>Total Test Time:</strong> {total_test_time:.2f} ms
                </div>
            </div>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    with open(f"{output_dir}/report{suffix}.html", 'w') as f:
        f.write(html_content)

import plotly.express as px

def create_performance_plot(benchmarks, output_dir, suffix=""):
    """Create performance visualization"""
    df = pd.DataFrame(benchmarks)

    # Create a bar plot for kernel time
    fig1 = px.bar(df, x='name', y='KernelTime_ms', title='Kernel Execution Time',
                  labels={'KernelTime_ms': 'Time (ms)', 'name': 'Benchmark Name'},
                  text='KernelTime_ms')
    fig1.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig1.update_layout(xaxis_tickangle=-45, width=800, height=400)

    # Create a bar plot for bandwidth
    fig2 = px.bar(df, x='name', y='Bandwidth_GB/s', title='Memory Bandwidth',
                  labels={'Bandwidth_GB/s': 'GB/s', 'name': 'Benchmark Name'},
                  text='Bandwidth_GB/s')
    fig2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig2.update_layout(xaxis_tickangle=-45, width=800, height=400)

    # Save the figures
    fig1.write_image(f"{output_dir}/kernel_time_plot{suffix}.svg")
    fig2.write_image(f"{output_dir}/bandwidth_plot{suffix}.svg")

def process_benchmark_results(json_file, suffix=""):
    """Process benchmark results and generate reports."""
    output_dir = os.path.dirname(json_file)

    # Read benchmark results
    with open(json_file) as f:
        data = json.load(f)

    benchmarks = data.get('benchmarks', [])

    # Get total time
    total_time = 0
    processed_benchmarks = []

    for benchmark in benchmarks:
        if not benchmark.get('error_occurred', False):
            total_time += benchmark.get('TotalTime_ms', 0)
            processed_benchmarks.append(benchmark)

    # Create performance plot
    create_performance_plot(processed_benchmarks, output_dir, suffix)

    # Generate HTML report
    create_html_report(processed_benchmarks, output_dir, total_time, suffix)

    # Create Markdown report
    report = []
    report.append("# CUDA Benchmark Performance Report\n")
    report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Track failed benchmarks
    failed_benchmarks = []
    successful_benchmarks = []

    for benchmark in processed_benchmarks:
        name = benchmark['name']

        # Check if this benchmark had an error
        if benchmark.get('error_occurred', False):
            error_msg = benchmark.get('error_message', 'Unknown error')
            failed_benchmarks.append((name, error_msg))
            continue

        run_type = benchmark.get('run_type', 'N/A')
        repetition_index = benchmark.get('repetition_index', 0)
        kernel_time = benchmark.get('KernelTime_ms', 'N/A')
        bandwidth = benchmark.get('Bandwidth_GB/s', 0)
        gflops = benchmark.get('GFLOPS', 0)
        size_kb = benchmark.get('Size_KB', 0)
        total_test_time = benchmark.get('TotalTime_ms', 0)

        successful_benchmarks.append((name, run_type,repetition_index, size_kb, kernel_time, bandwidth, gflops, total_test_time))

    # First report failed benchmarks
    if failed_benchmarks:
        report.append("\n## Failed Benchmarks\n")
        for name, error in failed_benchmarks:
            report.append(f"### {name}\n")
            report.append(f"Error: {error}\n")

    # Then report successful benchmarks
    for name, run_type, repetition_index, size_kb, kernel_time, bandwidth, gflops, total_test_time in successful_benchmarks:
        print(f"## {name}\n")
        print(f"Run Type: {run_type}")
        print(f"Repetition Index: {repetition_index}")
        print(f"- Data Size: {size_kb:.2f} KB")
        print(f"- Kernel Time: {kernel_time:.6f} ms")
        print(f"- Memory Bandwidth: {bandwidth:.2f} GB/s")
        print(f"- Compute Throughput: {gflops:.2f} GFLOPS")
        print(f"- Total Test Time: {total_test_time:.2f} ms\n")
    
        if run_type == 'iteration':
            report.append(f"## {name} (Repetition {repetition_index})\n")
        else:
            report.append(f"## {name}\n")
        report.append(f"- Data Size: {size_kb:.2f} KB")
        report.append(f"- Kernel Time: {kernel_time:.6f} ms")
        report.append(f"- Memory Bandwidth: {bandwidth:.2f} GB/s")
        report.append(f"- Compute Throughput: {gflops:.2f} GFLOPS")
        report.append(f"- Total Test Time: {total_test_time:.2f} ms\n")

    report.append("\n## Summary\n")
    report.append(f"Total Execution Time: {total_time/1000:.2f} s\n")

    # Write Markdown report
    with open(f"{output_dir}/report.md", 'w') as f:
        f.write('\n'.join(report))

def load_benchmark_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def create_visualization(df, output_dir, suffix=""):
    # Create subplots
    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=('Kernel Time (ms)', 'Bandwidth (GB/s)', 'GFLOPS'),
                        vertical_spacing=0.1)

    # Add traces for each metric
    fig.add_trace(go.Bar(name='Kernel Time', x=df['name'], y=df['KernelTime_ms']), row=1, col=1)
    fig.add_trace(go.Bar(name='Bandwidth', x=df['name'], y=df['Bandwidth_GB/s']), row=2, col=1)
    fig.add_trace(go.Bar(name='GFLOPS', x=df['name'], y=df['GFLOPS']), row=3, col=1)

    # Update layout
    fig.update_layout(
        title=f'CUDA Benchmark Performance{suffix}',
        height=1200,
        showlegend=False
    )

    # Save plot
    fig.write_html(os.path.join(output_dir, f'report{suffix}.html'))
    fig.write_image(os.path.join(output_dir, f'benchmark_plot{suffix}.png'))

def generate_markdown_report(df, total_time, output_dir, suffix=""):
    report = f"""# CUDA Benchmark Report{suffix}

## Summary
- Total Execution Time: {total_time:.2f} ms
- Number of Benchmarks: {len(df)}

## Performance Results

### Kernel Time (ms)
```
{df[['name', 'KernelTime_ms']].to_markdown(index=False)}
```

### Bandwidth (GB/s)
```
{df[['name', 'Bandwidth_GB/s']].to_markdown(index=False)}
```

### GFLOPS
```
{df[['name', 'GFLOPS']].to_markdown(index=False)}
```

### Data Size (KB)
```
{df[['name', 'Size_KB']].to_markdown(index=False)}
```
"""

    with open(os.path.join(output_dir, f'report{suffix}.md'), 'w') as f:
        f.write(report)

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_benchmark.py <benchmark_results.json> [suffix]")
        sys.exit(1)

    suffix = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""
    process_benchmark_results(sys.argv[1], suffix)

if __name__ == '__main__':
    main()

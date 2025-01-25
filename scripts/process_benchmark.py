#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import json
import os
from datetime import datetime

def create_html_report(benchmarks, plot_path, output_dir, total_time_ms):
    """Create HTML format report"""
    html_content = f"""
    <html>
    <head>
        <title>CUDA Benchmark Report</title>
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
        <h1>CUDA Benchmark Performance Report</h1>
        <p class="timestamp">Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="total-time">
            <h2>Total Test Time: {total_time_ms:.2f} ms</h2>
        </div>
        
        <div class="plot">
            <h2>Performance Overview</h2>
            <img src="{os.path.basename(plot_path)}" alt="Performance Plot" style="max-width: 100%;">
        </div>
        
        <h2>Detailed Results</h2>
    """

    # Add benchmark results
    for benchmark in benchmarks:
        name = benchmark['name']
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
    
    with open(f"{output_dir}/report.html", 'w') as f:
        f.write(html_content)

def create_performance_plot(benchmarks, output_dir):
    """Create performance visualization"""
    df = pd.DataFrame(benchmarks)
    
    plt.figure(figsize=(12, 6))
    
    # Create subplot for kernel time
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x='name', y='KernelTime_ms')
    plt.xticks(rotation=45, ha='right')
    plt.title('Kernel Execution Time')
    plt.ylabel('Time (ms)')
    
    # Create subplot for bandwidth
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x='name', y='Bandwidth_GB/s')
    plt.xticks(rotation=45, ha='right')
    plt.title('Memory Bandwidth')
    plt.ylabel('GB/s')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark_plot.png")
    plt.close()

def process_benchmark_results(json_file):
    """Process benchmark results and generate reports"""
    output_dir = os.path.dirname(json_file)
    
    # Read benchmark results
    with open(json_file) as f:
        data = json.load(f)
    
    benchmarks = data['benchmarks']
    
    # Create markdown report
    report = []
    report.append("# CUDA Benchmark Performance Report\n")
    report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    total_time = 0
    for benchmark in benchmarks:
        name = benchmark['name']
        kernel_time = benchmark.get('KernelTime_ms', 'N/A')
        bandwidth = benchmark.get('Bandwidth_GB/s', 0)
        gflops = benchmark.get('GFLOPS', 0)
        size_kb = benchmark.get('Size_KB', 0)
        total_test_time = benchmark.get('TotalTime_ms', 0)
        total_time += total_test_time

        report.append(f"## {name}\n")
        report.append(f"- Data Size: {size_kb:.2f} KB")
        report.append(f"- Kernel Time: {kernel_time:.6f} ms")
        report.append(f"- Memory Bandwidth: {bandwidth:.2f} GB/s")
        report.append(f"- Compute Throughput: {gflops:.2f} GFLOPS")
        report.append(f"- Total Test Time: {total_test_time:.2f} ms\n")

    report.append(f"\n## Summary\n")
    report.append(f"Total Execution Time: {total_time/1000:.2f} s")
    
    # Write report
    with open(f"{output_dir}/report.md", 'w') as f:
        f.write('\n'.join(report))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python process_benchmark.py <benchmark_results.json>")
        sys.exit(1)
    
    process_benchmark_results(sys.argv[1]) 
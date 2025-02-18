#!/usr/bin/env python3

import argparse
import json
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def load_benchmark_result(result_file):
    """Load benchmark result from JSON file"""
    with open(result_file, 'r') as f:
        data = json.load(f)

    processed_data = []
    for bench in data['benchmarks']:
        name = bench['name']
        # Only consider aggregate data
        if bench['run_type'] == 'iteration':
            continue
        if 'mean' in bench['aggregate_name'].lower():
            processed_data.append({
                'name': name,
                'type': 'mean',
                'KernelTime_ms': bench.get('KernelTime_ms', 0),
                # 'Bandwidth_GB/s': bench.get('Bandwidth_GB/s', 0),
                # 'GFLOPS': bench.get('GFLOPS', 0),
                'TotalTime_ms': bench.get('TotalTime_ms', 0)
            })
        elif 'median' in bench['aggregate_name'].lower():
            processed_data.append({
                'name': name,
                'type': 'median',
                'KernelTime_ms': bench.get('KernelTime_ms', 0),
                # 'Bandwidth_GB/s': bench.get('Bandwidth_GB/s', 0),
                # 'GFLOPS': bench.get('GFLOPS', 0),
                'TotalTime_ms': bench.get('TotalTime_ms', 0)
            })
    return processed_data

def calculate_speedup(base_data, opt_data):
    """Calculate performance speedup"""
    base_df = pd.DataFrame(base_data)
    opt_df = pd.DataFrame(opt_data)

    # metrics = ['KernelTime_ms', 'Bandwidth_GB/s', 'GFLOPS']
    metrics = ['KernelTime_ms', 'TotalTime_ms']
    results = {}

    for metric in metrics:
        df = pd.DataFrame()
        for type_ in ['mean', 'median']:
            base_values = base_df[base_df['type'] == type_][metric].values
            opt_values = opt_df[opt_df['type'] == type_][metric].values
            names = base_df[base_df['type'] == type_]['name'].values

            data = []
            for name, base_val, opt_val in zip(names, base_values, opt_values):
                speedup = opt_val / base_val if metric != 'KernelTime_ms' else base_val / opt_val
                data.append({
                    'Benchmark': name,
                    'Type': type_,
                    'Base': base_val,
                    'Optimized': opt_val,
                    'Speedup': speedup
                })
            df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
        results[metric] = df
    return results

def create_comparison_report(base_file, opt_file, output_file):
    base_data = load_benchmark_result(base_file)
    opt_data = load_benchmark_result(opt_file)
    results = calculate_speedup(base_data, opt_data)

    # metrics = ['KernelTime_ms', 'Bandwidth_GB/s', 'GFLOPS']
    metrics = ['KernelTime_ms', 'TotalTime_ms']

    # Generate HTML report
    html_content = """
    <html>
    <head>
        <title>CUDA Benchmark Performance Comparison</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
                max-width: 1800px;
                margin: 40px auto;
            }
            .metric-section {
                margin: 20px 0;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 8px;
                background: white;
            }
            .benchmark-list {
                list-style: none;
                padding: 0;
            }
            .benchmark-header {
                display: flex;
                align-items: center;
                padding: 10px 20px;
                background: #f8f9fa;
                border-radius: 5px;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .benchmark-item {
                margin: 5px 0;
                padding: 10px 20px;
                background: #f5f5f5;
                border-radius: 5px;
                display: flex;
                align-items: center;
            }
            .benchmark-name {
                flex: 0 0 500px;
                font-weight: bold;
                padding-right: 20px;
            }
            .header-name {
                flex: 0 0 500px;
                padding-right: 20px;
            }
            .stat-grid {
                display: flex;
                gap: 20px;
                flex: 1;
                min-width: 800px;
            }
            .stat-item {
                flex: 1;
                min-width: 180px;
                text-align: right;
                font-family: monospace;
                font-size: 14px;
            }
            .stat-value {
                background: white;
                padding: 8px 15px;
                border-radius: 3px;
                display: block;
            }
            .speedup { color: #2196F3; }
            .improvement { color: #4CAF50; }
            .title {
                color: #333;
                padding: 10px 0;
                border-bottom: 2px solid #eee;
                margin-bottom: 20px;
            }
            .type-label {
                color: #666;
                font-style: italic;
                margin: 15px 0;
            }
            .metric-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }
            .avg-speedup {
                background: #E3F2FD;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                color: #1976D2;
            }
            /* Speedup colors */
            .speedup-very-good { background: #4CAF50 !important; color: white !important; }
            .speedup-good { background: #C8E6C9 !important; }
            .speedup-neutral { background: #E3F2FD !important; }
            .speedup-bad { background: #FFCDD2 !important; }
            .speedup-very-bad { background: #F44336 !important; color: white !important; }

            /* Improvement colors */
            .improvement-very-good { background: #4CAF50 !important; color: white !important; }
            .improvement-good { background: #C8E6C9 !important; }
            .improvement-neutral { background: #E3F2FD !important; }
            .improvement-bad { background: #FFCDD2 !important; }
            .improvement-very-bad { background: #F44336 !important; color: white !important; }
        </style>
    </head>
    <body>
        <h1>CUDA Benchmark Performance Comparison</h1>
    """

    for metric in metrics:
        df = results[metric]
        html_content += f"""
        <div class="metric-section">
            <h2 class="title">{metric}</h2>
        """

        for type_ in ['mean', 'median']:
            df_type = df[df['Type'] == type_]
            avg_speedup = df_type['Speedup'].mean()

            html_content += f"""
            <div class="metric-header">
                <h3 class="type-label">{type_.capitalize()} Statistics</h3>
                <div class="avg-speedup">Average Speedup: {avg_speedup:.6f}x</div>
            </div>
            <div class="benchmark-header">
                <div class="header-name">Benchmark</div>
                <div class="stat-grid">
                    <div class="stat-item">Base</div>
                    <div class="stat-item">Optimized</div>
                    <div class="stat-item">Speedup</div>
                    <div class="stat-item">Improvement</div>
                </div>
            </div>
            <ul class="benchmark-list">
            """

            def get_speedup_class(speedup):
                if speedup >= 1.5:
                    return "speedup-very-good"
                elif speedup >= 1.1:
                    return "speedup-good"
                elif speedup > 0.9:
                    return "speedup-neutral"
                elif speedup > 0.5:
                    return "speedup-bad"
                else:
                    return "speedup-very-bad"

            def get_improvement_class(improvement):
                if improvement >= 50:
                    return "improvement-very-good"
                elif improvement >= 10:
                    return "improvement-good"
                elif improvement > -10:
                    return "improvement-neutral"
                elif improvement > -50:
                    return "improvement-bad"
                else:
                    return "improvement-very-bad"

            for _, row in df_type.iterrows():
                speedup = row['Speedup']
                improvement = (speedup - 1) * 100
                speedup_class = get_speedup_class(speedup)
                improvement_class = get_improvement_class(improvement)

                html_content += f"""
                <li class="benchmark-item">
                    <div class="benchmark-name">{row['Benchmark']}</div>
                    <div class="stat-grid">
                        <div class="stat-item">
                            <span class="stat-value">{row['Base']:.6f}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value">{row['Optimized']:.6f}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value speedup {speedup_class}">{speedup:.6f}x</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value improvement {improvement_class}">
                                {'+' if improvement > 0 else ''}{improvement:.1f}%
                            </span>
                        </div>
                    </div>
                </li>
                """

            html_content += "</ul>"
        html_content += "</div>"

    html_content += """
    </body>
    </html>
    """

    # Save HTML report
    with open(output_file, 'w') as f:
        f.write(html_content)

    # Generate Markdown summary
    summary_file = output_file.replace('.html', '_summary.md')
    with open(summary_file, 'w') as f:
        f.write("# CUDA Benchmark Performance Comparison Summary\n\n")

        for metric in metrics:
            df = results[metric]
            f.write(f"\n## {metric}\n\n")

            for type_ in ['mean', 'median']:
                df_type = df[df['Type'] == type_]
                avg_speedup = df_type['Speedup'].mean()
                f.write(f"### {type_.capitalize()} Statistics\n")
                f.write(f"- Average speedup: {avg_speedup:.6f}x\n\n")

                # Performance data table
                summary_data = df_type[['Benchmark', 'Base', 'Optimized', 'Speedup']]
                f.write(summary_data.to_markdown(index=False))
                f.write("\n\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, help='Base benchmark result file')
    parser.add_argument('--optimized', required=True, help='Optimized benchmark result file')
    parser.add_argument('--output', required=True, help='Output HTML file')
    args = parser.parse_args()

    try:
        create_comparison_report(args.base, args.optimized, args.output)
        print(f"Comparison report generated: {args.output}")
        print(f"Summary report generated: {args.output.replace('.html', '_summary.md')}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == '__main__':
    main()

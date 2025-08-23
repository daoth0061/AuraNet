"""
AuraNet Log Analysis Script

This script helps to analyze log files from AuraNet training and extract metrics.
It can also generate plots of metrics over time.

Usage:
    python analyze_logs.py --log_file logs/auranet_20240821_163505.log --metric accuracy
    python analyze_logs.py --log_file logs/auranet_20240821_163505.log --plot --metrics accuracy,loss
"""

import re
import os
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def extract_metrics(log_file, metrics=None):
    """
    Extract metrics from a log file
    
    Args:
        log_file: Path to log file
        metrics: List of metrics to extract (if None, extract all)
    
    Returns:
        DataFrame with metrics
    """
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found")
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract all validation metrics sections
    validation_sections = re.findall(
        r"VALIDATION Metrics - Epoch (\d+):\n==+\n(.*?)\n==+", 
        content, 
        re.DOTALL
    )
    
    data = []
    for epoch, section in validation_sections:
        entry = {'epoch': int(epoch)}
        
        # Extract all metrics from the section
        metrics_matches = re.findall(r"(\w+):\s+([\d\.]+)", section)
        for metric_name, value in metrics_matches:
            entry[metric_name] = float(value)
        
        data.append(entry)
    
    # Extract training loss
    training_losses = re.findall(r"Epoch (\d+) \| Batch \d+/\d+ \| Loss: ([\d\.]+)", content)
    loss_data = {}
    for epoch, loss in training_losses:
        epoch = int(epoch)
        if epoch not in loss_data:
            loss_data[epoch] = []
        loss_data[epoch].append(float(loss))
    
    # Add average training loss to data
    for entry in data:
        epoch = entry['epoch']
        if epoch in loss_data and loss_data[epoch]:
            entry['train_loss'] = sum(loss_data[epoch]) / len(loss_data[epoch])
    
    df = pd.DataFrame(data)
    
    # Filter for specific metrics if requested
    if metrics:
        metrics = ['epoch'] + metrics
        df = df[metrics]
    
    return df


def plot_metrics(df, metrics=None, output_dir=None):
    """
    Plot metrics from a DataFrame
    
    Args:
        df: DataFrame with metrics
        metrics: List of metrics to plot (if None, plot all except epoch)
        output_dir: Directory to save plots (if None, display only)
    """
    if df is None or df.empty:
        print("Error: No data to plot")
        return
    
    if metrics is None:
        metrics = [col for col in df.columns if col != 'epoch']
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot each metric
    for metric in metrics:
        if metric == 'epoch' or metric not in df.columns:
            continue
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df[metric], marker='o', linestyle='-')
        plt.title(f'{metric.replace("_", " ").title()} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace("_", " ").title())
        plt.grid(True)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'{metric}_trend.png'))
            plt.close()
        else:
            plt.show()


def export_to_json(df, output_file):
    """
    Export DataFrame to JSON file
    
    Args:
        df: DataFrame to export
        output_file: Path to output JSON file
    """
    if df is None or df.empty:
        print("Error: No data to export")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert DataFrame to dict and write to JSON
    data = df.to_dict(orient='records')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Data exported to {output_file}")


def analyze_memory_usage(log_file):
    """
    Analyze memory usage from log file
    
    Args:
        log_file: Path to log file
    
    Returns:
        DataFrame with memory usage
    """
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found")
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract memory usage
    memory_matches = re.findall(r"Memory: ([\d\.]+)GB", content)
    memory_usage = [float(m) for m in memory_matches]
    
    if not memory_usage:
        print("No memory usage information found in log file")
        return None
    
    # Create DataFrame
    df = pd.DataFrame({
        'index': range(len(memory_usage)),
        'memory_gb': memory_usage
    })
    
    # Print summary statistics
    print("\nMemory Usage Statistics (GB):")
    print(f"Min: {df['memory_gb'].min():.2f}")
    print(f"Max: {df['memory_gb'].max():.2f}")
    print(f"Mean: {df['memory_gb'].mean():.2f}")
    print(f"Std: {df['memory_gb'].std():.2f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Analyze AuraNet log files')
    parser.add_argument('--log_file', required=True, help='Path to log file')
    parser.add_argument('--metrics', help='Comma-separated list of metrics to extract')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--output_dir', help='Directory to save plots and JSON output')
    parser.add_argument('--memory', action='store_true', help='Analyze memory usage')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'log_analysis_{timestamp}'
    
    # Extract metrics from log file
    metrics = args.metrics.split(',') if args.metrics else None
    df = extract_metrics(args.log_file, metrics)
    
    if df is not None:
        # Print metrics
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\nExtracted Metrics:")
        print(df)
        
        # Export to JSON
        export_to_json(df, os.path.join(output_dir, 'metrics.json'))
        
        # Generate plots
        if args.plot:
            plot_metrics(df, metrics, output_dir)
    
    # Analyze memory usage if requested
    if args.memory:
        memory_df = analyze_memory_usage(args.log_file)
        if memory_df is not None and args.plot:
            plt.figure(figsize=(10, 6))
            plt.plot(memory_df['index'], memory_df['memory_gb'], marker='.', linestyle='-')
            plt.title('Memory Usage over Time')
            plt.xlabel('Log Entry')
            plt.ylabel('Memory Usage (GB)')
            plt.grid(True)
            
            plt.savefig(os.path.join(output_dir, 'memory_usage.png'))
            if not args.output_dir:
                plt.show()


if __name__ == '__main__':
    main()

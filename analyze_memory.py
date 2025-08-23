"""
AuraNet Memory Usage Analysis Script

This script analyzes memory usage patterns from AuraNet log files 
and provides insights into memory consumption trends.

Usage:
    python analyze_memory.py --log_file logs/auranet_20240821_163505.log
    python analyze_memory.py --log_file logs/auranet_20240821_163505.log --plot --detailed
"""

import re
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def extract_memory_info(log_file):
    """
    Extract memory usage information from a log file
    
    Args:
        log_file: Path to the log file
    
    Returns:
        DataFrame with memory usage information
    """
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found")
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract memory usage with timestamps and context
    memory_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ([^-]+) - ([^-]+) - (.+?Memory: ([\d\.]+)GB.+?)"
    memory_matches = re.findall(memory_pattern, content)
    
    if not memory_matches:
        # Try a simpler pattern without the full context
        memory_pattern = r"Memory: ([\d\.]+)GB"
        simple_matches = re.findall(memory_pattern, content)
        
        if not simple_matches:
            print("No memory usage information found in log file")
            return None
        
        # Create basic DataFrame without timestamps
        data = []
        for i, mem in enumerate(simple_matches):
            data.append({
                'timestamp': i,
                'memory_gb': float(mem),
                'module': 'unknown',
                'level': 'unknown',
                'context': 'Memory usage only'
            })
    else:
        # Create detailed DataFrame with timestamps and context
        data = []
        for timestamp, module, level, context, memory in memory_matches:
            data.append({
                'timestamp': datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f'),
                'memory_gb': float(memory),
                'module': module.strip(),
                'level': level.strip(),
                'context': context.strip()
            })
    
    df = pd.DataFrame(data)
    return df


def analyze_memory_trends(df):
    """
    Analyze memory usage trends
    
    Args:
        df: DataFrame with memory usage information
    
    Returns:
        Dictionary with analysis results
    """
    if df is None or df.empty:
        return None
    
    results = {
        'min_memory': df['memory_gb'].min(),
        'max_memory': df['memory_gb'].max(),
        'avg_memory': df['memory_gb'].mean(),
        'std_memory': df['memory_gb'].std(),
        'memory_range': df['memory_gb'].max() - df['memory_gb'].min(),
        'num_samples': len(df)
    }
    
    # Calculate growth rate if we have timestamp information
    if isinstance(df['timestamp'].iloc[0], datetime):
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Calculate time elapsed in seconds
        time_elapsed = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
        results['time_elapsed_seconds'] = time_elapsed
        
        # Calculate memory growth
        memory_growth = df['memory_gb'].iloc[-1] - df['memory_gb'].iloc[0]
        results['memory_growth_gb'] = memory_growth
        
        if time_elapsed > 0:
            results['memory_growth_rate_gb_per_hour'] = memory_growth / (time_elapsed / 3600)
    
    # Identify potential memory leaks (continuous growth patterns)
    # A simple heuristic: if memory consistently increases over time
    if len(df) >= 10:
        # Check if memory consistently increases
        is_increasing = True
        for i in range(1, len(df)):
            if df['memory_gb'].iloc[i] < df['memory_gb'].iloc[i-1]:
                is_increasing = False
                break
        
        results['potential_memory_leak'] = is_increasing and results['memory_range'] > 1.0
    
    # Find memory spikes
    if len(df) >= 5:
        rolling_mean = df['memory_gb'].rolling(window=5, min_periods=1).mean()
        spikes = df[df['memory_gb'] > rolling_mean * 1.5]
        results['memory_spikes_count'] = len(spikes)
        
        if not spikes.empty:
            results['memory_spikes'] = spikes[['timestamp', 'memory_gb', 'context']].to_dict('records')
    
    return results


def plot_memory_usage(df, output_dir=None, detailed=False):
    """
    Plot memory usage over time
    
    Args:
        df: DataFrame with memory usage information
        output_dir: Directory to save plots (if None, display only)
        detailed: Whether to create detailed plots
    """
    if df is None or df.empty:
        print("Error: No data to plot")
        return
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Basic memory usage plot
    plt.figure(figsize=(12, 6))
    
    if isinstance(df['timestamp'].iloc[0], datetime):
        # Plot with actual timestamps
        plt.plot(df['timestamp'], df['memory_gb'], marker='.', linestyle='-')
        plt.gcf().autofmt_xdate()  # Rotate date labels
    else:
        # Plot with indices
        plt.plot(df.index, df['memory_gb'], marker='.', linestyle='-')
    
    plt.title('GPU Memory Usage Over Time')
    plt.xlabel('Time')
    plt.ylabel('Memory Usage (GB)')
    plt.grid(True)
    
    # Add horizontal line for average memory usage
    avg_memory = df['memory_gb'].mean()
    plt.axhline(y=avg_memory, color='r', linestyle='--', label=f'Avg: {avg_memory:.2f} GB')
    
    # Add horizontal line for max memory usage
    max_memory = df['memory_gb'].max()
    plt.axhline(y=max_memory, color='g', linestyle=':', label=f'Max: {max_memory:.2f} GB')
    
    plt.legend()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'memory_usage.png'))
        plt.close()
    else:
        plt.show()
    
    if detailed and 'module' in df.columns and df['module'].nunique() > 1:
        # Plot memory usage by module
        plt.figure(figsize=(12, 8))
        modules = df['module'].unique()
        
        for module in modules:
            module_df = df[df['module'] == module]
            if isinstance(module_df['timestamp'].iloc[0], datetime):
                plt.plot(module_df['timestamp'], module_df['memory_gb'], marker='.', label=module)
            else:
                plt.plot(module_df.index, module_df['memory_gb'], marker='.', label=module)
        
        plt.title('GPU Memory Usage by Module')
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (GB)')
        plt.grid(True)
        plt.legend()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'memory_usage_by_module.png'))
            plt.close()
        else:
            plt.show()
    
    # Histogram of memory usage
    plt.figure(figsize=(10, 6))
    plt.hist(df['memory_gb'], bins=20, alpha=0.7, color='blue')
    plt.title('Distribution of GPU Memory Usage')
    plt.xlabel('Memory Usage (GB)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'memory_usage_histogram.png'))
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze AuraNet memory usage')
    parser.add_argument('--log_file', required=True, help='Path to log file')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--detailed', action='store_true', help='Generate detailed analysis')
    parser.add_argument('--output_dir', help='Directory to save plots and analysis')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'memory_analysis_{timestamp}'
    
    # Extract memory information from log file
    df = extract_memory_info(args.log_file)
    
    if df is not None:
        # Print basic statistics
        print("\nMemory Usage Statistics (GB):")
        print(f"Min: {df['memory_gb'].min():.2f}")
        print(f"Max: {df['memory_gb'].max():.2f}")
        print(f"Mean: {df['memory_gb'].mean():.2f}")
        print(f"Std: {df['memory_gb'].std():.2f}")
        print(f"Range: {df['memory_gb'].max() - df['memory_gb'].min():.2f}")
        print(f"Samples: {len(df)}")
        
        # Analyze memory trends
        analysis = analyze_memory_trends(df)
        if analysis:
            print("\nMemory Trend Analysis:")
            for key, value in analysis.items():
                if key != 'memory_spikes':
                    print(f"{key}: {value}")
            
            if 'potential_memory_leak' in analysis and analysis['potential_memory_leak']:
                print("\nWARNING: Potential memory leak detected!")
                print("Memory usage consistently increases over time.")
            
            if 'memory_spikes_count' in analysis and analysis['memory_spikes_count'] > 0:
                print(f"\nDetected {analysis['memory_spikes_count']} memory spikes!")
                if args.detailed and 'memory_spikes' in analysis:
                    print("\nMemory Spikes Details:")
                    for i, spike in enumerate(analysis['memory_spikes'][:5], 1):  # Show only first 5 spikes
                        timestamp = spike['timestamp']
                        if isinstance(timestamp, datetime):
                            timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                        print(f"Spike {i}: {timestamp} - {spike['memory_gb']:.2f}GB")
                        if 'context' in spike:
                            print(f"  Context: {spike['context'][:100]}...")
        
        # Generate plots if requested
        if args.plot:
            plot_memory_usage(df, output_dir, args.detailed)
            print(f"\nPlots saved to {output_dir}/")


if __name__ == '__main__':
    main()

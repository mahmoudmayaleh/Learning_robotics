#!/usr/bin/env python3
"""Generate training curves and evaluation plots from CSV logs."""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def smooth(data, window=10):
    return data.rolling(window=window, min_periods=1).mean()


def plot_training_curves(log_files, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for log_file, label in zip(log_files, labels):
        df = pd.read_csv(log_file)
        
        axes[0, 0].plot(df['episode'], smooth(df['return']), label=label)
        axes[0, 1].plot(df['episode'], smooth(df['success'].astype(float) * 100), label=label)
        axes[1, 0].plot(df['episode'], smooth(df['collisions']), label=label)
        axes[1, 1].plot(df['episode'], smooth(df['steps']), label=label)
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].set_title('Episode Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Success Rate (%)')
    axes[0, 1].set_title('Success Rate (smoothed)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Collisions')
    axes[1, 0].set_title('Collisions per Episode')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Steps')
    axes[1, 1].set_title('Episode Length')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Saved training_curves.png")


def plot_comparison_bar(log_files, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for log_file, label in zip(log_files, labels):
        df = pd.read_csv(log_file)
        last_100 = df.tail(100)
        results.append({
            'maze': label,
            'avg_return': last_100['return'].mean(),
            'success_rate': last_100['success'].mean() * 100,
            'avg_collisions': last_100['collisions'].mean(),
        })
    
    results_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].bar(results_df['maze'], results_df['avg_return'], color=['#4CAF50', '#2196F3', '#FF5722'])
    axes[0].set_ylabel('Average Return')
    axes[0].set_title('Final Performance: Return')
    
    axes[1].bar(results_df['maze'], results_df['success_rate'], color=['#4CAF50', '#2196F3', '#FF5722'])
    axes[1].set_ylabel('Success Rate (%)')
    axes[1].set_title('Final Performance: Success')
    
    axes[2].bar(results_df['maze'], results_df['avg_collisions'], color=['#4CAF50', '#2196F3', '#FF5722'])
    axes[2].set_ylabel('Avg Collisions')
    axes[2].set_title('Final Performance: Collisions')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'maze_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved maze_comparison.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', nargs='+', required=True, help='CSV log files')
    parser.add_argument('--labels', nargs='+', required=True, help='Labels for each log')
    parser.add_argument('--output', default='plots', help='Output directory')
    args = parser.parse_args()
    
    plot_training_curves(args.logs, args.labels, args.output)
    plot_comparison_bar(args.logs, args.labels, args.output)


if __name__ == '__main__':
    main()

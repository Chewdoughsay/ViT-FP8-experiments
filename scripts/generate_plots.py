"""
Generate publication-quality plots from training metrics.

This script creates visualizations of ViT training experiments including:
- Training/validation loss curves
- Training/validation accuracy curves
- Learning rate schedules
- Multi-experiment comparisons
- Hardware monitoring plots (CPU, memory, GPU)
- Convergence analysis

All plots are saved as high-resolution PNG files suitable for papers,
presentations, and reports. The script uses matplotlib with a clean,
professional style optimized for scientific publications.

Usage:
    Plot single experiment:
        $ python scripts/generate_plots.py --experiment results/BaseFP32

    Compare multiple experiments:
        $ python scripts/generate_plots.py --compare results/BaseFP32 results/BaseFP16 results/AugmFP16

    Plot all experiments:
        $ python scripts/generate_plots.py --all

    Custom output directory:
        $ python scripts/generate_plots.py --experiment results/BaseFP32 --output plots/

Output:
    For single experiment, generates:
    - {experiment}/plots/training_curves.png: Loss and accuracy over epochs
    - {experiment}/plots/learning_rate.png: LR schedule
    - {experiment}/plots/hardware_stats.png: CPU, memory, thermal (if available)

    For comparisons, generates:
    - results/comparison_plots/accuracy_comparison.png
    - results/comparison_plots/loss_comparison.png

Example:
    >>> from generate_plots import plot_training_curves, load_metrics
    >>>
    >>> # Load and plot single experiment
    >>> metrics = load_metrics('results/BaseFP32/metrics/final_metrics.json')
    >>> plot_training_curves(metrics, 'BaseFP32', save_dir='plots/')
    ✓ Saved: plots/BaseFP32_training_curves.png
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

# Import matplotlib with non-interactive backend for server use
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'figure.dpi': 100,
    'savefig.dpi': 300,  # High resolution for papers
    'savefig.bbox': 'tight',
    'figure.autolayout': True
})


def load_metrics(metrics_path):
    """
    Load training metrics from JSON file.

    Args:
        metrics_path (str or Path): Path to final_metrics.json

    Returns:
        dict: Metrics dictionary with time-series data

    Raises:
        FileNotFoundError: If metrics file doesn't exist

    Example:
        >>> metrics = load_metrics('results/BaseFP32/metrics/final_metrics.json')
        >>> print(f"Epochs: {len(metrics['train_loss'])}")
    """
    metrics_path = Path(metrics_path)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    return metrics


def load_hardware_stats(stats_path):
    """
    Load hardware monitoring statistics from JSON file.

    Args:
        stats_path (str or Path): Path to hardware_stats.json

    Returns:
        dict: Hardware stats dictionary, or None if file doesn't exist

    Example:
        >>> hw_stats = load_hardware_stats('results/BaseFP32/metrics/hardware_stats.json')
        >>> if hw_stats:
        ...     print(f"Max CPU: {hw_stats['summary']['max_cpu']:.1f}%")
    """
    stats_path = Path(stats_path)

    if not stats_path.exists():
        return None

    try:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        return stats
    except json.JSONDecodeError:
        return None


def plot_training_curves(metrics, experiment_name, save_dir):
    """
    Plot training and validation loss/accuracy curves.

    Creates a 2x1 subplot figure showing:
    - Top: Training and validation loss over epochs
    - Bottom: Training and validation accuracy over epochs

    Args:
        metrics (dict): Metrics dictionary from load_metrics()
        experiment_name (str): Name for plot title (e.g., 'BaseFP32')
        save_dir (str or Path): Directory to save the plot

    Returns:
        Path: Path to saved plot file

    Example Output:
        Figure with two subplots:
        - Loss curves showing convergence and potential overfitting
        - Accuracy curves showing training progress
        - Vertical line at best validation accuracy epoch
        - Grid for easy value reading
        - Legend identifying train/val curves

    Example:
        >>> metrics = load_metrics('results/BaseFP32/metrics/final_metrics.json')
        >>> plot_training_curves(metrics, 'BaseFP32', 'results/BaseFP32/plots')
        PosixPath('results/BaseFP32/plots/training_curves.png')
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    epochs = range(1, len(metrics['train_loss']) + 1)
    train_loss = metrics['train_loss']
    val_loss = metrics['val_loss']
    train_acc = metrics['train_acc']
    val_acc = metrics['val_acc']

    # Find best validation accuracy epoch
    best_epoch = val_acc.index(max(val_acc)) + 1

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Loss curves
    ax1.plot(epochs, train_loss, label='Training Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2)
    ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, label=f'Best Val Acc (epoch {best_epoch})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{experiment_name} - Training and Validation Loss')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy curves
    ax2.plot(epochs, train_acc, label='Training Accuracy', color='#1f77b4', linewidth=2)
    ax2.plot(epochs, val_acc, label='Validation Accuracy', color='#ff7f0e', linewidth=2)
    ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, label=f'Best Val Acc (epoch {best_epoch})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{experiment_name} - Training and Validation Accuracy')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Set y-axis limits for accuracy (0 to 1)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()

    # Save figure
    output_path = save_dir / f'{experiment_name}_training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def plot_learning_rate(metrics, experiment_name, save_dir):
    """
    Plot learning rate schedule over training.

    Creates a line plot showing how the learning rate changed during training
    (typically showing cosine annealing or warmup schedules).

    Args:
        metrics (dict): Metrics dictionary from load_metrics()
        experiment_name (str): Name for plot title
        save_dir (str or Path): Directory to save the plot

    Returns:
        Path: Path to saved plot file

    Example:
        >>> metrics = load_metrics('results/BaseFP32/metrics/final_metrics.json')
        >>> plot_learning_rate(metrics, 'BaseFP32', 'results/BaseFP32/plots')
        PosixPath('results/BaseFP32/plots/learning_rate.png')
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if 'learning_rates' not in metrics or not metrics['learning_rates']:
        print(f"⚠️  No learning rate data found for {experiment_name}")
        return None

    epochs = range(1, len(metrics['learning_rates']) + 1)
    lr = metrics['learning_rates']

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, lr, color='#2ca02c', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title(f'{experiment_name} - Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visibility

    plt.tight_layout()

    output_path = save_dir / f'{experiment_name}_learning_rate.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def plot_hardware_stats(hw_stats, experiment_name, save_dir):
    """
    Plot hardware monitoring statistics (CPU, memory, thermal).

    Creates a multi-panel plot showing:
    - CPU utilization over time
    - Memory usage over time
    - Thermal throttling levels (if available)

    Args:
        hw_stats (dict): Hardware stats from load_hardware_stats()
        experiment_name (str): Name for plot title
        save_dir (str or Path): Directory to save the plot

    Returns:
        Path: Path to saved plot file, or None if no data available

    Example:
        >>> hw_stats = load_hardware_stats('results/BaseFP32/metrics/hardware_stats.json')
        >>> if hw_stats:
        ...     plot_hardware_stats(hw_stats, 'BaseFP32', 'results/BaseFP32/plots')
    """
    if hw_stats is None:
        return None

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Extract full stats (time series)
    full_stats = hw_stats.get('full_stats', {})

    if not full_stats.get('timestamps'):
        print(f"⚠️  No hardware time-series data found for {experiment_name}")
        return None

    timestamps = np.array(full_stats['timestamps']) / 60  # Convert to minutes
    cpu_percent = full_stats['cpu_percent']
    mem_percent = full_stats['memory_percent']
    thermal = full_stats.get('thermal_pressure', [])

    # Create figure with 2 or 3 subplots depending on thermal data
    n_plots = 3 if thermal and any(thermal) else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3*n_plots))

    if n_plots == 2:
        ax1, ax2 = axes
    else:
        ax1, ax2, ax3 = axes

    # Plot 1: CPU usage
    ax1.plot(timestamps, cpu_percent, color='#d62728', linewidth=1.5)
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('CPU Usage (%)')
    ax1.set_title(f'{experiment_name} - CPU Utilization')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # Add mean line
    mean_cpu = np.mean(cpu_percent)
    ax1.axhline(y=mean_cpu, color='gray', linestyle='--', alpha=0.5, label=f'Mean: {mean_cpu:.1f}%')
    ax1.legend(loc='best')

    # Plot 2: Memory usage
    ax2.plot(timestamps, mem_percent, color='#9467bd', linewidth=1.5)
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Memory Usage (%)')
    ax2.set_title(f'{experiment_name} - Memory Utilization')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    # Add mean line
    mean_mem = np.mean(mem_percent)
    ax2.axhline(y=mean_mem, color='gray', linestyle='--', alpha=0.5, label=f'Mean: {mean_mem:.1f}%')
    ax2.legend(loc='best')

    # Plot 3: Thermal pressure (if available)
    if n_plots == 3:
        ax3.plot(timestamps, thermal, color='#ff7f0e', linewidth=1.5)
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Thermal Pressure Level')
        ax3.set_title(f'{experiment_name} - Thermal Throttling')
        ax3.grid(True, alpha=0.3)

        max_thermal = max(thermal)
        if max_thermal > 0:
            ax3.axhline(y=max_thermal, color='red', linestyle='--', alpha=0.5, label=f'Max: {max_thermal}')
            ax3.legend(loc='best')

    plt.tight_layout()

    output_path = save_dir / f'{experiment_name}_hardware_stats.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def plot_comparison(experiments_data, metric_name, save_dir):
    """
    Plot comparison of a specific metric across multiple experiments.

    Creates a line plot comparing the same metric (e.g., validation accuracy)
    across different experiments on the same axes.

    Args:
        experiments_data (list): List of dicts with 'name' and 'metrics' keys
        metric_name (str): Metric to plot (e.g., 'val_acc', 'train_loss')
        save_dir (str or Path): Directory to save the plot

    Returns:
        Path: Path to saved plot file

    Example:
        >>> exp1 = {'name': 'BaseFP32', 'metrics': load_metrics('results/BaseFP32/...')}
        >>> exp2 = {'name': 'BaseFP16', 'metrics': load_metrics('results/BaseFP16/...')}
        >>> plot_comparison([exp1, exp2], 'val_acc', 'results/comparison_plots')
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Map metric names to display names
    metric_labels = {
        'val_acc': 'Validation Accuracy',
        'train_acc': 'Training Accuracy',
        'val_loss': 'Validation Loss',
        'train_loss': 'Training Loss'
    }

    ylabel = metric_labels.get(metric_name, metric_name)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each experiment
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, exp_data in enumerate(experiments_data):
        name = exp_data['name']
        metrics = exp_data['metrics']

        if metric_name not in metrics:
            continue

        epochs = range(1, len(metrics[metric_name]) + 1)
        values = metrics[metric_name]
        color = colors[i % len(colors)]

        ax.plot(epochs, values, label=name, color=color, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Comparison: {ylabel} Across Experiments')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Set y-axis limits for accuracy metrics
    if 'acc' in metric_name:
        ax.set_ylim(0, 1.05)

    plt.tight_layout()

    # Save figure
    output_path = save_dir / f'comparison_{metric_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    """
    Main execution function for plot generation script.

    Parses command-line arguments and generates appropriate plots
    for single experiments or multi-experiment comparisons.

    Command-line Arguments:
        --experiment: Path to single experiment directory
        --compare: List of experiment directories to compare
        --all: Plot all experiments in results/ directory
        --output: Custom output directory for plots

    Examples:
        # Plot single experiment
        $ python scripts/generate_plots.py --experiment results/BaseFP32

        # Compare multiple experiments
        $ python scripts/generate_plots.py --compare results/BaseFP32 results/BaseFP16

        # Plot all experiments
        $ python scripts/generate_plots.py --all
    """
    parser = argparse.ArgumentParser(
        description='Generate plots from training metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot single experiment
  python scripts/generate_plots.py --experiment results/BaseFP32

  # Plot with custom output directory
  python scripts/generate_plots.py --experiment results/BaseFP32 --output custom_plots/

  # Compare multiple experiments
  python scripts/generate_plots.py --compare results/BaseFP32 results/BaseFP16 results/AugmFP16

  # Plot all experiments
  python scripts/generate_plots.py --all
        """
    )

    parser.add_argument(
        '--experiment',
        type=str,
        help='Path to single experiment directory'
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        help='List of experiment directories to compare'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Plot all experiments in results/ directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Custom output directory for plots'
    )

    args = parser.parse_args()

    # Single experiment mode
    if args.experiment:
        exp_path = Path(args.experiment)
        exp_name = exp_path.name
        metrics_file = exp_path / 'metrics' / 'final_metrics.json'
        hw_stats_file = exp_path / 'metrics' / 'hardware_stats.json'

        # Determine output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = exp_path / 'plots'

        try:
            print(f"📊 Generating plots for: {exp_name}")
            print(f"📁 Output directory: {output_dir}\n")

            # Load metrics
            metrics = load_metrics(metrics_file)

            # Plot training curves
            print("  Creating training curves...")
            curves_path = plot_training_curves(metrics, exp_name, output_dir)
            print(f"  ✓ Saved: {curves_path}")

            # Plot learning rate
            print("  Creating learning rate plot...")
            lr_path = plot_learning_rate(metrics, exp_name, output_dir)
            if lr_path:
                print(f"  ✓ Saved: {lr_path}")

            # Plot hardware stats if available
            hw_stats = load_hardware_stats(hw_stats_file)
            if hw_stats:
                print("  Creating hardware stats plots...")
                hw_path = plot_hardware_stats(hw_stats, exp_name, output_dir)
                if hw_path:
                    print(f"  ✓ Saved: {hw_path}")

            print(f"\n✓ All plots generated successfully!\n")

        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    # Comparison mode
    elif args.compare or args.all:
        if args.all:
            # Discover all experiments
            from extract_metrics import discover_experiments
            exp_paths = discover_experiments('results')
            print(f"📊 Found {len(exp_paths)} experiments: {[e.name for e in exp_paths]}\n")
        else:
            exp_paths = [Path(p) for p in args.compare]

        if not exp_paths:
            print("❌ No experiments found")
            sys.exit(1)

        # Determine output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = Path('results') / 'comparison_plots'

        print(f"📁 Output directory: {output_dir}\n")

        # Load all metrics
        experiments_data = []
        for exp_path in exp_paths:
            metrics_file = exp_path / 'metrics' / 'final_metrics.json'
            try:
                metrics = load_metrics(metrics_file)
                experiments_data.append({'name': exp_path.name, 'metrics': metrics})
            except FileNotFoundError:
                print(f"⚠️  Warning: Metrics not found for {exp_path.name}, skipping...")

        if not experiments_data:
            print("❌ No valid experiments to compare")
            sys.exit(1)

        # Generate comparison plots
        print(f"📊 Comparing {len(experiments_data)} experiments...\n")

        metrics_to_plot = ['val_acc', 'train_acc', 'val_loss', 'train_loss']

        for metric in metrics_to_plot:
            print(f"  Creating {metric} comparison...")
            comp_path = plot_comparison(experiments_data, metric, output_dir)
            print(f"  ✓ Saved: {comp_path}")

        print(f"\n✓ All comparison plots generated successfully!\n")

    else:
        print("❌ Error: Must specify --experiment, --compare, or --all")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

"""
Extract and analyze training metrics from experiment results.

This script provides utilities for loading, analyzing, and comparing metrics
from multiple ViT training experiments. It can:
- Load metrics from JSON files in experiment directories
- Compute summary statistics (best accuracy, convergence, stability)
- Compare multiple experiments side-by-side
- Export results to CSV for further analysis
- Generate formatted tables for reports and papers

The script automatically discovers experiments in the results/ directory
and extracts metrics from the standard metrics/final_metrics.json files.

Usage:
    Extract metrics from single experiment:
        $ python scripts/extract_metrics.py --experiment results/BaseFP32

    Compare multiple experiments:
        $ python scripts/extract_metrics.py --compare results/BaseFP32 results/BaseFP16

    Extract all experiments and save to CSV:
        $ python scripts/extract_metrics.py --all --output metrics_summary.csv

    Show detailed statistics:
        $ python scripts/extract_metrics.py --experiment results/AugmFP16 --detailed

Output:
    The script prints formatted tables to console and optionally saves
    CSV files for further analysis. Example output:

    Experiment Summary: BaseFP32
    ============================================================
    Best Validation Accuracy: 0.8234 (epoch 45)
    Final Training Loss: 0.4521
    Final Validation Loss: 0.6832
    Convergence Epoch: 38 (within 0.5% of best)
    Average Epoch Time: 125.3 seconds
    Total Training Time: 104.4 minutes
    ============================================================

Example:
    >>> from extract_metrics import load_experiment_metrics, compute_statistics
    >>>
    >>> # Load metrics
    >>> metrics = load_experiment_metrics('results/BaseFP32/metrics/final_metrics.json')
    >>>
    >>> # Compute statistics
    >>> stats = compute_statistics(metrics)
    >>> print(f"Best accuracy: {stats['best_val_acc']:.4f}")
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def load_experiment_metrics(metrics_path):
    """
    Load training metrics from JSON file.

    Reads a metrics JSON file (typically final_metrics.json) and returns
    the contents as a dictionary with time-series data for all tracked metrics.

    Args:
        metrics_path (str or Path): Path to metrics JSON file
            Example: 'results/BaseFP32/metrics/final_metrics.json'

    Returns:
        dict: Metrics dictionary with keys:
            - train_loss (list): Training loss per epoch
            - train_acc (list): Training accuracy per epoch
            - val_loss (list): Validation loss per epoch
            - val_acc (list): Validation accuracy per epoch
            - epoch_time (list): Duration of each epoch in seconds
            - learning_rates (list): Learning rate per epoch

    Raises:
        FileNotFoundError: If metrics file doesn't exist
        json.JSONDecodeError: If file is not valid JSON

    Example:
        >>> metrics = load_experiment_metrics('results/BaseFP32/metrics/final_metrics.json')
        >>> print(f"Trained for {len(metrics['train_loss'])} epochs")
        Trained for 50 epochs
        >>> print(f"Best accuracy: {max(metrics['val_acc']):.4f}")
        Best accuracy: 0.8234
    """
    metrics_path = Path(metrics_path)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    return metrics


def compute_statistics(metrics):
    """
    Compute summary statistics from training metrics.

    Analyzes time-series metrics and computes key statistics including
    best/final values, convergence behavior, and training stability.

    Args:
        metrics (dict): Metrics dictionary from load_experiment_metrics()

    Returns:
        dict: Statistics dictionary with keys:
            - best_val_acc (float): Best validation accuracy achieved
            - best_val_acc_epoch (int): Epoch where best accuracy occurred
            - final_train_loss (float): Training loss at final epoch
            - final_val_loss (float): Validation loss at final epoch
            - final_train_acc (float): Training accuracy at final epoch
            - final_val_acc (float): Validation accuracy at final epoch
            - avg_epoch_time (float): Mean epoch duration in seconds
            - total_time_minutes (float): Total training time in minutes
            - convergence_epoch (int): Epoch where model converged (within 0.5% of best)
            - generalization_gap (float): Final train_acc - final val_acc
            - overfitting_score (float): best_val_acc - final_val_acc (>0 indicates overfitting)

    Example:
        >>> stats = compute_statistics(metrics)
        >>> print(f"Best accuracy: {stats['best_val_acc']:.4f} at epoch {stats['best_val_acc_epoch']}")
        Best accuracy: 0.8234 at epoch 45
        >>> if stats['overfitting_score'] > 0.02:
        ...     print("Warning: Significant overfitting detected")

    Notes:
        - Convergence is defined as reaching within 0.5% of best validation accuracy
        - Generalization gap measures train/val performance difference
        - Overfitting score >0 indicates accuracy degradation after peak
    """
    # Extract time-series data
    train_loss = metrics.get('train_loss', [])
    train_acc = metrics.get('train_acc', [])
    val_loss = metrics.get('val_loss', [])
    val_acc = metrics.get('val_acc', [])
    epoch_time = metrics.get('epoch_time', [])

    # Compute statistics
    stats = {}

    # Best validation accuracy
    if val_acc:
        stats['best_val_acc'] = max(val_acc)
        stats['best_val_acc_epoch'] = val_acc.index(stats['best_val_acc']) + 1
    else:
        stats['best_val_acc'] = 0.0
        stats['best_val_acc_epoch'] = 0

    # Final values
    stats['final_train_loss'] = train_loss[-1] if train_loss else 0.0
    stats['final_val_loss'] = val_loss[-1] if val_loss else 0.0
    stats['final_train_acc'] = train_acc[-1] if train_acc else 0.0
    stats['final_val_acc'] = val_acc[-1] if val_acc else 0.0

    # Timing
    if epoch_time:
        stats['avg_epoch_time'] = np.mean(epoch_time)
        stats['total_time_minutes'] = sum(epoch_time) / 60
    else:
        stats['avg_epoch_time'] = 0.0
        stats['total_time_minutes'] = 0.0

    # Convergence analysis (when did model reach within 0.5% of best accuracy?)
    if val_acc:
        threshold = stats['best_val_acc'] * 0.995  # 99.5% of best
        convergence_epochs = [i+1 for i, acc in enumerate(val_acc) if acc >= threshold]
        stats['convergence_epoch'] = convergence_epochs[0] if convergence_epochs else len(val_acc)
    else:
        stats['convergence_epoch'] = 0

    # Generalization gap (train vs val accuracy at end)
    stats['generalization_gap'] = stats['final_train_acc'] - stats['final_val_acc']

    # Overfitting score (best vs final val accuracy)
    stats['overfitting_score'] = stats['best_val_acc'] - stats['final_val_acc']

    return stats


def print_experiment_summary(experiment_name, stats, detailed=False):
    """
    Print formatted summary of experiment statistics.

    Displays key metrics in a readable table format suitable for reports
    or console output.

    Args:
        experiment_name (str): Name of experiment (e.g., 'BaseFP32')
        stats (dict): Statistics from compute_statistics()
        detailed (bool): If True, show additional detailed metrics. Default: False

    Example Output (basic):
        Experiment Summary: BaseFP32
        ============================================================
        Best Validation Accuracy: 0.8234 (epoch 45)
        Final Training Loss: 0.4521
        Final Validation Loss: 0.6832
        Convergence Epoch: 38
        Average Epoch Time: 125.3 seconds
        Total Training Time: 104.4 minutes
        ============================================================

    Example Output (detailed):
        Experiment Summary: BaseFP32
        ============================================================
        Accuracy Metrics:
          Best Validation Accuracy: 0.8234 (epoch 45)
          Final Training Accuracy: 0.8512
          Final Validation Accuracy: 0.8156
          Generalization Gap: 0.0356 (train - val)

        Loss Metrics:
          Final Training Loss: 0.4521
          Final Validation Loss: 0.6832

        Training Dynamics:
          Convergence Epoch: 38
          Overfitting Score: 0.0078 (best - final val_acc)

        Timing:
          Average Epoch Time: 125.3 seconds
          Total Training Time: 104.4 minutes
        ============================================================
    """
    print(f"\nExperiment Summary: {experiment_name}")
    print("="*60)

    if detailed:
        # Detailed view with sections
        print("Accuracy Metrics:")
        print(f"  Best Validation Accuracy: {stats['best_val_acc']:.4f} (epoch {stats['best_val_acc_epoch']})")
        print(f"  Final Training Accuracy: {stats['final_train_acc']:.4f}")
        print(f"  Final Validation Accuracy: {stats['final_val_acc']:.4f}")
        print(f"  Generalization Gap: {stats['generalization_gap']:.4f} (train - val)")

        print("\nLoss Metrics:")
        print(f"  Final Training Loss: {stats['final_train_loss']:.4f}")
        print(f"  Final Validation Loss: {stats['final_val_loss']:.4f}")

        print("\nTraining Dynamics:")
        print(f"  Convergence Epoch: {stats['convergence_epoch']}")
        print(f"  Overfitting Score: {stats['overfitting_score']:.4f} (best - final val_acc)")

        print("\nTiming:")
        print(f"  Average Epoch Time: {stats['avg_epoch_time']:.1f} seconds")
        print(f"  Total Training Time: {stats['total_time_minutes']:.1f} minutes")
    else:
        # Compact view
        print(f"Best Validation Accuracy: {stats['best_val_acc']:.4f} (epoch {stats['best_val_acc_epoch']})")
        print(f"Final Training Loss: {stats['final_train_loss']:.4f}")
        print(f"Final Validation Loss: {stats['final_val_loss']:.4f}")
        print(f"Convergence Epoch: {stats['convergence_epoch']}")
        print(f"Average Epoch Time: {stats['avg_epoch_time']:.1f} seconds")
        print(f"Total Training Time: {stats['total_time_minutes']:.1f} minutes")

    print("="*60 + "\n")


def compare_experiments(experiment_paths, output_csv=None):
    """
    Compare metrics from multiple experiments side-by-side.

    Loads metrics from multiple experiment directories and creates a
    comparison table showing key statistics for each experiment.

    Args:
        experiment_paths (list): List of paths to experiment directories
            Example: ['results/BaseFP32', 'results/BaseFP16', 'results/AugmFP16']
        output_csv (str, optional): If provided, save comparison table to CSV file

    Returns:
        list: List of dictionaries, one per experiment, containing:
            - name: Experiment name
            - stats: Statistics dictionary from compute_statistics()

    Example Output:
        Experiment Comparison
        ============================================================
        Experiment       | Best Acc | Final Loss | Conv Epoch | Time (min)
        -----------------|----------|------------|------------|------------
        BaseFP32         |  0.8234  |   0.6832   |     38     |   104.4
        BaseFP16         |  0.8198  |   0.6901   |     40     |    87.2
        AugmFP16         |  0.8312  |   0.6654   |     42     |    92.1
        ============================================================

    Example:
        >>> experiments = ['results/BaseFP32', 'results/BaseFP16']
        >>> results = compare_experiments(experiments, output_csv='comparison.csv')
        >>> best_exp = max(results, key=lambda x: x['stats']['best_val_acc'])
        >>> print(f"Best experiment: {best_exp['name']}")
    """
    results = []

    # Load metrics from all experiments
    for exp_path in experiment_paths:
        exp_path = Path(exp_path)
        exp_name = exp_path.name

        # Try to find metrics file
        metrics_file = exp_path / 'metrics' / 'final_metrics.json'

        try:
            metrics = load_experiment_metrics(metrics_file)
            stats = compute_statistics(metrics)
            results.append({'name': exp_name, 'stats': stats})
        except FileNotFoundError:
            print(f"⚠️  Warning: Metrics not found for {exp_name}, skipping...")
            continue

    if not results:
        print("❌ No experiments found with valid metrics")
        return results

    # Print comparison table
    print("\nExperiment Comparison")
    print("="*80)
    print(f"{'Experiment':<16} | {'Best Acc':<8} | {'Final Loss':<10} | {'Conv Epoch':<10} | {'Time (min)':<10}")
    print("-"*80)

    for result in results:
        name = result['name']
        stats = result['stats']
        print(f"{name:<16} | {stats['best_val_acc']:>8.4f} | "
              f"{stats['final_val_loss']:>10.4f} | {stats['convergence_epoch']:>10} | "
              f"{stats['total_time_minutes']:>10.1f}")

    print("="*80 + "\n")

    # Save to CSV if requested
    if output_csv:
        import csv
        output_path = Path(output_csv)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Experiment', 'Best_Val_Acc', 'Best_Acc_Epoch',
                'Final_Train_Loss', 'Final_Val_Loss',
                'Final_Train_Acc', 'Final_Val_Acc',
                'Convergence_Epoch', 'Generalization_Gap', 'Overfitting_Score',
                'Avg_Epoch_Time_sec', 'Total_Time_min'
            ])

            # Data
            for result in results:
                name = result['name']
                stats = result['stats']
                writer.writerow([
                    name,
                    stats['best_val_acc'],
                    stats['best_val_acc_epoch'],
                    stats['final_train_loss'],
                    stats['final_val_loss'],
                    stats['final_train_acc'],
                    stats['final_val_acc'],
                    stats['convergence_epoch'],
                    stats['generalization_gap'],
                    stats['overfitting_score'],
                    stats['avg_epoch_time'],
                    stats['total_time_minutes']
                ])

        print(f"✓ Comparison table saved to: {output_path}\n")

    return results


def discover_experiments(results_dir='results'):
    """
    Automatically discover all experiments in results directory.

    Scans the results directory for subdirectories containing metrics files
    and returns a list of valid experiment paths.

    Args:
        results_dir (str): Path to results directory. Default: 'results'

    Returns:
        list: List of Path objects pointing to valid experiment directories

    Example:
        >>> experiments = discover_experiments('results')
        >>> print(f"Found {len(experiments)} experiments")
        Found 4 experiments
        >>> for exp in experiments:
        ...     print(exp.name)
        BaseFP32
        AugmFP32
        BaseFP16
        AugmFP16
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        return []

    experiments = []

    # Look for directories with metrics/final_metrics.json
    for exp_dir in results_path.iterdir():
        if exp_dir.is_dir():
            metrics_file = exp_dir / 'metrics' / 'final_metrics.json'
            if metrics_file.exists():
                experiments.append(exp_dir)

    return sorted(experiments)


def main():
    """
    Main execution function for metrics extraction script.

    Parses command-line arguments and executes the appropriate metrics
    extraction/comparison workflow.

    Command-line Arguments:
        --experiment: Path to single experiment directory
        --compare: List of experiment directories to compare
        --all: Process all experiments in results/ directory
        --output: CSV file path for saving comparison results
        --detailed: Show detailed statistics (for single experiment)

    Examples:
        # Single experiment with detailed stats
        $ python scripts/extract_metrics.py --experiment results/BaseFP32 --detailed

        # Compare multiple experiments
        $ python scripts/extract_metrics.py --compare results/BaseFP32 results/BaseFP16

        # Extract all experiments and save to CSV
        $ python scripts/extract_metrics.py --all --output metrics_summary.csv

    Workflow:
        1. Parse command-line arguments
        2. Load metrics from specified experiment(s)
        3. Compute statistics
        4. Print formatted output
        5. Optionally save to CSV
    """
    parser = argparse.ArgumentParser(
        description='Extract and analyze training metrics from experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single experiment
  python scripts/extract_metrics.py --experiment results/BaseFP32

  # Single experiment with detailed stats
  python scripts/extract_metrics.py --experiment results/BaseFP32 --detailed

  # Compare multiple experiments
  python scripts/extract_metrics.py --compare results/BaseFP32 results/BaseFP16 results/AugmFP16

  # Extract all experiments and save to CSV
  python scripts/extract_metrics.py --all --output metrics_summary.csv
        """
    )

    parser.add_argument(
        '--experiment',
        type=str,
        help='Path to single experiment directory (e.g., results/BaseFP32)'
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        help='List of experiment directories to compare'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all experiments in results/ directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file for comparison results'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed statistics (for single experiment)'
    )

    args = parser.parse_args()

    # Single experiment mode
    if args.experiment:
        exp_path = Path(args.experiment)
        metrics_file = exp_path / 'metrics' / 'final_metrics.json'

        try:
            print(f"📊 Loading metrics from: {metrics_file}")
            metrics = load_experiment_metrics(metrics_file)
            stats = compute_statistics(metrics)
            print_experiment_summary(exp_path.name, stats, detailed=args.detailed)
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing metrics JSON: {e}")
            sys.exit(1)

    # Comparison mode
    elif args.compare:
        print(f"📊 Comparing {len(args.compare)} experiments...")
        compare_experiments(args.compare, output_csv=args.output)

    # All experiments mode
    elif args.all:
        print("📊 Discovering experiments in results/ directory...")
        experiments = discover_experiments('results')

        if not experiments:
            print("❌ No experiments found in results/ directory")
            sys.exit(1)

        print(f"✓ Found {len(experiments)} experiments: {[e.name for e in experiments]}")
        compare_experiments(experiments, output_csv=args.output)

    else:
        print("❌ Error: Must specify --experiment, --compare, or --all")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

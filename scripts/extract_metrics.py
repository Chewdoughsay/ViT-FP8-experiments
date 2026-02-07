#!/usr/bin/env python3
"""
Extract Complete Metrics from All Experiments
Generates data for final report
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_experiment_metrics(exp_name, base_dir='results/checkpoints'):
    """Load all metrics for an experiment"""
    exp_dir = Path(base_dir) / exp_name

    data = {
        'name': exp_name,
        'exists': exp_dir.exists(),
        'metrics': None,
        'hardware': None,
        'timing': None
    }

    if not exp_dir.exists():
        return data

    # Load final_metrics.json
    metrics_file = exp_dir / 'final_metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        val_acc = metrics.get('val_acc', [])
        train_acc = metrics.get('train_acc', [])
        val_loss = metrics.get('val_loss', [])
        train_loss = metrics.get('train_loss', [])
        epoch_times = metrics.get('epoch_time', [])

        if val_acc and train_acc:
            data['metrics'] = {
                'num_epochs': len(val_acc),
                'best_val_acc': max(val_acc),
                'final_val_acc': val_acc[-1],
                'final_train_acc': train_acc[-1],
                'final_val_loss': val_loss[-1] if val_loss else 0,
                'final_train_loss': train_loss[-1] if train_loss else 0,
                'gap': train_acc[-1] - val_acc[-1],
                'val_acc_history': val_acc,
                'train_acc_history': train_acc,
            }

        if epoch_times:
            data['timing'] = {
                'total_time_hours': sum(epoch_times) / 3600,
                'avg_epoch_seconds': np.mean(epoch_times),
                'min_epoch_seconds': min(epoch_times),
                'max_epoch_seconds': max(epoch_times),
                'std_epoch_seconds': np.std(epoch_times),
            }

    # Load hardware_stats.json
    hw_file = exp_dir / 'hardware_stats.json'
    if hw_file.exists():
        with open(hw_file, 'r') as f:
            hw_data = json.load(f)

        cpu_vals = hw_data.get('cpu_percent', [])
        mem_vals = hw_data.get('memory_percent', [])
        thermal_vals = hw_data.get('thermal_pressure', [])

        if cpu_vals:
            data['hardware'] = {
                'avg_cpu': np.mean(cpu_vals),
                'max_cpu': np.max(cpu_vals),
                'avg_memory': np.mean(mem_vals) if mem_vals else 0,
                'max_memory': np.max(mem_vals) if mem_vals else 0,
                'max_thermal': np.max(thermal_vals) if thermal_vals else 0,
                'throttled': any(t > 0 for t in thermal_vals) if thermal_vals else False,
            }

    return data


def generate_comparison_table(experiments_data):
    """Generate comparison table for report"""

    print("=" * 100)
    print("COMPARISON TABLE - ALL 4 EXPERIMENTS")
    print("=" * 100)
    print()

    # Performance metrics
    print("📊 PERFORMANCE METRICS")
    print("-" * 100)
    print(f"{'Metric':<30} | {'Baseline':<12} | {'Regularized':<12} | {'FP16 Old':<12} | {'FP16 Fixed':<12}")
    print("-" * 100)

    for exp in experiments_data:
        if not exp['metrics']:
            continue

        m = exp['metrics']

        # Row: Epochs
        if exp == experiments_data[0]:
            epochs = [e['metrics']['num_epochs'] if e['metrics'] else 0 for e in experiments_data]
            print(f"{'Epochs Trained':<30} | {epochs[0]:<12} | {epochs[1]:<12} | {epochs[2]:<12} | {epochs[3]:<12}")

        # Row: Best Val Acc
        if exp == experiments_data[0]:
            vals = [f"{e['metrics']['best_val_acc'] * 100:.2f}%" if e['metrics'] else "N/A" for e in experiments_data]
            print(f"{'Best Val Accuracy':<30} | {vals[0]:<12} | {vals[1]:<12} | {vals[2]:<12} | {vals[3]:<12}")

        # Row: Final Val Acc
        if exp == experiments_data[0]:
            vals = [f"{e['metrics']['final_val_acc'] * 100:.2f}%" if e['metrics'] else "N/A" for e in experiments_data]
            print(f"{'Final Val Accuracy':<30} | {vals[0]:<12} | {vals[1]:<12} | {vals[2]:<12} | {vals[3]:<12}")

        # Row: Final Train Acc
        if exp == experiments_data[0]:
            vals = [f"{e['metrics']['final_train_acc'] * 100:.2f}%" if e['metrics'] else "N/A" for e in
                    experiments_data]
            print(f"{'Final Train Accuracy':<30} | {vals[0]:<12} | {vals[1]:<12} | {vals[2]:<12} | {vals[3]:<12}")

        # Row: Gap
        if exp == experiments_data[0]:
            vals = [f"{e['metrics']['gap'] * 100:.2f}%" if e['metrics'] else "N/A" for e in experiments_data]
            print(f"{'Overfitting Gap':<30} | {vals[0]:<12} | {vals[1]:<12} | {vals[2]:<12} | {vals[3]:<12}")

        # Row: Final Val Loss
        if exp == experiments_data[0]:
            vals = [f"{e['metrics']['final_val_loss']:.4f}" if e['metrics'] else "N/A" for e in experiments_data]
            print(f"{'Final Val Loss':<30} | {vals[0]:<12} | {vals[1]:<12} | {vals[2]:<12} | {vals[3]:<12}")

    print()

    # Timing metrics
    print("⏱️  TIMING METRICS")
    print("-" * 100)

    if experiments_data[0]['timing']:
        # Row: Total time
        vals = [f"{e['timing']['total_time_hours']:.2f}h" if e.get('timing') else "N/A" for e in experiments_data]
        print(f"{'Total Time':<30} | {vals[0]:<12} | {vals[1]:<12} | {vals[2]:<12} | {vals[3]:<12}")

        # Row: Avg epoch time
        vals = [f"{e['timing']['avg_epoch_seconds']:.1f}s" if e.get('timing') else "N/A" for e in experiments_data]
        print(f"{'Avg Time/Epoch':<30} | {vals[0]:<12} | {vals[1]:<12} | {vals[2]:<12} | {vals[3]:<12}")

        # Row: Std epoch time
        vals = [f"±{e['timing']['std_epoch_seconds']:.1f}s" if e.get('timing') else "N/A" for e in experiments_data]
        print(f"{'Epoch Time StdDev':<30} | {vals[0]:<12} | {vals[1]:<12} | {vals[2]:<12} | {vals[3]:<12}")

    print()

    # Hardware metrics
    print("💻 HARDWARE METRICS")
    print("-" * 100)

    if experiments_data[0]['hardware']:
        # Row: CPU avg
        vals = [f"{e['hardware']['avg_cpu']:.1f}%" if e.get('hardware') else "N/A" for e in experiments_data]
        print(f"{'CPU Avg':<30} | {vals[0]:<12} | {vals[1]:<12} | {vals[2]:<12} | {vals[3]:<12}")

        # Row: CPU max
        vals = [f"{e['hardware']['max_cpu']:.1f}%" if e.get('hardware') else "N/A" for e in experiments_data]
        print(f"{'CPU Max':<30} | {vals[0]:<12} | {vals[1]:<12} | {vals[2]:<12} | {vals[3]:<12}")

        # Row: Memory avg
        vals = [f"{e['hardware']['avg_memory']:.1f}%" if e.get('hardware') else "N/A" for e in experiments_data]
        print(f"{'Memory Avg':<30} | {vals[0]:<12} | {vals[1]:<12} | {vals[2]:<12} | {vals[3]:<12}")

        # Row: Thermal
        vals = ["No" if not e.get('hardware', {}).get('throttled', False) else "Yes" for e in experiments_data]
        print(f"{'Thermal Throttling':<30} | {vals[0]:<12} | {vals[1]:<12} | {vals[2]:<12} | {vals[3]:<12}")

    print()


def generate_latex_table(experiments_data, output_file='metrics_comparison_4exp.tex'):
    """Generate LaTeX table"""

    latex = r"""\begin{table}[H]
\centering
\small
\begin{tabular}{lrrrr}
\toprule
\textbf{Metrică} & \textbf{Baseline} & \textbf{Regularized} & \textbf{FP16 Old} & \textbf{FP16 Fixed} \\
\midrule
\multicolumn{5}{l}{\textit{\textbf{Performanță Model}}} \\
"""

    # Extract data
    exp_names = ['baseline_fp32', 'experiment2_regularized', 'experiment3_fp16', 'exp3_fp16_fixed']
    data_map = {e['name']: e for e in experiments_data}
    ordered_data = [data_map.get(name, {'metrics': None}) for name in exp_names]

    # Epochs
    vals = [str(e['metrics']['num_epochs']) if e.get('metrics') else "N/A" for e in ordered_data]
    latex += f"Epoci Antrenate & {' & '.join(vals)} \\\\\n"

    # Best Val Acc
    vals = [f"{e['metrics']['best_val_acc'] * 100:.2f}\\%" if e.get('metrics') else "N/A" for e in ordered_data]
    latex += f"Best Val Accuracy & {' & '.join(vals)} \\\\\n"

    # Final Val Acc
    vals = [f"{e['metrics']['final_val_acc'] * 100:.2f}\\%" if e.get('metrics') else "N/A" for e in ordered_data]
    latex += f"Final Val Accuracy & {' & '.join(vals)} \\\\\n"

    # Final Train Acc
    vals = [f"{e['metrics']['final_train_acc'] * 100:.2f}\\%" if e.get('metrics') else "N/A" for e in ordered_data]
    latex += f"Final Train Accuracy & {' & '.join(vals)} \\\\\n"

    # Gap
    vals = [f"{e['metrics']['gap'] * 100:.2f}\\%" if e.get('metrics') else "N/A" for e in ordered_data]
    latex += f"Overfitting Gap & {' & '.join(vals)} \\\\\n"

    # Final Val Loss
    vals = [f"{e['metrics']['final_val_loss']:.4f}" if e.get('metrics') else "N/A" for e in ordered_data]
    latex += f"Final Val Loss & {' & '.join(vals)} \\\\\n"

    latex += r"""\midrule
\multicolumn{5}{l}{\textit{\textbf{Timp Antrenare}}} \\
"""

    # Timing
    if ordered_data[0].get('timing'):
        vals = [f"{e['timing']['total_time_hours']:.2f}h" if e.get('timing') else "N/A" for e in ordered_data]
        latex += f"Timp Total (ore) & {' & '.join(vals)} \\\\\n"

        vals = [f"{e['timing']['avg_epoch_seconds']:.1f}s" if e.get('timing') else "N/A" for e in ordered_data]
        latex += f"Timp/Epocă (sec) & {' & '.join(vals)} \\\\\n"

        vals = [f"±{e['timing']['std_epoch_seconds']:.1f}s" if e.get('timing') else "N/A" for e in ordered_data]
        latex += f"Timp/Epocă StdDev & {' & '.join(vals)} \\\\\n"

    latex += r"""\midrule
\multicolumn{5}{l}{\textit{\textbf{Utilizare Hardware}}} \\
"""

    # Hardware
    if ordered_data[0].get('hardware'):
        vals = [f"{e['hardware']['avg_cpu']:.1f}\\%" if e.get('hardware') else "N/A" for e in ordered_data]
        latex += f"CPU Avg & {' & '.join(vals)} \\\\\n"

        vals = [f"{e['hardware']['max_cpu']:.1f}\\%" if e.get('hardware') else "N/A" for e in ordered_data]
        latex += f"CPU Max & {' & '.join(vals)} \\\\\n"

        vals = [f"{e['hardware']['avg_memory']:.1f}\\%" if e.get('hardware') else "N/A" for e in ordered_data]
        latex += f"Memory Avg & {' & '.join(vals)} \\\\\n"

        vals = ["Nu" if not e.get('hardware', {}).get('throttled', False) else "Da" for e in ordered_data]
        latex += f"Thermal Throttling & {' & '.join(vals)} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\caption{Comparație completă a celor 4 experimente}
\label{tab:full_comparison_4exp}
\end{table}
"""

    with open(output_file, 'w') as f:
        f.write(latex)

    print(f"✅ LaTeX table saved to: {output_file}")


def save_json_summary(experiments_data, output_file='all_experiments_summary.json'):
    """Save complete data to JSON"""

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj

    clean_data = convert(experiments_data)

    with open(output_file, 'w') as f:
        json.dump(clean_data, f, indent=2, default=convert)

    print(f"✅ Complete data saved to: {output_file}")


def main():
    print("🔍 Extracting metrics from all 4 experiments...\n")

    # Experiment names
    experiment_names = [
        'baseline_fp32',
        'experiment2_regularized',
        'experiment3_fp16',
        'exp3_fp16_fixed'
    ]

    # Load all data
    experiments_data = []
    for exp_name in experiment_names:
        data = load_experiment_metrics(exp_name)
        experiments_data.append(data)

        if data['exists'] and data['metrics']:
            print(f"✅ Loaded: {exp_name}")
        elif data['exists']:
            print(f"⚠️  Found but incomplete: {exp_name}")
        else:
            print(f"❌ Not found: {exp_name}")

    print()

    # Generate comparison table
    generate_comparison_table(experiments_data)

    # Generate LaTeX table
    generate_latex_table(experiments_data)

    # Save complete JSON
    save_json_summary(experiments_data)

    print("\n" + "=" * 100)
    print("✨ SUMMARY")
    print("=" * 100)
    print()

    # Find best model
    completed = [e for e in experiments_data if e.get('metrics')]
    if completed:
        best = max(completed, key=lambda e: e['metrics']['best_val_acc'])
        print(f"🏆 Best Model: {best['name']}")
        print(f"   Accuracy: {best['metrics']['best_val_acc'] * 100:.2f}%")
        print(f"   Gap: {best['metrics']['gap'] * 100:.2f}%")
        print()

    # Check FP16 comparison
    fp16_exps = [e for e in experiments_data if 'fp16' in e['name'] and e.get('metrics')]
    if len(fp16_exps) == 2:
        old = next((e for e in fp16_exps if 'fixed' not in e['name']), None)
        fixed = next((e for e in fp16_exps if 'fixed' in e['name']), None)

        if old and fixed:
            print("🔄 FP16 Comparison (Old vs Fixed):")
            print(f"   Old:   {old['metrics']['best_val_acc'] * 100:.2f}% acc, {old['metrics']['gap'] * 100:.2f}% gap")
            print(
                f"   Fixed: {fixed['metrics']['best_val_acc'] * 100:.2f}% acc, {fixed['metrics']['gap'] * 100:.2f}% gap")

            acc_improvement = (fixed['metrics']['best_val_acc'] - old['metrics']['best_val_acc']) * 100
            gap_improvement = (old['metrics']['gap'] - fixed['metrics']['gap']) * 100

            print(f"   Δ Accuracy: {acc_improvement:+.2f}%")
            print(f"   Δ Gap: {gap_improvement:+.2f}% (reduction)")
            print()

    print("📁 Files generated:")
    print("   - metrics_comparison_4exp.tex (LaTeX table)")
    print("   - all_experiments_summary.json (Complete data)")
    print()
    print("🎯 Next: Use these files in your final report!")


if __name__ == '__main__':
    main()
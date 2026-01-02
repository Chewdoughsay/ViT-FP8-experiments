#!/usr/bin/env python3
"""
Script actualizat pentru extragerea completă a metricilor din experimentele ViT
Funcționează cu structura reală de fișiere din results/checkpoints/
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta


def load_experiment_data(exp_name, base_dir='results/checkpoints'):
    """Încarcă toate datele pentru un experiment"""
    exp_dir = Path(base_dir) / exp_name

    data = {
        'name': exp_name,
        'exists': exp_dir.exists(),
        'metrics': None,
        'hardware': None,
        'timing': None,
        'gpu_power': None
    }

    if not exp_dir.exists():
        print(f"⚠️  Experiment folder not found: {exp_dir}")
        return data

    # 1. Load final_metrics.json
    metrics_file = exp_dir / 'final_metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        data['metrics'] = {
            'train_loss': metrics.get('train_loss', []),
            'train_acc': metrics.get('train_acc', []),
            'val_loss': metrics.get('val_loss', []),
            'val_acc': metrics.get('val_acc', []),
            'best_val_acc': max(metrics.get('val_acc', [0])),
            'final_train_acc': metrics['train_acc'][-1] if metrics.get('train_acc') else 0,
            'final_val_acc': metrics['val_acc'][-1] if metrics.get('val_acc') else 0,
            'final_train_loss': metrics['train_loss'][-1] if metrics.get('train_loss') else 0,
            'final_val_loss': metrics['val_loss'][-1] if metrics.get('val_loss') else 0,
            'num_epochs': len(metrics.get('train_loss', []))
        }

        # Calculate overfitting gap
        if data['metrics']['final_train_acc'] and data['metrics']['final_val_acc']:
            data['metrics']['gap'] = data['metrics']['final_train_acc'] - data['metrics']['final_val_acc']

        # Timing statistics
        epoch_times = metrics.get('epoch_time', [])
        if epoch_times:
            data['timing'] = {
                'total_seconds': sum(epoch_times),
                'total_minutes': sum(epoch_times) / 60,
                'total_hours': sum(epoch_times) / 3600,
                'avg_epoch_seconds': np.mean(epoch_times),
                'min_epoch_seconds': min(epoch_times),
                'max_epoch_seconds': max(epoch_times),
                'std_epoch_seconds': np.std(epoch_times)
            }

    # 2. Load hardware_stats.json
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
                'min_cpu': np.min(cpu_vals),
                'std_cpu': np.std(cpu_vals),
                'avg_memory': np.mean(mem_vals) if mem_vals else 0,
                'max_memory': np.max(mem_vals) if mem_vals else 0,
                'max_thermal': np.max(thermal_vals) if thermal_vals else 0,
                'throttled': any(t > 0 for t in thermal_vals) if thermal_vals else False,
                'num_samples': len(cpu_vals)
            }

    return data


def analyze_gpu_power_log(csv_file='results/logs/full_night_run.csv'):
    """Analizează log-ul GPU power din CSV"""
    csv_path = Path(csv_file)

    if not csv_path.exists():
        print(f"⚠️  GPU log file not found: {csv_path}")
        return None

    # Load CSV
    df = pd.read_csv(csv_path)

    # Remove carriage returns if present
    df.columns = df.columns.str.strip()

    # Convert to numeric
    df['gpu_power_mW'] = pd.to_numeric(df['gpu_power_mW'], errors='coerce')
    df['cpu_power_mW'] = pd.to_numeric(df['cpu_power_mW'], errors='coerce')

    # Remove rows with NaN or zero values (idle periods)
    df_active = df[(df['gpu_power_mW'] > 100) & (df['cpu_power_mW'] > 100)]

    stats = {
        'total_measurements': len(df),
        'active_measurements': len(df_active),
        'duration_hours': len(df) / 3600,  # Assuming 1 sample/second
        # GPU Stats
        'gpu_avg_power_mW': df_active['gpu_power_mW'].mean(),
        'gpu_max_power_mW': df_active['gpu_power_mW'].max(),
        'gpu_min_power_mW': df_active['gpu_power_mW'].min(),
        'gpu_std_power_mW': df_active['gpu_power_mW'].std(),
        # CPU Stats
        'cpu_avg_power_mW': df_active['cpu_power_mW'].mean(),
        'cpu_max_power_mW': df_active['cpu_power_mW'].max(),
        'cpu_min_power_mW': df_active['cpu_power_mW'].min(),
        'cpu_std_power_mW': df_active['cpu_power_mW'].std(),
        # Total power
        'total_avg_power_mW': df_active['gpu_power_mW'].mean() + df_active['cpu_power_mW'].mean(),
        'total_avg_power_W': (df_active['gpu_power_mW'].mean() + df_active['cpu_power_mW'].mean()) / 1000,
    }

    # Energy estimation (if we know duration)
    if stats['total_measurements'] > 0:
        # Approximate total energy (Wh)
        avg_power_W = stats['total_avg_power_W']
        duration_h = stats['duration_hours']
        stats['estimated_energy_Wh'] = avg_power_W * duration_h

    return stats


def generate_summary_table(experiments_data):
    """Generează tabel rezumat cu toate metricile"""

    print("\n" + "=" * 100)
    print("REZUMAT COMPLET EXPERIMENTE")
    print("=" * 100)

    for data in experiments_data:
        print(f"\n{'=' * 100}")
        print(f"📊 {data['name'].upper().replace('_', ' ')}")
        print(f"{'=' * 100}")

        if not data['exists']:
            print("   ❌ Experiment folder not found")
            continue

        # Performance Metrics
        if data['metrics']:
            m = data['metrics']
            print(f"\n   🎯 PERFORMANCE METRICS:")
            print(f"      Epochs Trained: {m['num_epochs']}")
            print(f"      Best Val Accuracy: {m['best_val_acc'] * 100:.2f}%")
            print(f"      Final Train Accuracy: {m['final_train_acc'] * 100:.2f}%")
            print(f"      Final Val Accuracy: {m['final_val_acc'] * 100:.2f}%")
            print(f"      Overfitting Gap: {m.get('gap', 0) * 100:.2f}%")
            print(f"      Final Train Loss: {m['final_train_loss']:.4f}")
            print(f"      Final Val Loss: {m['final_val_loss']:.4f}")

        # Timing
        if data['timing']:
            t = data['timing']
            print(f"\n   ⏱️  TIMING:")
            print(f"      Total Training Time: {t['total_hours']:.2f}h ({t['total_minutes']:.1f} min)")
            print(f"      Avg Time/Epoch: {t['avg_epoch_seconds']:.1f}s")
            print(f"      Min Time/Epoch: {t['min_epoch_seconds']:.1f}s")
            print(f"      Max Time/Epoch: {t['max_epoch_seconds']:.1f}s")
            print(f"      Std Dev: ±{t['std_epoch_seconds']:.1f}s")

        # Hardware
        if data['hardware']:
            h = data['hardware']
            print(f"\n   💻 HARDWARE UTILIZATION:")
            print(f"      Avg CPU: {h['avg_cpu']:.1f}% (Max: {h['max_cpu']:.1f}%)")
            print(f"      Avg Memory: {h['avg_memory']:.1f}% (Max: {h['max_memory']:.1f}%)")
            print(f"      Thermal Pressure: {'YES ⚠️ ' if h['throttled'] else 'NO ✅'}")
            if h['throttled']:
                print(f"      Max Thermal Level: {h['max_thermal']}")
            print(f"      Monitoring Samples: {h['num_samples']}")

    print(f"\n{'=' * 100}\n")


def generate_latex_comparison_table(experiments_data, output_file='metrics_comparison.tex'):
    """Generează tabel LaTeX pentru comparație"""

    latex = r"""\begin{table}[H]
\centering
\small
\begin{tabular}{lrrr}
\toprule
\textbf{Metrică} & \textbf{Baseline FP32} & \textbf{Regularized FP32} & \textbf{Mixed Precision FP16} \\
\midrule
"""

    # Map experiment names to column order
    exp_map = {
        'baseline_fp32': 0,
        'experiment2_regularized': 1,
        'experiment3_fp16': 2
    }

    # Sort experiments by position
    sorted_data = [None, None, None]
    for exp in experiments_data:
        idx = exp_map.get(exp['name'])
        if idx is not None:
            sorted_data[idx] = exp

    # Performance metrics
    latex += r"\multicolumn{4}{l}{\textit{\textbf{Performanță Model}}} \\" + "\n"

    metrics_rows = [
        ('Epoci Antrenate', lambda e: e['metrics']['num_epochs'], '{}'),
        ('Best Val Accuracy', lambda e: e['metrics']['best_val_acc'] * 100, '{:.2f}\\%'),
        ('Final Val Accuracy', lambda e: e['metrics']['final_val_acc'] * 100, '{:.2f}\\%'),
        ('Final Train Accuracy', lambda e: e['metrics']['final_train_acc'] * 100, '{:.2f}\\%'),
        ('Overfitting Gap', lambda e: e['metrics'].get('gap', 0) * 100, '{:.2f}\\%'),
        ('Final Val Loss', lambda e: e['metrics']['final_val_loss'], '{:.4f}'),
    ]

    for label, getter, fmt in metrics_rows:
        values = []
        for exp in sorted_data:
            if exp and exp['exists'] and exp['metrics']:
                val = getter(exp)
                values.append(fmt.format(val))
            else:
                values.append("N/A")
        latex += f"{label} & {' & '.join(values)} \\\\\n"

    # Timing
    latex += r"\midrule" + "\n"
    latex += r"\multicolumn{4}{l}{\textit{\textbf{Timp Antrenare}}} \\" + "\n"

    timing_rows = [
        ('Timp Total (ore)', lambda e: e['timing']['total_hours'], '{:.2f}h'),
        ('Timp/Epocă (sec)', lambda e: e['timing']['avg_epoch_seconds'], '{:.1f}s'),
        ('Timp/Epocă StdDev', lambda e: e['timing']['std_epoch_seconds'], '±{:.1f}s'),
    ]

    for label, getter, fmt in timing_rows:
        values = []
        for exp in sorted_data:
            if exp and exp['exists'] and exp['timing']:
                val = getter(exp)
                values.append(fmt.format(val))
            else:
                values.append("N/A")
        latex += f"{label} & {' & '.join(values)} \\\\\n"

    # Hardware
    latex += r"\midrule" + "\n"
    latex += r"\multicolumn{4}{l}{\textit{\textbf{Utilizare Hardware}}} \\" + "\n"

    hw_rows = [
        ('CPU Avg', lambda e: e['hardware']['avg_cpu'], '{:.1f}\\%'),
        ('CPU Max', lambda e: e['hardware']['max_cpu'], '{:.1f}\\%'),
        ('Memory Avg', lambda e: e['hardware']['avg_memory'], '{:.1f}\\%'),
        ('Memory Max', lambda e: e['hardware']['max_memory'], '{:.1f}\\%'),
        ('Thermal Throttling', lambda e: "Da" if e['hardware']['throttled'] else "Nu", '{}'),
    ]

    for label, getter, fmt in hw_rows:
        values = []
        for exp in sorted_data:
            if exp and exp['exists'] and exp['hardware']:
                val = getter(exp)
                values.append(fmt.format(val) if '{}' in fmt else fmt.format(val))
            else:
                values.append("N/A")
        latex += f"{label} & {' & '.join(values)} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\caption{Comparație completă a celor 3 experimente}
\label{tab:full_comparison}
\end{table}
"""

    with open(output_file, 'w') as f:
        f.write(latex)

    print(f"✅ LaTeX table saved to: {output_file}")
    return latex


def main():
    print("🔍 Extragere metrici din checkpoint-uri și log-uri...\n")

    # Experiment names
    experiments = [
        'baseline_fp32',
        'experiment2_regularized',
        'experiment3_fp16'
    ]

    # Load all experiment data
    experiments_data = []
    for exp_name in experiments:
        data = load_experiment_data(exp_name)
        experiments_data.append(data)

    # Display summary
    generate_summary_table(experiments_data)

    # Generate LaTeX table
    generate_latex_comparison_table(experiments_data, '../results/metrics/metrics_comparison_full.tex')

    # Analyze GPU power if available
    print("\n📊 Analizare log GPU power...")
    gpu_stats = analyze_gpu_power_log()

    if gpu_stats:
        print("\n" + "=" * 80)
        print("GPU/CPU POWER ANALYSIS (full_night_run.csv)")
        print("=" * 80)
        print(f"  Total Measurements: {gpu_stats['total_measurements']:,}")
        print(f"  Active Measurements: {gpu_stats['active_measurements']:,}")
        print(f"  Duration: {gpu_stats['duration_hours']:.2f} hours")
        print(f"\n  GPU Power:")
        print(f"    Average: {gpu_stats['gpu_avg_power_mW']:.0f} mW ({gpu_stats['gpu_avg_power_mW'] / 1000:.2f} W)")
        print(f"    Max: {gpu_stats['gpu_max_power_mW']:.0f} mW ({gpu_stats['gpu_max_power_mW'] / 1000:.2f} W)")
        print(f"    StdDev: ±{gpu_stats['gpu_std_power_mW']:.0f} mW")
        print(f"\n  CPU Power:")
        print(f"    Average: {gpu_stats['cpu_avg_power_mW']:.0f} mW ({gpu_stats['cpu_avg_power_mW'] / 1000:.2f} W)")
        print(f"    Max: {gpu_stats['cpu_max_power_mW']:.0f} mW ({gpu_stats['cpu_max_power_mW'] / 1000:.2f} W)")
        print(f"    StdDev: ±{gpu_stats['cpu_std_power_mW']:.0f} mW")
        print(f"\n  Total System Power:")
        print(f"    Average: {gpu_stats['total_avg_power_W']:.2f} W")
        print(f"    Estimated Energy Consumption: {gpu_stats['estimated_energy_Wh']:.2f} Wh")
        print("=" * 80 + "\n")

        # Save GPU stats to JSON
        with open('../results/metrics/gpu_power_analysis.json', 'w') as f:
            json.dump(gpu_stats, f, indent=2)
        print("✅ GPU power analysis saved to: gpu_power_analysis.json")

    # Save all data to JSON
    with open('../results/metrics/all_experiments_data.json', 'w') as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(experiments_data, f, indent=2, default=convert)

    print("✅ Complete experiment data saved to: all_experiments_data.json")

    print("\n" + "=" * 80)
    print("💡 NEXT STEPS:")
    print("=" * 80)
    print("1. Copiază conținutul din 'metrics_comparison_full.tex' în raportul LaTeX")
    print("2. Adaugă secțiunea cu GPU power analysis în raport")
    print("3. Verifică toate cifrele și recompilează PDF-ul")
    print("4. Fișierele JSON conțin toate datele pentru referință ulterioară")
    print("=" * 80)


if __name__ == '__main__':
    main()
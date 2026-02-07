#!/usr/bin/env python3
"""
Auto-Discovery Plot Generator for ViT Experiments
==================================================
Scanează automat results/checkpoints/ și generează ploturi din datele găsite.

Usage (din folderul scripts/):
    python generate_plots.py
    python generate_plots.py --base-dir ../results/checkpoints --output-dir ../results/plots_v2
    python generate_plots.py --scan  # doar listează ce găsește

Structura așteptată:
    results/checkpoints/
    ├── experiment_name_1/
    │   ├── final_metrics.json      # obligatoriu pentru ploturi
    │   ├── hardware_stats.json     # opțional
    │   └── best_model.pt           # opțional
    ├── experiment_name_2/
    │   └── ...
    └── exp4_fp8_test/
        └── fp8_test_results.json   # pentru analiza FP8
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# =============================================================================
# LABEL MAPPING - Editează aici pentru a schimba numele afișate
# =============================================================================

# Mapare: nume_folder -> (label afișat, label scurt pentru bar charts, ordine)
# Ordinea determină poziția în legendă și bar charts
LABEL_MAPPING = {
    # FP32 experiments
    'baseline_fp32': ('Baseline FP32', 'Base FP32', 1),
    'experiment2_regularized': ('Augmented FP32', 'Aug FP32', 2),

    # FP16 experiments
    'experiment3_fp16': ('Baseline FP16', 'Base FP16', 3),
    'exp3_fp16_fixed': ('Augmented FP16', 'Aug FP16', 4),

    # Adaugă aici alte experimente dacă e nevoie
    # 'nume_folder': ('Label Lung', 'Label Scurt', ordine),
}

# Culori fixe per experiment (pentru consistență între grafice)
COLOR_MAPPING = {
    'baseline_fp32': '#e74c3c',  # Roșu
    'experiment2_regularized': '#2ecc71',  # Verde
    'experiment3_fp16': '#e67e22',  # Portocaliu
    'exp3_fp16_fixed': '#3498db',  # Albastru
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExperimentMetrics:
    """Metrici extrase din final_metrics.json"""
    name: str
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    epoch_time: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)

    @property
    def num_epochs(self) -> int:
        return len(self.val_acc)

    @property
    def best_val_acc(self) -> float:
        return max(self.val_acc) if self.val_acc else 0.0

    @property
    def final_val_acc(self) -> float:
        return self.val_acc[-1] if self.val_acc else 0.0

    @property
    def final_train_acc(self) -> float:
        return self.train_acc[-1] if self.train_acc else 0.0

    @property
    def overfitting_gap(self) -> float:
        return self.final_train_acc - self.final_val_acc

    @property
    def total_time_hours(self) -> float:
        return sum(self.epoch_time) / 3600 if self.epoch_time else 0.0

    @property
    def avg_epoch_time(self) -> float:
        return np.mean(self.epoch_time) if self.epoch_time else 0.0


@dataclass
class HardwareStats:
    """Statistici hardware din hardware_stats.json"""
    cpu_percent: List[float] = field(default_factory=list)
    memory_percent: List[float] = field(default_factory=list)
    thermal_pressure: List[float] = field(default_factory=list)

    @property
    def avg_cpu(self) -> float:
        return np.mean(self.cpu_percent) if self.cpu_percent else 0.0

    @property
    def max_cpu(self) -> float:
        return max(self.cpu_percent) if self.cpu_percent else 0.0

    @property
    def avg_memory(self) -> float:
        return np.mean(self.memory_percent) if self.memory_percent else 0.0

    @property
    def max_memory(self) -> float:
        return max(self.memory_percent) if self.memory_percent else 0.0

    @property
    def throttled(self) -> bool:
        return any(t > 0 for t in self.thermal_pressure) if self.thermal_pressure else False


@dataclass
class Experiment:
    """Container complet pentru un experiment"""
    name: str
    folder: Path
    metrics: Optional[ExperimentMetrics] = None
    hardware: Optional[HardwareStats] = None

    # Display settings (auto-assigned)
    label: str = ""
    short_label: str = ""  # Pentru bar charts și spații compacte
    color: str = "#333333"
    linestyle: str = "-"
    marker: str = "o"


@dataclass
class FP8Results:
    """Rezultate test FP8 quantization"""
    original_accuracy: float = 0.0
    original_loss: float = 0.0
    fp8_accuracy: float = 0.0
    fp8_loss: float = 0.0
    accuracy_degradation: float = 0.0
    loss_increase: float = 0.0


# =============================================================================
# DATA LOADING
# =============================================================================

def load_json_safe(filepath: Path) -> Optional[Dict]:
    """Încarcă JSON cu handling de erori"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  ⚠️  Nu pot citi {filepath.name}: {e}")
        return None


def load_experiment(folder: Path) -> Optional[Experiment]:
    """Încarcă un experiment din folder"""
    exp = Experiment(name=folder.name, folder=folder)

    # 1. Load metrics (obligatoriu)
    metrics_file = folder / 'final_metrics.json'
    if not metrics_file.exists():
        return None

    data = load_json_safe(metrics_file)
    if data is None:
        return None

    exp.metrics = ExperimentMetrics(
        name=folder.name,
        train_loss=data.get('train_loss', []),
        train_acc=data.get('train_acc', []),
        val_loss=data.get('val_loss', []),
        val_acc=data.get('val_acc', []),
        epoch_time=data.get('epoch_time', []),
        learning_rates=data.get('learning_rates', [])
    )

    # 2. Load hardware stats (opțional)
    hw_file = folder / 'hardware_stats.json'
    if hw_file.exists():
        hw_data = load_json_safe(hw_file)
        if hw_data:
            exp.hardware = HardwareStats(
                cpu_percent=hw_data.get('cpu_percent', []),
                memory_percent=hw_data.get('memory_percent', []),
                thermal_pressure=hw_data.get('thermal_pressure', [])
            )

    return exp


def load_fp8_results(base_dir: Path) -> Optional[FP8Results]:
    """Caută și încarcă rezultatele FP8 din orice folder"""
    # Caută în toate subfolderele
    for folder in base_dir.iterdir():
        if not folder.is_dir():
            continue

        fp8_file = folder / 'fp8_test_results.json'
        if fp8_file.exists():
            data = load_json_safe(fp8_file)
            if data:
                return FP8Results(
                    original_accuracy=data.get('original_fp16', {}).get('accuracy', 0),
                    original_loss=data.get('original_fp16', {}).get('loss', 0),
                    fp8_accuracy=data.get('fp8_simulated', {}).get('accuracy', 0),
                    fp8_loss=data.get('fp8_simulated', {}).get('loss', 0),
                    accuracy_degradation=data.get('degradation', {}).get('accuracy_loss_pp', 0),
                    loss_increase=data.get('degradation', {}).get('loss_increase', 0)
                )

    return None


def discover_experiments(base_dir: Path) -> List[Experiment]:
    """Descoperă automat toate experimentele din directorul dat"""
    experiments = []

    if not base_dir.exists():
        print(f"❌ Directorul nu există: {base_dir}")
        return experiments

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue

        # Skip foldere care nu au metrics
        metrics_file = folder / 'final_metrics.json'
        if not metrics_file.exists():
            # Verifică dacă e folder FP8
            if (folder / 'fp8_test_results.json').exists():
                print(f"  📦 {folder.name}/ (FP8 test results)")
            continue

        exp = load_experiment(folder)
        if exp and exp.metrics and exp.metrics.num_epochs > 0:
            experiments.append(exp)
            hw_status = "✓" if exp.hardware else "✗"
            print(f"  ✓ {folder.name}/ ({exp.metrics.num_epochs} epochs, hw:{hw_status})")

    return experiments


def assign_display_properties(experiments: List[Experiment]):
    """Asignează culori și stiluri bazat pe mapping-ul definit"""
    # Fallback colors pentru experimente necunoscute
    fallback_colors = ['#9b59b6', '#1abc9c', '#f39c12', '#34495e', '#95a5a6']
    markers = ['o', 's', '^', 'd', 'v', 'p', 'h', '*']

    # Detectare pattern-uri în nume pentru linestyle
    fp16_keywords = ['fp16', 'mixed', 'half']

    for i, exp in enumerate(experiments):
        name_lower = exp.name.lower()

        # 1. Labels din mapping sau fallback
        if exp.name in LABEL_MAPPING:
            mapping = LABEL_MAPPING[exp.name]
            exp.label = mapping[0]  # Label lung
            exp.short_label = mapping[1]  # Label scurt
        else:
            # Fallback: convertește numele folderului
            exp.label = exp.name.replace('_', ' ').replace('-', ' ').title()
            exp.short_label = exp.label[:10] if len(exp.label) > 10 else exp.label

        # 2. Culoare din mapping sau fallback
        if exp.name in COLOR_MAPPING:
            exp.color = COLOR_MAPPING[exp.name]
        else:
            exp.color = fallback_colors[i % len(fallback_colors)]

        # 3. Marker
        exp.marker = markers[i % len(markers)]

        # 4. Linestyle bazat pe precizie
        if any(kw in name_lower for kw in fp16_keywords):
            exp.linestyle = '--'
        else:
            exp.linestyle = '-'


def sort_experiments(experiments: List[Experiment]) -> List[Experiment]:
    """Sortează experimentele după ordinea definită în LABEL_MAPPING"""

    def get_order(exp):
        if exp.name in LABEL_MAPPING:
            return LABEL_MAPPING[exp.name][2]  # Index 2 pentru ordine
        return 999  # Pune experimentele necunoscute la final

    return sorted(experiments, key=get_order)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def setup_plotting():
    """Configurează matplotlib"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10


def plot_val_accuracy(experiments: List[Experiment], output_dir: Path):
    """Grafic principal: Validation Accuracy over epochs"""
    fig, ax = plt.subplots(figsize=(12, 7))

    for exp in experiments:
        if not exp.metrics:
            continue

        val_acc = [v * 100 for v in exp.metrics.val_acc]
        epochs = range(1, len(val_acc) + 1)

        ax.plot(epochs, val_acc,
                label=f"{exp.label} ({exp.metrics.best_val_acc * 100:.1f}%)",
                color=exp.color,
                linestyle=exp.linestyle,
                linewidth=2.5,
                marker=exp.marker,
                markevery=max(1, len(val_acc) // 10),
                markersize=6)

    ax.set_xlabel('Epoca')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Evoluția Acurateții pe Setul de Validare')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Auto-adjust y limits
    all_accs = [v * 100 for exp in experiments if exp.metrics for v in exp.metrics.val_acc]
    if all_accs:
        ax.set_ylim([max(0, min(all_accs) - 10), min(100, max(all_accs) + 5)])

    plt.tight_layout()
    save_path = output_dir / '01_val_accuracy.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"  ✓ {save_path.name}")
    plt.close()


def plot_train_val_comparison(experiments: List[Experiment], output_dir: Path):
    """Grid de grafice Train vs Val pentru fiecare experiment"""
    n = len(experiments)
    if n == 0:
        return

    # Calculează grid dimensions - forțează 2x2 pentru 4 experimente
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.5 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, exp in enumerate(experiments):
        if not exp.metrics:
            continue

        ax = axes[idx]

        train_acc = [v * 100 for v in exp.metrics.train_acc]
        val_acc = [v * 100 for v in exp.metrics.val_acc]
        epochs = range(1, len(train_acc) + 1)

        ax.plot(epochs, train_acc, label='Train', color=exp.color,
                linestyle='--', linewidth=2, alpha=0.7)
        ax.plot(epochs, val_acc, label='Validation', color=exp.color,
                linestyle='-', linewidth=2.5)

        # Fill gap area
        ax.fill_between(epochs, train_acc, val_acc, alpha=0.15, color=exp.color)

        # Show gap value - MUTAT în stânga jos pentru a evita overlap cu legenda
        gap = exp.metrics.overfitting_gap * 100
        ax.text(0.03, 0.03, f'Gap: {gap:.1f}%',
                transform=ax.transAxes, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'),
                fontsize=10, fontweight='bold')

        # Titlu cu label-ul experimentului
        ax.set_title(exp.label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoca')
        ax.set_ylabel('Accuracy (%)')

        # Legenda în dreapta sus pentru a nu se suprapune cu gap
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Fix y limits pentru consistență
        ax.set_ylim([20, 100])
        ax.set_xlim([1, max(epochs)])

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Analiza Overfitting: Train vs Validation', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_path = output_dir / '02_overfitting_analysis.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"  ✓ {save_path.name}")
    plt.close()


def plot_loss_curves(experiments: List[Experiment], output_dir: Path):
    """Train și Val Loss pentru toate experimentele"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for exp in experiments:
        if not exp.metrics:
            continue

        epochs = range(1, exp.metrics.num_epochs + 1)

        # Train loss
        ax1.plot(epochs, exp.metrics.train_loss,
                 label=exp.label, color=exp.color,
                 linestyle=exp.linestyle, linewidth=2)

        # Val loss
        ax2.plot(epochs, exp.metrics.val_loss,
                 label=exp.label, color=exp.color,
                 linestyle=exp.linestyle, linewidth=2)

    ax1.set_xlabel('Epoca')
    ax1.set_ylabel('Train Loss')
    ax1.set_title('Training Loss')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoca')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / '03_loss_curves.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"  ✓ {save_path.name}")
    plt.close()


def plot_summary_bars(experiments: List[Experiment], output_dir: Path):
    """Bar charts cu metrici sumare"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Folosim short_label pentru bar charts
    labels = [exp.short_label for exp in experiments if exp.metrics]
    colors = [exp.color for exp in experiments if exp.metrics]

    x = np.arange(len(labels))
    width = 0.6

    # 1. Best Validation Accuracy
    best_accs = [exp.metrics.best_val_acc * 100 for exp in experiments if exp.metrics]
    bars = axes[0].bar(x, best_accs, width, color=colors, edgecolor='black', linewidth=1)
    axes[0].set_ylabel('Best Val Accuracy (%)')
    axes[0].set_title('Best Validation Accuracy', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    axes[0].grid(True, axis='y', alpha=0.3)

    # Auto y-limit
    if best_accs:
        axes[0].set_ylim([max(0, min(best_accs) - 10), min(100, max(best_accs) + 5)])

    for bar, val in zip(bars, best_accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 2. Overfitting Gap
    gaps = [exp.metrics.overfitting_gap * 100 for exp in experiments if exp.metrics]
    bars = axes[1].bar(x, gaps, width, color=colors, edgecolor='black', linewidth=1)
    axes[1].set_ylabel('Overfitting Gap (%)')
    axes[1].set_title('Overfitting Gap (Train - Val)', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    axes[1].axhline(y=5, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Ideal (<5%)')
    axes[1].grid(True, axis='y', alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=8)

    for bar, val in zip(bars, gaps):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 3. Training Time
    times = [exp.metrics.total_time_hours for exp in experiments if exp.metrics]
    bars = axes[2].bar(x, times, width, color=colors, edgecolor='black', linewidth=1)
    axes[2].set_ylabel('Total Time (hours)')
    axes[2].set_title('Training Time', fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    axes[2].grid(True, axis='y', alpha=0.3)

    for bar, val in zip(bars, times):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{val:.1f}h', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = output_dir / '04_summary_bars.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"  ✓ {save_path.name}")
    plt.close()


def plot_hardware_utilization(experiments: List[Experiment], output_dir: Path):
    """Hardware utilization (CPU, Memory)"""
    # Filtrează doar experimente cu date hardware
    hw_experiments = [exp for exp in experiments if exp.hardware]

    if not hw_experiments:
        print("  ⚠️  Nu există date hardware pentru plotare")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    labels = [exp.short_label for exp in hw_experiments]
    colors = [exp.color for exp in hw_experiments]

    x = np.arange(len(labels))
    width = 0.35

    # CPU
    cpu_avg = [exp.hardware.avg_cpu for exp in hw_experiments]
    cpu_max = [exp.hardware.max_cpu for exp in hw_experiments]

    ax1.bar(x - width / 2, cpu_avg, width, label='Average', color=colors, alpha=0.7, edgecolor='black')
    ax1.bar(x + width / 2, cpu_max, width, label='Maximum', color=colors, edgecolor='black', hatch='//')
    ax1.set_ylabel('CPU Utilization (%)')
    ax1.set_title('CPU Usage', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax1.legend(loc='upper right')
    ax1.grid(True, axis='y', alpha=0.3)

    # Memory
    mem_avg = [exp.hardware.avg_memory for exp in hw_experiments]
    mem_max = [exp.hardware.max_memory for exp in hw_experiments]

    ax2.bar(x - width / 2, mem_avg, width, label='Average', color=colors, alpha=0.7, edgecolor='black')
    ax2.bar(x + width / 2, mem_max, width, label='Maximum', color=colors, edgecolor='black', hatch='//')
    ax2.set_ylabel('Memory Utilization (%)')
    ax2.set_title('Memory Usage', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax2.legend(loc='lower right')
    ax2.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Hardware Utilization', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = output_dir / '05_hardware_utilization.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"  ✓ {save_path.name}")
    plt.close()


def plot_fp8_analysis(fp8_results: FP8Results, output_dir: Path):
    """FP8 Quantization impact analysis"""
    if not fp8_results:
        print("  ⚠️  Nu există date FP8 pentru plotare")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    models = ['Original\nFP16', 'FP8\n(E4M3)']
    colors = ['#2ecc71', '#e74c3c']

    # Accuracy
    accuracies = [fp8_results.original_accuracy, fp8_results.fp8_accuracy]
    bars1 = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Impact FP8 pe Accuracy', fontweight='bold')

    # Auto y-limits
    y_min = min(accuracies) - 5
    y_max = max(accuracies) + 3
    ax1.set_ylim([y_min, y_max])
    ax1.grid(True, axis='y', alpha=0.3)

    for bar, val in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Degradation annotation
    deg = fp8_results.accuracy_degradation
    ax1.annotate(f'-{deg:.2f}%', xy=(0.5, (accuracies[0] + accuracies[1]) / 2),
                 fontsize=14, color='red', fontweight='bold', ha='center',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Loss
    losses = [fp8_results.original_loss, fp8_results.fp8_loss]
    bars2 = ax2.bar(models, losses, color=colors, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Loss')
    ax2.set_title('Impact FP8 pe Loss', fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)

    for bar, val in zip(bars2, losses):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Loss increase annotation
    inc = fp8_results.loss_increase
    ax2.annotate(f'+{inc:.4f}', xy=(0.5, (losses[0] + losses[1]) / 2),
                 fontsize=12, color='red', fontweight='bold', ha='center',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Evaluare FP8 Post-Training Quantization', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = output_dir / '06_fp8_quantization.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"  ✓ {save_path.name}")
    plt.close()


def plot_comprehensive_summary(experiments: List[Experiment], fp8_results: Optional[FP8Results], output_dir: Path):
    """Grafic rezumat comprehensiv (poster-style)"""
    n_exp = len(experiments)
    has_fp8 = fp8_results is not None

    fig = plt.figure(figsize=(16, 10))

    # Layout: 2 rows, 3 columns
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # 1. Main accuracy plot (top, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    for exp in experiments:
        if not exp.metrics:
            continue
        val_acc = [v * 100 for v in exp.metrics.val_acc]
        epochs = range(1, len(val_acc) + 1)
        ax1.plot(epochs, val_acc, label=exp.label, color=exp.color,
                 linewidth=2, linestyle=exp.linestyle)
    ax1.set_xlabel('Epoca')
    ax1.set_ylabel('Val Accuracy (%)')
    ax1.set_title('Evoluția Accuracy', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. FP8 or additional info (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    if has_fp8:
        accs = [fp8_results.original_accuracy, fp8_results.fp8_accuracy]
        bars = ax2.bar(['FP16', 'FP8'], accs, color=['#2ecc71', '#e74c3c'], edgecolor='black')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('FP8 Quantization', fontweight='bold')
        y_min = min(accs) - 5
        ax2.set_ylim([y_min, max(accs) + 3])
        for bar, val in zip(bars, accs):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
        ax2.text(0.5, 0.5, f'-{fp8_results.accuracy_degradation:.2f}%',
                 transform=ax2.transAxes, ha='center', va='center',
                 fontsize=14, color='red', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No FP8 data', transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('FP8 Quantization', fontweight='bold')

    # 3. Overfitting gap (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    labels = [exp.short_label for exp in experiments if exp.metrics]
    gaps = [exp.metrics.overfitting_gap * 100 for exp in experiments if exp.metrics]
    colors = [exp.color for exp in experiments if exp.metrics]
    bars = ax3.bar(labels, gaps, color=colors, edgecolor='black')
    ax3.axhline(y=5, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax3.set_ylabel('Gap (%)')
    ax3.set_title('Overfitting Gap', fontweight='bold')
    ax3.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, gaps):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')

    # 4. Training time (bottom middle)
    ax4 = fig.add_subplot(gs[1, 1])
    times = [exp.metrics.total_time_hours for exp in experiments if exp.metrics]
    bars = ax4.bar(labels, times, color=colors, edgecolor='black')
    ax4.set_ylabel('Time (h)')
    ax4.set_title('Training Time', fontweight='bold')
    ax4.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, times):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{val:.1f}h', ha='center', fontsize=8, fontweight='bold')

    # 5. Best accuracy (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    best_accs = [exp.metrics.best_val_acc * 100 for exp in experiments if exp.metrics]
    bars = ax5.bar(labels, best_accs, color=colors, edgecolor='black')
    ax5.set_ylabel('Accuracy (%)')
    ax5.set_title('Best Val Accuracy', fontweight='bold')
    ax5.tick_params(axis='x', rotation=30)
    if best_accs:
        ax5.set_ylim([max(0, min(best_accs) - 8), min(100, max(best_accs) + 5)])
    for bar, val in zip(bars, best_accs):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')

    plt.suptitle('Rezumat Experimente ViT', fontsize=16, fontweight='bold', y=0.98)
    save_path = output_dir / '07_comprehensive_summary.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"  ✓ {save_path.name}")
    plt.close()


def export_summary_json(experiments: List[Experiment], fp8_results: Optional[FP8Results], output_dir: Path):
    """Exportă rezumatul în JSON pentru referință"""
    summary = {
        'experiments': [],
        'fp8_results': None
    }

    for exp in experiments:
        if not exp.metrics:
            continue

        exp_data = {
            'name': exp.name,
            'num_epochs': exp.metrics.num_epochs,
            'best_val_acc': exp.metrics.best_val_acc,
            'final_val_acc': exp.metrics.final_val_acc,
            'final_train_acc': exp.metrics.final_train_acc,
            'overfitting_gap': exp.metrics.overfitting_gap,
            'total_time_hours': exp.metrics.total_time_hours,
            'avg_epoch_seconds': exp.metrics.avg_epoch_time
        }

        if exp.hardware:
            exp_data['hardware'] = {
                'avg_cpu': exp.hardware.avg_cpu,
                'max_cpu': exp.hardware.max_cpu,
                'avg_memory': exp.hardware.avg_memory,
                'max_memory': exp.hardware.max_memory,
                'throttled': exp.hardware.throttled
            }

        summary['experiments'].append(exp_data)

    if fp8_results:
        summary['fp8_results'] = {
            'original_accuracy': fp8_results.original_accuracy,
            'fp8_accuracy': fp8_results.fp8_accuracy,
            'accuracy_degradation': fp8_results.accuracy_degradation,
            'original_loss': fp8_results.original_loss,
            'fp8_loss': fp8_results.fp8_loss,
            'loss_increase': fp8_results.loss_increase
        }

    save_path = output_dir / 'experiments_summary.json'
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ {save_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Auto-generate plots from ViT experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (rulat din scripts/):
    python generate_plots.py
    python generate_plots.py --base-dir ../results/checkpoints --output-dir ../results/plots_v2
    python generate_plots.py --scan
        """
    )
    parser.add_argument('--base-dir', type=str, default='../results/checkpoints',
                        help='Directory containing experiment folders (default: ../results/checkpoints)')
    parser.add_argument('--output-dir', type=str, default='../results/plots_v2',
                        help='Directory for output plots (default: ../results/plots_v2)')
    parser.add_argument('--scan', action='store_true',
                        help='Only scan and list experiments, do not generate plots')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("AUTO-DISCOVERY PLOT GENERATOR")
    print("=" * 70)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Discover experiments
    print("Scanning for experiments...")
    experiments = discover_experiments(base_dir)

    if not experiments:
        print(f"\n❌ Nu s-au găsit experimente în {base_dir}")
        print("   Asigură-te că există foldere cu final_metrics.json")
        return 1

    print(f"\n✓ Găsite {len(experiments)} experimente")

    # Load FP8 results
    fp8_results = load_fp8_results(base_dir)
    if fp8_results:
        print(f"✓ Găsite rezultate FP8 (degradare: -{fp8_results.accuracy_degradation:.2f}%)")

    if args.scan:
        print("\n[Mod scanare - nu se generează grafice]")
        return 0

    # Setup
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_plotting()
    assign_display_properties(experiments)

    # Sortează experimentele în ordinea din LABEL_MAPPING
    experiments = sort_experiments(experiments)
    print(f"   Ordine: {', '.join(exp.label for exp in experiments)}")

    # Generate plots
    print("\nGenerare grafice...")
    print("-" * 40)

    plot_val_accuracy(experiments, output_dir)
    plot_train_val_comparison(experiments, output_dir)
    plot_loss_curves(experiments, output_dir)
    plot_summary_bars(experiments, output_dir)
    plot_hardware_utilization(experiments, output_dir)

    if fp8_results:
        plot_fp8_analysis(fp8_results, output_dir)

    plot_comprehensive_summary(experiments, fp8_results, output_dir)
    export_summary_json(experiments, fp8_results, output_dir)

    print("-" * 40)
    print(f"\n✅ Toate fișierele salvate în: {output_dir}")

    # List generated files
    print("\nFișiere generate:")
    for f in sorted(output_dir.glob('*')):
        size_kb = f.stat().st_size / 1024
        print(f"  {'📊' if f.suffix == '.png' else '📄'} {f.name} ({size_kb:.1f} KB)")

    return 0


if __name__ == '__main__':
    exit(main())
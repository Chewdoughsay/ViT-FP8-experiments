import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configurare stil grafice
plt.style.use('ggplot')
COLORS = {
    'baseline_fp32': '#e74c3c',  # Rosu (Pericol/Overfitting)
    'experiment2_regularized': '#2ecc71',  # Verde (Succes)
    'experiment3_fp16': '#3498db'  # Albastru (Tehnologie/Nou)
}
LABELS = {
    'baseline_fp32': 'Exp 1: Baseline (FP32)',
    'experiment2_regularized': 'Exp 2: Regularized (FP32)',
    'experiment3_fp16': 'Exp 3: Mixed Precision (FP16)'
}


def load_metrics(exp_name, base_dir='experiments/results/checkpoints'):
    path = Path(base_dir) / exp_name / 'final_metrics.json'
    if not path.exists():
        print(f"⚠️ Nu am gasit date pentru {exp_name}")
        return None

    with open(path, 'r') as f:
        return json.load(f)


def plot_comparison_metric(experiments_data, metric_key, title, ylabel, filename):
    plt.figure(figsize=(10, 6))

    for exp_name, data in experiments_data.items():
        if not data: continue

        values = data.get(metric_key, [])
        epochs = range(1, len(values) + 1)

        # Plotare linie
        plt.plot(epochs, values, label=LABELS.get(exp_name, exp_name),
                 color=COLORS.get(exp_name), linewidth=2)

        # Marcare punct final/maxim
        if 'acc' in metric_key:
            best_val = max(values)
            best_idx = values.index(best_val)
            plt.plot(best_idx + 1, best_val, 'o', color=COLORS.get(exp_name))
            plt.annotate(f'{best_val:.1%}', (best_idx + 1, best_val),
                         textcoords="offset points", xytext=(0, 10), ha='center')

    plt.title(title, fontsize=14)
    plt.xlabel('Epoci')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = Path('results/plots') / filename
    plt.savefig(save_path, dpi=300)
    print(f"📈 Grafic salvat: {save_path}")
    plt.close()


def plot_individual_dashboard(exp_name, data):
    if not data: return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    epochs = range(1, len(data['train_loss']) + 1)

    # 1. Loss Chart
    ax1.plot(epochs, data['train_loss'], label='Train Loss', color='orange', linestyle='--')
    ax1.plot(epochs, data['val_loss'], label='Val Loss', color='red')
    ax1.set_title(f'{LABELS[exp_name]} - Loss Curve')
    ax1.set_xlabel('Epoci')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy Chart
    ax2.plot(epochs, data['train_acc'], label='Train Acc', color='lightblue', linestyle='--')
    ax2.plot(epochs, data['val_acc'], label='Val Acc', color='blue')
    ax2.set_title(f'{LABELS[exp_name]} - Accuracy Curve')
    ax2.set_xlabel('Epoci')
    ax2.set_ylabel('Acuratețe')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Annotate gap
    final_train = data['train_acc'][-1]
    final_val = data['val_acc'][-1]
    gap = final_train - final_val
    ax2.text(0.5, 0.5, f'Gap: {gap:.1%}', transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_path = Path('results/plots') / f'dashboard_{exp_name}.png'
    plt.savefig(save_path, dpi=300)
    print(f"📊 Dashboard salvat: {save_path}")
    plt.close()


def main():
    # Asigura folderul de ploturi
    Path('results/plots').mkdir(parents=True, exist_ok=True)

    experiments = [
        'baseline_fp32',
        'experiment2_regularized',
        'experiment3_fp16'
    ]

    # 1. Incarca datele
    data_store = {}
    for exp in experiments:
        data_store[exp] = load_metrics(exp)

    # 2. Genereaza Comparatii
    # Acuratete Validare (Cel mai important grafic)
    plot_comparison_metric(data_store, 'val_acc',
                           'Evoluția Acurateței pe Validare (Test Set)',
                           'Acuratețe', 'comparison_accuracy.png')

    # Loss Validare (Arata stabilitatea)
    plot_comparison_metric(data_store, 'val_loss',
                           'Evoluția Loss-ului pe Validare',
                           'Cross Entropy Loss', 'comparison_loss.png')

    # Train vs Val Gap (Arata overfitting-ul la Baseline)
    # Aici facem un grafic special doar cu Train Loss
    plot_comparison_metric(data_store, 'train_loss',
                           'Evoluția Loss-ului pe Antrenare (Learning Speed)',
                           'Train Loss', 'comparison_train_loss.png')

    # 3. Genereaza Dashboards individuale
    for exp in experiments:
        plot_individual_dashboard(exp, data_store[exp])


if __name__ == '__main__':
    main()
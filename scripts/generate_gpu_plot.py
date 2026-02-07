import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configurare ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, '../results/logs')
OUTPUT_DIR = os.path.join(BASE_DIR, '../results/plots_v2')

os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_NIGHT = os.path.join(LOGS_DIR, 'full_night_run.csv')
FILE_FP16_EXT = os.path.join(LOGS_DIR, 'run_fp16_extended.csv')

COLORS = {
    "Baseline FP32": "#e74c3c",  # Roșu
    "Augmented FP32": "#2ecc71",  # Verde
    "Baseline FP16": "#f39c12",  # Portocaliu
    "Augmented FP16": "#3498db"  # Albastru
}


def trim_idle_tail(df, threshold=1.0, buffer_seconds=300):
    """
    Taie dataframe-ul după ultimul moment de activitate GPU.
    Păstrează un buffer de 'buffer_seconds' (default 5 min) pentru context.
    """
    # Găsim indicii unde consumul este > 1W
    active_indices = df.index[df['gpu_power_W'] > threshold].tolist()

    if not active_indices:
        return df  # Nu am găsit activitate, returnăm tot

    last_active_idx = active_indices[-1]

    # Calculăm punctul de tăiere (ultimul activ + buffer)
    cut_idx = min(last_active_idx + buffer_seconds, len(df))

    # Debug info
    original_duration = len(df) / 3600.0
    new_duration = cut_idx / 3600.0
    if original_duration - new_duration > 0.1:
        print(f"   -> Auto-Trim: Scurtat de la {original_duration:.2f}h la {new_duration:.2f}h (eliminat idle final)")

    return df.iloc[:cut_idx].copy()


def load_and_split_night_run(filepath):
    if not os.path.exists(filepath):
        print(f"Eroare: {filepath} lipsă.")
        return None, None, None

    df = pd.read_csv(filepath)
    df['gpu_power_W'] = df['gpu_power_mW'] / 1000.0

    active_indices = df.index[df['gpu_power_W'] > 1.0].to_numpy()
    if len(active_indices) == 0: return None, None, None

    gaps = np.diff(active_indices)
    split_points = np.where(gaps > 60)[0]

    if len(split_points) < 2:
        sorted_gaps_idx = np.argsort(gaps)[::-1]
        split_points = sorted(sorted_gaps_idx[:2])

    idx_split_1 = active_indices[split_points[0]]
    idx_start_2 = active_indices[split_points[0] + 1]
    idx_split_2 = active_indices[split_points[1]]
    idx_start_3 = active_indices[split_points[1] + 1]

    exp1 = df.iloc[:idx_split_1 + 10].copy()
    exp2 = df.iloc[idx_start_2 - 10: idx_split_2 + 10].copy()
    exp3 = df.iloc[idx_start_3 - 10:].copy()

    return exp1, exp2, exp3


def process_single_run(filepath):
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    df['gpu_power_W'] = df['gpu_power_mW'] / 1000.0

    # Aici aplicăm funcția de tăiere (TRIM)
    print(f"Procesare {os.path.basename(filepath)}...")
    df = trim_idle_tail(df)

    return df


def get_stats(df):
    active = df[df['gpu_power_W'] > 1.0]
    return {
        "avg": active['gpu_power_W'].mean(),
        "peak": df['gpu_power_W'].max()
    }


def main():
    print("--- Generare Grafic GPU Power (v3 - Auto Trim) ---")

    exp1_df, exp2_df, exp3_df = load_and_split_night_run(FILE_NIGHT)
    exp4_df = process_single_run(FILE_FP16_EXT)

    if any(x is None for x in [exp1_df, exp2_df, exp3_df, exp4_df]):
        print("Eroare la date.")
        return

    experiments = [
        {"name": "Baseline FP32", "data": exp1_df, "color": COLORS["Baseline FP32"]},
        {"name": "Augmented FP32", "data": exp2_df, "color": COLORS["Augmented FP32"]},
        {"name": "Baseline FP16", "data": exp3_df, "color": COLORS["Baseline FP16"]},
        {"name": "Augmented FP16", "data": exp4_df, "color": COLORS["Augmented FP16"]}
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.25)
    axes_flat = axes.flatten()

    stats_summary = []

    for i, exp in enumerate(experiments):
        ax = axes_flat[i]
        df = exp['data']
        name = exp['name']

        # Reset timp la 0
        df = df.reset_index(drop=True)
        time_hours = df.index / 3600.0

        stats = get_stats(df)
        stats_summary.append((name, stats['avg'], stats['peak']))

        sns.lineplot(x=time_hours, y=df['gpu_power_W'], ax=ax, color=exp['color'], linewidth=0.5, alpha=0.9)

        ax.axhline(stats['avg'], color='black', linestyle='--', linewidth=1, alpha=0.7,
                   label=f"Avg Active: {stats['avg']:.2f}W")

        ax.set_title(f"{name}", fontsize=12, fontweight='bold', color='#333333')
        ax.set_ylabel("GPU Power (W)")
        ax.set_xlabel("Durată (ore)")
        ax.set_ylim(0, 16)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        ax.text(0.02, 0.9, f"Peak: {stats['peak']:.2f}W", transform=ax.transAxes,
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.suptitle("Analiza Consumului GPU pe cele 4 Experimente (Apple M4)\n(Sincronizat la durata reală de antrenare)",
                 fontsize=16, y=0.96)

    out_file = os.path.join(OUTPUT_DIR, 'gpu_power_4_experiments_trimmed.png')
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    print(f"\n✅ Grafic salvat în: {out_file}")

    print("\n--- Rezultate Finale pentru Raport ---")
    print(f"{'Experiment':<20} | {'Avg Power (W)':<15} | {'Peak Power (W)':<15}")
    print("-" * 56)
    for name, avg, peak in stats_summary:
        print(f"{name:<20} | {avg:.2f}            | {peak:.2f}")


if __name__ == "__main__":
    main()
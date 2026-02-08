import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# --- CONFIGURARE ---
BASE_DIR = Path(__file__).resolve().parent.parent
# Calea către log-ul VECHI (cel cu 3 experimente)
OLD_LOG_PATH = BASE_DIR / "results/logs/full_night_run.csv"
# Calea către log-ul NOU (generat de tine acum)
NEW_LOG_PATH = BASE_DIR / "results/logs/verification_fp32.csv"

# Unde salvăm rezultatele
OUTPUT_DIR = BASE_DIR / "results/verification_run_fp32/plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Culori
COLOR_OLD = "#e74c3c"  # Roșu (Vechi)
COLOR_NEW = "#2ecc71"  # Verde (Nou)


def get_old_augmented_run(filepath):
    """Extrage DOAR experimentul 2 (Augmented FP32) din log-ul vechi"""
    if not filepath.exists():
        print(f"❌ Nu găsesc log-ul vechi: {filepath}")
        return None

    df = pd.read_csv(filepath)
    df['gpu_power_W'] = df['gpu_power_mW'] / 1000.0

    # Găsim zonele active
    active_indices = df.index[df['gpu_power_W'] > 1.0].to_numpy()
    if len(active_indices) == 0: return None

    # Găsim pauzele dintre experimente
    gaps = np.diff(active_indices)
    split_points = np.where(gaps > 60)[0]  # Pauze > 60s

    if len(split_points) < 2:
        print("⚠️ Atenție: Nu am putut separa perfect experimentele vechi. Încerc cea mai bună potrivire.")
        sorted_gaps = np.argsort(gaps)[::-1]
        split_points = sorted(sorted_gaps[:2])

    # Experimentul 2 este între prima și a doua pauză
    idx_start = active_indices[split_points[0] + 1]
    idx_end = active_indices[split_points[1]]

    # Extragem cu puțin buffer
    segment = df.iloc[idx_start - 10: idx_end + 10].copy()
    print(f"✅ Extras 'Old Augmented FP32': {len(segment) / 3600:.2f} ore")
    return segment


def get_new_run(filepath):
    """Încarcă log-ul nou de verificare"""
    if not filepath.exists():
        print(f"❌ Nu găsesc log-ul nou: {filepath}")
        print("   Verifică dacă l-ai salvat cu numele corect (--name verification_fp32)")
        return None

    df = pd.read_csv(filepath)
    df['gpu_power_W'] = df['gpu_power_mW'] / 1000.0

    # Tăiem cozile de idle (auto-trim)
    active_indices = df.index[df['gpu_power_W'] > 1.0].tolist()
    if active_indices:
        last_active = active_indices[-1]
        # Păstrăm 2 minute după final
        cut_idx = min(last_active + 120, len(df))
        df = df.iloc[:cut_idx].copy()

    print(f"✅ Încărcat 'New Verification Run': {len(df) / 3600:.2f} ore")
    return df


def main():
    print("--- COMPARARE RERUN FP32 ---")

    df_old = get_old_augmented_run(OLD_LOG_PATH)
    df_new = get_new_run(NEW_LOG_PATH)

    if df_old is None or df_new is None:
        return

    # Pregătire date pentru plot (reset index la 0)
    df_old = df_old.reset_index(drop=True)
    df_old['Time_h'] = df_old.index / 3600.0

    df_new = df_new.reset_index(drop=True)
    df_new['Time_h'] = df_new.index / 3600.0

    # Calcul metrici
    def get_stats(df):
        active = df[df['gpu_power_W'] > 1.0]
        return active['gpu_power_W'].mean(), df['gpu_power_W'].max()

    avg_old, peak_old = get_stats(df_old)
    avg_new, peak_new = get_stats(df_new)

    # --- PLOTTING ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 1. Old Run
    sns.lineplot(data=df_old, x='Time_h', y='gpu_power_W', ax=axes[0], color=COLOR_OLD, linewidth=0.8)
    axes[0].set_title(f"OLD Run (Din Full Night) - Avg: {avg_old:.2f}W | Peak: {peak_old:.2f}W", fontweight='bold')
    axes[0].set_ylabel("Power (W)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 18)

    # 2. New Run
    sns.lineplot(data=df_new, x='Time_h', y='gpu_power_W', ax=axes[1], color=COLOR_NEW, linewidth=0.8)
    axes[1].set_title(f"NEW Verification Run - Avg: {avg_new:.2f}W | Peak: {peak_new:.2f}W", fontweight='bold')
    axes[1].set_ylabel("Power (W)")
    axes[1].set_xlabel("Time (hours)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 18)

    # Adăugăm observația
    plt.suptitle("Verificare Reproductibilitate: Augmented FP32 GPU Power", fontsize=14)

    save_path = OUTPUT_DIR / "comparison_fp32_augmented.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\n📈 Grafic salvat în: {save_path}")

    # --- CONCLUZIE AUTOMATĂ ---
    print("\n--- REZULTATE COMPARATIVE ---")
    print(f"{'Metric':<10} | {'OLD Run':<10} | {'NEW Run':<10} | {'Diff'}")
    print("-" * 45)
    print(f"{'Avg Power':<10} | {avg_old:.2f} W     | {avg_new:.2f} W     | {avg_new - avg_old:+.2f} W")
    print(f"{'Peak':<10} | {peak_old:.2f} W     | {peak_new:.2f} W     | {peak_new - peak_old:+.2f} W")

    if abs(avg_new - avg_old) < 1.0 and abs(peak_new - peak_old) < 2.0:
        print("\n✅ CONCLUZIE: Rezultatele se confirmă! Profilul este similar.")
    else:
        print("\n⚠️ CONCLUZIE: Există diferențe notabile. Verifică graficele.")


if __name__ == "__main__":
    main()
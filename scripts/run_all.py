import subprocess
import time
import sys
from datetime import datetime

# Lista experimentelor în ordinea dorită
experiments = [
    "experiments/baseline_fp32.py",
    "experiments/experiment2_regularized.py",
    "experiments/experiment3_fp16.py"
]


def run_experiment(script_path):
    print(f"\n{'=' * 80}")
    print(f"🚀 PORNIRE EXPERIMENT: {script_path}")
    print(f"🕒 Ora: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'=' * 80}\n")

    start = time.time()

    # Rulăm procesul și așteptăm să termine
    result = subprocess.run([sys.executable, script_path])

    duration = (time.time() - start) / 60

    if result.returncode == 0:
        print(f"\n✅ {script_path} FINALIZAT cu succes în {duration:.1f} minute.")
    else:
        print(f"\n❌ EROARE la {script_path}. Cod eroare: {result.returncode}")
        # Putem alege să oprim totul sau să continuăm. Aici continuăm.


def cooldown(seconds=60):
    print(f"\n❄️  Cooldown period ({seconds}s) to reset thermals...")
    time.sleep(seconds)


def main():
    print("🎯 Starting Master Run Sequence (All 3 Experiments)")

    for i, exp in enumerate(experiments):
        run_experiment(exp)

        # Pauză de răcire între experimente (dar nu după ultimul)
        if i < len(experiments) - 1:
            cooldown(60)

    print(f"\n{'=' * 80}")
    print("🏁 TOATE EXPERIMENTELE S-AU ÎNCHEIAT!")
    print(f"🕒 Ora: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
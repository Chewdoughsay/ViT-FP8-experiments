import subprocess
import csv
import argparse
import re
from pathlib import Path
from datetime import datetime
import sys


def monitor_stream(output_file, interval=1000):
    print(f"🔒 Starting GPU Monitor (Robust Stream)...")
    print(f"💾 Saving logs to: {output_file}")

    # Header CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'gpu_utilization_percent', 'gpu_power_mW', 'cpu_power_mW'])

    cmd = [
        "powermetrics",
        "-i", str(interval),
        "--samplers", "cpu_power,gpu_power",
        "--show-initial-usage"
    ]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Valori implicite
        current_metrics = {'gpu_res': 0.0, 'gpu_pwr': 0.0, 'cpu_pwr': 0.0}

        while True:
            line = process.stdout.readline()
            if not line:
                break

            # Convertim la minuscule pentru a nu conta daca e "GPU" sau "gpu"
            line_lower = line.lower()

            # 1. GPU Utilization (Residency)
            # Cautam "gpu active residency: 95.5%"
            if "gpu active residency" in line_lower:
                match = re.search(r'([\d\.]+)\s*%', line)
                if match:
                    current_metrics['gpu_res'] = float(match.group(1))

            # 2. GPU Power
            # Cautam "gpu power: 123 mW"
            elif "gpu power" in line_lower:
                match = re.search(r'([\d]+)\s*mw', line_lower)
                if match:
                    current_metrics['gpu_pwr'] = float(match.group(1))

            # 3. CPU Power
            # Folosim CPU Power ca "trigger" pentru a salva linia (de obicei apare ultima)
            elif "cpu power" in line_lower:
                match = re.search(r'([\d]+)\s*mw', line_lower)
                if match:
                    current_metrics['cpu_pwr'] = float(match.group(1))

                    # --- SALVARE ȘI AFIȘARE ---
                    timestamp = datetime.now().strftime('%H:%M:%S')

                    # Logica vizuală: Dacă avem consum mare (>1000mW) dar util 0%,
                    # înseamnă că probabil util-ul e citit greșit, dar e clar că muncește.
                    # Totuși, cu fix-ul de mai sus ar trebui să meargă.

                    print(
                        f"[{timestamp}] GPU Util: {current_metrics['gpu_res']:5.1f}% | GPU Pwr: {current_metrics['gpu_pwr']:5.0f} mW | CPU Pwr: {current_metrics['cpu_pwr']:5.0f} mW")

                    with open(output_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            timestamp,
                            current_metrics['gpu_res'],
                            current_metrics['gpu_pwr'],
                            current_metrics['cpu_pwr']
                        ])

    except KeyboardInterrupt:
        print("\n🛑 Monitor stopped.")
        process.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='monitor_log', help='Nume fisier log')
    args = parser.parse_args()

    Path("results/logs").mkdir(parents=True, exist_ok=True)
    file_path = f"results/logs/{args.name}.csv"

    monitor_stream(file_path)
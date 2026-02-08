"""
GPU and power monitoring for Apple Silicon (M1/M2/M3) using powermetrics.

This module provides real-time monitoring of GPU utilization and power consumption
on Apple Silicon Macs. It uses the macOS powermetrics tool to capture:
- GPU active residency (utilization percentage)
- GPU power consumption (milliwatts)
- CPU power consumption (milliwatts)

The monitor saves data to CSV for post-training analysis and plotting.

Requirements:
    - macOS with Apple Silicon (M1/M2/M3)
    - sudo privileges (powermetrics requires root access)

Usage:
    Terminal:
        $ sudo python src/utils/gpu_monitor.py --name experiment_name

    Python:
        >>> from src.utils.gpu_monitor import monitor_stream
        >>> monitor_stream('results/logs/training_run.csv', interval=1000)

    Run alongside training:
        # Terminal 1: Start GPU monitoring
        $ sudo python src/utils/gpu_monitor.py --name my_experiment

        # Terminal 2: Run training
        $ python scripts/train.py --config configs/BaseFP32.yaml

Notes:
    - Requires sudo because powermetrics accesses kernel-level hardware data
    - Default interval: 1000ms (1 second) per sample
    - Press Ctrl+C to stop monitoring gracefully
    - CSV output includes timestamp, GPU%, GPU power, CPU power

Example Output:
    🔒 Starting GPU Monitor (Robust Stream)...
    💾 Saving logs to: results/logs/experiment.csv
    [14:32:01] GPU Util:  45.2% | GPU Pwr:  3420 mW | CPU Pwr:  1250 mW
    [14:32:02] GPU Util:  67.8% | GPU Pwr:  4820 mW | CPU Pwr:  1680 mW
    ...
"""
import subprocess
import csv
import argparse
import re
from pathlib import Path
from datetime import datetime
import sys


def monitor_stream(output_file, interval=1000):
    """
    Monitor GPU utilization and power consumption in real-time (Apple Silicon).

    Streams data from macOS powermetrics tool and parses GPU active residency,
    GPU power, and CPU power. Saves measurements to CSV file with timestamps.

    Args:
        output_file (str): Path to CSV output file (e.g., 'results/logs/run.csv')
        interval (int): Sampling interval in milliseconds. Default: 1000 (1 second)

    Output CSV Format:
        timestamp, gpu_utilization_percent, gpu_power_mW, cpu_power_mW
        14:32:01, 45.2, 3420, 1250
        14:32:02, 67.8, 4820, 1680
        ...

    Behavior:
        - Creates CSV file with header (overwrites if exists)
        - Continuously reads from powermetrics stdout stream
        - Parses lines for GPU residency, GPU power, CPU power
        - Writes complete measurement when all three values are captured
        - Displays real-time data to console with formatted output
        - Handles Ctrl+C gracefully (terminates powermetrics subprocess)

    Requirements:
        - Must run with sudo (powermetrics requires root access)
        - macOS with Apple Silicon (M1/M2/M3)
        - powermetrics command available (built into macOS)

    Example:
        >>> # Monitor with 500ms sampling
        >>> monitor_stream('results/logs/fp16_run.csv', interval=500)
        🔒 Starting GPU Monitor (Robust Stream)...
        💾 Saving logs to: results/logs/fp16_run.csv
        [14:32:01] GPU Util:  45.2% | GPU Pwr:  3420 mW | CPU Pwr:  1250 mW
        ...
        ^C
        🛑 Monitor stopped.

    Notes:
        - Uses regex matching for robust parsing (case-insensitive)
        - Default values (0.0) used until first measurement captured
        - CPU Power triggers row save (typically appears last in output)
        - Redirect stderr to avoid powermetrics warnings in console
        - Press Ctrl+C to stop monitoring (sends SIGTERM to subprocess)

    Typical Usage:
        # Run in separate terminal before training:
        $ sudo python src/utils/gpu_monitor.py --name my_experiment

        # Or call from Python (requires running script with sudo):
        >>> from src.utils.gpu_monitor import monitor_stream
        >>> monitor_stream('results/logs/experiment.csv')
    """
    print(f"🔒 Starting GPU Monitor (Robust Stream)...")
    print(f"💾 Saving logs to: {output_file}")

    # Write CSV header
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

        # Default values
        current_metrics = {'gpu_res': 0.0, 'gpu_pwr': 0.0, 'cpu_pwr': 0.0}

        while True:
            line = process.stdout.readline()
            if not line:
                break

            # Convert to lowercase for case-insensitive matching
            line_lower = line.lower()

            # 1. GPU Utilization (Residency)
            # Look for "gpu active residency: 95.5%"
            if "gpu active residency" in line_lower:
                match = re.search(r'([\d\.]+)\s*%', line)
                if match:
                    current_metrics['gpu_res'] = float(match.group(1))

            # 2. GPU Power
            # Look for "gpu power: 123 mW"
            elif "gpu power" in line_lower:
                match = re.search(r'([\d]+)\s*mw', line_lower)
                if match:
                    current_metrics['gpu_pwr'] = float(match.group(1))

            # 3. CPU Power
            # Use CPU Power as trigger to save row (usually appears last)
            elif "cpu power" in line_lower:
                match = re.search(r'([\d]+)\s*mw', line_lower)
                if match:
                    current_metrics['cpu_pwr'] = float(match.group(1))

                    # Save and display metrics
                    timestamp = datetime.now().strftime('%H:%M:%S')

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
    parser.add_argument('--name', type=str, default='monitor_log', help='Log file name')
    args = parser.parse_args()

    Path("../../results/logs").mkdir(parents=True, exist_ok=True)
    file_path = f"results/logs/{args.name}.csv"

    monitor_stream(file_path)

    # Example: sudo python src/utils/gpu_monitor.py --name run_fp16_extended
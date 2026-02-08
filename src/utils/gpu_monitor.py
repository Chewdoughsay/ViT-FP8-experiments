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
    Integrated (automatically requests sudo):
        >>> from src.utils.gpu_monitor import GPUMonitor
        >>> monitor = GPUMonitor(output_file='results/BaseFP32/metrics/gpu_stats.csv')
        >>> monitor.start()  # Will prompt for sudo password
        >>> # ... training happens ...
        >>> monitor.stop()

    Standalone:
        $ sudo python src/utils/gpu_monitor.py --name experiment_name

Notes:
    - Requires sudo because powermetrics accesses kernel-level hardware data
    - Automatically prompts for sudo password when started
    - Default interval: 1000ms (1 second) per sample
    - CSV output includes timestamp, GPU%, GPU power, CPU power
    - Gracefully handles sudo denial (training continues without GPU monitoring)

Example Output:
    🔒 GPU Monitor requires sudo access for powermetrics...
    Password: [user enters password]
    🎮 GPU Monitor started...
    💾 Saving to: results/BaseFP32/metrics/gpu_stats.csv
"""
import subprocess
import csv
import argparse
import re
import os
import threading
import time
from pathlib import Path
from datetime import datetime
import sys


class GPUMonitor:
    """
    Background GPU monitor for Apple Silicon (requires sudo).

    Runs powermetrics in a background thread to monitor GPU utilization and power.
    Automatically prompts for sudo password when started.

    Args:
        output_file (str): Path to CSV output file (e.g., 'results/BaseFP32/metrics/gpu_stats.csv')
        interval (int): Sampling interval in milliseconds. Default: 1000

    Attributes:
        output_file (Path): Where GPU stats are saved
        interval (int): Sampling interval in ms
        running (bool): Whether monitoring is active
        process (Popen): Background powermetrics process
        thread (Thread): Thread for parsing powermetrics output
        has_sudo (bool): Whether sudo access was granted

    Example:
        >>> monitor = GPUMonitor('results/BaseFP32/metrics/gpu_stats.csv')
        >>> monitor.start()  # Prompts for sudo password
        GPU Monitor requires sudo access...
        Password: ****
        GPU Monitor started...
        >>> # ... training happens ...
        >>> monitor.stop()
        GPU Monitor stopped.

    Notes:
        - Gracefully handles sudo denial (prints warning, continues without monitoring)
        - Runs in background thread (non-blocking)
        - CSV format: timestamp, gpu_util_%, gpu_power_mW, cpu_power_mW
        - Only works on macOS with Apple Silicon
    """

    def __init__(self, output_file, interval=1000):
        self.output_file = Path(output_file)
        self.interval = interval
        self.running = False
        self.process = None
        self.thread = None
        self.has_sudo = False

        # Create output directory
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def _check_sudo(self):
        """
        Check if we have sudo access (prompts for password if needed).

        Returns:
            bool: True if sudo access granted, False otherwise
        """
        try:
            # Test sudo access with a simple command
            # This will prompt for password if needed
            result = subprocess.run(
                ['sudo', '-v'],
                capture_output=True,
                timeout=60  # Give user 60 seconds to enter password
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            return False
        except Exception:
            return False

    def _monitor_loop(self):
        """
        Internal monitoring loop that runs in background thread.
        Parses powermetrics output and saves to CSV.
        """
        # Write CSV header
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'gpu_utilization_percent', 'gpu_power_mW', 'cpu_power_mW'])

        cmd = [
            "sudo",
            "powermetrics",
            "-i", str(self.interval),
            "--samplers", "cpu_power,gpu_power",
            "--show-initial-usage"
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Default values
            current_metrics = {'gpu_res': 0.0, 'gpu_pwr': 0.0, 'cpu_pwr': 0.0}

            while self.running and self.process.poll() is None:
                line = self.process.stdout.readline()
                if not line:
                    break

                line_lower = line.lower()

                # Parse GPU utilization
                if "gpu active residency" in line_lower:
                    match = re.search(r'([\d\.]+)\s*%', line)
                    if match:
                        current_metrics['gpu_res'] = float(match.group(1))

                # Parse GPU power
                elif "gpu power" in line_lower:
                    match = re.search(r'([\d]+)\s*mw', line_lower)
                    if match:
                        current_metrics['gpu_pwr'] = float(match.group(1))

                # Parse CPU power and save row
                elif "cpu power" in line_lower:
                    match = re.search(r'([\d]+)\s*mw', line_lower)
                    if match:
                        current_metrics['cpu_pwr'] = float(match.group(1))

                        # Save metrics
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        with open(self.output_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                timestamp,
                                current_metrics['gpu_res'],
                                current_metrics['gpu_pwr'],
                                current_metrics['cpu_pwr']
                            ])

        except Exception as e:
            if self.running:
                print(f"GPU Monitor error: {e}")

    def start(self):
        """
        Start GPU monitoring (prompts for sudo password).

        If sudo access is denied, prints a warning and returns without monitoring.
        Training will continue without GPU stats.
        """
        print("GPU Monitor requires sudo access for powermetrics...")

        if not self._check_sudo():
            print("Sudo access denied. GPU monitoring disabled.")
            print("(Training will continue without GPU stats)")
            self.has_sudo = False
            return

        self.has_sudo = True
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

        print("GPU Monitor started...")
        print(f"Saving to: {self.output_file}")

    def stop(self):
        """
        Stop GPU monitoring and clean up.

        Returns:
            dict: Summary statistics (empty dict if monitoring wasn't active)
        """
        if not self.has_sudo or not self.running:
            return {}

        self.running = False

        # Terminate powermetrics process
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

        # Wait for thread
        if self.thread:
            self.thread.join(timeout=5)

        print("GPU Monitor stopped.")
        return {}


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
        Starting GPU Monitor (Robust Stream)...
        Saving logs to: results/logs/fp16_run.csv
        [14:32:01] GPU Util:  45.2% | GPU Pwr:  3420 mW | CPU Pwr:  1250 mW
        ...
        ^C
        Monitor stopped.

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
    print(f"Starting GPU Monitor (Robust Stream)...")
    print(f"Saving logs to: {output_file}")

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
        print("\nMonitor stopped.")
        process.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='monitor_log', help='Log file name')
    args = parser.parse_args()

    Path("../../results/logs").mkdir(parents=True, exist_ok=True)
    file_path = f"results/logs/{args.name}.csv"

    monitor_stream(file_path)

    # Example: sudo python src/utils/gpu_monitor.py --name run_fp16_extended
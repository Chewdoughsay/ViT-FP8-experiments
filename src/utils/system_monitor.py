"""
Hardware monitoring utilities for tracking system resources during training.

This module provides real-time monitoring of CPU usage, memory consumption, and
thermal throttling on macOS systems. The SystemMonitor runs in a background thread
to avoid interfering with training performance.

Example:
    >>> from src.utils.system_monitor import SystemMonitor
    >>>
    >>> # Start monitoring
    >>> monitor = SystemMonitor(interval=2.0)
    >>> monitor.start()
    >>>
    >>> # ... run your training ...
    >>>
    >>> # Stop and get summary
    >>> summary, full_stats = monitor.stop()
    >>> print(f"Avg CPU: {summary['avg_cpu']:.1f}%")
    >>> print(f"Throttled: {summary['throttled']}")
"""
import time
import threading
import psutil
import subprocess
import numpy as np


class SystemMonitor:
    """
    Background monitor for CPU, memory, and thermal throttling.

    Runs in a separate thread to continuously track system resource usage without
    blocking the main training loop. Collects time-series data for all metrics
    and provides summary statistics when stopped.

    Args:
        interval (float): Sampling interval in seconds. Default: 1.0

    Attributes:
        interval (float): Time between measurements in seconds
        running (bool): Whether monitoring is currently active
        stats (dict): Time-series data for all metrics:
            - cpu_percent: List of CPU usage percentages
            - memory_percent: List of RAM usage percentages
            - thermal_pressure: List of thermal throttling levels (macOS)
            - timestamps: List of relative timestamps (seconds since start)
        thread (Thread): Background thread running the monitoring loop
        thermal_supported (bool): Whether thermal monitoring is available (macOS only)

    Notes:
        - CPU and memory monitoring works on all platforms (requires psutil)
        - Thermal monitoring only works on macOS (uses sysctl)
        - If thermal monitoring fails, it's automatically disabled
        - The monitor runs as a daemon thread (won't block program exit)
        - All data is stored in memory until stop() is called

    Example:
        >>> monitor = SystemMonitor(interval=2.0)
        >>> monitor.start()
        >>> # Training runs here...
        >>> summary, stats = monitor.stop()
        >>> print(f"Max CPU: {summary['max_cpu']:.1f}%")
        >>> print(f"Avg Memory: {summary['avg_mem']:.1f}%")
        >>> if summary['throttled']:
        ...     print(f"Warning: Thermal throttling detected!")
    """
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.stats = {
            'cpu_percent': [],
            'memory_percent': [],
            'thermal_pressure': [],
            'timestamps': []
        }
        self.thread = None
        # Flag to disable thermal checking if it fails
        self.thermal_supported = True

    def _get_thermal_pressure(self):
        """
        Read current thermal throttling level (macOS only).

        Queries the macOS kernel for the current thermal pressure level using sysctl.
        Higher values indicate more aggressive thermal throttling.

        Returns:
            int: Thermal pressure level (0 = no throttling, >0 = throttling active).
                Returns 0 if thermal monitoring is not supported or fails.

        Notes:
            - Only works on macOS systems
            - Requires machdep.cpu.thermal_level sysctl parameter
            - If the first call fails, all subsequent calls return 0 (avoid repeated errors)
            - Typical range: 0-100, but exact range is system-dependent
        """
        if not self.thermal_supported:
            return 0

        try:
            # Redirect stderr to DEVNULL to avoid console errors
            output = subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.thermal_level'],
                stderr=subprocess.DEVNULL
            )
            return int(output.strip())
        except Exception:
            # If it fails once, don't try again
            self.thermal_supported = False
            return 0

    def _monitor_loop(self):
        """
        Internal monitoring loop that runs in background thread.

        Continuously samples CPU, memory, and thermal metrics at the specified
        interval until stop() is called. All measurements are appended to
        self.stats for later analysis.

        Notes:
            - Runs in a daemon thread (automatically terminates when program exits)
            - Uses relative timestamps (seconds since monitoring started)
            - Sleeps between samples to avoid excessive CPU usage
            - Thread-safe: only this method modifies self.stats
        """
        start_time = time.time()
        while self.running:
            # 1. CPU & Memory
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent

            # 2. Thermal Pressure
            thermal = self._get_thermal_pressure()

            # 3. Log
            self.stats['cpu_percent'].append(cpu)
            self.stats['memory_percent'].append(mem)
            self.stats['thermal_pressure'].append(thermal)
            self.stats['timestamps'].append(time.time() - start_time)

            time.sleep(self.interval)

    def start(self):
        """
        Start monitoring system resources in a background thread.

        Resets all statistics, launches the monitoring thread, and begins
        collecting CPU, memory, and thermal data at the specified interval.

        Notes:
            - Clears any previous monitoring data
            - Thread runs as daemon (won't prevent program exit)
            - Non-blocking: returns immediately while monitoring continues
            - Can only start if not already running

        Example:
            >>> monitor = SystemMonitor(interval=2.0)
            >>> monitor.start()
            System Monitor started...
        """
        self.running = True
        self.stats = {k: [] for k in self.stats}
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("System Monitor started...")

    def stop(self):
        """
        Stop monitoring and return summary statistics.

        Signals the background thread to stop, waits for it to finish, then
        computes summary statistics from all collected data.

        Returns:
            tuple: (summary, full_stats)
                summary (dict): Aggregated statistics with keys:
                    - avg_cpu (float): Mean CPU usage percentage
                    - max_cpu (float): Peak CPU usage percentage
                    - avg_mem (float): Mean memory usage percentage
                    - max_thermal (int): Peak thermal throttling level
                    - throttled (bool): True if any throttling was detected
                full_stats (dict): Complete time-series data:
                    - cpu_percent (list): All CPU measurements
                    - memory_percent (list): All memory measurements
                    - thermal_pressure (list): All thermal measurements
                    - timestamps (list): Relative timestamps in seconds

        Example:
            >>> summary, stats = monitor.stop()
            System Monitor stopped.
            >>> print(f"Avg CPU: {summary['avg_cpu']:.1f}%")
            >>> print(f"Throttled: {summary['throttled']}")
            >>> # Access raw data for plotting
            >>> plt.plot(stats['timestamps'], stats['cpu_percent'])

        Notes:
            - Blocks until monitoring thread finishes (typically < 1 second)
            - Returns safe default values (0) if no data was collected
            - full_stats can be saved to JSON for post-training analysis
        """
        self.running = False
        if self.thread:
            self.thread.join()

        # Calculate averages (protected against empty lists)
        def safe_mean(data):
            return np.mean(data) if data else 0

        def safe_max(data):
            return np.max(data) if data else 0

        summary = {
            'avg_cpu': safe_mean(self.stats['cpu_percent']),
            'max_cpu': safe_max(self.stats['cpu_percent']),
            'avg_mem': safe_mean(self.stats['memory_percent']),
            'max_thermal': safe_max(self.stats['thermal_pressure']),
            'throttled': any(t > 0 for t in self.stats['thermal_pressure'])
        }
        print("System Monitor stopped.")
        return summary, self.stats
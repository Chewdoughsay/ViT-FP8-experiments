import time
import threading
import psutil
import subprocess
import numpy as np


class SystemMonitor:
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
        # Flag pentru a dezactiva verificarea termică dacă nu merge
        self.thermal_supported = True

    def _get_thermal_pressure(self):
        """Citeste nivelul de throttling (Safe version)"""
        if not self.thermal_supported:
            return 0

        try:
            # Redirectionam stderr la DEVNULL ca sa nu apara erori in consola
            output = subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.thermal_level'],
                stderr=subprocess.DEVNULL
            )
            return int(output.strip())
        except Exception:
            # Daca da eroare o data, nu mai incercam niciodata
            self.thermal_supported = False
            return 0

    def _monitor_loop(self):
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
        """Porneste monitorizarea in background"""
        self.running = True
        self.stats = {k: [] for k in self.stats}
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("🖥️  System Monitor started...")

    def stop(self):
        """Opreste monitorizarea si returneaza media"""
        self.running = False
        if self.thread:
            self.thread.join()

        # Calculeaza medii (protejat la liste goale)
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
        print("🛑 System Monitor stopped.")
        return summary, self.stats
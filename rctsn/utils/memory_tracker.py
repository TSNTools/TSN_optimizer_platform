import os
import time
import subprocess
import json
import multiprocessing as mp
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

def get_system_memory():
    """Get system memory information using btop or free command"""
    try:
        # First try btop in JSON mode
        result = subprocess.run(
            ["btop", "--json", "1"],
            capture_output=True,
            text=True,
            timeout=2 # Short timeout
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            total_memory = data["memory"]["total"] / (1024 * 1024 * 1024)  # Convert to GB
            used_memory = data["memory"]["used"] / (1024 * 1024 * 1024)
            available_memory = data["memory"]["available"] / (1024 * 1024 * 1024)

            return {
                "total_gb": round(total_memory, 3),
                "used_gb": round(used_memory, 3),
                "available_gb": round(available_memory, 3),
                "percent_used": round(used_memory / total_memory * 100, 1) if total_memory else 0,
                "source": "btop"
            }
    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, FileNotFoundError, TimeoutError):
        pass # Silently ignore errors and try fallback

    # Fall back to free command
    try:
        result = subprocess.run(
            ["free", "-b"], # Use bytes for better precision before converting
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                memory_info = lines[1].split()
                if len(memory_info) >= 7:
                    total_memory = float(memory_info[1]) / (1024 * 1024 * 1024) # Bytes to GB
                    used_memory = float(memory_info[2]) / (1024 * 1024 * 1024)
                    # 'available' (col 7) is generally preferred over 'free' (col 4)
                    available_memory = float(memory_info[6]) / (1024 * 1024 * 1024)

                    return {
                        "total_gb": round(total_memory, 3),
                        "used_gb": round(used_memory, 3),
                        "available_gb": round(available_memory, 3),
                        "percent_used": round(used_memory / total_memory * 100, 1) if total_memory else 0,
                        "source": "free"
                    }
    except (subprocess.SubprocessError, IndexError, ValueError, FileNotFoundError, TimeoutError):
        pass # Silently ignore errors

    return {
        "total_gb": -1, "used_gb": -1, "available_gb": -1, "percent_used": -1, "source": "none"
    }

def get_process_memory_by_pid(pid):
    """Get memory usage of a specific process by PID"""
    try:
        process = psutil.Process(pid)
        memory_info = process.memory_info()
        rss_gb = memory_info.rss / (1024 * 1024 * 1024)  # Bytes to GB
        vms_gb = memory_info.vms / (1024 * 1024 * 1024)  # Bytes to GB
        return {
            "rss_gb": round(rss_gb, 3),
            "vms_gb": round(vms_gb, 3),
            "percent": round(process.memory_percent(), 1)
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
        return {"rss_gb": -1, "vms_gb": -1, "percent": -1}

def _memory_monitor_worker(output_file, interval, stop_event, parent_pid):
    """Worker function for memory monitoring process (intended for internal use)"""
    print(f"[MemoryMonitor Worker PID: {os.getpid()}] Started, monitoring parent PID: {parent_pid}")
    start_time = time.time()
    try:
        while not stop_event.is_set():
            system_memory = get_system_memory()
            python_memory = get_process_memory_by_pid(parent_pid) # Monitor parent
            timestamp = time.time() - start_time

            try:
                # Ensure file exists and write header if needed (though header is written on init)
                if not os.path.exists(output_file):
                     print(f"[MemoryMonitor Worker] Output file {output_file} missing, cannot write.")
                     time.sleep(interval) # Avoid busy-looping if file disappears
                     continue

                with open(output_file, 'a') as f:
                    f.write(f"{timestamp:.2f},{system_memory['total_gb']},{system_memory['used_gb']},"
                            f"{system_memory['available_gb']},{system_memory['percent_used']},{system_memory['source']},"
                            f"{python_memory['rss_gb']},{python_memory['vms_gb']},{python_memory['percent']}\n")
            except IOError as e:
                print(f"[MemoryMonitor Worker] Error writing to file {output_file}: {e}")
            except Exception as e:
                 print(f"[MemoryMonitor Worker] Unexpected error during write: {e}")

            # Wait for the interval or until stop event is set
            stop_event.wait(timeout=interval)

    except Exception as e:
        print(f"[MemoryMonitor Worker] Error: {e}")
    finally:
        print(f"[MemoryMonitor Worker PID: {os.getpid()}] Stopping.")


class MemoryTracker:
    """Manages memory usage monitoring in a separate process."""

    def __init__(self, output_folder: str, interval: int = 5):
        """
        Initializes the MemoryTracker.

        Args:
            output_folder: The directory where memory data will be saved directly.
            interval: The sampling interval in seconds.
        """
        # Use the given output_folder directly without creating a nested memory folder
        self.output_folder = output_folder
        try:
            os.makedirs(self.output_folder, exist_ok=True)
            print(f"Ensuring output folder exists: {self.output_folder}")
        except OSError as e:
            raise ValueError(f"Could not access or create output folder: {self.output_folder} - {e}")
                
        self.output_file = os.path.join(self.output_folder, "memory_usage.csv")
        self.interval = interval
        self.parent_pid = os.getpid() # PID of the process creating this instance
        self._process = None
        self._stop_event = None
        self._write_header()

    def _write_header(self):
        """Writes the CSV header to the output file."""
        try:
            with open(self.output_file, 'w') as f:
                f.write("timestamp,total_gb,used_gb,available_gb,percent_used,source,python_rss_gb,python_vms_gb,python_percent\n")
        except IOError as e:
            print(f"ERROR: Could not write memory file header to {self.output_file}: {e}")
            # Consider raising the error depending on desired behavior
            # raise

    def start(self):
        """Starts the memory monitoring process."""
        if self._process and self._process.is_alive():
            print("Memory tracking is already running.")
            return

        self._stop_event = mp.Event()
        self._process = mp.Process(
            target=_memory_monitor_worker,
            args=(self.output_file, self.interval, self._stop_event, self.parent_pid),
            daemon=True # Ensures process exits if main script crashes
        )
        try:
            self._process.start()
            print(f"Memory tracking process started (PID: {self._process.pid}), saving to {self.output_file}")
        except Exception as e:
             print(f"ERROR: Failed to start memory tracking process: {e}")
             self._process = None
             self._stop_event = None

    def stop(self):
        """Stops the memory monitoring process."""
        if not self._process or not self._process.is_alive():
            print("Memory tracking is not running or already stopped.")
            return

        print("Stopping memory tracking process...")
        try:
            if self._stop_event:
                self._stop_event.set()
            self._process.join(timeout=max(1, self.interval + 1)) # Wait a bit longer than interval
            if self._process.is_alive():
                print("Memory tracking process did not terminate gracefully, forcing termination.")
                self._process.terminate()
                self._process.join(timeout=1) # Wait briefly after terminate
            print("Memory tracking stopped.")
        except Exception as e:
            print(f"Error stopping memory tracker: {e}")
        finally:
             self._process = None
             self._stop_event = None

    def plot_usage(self, output_folder=None):
        """
        Generates plots for system and Python process memory usage.

        Args:
            output_folder: Optional. Directory to save plots. Defaults to the same folder where CSV is stored.
        """
        if output_folder is None:
            output_folder = self.output_folder
        # Removed the code that creates a nested memory folder
        
        if not os.path.exists(self.output_file):
            print(f"WARNING: Memory usage file not found: {self.output_file}. Skipping plot.")
            return

        try:
            # Check if file is empty or has only header
            if os.path.getsize(self.output_file) < 150:
                print(f"WARNING: Memory usage file is empty or too small: {self.output_file}. Skipping plot.")
                return
            
            df = pd.read_csv(self.output_file)
            if df.empty:
                print(f"WARNING: Memory usage data is empty in {self.output_file}. Skipping plot.")
                return

            # Filter out any invalid values (like -1) that might exist if process wasn't found initially
            valid_df = df[(df['python_rss_gb'] >= 0) & (df['total_gb'] > 0)]
            
            # --- Combined Plot ---
            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'], df['used_gb'], label='System Used (GB)', color='red', alpha=0.8)
            plt.plot(df['timestamp'], df['available_gb'], label='System Available (GB)', color='green', alpha=0.6)
            plt.plot(df['timestamp'], df['python_rss_gb'], label='Python Process RSS (GB)', color='blue', linestyle='-', linewidth=2)

            plt.grid(alpha=0.4)
            plt.legend()
            plt.xlabel('Time (seconds)')
            plt.ylabel('Memory (GB)')
            plt.title('System and Python Process Memory Usage')
            plt.ylim(bottom=0) # Ensure y-axis starts at 0

            plot_file = os.path.join(output_folder, "combined_chart.png")
            plt.tight_layout()
            plt.savefig(plot_file, dpi=300)
            plt.close()

            # --- Python Process Plot ---
            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'], df['python_rss_gb'], label='Python Physical (RSS)', color='blue')
            plt.plot(df['timestamp'], df['python_vms_gb'], label='Python Virtual (VMS)', color='purple', alpha=0.7)

            plt.grid(alpha=0.4)
            plt.legend()
            plt.xlabel('Time (seconds)')
            plt.ylabel('Memory (GB)')
            plt.title('Python Process Memory Usage (RSS & VMS)')
            plt.ylim(bottom=0)

            # Find peak RSS using valid data only
            if not valid_df.empty:
                py_peak_idx = valid_df['python_rss_gb'].idxmax()
                py_peak_time = df.loc[py_peak_idx, 'timestamp']
                py_peak_memory = df.loc[py_peak_idx, 'python_rss_gb']
                if py_peak_memory > 0: # Only annotate if there's usage
                    plt.annotate(f'Peak RSS: {py_peak_memory:.3f} GB',
                                xy=(py_peak_time, py_peak_memory),
                                xytext=(py_peak_time, py_peak_memory * 1.1), # Adjust position
                                arrowprops=dict(facecolor='blue', arrowstyle='->'),
                                fontsize=9)

            py_plot_file = os.path.join(output_folder, "python_chart.png")
            plt.tight_layout()
            plt.savefig(py_plot_file, dpi=300)
            plt.close()

            print(f"Memory usage charts saved to {output_folder}")

            # --- Save Summary ---
            summary_file = os.path.join(output_folder, "summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"Memory Usage Summary ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
                f.write("===================================\n\n")

                # Use valid data for summary statistics
                if not valid_df.empty:
                    f.write("System Memory:\n")
                    f.write(f"  Peak Usage: {valid_df['used_gb'].max():.3f} GB\n")
                    f.write(f"  Average Usage: {valid_df['used_gb'].mean():.3f} GB\n")
                    f.write(f"  Initial Available: {valid_df['available_gb'].iloc[0]:.3f} GB\n")
                    f.write(f"  Minimum Available: {valid_df['available_gb'].min():.3f} GB\n")
                    f.write(f"  Total System Memory: {valid_df['total_gb'].iloc[0]:.3f} GB\n\n")

                    f.write("Python Process Memory (Parent PID):\n")
                    f.write(f"  Peak Physical (RSS): {valid_df['python_rss_gb'].max():.3f} GB\n")
                    f.write(f"  Average Physical (RSS): {valid_df['python_rss_gb'].mean():.3f} GB\n")
                    f.write(f"  Peak Virtual (VMS): {valid_df['python_vms_gb'].max():.3f} GB\n")
                    f.write(f"  Average Virtual (VMS): {valid_df['python_vms_gb'].mean():.3f} GB\n")
                    f.write(f"  Maximum Percentage of System: {valid_df['python_percent'].max():.1f}%\n")
                else:
                    f.write("No valid memory data was collected or all readings were invalid.\n")

            if output_folder != self.output_folder:
                shutil.copy2(self.output_file, os.path.join(output_folder, "memory_usage.csv"))

            print(f"Memory usage summary saved to {summary_file}")

        except ImportError:
            print("WARNING: pandas or matplotlib not found. Cannot generate memory plots/summary.")
        except pd.errors.EmptyDataError:
            print(f"WARNING: Memory usage file is empty: {self.output_file}. Skipping plot/summary.")
        except Exception as e:
            print(f"ERROR: Failed to generate memory usage plot/summary: {e}")
            import traceback
            traceback.print_exc()


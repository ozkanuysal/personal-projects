import threading
import time

from collections import deque
from typing import List

class LatencyTracker:
    def __init__(self, window_size: int):
        """
        Initialize the LatencyTracker.

        :param window_size: The size of the sliding time window in seconds.
        """
        self.window_size = window_size
        self.latencies = deque()  # Store (timestamp, latency)
        self.lock = threading.Lock()

    def report_latency(self, latency: float):
        """
        Report a latency value.

        :param latency: The latency value to report.
        """
        with self.lock:
            current_time = time.time()
            self.latencies.append((current_time, latency))
            self._remove_old_entries()

    def get_99th_percentile(self) -> float:
        """
        Calculate the 99th percentile latency over the current time window.

        :return: The 99th percentile latency.
        """
        with self.lock:
            self._remove_old_entries()
            if not self.latencies:
                return 0.0  # No data available

            sorted_latencies = sorted(latency for _, latency in self.latencies)
            index = int(0.99 * len(sorted_latencies)) - 1
            return sorted_latencies[max(index, 0)]

    def _remove_old_entries(self):
        """
        Remove latency entries that are outside the sliding time window.
        """
        current_time = time.time()
        while self.latencies and self.latencies[0][0] < current_time - self.window_size:
            self.latencies.popleft()

# Example Usage
if __name__ == "__main__":
    tracker = LatencyTracker(window_size=10)  # 10-second sliding window

    tracker.report_latency(100)
    time.sleep(1)
    tracker.report_latency(200)
    time.sleep(1)
    tracker.report_latency(300)

    print("99th Percentile:", tracker.get_99th_percentile())

    tracker.report_latency(400)
    tracker.report_latency(500)
    
    print("99th Percentile:", tracker.get_99th_percentile())
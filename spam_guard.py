# spam_guard.py
import time
import threading

class SpamGuard:
    def __init__(self, debounce_seconds=30, min_interval=10, max_per_minute=3):
        self.debounce_seconds = debounce_seconds  # chặn lặp lại cùng 1 loại cảnh báo
        self.min_interval = min_interval          # khoảng cách tối thiểu giữa 2 alert bất kỳ
        self.max_per_minute = max_per_minute      # số lần tối đa cho phép trong 1 phút
        self.last_alert_time = {}
        self.alert_history = []  # [(timestamp, reason)]
        self.lock = threading.Lock()

    def allow(self, reason: str) -> bool:
        now = time.time()
        with self.lock:
            # --- kiểm tra debounce cho cùng 1 loại ---
            last = self.last_alert_time.get(reason, 0)
            if now - last < self.debounce_seconds:
                return False

            # --- kiểm tra min interval toàn cục ---
            if self.alert_history and now - self.alert_history[-1][0] < self.min_interval:
                return False

            # --- kiểm tra số lượng trong 60s ---
            self.alert_history = [(t, r) for (t, r) in self.alert_history if now - t < 60]
            if len(self.alert_history) >= self.max_per_minute:
                return False

            # --- hợp lệ => cập nhật ---
            self.last_alert_time[reason] = now
            self.alert_history.append((now, reason))
            return True

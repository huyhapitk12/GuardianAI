# spam_guard.py
import threading
import time
from typing import Tuple, Union

from config import (DEBOUNCE_SECONDS, SPAM_GUARD_MAX_PER_MINUTE,
                    SPAM_GUARD_MIN_INTERVAL)

class SpamGuard:
    """Lớp quản lý và ngăn chặn việc gửi cảnh báo lặp lại quá nhanh (spam)."""
    def __init__(self, debounce_seconds=DEBOUNCE_SECONDS, min_interval=SPAM_GUARD_MIN_INTERVAL, max_per_minute=SPAM_GUARD_MAX_PER_MINUTE):
        self.debounce_seconds = debounce_seconds  # Chặn lặp lại cùng 1 key cảnh báo
        self.min_interval = min_interval          # Khoảng cách tối thiểu giữa 2 alert bất kỳ
        self.max_per_minute = max_per_minute      # Số lần tối đa cho phép trong 1 phút
        self.last_alert_time = {}                 # Lưu {key: timestamp}
        self.alert_history = []                   # Lưu [(timestamp, key)]
        self.muted_until = {}                     # Lưu {key: timestamp}
        self.lock = threading.Lock()

    def mute(self, key: Union[str, Tuple], duration_seconds: int):
        """Tạm thời bỏ qua tất cả các cảnh báo cho một key cụ thể."""
        with self.lock:
            self.muted_until[key] = time.time() + duration_seconds
            print(f"[SpamGuard] Key '{key}' đã bị tắt tiếng trong {duration_seconds} giây.")

    def allow(self, key: Union[str, Tuple]) -> bool:
        """Kiểm tra xem một cảnh báo với key cụ thể có được phép gửi đi hay không."""
        now = time.time()
        with self.lock:
            # 0. Kiểm tra nếu key đang bị tắt tiếng
            if self.muted_until.get(key, 0) > now:
                return False

            # 1. Kiểm tra debounce cho cùng 1 key
            if now - self.last_alert_time.get(key, 0) < self.debounce_seconds:
                return False

            # 2. Kiểm tra khoảng cách tối thiểu giữa 2 alert bất kỳ
            if self.alert_history and now - self.alert_history[-1][0] < self.min_interval:
                return False

            # 3. Kiểm tra số lượng tối đa trong 1 phút
            self.alert_history = [(t, k) for (t, k) in self.alert_history if now - t < 60]
            if len(self.alert_history) >= self.max_per_minute:
                return False

            # Hợp lệ -> Cập nhật trạng thái và cho phép
            self.last_alert_time[key] = now
            self.alert_history.append((now, key))
            return True
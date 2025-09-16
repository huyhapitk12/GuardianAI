# spam_guard.py
import time
import threading
from typing import Union, Tuple

class SpamGuard:
    def __init__(self, debounce_seconds=30, min_interval=10, max_per_minute=3):
        self.debounce_seconds = debounce_seconds  # chặn lặp lại cùng 1 key cảnh báo
        self.min_interval = min_interval          # khoảng cách tối thiểu giữa 2 alert bất kỳ
        self.max_per_minute = max_per_minute      # số lần tối đa cho phép trong 1 phút
        self.last_alert_time = {}                 # Sẽ lưu key -> timestamp
        self.alert_history = []                   # Sẽ lưu (timestamp, key)
        self.lock = threading.Lock()

    def allow(self, key: Union[str, Tuple]) -> bool:
        """
        Kiểm tra xem một cảnh báo với key cụ thể có được phép hay không.
        Key có thể là một chuỗi (ví dụ: "lua_chay") hoặc một tuple (ví dụ: ("nguoi_quen", "TenNguoiA")).
        """
        now = time.time()
        with self.lock:
            # --- 1. Kiểm tra debounce cho cùng 1 key cụ thể ---
            last_for_key = self.last_alert_time.get(key, 0)
            if now - last_for_key < self.debounce_seconds:
                # Vẫn trong thời gian chờ cho key này, bỏ qua
                return False

            # --- 2. Kiểm tra khoảng cách tối thiểu giữa 2 alert bất kỳ ---
            if self.alert_history and now - self.alert_history[-1][0] < self.min_interval:
                # Gửi cảnh báo quá nhanh, bỏ qua
                return False

            # --- 3. Kiểm tra số lượng tối đa trong 1 phút ---
            # Lọc ra các cảnh báo trong vòng 60 giây gần nhất
            self.alert_history = [(t, k) for (t, k) in self.alert_history if now - t < 60]
            if len(self.alert_history) >= self.max_per_minute:
                # Đã đạt giới hạn, bỏ qua
                return False

            # --- Hợp lệ => Cập nhật trạng thái và cho phép ---
            self.last_alert_time[key] = now
            self.alert_history.append((now, key))
            return True
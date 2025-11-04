# state_manager.py
import time, threading, uuid
from config import DEBOUNCE_SECONDS
from typing import Union, Tuple, Optional, Dict, Any

class StateManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = {}  # alert_id -> info
        self.person_detection_enabled = True # <-- THÊM DÒNG NÀY

    # <--- THÊM CÁC HÀM MỚI DƯỚI ĐÂY --->
    def is_person_detection_enabled(self) -> bool:
        """Kiểm tra xem tính năng nhận diện người có đang bật không."""
        with self.lock:
            return self.person_detection_enabled

    def set_person_detection_enabled(self, enabled: bool):
        """Bật hoặc tắt tính năng nhận diện người."""
        with self.lock:
            self.person_detection_enabled = enabled
            print(f"[StateManager] Person detection set to: {enabled}")
    # <--- KẾT THÚC PHẦN THÊM MỚI --->

    # <--- THAY ĐỔI MỚI: Thêm hàm kiểm tra cảnh báo đang chờ --->
    def has_unresolved_alert(self, key: Union[str, Tuple]) -> bool:
        """
        Kiểm tra xem có cảnh báo nào chưa được giải quyết cho một key cụ thể không.
        Key có thể là "nguoi_la" hoặc ("nguoi_quen", "TenNguoiA").
        """
        with self.lock:
            for alert in self.active.values():
                if alert['resolved']:
                    continue

                # Xây dựng key của alert đang xét để so sánh
                current_key = None
                if alert['type'] == 'nguoi_quen':
                    current_key = (alert['type'], alert['asked_for'])
                else:
                    current_key = alert['type']

                if current_key == key:
                    # Tìm thấy một alert chưa giải quyết trùng khớp
                    return True
        return False

    def create_alert(self, typ, chat_id, asked_for=None, image_path=None):
    # --- KẾT THÚC THAY ĐỔI ---
        current_time = time.time()
        # Logic debounce này vẫn hữu ích để tránh tạo alert trùng lặp trong tích tắc
        for alert in self.active.values():
            if (alert['type'] == typ and str(alert['chat_id']) == str(chat_id) and 
                not alert['resolved'] and (current_time - alert['ts'] < DEBOUNCE_SECONDS)):
                if typ == 'nguoi_quen' and alert['asked_for'] == asked_for:
                    return alert['id']
                elif typ != 'nguoi_quen':
                    return alert['id']

        aid = uuid.uuid4().hex
        info = {
            "id": aid, "type": typ, "chat_id": chat_id, 
            "asked_for": asked_for, "ts": current_time, 
            "resolved": False, "reply": None,
            "image_path": image_path # Dòng này giờ sẽ hoạt động đúng
        }
        with self.lock:
            self.active[aid] = info
        return aid

    def resolve_alert(self, aid, reply_text):
        with self.lock:
            if aid in self.active:
                self.active[aid]["resolved"] = True
                self.active[aid]["reply"] = reply_text

    def get_alert_by_id(self, aid: str) -> Optional[Dict[str, Any]]:
        """Lấy thông tin của một cảnh báo bằng ID của nó."""
        with self.lock:
            return self.active.get(aid)

    def latest_unresolved_for_chat(self, chat_id):
        with self.lock:
            cands = [a for a in self.active.values() if (not a["resolved"]) and str(a["chat_id"])==str(chat_id)]
            cands.sort(key=lambda x: x["ts"], reverse=True)
            return cands[0] if cands else None

    def list_alerts(self):
        with self.lock:
            return list(self.active.values())
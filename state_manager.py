# state_manager.py
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple, Union

from config import DEBOUNCE_SECONDS

class StateManager:
    """Quản lý trạng thái của ứng dụng, đặc biệt là các cảnh báo."""
    def __init__(self):
        self.lock = threading.Lock()
        self.active = {}  # {alert_id: info}
        self.person_detection_enabled = True

    def is_person_detection_enabled(self) -> bool:
        """Kiểm tra xem tính năng nhận diện người có đang bật không."""
        with self.lock:
            return self.person_detection_enabled

    def set_person_detection_enabled(self, enabled: bool):
        """Bật hoặc tắt tính năng nhận diện người."""
        with self.lock:
            self.person_detection_enabled = enabled
            print(f"[StateManager] Person detection set to: {enabled}")

    def has_unresolved_alert(self, key: Union[str, Tuple]) -> bool:
        """Kiểm tra xem có cảnh báo nào chưa được giải quyết cho một key cụ thể không."""
        with self.lock:
            for alert in self.active.values():
                if alert['resolved']:
                    continue
                current_key = (alert['type'], alert['asked_for']) if alert['type'] == 'nguoi_quen' else alert['type']
                if current_key == key:
                    return True
        return False

    def create_alert(self, typ, chat_id, asked_for=None, image_path=None):
        """Tạo một cảnh báo mới và trả về ID của nó."""
        current_time = time.time()
        # Debounce nhanh để tránh tạo alert trùng lặp trong tích tắc
        for alert in self.active.values():
            if (alert['type'] == typ and str(alert['chat_id']) == str(chat_id) and
                not alert['resolved'] and (current_time - alert['ts'] < DEBOUNCE_SECONDS)):
                if typ != 'nguoi_quen' or alert['asked_for'] == asked_for:
                    return alert['id']

        aid = uuid.uuid4().hex
        info = {
            "id": aid, "type": typ, "chat_id": chat_id, "asked_for": asked_for,
            "ts": current_time, "resolved": False, "reply": None, "image_path": image_path
        }
        with self.lock:
            self.active[aid] = info
        return aid

    def resolve_alert(self, aid, reply_text):
        """Đánh dấu một cảnh báo là đã được giải quyết."""
        with self.lock:
            if aid in self.active:
                self.active[aid]["resolved"] = True
                self.active[aid]["reply"] = reply_text

    def get_alert_by_id(self, aid: str) -> Optional[Dict[str, Any]]:
        """Lấy thông tin của một cảnh báo bằng ID."""
        with self.lock:
            return self.active.get(aid)

    def latest_unresolved_for_chat(self, chat_id):
        """Tìm cảnh báo gần nhất chưa được giải quyết cho một chat_id."""
        with self.lock:
            cands = [a for a in self.active.values() if not a["resolved"] and str(a["chat_id"]) == str(chat_id)]
            cands.sort(key=lambda x: x["ts"], reverse=True)
            return cands[0] if cands else None

    def list_alerts(self):
        """Liệt kê tất cả các cảnh báo đã được tạo."""
        with self.lock:
            return list(self.active.values())
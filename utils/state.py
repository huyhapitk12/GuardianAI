# Quản lý trạng thái và xử lý cảnh báo
import threading
import time
from collections import deque
from config import settings, AlertType, AlertPriority


# Thông tin cảnh báo
class Alert:
    def __init__(self, id, type, timestamp, source_id=None, chat_id=None, 
                 image_path=None, name=None, resolved=False, resolution=None):
        self.id = id
        self.type = type
        self.timestamp = timestamp
        self.source_id = source_id
        self.chat_id = chat_id
        self.image_path = image_path
        self.name = name
        self.resolved = resolved
        self.resolution = resolution


# Quản lý trạng thái luồng an toàn
class StateManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._states = {}
        self._alerts = {}
        self._detection_global = True
        self._detection_cameras = {}
        self._alert_counter = 0
        self._unresolved = set()
    
    # Quản lý trạng thái
    def set(self, key, value):
        with self._lock:
            self._states[key] = value
    
    def get(self, key, default=None):
        with self._lock:
            return self._states.get(key, default)
    
    # Điều khiển phát hiện
    def is_detection_enabled(self, source_id=None):
        with self._lock:
            if source_id is None:
                return self._detection_global
            return self._detection_global and self._detection_cameras.get(source_id, True)
    
    def set_detection(self, enabled, source_id=None):
        with self._lock:
            if source_id is None:
                self._detection_global = enabled
            else:
                self._detection_cameras[source_id] = enabled
    

    
    # Quản lý cảnh báo
    def create_alert(self, alert_type, source_id=None,
                     chat_id=None, image_path=None,
                     name=None):
        with self._lock:
            self._alert_counter += 1
            alert_id = f"alert_{self._alert_counter}_{int(time.time())}"
            
            self._alerts[alert_id] = Alert(
                id=alert_id,
                type=alert_type,
                timestamp=time.time(),
                source_id=source_id,
                chat_id=chat_id,
                image_path=image_path,
                name=name
            )
            
            self._unresolved.add((alert_type, source_id or 'default'))
            return alert_id
    
    def resolve_alert(self, alert_id, resolution=None):
        with self._lock:
            alert = self._alerts.get(alert_id)
            if not alert:
                return False
            
            alert.resolved = True
            alert.resolution = resolution
            self._unresolved.discard((alert.type, alert.source_id or 'default'))
            return True
    
    def get_alert(self, alert_id):
        with self._lock:
            return self._alerts.get(alert_id)
    
    def list_alerts(self, limit=100):
        with self._lock:
            alerts = sorted(self._alerts.values(), key=lambda a: a.timestamp, reverse=True)
            return alerts[:limit]
    
    def has_unresolved(self, alert_type, source_id=None):
        with self._lock:
            key = (alert_type, source_id or 'default')
            return key in self._unresolved
    

# Ngăn chặn spam cảnh báo
class SpamGuard:
    def __init__(self):
        self._lock = threading.Lock()
        self._last_alert = {}
        self._history = deque(maxlen=100)
        self._muted = {}
        self.config = settings.spam_guard
    
    def allow(self, key, is_critical=False):
        now = time.time()
        
        with self._lock:
            # Check mute
            if self._muted.get(key, 0) > now:
                return False
            
            # Kiểm tra thời gian chờ
            if now - self._last_alert.get(key, 0) < self.config.debounce_seconds:
                return False
            
            # Kiểm tra khoảng thời gian tối thiểu (bỏ qua nếu khẩn cấp)
            if not is_critical and self._history:
                if now - self._history[-1][0] < self.config.min_interval:
                    return False
            
            # Kiểm tra giới hạn tần suất
            self._history = deque(
                [(t, k) for t, k in self._history if now - t < 60],
                maxlen=100
            )
            if len(self._history) >= self.config.max_per_minute:
                return False
            
            # Cho phép
            self._last_alert[key] = now
            self._history.append((now, key))
            return True
    
    def mute(self, key, duration):
        with self._lock:
            self._muted[key] = time.time() + duration


# Các instance toàn cục
state_manager = StateManager()
spam_guard = SpamGuard()
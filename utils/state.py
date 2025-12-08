"""State management và alert handling"""

from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import deque

from config import settings, AlertType, AlertPriority


@dataclass
class Alert:
    """Alert information"""
    id: str
    type: str
    timestamp: float
    source_id: Optional[str] = None
    chat_id: Optional[str] = None
    image_path: Optional[str] = None
    name: Optional[str] = None
    resolved: bool = False
    resolution: Optional[str] = None


class StateManager:
    """Thread-safe state management"""
    
    __slots__ = ('_lock', '_states', '_alerts', '_detection_global', 
                 '_detection_cameras', '_alert_counter', '_unresolved')
    
    def __init__(self):
        self._lock = threading.RLock()
        self._states: Dict[str, Any] = {}
        self._alerts: Dict[str, Alert] = {}
        self._detection_global = True
        self._detection_cameras: Dict[str, bool] = {}
        self._alert_counter = 0
        self._unresolved: Set[Tuple] = set()
    
    # State management
    def set(self, key: str, value: Any):
        with self._lock:
            self._states[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._states.get(key, default)
    
    # Detection control
    def is_detection_enabled(self, source_id: Optional[str] = None) -> bool:
        with self._lock:
            if source_id is None:
                return self._detection_global
            return self._detection_global and self._detection_cameras.get(source_id, True)
    
    def set_detection(self, enabled: bool, source_id: Optional[str] = None):
        with self._lock:
            if source_id is None:
                self._detection_global = enabled
            else:
                self._detection_cameras[source_id] = enabled
    
    # Aliases for compatibility
    def is_person_detection_enabled(self, source_id: Optional[str] = None) -> bool:
        return self.is_detection_enabled(source_id)
    
    def set_person_detection_enabled(self, enabled: bool, source_id: Optional[str] = None):
        self.set_detection(enabled, source_id)
    
    # Alert management
    def create_alert(self, alert_type: str, source_id: Optional[str] = None,
                     chat_id: Optional[str] = None, image_path: Optional[str] = None,
                     name: Optional[str] = None) -> str:
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
    
    def resolve_alert(self, alert_id: str, resolution: Optional[str] = None) -> bool:
        with self._lock:
            alert = self._alerts.get(alert_id)
            if not alert:
                return False
            
            alert.resolved = True
            alert.resolution = resolution
            self._unresolved.discard((alert.type, alert.source_id or 'default'))
            return True
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        with self._lock:
            return self._alerts.get(alert_id)
    
    def list_alerts(self, limit: int = 100) -> List[Alert]:
        with self._lock:
            alerts = sorted(self._alerts.values(), key=lambda a: a.timestamp, reverse=True)
            return alerts[:limit]
    
    def has_unresolved(self, key: Tuple) -> bool:
        with self._lock:
            return key in self._unresolved


class SpamGuard:
    """Prevent alert spam"""
    
    __slots__ = ('_lock', '_last_alert', '_history', '_muted', 'config')
    
    def __init__(self):
        self._lock = threading.Lock()
        self._last_alert: Dict[Union[str, Tuple], float] = {}
        self._history: deque = deque(maxlen=100)
        self._muted: Dict[Union[str, Tuple], float] = {}
        self.config = settings.spam_guard
    
    def allow(self, key: Union[str, Tuple], is_critical: bool = False) -> bool:
        now = time.time()
        
        with self._lock:
            # Check mute
            if self._muted.get(key, 0) > now:
                return False
            
            # Check debounce
            if now - self._last_alert.get(key, 0) < self.config.debounce_seconds:
                return False
            
            # Check min interval (skip for critical)
            if not is_critical and self._history:
                if now - self._history[-1][0] < self.config.min_interval:
                    return False
            
            # Check rate limit
            self._history = deque(
                [(t, k) for t, k in self._history if now - t < 60],
                maxlen=100
            )
            if len(self._history) >= self.config.max_per_minute:
                return False
            
            # Allow
            self._last_alert[key] = now
            self._history.append((now, key))
            return True
    
    def mute(self, key: Union[str, Tuple], duration: int):
        with self._lock:
            self._muted[key] = time.time() + duration


# Global instances
state_manager = StateManager()
spam_guard = SpamGuard()
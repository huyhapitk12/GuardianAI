"""State management for alerts and system status"""
import time
import threading
import uuid
from typing import Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from config.constants import DEBOUNCE_SECONDS

@dataclass
class AlertInfo:
    """Alert information"""
    id: str
    type: str
    chat_id: str
    timestamp: float
    resolved: bool = False
    reply: Optional[str] = None
    asked_for: Optional[str] = None
    image_path: Optional[str] = None

class StateManager:
    """Thread-safe state manager for the application"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._active_alerts: Dict[str, AlertInfo] = {}
        self._person_detection_enabled = True

    def is_person_detection_enabled(self) -> bool:
        """Check if person detection is enabled"""
        with self._lock:
            return self._person_detection_enabled

    def set_person_detection_enabled(self, enabled: bool) -> None:
        """Enable or disable person detection"""
        with self._lock:
            self._person_detection_enabled = enabled

    def has_unresolved_alert(self, key: Union[str, Tuple[str, str]]) -> bool:
        """Check if there's an unresolved alert for the given key"""
        with self._lock:
            for alert in self._active_alerts.values():
                if alert.resolved:
                    continue
                
                current_key = self._get_alert_key(alert)
                if current_key == key:
                    return True
        return False

    def create_alert(
        self,
        alert_type: str,
        chat_id: str,
        asked_for: Optional[str] = None,
        image_path: Optional[str] = None
    ) -> str:
        """Create a new alert and return its ID"""
        current_time = time.time()
        
        # Check for duplicate alerts (debounce)
        with self._lock:
            for alert in self._active_alerts.values():
                if (alert.type == alert_type and 
                    str(alert.chat_id) == str(chat_id) and 
                    not alert.resolved and 
                    (current_time - alert.timestamp < DEBOUNCE_SECONDS)):
                    
                    if alert_type == 'nguoi_quen' and alert.asked_for == asked_for:
                        return alert.id
                    elif alert_type != 'nguoi_quen':
                        return alert.id
            
            # Create new alert
            alert_id = uuid.uuid4().hex
            alert = AlertInfo(
                id=alert_id,
                type=alert_type,
                chat_id=chat_id,
                timestamp=current_time,
                asked_for=asked_for,
                image_path=image_path
            )
            self._active_alerts[alert_id] = alert
        
        return alert_id

    def resolve_alert(self, alert_id: str, reply_text: str) -> None:
        """Mark an alert as resolved"""
        with self._lock:
            if alert_id in self._active_alerts:
                self._active_alerts[alert_id].resolved = True
                self._active_alerts[alert_id].reply = reply_text

    def get_alert_by_id(self, alert_id: str) -> Optional[AlertInfo]:
        """Get alert information by ID"""
        with self._lock:
            return self._active_alerts.get(alert_id)

    def get_latest_unresolved_for_chat(self, chat_id: str) -> Optional[AlertInfo]:
        """Get the latest unresolved alert for a chat"""
        with self._lock:
            candidates = [
                alert for alert in self._active_alerts.values()
                if not alert.resolved and str(alert.chat_id) == str(chat_id)
            ]
            if not candidates:
                return None
            return max(candidates, key=lambda x: x.timestamp)

    def list_alerts(self) -> list:
        """Get all alerts"""
        with self._lock:
            return list(self._active_alerts.values())

    @staticmethod
    def _get_alert_key(alert: AlertInfo) -> Union[str, Tuple[str, str]]:
        """Get the key for an alert"""
        if alert.type == 'nguoi_quen':
            return (alert.type, alert.asked_for)
        return alert.type
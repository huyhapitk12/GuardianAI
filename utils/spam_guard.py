"""Spam protection for alerts"""
import time
import threading
from typing import Union, Tuple, Dict, List
from config.constants import (
    DEBOUNCE_SECONDS,
    SPAM_GUARD_MIN_INTERVAL,
    SPAM_GUARD_MAX_PER_MINUTE
)

class SpamGuard:
    """Prevents alert spam with multiple strategies"""
    
    def __init__(
        self,
        debounce_seconds: int = DEBOUNCE_SECONDS,
        min_interval: int = SPAM_GUARD_MIN_INTERVAL,
        max_per_minute: int = SPAM_GUARD_MAX_PER_MINUTE
    ):
        self.debounce_seconds = debounce_seconds
        self.min_interval = min_interval
        self.max_per_minute = max_per_minute
        
        self._lock = threading.Lock()
        self._last_alert_time: Dict[Union[str, Tuple], float] = {}
        self._alert_history: List[Tuple[float, Union[str, Tuple]]] = []
        self._muted_until: Dict[Union[str, Tuple], float] = {}

    def mute(self, key: Union[str, Tuple], duration_seconds: int) -> None:
        """Temporarily mute alerts for a specific key"""
        with self._lock:
            now = time.time()
            self._muted_until[key] = now + duration_seconds

    def allow(self, key: Union[str, Tuple]) -> bool:
        """Check if an alert with the given key is allowed"""
        now = time.time()
        
        with self._lock:
            # Check if muted
            if self._muted_until.get(key, 0) > now:
                return False

            # Check debounce for same key
            last_for_key = self._last_alert_time.get(key, 0)
            if now - last_for_key < self.debounce_seconds:
                return False

            # Check minimum interval between any alerts
            if self._alert_history and now - self._alert_history[-1][0] < self.min_interval:
                return False

            # Check max alerts per minute
            self._alert_history = [
                (t, k) for (t, k) in self._alert_history 
                if now - t < 60
            ]
            if len(self._alert_history) >= self.max_per_minute:
                return False

            # Allow the alert
            self._last_alert_time[key] = now
            self._alert_history.append((now, key))
            return True
"""Activity logging widgets"""

from __future__ import annotations
import threading
from collections import deque
from datetime import datetime
from typing import Literal, List, Deque

import customtkinter as ctk
from gui.styles import Colors, Fonts, Sizes


ActivityType = Literal["info", "success", "warning", "error", "detection", "alert"]
LogLevel = Literal["info", "success", "warning", "error"]


class ActivityLogger:
    """Singleton activity logger"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance
    
    def _init(self):
        self.activities: Deque = deque(maxlen=100)
        self.logs: Deque = deque(maxlen=200)
        self._activity_widgets: List = []
        self._log_widgets: List = []
    
    def log_activity(self, message: str, activity_type: ActivityType = "info"):
        entry = {'message': message, 'type': activity_type, 'time': datetime.now()}
        self.activities.append(entry)
        
        for widget in self._activity_widgets:
            try:
                widget.add_entry(message, activity_type)
            except Exception:
                pass
    
    def log_system(self, message: str, level: LogLevel = "info"):
        entry = {'message': message, 'level': level, 'time': datetime.now()}
        self.logs.append(entry)
        
        for widget in self._log_widgets:
            try:
                widget.add_entry(message, level)
            except Exception:
                pass
    
    def register_activity(self, widget):
        if widget not in self._activity_widgets:
            self._activity_widgets.append(widget)
    
    def unregister_activity(self, widget):
        if widget in self._activity_widgets:
            self._activity_widgets.remove(widget)
    
    def register_log(self, widget):
        if widget not in self._log_widgets:
            self._log_widgets.append(widget)
    
    def unregister_log(self, widget):
        if widget in self._log_widgets:
            self._log_widgets.remove(widget)


# Global logger instance
_logger = ActivityLogger()


def log_activity(message: str, activity_type: ActivityType = "info"):
    _logger.log_activity(message, activity_type)


def log_system(message: str, level: LogLevel = "info"):
    _logger.log_system(message, level)


# Removed ActivityFeed and SystemLogs classes
# Logging functionality is preserved for background tasks   
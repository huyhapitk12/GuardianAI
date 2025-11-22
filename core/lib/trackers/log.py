import os
import sys
from typing import Any, Dict, Final, Literal, Optional
import datetime

class PrintLogger:
    """A simple print-based logger to replace logging.Logger"""
    
    def __init__(self, name: str):
        self.name = name
        
    def _format_message(self, level: str, message: str) -> str:
        """Format message with timestamp and level"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"{timestamp} - {self.name} - {level} - {message}"
    
    def debug(self, message: str, *args, **kwargs):
        if os.environ.get("TRACKERS_LOG_LEVEL", "ERROR").upper() in ["DEBUG"]:
            print(self._format_message("DEBUG", message % args if args else message), file=sys.stderr)
    
    def info(self, message: str, *args, **kwargs):
        level_name = os.environ.get("TRACKERS_LOG_LEVEL", "ERROR").upper()
        if level_name in ["DEBUG", "INFO"]:
            print(self._format_message("INFO", message % args if args else message), file=sys.stderr)
    
    def warning(self, message: str, *args, **kwargs):
        level_name = os.environ.get("TRACKERS_LOG_LEVEL", "ERROR").upper()
        if level_name in ["DEBUG", "INFO", "WARNING"]:
            print(self._format_message("WARNING", message % args if args else message), file=sys.stderr)
    
    def error(self, message: str, *args, **kwargs):
        print(self._format_message("ERROR", message % args if args else message), file=sys.stderr)
    
    def critical(self, message: str, *args, **kwargs):
        print(self._format_message("CRITICAL", message % args if args else message), file=sys.stderr)

def get_logger(name: Optional[str]) -> PrintLogger:
    """
    Retrieves a print-based logger instance with the specified name.

    Args:
        name (str): The name for the logger, typically __name__.

    Returns:
        PrintLogger: Print-based logger instance.
    """
    return PrintLogger(name or "root")

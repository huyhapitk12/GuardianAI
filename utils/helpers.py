"""Common helper utilities"""

from __future__ import annotations
import gc
import os
import sys
import time
import threading
import functools
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import psutil

# Optional pygame
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None
    PYGAME_AVAILABLE = False

from config import settings


# ==================== DECORATORS ====================

def timed(name: Optional[str] = None):
    """Log execution time for slow functions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if elapsed > 0.1:
                print(f"⏱️ {name or func.__name__}: {elapsed:.3f}s")
            return result
        return wrapper
    return decorator


def retry(attempts: int = 3, delay: float = 1.0):
    """Retry on exception"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for i in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if i < attempts - 1:
                        time.sleep(delay)
            raise last_error
        return wrapper
    return decorator


def threaded(func: Callable) -> Callable:
    """Run function in background thread"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread
    return wrapper


# ==================== ALARM ====================

class AlarmPlayer:
    """Audio alarm player with fade-in"""
    
    __slots__ = ('_sound_file', '_stop_flag', '_thread')
    
    def __init__(self, sound_file: Path):
        self._sound_file = sound_file
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
    
    def initialize(self) -> bool:
        if not PYGAME_AVAILABLE:
            return False
        try:
            pygame.mixer.pre_init(22050, -16, 2, 512)
            pygame.mixer.init()
            return True
        except Exception:
            return False
    
    def play(self):
        if not PYGAME_AVAILABLE or not self._sound_file.exists():
            return
        
        try:
            if not pygame.mixer.music.get_busy():
                self._stop_flag.clear()
                pygame.mixer.music.load(str(self._sound_file))
                pygame.mixer.music.set_volume(settings.alarm.start_volume)
                pygame.mixer.music.play(loops=-1)
                self._thread = threading.Thread(target=self._fade_in, daemon=True)
                self._thread.start()
        except Exception as e:
            print(f"Alarm error: {e}")
    
    def stop(self):
        if not PYGAME_AVAILABLE:
            return
        
        try:
            self._stop_flag.set()
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
        except Exception:
            pass
    
    def _fade_in(self):
        if not PYGAME_AVAILABLE:
            return
            
        start = time.time()
        duration = settings.alarm.fade_duration
        start_vol = settings.alarm.start_volume
        max_vol = settings.alarm.max_volume
        
        while not self._stop_flag.is_set():
            elapsed = time.time() - start
            if elapsed >= duration:
                break
            
            vol = start_vol + (max_vol - start_vol) * (elapsed / duration)
            try:
                pygame.mixer.music.set_volume(min(vol, max_vol))
            except Exception:
                break
            time.sleep(0.1)
        
        if not self._stop_flag.is_set():
            try:
                pygame.mixer.music.set_volume(max_vol)
            except Exception:
                pass


# Global alarm instance
_alarm: Optional[AlarmPlayer] = None


def init_alarm() -> bool:
    global _alarm
    _alarm = AlarmPlayer(settings.paths.alarm_sound)
    return _alarm.initialize()


def play_alarm():
    if _alarm:
        _alarm.play()


def stop_alarm():
    if _alarm:
        _alarm.stop()


# ==================== MEMORY ====================

def get_memory_mb() -> float:
    """Get current process memory in MB"""
    return psutil.Process().memory_info().rss / 1024 / 1024


def cleanup_memory():
    """Force garbage collection"""
    gc.collect()


class MemoryMonitor:
    """Background memory cleanup"""
    
    __slots__ = ('_running', '_thread', '_threshold', '_interval')
    
    def __init__(self, threshold_mb: float = 800, interval: float = 60):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._threshold = threshold_mb
        self._interval = interval
    
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
    
    def _loop(self):
        while self._running:
            if get_memory_mb() > self._threshold:
                cleanup_memory()
            time.sleep(self._interval)


# ==================== THREAD POOL ====================

class TaskPool:
    """Simple thread pool for async tasks"""
    
    __slots__ = ('_executor', '_stats')
    
    def __init__(self, workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="Task")
        self._stats = {'submitted': 0, 'completed': 0, 'failed': 0}
    
    def submit(self, func: Callable, *args, **kwargs):
        """Submit task to pool"""
        try:
            future = self._executor.submit(func, *args, **kwargs)
            self._stats['submitted'] += 1
            future.add_done_callback(self._on_done)
            return future
        except Exception:
            return None
    
    def _on_done(self, future):
        try:
            future.result()
            self._stats['completed'] += 1
        except Exception:
            self._stats['failed'] += 1
    
    def shutdown(self, wait: bool = True):
        self._executor.shutdown(wait=wait)
    
    @property
    def stats(self) -> Dict[str, int]:
        return self._stats.copy()


# Global instances
memory_monitor = MemoryMonitor()
task_pool = TaskPool()  
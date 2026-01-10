import gc
import os
import time
import threading
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor
import psutil
import pygame
from config import settings

# Thử lại khi lỗi
def retry(attempts=3, delay=1.0):
    def decorator(func):
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


# Chạy hàm trong luồng nền
def threaded(func):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread
    return wrapper


# Phát âm thanh cảnh báo (fade-in)
class AlarmPlayer:
    def __init__(self, sound_file):
        self._sound_file = sound_file
        self._stop_flag = threading.Event()
        self._thread = None
    
    def initialize(self):
        pygame.mixer.pre_init(22050, -16, 2, 512)
        pygame.mixer.init()
        return True
    
    def play(self):
        if not self._sound_file.exists():
            return
        
        if not pygame.mixer.music.get_busy():
            self._stop_flag.clear()
            pygame.mixer.music.load(str(self._sound_file))
            pygame.mixer.music.set_volume(settings.alarm.start_volume)
            pygame.mixer.music.play(loops=-1)
            self._thread = threading.Thread(target=self._fade_in, daemon=True)
            self._thread.start()
    
    def stop(self):
        self._stop_flag.set()
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    
    def _fade_in(self):
        start = time.time()
        duration = settings.alarm.fade_in_duration
        start_vol = settings.alarm.start_volume
        max_vol = settings.alarm.max_volume
        
        while not self._stop_flag.is_set():
            elapsed = time.time() - start
            if elapsed >= duration:
                break
            
            vol = start_vol + (max_vol - start_vol) * (elapsed / duration)
            pygame.mixer.music.set_volume(min(vol, max_vol))
            time.sleep(0.1)
        
        if not self._stop_flag.is_set():
            pygame.mixer.music.set_volume(max_vol)


_alarm = None

def init_alarm():
    global _alarm
    _alarm = AlarmPlayer(settings.paths.alarm_sound)
    return _alarm.initialize()


def play_alarm():
    if _alarm:
        _alarm.play()


def stop_alarm():
    if _alarm:
        _alarm.stop()


# Lấy RAM sử dụng (MB)
def get_memory_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


# Ép buộc thu gom rác bộ nhớ
def cleanup_memory():
    gc.collect()


# Giám sát & dọn dẹp RAM
class MemoryMonitor:
    def __init__(self, threshold_mb=800, interval=60):
        self._running = False
        self._thread = None
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


# Thread pool đơn giản
class TaskPool:
    
    
    def __init__(self, workers=4):
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="Task")
        self._stats = {'submitted': 0, 'completed': 0, 'failed': 0}
    
    # Gửi task vào pool
    def submit(self, func, *args, **kwargs):
        future = self._executor.submit(func, *args, **kwargs)
        self._stats['submitted'] += 1
        future.add_done_callback(self._on_done)
        return future
    
    def _on_done(self, future):
        future.result()
        self._stats['completed'] += 1
    
    def shutdown(self, wait: bool = True):
        self._executor.shutdown(wait=wait)
    
    def stats(self):
        return self._stats.copy()


# Global instances
memory_monitor = MemoryMonitor()
task_pool = TaskPool()  
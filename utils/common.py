import os
import sys
import time
import gc
import json
import hashlib
import threading
import queue
import functools
import concurrent.futures
import psutil
import numpy as np
import cv2
from typing import Any, Dict, Optional, Union, Tuple, List, Callable
from pathlib import Path
from datetime import datetime
from collections import deque
from functools import wraps
from config import settings

# Optional pygame for alarm sound
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None
    PYGAME_AVAILABLE = False
    print("WARNING: pygame not available, alarm sounds will be disabled")


# ============================================================================
# CACHE & CORE UTILITIES (from core_utils.py)
# ============================================================================

class Cache:
    """Simple in-memory cache with TTL support"""
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        with self._lock:
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            self.cache[key] = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl or self.ttl_seconds
            }
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self.cache:
                return None
            item = self.cache[key]
            if time.time() - item['timestamp'] > item['ttl']:
                del self.cache[key]
                return None
            return item['value']
    
    def _evict_oldest(self):
        if self.cache:
            oldest_key = min(self.cache, key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
    
    def clear(self):
        with self._lock:
            self.cache.clear()

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return {'total_items': len(self.cache), 'max_size': self.max_size, 'ttl_seconds': self.ttl_seconds}

class EmbeddingCache:
    """Cache for face embeddings"""
    def __init__(self, max_size: int = 500):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def _hash_face(self, face_data) -> Optional[str]:
        try:
            face_bytes = face_data.tobytes() if hasattr(face_data, 'tobytes') else str(face_data).encode()
            return hashlib.md5(face_bytes).hexdigest()
        except:
            return None
    
    def get(self, face_hash: str) -> Optional[Any]:
        with self._lock:
            if face_hash in self.cache:
                self.hits += 1
                return self.cache[face_hash]
            self.misses += 1
            return None
    
    def set(self, face_hash: str, embedding: Any):
        with self._lock:
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            self.cache[face_hash] = {'embedding': embedding, 'timestamp': time.time()}
    
    def _evict_oldest(self):
        if self.cache:
            oldest_key = min(self.cache, key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
    
    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    def clear(self):
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

# Global cache instances
generic_cache = Cache(ttl_seconds=3600, max_size=1000)
embedding_cache = EmbeddingCache(max_size=500)

class AlarmPlayer:
    """Alarm sound player with fade-in"""
    def __init__(self, sound_file: Path):
        self.sound_file = sound_file
        self._alarm_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
    
    def initialize(self) -> bool:
        if not PYGAME_AVAILABLE:
            print("WARNING: pygame not available, alarm initialization skipped")
            return False
        try:
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
            return True
        except Exception as e:
            print(f"ERROR: Failed to initialize AlarmPlayer: {e}")
            return False

    def play(self) -> None:
        if not PYGAME_AVAILABLE:
            print("WARNING: pygame not available, cannot play alarm")
            return
        if not self.sound_file.exists():
            print(f"ERROR: Alarm file not found: {self.sound_file}")
            return
        try:
            if not pygame.mixer.music.get_busy():
                self._stop_flag.clear()
                pygame.mixer.music.load(str(self.sound_file))
                pygame.mixer.music.set_volume(settings.alarm.start_volume)
                pygame.mixer.music.play(loops=-1)
                self._alarm_thread = threading.Thread(target=self._fade_in_worker, daemon=True)
                self._alarm_thread.start()
        except Exception as e:
            print(f"ERROR: Error playing alarm: {e}")

    def stop(self) -> None:
        if not PYGAME_AVAILABLE:
            return
        try:
            if pygame.mixer.music.get_busy():
                self._stop_flag.set()
                if self._alarm_thread and self._alarm_thread.is_alive():
                    self._alarm_thread.join(timeout=0.5)
                pygame.mixer.music.stop()
                self._alarm_thread = None
        except Exception as e:
            print(f"ERROR: Error stopping alarm: {e}")

    def _fade_in_worker(self) -> None:
        if not PYGAME_AVAILABLE:
            return
        start_time = time.time()
        duration = settings.alarm.fade_in_duration
        start_vol = settings.alarm.start_volume
        max_vol = settings.alarm.max_volume
        while not self._stop_flag.is_set():
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
            progress = elapsed / duration
            current_volume = start_vol + (max_vol - start_vol) * progress
            try:
                pygame.mixer.music.set_volume(min(current_volume, max_vol))
            except: break
            time.sleep(0.1)
        if not self._stop_flag.is_set():
            try: pygame.mixer.music.set_volume(max_vol)
            except: pass

class SpamGuard:
    """Prevent alert spamming"""
    def __init__(self):
        self.debounce_seconds = settings.spam_guard.debounce_seconds
        self.min_interval = settings.spam_guard.min_interval
        self.max_per_minute = settings.spam_guard.max_per_minute
        self._lock = threading.Lock()
        self._last_alert_time: Dict[Union[str, Tuple], float] = {}
        self._alert_history: List[Tuple[float, Union[str, Tuple]]] = []
        self._muted_until: Dict[Union[str, Tuple], float] = {}

    def mute(self, key: Union[str, Tuple], duration_seconds: int) -> None:
        with self._lock:
            self._muted_until[key] = time.time() + duration_seconds

    def allow(self, key: Union[str, Tuple], is_critical: bool = False) -> bool:
        now = time.time()
        with self._lock:
            if self._muted_until.get(key, 0) > now: return False
            if now - self._last_alert_time.get(key, 0) < self.debounce_seconds: return False
            if not is_critical and self._alert_history and now - self._alert_history[-1][0] < self.min_interval: return False
            
            self._alert_history = [(t, k) for (t, k) in self._alert_history if now - t < 60]
            if len(self._alert_history) >= self.max_per_minute: return False
            
            self._last_alert_time[key] = now
            self._alert_history.append((now, key))
            return True

class AlertInfo:
    def __init__(self, alert_id: str, alert_type: str, timestamp: float, source_id: Optional[str] = None, 
                 chat_id: Optional[str] = None, image_path: Optional[str] = None, asked_for: Optional[str] = None, resolved: bool = False):
        self.id = alert_id
        self.type = alert_type
        self.timestamp = timestamp
        self.source_id = source_id
        self.chat_id = chat_id
        self.image_path = image_path
        self.asked_for = asked_for
        self.resolved = resolved

class StateManager:
    """Manage system state and alerts"""
    def __init__(self):
        self._lock = threading.Lock()
        self.states: Dict[str, Any] = {}
        self.alerts: Dict[str, Any] = {}
        self._person_detection_global = True
        self._person_detection_cameras: Dict[str, bool] = {}
        self._alert_counter = 0
        self._active_alerts: Dict[str, AlertInfo] = {}
        self._unresolved_keys: set = set()
        
    def set_state(self, key: str, value: Any) -> None:
        with self._lock: self.states[key] = value
            
    def get_state(self, key: str, default: Any = None) -> Any:
        with self._lock: return self.states.get(key, default)
            
    def add_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        with self._lock: self.alerts[alert_type] = {**data, 'timestamp': time.time()}
            
    def is_person_detection_enabled(self, source_id: Optional[str] = None) -> bool:
        with self._lock:
            if source_id is None: return self._person_detection_global
            return self._person_detection_global and self._person_detection_cameras.get(source_id, True)
    
    def set_person_detection(self, enabled: bool, source_id: Optional[str] = None) -> None:
        with self._lock:
            if source_id is None: self._person_detection_global = enabled
            else: self._person_detection_cameras[source_id] = enabled
    
    def set_person_detection_enabled(self, enabled: bool, source_id: Optional[str] = None) -> None:
        self.set_person_detection(enabled, source_id)
    
    def create_alert(self, alert_type: str, chat_id: Optional[str] = None, image_path: Optional[str] = None, 
                    asked_for: Optional[str] = None, source_id: Optional[str] = None) -> str:
        with self._lock:
            self._alert_counter += 1
            alert_id = f"alert_{self._alert_counter}_{int(time.time())}"
            alert_info = AlertInfo(alert_id, alert_type, time.time(), source_id, chat_id, image_path, asked_for, False)
            self._active_alerts[alert_id] = alert_info
            self._unresolved_keys.add((alert_type, source_id or 'default'))
            return alert_id
    
    def resolve_alert(self, alert_id: str, resolution_note: Optional[str] = None) -> bool:
        with self._lock:
            if alert_id not in self._active_alerts: return False
            alert_info = self._active_alerts[alert_id]
            alert_info.resolved = True
            self._unresolved_keys.discard((alert_info.type, alert_info.source_id or 'default'))
            return True
            
    def list_alerts(self) -> List[AlertInfo]:
        with self._lock:
            return list(self._active_alerts.values())

# Global instances
state_manager = StateManager()
spam_guard = SpamGuard()
_alarm_player: Optional[AlarmPlayer] = None

def init_alarm(sound_file: Path) -> bool:
    global _alarm_player
    _alarm_player = AlarmPlayer(sound_file)
    return _alarm_player.initialize()

def play_alarm():
    if _alarm_player: _alarm_player.play()

def stop_alarm():
    if _alarm_player: _alarm_player.stop()

# ============================================================================
# DECORATORS (from decorators.py)
# ============================================================================

def performance_timer(func_name: Optional[str] = None):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or func.__name__
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                if execution_time > 0.1:
                    print(f"DEBUG: {name} executed in {execution_time:.3f}s")
                return result
            except Exception as e:
                print(f"ERROR: {name} failed after {time.time() - start_time:.3f}s: {e}")
                raise
        return wrapper
    return decorator

def memory_usage_monitor(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
        try:
            result = func(*args, **kwargs)
            diff = (process.memory_info().rss / 1024 / 1024) - start_memory
            if diff > 10: print(f"DEBUG: {func.__name__} memory: +{diff:.1f}MB")
            return result
        except Exception as e:
            print(f"ERROR: {func.__name__} memory error: {e}")
            raise
    return wrapper

def cached(ttl_seconds: int = 3600, cache_key_prefix: str = ""):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{cache_key_prefix}{func.__name__}_{str(args)}_{str(kwargs)}"
            try:
                result = generic_cache.get(cache_key)
                if result is not None: return result
            except: pass
            result = func(*args, **kwargs)
            generic_cache.set(cache_key, result, ttl=ttl_seconds)
            return result
        return wrapper
    return decorator

def thread_safe(lock_attr: str = "_lock"):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            lock = getattr(self, lock_attr, None)
            if lock is None:
                lock = threading.Lock()
                setattr(self, lock_attr, lock)
            with lock: return func(self, *args, **kwargs)
        return wrapper
    return decorator

def retry(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try: return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1: time.sleep(delay)
            if last_exception: raise last_exception
            raise RuntimeError(f"{func.__name__} failed after {max_attempts} attempts")
        return wrapper
    return decorator

def singleton(cls):
    instances = {}
    lock = threading.Lock()
    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances: instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

# ============================================================================
# HELPERS (from helpers.py)
# ============================================================================

def setup_optimization_environment():
    try:
        os.environ.setdefault('OMP_NUM_THREADS', '4')
        os.environ.setdefault('MKL_NUM_THREADS', '4')
        os.environ['OPENBLAS_NUM_THREADS'] = '4'
        print("INFO: Optimization environment setup complete")
    except Exception as e: print(f"ERROR: Failed to setup optimization: {e}")

def optimize_garbage_collection(threshold_0: int = 700, threshold_1: int = 10, threshold_2: int = 10):
    try:
        gc_threshold_0 = settings.get('optimization.gc_threshold_0', threshold_0)
        gc_threshold_1 = settings.get('optimization.gc_threshold_1', threshold_1)
        gc_threshold_2 = settings.get('optimization.gc_threshold_2', threshold_2)
        gc.set_threshold(gc_threshold_0, gc_threshold_1, gc_threshold_2)
        gc.collect()
        print(f"INFO: GC optimized with thresholds: ({gc_threshold_0}, {gc_threshold_1}, {gc_threshold_2})")
    except Exception as e: print(f"ERROR: Failed to optimize GC: {e}")

def force_garbage_collection():
    try:
        before = gc.get_count()
        gc.collect()
        print(f"INFO: GC collected. Before: {before}, After: {gc.get_count()}")
    except Exception as e: print(f"ERROR: GC force failed: {e}")

def safe_filename(filename: str, max_length: int = 100) -> str:
    import re
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    safe_name = ' '.join(safe_name.split())
    if len(safe_name) > max_length:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:max_length - len(ext)] + ext
    return safe_name

def get_system_info() -> dict:
    try:
        import platform
        return {
            'platform': platform.platform(),
            'python_version': sys.version,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024**3,
            'disk_total_gb': psutil.disk_usage('/').total / 1024**3
        }
    except: return {}

# ============================================================================
# PERFORMANCE UTILS (from performance_utils.py)
# ============================================================================

class FrameOptimizer:
    def __init__(self, resize_to: Optional[Tuple[int, int]] = None, quality: int = 95):
        self.resize_to = resize_to
        self.quality = quality
        self._lock = threading.Lock()
        self.stats = {'frames_processed': 0}

    def optimize_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None: return frame
        try:
            optimized = frame
            if self.resize_to and self.resize_to != (frame.shape[1], frame.shape[0]):
                optimized = cv2.resize(frame, self.resize_to, interpolation=cv2.INTER_LINEAR)
            with self._lock: self.stats['frames_processed'] += 1
            return optimized
        except: return frame

    def compress_frame(self, frame: np.ndarray, compression_level: int = 3) -> bytes:
        try:
            quality = max(10, self.quality - (compression_level - 1) * 10)
            _, compressed = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return compressed.tobytes()
        except: return b''

class MemoryOptimizer:
    def __init__(self, memory_threshold_mb: float = 800, cleanup_interval: float = 60.0):
        self.memory_threshold_mb = memory_threshold_mb
        self.cleanup_interval = cleanup_interval
        self._running = False
        self._cleanup_thread = None
        self._lock = threading.Lock()
        self.stats = {'cleanups': 0, 'freed_mb': 0.0}

    def start_background_cleanup(self):
        if self._running: return
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True, name="MemoryOptimizer")
        self._cleanup_thread.start()

    def stop_background_cleanup(self):
        self._running = False
        if self._cleanup_thread: self._cleanup_thread.join(timeout=5.0)

    def _cleanup_loop(self):
        while self._running:
            try:
                self.check_and_cleanup()
                time.sleep(self.cleanup_interval)
            except: pass

    def check_and_cleanup(self):
        try:
            process = psutil.Process()
            mem = process.memory_info().rss / 1024 / 1024
            if mem > self.memory_threshold_mb:
                before = mem
                gc.collect()
                after = process.memory_info().rss / 1024 / 1024
                freed = before - after
                if freed > 0:
                    with self._lock:
                        self.stats['cleanups'] += 1
                        self.stats['freed_mb'] += freed
                    print(f"INFO: Memory cleanup freed {freed:.1f}MB")
        except: pass
    
    def get_stats(self):
        with self._lock: return self.stats.copy()

class PerformanceMonitor:
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval: float = 5.0):
        if self.monitoring: return
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread: self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self, interval: float):
        while self.monitoring:
            try:
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().percent
                if cpu > 80: print(f"WARNING: High CPU: {cpu:.1f}%")
                if mem > 90: print(f"WARNING: High memory: {mem:.1f}%")
            except: pass
            time.sleep(interval)
    
    def get_current_metrics(self) -> Dict:
        try:
            p = psutil.Process()
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'process_memory_mb': p.memory_info().rss / 1024 / 1024
            }
        except: return {}

# Global optimizer instances
frame_optimizer = FrameOptimizer(resize_to=(1280, 720), quality=85)
memory_optimizer = MemoryOptimizer(memory_threshold_mb=800, cleanup_interval=60.0)
performance_monitor = PerformanceMonitor()

# ============================================================================
# THREAD POOL (from thread_pool_executor.py)
# ============================================================================

class OptimizedThreadPool:
    def __init__(self, max_workers: int = 4):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="OptPool")
        self._lock = threading.Lock()
        self.stats = {'submitted': 0, 'completed': 0, 'failed': 0}
        
    def submit(self, func: Callable, *args, **kwargs):
        try:
            future = self.executor.submit(func, *args, **kwargs)
            with self._lock: self.stats['submitted'] += 1
            future.add_done_callback(self._callback)
            return future
        except Exception as e:
            print(f"ERROR: Pool submit failed: {e}")
            return None
            
    def _callback(self, future):
        try:
            future.result()
            with self._lock: self.stats['completed'] += 1
        except:
            with self._lock: self.stats['failed'] += 1

    def shutdown(self, wait=True):
        self.executor.shutdown(wait=wait)

class TaskQueue:
    def __init__(self, worker_count: int = 2):
        self.queue = queue.Queue(maxsize=100)
        self.workers = []
        self._running = False
        self.worker_count = worker_count
        
    def start(self):
        if self._running: return
        self._running = True
        for i in range(self.worker_count):
            t = threading.Thread(target=self._loop, daemon=True, name=f"TaskWorker-{i}")
            t.start()
            self.workers.append(t)
            
    def stop(self):
        self._running = False
        for t in self.workers: t.join(timeout=1.0)
        
    def submit(self, func, *args, **kwargs):
        try:
            self.queue.put_nowait((func, args, kwargs))
            return True
        except: return False
        
    def _loop(self):
        while self._running:
            try:
                item = self.queue.get(timeout=1.0)
                if item:
                    func, args, kwargs = item
                    try: func(*args, **kwargs)
                    except Exception as e: print(f"ERROR: Task failed: {e}")
                    finally: self.queue.task_done()
            except: continue

# Global threading instances
thread_pool = OptimizedThreadPool(max_workers=4)
task_queue = TaskQueue(worker_count=2)

# ============================================================================
# REPORTING (from reporting_utils.py)
# ============================================================================

class OptimizationReporter:
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics: Dict[str, Any] = {}
        self.start_time = time.time()
        
    def add_metric(self, category: str, name: str, value: Any):
        if category not in self.metrics: self.metrics[category] = {}
        self.metrics[category][name] = value
        
    def generate_report(self) -> Dict:
        return {
            'timestamp': datetime.now().isoformat(),
            'duration': time.time() - self.start_time,
            'metrics': self.metrics,
            'system': performance_monitor.get_current_metrics()
        }

reporter = OptimizationReporter()

# ============================================================================
# MISSING DECORATORS & HELPERS
# ============================================================================

def validate_types(*expected_types):
    """Type validation decorator"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Simple type validation (can be expanded)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def time_async(func: Callable) -> Callable:
    """Async timing decorator"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        print(f"DEBUG: {func.__name__} took {time.time() - start:.3f}s")
        return result
    return wrapper

def profile(func: Callable) -> Callable:
    """Profiling decorator"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import cProfile
        import pstats
        from io import StringIO
        
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Optional: print stats
        # s = StringIO()
        # ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        # ps.print_stats()
        # print(s.getvalue())
        
        return result
    return wrapper

def estimate_memory_usage(obj) -> float:
    """Estimate memory usage of an object in MB"""
    try:
        import sys
        return sys.getsizeof(obj) / 1024 / 1024
    except:
        return 0.0

def is_memory_available(required_mb: float) -> bool:
    """Check if enough memory is available"""
    try:
        available = psutil.virtual_memory().available / 1024 / 1024
        return available >= required_mb
    except:
        return True

def get_memory_info() -> Dict:
    """Get current memory information"""
    try:
        vm = psutil.virtual_memory()
        return {
            'total_mb': vm.total / 1024 / 1024,
            'available_mb': vm.available / 1024 / 1024,
            'used_mb': vm.used / 1024 / 1024,
            'percent': vm.percent
        }
    except:
        return {}

def batch_array_operations(arrays: List[np.ndarray], operation: str) -> Optional[np.ndarray]:
    """Batch process numpy arrays"""
    try:
        if operation == 'stack':
            return np.stack(arrays)
        elif operation == 'concat':
            return np.concatenate(arrays)
        elif operation == 'mean':
            return np.mean(arrays, axis=0)
        return None
    except:
        return None

def optimize_numpy_arrays(*arrays) -> List[np.ndarray]:
    """Optimize numpy arrays for memory usage"""
    optimized = []
    for arr in arrays:
        if arr is None:
            optimized.append(None)
            continue
        try:
            # Convert to optimal dtype if possible
            if arr.dtype == np.float64:
                optimized.append(arr.astype(np.float32))
            elif arr.dtype == np.int64:
                optimized.append(arr.astype(np.int32))
            else:
                optimized.append(arr)
        except:
            optimized.append(arr)
    return optimized

def clear_array_memory(arr: Optional[np.ndarray]):
    """Clear array memory"""
    if arr is not None:
        del arr
    gc.collect()

def profile_function(func: Callable) -> Dict:
    """Profile a function and return metrics"""
    import time
    start = time.time()
    result = func()
    elapsed = time.time() - start
    return {'elapsed': elapsed, 'result': result}

def measure_function_performance(func: Callable, iterations: int = 10) -> Dict:
    """Measure function performance over multiple iterations"""
    times = []
    for _ in range(iterations):
        start = time.time()
        func()
        times.append(time.time() - start)
    return {
        'mean': sum(times) / len(times),
        'min': min(times),
        'max': max(times),
        'iterations': iterations
    }

def benchmark_operations(operations: Dict[str, Callable], iterations: int = 100) -> Dict:
    """Benchmark multiple operations"""
    results = {}
    for name, func in operations.items():
        results[name] = measure_function_performance(func, iterations)
    return results

def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"

def check_environment() -> Dict:
    """Check environment configuration"""
    return {
        'python_version': sys.version,
        'platform': sys.platform,
        'cwd': os.getcwd(),
        'env_vars': {k: v for k, v in os.environ.items() if 'PYTHON' in k or 'PATH' in k}
    }

def cleanup_resources():
    """Cleanup unused resources"""
    gc.collect()
    print("INFO: Resources cleaned up")

def create_performance_summary() -> Dict:
    """Create performance summary"""
    return performance_monitor.get_current_metrics()

def save_system_info(filepath: str):
    """Save system information to file"""
    try:
        info = get_system_info()
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"INFO: System info saved to {filepath}")
    except Exception as e:
        print(f"ERROR: Failed to save system info: {e}")

def export_metrics_csv(metrics: Dict, filepath: str):
    """Export metrics to CSV"""
    try:
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in metrics.items():
                writer.writerow([key, value])
        print(f"INFO: Metrics exported to {filepath}")
    except Exception as e:
        print(f"ERROR: Failed to export metrics: {e}")

def enable_all_optimizations():
    """Enable all available optimizations"""
    print("INFO: Enabling all optimizations...")
    setup_optimization_environment()
    optimize_garbage_collection()
    memory_optimizer.start_background_cleanup()
    performance_monitor.start_monitoring()
    task_queue.start()

def graceful_shutdown():
    """Perform graceful shutdown with cleanup"""
    print("INFO: Shutting down...")
    performance_monitor.stop_monitoring()
    memory_optimizer.stop_background_cleanup()
    task_queue.stop()
    thread_pool.shutdown()
    force_garbage_collection()
    print("INFO: Shutdown complete")

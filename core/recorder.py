"""Video recording"""

from __future__ import annotations
import cv2
import time
import uuid
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from config import settings
from utils import security


class Recorder:
    """Video recorder with encryption"""
    
    __slots__ = ('_lock', '_current', '_out_dir', '_fps', '_fourcc')
    
    def __init__(self):
        self._lock = threading.Lock()
        self._current: Optional[Dict[str, Any]] = None
        self._out_dir = settings.paths.tmp_dir
        self._fps = settings.recorder.fps
        self._fourcc = cv2.VideoWriter.fourcc(*settings.recorder.fourcc)
    
    @property
    def current(self) -> Optional[Dict]:
        with self._lock:
            return self._current
    
    def start(self, source_id: str, reason: str = "alert", 
              duration: int = None, wait_for_user: bool = False) -> Optional[Dict]:
        """Start recording"""
        duration = duration or settings.recorder.duration
        
        with self._lock:
            if self._current is not None:
                return None
            
            filename = f"rec_{reason}_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
            path = self._out_dir / filename
            
            self._current = {
                'path': path,
                'writer': None,
                'end_time': time.time() + duration,
                'source_id': source_id,
                'reason': reason,
                'alert_ids': [],
                'wait_for_user': wait_for_user,
            }
            
            return self._current
    
    def write(self, frame: np.ndarray) -> bool:
        """Write frame"""
        with self._lock:
            if self._current is None:
                return False
            
            # Initialize writer on first frame
            if self._current['writer'] is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(self._current['path']),
                    self._fourcc,
                    self._fps,
                    (w, h)
                )
                if not writer.isOpened():
                    self._current = None
                    return False
                self._current['writer'] = writer
            
            self._current['writer'].write(frame)
            return True
    
    def check_finalize(self) -> Optional[Dict]:
        """Check if recording should end"""
        with self._lock:
            if self._current is None:
                return None
            
            if time.time() < self._current['end_time']:
                return None
            
            if self._current.get('wait_for_user'):
                return None
            
            return self._finalize()
    
    def _finalize(self) -> Dict:
        """Finalize recording (must hold lock)"""
        rec = self._current
        self._current = None
        
        if rec['writer']:
            rec['writer'].release()
        
        # Encrypt file
        if rec['path'].exists():
            security.encrypt_file(rec['path'])
        
        return {
            'path': rec['path'],
            'source_id': rec['source_id'],
            'alert_ids': rec['alert_ids'],
        }
    
    def stop(self):
        """Stop recording"""
        with self._lock:
            if self._current:
                self._current['end_time'] = time.time()
                self._current['wait_for_user'] = False
    
    def discard(self) -> bool:
        """Discard current recording"""
        with self._lock:
            if self._current is None:
                return False
            
            if self._current['writer']:
                self._current['writer'].release()
            
            if self._current['path'].exists():
                self._current['path'].unlink()
            
            self._current = None
            return True
    
    def extend(self, seconds: float):
        """Extend recording duration"""
        with self._lock:
            if self._current:
                self._current['end_time'] += seconds
    
    def resolve_user_wait(self):
        """Allow finalization after user response"""
        with self._lock:
            if self._current:
                self._current['wait_for_user'] = False
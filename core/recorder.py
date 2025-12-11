"""Video recording"""

from __future__ import annotations
import cv2
import time
import uuid
import threading
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

from config import settings
from utils import security


class Recorder:
    """Video recorder with encryption"""
    
    __slots__ = ('lock', 'active_recording', 'out_dir', 'fps', 'fourcc')
    
    def __init__(self):
        self.lock = threading.Lock()
        self.active_recording: Optional[Dict[str, Any]] = None
        self.out_dir = settings.paths.tmp_dir
        self.fps = settings.recorder.fps
        self.fourcc = cv2.VideoWriter.fourcc(*settings.recorder.fourcc)
    
    @property
    def current(self) -> Optional[Dict]:
        with self.lock:
            return self.active_recording
    
    def start(self, source_id: str, reason: str = "alert", 
              duration: int = None, wait_for_user: bool = False) -> Optional[Dict]:
        """Start recording"""
        duration = duration or settings.recorder.duration
        
        with self.lock:
            if self.active_recording is not None:
                return None
            
            filename = f"rec_{reason}_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
            path = self.out_dir / filename
            
            self.active_recording = {
                'path': path,
                'writer': None,
                'end_time': time.time() + duration,
                'source_id': source_id,
                'reason': reason,
                'alert_ids': [],
                'wait_for_user': wait_for_user,
            }
            
            return self.active_recording
    
    def write(self, frame: np.ndarray) -> bool:
        """Write frame"""
        with self.lock:
            if self.active_recording is None:
                return False
            
            # Initialize writer on first frame
            if self.active_recording['writer'] is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(self.active_recording['path']),
                    self.fourcc,
                    self.fps,
                    (w, h)
                )
                if not writer.isOpened():
                    self.active_recording = None
                    return False
                self.active_recording['writer'] = writer
            
            self.active_recording['writer'].write(frame)
            return True
    
    def check_finalize(self) -> Optional[Dict]:
        """Check if recording should end"""
        with self.lock:
            if self.active_recording is None:
                return None
            
            if time.time() < self.active_recording['end_time']:
                return None
            
            if self.active_recording.get('wait_for_user'):
                return None
            
            return self.finalize()
    
    def finalize(self) -> Dict:
        """Finalize recording (must hold lock)"""
        rec = self.active_recording
        self.active_recording = None
        
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
        with self.lock:
            if self.active_recording:
                self.active_recording['end_time'] = time.time()
                self.active_recording['wait_for_user'] = False
    
    def discard(self) -> bool:
        """Discard current recording"""
        with self.lock:
            if self.active_recording is None:
                return False
            
            if self.active_recording['writer']:
                self.active_recording['writer'].release()
            
            if self.active_recording['path'].exists():
                self.active_recording['path'].unlink()
            
            self.active_recording = None
            return True
    
    def extend(self, seconds: float):
        """Extend recording duration"""
        with self.lock:
            if self.active_recording:
                self.active_recording['end_time'] += seconds
    
    def resolve_user_wait(self):
        """Allow finalization after user response"""
        with self.lock:
            if self.active_recording:
                self.active_recording['wait_for_user'] = False
"""Multi-camera management"""

from __future__ import annotations
import threading
from typing import Callable, Dict, Optional
import numpy as np

from config import settings
from core.camera import Camera
from core.detection import BehaviorAnalyzer, PersonTracker


class CameraManager:
    """Manage multiple cameras"""
    
    def __init__(self, on_person_alert: Callable = None, on_fire_alert: Callable = None):
        self.cameras: Dict[str, Camera] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        
        self._fire_detector = None
        self._face_detector = None
        self._behavior_analyzer: Optional[BehaviorAnalyzer] = None
        self._state = None
        
        self._on_person = on_person_alert
        self._on_fire = on_fire_alert
        
        self._create_cameras()
    

    def _create_cameras(self):
        """Create camera instances with shared model"""
        for source in settings.camera.sources:
            try:
                src = int(source) if source.isdigit() else source
                cam = Camera(src, self._on_person, self._on_fire, shared_model=None)
                self.cameras[str(source)] = cam
                print(f"✅ Camera created: {source}")
            except Exception as e:
                print(f"❌ Camera create failed {source}: {e}")
    
    def start(self, fire_detector, face_detector, state_manager, behavior_analyzer=None):
        """Start all cameras"""
        self._fire_detector = fire_detector
        self._face_detector = face_detector
        self._behavior_analyzer = behavior_analyzer
        self._state = state_manager
        
        with self._lock:
            for source, cam in self.cameras.items():
                cam.start_workers(fire_detector, face_detector, behavior_analyzer)
                
                thread = threading.Thread(
                    target=cam.process_loop,
                    args=(state_manager,),
                    daemon=True
                )
                self._threads[source] = thread
                thread.start()
                print(f"✅ Camera {source} started")
    
    def stop(self):
        """Stop all cameras"""
        print("Stopping cameras...")
        
        with self._lock:
            for cam in self.cameras.values():
                cam.quit = True
        
        for thread in self._threads.values():
            thread.join(timeout=5.0)
        
        print("✅ All cameras stopped")
    
    def get_camera(self, source: str) -> Optional[Camera]:
        with self._lock:
            return self.cameras.get(source)
    
    def get_all_frames(self) -> Dict[str, tuple]:
        frames = {}
        with self._lock:
            for source, cam in self.cameras.items():
                frames[source] = cam.read()
        return frames
    
    def get_status(self) -> Dict[str, bool]:
        status = {}
        with self._lock:
            for source, cam in self.cameras.items():
                status[source] = cam.get_connection_status()
        return status
    
    def add_camera(self, source: str) -> tuple:
        with self._lock:
            if source in self.cameras:
                return False, "Camera already exists"
        
        try:
            src = int(source) if source.isdigit() else source
            cam = Camera(src, self._on_person, self._on_fire, shared_model=None)
            
            if self._fire_detector and self._face_detector:
                cam.start_workers(self._fire_detector, self._face_detector, self._behavior_analyzer)
                
                thread = threading.Thread(
                    target=cam.process_loop,
                    args=(self._state,),
                    daemon=True
                )
                
                with self._lock:
                    self.cameras[source] = cam
                    self._threads[source] = thread
                
                thread.start()
                
                # Save to config for persistence
                from config.settings import add_camera_source_to_config
                save_success, save_msg = add_camera_source_to_config(source)
                if not save_success:
                    print(f"⚠️ Warning: Camera added to runtime but not saved to config: {save_msg}")
            
            return True, f"Camera {source} added"
            
        except Exception as e:
            return False, str(e)
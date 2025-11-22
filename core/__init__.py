# core/__init__.py
from .camera import Camera
from .camera_manager import CameraManager
from .recorder import Recorder, compress_video
from .detection import FaceDetector, FireDetector, PersonTracker

__all__ = ['Camera', 'CameraManager', 'Recorder', 'compress_video', 'FaceDetector', 'FireDetector', 'PersonTracker']
"""Core detection and processing modules"""
from .camera import Camera
from .recorder import Recorder, compress_video
from .detection import FaceDetector, FireDetector, PersonTracker

__all__ = ['Camera', 'Recorder', 'compress_video', 'FaceDetector', 'FireDetector', 'PersonTracker']
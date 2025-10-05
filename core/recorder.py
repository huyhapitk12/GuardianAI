"""Video recording functionality"""
import cv2
import os
import time
import uuid
import threading
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from shutil import which

from config.settings import settings
from config.constants import RECORDER_FPS, RECORDER_FOURCC

logger = logging.getLogger(__name__)

class Recorder:
    """Handles video recording with user wait capability"""
    
    def __init__(self, fps: float = RECORDER_FPS, fourcc_str: str = RECORDER_FOURCC):
        self.out_dir = settings.paths.tmp_dir
        self.fps = float(fps)
        self.fourcc = cv2.VideoWriter.fourcc(*fourcc_str)
        self._lock = threading.Lock()
        self.current: Optional[Dict[str, Any]] = None
    
    def start(
        self,
        reason: str = "alert",
        duration: int = 60,
        wait_for_user: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Start a new recording
        
        Args:
            reason: Reason for recording
            duration: Duration in seconds
            wait_for_user: If True, won't finalize until user responds
        """
        acquired = self._lock.acquire(timeout=0.3)
        if not acquired:
            logger.warning("Recorder busy, cannot start new recording")
            return None
        
        try:
            if self.current is not None:
                logger.warning("Recording already in progress")
                return None
            
            filename = f"rec_{reason}_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
            path = self.out_dir / filename
            
            self.current = {
                "path": path,
                "writer": None,
                "end_time": time.time() + float(duration),
                "meta": {"reason": reason},
                "alert_ids": [],
                "wait_for_user": wait_for_user
            }
            
            logger.info(f"Recording started: {path} (wait_for_user={wait_for_user})")
            return self.current
        finally:
            self._lock.release()
    
    def write(self, frame) -> bool:
        """Write a frame to the current recording"""
        with self._lock:
            rec = self.current
            if rec is None:
                return False
            
            # Initialize writer on first frame
            if rec["writer"] is None:
                h, w = frame.shape[:2]
                try:
                    writer = cv2.VideoWriter(
                        str(rec["path"]),
                        self.fourcc,
                        self.fps,
                        (w, h)
                    )
                    if not writer.isOpened():
                        logger.error(f"Failed to open writer: {rec['path']}")
                        self.current = None
                        return False
                    rec["writer"] = writer
                except Exception as e:
                    logger.error(f"Failed to create writer: {e}")
                    self.current = None
                    return False
            
            writer = rec["writer"]
        
        try:
            writer.write(frame)
            return True
        except Exception as e:
            logger.error(f"Error writing frame: {e}")
            return False
    
    def extend(self, extra_seconds: float) -> float:
        """Extend recording duration"""
        with self._lock:
            if self.current is None:
                raise RuntimeError("No active recording to extend")
            self.current["end_time"] += float(extra_seconds)
            logger.info(f"Recording extended by {extra_seconds}s")
            return self.current["end_time"]
    
    def resolve_user_wait(self):
        """Mark user wait as resolved, allowing finalization"""
        with self._lock:
            if self.current and self.current.get("wait_for_user", False):
                self.current["wait_for_user"] = False
                logger.info("User wait resolved, recording can finalize")
    
    def check_and_finalize(self) -> Optional[Dict[str, Any]]:
        """Check if recording should finalize and do so if ready"""
        with self._lock:
            rec = self.current
            if rec is None:
                return None
            
            # Don't finalize if time not reached
            if time.time() < rec["end_time"]:
                return None
            
            # Don't finalize if waiting for user
            if rec.get("wait_for_user", False):
                return None
            
            # Finalize
            writer = rec.get("writer")
            path = rec.get("path")
            alert_ids = list(rec.get("alert_ids", []))
            meta = rec.get("meta", {})
            
            self.current = None
        
        # Release writer outside lock
        try:
            if writer:
                writer.release()
        except Exception as e:
            logger.error(f"Error releasing writer: {e}")
        
        logger.info(f"Recording finalized: {path}")
        return {"path": path, "alert_ids": alert_ids, "meta": meta}
    
    def stop_and_discard(self) -> bool:
        """Stop current recording and delete the file"""
        with self._lock:
            rec = self.current
            if not rec:
                return False
            
            writer = rec.get("writer")
            path = rec.get("path")
            self.current = None
        
        # Release and delete outside lock
        try:
            if writer:
                writer.release()
        except Exception:
            pass
        
        try:
            if path and path.exists():
                path.unlink()
                logger.info(f"Recording discarded: {path}")
        except Exception as e:
            logger.error(f"Failed to delete recording: {e}")
        
        return True

def compress_video(input_path: Path) -> Optional[Path]:
    """Compress video using ffmpeg if available"""
    ffmpeg = which("ffmpeg")
    if not ffmpeg:
        return None
    
    output_path = input_path.parent / f"{input_path.stem}_compressed.mp4"
    cmd = [
        ffmpeg, "-y", "-i", str(input_path),
        "-vcodec", "libx264", "-crf", "28",
        "-preset", "veryfast",
        "-acodec", "aac", "-b:a", "96k",
        str(output_path)
    ]
    
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300
        )
        if output_path.exists():
            return output_path
    except Exception as e:
        logger.error(f"Video compression failed: {e}")
        if output_path.exists():
            output_path.unlink()
    
    return None
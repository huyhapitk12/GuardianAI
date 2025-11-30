# core/recorder.py
import cv2
import os
import time
import uuid
import threading
import subprocess
import datetime as dt
from pathlib import Path
from typing import Optional, Dict, Any
from shutil import which
from config import settings
from utils import security_manager

def print_msg(message):
    """Simple print function to replace logging"""
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - recorder - INFO - {message}")

def print_warning(message):
    """Simple print function for warnings"""
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - recorder - WARNING - {message}")

def print_error(message):
    """Simple print function for errors"""
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - recorder - ERROR - {message}")

class Recorder:
    """Xử lý quay video với khả năng chờ người dùng"""
    
    def __init__(self):
        self.out_dir = settings.paths.tmp_dir
        self.fps = float(settings.recorder.fps)
        self.fourcc = cv2.VideoWriter.fourcc(*settings.recorder.fourcc)
        self._lock = threading.Lock()
        self.current: Optional[Dict[str, Any]] = None
    
    def start(
        self,
        source_id: str,
        reason: str = "alert",
        duration: int = 60,
        wait_for_user: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Bắt đầu một bản ghi mới
        
        Đối số:
            source_id: ID của camera nguồn
            reason: Lý do ghi
            duration: Thời lượng tính bằng giây
            wait_for_user: Nếu là True, sẽ không hoàn tất cho đến khi người dùng phản hồi
        """
        acquired = self._lock.acquire(timeout=0.3)
        if not acquired:
            print_warning("Recorder busy, cannot start new recording")
            return None
        
        try:
            if self.current is not None:
                print_warning("Recording already in progress")
                return None
            
            filename = f"rec_{reason}_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
            path = self.out_dir / filename
            
            self.current = {
                "path": path,
                "writer": None,
                "end_time": time.time() + float(duration),
                "meta": {"reason": reason, "source_id": source_id},
                "alert_ids": [],
                "wait_for_user": wait_for_user,
                "source_id": source_id
            }
            
            print_msg(f"Recording started: {path} (wait_for_user={wait_for_user})")
            return self.current
        finally:
            self._lock.release()
    
    def write(self, frame) -> bool:
        """Ghi một khung hình vào bản ghi hiện tại - được tối ưu hóa"""
        # Ghi khung hình với khóa để đảm bảo an toàn thread
        with self._lock:
            rec = self.current
            if rec is None:
                return False
            
            # Khởi tạo người ghi trên khung hình đầu tiên
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
                        print_error(f"Failed to open writer: {rec['path']}")
                        self.current = None
                        return False
                    rec["writer"] = writer
                except Exception as e:
                    print_error(f"Failed to create writer: {e}")
                    self.current = None
                    return False
            
            writer = rec["writer"]
            
            try:
                writer.write(frame)
                return True
            except Exception as e:
                print_error(f"Error writing frame: {e}")
                return False
    
    def extend(self, extra_seconds: float) -> float:
        """Kéo dài thời gian ghi"""
        with self._lock:
            if self.current is None:
                raise RuntimeError("No active recording to extend")
            self.current["end_time"] += float(extra_seconds)
            print_msg(f"Recording extended by {extra_seconds}s")
            return self.current["end_time"]
    
    def resolve_user_wait(self):
        """Đánh dấu chờ người dùng đã được giải quyết, cho phép hoàn tất"""
        with self._lock:
            if self.current and self.current.get("wait_for_user", False):
                self.current["wait_for_user"] = False
                print_msg("User wait resolved, recording can finalize")
    
    def stop(self):
        """Dừng ghi ngay lập tức và lưu video"""
        with self._lock:
            if self.current:
                # Set end time to now to force finalization on next check
                self.current["end_time"] = time.time()
                self.current["wait_for_user"] = False
                print_msg("Recording stop requested")

    def check_and_finalize(self) -> Optional[Dict[str, Any]]:
        """Kiểm tra xem bản ghi có nên hoàn tất không và làm như vậy nếu đã sẵn sàng"""
        with self._lock:
            rec = self.current
            if rec is None:
                return None
            
            # Không hoàn tất nếu chưa đến thời gian
            if time.time() < rec["end_time"]:
                return None
            
            # Không hoàn tất nếu đang chờ người dùng
            if rec.get("wait_for_user", False):
                return None
            
            # Hoàn tất
            writer = rec.get("writer")
            path = rec.get("path")
            alert_ids = list(rec.get("alert_ids", []))
            meta = rec.get("meta", {})
            
            self.current = None
        
        # Giải phóng người ghi bên ngoài khóa
        try:
            if writer:
                writer.release()
        except Exception as e:
            print_error(f"Error releasing writer: {e}")
        
        # Encrypt the video file
        try:
            if path and path.exists():
                security_manager.encrypt_file(path)
                print_msg(f"Recording encrypted: {path}")
        except Exception as e:
            print_error(f"Failed to encrypt recording: {e}")

        print_msg(f"Recording finalized: {path}")
        return {"path": path, "alert_ids": alert_ids, "meta": meta}
    
    def stop_and_discard(self) -> bool:
        """Dừng ghi hiện tại và xóa tệp"""
        with self._lock:
            rec = self.current
            if not rec:
                return False
            
            writer = rec.get("writer")
            path = rec.get("path")
            self.current = None
        
        # Giải phóng và xóa bên ngoài khóa
        try:
            if writer:
                writer.release()
        except Exception:
            pass
        
        try:
            if path and path.exists():
                path.unlink()
                print_msg(f"Recording discarded: {path}")
        except Exception as e:
            print_error(f"Failed to delete recording: {e}")
        
        return True

def compress_video(input_path: Path) -> Optional[Path]:
    """Nén video bằng ffmpeg nếu có (Handles encrypted files)"""
    ffmpeg = which("ffmpeg")
    if not ffmpeg:
        return None
    
    # Decrypt temporarily for ffmpeg
    temp_decrypted = input_path.parent / f"temp_dec_{input_path.name}"
    try:
        decrypted_data = security_manager.try_decrypt_file(input_path)
        if decrypted_data:
            with open(temp_decrypted, 'wb') as f:
                f.write(decrypted_data)
            input_source = temp_decrypted
        else:
            # Assume it's not encrypted yet or decryption failed (try using original)
            input_source = input_path
            
        output_path = input_path.parent / f"{input_path.stem}_compressed.mp4"
        cmd = [
            ffmpeg, "-y", "-i", str(input_source),
            "-vcodec", "libx264", "-crf", "28",
            "-preset", "veryfast",
            "-acodec", "aac", "-b:a", "96k",
            str(output_path)
        ]
        
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300
        )
        
        if output_path.exists():
            # Encrypt the compressed output
            security_manager.encrypt_file(output_path)
            return output_path
            
    except Exception as e:
        print_error(f"Video compression failed: {e}")
        if output_path.exists():
            output_path.unlink()
    finally:
        # Cleanup temp decrypted file
        if temp_decrypted.exists():
            temp_decrypted.unlink()
    
    return None

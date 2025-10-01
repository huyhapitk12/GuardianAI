# video_recorder.py
import os
import subprocess
import threading
import time
import uuid
from shutil import which

import cv2
import httpx

from config import (FFMPEG_TIMEOUT, HTTPX_TIMEOUT, RECORDER_FPS,
                    RECORDER_FOURCC, TELEGRAM_CHAT_ID, TELEGRAM_TOKEN,
                    VIDEO_PREVIEW_LIMIT_MB)

def file_size_mb(p):
    try: return os.path.getsize(p) / (1024 * 1024)
    except: return 0.0

def try_compress_ffmpeg(inp):
    """Nén video bằng ffmpeg nếu có thể."""
    ff = which("ffmpeg")
    if not ff: return None
    out = inp.replace(".mp4", f"_cmp_{uuid.uuid4().hex}.mp4")
    cmd = [ff, "-y", "-i", inp, "-vcodec", "libx264", "-crf", "28", "-preset", "veryfast", "-acodec", "aac", "-b:a", "96k", out]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=FFMPEG_TIMEOUT)
        return out if os.path.exists(out) else None
    except Exception:
        if os.path.exists(out): os.remove(out)
        return None

def send_photo(token, chat_id, image_path, caption=""):
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        with open(image_path, "rb") as f:
            r = httpx.post(url, data={"chat_id": chat_id, "caption": caption}, files={"photo": f}, timeout=HTTPX_TIMEOUT)
            return r.is_success
    except Exception as e:
        print(f"send_photo error: {e}")
        return False

def send_video_or_document(token, chat_id, video_path, caption=""):
    """Gửi video, tự động nén và gửi dưới dạng document nếu file quá lớn."""
    url_base = f"https://api.telegram.org/bot{token}"
    if file_size_mb(video_path) <= VIDEO_PREVIEW_LIMIT_MB:
        try:
            with open(video_path, "rb") as f:
                r = httpx.post(f"{url_base}/sendVideo", data={"chat_id": chat_id, "caption": caption}, files={"video": f}, timeout=HTTPX_TIMEOUT)
                return r.is_success
        except Exception as e:
            print(f"sendVideo error: {e}")

    cmp = try_compress_ffmpeg(video_path)
    try_path = cmp if cmp else video_path
    try:
        with open(try_path, "rb") as f:
            r = httpx.post(f"{url_base}/sendDocument", data={"chat_id": chat_id, "caption": caption}, files={"document": (os.path.basename(try_path), f)}, timeout=HTTPX_TIMEOUT)
            return r.is_success
    except Exception as e:
        print(f"sendDocument error: {e}")
        return False
    finally:
        if cmp and os.path.exists(cmp):
            os.remove(cmp)

class Recorder:
    """Lớp quản lý việc ghi video từ các khung hình (frame)."""
    def __init__(self, out_dir="tmp", fps=RECORDER_FPS, fourcc_str=RECORDER_FOURCC):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.fps = float(fps)
        self.fourcc = cv2.VideoWriter.fourcc(*fourcc_str)
        self._lock = threading.Lock()
        self.current = None

    def start(self, reason="alert", duration=60, wait_for_user=False):
        """Bắt đầu một phiên ghi hình mới."""
        if not self._lock.acquire(timeout=0.3):
            print("Recorder.start: lock busy -> returning None")
            return None
        try:
            if self.current is not None:
                print("Recorder.start: current exists -> busy")
                return None
            fname = f"rec_{reason}_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
            path = os.path.join(self.out_dir, fname)
            rec = {
                "path": path, "writer": None, "end_time": time.time() + float(duration),
                "meta": {"reason": reason}, "alert_ids": [], "wait_for_user": wait_for_user,
            }
            self.current = rec
            print(f"Recorder.start: created record meta (wait_for_user={wait_for_user}): {path}")
            return rec
        finally:
            self._lock.release()

    def extend(self, extra_seconds):
        """Kéo dài thời gian ghi hình của phiên hiện tại."""
        with self._lock:
            if self.current is None:
                raise RuntimeError("No active recorder to extend")
            self.current["end_time"] += float(extra_seconds)
            print(f"Recorder.extend: extended by {extra_seconds}s")

    def write(self, frame):
        """Ghi một khung hình vào video."""
        with self._lock:
            rec = self.current
            if rec is None: return False
            if rec["writer"] is None:
                h, w = frame.shape[:2]
                try:
                    writer = cv2.VideoWriter(rec["path"], self.fourcc, self.fps, (w, h))
                    if not writer.isOpened():
                        self.current = None
                        return False
                    rec["writer"] = writer
                except Exception as e:
                    print(f"Recorder.write: failed to open writer: {e}")
                    self.current = None
                    return False
        try:
            rec["writer"].write(frame)
        except Exception as e:
            print(f"Recorder.write: exception writing frame: {e}")
            return False
        return True

    def check_and_finalize(self):
        """Kiểm tra nếu hết thời gian ghi hình và hoàn tất file video."""
        with self._lock:
            rec = self.current
            if rec is None: return None
            # Nếu hết giờ nhưng đang chờ người dùng, không hoàn tất
            if time.time() < rec["end_time"] or rec.get("wait_for_user", False):
                return None
            self.current = None

        try:
            if rec.get("writer"): rec["writer"].release()
        except Exception as e:
            print(f"Recorder.check_and_finalize: error releasing writer: {e}")
        print(f"Recorder.check_and_finalize: finalized {rec.get('path')}")
        return rec

    def resolve_user_wait(self):
        """Báo hiệu rằng thời gian chờ người dùng đã kết thúc để recorder có thể hoàn tất."""
        with self._lock:
            if self.current and self.current.get("wait_for_user", False):
                print("Recorder.resolve_user_wait: Đã mở khóa chờ người dùng.")
                self.current["wait_for_user"] = False

    def stop_and_discard(self):
        """Dừng ghi hình và xóa file video đang ghi."""
        with self._lock:
            rec = self.current
            if not rec: return False
            self.current = None
        try:
            if rec.get("writer"): rec["writer"].release()
            if rec.get("path") and os.path.exists(rec["path"]):
                os.remove(rec["path"])
                print(f"Recorder.stop_and_discard: removed {rec['path']}")
        except Exception as e:
            print(f"Recorder.stop_and_discard: failed to remove file: {e}")
        return True
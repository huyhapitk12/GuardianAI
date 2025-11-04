# video_recorder.py
import cv2, time, uuid, os, subprocess
from config import TMP_DIR, RECORD_SECONDS, VIDEO_PREVIEW_LIMIT_MB, HTTPX_TIMEOUT, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, FFMPEG_TIMEOUT, RECORDER_FPS, RECORDER_FOURCC
import httpx
from shutil import which
import threading

def file_size_mb(p): 
    try: return os.path.getsize(p)/(1024*1024)
    except: return 0.0

def try_compress_ffmpeg(inp):
    ff = which("ffmpeg")
    if not ff: return None
    out = inp.replace(".mp4", f"_cmp_{uuid.uuid4().hex}.mp4")
    cmd = [ff, "-y", "-i", inp, "-vcodec", "libx264", "-crf", "28", "-preset", "veryfast", "-acodec", "aac", "-b:a", "96k", out]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=FFMPEG_TIMEOUT)
        return out if os.path.exists(out) else None
    except Exception as e:
        if os.path.exists(out): os.remove(out)
        return None

def send_photo(token, chat_id, image_path, caption=""):
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        with open(image_path, "rb") as f:
            r = httpx.post(url, data={"chat_id":chat_id, "caption":caption}, files={"photo":f}, timeout=HTTPX_TIMEOUT)
            return r.is_success
    except Exception as e:
        print("send_photo error", e)
        return False

def send_video_or_document(token, chat_id, video_path, caption=""):
    url_base = f"https://api.telegram.org/bot{token}"
    size = file_size_mb(video_path)
    if size <= VIDEO_PREVIEW_LIMIT_MB:
        try:
            with open(video_path, "rb") as f:
                r = httpx.post(f"{url_base}/sendVideo", data={"chat_id":chat_id, "caption":caption}, files={"video":f}, timeout=HTTPX_TIMEOUT)
                return r.is_success
        except Exception as e:
            print("sendVideo error", e)
    cmp = try_compress_ffmpeg(video_path)
    try_path = cmp if cmp else video_path
    try:
        with open(try_path, "rb") as f:
            r = httpx.post(f"{url_base}/sendDocument", data={"chat_id":chat_id, "caption":caption}, files={"document":(os.path.basename(try_path), f)}, timeout=HTTPX_TIMEOUT)
            ok = r.is_success
    except Exception as e:
        print("sendDocument error", e)
        ok = False
    finally:
        if cmp and os.path.exists(cmp):
            os.remove(cmp)
    return ok

class Recorder:
    def __init__(self, out_dir="tmp", fps=RECORDER_FPS, fourcc_str=RECORDER_FOURCC):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.fps = float(fps)
        self.fourcc = cv2.VideoWriter.fourcc(*fourcc_str)
        self._lock = threading.Lock()
        self.current = None

    # <--- THAY ĐỔI: Thêm tham số wait_for_user --->
    def start(self, reason="alert", duration=60, wait_for_user=False):
        got = self._lock.acquire(timeout=0.3)
        if not got:
            print("Recorder.start: lock busy -> returning None")
            return None
        try:
            if self.current is not None:
                print("Recorder.start: current exists -> busy")
                return None
            fname = f"rec_{reason}_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
            path = os.path.join(self.out_dir, fname)
            rec = {
                "path": path,
                "writer": None,
                "end_time": time.time() + float(duration),
                "meta": {"reason": reason},
                "alert_ids": [],
                "wait_for_user": wait_for_user, # <--- THAY ĐỔI: Lưu trạng thái chờ
            }
            self.current = rec
            print(f"Recorder.start: created record meta (wait_for_user={wait_for_user}):", path)
            return rec
        finally:
            try: self._lock.release()
            except RuntimeError: pass

    def extend(self, extra_seconds):
        with self._lock:
            if self.current is None:
                raise RuntimeError("No active recorder to extend")
            self.current["end_time"] += float(extra_seconds)
            print(f"Recorder.extend: extended by {extra_seconds}s, new end_time:", self.current["end_time"])
            return self.current["end_time"]

    def write(self, frame):
        with self._lock:
            rec = self.current
            if rec is None: return False
            if rec["writer"] is None:
                h, w = frame.shape[:2]
                try:
                    writer = cv2.VideoWriter(rec["path"], self.fourcc, self.fps, (w, h))
                    if not writer.isOpened():
                        print("Recorder.write: VideoWriter failed to open", rec["path"])
                        self.current = None
                        return False
                    rec["writer"] = writer
                    print("Recorder.write: opened writer:", rec["path"], "frame size:", (w,h))
                except Exception as e:
                    print("Recorder.write: failed to open writer:", e)
                    self.current = None
                    return False
            writer = rec["writer"]
        try:
            writer.write(frame)
        except Exception as e:
            print("Recorder.write: exception writing frame:", e)
            return False
        return True

    def check_and_finalize(self):
        with self._lock:
            rec = self.current
            if rec is None: return None
            if time.time() < rec["end_time"]: return None

            # <--- THAY ĐỔI: Logic "Grace Period" --->
            # Nếu hết giờ nhưng vẫn đang trong trạng thái chờ người dùng, KHÔNG hoàn tất.
            if rec.get("wait_for_user", False):
                return None

            writer = rec.get("writer")
            path = rec.get("path")
            alert_ids = list(rec.get("alert_ids", []))
            meta = rec.get("meta", {})
            self.current = None
        try:
            if writer: writer.release()
        except Exception as e:
            print("Recorder.check_and_finalize: error releasing writer:", e)
        print("Recorder.check_and_finalize: finalized", path, "alerts:", alert_ids)
        return {"path": path, "alert_ids": alert_ids, "meta": meta}

    # <--- THAY ĐỔI: Hàm mới để "mở khóa" recorder --->
    def resolve_user_wait(self):
        """Báo hiệu rằng thời gian chờ người dùng đã kết thúc (do có phản hồi hoặc hết giờ)."""
        with self._lock:
            if self.current:
                if self.current.get("wait_for_user", False):
                    print("Recorder.resolve_user_wait: Đã mở khóa chờ người dùng.")
                    self.current["wait_for_user"] = False
                
    def stop_and_discard(self):
        with self._lock:
            rec = self.current
            if not rec: return False
            writer = rec.get("writer")
            path = rec.get("path")
            self.current = None
        try:
            if writer: writer.release()
        except Exception: pass
        try:
            if path and os.path.exists(path):
                os.remove(path)
                print("Recorder.stop_and_discard: removed", path)
        except Exception as e:
            print("Recorder.stop_and_discard: failed to remove file:", e)
        return True
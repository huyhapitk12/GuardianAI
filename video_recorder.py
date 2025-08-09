# video_recorder.py
import cv2, time, uuid, os, subprocess
from config import TMP_DIR, RECORD_SECONDS, VIDEO_PREVIEW_LIMIT_MB, HTTPX_TIMEOUT, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
import httpx
from shutil import which

def file_size_mb(p): 
    try: return os.path.getsize(p)/(1024*1024)
    except: return 0.0

def try_compress_ffmpeg(inp):
    ff = which("ffmpeg")
    if not ff: return None
    out = inp.replace(".mp4", f"_cmp_{uuid.uuid4().hex}.mp4")
    cmd = [ff, "-y", "-i", inp, "-vcodec", "libx264", "-crf", "28", "-preset", "veryfast", "-acodec", "aac", "-b:a", "96k", out]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
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
    # small enough -> sendVideo
    if size <= VIDEO_PREVIEW_LIMIT_MB:
        try:
            with open(video_path, "rb") as f:
                r = httpx.post(f"{url_base}/sendVideo", data={"chat_id":chat_id, "caption":caption}, files={"video":f}, timeout=HTTPX_TIMEOUT)
                return r.is_success
        except Exception as e:
            print("sendVideo error", e)
    # try compress
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
    def __init__(self, fps=20.0):
        self.current = None  # {path, writer, start_ts, duration}
        self.lock = False
        self.fps = fps

    def start(self, reason="alert", duration=RECORD_SECONDS):
        if self.current:
            print("Recorder busy")
            return None
        filename = f"{reason}_{uuid.uuid4().hex}.mp4"
        path = os.path.join(TMP_DIR, filename)
        self.current = {"path":path, "writer":None, "start":None, "duration":duration}
        return self.current

    def write(self, frame):
        if not self.current: return
        if self.current["writer"] is None:
            h,w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.current["writer"] = cv2.VideoWriter(self.current["path"], fourcc, self.fps, (w,h))
            self.current["start"] = time.time()
        self.current["writer"].write(frame)

    def check_and_finalize(self):
        if not self.current: return None
        if self.current["start"] and (time.time() - self.current["start"] >= self.current["duration"]):
            # finalize
            try:
                if self.current["writer"]:
                    self.current["writer"].release()
            except:
                pass
            p = self.current["path"]
            self.current = None
            return p
        return None

    def stop_and_discard(self):
        if not self.current: return
        try:
            if self.current["writer"]:
                self.current["writer"].release()
        except:
            pass
        try:
            if os.path.exists(self.current["path"]):
                os.remove(self.current["path"])
        except:
            pass
        self.current = None

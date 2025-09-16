# main.py
import threading
import time
import os
import uuid
import queue
import logging
import cv2
import customtkinter as ctk

from detection_core import Camera
import detection_core
from telegram_bot import run_bot, response_queue, state
from video_recorder import Recorder, send_photo, send_video_or_document
from gui_manager import FaceManagerApp
from config import TELEGRAM_CHAT_ID, TELEGRAM_TOKEN, TMP_DIR, RECORD_SECONDS, IP_CAMERA_URL, DEBOUNCE_SECONDS
from spam_guard import SpamGuard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("guardian")

recorder = Recorder()
sm = state
response_q = response_queue
guard = SpamGuard(debounce_seconds=DEBOUNCE_SECONDS, min_interval=10, max_per_minute=4)

USER_RESPONSE_WINDOW = 60

def _on_alert(frame, reason, name, meta):
    if reason == "nguoi_quen":
        alert_key = (reason, name)
    else:
        alert_key = reason

    # <--- THAY ĐỔI MỚI: Lớp kiểm tra đầu tiên và quan trọng nhất --->
    # Nếu đã có một cảnh báo cho người/sự kiện này đang chờ phản hồi, hãy bỏ qua hoàn toàn.
    if sm.has_unresolved_alert(alert_key):
        log.info("Bỏ qua cảnh báo %s vì đã có một cảnh báo khác đang chờ phản hồi.", alert_key)
        return

    # Lớp kiểm tra thứ hai: SpamGuard để chống các phát hiện dồn dập
    if not guard.allow(alert_key):
        log.info("Bỏ qua cảnh báo %s để tránh spam (debounce)", alert_key)
        return

    log.info(">>> CẢNH BÁO MỚI ĐƯỢC PHÉP: %s", alert_key)

    chat_id = TELEGRAM_CHAT_ID
    img_path = os.path.join(TMP_DIR, f"alert_{reason}_{uuid.uuid4().hex}.jpg")

    try:
        cv2.imwrite(img_path, frame)
    except Exception as e:
        log.exception("Failed to write img: %s", e)
        return

    alert_id = sm.create_alert(reason, chat_id, asked_for=name)

    if reason == "nguoi_la":
        caption = "⚠️ Phát hiện người lạ"
    elif reason == "nguoi_quen":
        caption = f"👋 Phát hiện {name}"
    else:
        caption = "🔥 CẢNH BÁO CHÁY"

    if reason != "lua_chay":
        caption += f"\n\nBạn có nhận ra người này không? (Trả lời trong {USER_RESPONSE_WINDOW}s: có/không/đã ra khỏi nhà)"
    else:
        caption += "\n\nVui lòng kiểm tra ngay lập tức!"

    if reason == "nguoi_la":
        cam_obj = globals().get("cam", None)
        try:
            start_clip_for_alert(cam_obj, frame, alert_id, duration=8, fps=recorder.fps)
            log.info("Started immediate clip worker for stranger alert %s", alert_id)
        except Exception as e:
            log.exception("Failed to start immediate clip for alert %s: %s", alert_id, e)

    threading.Thread(target=lambda: send_photo(TELEGRAM_TOKEN, chat_id, img_path, caption), daemon=True).start()

    def _try_start_recorder(reason, duration, timeout=3.0, **kwargs):
        q = queue.Queue()
        def target():
            try:
                r = recorder.start(reason=reason, duration=duration, **kwargs)
                q.put(("ok", r))
            except Exception as e:
                q.put(("exc", e))
        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            log.error("recorder.start() hung (still alive after %.1fs)", timeout)
            return None
        try:
            status, val = q.get_nowait()
        except queue.Empty:
            log.error("recorder.start() returned nothing")
            return None
        if status == "ok":
            return val
        else:
            log.exception("recorder.start() raised exception: %s", val)
            return None

    log.debug("Attempting to start recorder for alert %s", alert_id)
    
    wait_for_user_reply = (reason == "nguoi_quen")
    rec = _try_start_recorder(reason, RECORD_SECONDS, timeout=3.0, wait_for_user=wait_for_user_reply)
    
    if rec is None:
        log.warning("Recorder returned None (busy/timeout/failed). Attaching/extending if possible.")
        current = getattr(recorder, "current", None)
        if current:
            current.setdefault("alert_ids", []).append(alert_id)
            log.debug("Attached alert %s to current recorder", alert_id)
            try:
                if hasattr(recorder, "extend"):
                    recorder.extend(RECORD_SECONDS)
                    log.debug("Extended recorder by %s seconds", RECORD_SECONDS)
            except Exception:
                log.exception("Failed to extend recorder")
        else:
            log.warning("No active recorder to attach to.")
    else:
        rec.setdefault("alert_ids", []).append(alert_id)
        rec["alert_id"] = alert_id
        log.info("Started new recorder for alert %s -> %s", alert_id, rec.get("path", "<no-path>"))

    def watcher(aid):
        start = time.time()
        reply_received = False
        while time.time() - start < USER_RESPONSE_WINDOW:
            try:
                resp = response_q.get(timeout=1.0)
            except queue.Empty:
                continue
            
            if resp and resp.get("alert_id") == aid:
                reply_received = True
                decision = resp.get("decision")
                raw = resp.get("raw_text")
                
                recorder.resolve_user_wait()
                
                sm.resolve_alert(aid, raw)
                if decision in ("yes", "left"):
                    log.info("Owner replied safe -> stop and delete rec")
                    recorder.stop_and_discard()
                else:
                    log.info("Negative/unclear reply -> keep recording")
                
                return

        if not reply_received:
            log.info("No reply in %ds for alert %s. Resolving user wait.", USER_RESPONSE_WINDOW, aid)
            recorder.resolve_user_wait()

    threading.Thread(target=watcher, args=(alert_id,), daemon=True).start()

# (Phần còn lại của file main.py giữ nguyên, không cần thay đổi)
# ...
# (Các hàm start_clip_for_alert, recorder_monitor_loop, run_gui, và __main__ giữ nguyên)
# ...
detection_core.on_alert_callback = _on_alert


def start_clip_for_alert(cam, initial_frame, alert_id, duration=8, fps=20.0, reason="clip"):
    def worker():
        os.makedirs(TMP_DIR, exist_ok=True)
        fname = f"clip_{reason}_{alert_id[:8]}_{uuid.uuid4().hex[:8]}.mp4"
        path = os.path.join(TMP_DIR, fname)
        try:
            h, w = initial_frame.shape[:2]
        except Exception:
            h, w = 480, 640
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        try:
            writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
            if not writer.isOpened():
                log.error("Clip worker: VideoWriter failed to open %s", path)
                return
        except Exception as e:
            log.exception("Clip worker: failed to create VideoWriter: %s", e)
            return
        t0 = time.time()
        try:
            writer.write(initial_frame)
        except Exception as e:
            log.exception("Clip worker: write initial_frame failed: %s", e)
        while time.time() - t0 < float(duration):
            try:
                if hasattr(cam, "read"):
                    ret, frame = cam.read()
                else:
                    frame = getattr(cam, "_last_frame", None)
                    ret = frame is not None
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue
                try:
                    fh, fw = frame.shape[:2]
                    if (fw, fh) != (w, h):
                        frame = cv2.resize(frame, (w, h))
                except Exception:
                    pass
                writer.write(frame)
            except Exception as e:
                log.exception("Clip worker: exception while reading/writing frame: %s", e)
                time.sleep(0.02)
                continue
        try:
            writer.release()
        except Exception:
            pass
        try:
            threading.Thread(target=lambda p=path: send_video_or_document(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, p, caption=f"📹 Clip cảnh báo ({reason})"), daemon=True).start()
        except Exception as e:
            log.exception("Clip worker: failed to spawn send thread: %s", e)
    threading.Thread(target=worker, daemon=True).start()

def recorder_monitor_loop(cam):
    log.info("recorder_monitor_loop started, cam type: %s", type(cam))
    while True:
        if getattr(cam, "quit", False):
            log.info("recorder_monitor_loop quitting due to cam.quit")
            break
        ret, frame = False, None
        try:
            if hasattr(cam, "read"):
                ret, frame = cam.read()
            if (not ret) and hasattr(cam, "_last_frame"):
                f = getattr(cam, "_last_frame", None)
                if f is not None:
                    ret, frame = True, f
        except Exception as e:
            log.exception("Exception while reading frame: %s", e)
            ret, frame = False, None
        if not ret:
            time.sleep(0.05)
            continue
        try:
            if recorder.current:
                recorder.write(frame)
                finalized = recorder.check_and_finalize()
                if finalized:
                    path = finalized if isinstance(finalized, str) else finalized.get("path")
                    log.info("Recorder finalized: %s", path)
                    threading.Thread(target=lambda p=path: send_video_or_document(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, p, caption='📹 Bản ghi cảnh báo'), daemon=True).start()
        except Exception as e:
            log.exception("Error during recorder write/finalize: %s", e)
        time.sleep(0.02)

def run_gui():
    root = ctk.CTk()
    app = FaceManagerApp(root)
    root.mainloop()

if __name__ == "__main__":
    tbot = threading.Thread(target=run_bot, daemon=True)
    tbot.start()
    log.info("Telegram bot thread started.")
    try:
        gui_thread = threading.Thread(target=run_gui, daemon=True)
        gui_thread.start()
        log.info("GUI thread started.")
    except Exception as e:
        log.exception("Failed to start GUI thread: %s", e)
    try:
        cam = Camera(IP_CAMERA_URL)
        globals()["cam"] = cam
    except Exception as e:
        log.exception("Failed to create Camera: %s", e)
        raise
    threading.Thread(target=recorder_monitor_loop, args=(cam,), daemon=True).start()
    log.info("Recorder monitor thread started.")
    try:
        cam.detect()
    except KeyboardInterrupt:
        log.info("Interrupted by user, quitting...")
    except Exception as e:
        log.exception("Unhandled exception in cam.detect: %s", e)
    finally:
        try:
            cam.delete()
        except Exception:
            pass
        log.info("Main exiting.")
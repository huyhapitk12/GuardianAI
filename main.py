# main.py
import threading
import time
import os
import uuid
import queue
import logging
import cv2
import customtkinter as ctk

from detection_core import Camera, on_alert_callback  # detection_core should export Camera and allow callback
from telegram_bot import run_bot, response_queue, state
from video_recorder import Recorder, send_photo, send_video_or_document
from state_manager import StateManager
from gui_manager import FaceManagerApp
from config import TELEGRAM_CHAT_ID, TELEGRAM_TOKEN, TMP_DIR, RECORD_SECONDS, FIRE_WINDOW_SECONDS, FIRE_REQUIRED_COUNT
import detection_core

# setup logging to stdout
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("guardian")

# shared objects
recorder = Recorder()
sm = state  # state manager from telegram_bot
response_q = response_queue

# fire aggregation (if your project uses it)
from collections import deque
fire_timestamps = deque()

# ---------- Alert callback used by detection_core ----------

def _on_alert(frame, reason, name, meta):
    """
    Called by detection_core when an event occurs.
    reason: 'nguoi_la' | 'nguoi_quen' | 'lua_chay' (example)
    name: detected person name if known, else None
    meta: additional metadata
    """
    chat_id = TELEGRAM_CHAT_ID
    ts = time.time()
    img_path = os.path.join(TMP_DIR, f"alert_{reason}_{uuid.uuid4().hex}.jpg")

    # save image (best-effort)
    try:
        import cv2
        cv2.imwrite(img_path, frame)
    except Exception as e:
        log.exception("Failed to write img: %s", e)
        return

    # create alert entry in state manager
    alert_id = sm.create_alert(reason, chat_id, asked_for=name)

    if reason == "nguoi_la":
        caption = "⚠️ Phát hiện người lạ"
    elif reason == "nguoi_quen":
        caption = "⚠️ Phát hiện người quen"
    else:
        caption = "🔥 Phát hiện lửa cháy"

    if name:
        caption += f" - tên: {name}\nCó phải {name} đang ở trong khu vực không? (Trả lời trong 60s: có/không/đã ra khỏi nhà)"
    else:
        caption += "\nVui lòng phản hồi trong 60s nếu có mặt trong nhà."

    if reason == "nguoi_la":
        # try to access global cam (we create cam below in __main__)
        cam_obj = globals().get("cam", None)
        try:
            # use frame argument as initial_frame
            start_clip_for_alert(cam_obj, frame, alert_id, duration=8, fps=recorder.fps if hasattr(recorder,"fps") else 20.0, reason="nguoi_la")
            log.info("Started immediate clip worker for stranger alert %s", alert_id)
        except Exception as e:
            log.exception("Failed to start immediate clip for alert %s: %s", alert_id, e)

    # send photo in background
    threading.Thread(target=lambda: send_photo(TELEGRAM_TOKEN, chat_id, img_path, caption), daemon=True).start()

    # --------- start recorder (in a safe wrapper) ----------
    def _try_start_recorder(reason, duration, timeout=3.0):
        q = queue.Queue()
        def target():
            try:
                r = recorder.start(reason=reason, duration=duration)
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
    rec = _try_start_recorder(reason, RECORD_SECONDS, timeout=3.0)
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

    # --------- watcher thread: wait for user response in response_queue ---------
    def watcher(aid):
        start = time.time()
        while time.time() - start < 60:
            try:
                resp = response_q.get(timeout=1.0)
            except queue.Empty:
                resp = None
            if resp and resp.get("alert_id") == aid:
                decision = resp.get("decision")
                raw = resp.get("raw_text")
                if decision in ("yes", "left"):
                    log.info("Owner replied safe -> stop and delete rec")
                    recorder.stop_and_discard()
                    sm.resolve_alert(aid, raw)
                    return
                elif decision == "no" or decision is None:
                    log.info("Negative reply -> keep recording")
                    sm.resolve_alert(aid, raw)
                    return
        log.info("No reply in 60s for alert %s", aid)

    threading.Thread(target=watcher, args=(alert_id,), daemon=True).start()


# bind the callback so detection_core will call us
detection_core.on_alert_callback = _on_alert

def start_clip_for_alert(cam, initial_frame, alert_id, duration=8, fps=20.0, reason="clip"):
    """
    Bắt 1 thread riêng, ghi initial_frame + các frame tiếp theo lấy từ cam.read()
    trong `duration` giây, lưu vào TMP_DIR và gửi bằng send_video_or_document.
    - cam: object camera (detection_core.Camera) phải có method read() trả (ret, frame)
    - initial_frame: frame đã nhận khi phát hiện (numpy array)
    """
    def worker():
        os.makedirs(TMP_DIR, exist_ok=True)
        fname = f"clip_{reason}_{alert_id[:8]}_{uuid.uuid4().hex[:8]}.mp4"
        path = os.path.join(TMP_DIR, fname)

        # determine frame size from initial_frame
        try:
            h, w = initial_frame.shape[:2]
        except Exception:
            # fallback size
            h, w = 480, 640

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        try:
            writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
            if not writer.isOpened():
                print("Clip worker: VideoWriter failed to open", path)
                return
        except Exception as e:
            print("Clip worker: failed to create VideoWriter:", e)
            return

        t0 = time.time()
        # write initial frame first
        try:
            writer.write(initial_frame)
        except Exception as e:
            print("Clip worker: write initial_frame failed:", e)

        # keep reading frames from cam.read() for duration
        while time.time() - t0 < float(duration):
            try:
                # prefer cam.read() API (we added this earlier)
                if hasattr(cam, "read"):
                    ret, frame = cam.read()
                else:
                    # fallback: try to use _last_frame
                    frame = getattr(cam, "_last_frame", None)
                    ret = frame is not None

                if not ret or frame is None:
                    # small sleep and continue
                    time.sleep(0.02)
                    continue
                # ensure shape matches writer or resize if necessary
                try:
                    fh, fw = frame.shape[:2]
                    if (fw, fh) != (w, h):
                        frame = cv2.resize(frame, (w, h))
                except Exception:
                    pass
                writer.write(frame)
            except Exception as e:
                # don't kill worker on single error; log and continue
                print("Clip worker: exception while reading/writing frame:", e)
                time.sleep(0.02)
                continue

        # finalize
        try:
            writer.release()
        except Exception:
            pass

        # send file async (this will do http post)
        try:
            threading.Thread(target=lambda p=path: send_video_or_document(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, p, caption=f"📹 Clip cảnh báo ({reason})"), daemon=True).start()
        except Exception as e:
            print("Clip worker: failed to spawn send thread:", e)

# ---------- recorder monitor loop (background thread) ----------
def recorder_monitor_loop(cam):
    cap = cam
    print("recorder_monitor_loop started, cam type:", type(cam))
    last_status_log = time.time()
    loop_count = 0
    while True:
        loop_count += 1
        if getattr(cap, "quit", False):
            print("recorder_monitor_loop quitting due to cam.quit")
            break
        ret, frame = False, None
        try:
            if hasattr(cap, "read"):
                ret, frame = cap.read()
            if (not ret) and hasattr(cap, "_last_frame"):
                # fallback to last_frame stored by detection_core (non-blocking)
                f = getattr(cap, "_last_frame", None)
                if f is not None:
                    ret, frame = True, f
            if (not ret) and hasattr(cap, "capture_array"):
                f = cap.capture_array()
                ret = f is not None
                frame = f
        except Exception as e:
            print("Exception while reading frame:", e)
            ret, frame = False, None

        if not ret:
            if time.time() - last_status_log > 5.0:
                print("recorder_monitor_loop: no frame (ret=False). loop_count=", loop_count, "recorder.current:", bool(getattr(recorder,'current',None)))
                last_status_log = time.time()
            time.sleep(0.05)
            continue

        # log frame info
        try:
            if frame is not None:
                h, w = frame.shape[:2]
                print(f"recorder_monitor_loop: got frame size={w}x{h}, recorder.current={'yes' if recorder.current else 'no'}")
            else:
                print("recorder_monitor_loop: frame is None, skipping shape logging")
        except Exception as e:
            print("recorder_monitor_loop: got frame but cannot read shape:", e)

        # write to recorder if active
        try:
            if recorder.current:
                print("recorder_monitor_loop: calling recorder.write(...) path=", recorder.current.get('path'))
                wrote = recorder.write(frame)
                print("recorder_monitor_loop: recorder.write returned", wrote)
                finalized = recorder.check_and_finalize()
                if finalized:
                    path = finalized if isinstance(finalized, str) else finalized.get("path")
                    print("Recorder finalized:", path)
                    threading.Thread(target=lambda p=path: send_video_or_document(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, p, caption='📹 Bản ghi cảnh báo'), daemon=True).start()
        except Exception as e:
            print("Error during recorder write/finalize:", e)
        time.sleep(0.02)

def run_gui():
    root = ctk.CTk()
    app = FaceManagerApp(root)
    root.mainloop()


if __name__ == "__main__":
    # start telegram bot
    tbot = threading.Thread(target=run_bot, daemon=True)
    tbot.start()

    # start GUI optionally
    gui_thread = threading.Thread(target=run_gui, daemon=True)
    gui_thread.start()

    # create camera and run detection (detect() is blocking)
    cam = Camera("rtsp://admin:XGZBPX@192.168.1.6:554/h264/ch1/main/av_stream")
    # start recorder monitor thread
    threading.Thread(target=recorder_monitor_loop, args=(cam,), daemon=True).start()
    # run detection (this will likely block until you quit)
    cam.detect()

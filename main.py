# main.py
import threading, time, os, uuid
from detection_core import Camera, on_alert_callback, get_known_data
from telegram_bot import run_bot, response_queue, state
from video_recorder import Recorder, send_photo, send_video_or_document
from state_manager import StateManager
from gui_manager import GUI
from config import TELEGRAM_CHAT_ID, TELEGRAM_TOKEN, TMP_DIR, RECORD_SECONDS, FIRE_WINDOW_SECONDS, FIRE_REQUIRED_COUNT
import detection_core
import queue

# shared objects
recorder = Recorder()
sm = state  # state manager from telegram_bot
response_q = response_queue

# fire aggregation
from collections import deque
fire_timestamps = deque()

# set callback used by detection_core
def _on_alert(frame, reason, name, meta):
    """
    Called from detection_core when it sees:
      reason = 'nguoi_la' | 'nguoi_quen' | 'lua_chay'
    We'll:
     - Save one image
     - Create alert in state manager
     - Send photo to telegram (async)
     - Start recorder (100s)
     - Start watcher thread (60s) to wait for reply in response_queue
    """
    chat_id = TELEGRAM_CHAT_ID
    ts = time.time()
    img_path = os.path.join(TMP_DIR, f"alert_{reason}_{uuid.uuid4().hex}.jpg")
    try:
        import cv2
        cv2.imwrite(img_path, frame)
    except Exception as e:
        print("Failed to write img", e)
        return

    # create alert entry
    alert_id = sm.create_alert(reason, chat_id, asked_for=name)
    caption = f"⚠️ Phát hiện {reason}"
    if name:
        caption += f" - tên: {name}\nCó phải {name} đang ở trong khu vực không? (Trả lời trong 60s: có/không/đã ra khỏi nhà)"
    else:
        caption += "\nVui lòng phản hồi trong 60s nếu có mặt trong nhà."

    # send photo async
    threading.Thread(target=lambda: send_photo(TELEGRAM_TOKEN, chat_id, img_path, caption), daemon=True).start()

    # start recording
    rec = recorder.start(reason=reason, duration=RECORD_SECONDS)
    # attach alert id to recorder record (we store as metadata)
    if rec is not None:
        rec["alert_id"] = alert_id

    # watcher thread to wait 60s for reply
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
                # if yes or left -> stop & delete recording
                if decision in ("yes","left"):
                    print("Owner replied safe -> stop and delete rec")
                    recorder.stop_and_discard()
                    sm.resolve_alert(aid, raw)
                    return
                elif decision == "no" or decision is None:
                    print("Negative reply -> keep recording")
                    sm.resolve_alert(aid, raw)
                    return
        # timeout -> keep recording and escalate (do nothing here)
        print("No reply in 60s for alert", aid)

    threading.Thread(target=watcher, args=(alert_id,), daemon=True).start()

# bind the callback
detection_core.on_alert_callback = _on_alert

def recorder_monitor_loop(cam):
    """
    Called in background to write frames to recorder if active, and when recorder finalizes send it.
    We'll pull frames from camera by reading camera.read() ourselves? No:
    Since Camera.detect() already reads frames and shows them,
    we instead run a side thread that grabs frames from the camera object using cap.read() too.
    NOTE: On some cameras multiple simultaneous reads may fail; in that case we could modify detection_core.Camera to expose frames.
    """
    cap = cam  # detection_core.Camera is subclass of cv2.VideoCapture
    while True:
        if cap.quit:
            break
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        # write to recorder if active
        if recorder.current:
            recorder.write(frame)
            finalized = recorder.check_and_finalize()
            if finalized:
                # send video async
                threading.Thread(target=lambda p=finalized: send_video_or_document(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, p, caption="📹 Bản ghi cảnh báo"), daemon=True).start()
        time.sleep(0.02)

if __name__ == "__main__":
    # start telegram bot in background
    tbot = threading.Thread(target=run_bot, daemon=True)
    tbot.start()

    # start GUI optionally (comment out if running headless)
    gui = threading.Thread(target=lambda: GUI().run(), daemon=True)
    gui.start()

    # create camera and run detection (this will block in detect())
    cam = Camera()
    # start recorder writer thread
    threading.Thread(target=recorder_monitor_loop, args=(cam,), daemon=True).start()
    # run detect (this will show window and block until 'q')
    cam.detect()

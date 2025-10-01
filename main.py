# main.py
import logging
import os
import queue
import threading
import time
import uuid
from functools import partial

import cv2
import customtkinter as ctk
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from alarm_player import init_alarm, play_alarm
from config import (IP_CAMERAS, RECORD_SECONDS, STRANGER_CLIP_DURATION,
                    TELEGRAM_CHAT_ID, TELEGRAM_TOKEN, TMP_DIR,
                    USER_RESPONSE_WINDOW_SECONDS)
from detection_core import Camera
from gui_manager import FaceManagerApp
from shared_state import guard, recorder, response_queue, state
import shared_state
from telegram_bot import (add_system_message_to_history, run_bot,
                          schedule_send_alert)
from video_recorder import send_photo, send_video_or_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
log = logging.getLogger("guardian")

def fire_alert_watcher(alert_id):
    """Theo dõi cảnh báo cháy, nếu không có phản hồi sau một thời gian sẽ tự bật còi."""
    log.info(f"Bắt đầu theo dõi cảnh báo cháy ID: {alert_id} trong {USER_RESPONSE_WINDOW_SECONDS} giây.")
    time.sleep(USER_RESPONSE_WINDOW_SECONDS)
    alert_info = state.get_alert_by_id(alert_id)
    if alert_info and not alert_info.get('resolved', False):
        log.warning(f"Không có phản hồi cho cảnh báo cháy {alert_id}. KÍCH HOẠT CÒI BÁO ĐỘNG!")
        play_alarm()

def _on_alert(frame, reason, name, meta, camera_name="Unknown"):
    """Callback được gọi khi có một sự kiện cảnh báo từ detection_core."""
    alert_key = (reason, name) if reason == "nguoi_quen" else ("lua_chay" if "lua_chay" in reason else reason)

    if state.has_unresolved_alert(alert_key):
        log.info(f"Bỏ qua cảnh báo '{alert_key}' vì đã có cảnh báo khác đang chờ.")
        return
    if not guard.allow(alert_key):
        log.info(f"Bỏ qua cảnh báo '{alert_key}' để tránh spam.")
        return

    log.info(f">>> CẢNH BÁO MỚI: {alert_key} từ camera {camera_name}")
    img_path = os.path.join(TMP_DIR, f"alert_{reason}_{uuid.uuid4().hex}.jpg")
    cv2.imwrite(img_path, frame)
    alert_id = state.create_alert(reason, TELEGRAM_CHAT_ID, asked_for=name, image_path=img_path)

    # Tạo caption và nút bấm
    is_fire_alert = "lua_chay" in reason
    caption = ""
    if reason == "nguoi_la":
        caption = f"⚠️ [{camera_name}] Phát hiện người lạ\n\nBạn có nhận ra người này không? (có/không)"
    elif reason == "nguoi_quen":
        caption = f"👋 [{camera_name}] Phát hiện {name}\n\nBạn có nhận ra người này không? (có/không)"
    elif reason == "lua_chay_nghi_ngo":
        caption = f"🟡 [{camera_name}] CẢNH BÁO VÀNG: Nghi ngờ có cháy. Vui lòng xác nhận."
    elif reason == "lua_chay_khan_cap":
        caption = f"🔴 [{camera_name}] CẢNH BÁO ĐỎ: Phát hiện đám cháy. Yêu cầu kiểm tra ngay!"

    if is_fire_alert:
        keyboard = [[InlineKeyboardButton("✅ Cháy thật", callback_data=f"fire_real:{alert_id}"),
                     InlineKeyboardButton("❌ Báo động giả", callback_data=f"fire_false:{alert_id}")],
                    [InlineKeyboardButton("📞 Gọi PCCC (114)", callback_data=f"fire_call:{alert_id}")]]
        schedule_send_alert(TELEGRAM_CHAT_ID, img_path, caption, InlineKeyboardMarkup(keyboard))
        if reason == "lua_chay_khan_cap":
            threading.Thread(target=fire_alert_watcher, args=(alert_id,), daemon=True).start()
    else:
        threading.Thread(target=lambda: send_photo(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, img_path, caption), daemon=True).start()

    add_system_message_to_history(TELEGRAM_CHAT_ID, caption)

    # Bắt đầu ghi hình và các tác vụ khác
    if reason == "nguoi_la":
        start_clip_for_alert(shared_state.active_cameras.get(camera_name), frame, alert_id, duration=STRANGER_CLIP_DURATION)

    rec = recorder.start(reason=reason, duration=RECORD_SECONDS, wait_for_user=(reason == "nguoi_quen"))
    if rec:
        rec.setdefault("alert_ids", []).append(alert_id)
        log.info(f"Đã bắt đầu ghi hình mới cho cảnh báo {alert_id} -> {rec.get('path')}")
    else: # Nếu recorder bận, thử đính kèm vào bản ghi hiện tại
        if recorder.current:
            recorder.current.setdefault("alert_ids", []).append(alert_id)
            recorder.extend(RECORD_SECONDS)
            log.info(f"Đã đính kèm cảnh báo {alert_id} vào bản ghi hiện tại và kéo dài thời gian.")

    if not is_fire_alert:
        threading.Thread(target=user_response_watcher, args=(alert_id,), daemon=True).start()

def user_response_watcher(alert_id):
    """Theo dõi phản hồi của người dùng cho cảnh báo người quen/lạ."""
    try:
        resp = response_queue.get(timeout=USER_RESPONSE_WINDOW_SECONDS)
        if resp and resp.get("alert_id") == alert_id:
            recorder.resolve_user_wait()
            state.resolve_alert(alert_id, resp.get("raw_text"))
            if resp.get("decision") == "yes":
                log.info("Phản hồi an toàn -> dừng và xóa bản ghi.")
                recorder.stop_and_discard()
            return
    except queue.Empty:
        log.info(f"Không có phản hồi trong {USER_RESPONSE_WINDOW_SECONDS}s cho cảnh báo {alert_id}. Mở khóa ghi hình.")
        recorder.resolve_user_wait()

def start_clip_for_alert(cam, initial_frame, alert_id, duration=8, fps=20.0):
    """Tạo một video clip ngắn và gửi ngay lập tức khi có cảnh báo."""
    def worker():
        path = os.path.join(TMP_DIR, f"clip_{alert_id[:8]}_{uuid.uuid4().hex[:8]}.mp4")
        h, w = initial_frame.shape[:2]
        try:
            writer = cv2.VideoWriter(path, cv2.VideoWriter.fourcc(*"mp4v"), float(fps), (w, h))
            if not writer.isOpened(): return
            t0 = time.time()
            writer.write(initial_frame)
            while time.time() - t0 < float(duration):
                ret, frame = cam.read_raw() if hasattr(cam, "read_raw") else cam.read()
                if ret and frame is not None:
                    if frame.shape[:2] != (h, w): frame = cv2.resize(frame, (w, h))
                    writer.write(frame)
                time.sleep(1/fps)
            writer.release()
            send_video_or_document(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, path, caption=f"📹 Clip cảnh báo")
        except Exception as e:
            log.exception(f"Worker tạo clip thất bại: {e}")
    threading.Thread(target=worker, daemon=True).start()

def recorder_monitor_loop(cam):
    """Vòng lặp liên tục đọc frame và ghi vào video nếu recorder đang hoạt động."""
    log.info("Vòng lặp giám sát ghi hình đã bắt đầu.")
    while not getattr(cam, "quit", False):
        ret, frame = cam.read_raw() if hasattr(cam, "read_raw") else cam.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue
        try:
            if recorder.current:
                recorder.write(frame)
                finalized = recorder.check_and_finalize()
                if finalized:
                    path = finalized.get("path")
                    log.info(f"Bản ghi đã hoàn tất: {path}")
                    threading.Thread(target=lambda p=path: send_video_or_document(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, p, caption='📹 Bản ghi cảnh báo'), daemon=True).start()
        except Exception as e:
            log.exception(f"Lỗi trong quá trình ghi/hoàn tất bản ghi: {e}")
        time.sleep(0.02)

def run_gui(cam_instance):
    """Khởi chạy giao diện đồ họa trong luồng riêng."""
    root = ctk.CTk()
    FaceManagerApp(root, cam_instance)
    root.mainloop()

if __name__ == "__main__":
    init_alarm()
    threading.Thread(target=run_bot, daemon=True).start()
    log.info("Luồng Telegram bot đã bắt đầu.")

    camera_threads = []
    for name, src in IP_CAMERAS.items():
        try:
            log.info(f"Đang khởi tạo camera: {name} (Nguồn: {src})")
            cam = Camera(src, show_window=False)
            shared_state.active_cameras[name] = cam
            cam.on_alert_callback = partial(_on_alert, camera_name=name)
            thread = threading.Thread(target=cam.detect, name=f"CamThread-{name}", daemon=True)
            thread.start()
            camera_threads.append(thread)
            log.info(f"Đã bắt đầu luồng cho camera '{name}'.")
        except Exception as e:
            log.exception(f"Không thể bắt đầu camera '{name}': {e}")

    if shared_state.active_cameras:
        main_cam_instance = list(shared_state.active_cameras.values())[0]
        threading.Thread(target=run_gui, args=(main_cam_instance,), daemon=True).start()
        log.info("Luồng giao diện đồ họa (GUI) đã bắt đầu.")
        threading.Thread(target=recorder_monitor_loop, args=(main_cam_instance,), daemon=True).start()
    else:
        log.error("Không có camera nào được khởi tạo thành công. Thoát chương trình.")
        exit()

    try:
        while any(t.is_alive() for t in camera_threads):
            time.sleep(10)
        log.warning("Tất cả các luồng camera đã dừng. Thoát chương trình chính.")
    except KeyboardInterrupt:
        log.info("Bị ngắt bởi người dùng, đang thoát...")
    finally:
        for cam in shared_state.active_cameras.values():
            if hasattr(cam, 'delete'): cam.delete()
        log.info("Chương trình chính đã thoát.")
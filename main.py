# main.py
import threading
import time
import os
import uuid
import queue
import logging
import cv2
import customtkinter as ctk
import asyncio

from detection_core import Camera
import detection_core
from telegram_bot import run_bot, schedule_send_alert, add_system_message_to_history
from video_recorder import send_photo, send_video_or_document
from gui_manager import FaceManagerApp
from config import TELEGRAM_CHAT_ID, TELEGRAM_TOKEN, TMP_DIR, RECORD_SECONDS, IP_CAMERA_URL, USER_RESPONSE_WINDOW_SECONDS, STRANGER_CLIP_DURATION
from shared_state import state, response_queue, recorder, guard
import shared_state
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from alarm_player import init_alarm, play_alarm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
log = logging.getLogger("guardian")

sm = state
response_q = response_queue

def fire_alert_watcher(alert_id):
    """
    Một thread theo dõi cảnh báo cháy. Nếu không có phản hồi sau một khoảng thời gian,
    nó sẽ kích hoạt còi báo động.
    """
    log.info(f"Bắt đầu theo dõi cảnh báo cháy ID: {alert_id}. Sẽ chờ {USER_RESPONSE_WINDOW_SECONDS} giây.")
    time.sleep(USER_RESPONSE_WINDOW_SECONDS)

    alert_info = sm.get_alert_by_id(alert_id)
    if alert_info and not alert_info.get('resolved', False):
        log.warning(f"Không có phản hồi cho cảnh báo cháy ID {alert_id} sau {USER_RESPONSE_WINDOW_SECONDS} giây. KÍCH HOẠT CÒI BÁO ĐỘNG!")
        play_alarm()
    else:
        log.info(f"Cảnh báo cháy ID {alert_id} đã được xử lý. Không bật còi báo động.")


def _on_alert(frame, reason, name, meta):
    if reason == "nguoi_quen":
        alert_key = (reason, name)
    elif reason in ("lua_chay_khan_cap", "lua_chay_nghi_ngo"):
        alert_key = "lua_chay"
    else:
        alert_key = reason

    if sm.has_unresolved_alert(alert_key):
        log.info("Bỏ qua cảnh báo '%s' vì đã có một cảnh báo khác đang chờ phản hồi.", alert_key)
        return

    if not guard.allow(alert_key):
        log.info("Bỏ qua cảnh báo '%s' để tránh spam (debounce/muted).", alert_key)
        return

    log.info(">>> CẢNH BÁO MỚI ĐƯỢC PHÉP: %s", alert_key)

    chat_id = TELEGRAM_CHAT_ID
    img_path = os.path.join(TMP_DIR, f"alert_{reason}_{uuid.uuid4().hex}.jpg")

    try:
        cv2.imwrite(img_path, frame)
    except Exception as e:
        log.exception("Không thể ghi ảnh cảnh báo: %s", e)
        return

    alert_id = sm.create_alert(reason, chat_id, asked_for=name, image_path=img_path)

    caption = ""
    reply_markup = None
    is_fire_alert = False

    if reason == "nguoi_la":
        caption = f"⚠️ Phát hiện người lạ\n\nBạn có nhận ra người này không? (Trả lời trong {USER_RESPONSE_WINDOW_SECONDS}s: có/không)"
    elif reason == "nguoi_quen":
        caption = f"👋 Phát hiện {name}\n\nBạn có nhận ra người này không? (Trả lời trong {USER_RESPONSE_WINDOW_SECONDS}s: có/không)"
    elif reason == "lua_chay_nghi_ngo":
        is_fire_alert = True
        caption = "🟡 CẢNH BÁO VÀNG: Phát hiện dấu hiệu nghi ngờ cháy. Vui lòng kiểm tra hình ảnh và xác nhận."
    elif reason == "lua_chay_khan_cap":
        is_fire_alert = True
        caption = "🔴 CẢNH BÁO ĐỎ KHẨN CẤP: Phát hiện đám cháy đang phát triển hoặc có cả lửa và khói. Yêu cầu kiểm tra ngay lập tức!"

    if is_fire_alert:
        keyboard = [
            [
                InlineKeyboardButton("✅ Cháy thật", callback_data=f"fire_real:{alert_id}"),
                InlineKeyboardButton("❌ Báo động giả", callback_data=f"fire_false:{alert_id}"),
            ],
            [InlineKeyboardButton("📞 Gọi PCCC (114)", callback_data=f"fire_call:{alert_id}")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        schedule_send_alert(chat_id, img_path, caption, reply_markup)
        add_system_message_to_history(chat_id, caption)

        if reason == "lua_chay_khan_cap":
            threading.Thread(target=fire_alert_watcher, args=(alert_id,), daemon=True).start()

    else:
        threading.Thread(target=lambda: send_photo(TELEGRAM_TOKEN, chat_id, img_path, caption), daemon=True).start()
        add_system_message_to_history(chat_id, caption)

    if reason == "nguoi_la":
        cam_obj = globals().get("cam", None)
        try:
            start_clip_for_alert(cam_obj, frame, alert_id, duration=STRANGER_CLIP_DURATION, fps=recorder.fps)
            log.info("Đã bắt đầu worker tạo clip ngắn cho cảnh báo người lạ %s", alert_id)
        except Exception as e:
            log.exception("Không thể bắt đầu tạo clip ngắn cho cảnh báo %s: %s", alert_id, e)

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
            log.error("Hàm recorder.start() bị treo (vẫn chạy sau %.1fs)", timeout)
            return None
        try:
            status, val = q.get_nowait()
        except queue.Empty:
            log.error("Hàm recorder.start() không trả về giá trị nào")
            return None
        if status == "ok":
            return val
        else:
            log.exception("Hàm recorder.start() gây ra lỗi: %s", val)
            return None

    log.debug("Đang thử bắt đầu ghi hình cho cảnh báo %s", alert_id)
    
    wait_for_user_reply = (reason == "nguoi_quen")
    rec = _try_start_recorder(reason, RECORD_SECONDS, timeout=3.0, wait_for_user=wait_for_user_reply)
    
    if rec is None:
        log.warning("Trình ghi hình trả về None (bận/hết giờ/lỗi). Sẽ thử đính kèm hoặc mở rộng nếu có thể.")
        current = getattr(recorder, "current", None)
        if current:
            current.setdefault("alert_ids", []).append(alert_id)
            log.debug("Đã đính kèm cảnh báo %s vào bản ghi hiện tại", alert_id)
            try:
                if hasattr(recorder, "extend"):
                    recorder.extend(RECORD_SECONDS)
                    log.debug("Đã mở rộng thời gian ghi hình thêm %s giây", RECORD_SECONDS)
            except Exception:
                log.exception("Không thể mở rộng thời gian ghi hình")
        else:
            log.warning("Không có bản ghi nào đang hoạt động để đính kèm.")
    else:
        rec.setdefault("alert_ids", []).append(alert_id)
        rec["alert_id"] = alert_id
        log.info("Đã bắt đầu ghi hình mới cho cảnh báo %s -> %s", alert_id, rec.get("path", "<không-có-đường-dẫn>"))

    if not is_fire_alert:
        def watcher(aid):
            start = time.time()
            reply_received = False
            while time.time() - start < USER_RESPONSE_WINDOW_SECONDS:
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
                        log.info("Chủ nhà phản hồi an toàn -> dừng và xóa bản ghi")
                        recorder.stop_and_discard()
                    else:
                        log.info("Phản hồi không an toàn/không rõ -> tiếp tục ghi hình")
                    
                    return

            if not reply_received:
                log.info("Không có phản hồi trong %ds cho cảnh báo %s. Mở khóa chờ người dùng.", USER_RESPONSE_WINDOW_SECONDS, aid)
                recorder.resolve_user_wait()

        threading.Thread(target=watcher, args=(alert_id,), daemon=True).start()

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
                log.error("Worker tạo clip: VideoWriter không thể mở %s", path)
                return
        except Exception as e:
            log.exception("Worker tạo clip: không thể tạo VideoWriter: %s", e)
            return
        t0 = time.time()
        try:
            writer.write(initial_frame)
        except Exception as e:
            log.exception("Worker tạo clip: ghi frame đầu tiên thất bại: %s", e)
        while time.time() - t0 < float(duration):
            try:
                if hasattr(cam, "read_raw"):
                    ret, frame = cam.read_raw()
                else:
                    ret, frame = cam.read()

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
                log.exception("Worker tạo clip: lỗi khi đọc/ghi frame: %s", e)
                time.sleep(0.02)
                continue
        try:
            writer.release()
        except Exception:
            pass
        try:
            threading.Thread(target=lambda p=path: send_video_or_document(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, p, caption=f"📹 Clip cảnh báo ({reason})"), daemon=True).start()
        except Exception as e:
            log.exception("Worker tạo clip: không thể tạo luồng gửi video: %s", e)
    threading.Thread(target=worker, daemon=True).start()

def recorder_monitor_loop(cam):
    log.info("Vòng lặp giám sát ghi hình đã bắt đầu, loại camera: %s", type(cam))
    while True:
        if getattr(cam, "quit", False):
            log.info("Vòng lặp giám sát ghi hình đang thoát do cam.quit")
            break
        
        ret, frame = False, None
        try:
            if hasattr(cam, "read_raw"):
                ret, frame = cam.read_raw()
            else:
                ret, frame = cam.read()

            if (not ret) and hasattr(cam, "_raw_frame"):
                f = getattr(cam, "_raw_frame", None)
                if f is not None:
                    ret, frame = True, f.copy()
        except Exception as e:
            log.exception("Lỗi khi đọc frame cho trình ghi hình: %s", e)
            ret, frame = False, None
        
        if not ret or frame is None:
            time.sleep(0.05)
            continue
        
        try:
            if recorder.current:
                recorder.write(frame)
                finalized = recorder.check_and_finalize()
                if finalized:
                    path = finalized if isinstance(finalized, str) else finalized.get("path")
                    log.info("Bản ghi đã hoàn tất: %s", path)
                    threading.Thread(target=lambda p=path: send_video_or_document(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, p, caption='📹 Bản ghi cảnh báo'), daemon=True).start()
        except Exception as e:
            log.exception("Lỗi trong quá trình ghi/hoàn tất bản ghi: %s", e)
        time.sleep(0.02)

def run_gui(cam_instance):
    root = ctk.CTk()
    app = FaceManagerApp(root, cam_instance)
    root.mainloop()

if __name__ == "__main__":
    init_alarm()
    
    tbot = threading.Thread(target=run_bot, daemon=True)
    tbot.start()
    log.info("Luồng Telegram bot đã bắt đầu.")

    try:
        cam = Camera(IP_CAMERA_URL, show_window=False) 
        shared_state.camera_instance = cam
    except Exception as e:
        log.exception("Không thể tạo đối tượng Camera: %s", e)
        raise

    try:
        gui_thread = threading.Thread(target=run_gui, args=(cam,), daemon=True)
        gui_thread.start()
        log.info("Luồng giao diện đồ họa (GUI) đã bắt đầu.")
    except Exception as e:
        log.exception("Không thể bắt đầu luồng GUI: %s", e)

    threading.Thread(target=recorder_monitor_loop, args=(cam,), daemon=True).start()
    log.info("Luồng giám sát ghi hình đã bắt đầu.")

    try:
        cam.detect()
    except KeyboardInterrupt:
        log.info("Bị ngắt bởi người dùng, đang thoát...")
    except Exception as e:
        log.exception("Lỗi không xác định trong cam.detect: %s", e)
    finally:
        try:
            cam.delete()
        except Exception:
            pass
        log.info("Chương trình chính đang thoát.")
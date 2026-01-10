# main.py
import os
import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

import time
import queue
import uuid
import threading

os.environ['YOLO_VERBOSE'] = 'False'

# Import module
from config import settings, AlertType, AlertPriority
from core import CameraManager, Recorder, FaceDetector, FireDetector
from utils import state_manager, spam_guard, security, init_alarm, play_alarm, stop_alarm, memory_monitor, task_pool
from bot import GuardianBot, AIAssistant, send_photo, send_video
from gui import run_gui


# Controller hệ thống
class GuardianApp:
    
    def __init__(self):
        
        self.state = state_manager              # Quản lý trạng thái
        self.spam_guard = spam_guard            # Chống spam
        self.recorder = Recorder()              # Trình ghi hình
        self.response_queue = queue.Queue()     # Hàng đợi phản hồi
        self.shutdown_event = threading.Event()
        self.threads = []
        self.is_alarm_playing = False
    
    # Khởi tạo hệ thống
    def initialize(self):
        # Theo dõi RAM
        memory_monitor.start()
        
        # Khởi tạo Còi báo động
        if not init_alarm():
            print("[WARN] Còi báo động lỗi!")
            return False
        
        # Khởi tạo Nhận diện khuôn mặt
        self.face_detector = FaceDetector()
        
        if not self.face_detector.initialize():
            print("[ERR] Face Detector lỗi!")
            return False
            
        self.face_detector.load_known_faces()
        print("[OK] Face Detector đã sẵn sàng!")
        
        # Khởi tạo Nhận diện lửa
        self.fire_detector = FireDetector()
        
        if not self.fire_detector.initialize():
            print("[ERR] Fire Detector lỗi!")
            return False

        print("[OK] Fire Detector đã sẵn sàng!")
        
        # Khởi tạo Quản lý Camera
        self.camera_manager = CameraManager(
            person_alert=self.person_alert,
            fire_alert=self.fire_alert,
            fall_alert=self.fall_alert
        )
        
        self.camera_manager.start(
            self.fire_detector,
            self.face_detector,
            self.state
        )
        print("[OK] Camera Manager đã sẵn sàng!")
        
        # Khởi tạo AI & Bot
        self.ai_assistant = AIAssistant()
        
        self.bot = GuardianBot(
            self.ai_assistant,
            self,
            self.get_snapshot,
            self.camera_manager,
            self.response_queue
        )
        print("[OK] Telegram Bot đã sẵn sàng!")
        
        print("[OK] KHỞI TẠO HOÀN TẤT!")
        return True
    
    # Xử lý cảnh báo người
    def person_alert(self, source_id, frame, alert_type, metadata):
        # Check phải là dictionary
        if not isinstance(metadata, dict):
            metadata = {}
        
        # Tạo key anti-spam
        if alert_type == AlertType.KNOWN_PERSON:
            key = (alert_type, metadata.get('name'), source_id) # Tạo key cho người quen
        else:
            key = (alert_type, source_id) # Tạo key cho người lạ
        
        # Bỏ qua nếu đã có cảnh báo (chưa giải quyết)
        if alert_type == AlertType.STRANGER and self.state.has_unresolved(alert_type, source_id):
            return
        
        # Check có được gửi không
        if not self.spam_guard.allow(key):
            return
        
        # Lưu ảnh vào folder temp
        img_path = settings.paths.tmp_dir / f"alert_{uuid.uuid4().hex}.jpg"
        security.save_image(img_path, frame)
        
        # Tạo cảnh báo trong hệ thống
        alert_id = self.state.create_alert(
            alert_type=alert_type,
            source_id=source_id,
            chat_id=settings.telegram.chat_id,
            image_path=str(img_path),
            name=metadata.get('name')
        )
        
        priority = self.get_priority(alert_type, metadata) # Xác định mức độ ưu tiên
        caption = self.get_caption(alert_type, source_id, metadata, priority) # Tạo nội dung tin nhắn
        
        # Gửi cảnh báo qua Tele
        if self.bot:
            self.bot.schedule_alert(
                settings.telegram.chat_id,
                str(img_path),
                caption,
                alert_id,
                is_fire=False,
                silent=(priority == AlertPriority.LOW)  # Không kêu nếu ưu tiên thấp
            )
        
        # Thêm vào ngữ cảnh cho AI
        if self.ai_assistant:
            self.ai_assistant.add_context(settings.telegram.chat_id, caption)
        
        # Ghi video & chờ phản hồi
        self.start_recording(source_id, alert_id)
        
        threading.Thread(
            target=self.watch_response,
            args=(alert_id,),
            daemon=True
        ).start()
    
    # Xử lý cảnh báo cháy
    def fire_alert(self, source_id, frame, alert_type):
        # Check có phải cảnh báo khẩn cấp        
        critical = (alert_type == AlertType.FIRE_CRITICAL)
        key = (alert_type, source_id)
        
        # Bỏ qua nếu đã có cảnh báo cháy chưa giải quyết
        if self.state.has_unresolved(alert_type, source_id):
            return
        
        # Anti-spam (Critical ưu tiên cao)
        if not self.spam_guard.allow(key, critical):
            return
        
        # Lưu ảnh
        img_path = settings.paths.tmp_dir / f"fire_{uuid.uuid4().hex}.jpg"
        security.save_image(img_path, frame)
        
        # Tạo cảnh báo
        alert_id = self.state.create_alert(
            alert_type=alert_type,
            source_id=source_id,
            chat_id=settings.telegram.chat_id,
            image_path=str(img_path)
        )
        
        # Tạo nội dung tin nhắn
        if critical:
            caption = f"[CẢNH BÁO ĐỎ] Phát hiện cháy tại camera {source_id}!"
        else:
            caption = f"[CẢNH BÁO VÀNG] Nghi ngờ có cháy tại camera {source_id}"
        
        # Gửi qua Telegram
        if self.bot:
            self.bot.schedule_alert(
                settings.telegram.chat_id,
                str(img_path),
                caption,
                alert_id,
                is_fire=True
            )
        
        # Thêm vào ngữ cảnh AI
        if self.ai_assistant:
            self.ai_assistant.add_context(settings.telegram.chat_id, caption)
        
        # Bắt đầu quay video
        self.start_recording(source_id, alert_id)
        
        # Nếu khẩn cấp, chờ phản hồi rồi bật còi
        if critical:
            threading.Thread(
                target=self.watch_fire_alert,
                args=(alert_id,),
                daemon=True
            ).start()
    
    # Xử lý té ngã
    def fall_alert(self, source_id, frame, alert_type):
        key = (alert_type, source_id)
        
        # Bỏ qua nếu đã có cảnh báo té ngã chưa giải quyết
        if self.state.has_unresolved(alert_type, source_id):
            return
        
        # Check chống spam
        if not self.spam_guard.allow(key, is_critical=True):
            return
        
        # Lưu ảnh
        img_path = settings.paths.tmp_dir / f"fall_{uuid.uuid4().hex}.jpg"
        security.save_image(img_path, frame)
        
        # Tạo cảnh báo
        alert_id = self.state.create_alert(
            alert_type=alert_type,
            source_id=source_id,
            chat_id=settings.telegram.chat_id,
            image_path=str(img_path)
        )
        
        # Tạo nội dung tin nhắn
        caption = f"[ALERT] CẢNH BÁO: Phát hiện người té ngã tại camera {source_id}!"
        
        # Gửi qua Telegram
        if self.bot:
            self.bot.schedule_alert(
                settings.telegram.chat_id,
                str(img_path),
                caption,
                alert_id,
                is_fire=False
            )
        
        # Thêm vào ngữ cảnh AI
        if self.ai_assistant:
            self.ai_assistant.add_context(settings.telegram.chat_id, caption)
        
        # Bắt đầu quay video
        self.start_recording(source_id, alert_id)
    
    # Các hàm hỗ trợ
    
    # Xác định mức độ ưu tiên
    def get_priority(self, alert_type, metadata):
        if alert_type in [AlertType.FIRE_CRITICAL, AlertType.FIRE_WARNING]:
            return AlertPriority.CRITICAL  # Cao nhất
        if alert_type == AlertType.STRANGER:
            return AlertPriority.MEDIUM    # Trung bình
        return AlertPriority.LOW           # Thấp
    
    # Tạo nội dung tin nhắn
    def get_caption(self, alert_type, source_id, metadata, priority):
        if priority == AlertPriority.CRITICAL:
            return f"[CRITICAL] KHẨN CẤP - Có cháy tại camera {source_id}!"
        elif priority == AlertPriority.MEDIUM:
            return f"[WARN] Phát hiện người lạ tại camera {source_id}"
        else:
            name = metadata.get('name', 'Unknown')
            return f"[INFO] {name} đang ở camera {source_id}"
    
    # Quay video
    def start_recording(self, source_id, alert_id):
        # Get thời gian quay từ config
        duration = settings.get('recorder.duration_seconds', 30)
        
        rec = self.recorder.start(
            source_id=source_id,
            reason="alert",
            duration=duration
        )
        
        # Add ID cảnh báo vào list
        if rec:
            rec['alert_ids'].append(alert_id)
    
    # Chờ user phản hồi
    def watch_response(self, alert_id):
        # Get thời gian chờ từ config
        timeout = settings.telegram.user_response_window_seconds
        start = time.time()
        
        # Chờ trong khoảng thời gian cho phép
        while time.time() - start < timeout:
            try:
                # Get phản hồi từ queue (chờ 1 giây)
                resp = self.response_queue.get(timeout=1.0)
                
                # Check có phải phản hồi cho cảnh báo này không
                if resp and resp.get('alert_id') == alert_id:
                    # Nếu người dùng nói không sao thì hủy video
                    if resp.get('decision') in ('yes', 'left'):
                        self.recorder.discard()
                    return 
            except queue.Empty:
                # Không có phản hồi, tiếp tục chờ
                continue
    
    # Báo động nếu không phản hồi
    def watch_fire_alert(self, alert_id):
        # Chờ người dùng phản hồi
        time.sleep(settings.telegram.user_response_window_seconds)
        
        # Kiểm tra cảnh báo đã được xử lý chưa
        alert = self.state.get_alert(alert_id)
        if alert and not alert.resolved:
            # Chưa xử lý -> Bật còi báo động!
            self.play()
    
    # Chụp ảnh từ camera
    def get_snapshot(self, chat_id, source=None):
        if not sel.camera_manager:
            return
        
        # Lấy danh sách camera
        cameras = list(self.camera_manager.cameras.keys())
        if not cameras:
            return
        
        # Xác định camera cần chụp
        cam_id = source or cameras[0]
        if source and source.isdigit():
            idx = int(source)
            if 0 <= idx < len(cameras):
                cam_id = cameras[idx]
        
        # Lấy camera
        cam = self.camera_manager.get_camera(cam_id)
        if not cam:
            return
        
        # Đọc frame
        ret, frame = cam.read_raw()
        if not ret or frame is None:
            return
        
        # Lưu ảnh
        img_path = settings.paths.tmp_dir / f"snap_{uuid.uuid4().hex}.jpg"
        security.save_image(img_path, frame)
        
        # Gửi ảnh (chạy trong thread để không block)
        threading.Thread(
            target=lambda: send_photo(chat_id, str(img_path), f"[SNAP] Camera {cam_id}"),
            daemon=True
        ).start()
    
    # Loop ghi video
    def recorder_loop(self):
        # Dùng target_fps từ camera config (FPS mà camera thực sự output)
        target_fps = settings.camera.target_fps
        frame_interval = 1.0 / target_fps
        last_frame_time = 0
        fps_updated = False
        
        while not self.shutdown_event.is_set():
            now = time.time()
            
            # Kiểm tra có đang ghi video không
            if self.recorder.dang_ghi and self.camera_manager:
                # Cập nhật FPS cho recorder (chỉ 1 lần mỗi phiên ghi)
                if not fps_updated:
                    self.recorder.fps = target_fps
                    fps_updated = True
                    print(f"[DEBUG] Target FPS config: {target_fps}")
                    print(f"[DEBUG] Frame interval: {frame_interval:.4f}s")
                
                # Điều khiển tốc độ ghi frame
                if now - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue
                last_frame_time = now
                
                source_id = self.recorder.dang_ghi.get('source_id')
                cam = self.camera_manager.get_camera(source_id) if source_id else None
                
                if cam:
                    # Đọc frame và ghi vào video
                    ret, frame = cam.read_raw()
                    if ret and frame is not None:
                        self.recorder.write(frame)
                    
                    # Kiểm tra xem có cần kéo dài thời gian ghi không
                    end_time = self.recorder.dang_ghi.get('end_time', 0)
                    
                    if 0 < end_time - now < 5.0:  # Còn dưới 5 giây
                        if cam.has_active_threat():  # Vẫn còn nguy hiểm
                            extension = settings.get('recorder.extension_seconds', 10)
                            self.recorder.extend(extension)
                
                # Kiểm tra hoàn thành ghi video
                result = self.recorder.check_finalize()
                if result:
                    # Reset flag để lần ghi tiếp theo cập nhật lại
                    fps_updated = False
                    
                    # Gửi video qua Telegram
                    task_pool.submit(
                        send_video,
                        settings.telegram.chat_id,
                        str(result['path']),
                        "[VIDEO] Video cảnh báo"
                    )
            else:
                # Khi không ghi, reset flag
                fps_updated = False
                time.sleep(0.5)
    
    # Check hệ thống còn sống
    def life_loop(self):
        interval = 300  # 5 phút
        last_beat = 0
        
        while not self.shutdown_event.is_set():
            now = time.time()
            
            if now - last_beat >= interval:
                if self.bot:
                    self.bot.send_heartbeat()
                    last_beat = now
            
            time.sleep(60)  # Kiểm tra mỗi phút
    
    # Điều khiển còi báo động
    def play(self):
        self.is_alarm_playing = True
        play_alarm()
    
    def stop(self):
        stop_alarm()
        self.is_alarm_playing = False
    
    def run(self):
        # Khởi tạo hệ thống
        if not self.initialize():
            print("[ERR] Khởi tạo thất bại! Đang tắt...")
            self.shutdown()
            return
        
        # Chạy Telegram Bot trong thread riêng
        if self.bot:
            t = threading.Thread(target=self.bot.run, daemon=True)
            t.start()
            self.threads.append(t)
        
        # Chạy kiểm tra sức khỏe hệ thống
        if self.bot:
            t = threading.Thread(target=self.life_loop, daemon=True)
            t.start()
            self.threads.append(t)
        
        # Chạy ghi video
        t = threading.Thread(target=self.recorder_loop, daemon=True)
        t.start()
        self.threads.append(t)
        
        # Chạy giao diện GUI
        t = threading.Thread(
            target=run_gui,
            args=(self.camera_manager, self.face_detector, self.state, self),
            daemon=True
        )
        t.start()
        self.threads.append(t)
        
        print("[OK] Hệ thống đang chạy!")
        
        # Xử lý sự kiện nhấn Ctrl + C
        try:
            while not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
    
    # Tắt hệ thống
    def shutdown(self):
        # Tránh gọi nhiều lần
        if self.shutdown_event.is_set():
            return
        
        print("Đang tắt hệ thống...")
        
        # Báo hiệu tất cả thread dừng
        self.shutdown_event.set()
        
        # Dừng giám sát bộ nhớ
        memory_monitor.stop()
        
        # Dừng task pool
        task_pool.shutdown()
        
        # Dừng Telegram Bot
        if hasattr(self, 'bot') and self.bot:
            self.bot.stop()
        
        # Dừng tất cả camera
        if hasattr(self, 'camera_manager') and self.camera_manager:
            self.camera_manager.stop()
        
        print("[OK] Đã tắt hệ thống hoàn toàn!")


# Bắt đầu chương trình
def main():
    app = GuardianApp()
    app.run()


if __name__ == "__main__":
    main()

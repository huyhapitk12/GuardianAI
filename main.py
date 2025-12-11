# main.py0
import os
import time
import queue
import uuid
import threading

# Tắt thông báo của YOLO
os.environ['YOLO_VERBOSE'] = 'False'

# Import các module của dự án
from config import settings, AlertType, AlertPriority
from core import CameraManager, Recorder, FaceDetector, FireDetector
from utils import state_manager, spam_guard, security, init_alarm, play_alarm, stop_alarm, memory_monitor, task_pool
from bot import GuardianBot, AIAssistant, send_photo, send_video
from core.detection import BehaviorAnalyzer
from gui import run_gui


# Class điều khiển hệ thống
class GuardianApp:
    
    def __init__(self):
        
        # Quản lý trạng thái hệ thống (bật/tắt camera, cảnh báo,...)
        self.state = state_manager
        
        # Chống spam tin nhắn
        self.spam_guard = spam_guard
        
        # Quay video
        self.recorder = Recorder()
        
        # Hàng đợi chờ phản hồi
        self.response_queue = queue.Queue()
        
        # Báo thread tắt chương trình
        self.shutdown_event = threading.Event()
        
        # Danh sách các thread
        self.threads = []
        
        # Trạng thái còi
        self.is_alarm_playing = False
    
    # Hàm này chạy khởi tạo hệ thống
    def initialize(self):
        print("🚀 Bắt đầu khởi tạo hệ thống...")
        
        # Theo dõi RAM
        memory_monitor.start()
        
        # Khởi tạo còi báo động
        if not init_alarm():
            print("⚠️ Còi báo động không chạy được!")
            return False
        
        # Khởi tạo nhận diện khuôn mặt
        print("📷 Đang tải bộ nhận diện khuôn mặt...")
        self.face_detector = FaceDetector()
        
        if not self.face_detector.initialize():
            print("❌ Bộ nhận diện khuôn mặt không chạy được!")
            return False
        
        # Tải danh sách face đã biết
        self.face_detector.load_known_faces()
        
        # Khởi tạo phát hiện cháy
        print("🔥 Đang tải bộ phát hiện cháy...")
        self.fire_detector = FireDetector()
        
        if not self.fire_detector.initialize():
            print("❌ Bộ phát hiện cháy không chạy được!")
            return False
        
        # Khởi tạo phân tích hành vi
        if settings.get('behavior.enabled', False):
            print("🧠 Đang tải bộ phân tích hành vi...")
            try:
                # Get path model từ config
                model_path = settings.get('behavior.model_path', 'Data/Model/anomaly_model.pth')
                model_path = settings.base_dir / model_path
                
                # Get thiết bị chạy
                device = settings.get('behavior.device', 'cpu')
                
                # Get ngưỡng
                threshold = settings.get('behavior.threshold', 0.5)
                
                # Check file model
                if not model_path.exists():
                    raise FileNotFoundError(f"Không tìm thấy model hành vi: {model_path}")
                
                # Bộ phân tích
                self.behavior_analyzer = BehaviorAnalyzer(
                    model_path=str(model_path),
                    device=device,
                    threshold=threshold
                )
                print("✅ Bộ phân tích hành vi đã sẵn sàng!")
                
            except Exception as e:
                print(f"⚠️ Không thể tải phân tích hành vi: {e}")
                self.behavior_analyzer = None
        else:
            print("🧠 Phân tích hành vi đang tắt (có thể bật trong config)")
            self.behavior_analyzer = None
        
        # Khởi tạo camera
        print("📹 Đang kết nối camera...")
        try:
            # Quản lý camera
            self.camera_manager = CameraManager(
                person_alert=self.person_alert,
                fire_alert=self.fire_alert
            )
            
            # Chạy camera
            self.camera_manager.start(
                self.fire_detector,
                self.face_detector,
                self.state,
                self.behavior_analyzer
            )
        except Exception as e:
            print(f"❌ Lỗi camera: {e}")
            return False
        
        # Khởi tạo AI Assistant
        self.ai_assistant = AIAssistant()
        
        # Khởi tạo Telegram Bot
        try:
            self.bot = GuardianBot(
                self.ai_assistant,
                self,
                self.get_snapshot,
                self.camera_manager,
                self.response_queue
            )
            print("✅ Telegram Bot đã sẵn sàng!")
        except Exception as e:
            print(f"⚠️ Telegram Bot không chạy được: {e}")
            self.bot = None
        
        print("✅ KHỞI TẠO HOÀN TẤT!")
        return True
    
    # XỬ LÝ CẢNH BÁO NGƯỜI
    def person_alert(self, source_id, frame, alert_type, metadata): # source_id: ID camera, frame: Hình ảnh, alert_type: Loại cảnh báo, metadata: Thông tin thêm
        # Check phải là dictionary
        if not isinstance(metadata, dict):
            metadata = {}
        
        # Tạo key cho cảnh báo chống spam
        if alert_type == AlertType.KNOWN_PERSON:
            key = (alert_type, metadata.get('name'), source_id) # Tạo key cho người quen
        else:
            key = (alert_type, source_id) # Tạo key cho người lạ
        
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
        
        # Bắt đầu quay video
        self.start_recording(source_id, alert_id)
        
        # Chạy thread chờ phản hồi người dùng
        threading.Thread(
            target=self.watch_response,
            args=(alert_id,),
            daemon=True
        ).start()
    
    # XỬ LÝ CẢNH BÁO CHÁY
    def fire_alert(self, source_id, frame, alert_type): # source_id: ID camera, frame: Hình ảnh, alert_type: Loại cảnh báo
        # Check có phải cảnh báo khẩn cấp        
        critical = (alert_type == AlertType.FIRE_CRITICAL)
        key = (alert_type, source_id)
        
        # Check chống spam (ưu tiên cảnh báo khẩn cấp)
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
            caption = f"🔴 NGUY HIỂM: Phát hiện cháy tại camera {source_id}!"
        else:
            caption = f"🟡 CẢNH BÁO: Nghi ngờ có cháy tại camera {source_id}"
        
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
    
    # CÁC HÀM HỖ TRỢ
    
    # Xác định độ ưu tiên cảnh báo
    def get_priority(self, alert_type, metadata): # alert_type: Loại cảnh báo, metadata: Thông tin cảnh báo
        if alert_type in [AlertType.FIRE_CRITICAL, AlertType.FIRE_WARNING]:
            return AlertPriority.CRITICAL  # Cao nhất
        if alert_type == AlertType.ANOMALOUS_BEHAVIOR:
            return AlertPriority.HIGH      # Cao
        if alert_type == AlertType.STRANGER:
            return AlertPriority.MEDIUM    # Trung bình
        return AlertPriority.LOW           # Thấp
    
    # Tạo nội dung tin nhắn cảnh báo
    def get_caption(self, alert_type, source_id, metadata, priority): # alert_type: Loại cảnh báo, source_id: ID camera, metadata: Thông tin cảnh báo, priority: Độ ưu tiên
        if priority == AlertPriority.CRITICAL:
            return f"🚨🔥 KHẨN CẤP - Có cháy tại camera {source_id}!"
        elif priority == AlertPriority.HIGH:
            score = metadata.get('score', 0)
            return f"⚠️🚨 CẢNH BÁO - Hành vi bất thường ({score:.2f}) tại camera {source_id}"
        elif priority == AlertPriority.MEDIUM:
            return f"⚠️ Phát hiện người lạ tại camera {source_id}"
        else:
            name = metadata.get('name', 'Ai đó')
            return f"👋 {name} đang ở camera {source_id}"
    
    # Quay video khi có cảnh báo
    def start_recording(self, source_id, alert_id): # source_id: ID camera, alert_id: ID cảnh báo
        try:
            # Get thời gian quay từ config
            duration = settings.get('recorder.duration', 30)
            
            rec = self.recorder.start(
                source_id=source_id,
                reason="alert",
                duration=duration
            )
            
            # Add ID cảnh báo vào list
            if rec:
                rec['alert_ids'].append(alert_id)
                
        except Exception as e:
            print(f"Lỗi quay video: {e}")
    
    # Chờ phản hồi người dùng
    def watch_response(self, alert_id): # alert_id: ID cảnh báo
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
        if not self.camera_manager:
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
            target=lambda: send_photo(chat_id, str(img_path), f"📸 Camera {cam_id}"),
            daemon=True
        ).start()
    
    # Ghi video khi cần
    def recorder_loop(self):
        while not self.shutdown_event.is_set():
            try:
                # Kiểm tra có đang ghi video không
                if self.recorder.current and self.camera_manager:
                    source_id = self.recorder.current.get('source_id')
                    cam = self.camera_manager.get_camera(source_id) if source_id else None
                    
                    if cam:
                        # Đọc frame và ghi vào video
                        ret, frame = cam.read_raw()
                        if ret and frame is not None:
                            self.recorder.write(frame)
                        
                        # Kiểm tra xem có cần kéo dài thời gian ghi không
                        end_time = self.recorder.current.get('end_time', 0)
                        now = time.time()
                        
                        if 0 < end_time - now < 5.0:  # Còn dưới 5 giây
                            if cam.has_active_threat():  # Vẫn còn nguy hiểm
                                extension = settings.get('recorder.extension_seconds', 10)
                                self.recorder.extend(extension)
                    
                    # Kiểm tra hoàn thành ghi video
                    result = self.recorder.check_finalize()
                    if result:
                        # Gửi video qua Telegram
                        task_pool.submit(
                            send_video,
                            settings.telegram.chat_id,
                            str(result['path']),
                            "📹 Video cảnh báo"
                        )
                else:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"Lỗi trong vòng lặp ghi video: {e}")
            
            time.sleep(0.1)
    
    # Kiểm tra hệ thống còn sống
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
    
    # Chạy chương trình chính
    def run(self):
        # Khởi tạo hệ thống
        if not self.initialize():
            print("❌ Khởi tạo thất bại! Đang tắt...")
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
        
        print("=" * 50)
        print("✅ HỆ THỐNG ĐANG CHẠY!")
        print("Nhấn Ctrl+C để tắt.")
        print("=" * 50)
        
        # Xử lý sự kiện nhấn Ctrl + C
        try:
            while not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⚠️ Nhận lệnh tắt từ bàn phím...")
        finally:
            self.shutdown()
    
    # Tắt chương trình
    def shutdown(self):
        # Tránh gọi nhiều lần
        if self.shutdown_event.is_set():
            return
        
        print("🛑 Đang tắt hệ thống...")
        
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
        
        print("✅ Đã tắt hệ thống hoàn toàn!")


# Bắt đầu chương trình
def main():
    print("=" * 60)
    print("       GUARDIANAI - HỆ THỐNG GIÁM SÁT AN NINH THÔNG MINH")
    print("=" * 60)
    
    # Tạo và chạy ứng dụng
    app = GuardianApp()
    app.run()


if __name__ == "__main__":
    main()

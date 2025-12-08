# main.py
import os, time, queue, uuid, threading
from typing import Optional

os.environ['YOLO_VERBOSE'] = 'False'

from config import settings, AlertType, AlertPriority
from core import CameraManager, Recorder, FaceDetector, FireDetector
from utils import state_manager, spam_guard, security, init_alarm, play_alarm, stop_alarm, memory_monitor, task_pool
from bot import GuardianBot, AIAssistant, send_photo, send_video
from core.detection import BehaviorAnalyzer
from gui import run_gui


class GuardianApp:
    def __init__(self):
        self.state = state_manager                  # Quản lý trạng thái
        self.spam_guard = spam_guard                # Chống spam
        self.recorder = Recorder()                  # Quay video
        self.response_queue = queue.Queue()         # Queue chờ phản hồi
        self.shutdown_event = threading.Event()     # Báo thread tắt
        
        self.threads = []                          # Danh sách thread
        self.is_alarm_playing = False              # Trạng thái còi
    
    def initialize(self) -> bool:
        print("🚀 Bắt đầu khởi tạo...")
        
        # Chạy giám sát bộ nhớ
        memory_monitor.start()
        
        if not init_alarm():
            print("⚠️ Còi không chạy được")
            return False
        
        print("📷 Chạy bộ nhận diện face...")
        self.face_detector = FaceDetector()
        if not self.face_detector.initialize():
            print("❌ Bộ nhận diện face không chạy được")
            return False
        self.face_detector.load_known_faces()
        
        print("🔥 Chạy phát hiện cháy...")
        self.fire_detector = FireDetector()
        if not self.fire_detector.initialize():
            print("❌ Phát hiện cháy không chạy được")
            return False
        
        # Kiểm tra tùy chọn hành vi bất thường
        if settings.get('behavior.enabled', False):
            print("🧠 Loading behavior analyzer...")
            try:
                mode_path = settings.get('behavior.model_path', 'Data/Model/anomaly_model.pth')
                mode_path = settings.base_dir / mode_path
                device = settings.get('behavior.device', 'cpu')
                threshold = settings.get('behavior.threshold', 0.5)
                
                if not mode_path.exists():
                    raise FileNotFoundError(f"Không phát hiện model hành vi: {mode_path}")

                self.behavior_analyzer = BehaviorAnalyzer(
                    model_path=str(mode_path),
                    device=device,
                    threshold=threshold
                )
            except Exception as e:
                print(f"⚠️ Không khởi tạo được phân tích hành vi bất thường: {e}")
                self.behavior_analyzer = None

        else:
            print("🧠 Hành vi bất thường không được bật")
            self.behavior_analyzer = None
        
        print("📹 Bắt đầu lấy video...")
        try:
            self.camera_manager = CameraManager(
                on_person_alert=self.person_alert,
                on_fire_alert=self.fire_alert
            )
            self.camera_manager.start(
                self.fire_detector,
                self.face_detector,
                self.state,
                self.behavior_analyzer
            )
        except Exception as e:
            print(f"❌ Lỗi camera: {e}")
            return False
        
        self.ai_assistant = AIAssistant()
        
        try:
            self.bot = GuardianBot(
                self.ai_assistant,
                self,
                self.get_snapshot,
                self.camera_manager,
                self.response_queue
            )
            print("✅ Bot đã sẵn sàng")
        except Exception as e:
            print(f"⚠️ Bot không chạy được: {e}")
        
        print("✅ Đã hoàn thành khởi tạo")
        return True
    
    # Xử lý cảnh báo người
    def person_alert(self, source_id: str, frame, alert_type: str, metadata: dict):
        if not isinstance(metadata, dict):
            metadata = {}
        
        if alert_type == AlertType.KNOWN_PERSON:
            key = (alert_type, metadata.get('name'), source_id)
        else:
            key = (alert_type, source_id)
        
        if not self.spam_guard.allow(key):
            return
        
        img_path = settings.paths.tmp_dir / f"alert_{uuid.uuid4().hex}.jpg"
        security.save_image(img_path, frame)
        
        alert_id = self.state.create_alert(
            alert_type=alert_type,
            source_id=source_id,
            chat_id=settings.telegram.chat_id,
            image_path=str(img_path),
            name=metadata.get('name')
        )
        
        priority = self._get_priority(alert_type, metadata)
        caption = self._get_caption(alert_type, source_id, metadata, priority)
        
        if self.bot:
            self.bot.schedule_alert(
                settings.telegram.chat_id,
                str(img_path),
                caption,
                alert_id,
                is_fire=False,
                silent=(priority == AlertPriority.LOW)
            )
        
        if self.ai_assistant:
            self.ai_assistant.add_context(settings.telegram.chat_id, caption)
        
        self._start_recording(source_id, alert_id)
        
        threading.Thread(
            target=self._watch_response,
            args=(alert_id,),
            daemon=True
        ).start()
    
    # Xử lý cảnh báo cháy
    def fire_alert(self, source_id: str, frame, alert_type: str):
        is_critical = (alert_type == AlertType.FIRE_CRITICAL)
        key = (alert_type, source_id)
        
        if not self.spam_guard.allow(key, is_critical):
            return
        
        img_path = settings.paths.tmp_dir / f"fire_{uuid.uuid4().hex}.jpg"
        security.save_image(img_path, frame)
        
        alert_id = self.state.create_alert(
            alert_type=alert_type,
            source_id=source_id,
            chat_id=settings.telegram.chat_id,
            image_path=str(img_path)
        )
        
        if is_critical:
            caption = f"🔴 NGUY HIỂM: Phát hiện cháy tại camera {source_id}!"
        else:
            caption = f"🟡 CẢNH BÁO: Nghi ngờ cháy tại camera {source_id}"
        
        if self.bot:
            self.bot.schedule_alert(
                settings.telegram.chat_id,
                str(img_path),
                caption,
                alert_id,
                is_fire=True
            )
        
        if self.ai_assistant:
            self.ai_assistant.add_context(settings.telegram.chat_id, caption)
        
        self._start_recording(source_id, alert_id)
        
        if is_critical:
            threading.Thread(
                target=self._watch_fire_alert,
                args=(alert_id,),
                daemon=True
            ).start()
    
    def _get_priority(self, alert_type: str, metadata: dict) -> AlertPriority:
        """Determine alert priority"""
        if alert_type in [AlertType.FIRE_CRITICAL, AlertType.FIRE_WARNING]:
            return AlertPriority.CRITICAL
        if alert_type == AlertType.ANOMALOUS_BEHAVIOR:
            return AlertPriority.HIGH
        if alert_type == AlertType.STRANGER:
            return AlertPriority.MEDIUM
        return AlertPriority.LOW
    
    def _get_caption(self, alert_type: str, source_id: str, 
                     metadata: dict, priority: AlertPriority) -> str:
        """Generate alert caption"""
        if priority == AlertPriority.CRITICAL:
            return f"🚨🔥 KHẨN CẤP - Cháy tại camera {source_id}!"
        elif priority == AlertPriority.HIGH:
            score = metadata.get('score', 0)
            return f"⚠️🚨 CẢNH BÁO - Hành vi bất thường ({score:.2f}) tại camera {source_id}"
        elif priority == AlertPriority.MEDIUM:
            return f"⚠️ Phát hiện người lạ tại camera {source_id}"
        else:
            name = metadata.get('name', 'Unknown')
            return f"👋 {name} tại camera {source_id}"
    
    def _start_recording(self, source_id: str, alert_id: str):
        """Start recording for alert"""
        try:
            rec = self.recorder.start(
                source_id=source_id,
                reason="alert",
                duration=settings.get('recorder.duration', 30)
            )
            if rec:
                rec['alert_ids'].append(alert_id)
        except Exception as e:
            print(f"Recording error: {e}")
    
    def _watch_response(self, alert_id: str):
        """Watch for user response"""
        timeout = settings.telegram.user_response_window_seconds
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                resp = self.response_queue.get(timeout=1.0)
                if resp and resp.get('alert_id') == alert_id:
                    if resp.get('decision') in ('yes', 'left'):
                        self.recorder.discard()
                    return
            except queue.Empty:
                continue
    
    def _watch_fire_alert(self, alert_id: str):
        """Watch fire alert and trigger alarm"""
        time.sleep(settings.telegram.user_response_window_seconds)
        
        alert = self.state.get_alert(alert_id)
        if alert and not alert.resolved:
            self.play()
    
    def get_snapshot(self, chat_id: str, source: str = None):
        if not self.camera_manager:
            return
        
        cameras = list(self.camera_manager.cameras.keys())
        if not cameras:
            return
        
        cam_id = source or cameras[0]
        if source and source.isdigit():
            idx = int(source)
            if 0 <= idx < len(cameras):
                cam_id = cameras[idx]
        
        cam = self.camera_manager.get_camera(cam_id)
        if not cam:
            return
        
        ret, frame = cam.read_raw()
        if not ret or frame is None:
            return
        
        img_path = settings.paths.tmp_dir / f"snap_{uuid.uuid4().hex}.jpg"
        security.save_image(img_path, frame)
        
        threading.Thread(
            target=lambda: send_photo(chat_id, str(img_path), f"📸 Camera {cam_id}"),
            daemon=True
        ).start()
    
    def _recorder_loop(self):
        """Monitor and finalize recordings"""
        while not self.shutdown_event.is_set():
            try:
                if self.recorder.current and self.camera_manager:
                    source_id = self.recorder.current.get('source_id')
                    cam = self.camera_manager.get_camera(source_id) if source_id else None
                    
                    if cam:
                        ret, frame = cam.read_raw()
                        if ret and frame is not None:
                            self.recorder.write(frame)
                        
                        # === SMART EXTEND LOGIC ===
                        # Check if recording is ending soon (< 5s)
                        end_time = self.recorder.current.get('end_time', 0)
                        now = time.time()
                        if 0 < end_time - now < 5.0:
                            # Check for active threat
                            if cam.has_active_threat():
                                extension = settings.get('recorder.extension_seconds', 10)
                                self.recorder.extend(extension)
                                print(f"🔄 Smart Extend: Adding {extension}s to recording (Camera {source_id})")
                    
                    result = self.recorder.check_finalize()
                    if result:
                        task_pool.submit(
                            send_video,
                            settings.telegram.chat_id,
                            str(result['path']),
                            "📹 Video cảnh báo"
                        )
                else:
                    time.sleep(0.5)
            except Exception as e:
                print(f"Recorder loop error: {e}")
            
            time.sleep(0.1)
    
    # Kiểm tra hệ thống còn sống
    def _life_loop(self):
        interval = 300
        last_beat = 0
        
        while not self.shutdown_event.is_set():
            now = time.time()
            
            if now - last_beat >= interval:
                if self.bot:
                    self.bot.send_heartbeat()
                    last_beat = now
            
            time.sleep(60)
    
    def play(self):
        self.is_alarm_playing = True
        play_alarm()
    
    def stop(self):
        stop_alarm()
        self.is_alarm_playing = False
    
    def run(self):
        # Kiểm tra khởi tạo
        if not self.initialize():
            print("❌ Khởi tạo thất bại")
            self.shutdown()
            return
        
        # Khởi tạo bot
        if self.bot:
            t = threading.Thread(target=self.bot.run, daemon=True)
            t.start()
            self.threads.append(t)
        
        # Khởi tạo Kiểm tra hệ thống còn sống :v
        if self.bot:
            t = threading.Thread(target=self._life_loop, daemon=True)
            t.start()
            self.threads.append(t)
        
        # Khởi tạo ghi video
        t = threading.Thread(target=self._recorder_loop, daemon=True)
        t.start()
        self.threads.append(t)
        
        # Khởi tạo giao diện
        t = threading.Thread(
            target=run_gui,
            args=(self.camera_manager, self.face_detector, self.state, self),
            daemon=True
        )
        t.start()
        self.threads.append(t)
        
        print("✅ Hệ thống đang chạy. Nhấn Ctrl+C để stop.")
        # Xử lý nhấn Ctrl+C
        try:
            while not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
    
    # Tắt ứng dụng
    def shutdown(self):
        if self.shutdown_event.is_set():
            return
        
        print("🛑 Đang tắt ứng dụng...")
        self.shutdown_event.set()
        
        memory_monitor.stop()
        task_pool.shutdown()
        
        if hasattr(self, 'bot') and self.bot:
            self.bot.stop()
        
        if hasattr(self, 'camera_manager') and self.camera_manager:
            self.camera_manager.stop()
        
        print("✅ Đã tắt ứng dụng")


def main():
    app = GuardianApp()
    app.run()


if __name__ == "__main__":
    main()

import os
import warnings
import threading
import time
import uuid
import queue
import subprocess
import numpy as np
from typing import Optional, Dict, Any

# Suppress pkg_resources deprecation warning from dependencies
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated")

from config import settings, AlertType, AlertPriority

# Set environment variables for optimization
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('TORCH_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')

from core import CameraManager, Recorder
from core.detection.all_detectors import FaceDetector, FireDetector
from utils.common import StateManager, SpamGuard, init_alarm, play_alarm, stop_alarm, performance_monitor, performance_timer, memory_optimizer, thread_pool, task_queue
from utils.security import security_manager
from bot.service import GuardianBot, AIAssistant, send_photo, send_video_or_document
from gui import run_gui

class GuardianApp:
    """Guardian main application"""
    
    def __init__(self):
        self.state = StateManager()
        self.spam_guard = SpamGuard()
        self.recorder = Recorder()
        self.response_queue = queue.Queue()
        self.shutdown_event = threading.Event()
        
        self.is_alarm_playing = False
        self.face_detector: Optional[FaceDetector] = None
        self.fire_detector: Optional[FireDetector] = None
        self.behavior_analyzer = None
        self.camera_manager: Optional[CameraManager] = None
        
        self.bot: Optional[GuardianBot] = None
        self.ai_assistant: Optional[AIAssistant] = None
        
        self.threads = []
    
    @performance_timer("GuardianApp.initialize")
    def initialize(self) -> bool:
        """Initialize all components"""
        print("INFO: Initializing Guardian system...")
        
        # Start optimization services
        performance_monitor.start_monitoring()
        memory_optimizer.start_background_cleanup()
        task_queue.start()
        
        if not init_alarm(settings.paths.alarm_sound):
            print("WARNING: Alarm initialization failed")
        
        print("INFO: Loading face detector...")
        self.face_detector = FaceDetector()
        if not self.face_detector.initialize():
            print("ERROR: Face detector initialization failed")
            return False
        self.face_detector.load_known_faces()
        
        print("INFO: Loading fire detector...")
        self.fire_detector = FireDetector()
        if not self.fire_detector.initialize():
            print("ERROR: Fire detector initialization failed")
            return False
        
        print("INFO: Warming up AI models...")
        self._warmup_models()
        
        # Initialize behavior analyzer if enabled
        if settings.get('behavior.enabled', False):
            print("INFO: Loading anomaly detector...")
            try:
                from core.detection.anomaly import AnomalyDetector
                model_path = str(settings.get('behavior.model_path'))
                self.behavior_analyzer = AnomalyDetector(
                    model_path=model_path,
                    device=settings.get('behavior.device', 'cpu'),
                    threshold=settings.get('behavior.threshold', 0.5)
                )
                print("INFO: Anomaly detector loaded successfully")
            except Exception as e:
                print(f"WARNING: Anomaly detector initialization failed: {e}")
                self.behavior_analyzer = None
        
        try:
            self.camera_manager = CameraManager(
                show_windows=False,
                on_person_alert=self._handle_person_alert,
                on_fire_alert=self._handle_fire_alert
            )
            self.camera_manager.start_workers(
                self.fire_detector,
                self.face_detector,
                self.state,
                self.behavior_analyzer
            )
        except Exception as e:
            print(f"ERROR: Camera manager initialization failed: {e}")
            return False
        
        self.ai_assistant = AIAssistant()
        
        try:
            self.bot = GuardianBot(
                self.state,
                self.ai_assistant,
                self.spam_guard,
                self,
                self.response_queue,
                self._get_camera_snapshot,
                self.camera_manager
            )
            print("INFO: Telegram bot initialized successfully")
        except Exception as e:
            print(f"ERROR: Failed to initialize Telegram bot: {e}")
            self.bot = None

        print("INFO: Guardian system initialized successfully")
        return True
    
    def _warmup_models(self):
        """Warm up AI models for better initial performance"""
        try:
            dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            
            print("INFO: Warming up face detector...")
            if self.face_detector:
                self.face_detector.detect_faces(dummy_frame)

            print("INFO: Warming up fire detector...")
            if self.fire_detector:
                self.fire_detector.detect(dummy_frame)

            print("INFO: Model warm-up completed")
        except Exception as e:
            print(f"ERROR: Model warm-up failed: {e}")
    
    def _calculate_alert_priority(self, alert_type: str, metadata: Optional[dict] = None) -> AlertPriority:
        """Calculate alert priority based on type and context"""
        
        # Fire alerts = CRITICAL
        if alert_type in [AlertType.FIRE_CRITICAL.value, AlertType.FIRE_WARNING.value]:
            return AlertPriority.CRITICAL
        
        # Stranger + Anomalous behavior = HIGH
        if alert_type == AlertType.ANOMALOUS_BEHAVIOR.value:
            return AlertPriority.HIGH
        
        # Stranger only = MEDIUM
        if alert_type == AlertType.STRANGER.value:
            return AlertPriority.MEDIUM
        
        # Known person = LOW
        return AlertPriority.LOW
    
    def _get_priority_caption(self, alert_type: str, source_id: str, metadata: Optional[dict], priority: AlertPriority) -> str:
        """Generate caption based on priority and alert type"""
        
        if priority == AlertPriority.CRITICAL:
            if alert_type == AlertType.FIRE_CRITICAL.value:
                return f"🚨🔥 **KHẨN CẤP** - Phát hiện lửa tại camera {source_id}\n⚠️ Cần kiểm tra ngay!"
            else:
                return f"🚨🔥 Cảnh báo cháy tại camera {source_id}\n⚠️ Cần kiểm tra!"
        
        elif priority == AlertPriority.HIGH:
            score = metadata.get('score', 0.0) if metadata else 0.0
            return f"⚠️🚨 **CẢNH BÁO CAO** - Hành vi bất thường tại camera {source_id}\nĐiểm số: {score:.2f}\n⚠️ Người lạ đang có hành động đáng ngờ!"
        
        elif priority == AlertPriority.MEDIUM:
            return f"⚠️ Cảnh báo - Phát hiện người lạ tại camera {source_id}\n📹 Đang theo dõi..."
        
        else:  # LOW
            name = metadata.get('name', 'Unknown') if metadata else 'Unknown'
            return f"👋 Phát hiện {name} tại camera {source_id}"
    
    def _handle_person_alert(self, source_id: str, frame, alert_type: str, metadata: Optional[dict]):
        """Handle person detection alert"""
        print(f"ALERT: Person detected on {source_id}, type={alert_type}")
        
        if not isinstance(metadata, dict):
            metadata = {}
        
        if alert_type == AlertType.KNOWN_PERSON.value:
            name = metadata.get('name', 'Unknown')
            alert_key = (alert_type, name, source_id)
        else:
            alert_key = (alert_type, source_id)
        
        if not self.spam_guard.allow(alert_key):
            return

        if hasattr(self.state, 'has_unresolved_alert') and self.state.has_unresolved_alert(alert_key):
            return
        
        img_path = settings.paths.tmp_dir / f"alert_{alert_type}_{uuid.uuid4().hex}.jpg"
        try:
            # Save encrypted image
            security_manager.save_encrypted_image(img_path, frame)
        except Exception as e:
            print(f"ERROR: Failed to save alert image: {e}")
            return
        
        alert_id = self.state.create_alert(
            alert_type=alert_type,
            chat_id=settings.telegram.chat_id,
            asked_for=metadata.get('name'),
            image_path=str(img_path),
            source_id=source_id
        )
        
        # Calculate priority and generate caption
        priority = self._calculate_alert_priority(alert_type, metadata)
        caption = self._get_priority_caption(alert_type, source_id, metadata, priority)
        
        # Determine if notification should be silent (LOW priority)
        disable_notification = (priority == AlertPriority.LOW)
        
        if self.bot:
            self.bot.schedule_alert(
                settings.telegram.chat_id,
                str(img_path),
                caption,
                alert_id,
                disable_notification=disable_notification
            )
        
        if self.ai_assistant:
            self.ai_assistant.add_system_message(settings.telegram.chat_id, caption)
        
        self._start_recording(source_id, alert_id, settings.recorder.duration_seconds, False)
        
        if alert_type != AlertType.FIRE_CRITICAL.value:
            self._start_response_watcher(alert_id)
    
    def _handle_fire_alert(self, source_id: str, frame, alert_type: str):
        """Handle fire detection alert"""
        alert_key = (alert_type, source_id)
        
        is_critical = (alert_type == AlertType.FIRE_CRITICAL.value)
        
        if not self.spam_guard.allow(alert_key, is_critical=is_critical):
            return
        
        if hasattr(self.state, 'has_unresolved_alert') and self.state.has_unresolved_alert(alert_key):
            return
        
        img_path = settings.paths.tmp_dir / f"alert_{alert_type}_{uuid.uuid4().hex}.jpg"
        # Save encrypted image
        security_manager.save_encrypted_image(img_path, frame)
        
        alert_id = self.state.create_alert(
            alert_type=alert_type,
            chat_id=settings.telegram.chat_id,
            image_path=str(img_path),
            source_id=source_id
        )
        
        if alert_type == AlertType.FIRE_CRITICAL.value:
            caption = f"🔴 NGUY HIỂM: Phát hiện cháy tại camera {source_id}. Cần hành động ngay!"
        else:
            caption = f"🟡 CẢNH BÁO: Nghi ngờ có cháy tại camera {source_id}. Vui lòng kiểm tra."
        
        if self.bot:
            self.bot.schedule_alert(
                settings.telegram.chat_id,
                str(img_path),
                caption,
                alert_id
            )
        
        if self.ai_assistant:
            self.ai_assistant.add_system_message(settings.telegram.chat_id, caption)
        
        self._start_recording(source_id, alert_id, settings.recorder.duration_seconds, False)

        if alert_type == AlertType.FIRE_CRITICAL.value:
            threading.Thread(
                target=self._watch_fire_alert,
                args=(alert_id,),
                daemon=True
            ).start()
    
    def _watch_for_response(self, alert_id: str):
        """Watch for user response to alert"""
        start = time.time()
        
        while time.time() - start < settings.telegram.user_response_window_seconds:
            try:
                resp = self.response_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            if resp and resp.get("alert_id") == alert_id:
                decision = resp.get("decision")
                
                if decision in ("yes", "left"):
                    self.recorder.stop_and_discard()
                
                return
    
    def _watch_fire_alert(self, alert_id: str):
        """Monitor fire alert and trigger alarm if no response"""
        time.sleep(settings.telegram.user_response_window_seconds)
        
        alert_info = self.state.get_alert_by_id(alert_id)
        if alert_info and not alert_info.resolved:
            play_alarm()
    
    def _compress_video(self, video_path: str) -> Optional[str]:
        """Compress video to reduce file size"""
        try:
            base_path, _ = os.path.splitext(video_path)
            compressed_path = f"{base_path}_compressed.mp4"
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-c:v', 'libx264',
                '-crf', '28',
                '-preset', 'fast',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                '-y',
                compressed_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(compressed_path):
                original_size = os.path.getsize(video_path)
                compressed_size = os.path.getsize(compressed_path)
                
                if compressed_size < original_size * 0.8:
                    return compressed_path
                else:
                    os.remove(compressed_path)
                    return None
            else:
                return None
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return None
    
    def _start_recording(self, source_id: str, alert_id: str, duration: int, wait_for_user: bool):
        """Start recording video"""
        try:
            rec = self.recorder.start(
                source_id=source_id,
                reason="alert",
                duration=duration,
                wait_for_user=wait_for_user
            )
            if rec:
                rec.setdefault("alert_ids", []).append(alert_id)
        except Exception as e:
            print(f"ERROR: Failed to start recording: {e}")
    
    def _recorder_monitor_loop(self):
        """Monitor recorder and finalize recordings"""
        frame_count = 0
        while not self.shutdown_event.is_set():
            try:
                if self.recorder.current and self.camera_manager:
                    source_id = self.recorder.current.get("source_id")
                    if not source_id:
                        time.sleep(0.1)
                        continue

                    cam = self.camera_manager.get_camera(source_id)
                    if not cam:
                        time.sleep(0.1)
                        continue

                    ret, frame = cam.read_raw()
                    if ret and frame is not None:
                        frame_count += 1
                        # Optimized: skip every other frame for better performance
                        if frame_count % 2 == 0:
                            # Use thread pool for async writing
                            task_queue.submit(self.recorder.write, frame)
                        
                        finalized = self.recorder.check_and_finalize()
                        
                        if finalized:
                            path = finalized.get("path")
                            # Send video async using thread pool
                            task_queue.submit(
                                self._send_video_async,
                                str(path),
                                "📹 Video cảnh báo"
                            )
                else:
                    time.sleep(0.5)

            except Exception as e:
                print(f"ERROR: Recorder monitor error: {e}")
                time.sleep(1)
            
            time.sleep(0.1)
    
    def _get_camera_snapshot(self, chat_id: str, source: Optional[str] = None):
        """Get and send camera snapshot"""
        try:
            if not self.camera_manager or not self.camera_manager.cameras:
                return

            cam_to_use = None
            cam_source_id_for_caption = "default"

            if source:
                if source.isdigit():
                    idx = int(source)
                    camera_sources_list = list(self.camera_manager.cameras.keys())
                    if 0 <= idx < len(camera_sources_list):
                        cam_source_id = camera_sources_list[idx]
                        cam_to_use = self.camera_manager.get_camera(cam_source_id)
                        cam_source_id_for_caption = f"#{idx}"
                
                if not cam_to_use:
                    cam_to_use = self.camera_manager.get_camera(source)
                    cam_source_id_for_caption = source
            else:
                first_cam_source = next(iter(self.camera_manager.cameras))
                cam_to_use = self.camera_manager.get_camera(first_cam_source)
                cam_source_id_for_caption = f"#0"

            if not cam_to_use:
                return

            ret, frame = cam_to_use.read_raw()
            if not ret or frame is None:
                return
            
            img_path = settings.paths.tmp_dir / f"snapshot_{uuid.uuid4().hex}.jpg"
            # Save encrypted image
            security_manager.save_encrypted_image(img_path, frame)
            
            threading.Thread(
                target=lambda: send_photo(
                    settings.telegram.token,
                    chat_id,
                    str(img_path),
                    f"📸 Ảnh chụp từ camera {cam_source_id_for_caption}"
                ),
                daemon=True
            ).start()
        except Exception as e:
            print(f"ERROR: Snapshot error: {e}")
    
    # Alarm control methods
    def play(self):
        """Play alarm"""
        self.is_alarm_playing = True
        play_alarm()
    
    def stop(self):
        """Stop alarm"""
        stop_alarm()
        self.is_alarm_playing = False
    
    def run(self):
        """Run application"""
        if not self.initialize():
            print("ERROR: Initialization failed")
            self.shutdown()
            return
        
        if self.bot:
            bot_thread = threading.Thread(target=self.bot.run, daemon=True)
            bot_thread.start()
            self.threads.append(bot_thread)

        if self.bot:
            heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            heartbeat_thread.start()
            self.threads.append(heartbeat_thread)

        gui_thread = threading.Thread(
            target=run_gui,
            args=(self.camera_manager, self.face_detector, self.state),
            daemon=True
        )
        gui_thread.start()
        self.threads.append(gui_thread)
        
        recorder_thread = threading.Thread(
            target=self._recorder_monitor_loop,
            daemon=True
        )
        recorder_thread.start()
        self.threads.append(recorder_thread)
        
        try:
            while not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            print("INFO: Interrupted by user")
        finally:
            self.shutdown()
    
    def _send_alert_async(self, img_path: str, caption: str):
        """Send alert asynchronously using thread pool"""
        task_queue.submit(
            send_photo,
            settings.telegram.token,
            settings.telegram.chat_id,
            img_path,
            caption
        )
    
    def _start_response_watcher(self, alert_id: str):
        """Start response watcher thread"""
        threading.Thread(
            target=self._watch_for_response,
            args=(alert_id,),
            daemon=True
        ).start()
    
    def _send_video_async(self, video_path: str, caption: str):
        """Send video asynchronously"""
        def _send():
            path_to_send = video_path
            compressed_path = self._compress_video(video_path)
            
            if compressed_path:
                path_to_send = compressed_path
            
            try:
                send_video_or_document(
                    settings.telegram.token,
                    settings.telegram.chat_id,
                    path_to_send,
                    caption
                )
            finally:
                if compressed_path and compressed_path != video_path:
                    try:
                        os.remove(compressed_path)
                    except Exception:
                        pass
        
        threading.Thread(target=_send, daemon=True).start()
    
    def _heartbeat_loop(self):
        """Send periodic heartbeat messages"""
        heartbeat_interval = 300
        last_heartbeat = 0
        last_camera_check = 0
        camera_check_interval = 60
        
        while not self.shutdown_event.is_set():
            try:
                current_time = time.time()
                
                if current_time - last_camera_check >= camera_check_interval:
                    self._check_camera_health()
                    last_camera_check = current_time
                
                if current_time - last_heartbeat >= heartbeat_interval:
                    if self.bot:
                        try:
                            self.bot.send_heartbeat()
                            last_heartbeat = current_time
                        except Exception as e:
                            print(f"ERROR: Heartbeat error: {e}")

                time.sleep(60)
            except Exception as e:
                print(f"ERROR: Heartbeat loop error: {e}")
                time.sleep(60)
    
    def _check_camera_health(self):
        """Check camera health and reconnect if needed"""
        if not self.camera_manager:
            return
        
        statuses = self.camera_manager.get_all_connection_statuses()
        for source, status in statuses.items():
            try:
                if not status:
                    cam = self.camera_manager.get_camera(source)
                    if cam and cam.force_reconnect():
                        print(f"INFO: Camera {source} reconnected")
                    else:
                        print(f"ERROR: Failed to reconnect camera {source}")
            except Exception as e:
                print(f"ERROR: Camera health check error for {source}: {e}")

    def shutdown(self):
        """Shutdown application"""
        if self.shutdown_event.is_set():
            return
        print("INFO: Shutting down Guardian...")
        self.shutdown_event.set()
        
        # Stop optimization services
        performance_monitor.stop_monitoring()
        memory_optimizer.stop_background_cleanup()
        task_queue.stop()
        thread_pool.shutdown()
        
        if self.bot:
            self.bot.stop()
        
        if self.camera_manager:
            self.camera_manager.stop_all()
        print("INFO: Shutdown complete")

def main():
    """Main entry point"""
    app = GuardianApp()
    app.run()

if __name__ == "__main__":
    main()

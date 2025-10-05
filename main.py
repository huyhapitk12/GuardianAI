"""
Guardian Security System - Main Application
Refactored and cleaned version
"""
import logging
import threading
import time
import uuid
import queue
import cv2

# Configuration
from config import settings
from config import (
    RECORD_SECONDS,
    USER_RESPONSE_WINDOW_SECONDS,
    STRANGER_CLIP_DURATION,
    AlertType
)

# Core modules
from core import Camera
from core import Recorder
from core import FaceDetector, FireDetector, PersonTracker

# Utilities
from utils import StateManager, SpamGuard, init_alarm, play_alarm, stop_alarm

# Telegram
from bot import GuardianBot, AIAssistant

# GUI
from gui import run_gui

# Telegram helpers
from bot import send_photo, send_video_or_document

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

class GuardianApp:
    """Main Guardian application"""
    
    def __init__(self):
        # Initialize components
        self.state = StateManager()
        self.spam_guard = SpamGuard()
        self.recorder = Recorder()
        self.response_queue = queue.Queue()
        
        # Detection components
        self.face_detector = None
        self.fire_detector = None
        self.person_tracker = None
        self.camera = None
        
        # Telegram
        self.bot = None
        self.ai_assistant = None
        
        # Threads
        self.threads = []
    
    def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("Initializing Guardian system...")
        
        # Initialize alarm
        if not init_alarm(settings.paths.alarm_sound):
            logger.warning("Alarm initialization failed")
        
        # Initialize detectors
        self.face_detector = FaceDetector()
        if not self.face_detector.initialize():
            logger.error("Face detector initialization failed")
            return False
        self.face_detector.load_known_faces()
        
        self.fire_detector = FireDetector()
        if not self.fire_detector.initialize():
            logger.error("Fire detector initialization failed")
            return False
        
        self.person_tracker = PersonTracker(self.face_detector)
        if not self.person_tracker.initialize():
            logger.error("Person tracker initialization failed")
            return False
        
        # Initialize camera
        try:
            self.camera = Camera(
                source=settings.camera.source,
                show_window=False,
                on_person_alert=self._handle_person_alert,
                on_fire_alert=self._handle_fire_alert
            )
            self.camera.start_workers(
                self.fire_detector,
                self.person_tracker,
                self.face_detector
            )
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
        
        # Initialize AI assistant
        self.ai_assistant = AIAssistant()
        
        # Initialize Telegram bot
        self.bot = GuardianBot(
            self.state,
            self.ai_assistant,
            self.spam_guard,
            self,  # For alarm control
            self.response_queue,
            self._get_camera_snapshot
        )
        
        logger.info("Guardian system initialized successfully")
        return True
    
    def _handle_person_alert(self, frame, alert_type: str, metadata: dict):
        """Handle person detection alerts"""
        # Build alert key
        if alert_type == AlertType.KNOWN_PERSON.value:
            alert_key = (alert_type, metadata.get('name'))
        else:
            alert_key = alert_type
        
        # Check spam guard
        if not self.spam_guard.allow(alert_key):
            logger.info(f"Alert blocked by spam guard: {alert_key}")
            return
        
        # Check for unresolved alerts
        if self.state.has_unresolved_alert(alert_key):
            logger.info(f"Skipping alert, already have unresolved: {alert_key}")
            return
        
        # Save alert image
        img_path = settings.paths.tmp_dir / f"alert_{alert_type}_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(img_path), frame)
        
        # Create alert
        alert_id = self.state.create_alert(
            alert_type=alert_type,
            chat_id=settings.telegram.chat_id,
            asked_for=metadata.get('name'),
            image_path=str(img_path)
        )
        
        # Prepare caption
        if alert_type == AlertType.STRANGER.value:
            caption = (
                f"⚠️ Phát hiện người lạ\n\n"
                f"Bạn có nhận ra người này không? "
                f"(Trả lời trong {USER_RESPONSE_WINDOW_SECONDS}s: có/không)"
            )
        else:  # Known person
            name = metadata.get('name', 'Unknown')
            caption = (
                f"👋 Phát hiện {name}\n\n"
                f"Bạn có nhận ra người này không? "
                f"(Trả lời trong {USER_RESPONSE_WINDOW_SECONDS}s: có/không)"
            )
        
        # Send alert
        threading.Thread(
            target=lambda: send_photo(
                settings.telegram.token,
                settings.telegram.chat_id,
                str(img_path),
                caption
            ),
            daemon=True
        ).start()
        
        self.ai_assistant.add_system_message(settings.telegram.chat_id, caption)
        
        # Start short clip for stranger
        if alert_type == AlertType.STRANGER.value:
            self._start_alert_clip(frame, alert_id, STRANGER_CLIP_DURATION)
        
        # Start recording
        wait_for_user = (alert_type == AlertType.KNOWN_PERSON.value)
        self._start_recording(alert_id, RECORD_SECONDS, wait_for_user)
        
        # Start response watcher
        if alert_type != AlertType.FIRE_CRITICAL.value:
            threading.Thread(
                target=self._watch_for_response,
                args=(alert_id,),
                daemon=True
            ).start()
    
    def _handle_fire_alert(self, frame, alert_type: str):
        """Handle fire detection alerts"""
        alert_key = "lua_chay"
        
        if not self.spam_guard.allow(alert_key):
            logger.info("Fire alert blocked by spam guard")
            return
        
        if self.state.has_unresolved_alert(alert_key):
            logger.info("Skipping fire alert, already have unresolved")
            return
        
        # Save alert image
        img_path = settings.paths.tmp_dir / f"alert_{alert_type}_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(img_path), frame)
        
        # Create alert
        alert_id = self.state.create_alert(
            alert_type=alert_type,
            chat_id=settings.telegram.chat_id,
            image_path=str(img_path)
        )
        
        # Prepare caption
        if alert_type == AlertType.FIRE_CRITICAL.value:
            caption = (
                "🔴 CẢNH BÁO ĐỎ KHẨN CẤP: Phát hiện đám cháy đang phát triển "
                "hoặc có cả lửa và khói. Yêu cầu kiểm tra ngay lập tức!"
            )
        else:  # Warning
            caption = (
                "🟡 CẢNH BÁO VÀNG: Phát hiện dấu hiệu nghi ngờ cháy. "
                "Vui lòng kiểm tra hình ảnh và xác nhận."
            )
        
        # Send with buttons
        self.bot.schedule_alert(
            settings.telegram.chat_id,
            str(img_path),
            caption,
            alert_id
        )
        
        self.ai_assistant.add_system_message(settings.telegram.chat_id, caption)
        
        # Start fire alert watcher for critical alerts
        if alert_type == AlertType.FIRE_CRITICAL.value:
            threading.Thread(
                target=self._watch_fire_alert,
                args=(alert_id,),
                daemon=True
            ).start()
    
    def _watch_for_response(self, alert_id: str):
        """Watch for user response to alert"""
        start = time.time()
        
        while time.time() - start < USER_RESPONSE_WINDOW_SECONDS:
            try:
                resp = self.response_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            if resp and resp.get("alert_id") == alert_id:
                decision = resp.get("decision")
                
                self.recorder.resolve_user_wait()
                
                if decision in ("yes", "left"):
                    logger.info("Safe response - stopping and discarding recording")
                    self.recorder.stop_and_discard()
                else:
                    logger.info("Unsafe/unclear response - continue recording")
                
                return
        
        logger.info(f"No response for alert {alert_id}, resolving user wait")
        self.recorder.resolve_user_wait()
    
    def _watch_fire_alert(self, alert_id: str):
        """Watch fire alert and activate alarm if no response"""
        time.sleep(USER_RESPONSE_WINDOW_SECONDS)
        
        alert_info = self.state.get_alert_by_id(alert_id)
        if alert_info and not alert_info.resolved:
            logger.warning(f"No response to fire alert {alert_id}, ACTIVATING ALARM!")
            play_alarm()
    
    def _start_alert_clip(self, initial_frame, alert_id: str, duration: int):
        """Start recording a short clip for alert"""
        def worker():
            path = settings.paths.tmp_dir / f"clip_{alert_id[:8]}_{uuid.uuid4().hex[:8]}.mp4"
            
            try:
                h, w = initial_frame.shape[:2]
                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(path), fourcc, 20.0, (w, h))
                
                if not writer.isOpened():
                    logger.error("Failed to create clip writer")
                    return
                
                writer.write(initial_frame)
                
                start = time.time()
                while time.time() - start < duration:
                    ret, frame = self.camera.read_raw()
                    if ret and frame is not None:
                        writer.write(frame)
                    time.sleep(0.02)
                
                writer.release()
                
                threading.Thread(
                    target=lambda: send_video_or_document(
                        settings.telegram.token,
                        settings.telegram.chat_id,
                        str(path),
                        "📹 Clip cảnh báo"
                    ),
                    daemon=True
                ).start()
            except Exception as e:
                logger.error(f"Clip recording error: {e}")
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _start_recording(self, alert_id: str, duration: int, wait_for_user: bool):
        """Start video recording"""
        try:
            rec = self.recorder.start(
                reason="alert",
                duration=duration,
                wait_for_user=wait_for_user
            )
            if rec:
                rec.setdefault("alert_ids", []).append(alert_id)
                logger.info(f"Recording started for alert {alert_id}")
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
    
    def _recorder_monitor_loop(self):
        """Monitor recorder and finalize recordings"""
        while not self.camera.quit:
            ret, frame = self.camera.read_raw()
            if ret and frame is not None:
                try:
                    if self.recorder.current:
                        self.recorder.write(frame)
                        finalized = self.recorder.check_and_finalize()
                        
                        if finalized:
                            path = finalized.get("path")
                            logger.info(f"Recording finalized: {path}")
                            
                            threading.Thread(
                                target=lambda: send_video_or_document(
                                    settings.telegram.token,
                                    settings.telegram.chat_id,
                                    str(path),
                                    "📹 Bản ghi cảnh báo"
                                ),
                                daemon=True
                            ).start()
                except Exception as e:
                    logger.error(f"Recorder monitor error: {e}")
            
            time.sleep(0.02)
    
    def _get_camera_snapshot(self, chat_id: str):
        """Get and send camera snapshot"""
        try:
            ret, frame = self.camera.read_raw()
            if not ret or frame is None:
                logger.error("Failed to get camera snapshot")
                return
            
            img_path = settings.paths.tmp_dir / f"snapshot_{uuid.uuid4().hex}.jpg"
            cv2.imwrite(str(img_path), frame)
            
            threading.Thread(
                target=lambda: send_photo(
                    settings.telegram.token,
                    chat_id,
                    str(img_path),
                    "📸 Ảnh chụp nhanh từ camera"
                ),
                daemon=True
            ).start()
        except Exception as e:
            logger.error(f"Snapshot error: {e}")
    
    # Alarm control methods (for bot)
    def play(self):
        """Play alarm"""
        play_alarm()
    
    def stop(self):
        """Stop alarm"""
        stop_alarm()
    
    def run(self):
        """Run the application"""
        if not self.initialize():
            logger.error("Initialization failed")
            return
        
        # Start Telegram bot thread
        bot_thread = threading.Thread(target=self.bot.run, daemon=True)
        bot_thread.start()
        self.threads.append(bot_thread)
        logger.info("Telegram bot thread started")
        
        # Start GUI thread
        gui_thread = threading.Thread(
            target=run_gui,
            args=(self.camera, self.face_detector, self.state),
            daemon=True
        )
        gui_thread.start()
        self.threads.append(gui_thread)
        logger.info("GUI thread started")
        
        # Start recorder monitor thread
        recorder_thread = threading.Thread(
            target=self._recorder_monitor_loop,
            daemon=True
        )
        recorder_thread.start()
        self.threads.append(recorder_thread)
        logger.info("Recorder monitor thread started")
        
        # Run camera processing (blocks)
        try:
            self.camera.process_frames(self.state)
        except KeyboardInterrupt:
            logger.info("Interrupted by user, shutting down...")
        except Exception as e:
            logger.error(f"Camera processing error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the application"""
        logger.info("Shutting down Guardian...")
        if self.camera:
            self.camera.release()
        logger.info("Shutdown complete")

def main():
    """Main entry point"""
    app = GuardianApp()
    app.run()

if __name__ == "__main__":
    main()
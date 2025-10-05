"""Camera capture and processing"""
import cv2
import logging
import time
import threading
import queue
from typing import Optional, Callable, Tuple
from collections import deque
import numpy as np

from config.settings import settings
from config.constants import (
    PROCESS_SIZE,
    PROCESS_EVERY_N_FRAMES,
    TARGET_FPS,
    FIRE_WINDOW_SECONDS
)

logger = logging.getLogger(__name__)

class Camera:
    """Handles camera capture and frame processing"""
    
    def __init__(
        self,
        source=None,
        show_window: bool = False,
        on_person_alert: Optional[Callable] = None,
        on_fire_alert: Optional[Callable] = None
    ):
        self.source = source or settings.camera.source
        self.show_window = show_window
        self.on_person_alert = on_person_alert
        self.on_fire_alert = on_fire_alert
        
        # Initialize capture
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.source)
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {self.source}")
        
        # State
        self.quit = False
        self._frame_lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._raw_frame: Optional[np.ndarray] = None
        self._frame_idx = 0
        
        # Processing queues
        self.fire_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=16)
        
        # Fire detection state
        self.recent_fire_detections = deque(
            maxlen=int(TARGET_FPS * FIRE_WINDOW_SECONDS)
        )
        self.current_fire_boxes = []
        self.red_alert_mode_active = False
        self.red_alert_mode_until = 0
        
        logger.info(f"Camera initialized: {self.source}")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the latest processed frame"""
        with self._frame_lock:
            if self._last_frame is not None:
                return True, self._last_frame.copy()
            return False, None
    
    def read_raw(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the latest raw frame"""
        with self._frame_lock:
            if self._raw_frame is not None:
                return True, self._raw_frame.copy()
            return False, None
    
    def start_workers(self, fire_detector, person_tracker, face_detector):
        """Start background worker threads"""
        self.fire_detector = fire_detector
        self.person_tracker = person_tracker
        self.face_detector = face_detector
        
        threading.Thread(
            target=self._fire_worker,
            daemon=True
        ).start()
        logger.info("Fire detection worker started")
    
    def _fire_worker(self):
        """Background worker for fire detection"""
        while not self.quit:
            try:
                frame = self.fire_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            if self.fire_detector is None:
                continue
            
            try:
                detections = self.fire_detector.detect(frame)
                if detections:
                    self.result_queue.put(("fire", detections))
            except Exception as e:
                logger.error(f"Fire worker error: {e}")
    
    def process_frames(self, state_manager):
        """Main frame processing loop"""
        frame_interval = 1.0 / TARGET_FPS
        last_time = 0
        
        while not self.quit:
            now = time.time()
            
            # Frame rate limiting
            if now - last_time < frame_interval:
                time.sleep(0.005)
                continue
            last_time = now
            
            # Skip buffered frames
            for _ in range(2):
                self.cap.grab()
            
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            self._frame_idx += 1
            display_frame = frame.copy()
            orig_h, orig_w = frame.shape[:2]
            
            # Check if person detection is enabled
            person_detection_enabled = state_manager.is_person_detection_enabled()
            
            # Prepare small frame for processing
            small_frame = cv2.resize(
                frame,
                PROCESS_SIZE,
                interpolation=cv2.INTER_AREA
            )
            scale_x = orig_w / float(PROCESS_SIZE[0])
            scale_y = orig_h / float(PROCESS_SIZE[1])
            
            # Person detection and tracking
            if (self._frame_idx % PROCESS_EVERY_N_FRAMES == 0 and 
                person_detection_enabled):
                self._process_persons(small_frame, frame, scale_x, scale_y, now)
            
            # Send frame for fire detection
            if not self.fire_queue.full():
                try:
                    self.fire_queue.put_nowait(small_frame)
                except queue.Full:
                    pass
            
            # Process fire detection results
            self.current_fire_boxes = []
            self._process_fire_results(scale_x, scale_y, now, frame)
            
            # Draw visualizations
            self._draw_visualizations(display_frame, person_detection_enabled)
            
            # Update stored frames
            self._update_frames(display_frame, frame)
            
            # Show window if enabled
            if self.show_window:
                cv2.imshow("Guardian Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.quit = True
        
        self.release()
    
    def _process_persons(
        self,
        small_frame: np.ndarray,
        full_frame: np.ndarray,
        scale_x: float,
        scale_y: float,
        now: float
    ):
        """Process person detection and tracking"""
        try:
            detections = self.person_tracker.detect_persons(small_frame)
            self.person_tracker.update_tracks(
                detections,
                full_frame,
                scale_x,
                scale_y
            )
            
            # Check for alerts
            alerts = self.person_tracker.check_confirmations()
            for track_id, alert_type, metadata in alerts:
                if self.on_person_alert:
                    self.on_person_alert(full_frame, alert_type, metadata)
        except Exception as e:
            logger.error(f"Person processing error: {e}")
    
    def _process_fire_results(
        self,
        scale_x: float,
        scale_y: float,
        now: float,
        frame: np.ndarray
    ):
        """Process fire detection results from queue"""
        try:
            while not self.result_queue.empty():
                result_type, results = self.result_queue.get_nowait()
                
                if result_type == "fire":
                    self._handle_fire_detections(
                        results,
                        scale_x,
                        scale_y,
                        now,
                        frame
                    )
        except queue.Empty:
            pass
    
    def _handle_fire_detections(
        self,
        detections: list,
        scale_x: float,
        scale_y: float,
        now: float,
        frame: np.ndarray
    ):
        """Handle fire detections and determine alert level"""
        from config.constants import (
            FIRE_YELLOW_ALERT_FRAMES,
            FIRE_RED_ALERT_GROWTH_THRESHOLD,
            FIRE_RED_ALERT_GROWTH_WINDOW,
            FIRE_RED_ALERT_LOCKDOWN_SECONDS
        )
        
        # Update current fire boxes for display
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            x1_orig = int(x1 * scale_x)
            y1_orig = int(y1 * scale_y)
            x2_orig = int(x2 * scale_x)
            y2_orig = int(y2 * scale_y)
            self.current_fire_boxes.append((x1_orig, y1_orig, x2_orig, y2_orig))
        
        # Add to history
        for det in detections:
            det['timestamp'] = now
            self.recent_fire_detections.append(det)
        
        # Check if red alert lockdown expired
        if self.red_alert_mode_active and now > self.red_alert_mode_until:
            self.red_alert_mode_active = False
            logger.info("Red alert lockdown expired")
        
        if not self.recent_fire_detections:
            return
        
        # Determine alert level
        is_red_alert = False
        
        # Check if in red alert lockdown
        if self.red_alert_mode_active and detections:
            is_red_alert = True
        else:
            # Check for fire growth
            fire_dets = [d for d in self.recent_fire_detections if d['class'] == 'fire']
            if len(fire_dets) > 2:
                current_fires = [d for d in fire_dets if now - d['timestamp'] < 0.5]
                past_fires = [
                    d for d in fire_dets 
                    if FIRE_RED_ALERT_GROWTH_WINDOW - 0.5 < now - d['timestamp'] < FIRE_RED_ALERT_GROWTH_WINDOW
                ]
                
                if current_fires and past_fires:
                    avg_current = sum(d['area'] for d in current_fires) / len(current_fires)
                    avg_past = sum(d['area'] for d in past_fires) / len(past_fires)
                    
                    if avg_current > avg_past * FIRE_RED_ALERT_GROWTH_THRESHOLD:
                        is_red_alert = True
            
            # Check for fire + smoke
            if not is_red_alert:
                classes = {d['class'] for d in self.recent_fire_detections}
                if 'fire' in classes and 'smoke' in classes:
                    is_red_alert = True
        
        # Handle red alert
        if is_red_alert:
            if not self.red_alert_mode_active:
                self.red_alert_mode_active = True
                self.red_alert_mode_until = now + FIRE_RED_ALERT_LOCKDOWN_SECONDS
                logger.warning("RED ALERT: Fire growing or fire+smoke detected")
            
            if self.on_fire_alert:
                alert_frame = frame.copy()
                for d in self.recent_fire_detections:
                    x1, y1, x2, y2 = d['bbox']
                    x1_orig = int(x1 * scale_x)
                    y1_orig = int(y1 * scale_y)
                    x2_orig = int(x2 * scale_x)
                    y2_orig = int(y2 * scale_y)
                    cv2.rectangle(
                        alert_frame,
                        (x1_orig, y1_orig),
                        (x2_orig, y2_orig),
                        (0, 0, 255),
                        3
                    )
                self.on_fire_alert(alert_frame, "lua_chay_khan_cap")
                self.recent_fire_detections.clear()
            return
        
        # Handle yellow alert
        if len(self.recent_fire_detections) >= FIRE_YELLOW_ALERT_FRAMES:
            if self.on_fire_alert:
                alert_frame = frame.copy()
                for d in self.recent_fire_detections:
                    x1, y1, x2, y2 = d['bbox']
                    x1_orig = int(x1 * scale_x)
                    y1_orig = int(y1 * scale_y)
                    x2_orig = int(x2 * scale_x)
                    y2_orig = int(y2 * scale_y)
                    cv2.rectangle(
                        alert_frame,
                        (x1_orig, y1_orig),
                        (x2_orig, y2_orig),
                        (0, 255, 255),
                        2
                    )
                self.on_fire_alert(alert_frame, "lua_chay_nghi_ngo")
                self.recent_fire_detections.clear()
    
    def _draw_visualizations(self, frame: np.ndarray, show_persons: bool):
        """Draw bounding boxes and labels on frame"""
        # Draw tracked persons
        if show_persons and self.person_tracker:
            for track_data in self.person_tracker.tracked_objects.values():
                x1, y1, x2, y2 = track_data['bbox']
                name = track_data.get('confirmed_name') or track_data.get('name') or "Nguoi la"
                dist = track_data.get('distance', 0.0)
                
                color = (0, 255, 0) if name != "Nguoi la" else (0, 0, 255)
                label = f"{name} ({dist:.2f})"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
        
        # Draw fire boxes
        for x1, y1, x2, y2 in self.current_fire_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
    
    def _update_frames(self, display_frame: np.ndarray, raw_frame: np.ndarray):
        """Update stored frames thread-safely"""
        with self._frame_lock:
            self._last_frame = display_frame.copy()
            self._raw_frame = raw_frame.copy()
    
    def release(self):
        """Release camera resources"""
        logger.info("Releasing camera")
        self.quit = True
        if self.cap:
            self.cap.release()
        if self.show_window:
            cv2.destroyAllWindows()
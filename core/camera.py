"""Camera handling with detection integration"""

from __future__ import annotations
import cv2
import time
import queue
import platform
import threading
import numpy as np
from typing import Callable, Dict, Optional, Tuple
from collections import deque

from config import settings, AlertType
from core.detection import PersonTracker, FireFilter, BehaviorAnalyzer, FireTracker
from core.motion_detector import MotionDetector


class Camera:
    """Single camera handler with detection"""
    
    def __init__(self, source, on_person_alert: Callable = None, on_fire_alert: Callable = None, shared_model=None):
        self.source = source
        self.source_id = str(source)
        self.cap = None
        self.quit = False
        
        self._frame_lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._raw_frame: Optional[np.ndarray] = None
        self._frame_idx = 0
        
        self._reconnect_attempts = 0
        self._last_frame_time = time.time()
        self._ai_active_until = 0
        
        self._is_ir = False
        self._ir_history = deque(maxlen=30)
        
        # Use get() with default False since we cleaned up config
        debug_fire = settings.get('camera.debug_fire_detection', False)
        self._fire_filter = FireFilter(debug=debug_fire)
        self._fire_boxes = []
        self._fire_history = deque(maxlen=150)
        self._fire_tracker = FireTracker()
        
        # Sử dụng shared model nếu có
        self.person_tracker = PersonTracker(shared_model=shared_model)
        self.behavior_analyzer: Optional[BehaviorAnalyzer] = None
        
        # Motion detector để giảm tải
        self._motion_detector = MotionDetector(
            motion_threshold=settings.get('camera.motion_threshold', 25.0),
            min_area=settings.get('camera.motion_min_area', 500)
        )
        
        self.on_person_alert = on_person_alert
        self.on_fire_alert = on_fire_alert
        
        self.fire_queue = queue.Queue(maxsize=2)
        self.behavior_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=16)
        
        self._last_detection_enabled = False
        
        self._init_capture()
    
    def _init_capture(self):
        """Initialize video capture"""
        try:
            if isinstance(self.source, int):
                backends = self._get_backends()
                for backend in backends:
                    self.cap = cv2.VideoCapture(self.source, backend)
                    if self.cap.isOpened():
                        break
            else:
                self.cap = cv2.VideoCapture(self.source)
            
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"✅ Camera {self.source_id} connected")
        except Exception as e:
            print(f"❌ Camera {self.source_id} init failed: {e}")
    
    def _get_backends(self) -> list:
        """Get platform-specific backends"""
        if platform.system() == 'Windows':
            return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
        elif platform.system() == 'Linux':
            return [cv2.CAP_V4L2, cv2.CAP_ANY]
        return [cv2.CAP_ANY]
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read processed frame"""
        with self._frame_lock:
            if self._last_frame is not None:
                return True, self._last_frame.copy()
            return False, None
    
    def read_raw(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read raw frame"""
        with self._frame_lock:
            if self._raw_frame is not None:
                return True, self._raw_frame.copy()
            return False, None
    
    def start_workers(self, fire_detector, face_detector, behavior_analyzer=None):
        """Start detection workers"""
        self.person_tracker.set_face_detector(face_detector)
        self.person_tracker.initialize()
        
        self.behavior_analyzer = behavior_analyzer
        
        # Fire worker
        threading.Thread(
            target=self._fire_worker,
            args=(fire_detector,),
            daemon=True
        ).start()
        
        # Behavior worker
        if self.behavior_analyzer:
            threading.Thread(
                target=self._behavior_worker,
                daemon=True
            ).start()
            print(f"✅ Behavior worker started for camera {self.source_id}")
    
    def _fire_worker(self, detector):
        """Background fire detection"""
        while not self.quit:
            try:
                frame = self.fire_queue.get(timeout=1.0)
                detections = detector.detect(frame)
                if detections:
                    self.result_queue.put(('fire', detections))
            except queue.Empty:
                continue
    
    def _behavior_worker(self):
        """Background behavior analysis"""
        skip_counter = 0
        skip_n = settings.get('behavior.process_every_n_frames', 3)
        
        while not self.quit:
            try:
                frame = self.behavior_queue.get(timeout=1.0)
                
                # Skip frames for performance
                skip_counter += 1
                if skip_counter % skip_n != 0:
                    continue
                
                result = self.behavior_analyzer.process_frame(frame)
                
                # Check for alert
                if result.is_anomaly and self.behavior_analyzer.should_alert():
                    self.result_queue.put(('behavior', result, frame.copy()))
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Behavior worker error: {e}")
    
    def process_loop(self, state_manager):
        """Main processing loop"""
        interval = 1.0 / settings.camera.target_fps
        last_time = 0
        cleanup_counter = 0
        
        while not self.quit:
            now = time.time()
            if now - last_time < interval:
                time.sleep(0.001)
                continue
            last_time = now
            
            # Read frame
            if not self.cap or not self.cap.isOpened():
                if not self._reconnect():
                    time.sleep(2.0)
                    continue
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if not self._check_health():
                    self._reconnect()
                continue
            
            self._last_frame_time = time.time()
            self._frame_idx += 1
            
            # Cleanup periodically
            cleanup_counter += 1
            if cleanup_counter >= 100:
                self._fire_filter.cleanup()
                cleanup_counter = 0
            
            # Detect IR mode
            if self._frame_idx % 10 == 0:
                self._detect_ir(frame)
            
            # Apply color filter based on mode
            frame = self._apply_color_filter(frame)
            
            # Resize for processing
            proc_size = settings.camera.process_size
            small = cv2.resize(frame, tuple(proc_size))
            
            h, w = frame.shape[:2]
            scale_x = w / proc_size[0]
            scale_y = h / proc_size[1]
            
            detection_enabled = state_manager.is_detection_enabled(self.source_id)
            self._last_detection_enabled = detection_enabled
            
            # Motion detection
            has_motion = self._motion_detector.detect(small)
            
            # === SMART KEEP-ALIVE LOGIC ===
            # 1. Motion activates AI for at least 5 seconds
            if has_motion:
                self._ai_active_until = now + 5.0
            
            # 2. Check if we should run AI
            # Run if detection enabled AND (timer active OR we just started)
            should_run_ai = detection_enabled and (now < self._ai_active_until or self._frame_idx < 30)

            if should_run_ai:
                self._process_persons(small, frame, scale_x, scale_y)
                
                # 3. If AI sees people, keep AI alive even if motion stops
                # This fixes the "lost ID when standing still" issue
                if self.person_tracker.has_tracks():
                    self._ai_active_until = now + 5.0
            
            # Fire detection (Always run on low freq or same logic? Let's keep separate logic or same)
            # Fire is critical, maybe keep running or bind to same logic? 
            # Usually fire has motion (flicker), but safest to run always if enabled or use same logic.
            # Let's run fire always to be safe, it's lightweight.
            if not self.fire_queue.full():
                self.fire_queue.put(small.copy())
                self.fire_queue.put(small.copy())
            
            # Behavior analysis (only when detection enabled)
            if detection_enabled and self.behavior_analyzer and not self.behavior_queue.full():
                self.behavior_queue.put(small.copy())
            
            # Process results
            self._process_results(frame, scale_x, scale_y)
            
            # Update frames
            display = frame.copy()
            self._draw_overlays(display, detection_enabled)
            
            with self._frame_lock:
                self._last_frame = display
                self._raw_frame = frame.copy()
        
        self.release()
    
    def _process_persons(self, small: np.ndarray, full: np.ndarray, scale_x: float, scale_y: float):
        """Process person detections"""
        try:
            threshold = settings.get('detection.person_confidence', 0.5)
            if self._is_ir:
                threshold = settings.get('camera.infrared.person_detection_threshold', 0.45)
            
            detections = self.person_tracker.detect(small, threshold)
            
            # Skip face recognition in IR mode (grayscale limitation)
            if self._is_ir:
                # Update without face checks
                self.person_tracker.update(detections, full, scale_x, scale_y, skip_face_check=True)
            else:
                # Normal update with face recognition
                self.person_tracker.update(detections, full, scale_x, scale_y)
            
            # Check alerts
            for tid, alert_type, metadata in self.person_tracker.check_alerts():
                if self.on_person_alert:
                    alert_frame = full.copy()
                    self._draw_overlays(alert_frame, True)
                    self.on_person_alert(self.source_id, alert_frame, alert_type, metadata)
                    
        except Exception as e:
            print(f"Person processing error: {e}")
    
    def _process_results(self, frame: np.ndarray, scale_x: float, scale_y: float):
        """Process detection results from queues"""
        self._fire_boxes = []
        
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                result_type = result[0]
                
                if result_type == 'fire':
                    detections = result[1]
                    self._handle_fire_detections(detections, frame, scale_x, scale_y)
                
                elif result_type == 'behavior':
                    behavior_result = result[1]
                    alert_frame = result[2]
                    self._handle_behavior_alert(behavior_result, alert_frame)
                    
        except queue.Empty:
            pass
    
    def _handle_fire_detections(self, detections: list, frame: np.ndarray, scale_x: float, scale_y: float):
        """Handle fire detection results with Red Alert Mode"""
        validated_dets = []
        
        for det in detections:
            bbox = det['bbox']
            
            # Validate with filter
            if not self._fire_filter.validate(frame, bbox, self._is_ir):
                continue
            
            # Scale to original size
            x1, y1, x2, y2 = bbox
            scaled_bbox = (
                int(x1 * scale_x), int(y1 * scale_y),
                int(x2 * scale_x), int(y2 * scale_y)
            )
            
            self._fire_boxes.append(scaled_bbox)
            self._fire_history.append({'time': time.time(), **det})
            validated_dets.append(det)
        
        # Update fire tracker and check alert conditions
        now = time.time()
        should_alert, is_yellow, is_red = self._fire_tracker.update(validated_dets, now)
        
        # Trigger alert if conditions met
        if should_alert and self.on_fire_alert:
            alert_frame = frame.copy()
            self._draw_overlays(alert_frame, True)
            
            # Red Alert = CRITICAL, Yellow Alert = WARNING
            alert_type = AlertType.FIRE_CRITICAL if is_red else AlertType.FIRE_WARNING
            
            if is_red:
                print(f"🔴 RED ALERT - Camera {self.source_id}")
            elif is_yellow:
                print(f"🟡 Yellow Alert - Camera {self.source_id}")
            
            self.on_fire_alert(self.source_id, alert_frame, alert_type)
    
    def _handle_behavior_alert(self, result, frame: np.ndarray):
        """Handle behavior anomaly alert"""
        if self.on_person_alert:
            # Draw behavior visualization
            if self.behavior_analyzer:
                self.behavior_analyzer.draw_on_frame(frame, result)
            
            metadata = {
                'score': result.score,
                'timestamp': result.timestamp
            }
            self.on_person_alert(
                self.source_id,
                frame,
                AlertType.ANOMALOUS_BEHAVIOR,
                metadata
            )
    
    def _draw_overlays(self, frame: np.ndarray, detection_enabled: bool):
        """Draw detection overlays"""
        # Fire boxes
        for box in self._fire_boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
            cv2.putText(frame, "🔥 FIRE", (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Person tracks với behavior score
        if detection_enabled:
            self._draw_persons_with_behavior(frame)
        
        # Motion boxes
        # Vẽ các vùng chuyển động (Cyan)
        if hasattr(self._motion_detector, 'motion_boxes'):
            for (mx1, my1, mx2, my2) in self._motion_detector.motion_boxes:
                # Scale motion box from process_size to display frame size if needed
                # Note: motion detector now returns coordinates relative to input frame (small size)
                # But wait, we modified motion detector to return SCALED back boxes relative to 'small' input?
                # No, we modified it to return coordinates scaled to 'frame' passed to detect()
                # Here we pass 'small' (1280x720) to detect().
                # So the boxes are in 1280x720 coordinates.
                
                # However, 'frame' here in draw_overlays is the original/display frame?
                # Let's check process_loop: 
                # small = cv2.resize(frame, tuple(proc_size)) -> 1280x720
                # display = frame.copy() -> original resolution (could be same or higher)
                
                # If display is same size as small (1280x720), we can draw directly.
                # If not, we need scaling.
                # Assuming display is same for now as we usually display what we process or similar based on request.
                # Actually in process_loop: display = frame.copy(). And frame is from cap.read().
                # And small is resized process_size.
                
                # Safe calculation:
                dh, dw = frame.shape[:2]  # Display/Original frame size
                ph, pw = settings.camera.process_size[1], settings.camera.process_size[0] # Process size (1280x720)
                
                sx = dw / pw
                sy = dh / ph
                
                # If sizes match, sx=1. If original is larger, sx > 1.
                # Motion boxes are returned relative to INPUT of detect(), which is 'small'.
                
                # Wait, let's re-read my motion detector update.
                # I scale UP inside detect() based on input frame size vs 320x180.
                # Result: motion_boxes are in coordinates of 'small'.
                
                final_x1 = int(mx1 * sx)
                final_y1 = int(my1 * sy)
                final_x2 = int(mx2 * sx)
                final_y2 = int(my2 * sy)
                
                cv2.rectangle(frame, (final_x1, final_y1), (final_x2, final_y2), (255, 255, 0), 1)
                cv2.putText(frame, "Motion", (final_x1, final_y1 - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # IR indicator
        if self._is_ir:
            cv2.putText(frame, "IR MODE", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        

    
    def _draw_persons_with_behavior(self, frame: np.ndarray):
        """Draw person boxes with behavior status"""
        tracks = self.person_tracker._tracks
        
        # Lấy behavior score hiện tại
        behavior_score = 0.0
        behavior_threshold = 0.5
        
        if self.behavior_analyzer:
            behavior_score = self.behavior_analyzer.current_score
            behavior_threshold = self.behavior_analyzer.threshold
        
        is_anomaly = behavior_score >= behavior_threshold
        
        for tid, track in tracks.items():
            x1, y1, x2, y2 = map(int, track.bbox)
            
            # Xác định tên hiển thị
            name = track.confirmed_name or track.name
            is_stranger = (name == "Stranger")
            
            # === XÁC ĐỊNH MÀU BOX ===
            if is_anomaly:
                # Đỏ - Hành vi bất thường
                color = (0, 0, 255)
                status = "BAT THUONG"
            elif is_stranger:
                # Vàng/Cam - Chưa xác định
                color = (0, 165, 255)
                status = "Chua xac dinh"
            else:
                # Xanh lá - Bình thường (đã nhận diện)
                color = (0, 255, 0)
                status = "Binh thuong"
            
            # === VẼ BOX ===
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # === TẠO LABEL ===
            # Format: "ID:X Name | Status (score)"
            if self.behavior_analyzer and self.behavior_analyzer.loaded:
                label = f"ID:{tid} {name} | {status} ({behavior_score:.2f})"
            else:
                label = f"ID:{tid} {name}"
            
            # === VẼ LABEL PHÍA TRÊN BOX ===
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Vị trí label (phía trên box)
            label_y1 = max(0, y1 - text_h - 10)
            label_y2 = y1 - 2
            label_x2 = min(frame.shape[1], x1 + text_w + 8)
            
            # Background cho label
            cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
            
            # Text
            cv2.putText(frame, label, (x1 + 4, label_y2 - 4), font, font_scale, (255, 255, 255), thickness)
        
        # === VẼ SKELETON (nếu có) ===
        if self.behavior_analyzer and self.behavior_analyzer.current_pose:
            self._draw_skeleton_only(frame, is_anomaly)


    def _draw_skeleton_only(self, frame: np.ndarray, is_anomaly: bool):
        """Draw skeleton without separate score display"""
        analyzer = self.behavior_analyzer
        pose = analyzer.current_pose
        
        if not pose or not pose.is_valid:
            return
        
        # Scale keypoints
        h, w = frame.shape[:2]
        proc_w, proc_h = settings.camera.process_size
        
        scale_x = w / proc_w
        scale_y = h / proc_h
        
        scaled_kps = pose.keypoints.copy()
        scaled_kps[:, 0] *= scale_x
        scaled_kps[:, 1] *= scale_y
        
        # Màu skeleton theo trạng thái
        if is_anomaly:
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 0)  # Green
        
        # COCO skeleton connections
        SKELETON = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Draw bones
        for i, j in SKELETON:
            if i < len(pose.confidence) and j < len(pose.confidence):
                if pose.confidence[i] > 0.3 and pose.confidence[j] > 0.3:
                    pt1 = tuple(scaled_kps[i].astype(int))
                    pt2 = tuple(scaled_kps[j].astype(int))
                    cv2.line(frame, pt1, pt2, color, 2)
        
        # Draw joints
        for pt, conf in zip(scaled_kps, pose.confidence):
            if conf > 0.3:
                center = tuple(pt.astype(int))
                cv2.circle(frame, center, 5, color, -1)
                cv2.circle(frame, center, 5, (255, 255, 255), 1)
    
    def _detect_ir(self, frame: np.ndarray):
        """Detect infrared mode"""
        sample = frame[::10, ::10]
        b, g, r = cv2.split(sample.astype(np.float32))
        
        means = [np.mean(r), np.mean(g), np.mean(b)]
        std = np.std(means)
        ratio = min(means) / max(means) if max(means) > 0 else 1.0
        
        hsv = cv2.cvtColor(sample.astype(np.uint8), cv2.COLOR_BGR2HSV)
        sat = np.mean(hsv[:, :, 1])
        
        is_ir = std < 2.0 and ratio > 0.98 and sat < 10
        self._ir_history.append(is_ir)
        
        if len(self._ir_history) >= 10:
            ir_ratio = sum(self._ir_history) / len(self._ir_history)
            new_mode = ir_ratio >= 0.7
            
            if new_mode != self._is_ir:
                self._is_ir = new_mode
                mode_name = 'IR (Night Vision)' if new_mode else 'RGB (Color)'
                print(f"📷 Camera {self.source_id}: Switched to {mode_name} mode")
                if new_mode:
                    print(f"   → Face detection disabled (grayscale)")
    
    
    def _apply_color_filter(self, frame: np.ndarray) -> np.ndarray:
        """Apply color filter based on IR/RGB mode"""
        if self._is_ir:
            # Just convert to grayscale for consistency, no enhancement
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Return raw frame for RGB - No enhancement features at all
        return frame
    
    def _reconnect(self) -> bool:
        """Attempt reconnection"""
        self._reconnect_attempts += 1
        
        max_attempts = settings.get('camera.max_reconnect_attempts', 10)
        if self._reconnect_attempts > max_attempts:
            print(f"❌ Camera {self.source_id}: Max reconnect attempts ({max_attempts}) reached")
            return False
        
        print(f"Reconnecting camera {self.source_id}... (attempt {self._reconnect_attempts}/{max_attempts})")
        
        if self.cap:
            self.cap.release()
        
        time.sleep(2.0)
        self._init_capture()
        
        if self.cap and self.cap.isOpened():
            self._reconnect_attempts = 0
            return True
        
        return False
    
    def _check_health(self) -> bool:
        """Check connection health"""
        return time.time() - self._last_frame_time < 10
    
    def get_connection_status(self) -> bool:
        return self.cap is not None and self.cap.isOpened() and self._check_health()
    
    def has_active_threat(self) -> bool:
        """Check if camera has any active threat (Fire or Person)"""
        # 1. Fire Threat (Red or Yellow)
        if self._fire_tracker.is_red_alert or self._fire_tracker.is_yellow_alert:
            return True
        
        # 2. Person Threat (Stranger)
        if self.person_tracker.has_active_threats():
            return True
            
        # 3. Behavior Threat
        if self.behavior_analyzer and self.behavior_analyzer.current_score >= self.behavior_analyzer.threshold:
            return True
            
        return False
    
    def get_infrared_status(self) -> bool:
        return self._is_ir
    
    def force_reconnect(self):
        self._reconnect_attempts = 0
        self._reconnect()
    
    def release(self):
        self.quit = True
        if self.cap:
            self.cap.release()
        if self.behavior_analyzer:
            self.behavior_analyzer.close()
        print(f"Camera {self.source_id} released")
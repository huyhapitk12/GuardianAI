# core/camera.py  # type: ignore
import cv2
import time
import threading
import queue
import gc
import platform
from typing import Optional, Callable, Tuple, Dict
from collections import deque
import numpy as np

from config import settings, AlertType

class Camera:
    """Xử lý việc chụp và xử lý khung hình của máy ảnh"""
    
    def __init__(
        self,
        source,
        show_window: bool = False,
        on_person_alert: Optional[Callable] = None,
        on_fire_alert: Optional[Callable] = None,
        person_tracker=None
    ):
        self.source = source
        self.source_id = str(source)
        self.show_window = show_window
        self.on_person_alert = on_person_alert
        self.on_fire_alert = on_fire_alert
        self.person_tracker = person_tracker
        
        # Initialize capture with better error handling
        self.cap = None
        self._init_camera()
        
        # Trạng thái
        self.quit = False
        self._frame_lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._raw_frame: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None  # Khung hình grayscale trước đó cho optical flow
        self._frame_idx = 0
        self._warmup_frames = 30  # Bỏ qua 30 khung hình đầu tiên để ổn định
        
        # Cài đặt kết nối lại (đã cải thiện)
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10  # Tăng số lần thử
        self._reconnect_delay = 5    # Tăng thời gian chờ
        self._last_successful_frame = time.time()
        self._connection_timeout = 10  # giây không có khung hình thành công
        
        # Hàng đợi xử lý
        self.fire_queue = queue.Queue(maxsize=2)
        self.behavior_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=16)
        
        # Trạng thái phát hiện cháy
        self.recent_fire_detections = deque(
            maxlen=int(settings.camera.target_fps * settings.fire_logic.window_seconds) # Lưu trữ các phát hiện trong X giây
        )
        self.current_fire_boxes = []
        self.red_alert_mode_active = False
        self.red_alert_mode_until = 0
        self.yellow_alert_mode_active = False
        self.yellow_alert_mode_until = 0

        self.fire_objects: Dict[int, Dict] = {}
        self.next_fire_object_id = 0
        self.alerted_fire_object_ids = set()
        
        print(f"INFO: Camera initialized: {self.source}")
        
        # Lưu trạng thái bật/tắt nhận diện người ở khung gần nhất để đồng bộ vẽ với GUI
        self._last_person_detection_enabled = False
        
        # Trạng thái phát hiện hồng ngoại (IR)
        self._is_infrared_mode = False
        self._ir_detection_history = deque(maxlen=30)  # Lịch sử 30 khung hình để xác định IR mode ổn định
        self._ir_mode_stable_frames = 0
        self._ir_mode_threshold = 0.7  # 70% khung hình phải là IR để xác nhận chế độ IR
        
        # Debug flag for fire detection
        self._debug_fire_detection = settings.camera.debug_fire_detection
        
        # Behavior analysis
        self.behavior_analyzer = None
        self._last_behavior_alert = 0
        self._behavior_alert_cooldown = settings.get('behavior.alert_cooldown', 30)
        
        # Stranger tracking for conditional behavior analysis
        self._has_stranger = False
        self._last_stranger_detection = 0
        self._stranger_timeout = 60  # seconds - how long to keep behavior analysis active after last stranger
        
        # IR Enhancement control
        self.ir_enhancement_enabled = settings.camera.infrared.enhancement.enabled

        # Visualization state
        self.current_pose = None
        self.current_anomaly_score = 0.0
    
    def _get_camera_backends(self):
        """Get camera backends based on platform"""
        if platform.system() == 'Windows':
            return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
        elif platform.system() == 'Linux':
            return [cv2.CAP_V4L2, cv2.CAP_ANY]
        else:
            return [cv2.CAP_ANY]
    
    def _open_camera(self, source, backends):
        """Try to open camera with multiple backends"""
        if isinstance(source, int):
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(source, backend)
                    if self.cap.isOpened():
                        print(f"INFO: Opened camera with backend {backend}")
                        return True
                except Exception:
                    continue
            return False
        else:
            self.cap = cv2.VideoCapture(source)
            return self.cap.isOpened()
    
    def _configure_camera(self):
        """Configure camera properties"""
        if not self.cap or not self.cap.isOpened():
            return False
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)
            self.cap.set(cv2.CAP_PROP_FPS, 10)
        except Exception:
            pass
        
        return True
    
    def _enhance_ir_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE to enhance IR frame contrast"""
        try:
            if not self.ir_enhancement_enabled:
                return frame
            
            # Convert to LAB to enhance L channel
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(
                clipLimit=settings.camera.infrared.enhancement.clip_limit,
                tileGridSize=tuple(settings.camera.infrared.enhancement.tile_grid_size)
            )
            cl = clahe.apply(l)
            
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return enhanced
        except Exception as e:
            print(f"ERROR: Enhancement error: {e}")
            return frame

    def _init_camera(self):
        """Initialize camera with fallback options"""
        try:
            backends = self._get_camera_backends()
            if not self._open_camera(self.source, backends):
                print(f"WARNING: Initial camera connection failed: {self.source}")
                if not self._reconnect():
                    print(f"ERROR: Could not open camera: {self.source}")
                    self.cap = None
                    return
            
            self._configure_camera()
        except Exception as e:
            print(f"ERROR: Failed to initialize camera {self.source}: {e}")
            self.cap = None
    
    def _detect_infrared_mode(self, frame: np.ndarray) -> bool:
        """Detect if camera is in infrared mode"""
        try:
            h, w = frame.shape[:2]
            sample_size = min(h, w, 100)
            step_h = max(1, h // sample_size)
            step_w = max(1, w // sample_size)
            sample = frame[::step_h, ::step_w]
            
            b, g, r = cv2.split(sample.astype(np.float32))
            
            channel_means = [float(np.mean(r.astype(np.float32))), float(np.mean(g.astype(np.float32))), float(np.mean(b.astype(np.float32)))]
            channel_std = np.std(np.array(channel_means))
            
            max_mean = max(channel_means)
            min_mean = min(channel_means)
            color_ratio = min_mean / max_mean if max_mean > 0 else 1.0
            
            hsv = cv2.cvtColor(sample.astype(np.uint8), cv2.COLOR_BGR2HSV)
            saturation_mean = float(np.mean(hsv[:, :, 1].astype(np.float32)))
            
            channel_std_threshold = settings.camera.infrared.detection.channel_std_threshold
            color_ratio_threshold = settings.camera.infrared.detection.color_ratio_threshold
            saturation_threshold = settings.camera.infrared.detection.saturation_threshold
            
            is_ir = (channel_std < channel_std_threshold) and (color_ratio > color_ratio_threshold) and (saturation_mean < saturation_threshold)
            
            self._ir_detection_history.append(is_ir)
            
            if len(self._ir_detection_history) >= 10:
                ir_count = sum(self._ir_detection_history)
                ir_ratio = ir_count / len(self._ir_detection_history)
                
                previous_mode = self._is_infrared_mode
                self._is_infrared_mode = ir_ratio >= self._ir_mode_threshold
                
                if previous_mode != self._is_infrared_mode:
                    mode_str = "INFRARED" if self._is_infrared_mode else "RGB"
                    print(f"Camera {self.source_id} switched to {mode_str} mode (ratio: {ir_ratio:.2%})")
            
            return self._is_infrared_mode
            
        except Exception as e:
            print(f"ERROR: Infrared detection error: {e}")
            return False
    
    def _reconnect(self) -> bool:
        """Attempt camera reconnection"""
        self._reconnect_attempts += 1
        
        if self._reconnect_attempts > self._max_reconnect_attempts:
            print(f"ERROR: Max reconnection attempts reached for camera {self.source_id}")
            return False
        
        print(f"INFO: Camera reconnect attempt ({self._reconnect_attempts}/{self._max_reconnect_attempts})")
        
        try:
            if self.cap:
                self.cap.release()
            
            time.sleep(self._reconnect_delay)
            
            backends = self._get_camera_backends()
            if not self._open_camera(self.source, backends):
                print(f"WARNING: Reconnection attempt {self._reconnect_attempts} failed")
                return False
            
            self._configure_camera()
            print("INFO: Camera reconnected successfully")
            self._reconnect_attempts = 0
            self._last_successful_frame = time.time()
            return True
                
        except Exception as e:
            print(f"ERROR: Reconnection error: {e}")
            return False
    
    def _check_connection_health(self) -> bool:
        """Check camera connection health"""
        current_time = time.time()
        time_since_last_frame = current_time - self._last_successful_frame
        
        if time_since_last_frame > self._connection_timeout:
            print(f"WARNING: Camera connection timeout ({time_since_last_frame:.1f}s)")
            return False
        
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read latest processed frame"""
        with self._frame_lock:
            if self._last_frame is not None:
                return True, self._last_frame.copy()
            return False, None
    
    def read_raw(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read latest raw frame"""
        with self._frame_lock:
            if self._raw_frame is not None:
                return True, self._raw_frame.copy()
            return False, None
    
    def start_workers(self, fire_detector, face_detector, behavior_analyzer=None):
        """Start background worker threads"""
        self.fire_detector = fire_detector
        self.face_detector = face_detector
        self.behavior_analyzer = behavior_analyzer
        
        threading.Thread(
            target=self._fire_worker,
            daemon=True
        ).start()
        print("INFO: Fire detection worker started")
        
        if self.behavior_analyzer:
            threading.Thread(
                target=self._behavior_worker,
                daemon=True
            ).start()
            print("INFO: Behavior analysis worker started")
    
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
                print(f"ERROR: Fire worker error: {e}")
    def _behavior_worker(self):
        """Background worker for behavior analysis"""
        skip_counter = 0
        process_every_n = settings.get('behavior.process_every_n_frames', 3)
        
        while not self.quit:
            try:
                frame = self.behavior_queue.get(timeout=1.0)
                
                # Skip frames for performance
                skip_counter += 1
                if skip_counter % process_every_n != 0:
                    continue
                
                # New AnomalyDetector returns (frame, score, is_anomaly, pose)
                annotated_frame, score, is_anomaly, pose = self.behavior_analyzer.process_frame(frame)
                
                # Update state for visualization
                self.current_pose = pose
                self.current_anomaly_score = score
                
                # Update person tracker with behavior info
                if self.person_tracker and pose and pose.bbox:
                    try:
                        # Calculate scale factors
                        # We need original frame size. Assuming self.cap is available and valid.
                        # If not, we can't scale accurately, but usually it is.
                        # Or we can use self._raw_frame size if available.
                        orig_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) if self.cap else 0
                        orig_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if self.cap else 0
                        
                        if orig_w > 0 and orig_h > 0:
                            proc_w, proc_h = settings.camera.process_size
                            scale_x = orig_w / proc_w
                            scale_y = orig_h / proc_h
                            
                            self.person_tracker.update_behavior_status(
                                pose.bbox, 
                                score, 
                                is_anomaly, 
                                scale_x, 
                                scale_y
                            )
                    except Exception as e:
                        print(f"ERROR: Failed to update behavior status: {e}")
                
                if is_anomaly:
                    # Check cooldown
                    now = time.time()
                    if now - self._last_behavior_alert >= self._behavior_alert_cooldown:
                        # Create a simple result object/dict
                        result = {
                            'score': score,
                            'timestamp': now,
                            'is_anomaly': is_anomaly
                        }
                        self.result_queue.put(('behavior', result, annotated_frame))
                        self._last_behavior_alert = now
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"ERROR: Behavior worker error: {e}")

    def process_frames(self, state_manager):
        """Main frame processing loop"""
        frame_interval = 1.0 / settings.camera.target_fps
        last_time = 0
        frame_buffer = None
        
        while not self.quit:
            now = time.time()
            
            if now - last_time < frame_interval:
                time.sleep(0.001)
                continue
            last_time = now
            
            if not self.cap or not self.cap.isOpened():
                if not self._reconnect():
                    time.sleep(2.0)
                    continue
            
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    if not self._check_connection_health():
                        if not self._reconnect():
                            time.sleep(1.0)
                            continue
                    else:
                        time.sleep(0.005)
                        continue
                        
            except cv2.error as cv_err:
                print(f"ERROR: OpenCV error: {cv_err}")
                time.sleep(0.5)
                continue
            except Exception as e:
                print(f"ERROR: Frame read error: {e}")
                time.sleep(0.5)
                continue
            
            self._last_successful_frame = time.time()
            self._frame_idx += 1
            orig_h, orig_w = frame.shape[:2]
            
            if self._frame_idx % 10 == 0:
                self._detect_infrared_mode(frame)
            
            # Apply IR enhancement if needed
            if self._is_infrared_mode:
                frame = self._enhance_ir_frame(frame)
            
            person_detection_enabled = state_manager.is_person_detection_enabled(self.source_id)
            self._last_person_detection_enabled = person_detection_enabled
            
            try:
                if frame_buffer is None or frame_buffer.shape[:2] != tuple(settings.camera.process_size):
                    small_frame = cv2.resize(frame, tuple(settings.camera.process_size), interpolation=cv2.INTER_LINEAR)
                    frame_buffer = small_frame.copy()
                else:
                    cv2.resize(frame, tuple(settings.camera.process_size), dst=frame_buffer, interpolation=cv2.INTER_LINEAR)
                    small_frame = frame_buffer
            except Exception as e:
                print(f"ERROR: Resize error: {e}")
            
            # Calculate scale factors for mapping detections back to original frame
            scale_x = orig_w / settings.camera.process_size[0]
            scale_y = orig_h / settings.camera.process_size[1]
            
            # Person Detection
            if person_detection_enabled:
                self._process_persons(small_frame, frame, scale_x, scale_y, now)

            # Fire Detection
            if self.fire_detector and not self.fire_queue.full():
                self.fire_queue.put(small_frame.copy())
                
            # Behavior Analysis
            if self.behavior_analyzer and not self.behavior_queue.full():
                self.behavior_queue.put(small_frame.copy())
            
            self.current_fire_boxes = []
            self._process_fire_results(scale_x, scale_y, now, frame)
            
            if self.show_window or person_detection_enabled:
                display_frame = frame.copy()
                self._draw_visualizations(display_frame, person_detection_enabled)
                
                # Cập nhật các khung hình đã lưu
                self._update_frames(display_frame, frame)
                
                # Hiển thị cửa sổ nếu được bật
                if self.show_window:
                    try:
                        cv2.imshow("Guardian Detection", display_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.quit = True
                    except Exception as show_err:
                        print(f"WARNING: Failed to show window: {show_err}")
            else:
                # Khi nhận diện tắt, vẫn cập nhật cả hai khung hình để GUI hoạt động
                self._update_frames(frame, frame)
        
        self.release()

    def _process_persons(
        self,
        small_frame: np.ndarray,
        full_frame: np.ndarray,
        scale_x: float,
        scale_y: float,
        now: float
    ):
        """Xử lý phát hiện và theo dõi người"""
        try:
            if not self.person_tracker:
                return
            
            # Determine threshold
            threshold = settings.detection.person_confidence_threshold
            if self._is_infrared_mode:
                threshold = settings.camera.infrared.person_detection_threshold
                
            detections = self.person_tracker.detect_persons(small_frame, conf_threshold=threshold)
            self.person_tracker.update_tracks(
                detections,
                full_frame,
                scale_x,
                scale_y
            )
            
            # Kiểm tra cảnh báo
            alerts = self.person_tracker.check_confirmations()
            for track_id, alert_type, metadata in alerts:
                # Track stranger detection for conditional behavior analysis
                if alert_type == "nguoi_la":  # AlertType.STRANGER.value
                    self._has_stranger = True
                    self._last_stranger_detection = now
                    print(f"INFO: Stranger detected - activating behavior analysis for {self._stranger_timeout}s")
                
                if self.on_person_alert:
                    # Tạo frame chú thích giống GUI để gửi Telegram
                    alert_frame = full_frame.copy()
                    self._draw_visualizations(alert_frame, True)
                    self.on_person_alert(self.source_id, alert_frame, alert_type, metadata)
        except Exception as e:
            print(f"ERROR: Person processing error: {e}")

    def _process_fire_results(
        self,
        scale_x: float,
        scale_y: float,
        now: float,
        frame: np.ndarray
    ):
        """Xử lý kết quả phát hiện cháy và hành vi từ hàng đợi"""
        try:
            while not self.result_queue.empty():
                result_tuple = self.result_queue.get_nowait()
                result_type = result_tuple[0]
                
                if result_type == "fire":
                    results = result_tuple[1]
                    self._handle_fire_detections(
                        results,
                        scale_x,
                        scale_y,
                        now,
                        frame
                    )
                elif result_type == "behavior":
                    behavior_result = result_tuple[1]
                    alert_frame = result_tuple[2]
                    self._handle_behavior_detection(behavior_result, alert_frame)
        except queue.Empty:
            pass
    
    def _handle_behavior_detection(self, result, frame):
        """Handle behavior anomaly detection"""
        if self.on_person_alert:
            # Handle both object and dict for backward compatibility/flexibility
            score = result.get('score') if isinstance(result, dict) else result.score
            timestamp = result.get('timestamp') if isinstance(result, dict) else result.timestamp
            
            metadata = {
                'score': score,
                'timestamp': timestamp
            }
            self.on_person_alert(
                self.source_id,
                frame,
                AlertType.ANOMALOUS_BEHAVIOR.value,
                metadata
            )

    def _check_motion_infrared(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], motion_threshold: float, motion_std_min: float, debug: bool = False) -> bool:
        """
        Kiểm tra chuyển động trong vùng để xác định lửa thực
        Phiên bản đơn giản cho IR mode
        
        Args:
            frame: Khung hình đầy đủ
            bbox: Bounding box cần kiểm tra
            motion_threshold: Ngưỡng magnitude chuyển động
            motion_std_min: Độ lệch chuẩn magnitude tối thiểu
            debug: Bật/tắt debug log
        """
        try:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Mở rộng vùng kiểm tra một chút
            margin = 5
            x1_ext = max(0, x1 - margin)
            y1_ext = max(0, y1 - margin)
            x2_ext = min(frame.shape[1], x2 + margin)
            y2_ext = min(frame.shape[0], y2 + margin)
            
            roi = frame[y1_ext:y2_ext, x1_ext:x2_ext]
            if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
                return True  # Vùng quá nhỏ, bỏ qua kiểm tra motion
            
            # Chuyển sang grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Kiểm tra khung hình trước
            if self._prev_gray is None or self._prev_gray.shape != gray.shape:
                self._prev_gray = gray.copy()
                return True  # Chưa có khung hình trước, chấp nhận
            
            # Tính optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray,
                np.zeros_like(self._prev_gray),
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Cập nhật khung hình trước
            self._prev_gray = gray.copy()
            
            # Tính độ lớn của vector chuyển động
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            
            # Tính các chỉ số chuyển động
            magnitude_mean = float(np.mean(magnitude))
            magnitude_std = float(np.std(magnitude))
            strong_motion_pixels = np.sum(magnitude > motion_threshold)
            motion_ratio = strong_motion_pixels / magnitude.size if magnitude.size > 0 else 0
            
            # Lửa có chuyển động: magnitude_std > ngưỡng
            # Trong chế độ IR, giảm ngưỡng một chút để nhạy hơn
            std_threshold = motion_std_min * 0.8 if self._is_infrared_mode else motion_std_min
            has_motion = magnitude_std > std_threshold
            
            if debug and self._debug_fire_detection:
                print(f"━━━ MOTION CHECK ━━━")
                print(f"  📊 Magnitude: mean={magnitude_mean:.3f}, std={magnitude_std:.3f}")
                print(f"  🎯 Threshold: motion_std_min={motion_std_min:.3f} (adj: {std_threshold:.3f})")
                print(f"  📈 Strong motion pixels: {strong_motion_pixels} ({motion_ratio:.1%})")
                if has_motion:
                    print(f"  ✅ Motion detected: std={magnitude_std:.3f} > {std_threshold:.3f}")
                else:
                    print(f"  ❌ Static/slow motion: std={magnitude_std:.3f} <= {std_threshold:.3f}")
            
            return has_motion
            
        except Exception as e:
            if debug and self._debug_fire_detection:
                print(f"⚠️  Motion check error: {e}")
            # Khi có lỗi, chấp nhận (không reject)
            return True

    def _is_valid_yellow_alert_infrared(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], debug: bool = False) -> bool:
            """
            Bộ lọc cho cảnh báo vàng trong chế độ hồng ngoại
            
            Cảnh báo vàng ở chế độ IR cần:
            - Tiêu chí lỏng hơn cảnh báo đỏ
            - Vẫn loại bỏ các false positive rõ ràng
            - Cho phép các phát hiện nghi ngờ để người dùng xem xét
            """
            try:
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Bỏ qua bbox quá nhỏ (lỏng hơn red alert)
                roi_width = x2 - x1
                roi_height = y2 - y1
                if roi_width < 3 or roi_height < 3:
                    if debug:
                        print(f"❌ YELLOW IR FAIL: ROI quá nhỏ ({roi_width}x{roi_height})")
                    return False
                
                # Clamp bbox
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    return False
                
                # Chuyển sang grayscale
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
                
                # ===== TIÊU CHÍ 1: Độ sáng (lỏng hơn) =====
                brightness_mean = np.mean(gray_roi)
                brightness_max = np.max(gray_roi)
                
                # Lấy ngưỡng từ config
                brightness_mean_min = settings.camera.infrared.yellow_alert.brightness_mean_min
                brightness_max_min = settings.camera.infrared.yellow_alert.brightness_max_min
                
                if brightness_mean < brightness_mean_min and brightness_max < brightness_max_min:
                    if debug:
                        print(f"❌ YELLOW IR FAIL T1: Độ sáng thấp")
                        print(f"   Đo được: mean={brightness_mean:.1f}, max={brightness_max:.1f}")
                        print(f"   Yêu cầu: mean>={brightness_mean_min} OR max>={brightness_max_min}")
                    return False
                
                if debug:
                    print(f"✅ YELLOW IR PASS T1: Độ sáng OK")
                    print(f"   Đo được: mean={brightness_mean:.1f}, max={brightness_max:.1f}")
                    print(f"   Yêu cầu: mean>={brightness_mean_min} OR max>={brightness_max_min}")
                
                # ===== TIÊU CHÍ 2: Biến đổi cường độ (lỏng hơn) =====
                brightness_std = np.std(gray_roi)
                
                # Lấy ngưỡng từ config
                brightness_std_min = settings.camera.infrared.yellow_alert.brightness_std_min
                
                if brightness_std < brightness_std_min:
                    if debug:
                        print(f"❌ YELLOW IR FAIL T2: Quá đồng nhất")
                        print(f"   Đo được: std={brightness_std:.1f}")
                        print(f"   Yêu cầu: std>={brightness_std_min}")
                    return False
                
                if debug:
                    print(f"✅ YELLOW IR PASS T2: Biến đổi OK")
                    print(f"   Đo được: std={brightness_std:.1f}")
                    print(f"   Yêu cầu: std>={brightness_std_min}")
                
                # ===== TIÊU CHÍ 3: Loại bỏ vùng quá sáng (glare) =====
                # Lấy ngưỡng từ config
                very_bright_threshold = settings.camera.infrared.yellow_alert.very_bright_threshold
                very_bright_ratio_max = settings.camera.infrared.yellow_alert.very_bright_ratio_max
                
                very_bright_pixels = np.sum(gray_roi > very_bright_threshold)
                very_bright_ratio = very_bright_pixels / gray_roi.size if gray_roi.size > 0 else 0
                
                if very_bright_ratio > very_bright_ratio_max:
                    if debug:
                        print(f"❌ YELLOW IR FAIL T3: Phản xạ/glare (quá sáng đồng nhất)")
                        print(f"   Đo được: {very_bright_pixels} pixels ({very_bright_ratio:.2%})")
                        print(f"   Yêu cầu: >{very_bright_threshold} brightness, <={very_bright_ratio_max:.2%} ratio")
                    return False
                
                if debug:
                    print(f"✅ YELLOW IR PASS T3: Không phải glare")
                    print(f"   Đo được: {very_bright_pixels} pixels ({very_bright_ratio:.2%})")
                    print(f"   Yêu cầu: >{very_bright_threshold} brightness, <={very_bright_ratio_max:.2%} ratio")
                
                # ===== TIÊU CHÍ 4: Loại bỏ vùng quá tối đồng nhất =====
                # Lấy ngưỡng từ config
                very_dark_threshold = settings.camera.infrared.yellow_alert.very_dark_threshold
                very_dark_ratio_max = settings.camera.infrared.yellow_alert.very_dark_ratio_max
                
                very_dark_pixels = np.sum(gray_roi < very_dark_threshold)
                very_dark_ratio = very_dark_pixels / gray_roi.size if gray_roi.size > 0 else 0
                
                if very_dark_ratio > very_dark_ratio_max:
                    if debug:
                        print(f"❌ YELLOW IR FAIL T4: Vùng quá tối")
                        print(f"   Đo được: {very_dark_pixels} pixels ({very_dark_ratio:.2%})")
                        print(f"   Yêu cầu: <{very_dark_threshold} brightness, <={very_dark_ratio_max:.2%} ratio")
                    return False
                
                if debug:
                    print(f"✅ YELLOW IR PASS T4: Không quá tối")
                    print(f"   Đo được: {very_dark_pixels} pixels ({very_dark_ratio:.2%})")
                    print(f"   Yêu cầu: <{very_dark_threshold} brightness, <={very_dark_ratio_max:.2%} ratio")
                
                # ===== TIÊU CHÍ 5: Kiểm tra chuyển động (nếu được bật) =====
                if settings.camera.infrared.yellow_alert.check_motion:
                    motion_threshold = settings.camera.infrared.yellow_alert.motion_threshold
                    motion_std_min = settings.camera.infrared.yellow_alert.motion_std_min
                    
                    has_motion = self._check_motion_infrared(frame, bbox, motion_threshold, motion_std_min, debug)
                    if not has_motion:
                        if debug:
                            print(f"❌ YELLOW IR FAIL T5: Không có chuyển động đặc trưng của lửa")
                        return False
                    
                    if debug:
                        print(f"✅ YELLOW IR PASS T5: Có chuyển động")
                
                # ✅ PASS TẤT CẢ CÁC TIÊU CHÍ
                if debug:
                    print(f"🟡 ✅ YELLOW ALERT VALIDATED (IR MODE): bright={brightness_mean:.0f}, std={brightness_std:.0f}\n")
                
                return True
                
            except Exception as e:
                if debug:
                    print(f"⚠️  Yellow IR filter error: {e}")
                # Khi có lỗi, chấp nhận detection từ YOLO
                return True
    
    def _is_valid_fire_infrared(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], debug: bool = False) -> bool:
        """
        Bộ lọc cho chế độ hồng ngoại
        
        Ở chế độ IR, không thể dựa vào màu sắc, chỉ có thể dùng:
        - Độ sáng cao (nhiệt phát ra ánh sáng hồng ngoại)
        - Biến đổi cường độ (lửa nhấp nháy)
        - Vùng sáng tập trung (không phải ánh sáng môi trường)
        """
        try:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Bỏ qua bbox quá nhỏ
            roi_width = x2 - x1
            roi_height = y2 - y1
            if roi_width < 5 or roi_height < 5:
                if debug:
                    print(f"❌ IR FAIL: ROI quá nhỏ ({roi_width}x{roi_height})")
                return False
            
            # Clamp bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return False
            
            # Chuyển sang grayscale để phân tích độ sáng
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            # ===== TIÊU CHÍ 1: Độ sáng cao =====
            # Lửa trong IR thường rất sáng do phát nhiệt
            brightness_mean = np.mean(gray_roi)
            brightness_max = np.max(gray_roi)
            
            # Lấy ngưỡng từ config
            brightness_mean_min = settings.camera.infrared.red_alert.brightness_mean_min
            brightness_max_min = settings.camera.infrared.red_alert.brightness_max_min
            
            if brightness_mean < brightness_mean_min and brightness_max < brightness_max_min:
                if debug:
                    print(f"❌ IR FAIL T1: Độ sáng thấp")
                    print(f"   Đo được: mean={brightness_mean:.1f}, max={brightness_max:.1f}")
                    print(f"   Yêu cầu: mean>={brightness_mean_min} OR max>={brightness_max_min}")
                return False
            
            if debug:
                print(f"✅ IR PASS T1: Độ sáng OK")
                print(f"   Đo được: mean={brightness_mean:.1f}, max={brightness_max:.1f}")
                print(f"   Yêu cầu: mean>={brightness_mean_min} OR max>={brightness_max_min}")
            
            # ===== TIÊU CHÍ 2: Biến đổi cường độ =====
            # Lửa có biến đổi độ sáng, không đồng nhất
            brightness_std = np.std(gray_roi)
            
            # Lấy ngưỡng từ config
            brightness_std_min = settings.camera.infrared.red_alert.brightness_std_min
            
            # Với ảnh đã enhance, độ tương phản tăng nên std cũng tăng, nhưng ta vẫn giảm ngưỡng check
            # để đảm bảo bắt được lửa nhỏ
            if brightness_std < (brightness_std_min * 0.8):
                if debug:
                    print(f"❌ IR FAIL T2: Quá đồng nhất")
                    print(f"   Đo được: std={brightness_std:.1f}")
                    print(f"   Yêu cầu: std>={brightness_std_min}")
                return False
            
            if debug:
                print(f"✅ IR PASS T2: Biến đổi OK")
                print(f"   Đo được: std={brightness_std:.1f}")
                print(f"   Yêu cầu: std>={brightness_std_min}")
            
            # ===== TIÊU CHÍ 3: Vùng sáng tập trung =====
            # Lửa có vùng sáng tập trung, không phải ánh sáng môi trường rải rác
            # Lấy ngưỡng từ config
            bright_pixel_threshold = settings.camera.infrared.red_alert.bright_pixel_threshold
            bright_pixel_ratio_min = settings.camera.infrared.red_alert.bright_pixel_ratio_min
            
            bright_pixels = np.sum(gray_roi > bright_pixel_threshold)
            bright_ratio = bright_pixels / gray_roi.size if gray_roi.size > 0 else 0
            
            if bright_ratio < bright_pixel_ratio_min:
                if debug:
                    print(f"❌ IR FAIL T3: Không đủ vùng sáng tập trung")
                    print(f"   Đo được: {bright_pixels} pixels ({bright_ratio:.2%})")
                    print(f"   Yêu cầu: >{bright_pixel_threshold} brightness, >={bright_pixel_ratio_min:.2%} ratio")
                return False
            
            if debug:
                print(f"✅ IR PASS T3: Vùng sáng OK")
                print(f"   Đo được: {bright_pixels} pixels ({bright_ratio:.2%})")
                print(f"   Yêu cầu: >{bright_pixel_threshold} brightness, >={bright_pixel_ratio_min:.2%} ratio")
            
            # ===== TIÊU CHÍ 4: Loại bỏ vùng quá sáng đồng nhất (ánh sáng phản xạ) =====
            # Lấy ngưỡng từ config
            very_bright_threshold = settings.camera.infrared.red_alert.very_bright_threshold
            very_bright_ratio_max = settings.camera.infrared.red_alert.very_bright_ratio_max
            
            very_bright_pixels = np.sum(gray_roi > very_bright_threshold)
            very_bright_ratio = very_bright_pixels / gray_roi.size if gray_roi.size > 0 else 0
            
            if very_bright_ratio > very_bright_ratio_max:
                if debug:
                    print(f"❌ IR FAIL T4: Phản xạ/glare (quá sáng đồng nhất)")
                    print(f"   Đo được: {very_bright_pixels} pixels ({very_bright_ratio:.2%})")
                    print(f"   Yêu cầu: >{very_bright_threshold} brightness, <={very_bright_ratio_max:.2%} ratio")
                return False
            
            if debug:
                print(f"✅ IR PASS T4: Không phải glare")
                print(f"   Đo được: {very_bright_pixels} pixels ({very_bright_ratio:.2%})")
                print(f"   Yêu cầu: >{very_bright_threshold} brightness, <={very_bright_ratio_max:.2%} ratio")
            
            # ===== TIÊU CHÍ 5: Kiểm tra chuyển động (nếu được bật) =====
            if settings.camera.infrared.red_alert.check_motion:
                motion_threshold = settings.camera.infrared.red_alert.motion_threshold
                motion_std_min = settings.camera.infrared.red_alert.motion_std_min
                
                has_motion = self._check_motion_infrared(frame, bbox, motion_threshold, motion_std_min, debug)
                if not has_motion:
                    if debug:
                        print(f"❌ IR FAIL T5: Không có chuyển động đặc trưng của lửa")
                    return False
                
                if debug:
                    print(f"✅ IR PASS T5: Có chuyển động")
            
            # ✅ PASS TẤT CẢ CÁC TIÊU CHÍ
            if debug:
                print(f"🔥 ✅ FIRE VALIDATED (IR MODE): bright={brightness_mean:.0f}, std={brightness_std:.0f}, bright_ratio={bright_ratio:.2%}\n")
            
            return True
            
        except Exception as e:
            if debug:
                print(f"⚠️  IR filter error: {e}")
            # Khi có lỗi, chấp nhận detection từ YOLO
            return True
    
    def _is_valid_fire_color(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], debug: bool = False) -> bool:
        """
        Bộ lọc phân biệt lửa thực vs ánh sáng chói/glare
        
        Chiến lược:
        - Lửa thực: Màu đỏ/cam/vàng + độ sáng cao + có biến đổi cường độ
        - Ánh sáng chói: Trắng sáng + saturation thấp + R≈G≈B
        - Khói: Có thể xám nhạt, cần relax hơn
        - Chế độ hồng ngoại: Bỏ qua kiểm tra màu sắc, chỉ dựa vào độ sáng và biến đổi
        
        Được tối ưu để cân bằng giữa độ chính xác và recall
        """
        try:
            # Nếu ở chế độ hồng ngoại, áp dụng logic đơn giản hơn
            if self._is_infrared_mode:
                return self._is_valid_fire_infrared(frame, bbox, debug)
            
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Bỏ qua bbox quá nhỏ
            roi_width = x2 - x1
            roi_height = y2 - y1
            if roi_width < 5 or roi_height < 5:  # Giảm từ 10 -> 5
                return False
            
            # Clamp bbox vào frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return False

            # Chuyển sang HSV để phân tích màu sắc
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv_roi)
            
            # ===== TIÊU CHÍ 1: Hue (màu sắc) - RELAXED =====
            # Chấp nhận dải màu rộng hơn: đỏ, cam, vàng
            red_mask = ((h >= 0) & (h <= 15)) | ((h >= 165) & (h <= 180))  # Mở rộng từ 10/170 -> 15/165
            orange_yellow_mask = (h >= 5) & (h <= 50)  # Mở rộng từ 48 -> 50
            hue_mask = red_mask | orange_yellow_mask
            
            hue_pixels = np.sum(hue_mask)
            hue_ratio = hue_pixels / hue_mask.size if hue_mask.size > 0 else 0
            
            # Giảm threshold từ 0.1993 -> 0.10 (10% thay vì 20%)
            if hue_ratio < 0.10:
                if debug:
                    print(f"❌ FAIL T1: Không đủ màu lửa")
                    print(f"   Đo được: {hue_pixels} pixels ({hue_ratio:.2%})")
                    print(f"   Yêu cầu: Hue 0-15° hoặc 165-180° (đỏ) hoặc 5-50° (cam-vàng), >=10% pixels")
                return False
            
            if debug:
                print(f"✅ PASS T1: Hue OK")
                print(f"   Đo được: {hue_pixels} pixels ({hue_ratio:.2%}), yêu cầu >=10%")
            
            # ===== TIÊU CHÍ 2: Brightness - RELAXED =====
            v_mean = float(np.mean(v.astype(np.float32)))
            # Giảm từ 101.3892 -> 80 để chấp nhận lửa tối hơn
            if v_mean < 80:
                if debug:
                    print(f"❌ FAIL T2: Quá tối")
                    print(f"   Đo được: V_mean={v_mean:.1f}")
                    print(f"   Yêu cầu: V_mean>=80")
                return False
            
            if debug:
                print(f"✅ PASS T2: Đủ sáng")
                print(f"   Đo được: V_mean={v_mean:.1f}, yêu cầu >=80")
            
            # ===== TIÊU CHÍ 3: Saturation - RELAXED =====
            s_mean = float(np.mean(s.astype(np.float32)))
            s_std = float(np.std(s.astype(np.float32)))
            
            # Detect loại camera để áp dụng logic khác nhau
            v_std = float(np.std(v.astype(np.float32)))
            is_thermal_camera = (s_mean < 15) and (v_std > 25)
            
            if debug:
                camera_type = "🔴 THERMAL" if is_thermal_camera else "📱 RGB"
                print(f"📷 Camera: {camera_type} (S_mean={s_mean:.1f}, V_std={v_std:.1f})")
            
            # Với RGB camera: kiểm tra saturation để loại bỏ ánh sáng trắng
            # Giảm từ 20 -> 15 để chấp nhận nhiều hơn
            if not is_thermal_camera and s_mean < 15:
                if debug:
                    print(f"❌ FAIL T3: Ánh sáng trắng (saturation thấp)")
                    print(f"   Đo được: S_mean={s_mean:.1f}")
                    print(f"   Yêu cầu: S_mean>=15 (hoặc thermal camera)")
                return False
            
            if debug:
                print(f"✅ PASS T3: Saturation OK")
                print(f"   Đo được: S_mean={s_mean:.1f}, yêu cầu >=15")
            
            # ===== TIÊU CHÍ 4: RGB Ratio - RELAXED =====
            b, g, r = cv2.split(roi.astype(np.float32))
            r_mean = float(np.mean(r.astype(np.float32)))
            g_mean = float(np.mean(g.astype(np.float32)))
            b_mean = float(np.mean(b.astype(np.float32)))
            
            channel_max = max(r_mean, g_mean, b_mean)
            channel_min = min(r_mean, g_mean, b_mean)
            
            rgb_ratio = channel_min / channel_max if channel_max > 0 else 1.0
            
            # Chỉ loại bỏ ánh sáng trắng rõ ràng
            # Giảm từ 0.8923 -> 0.92 để strict hơn với white light
            if not is_thermal_camera and rgb_ratio > 0.92:
                if debug:
                    print(f"❌ FAIL T4: Ánh sáng trắng (R≈G≈B)")
                    print(f"   Đo được: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}, ratio={rgb_ratio:.3f}")
                    print(f"   Yêu cầu: RGB_ratio<=0.92")
                return False
            
            if debug:
                print(f"✅ PASS T4: RGB ratio OK")
                print(f"   Đo được: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}, ratio={rgb_ratio:.3f}, yêu cầu <=0.92")
            
            # ===== TIÊU CHÍ 5: Value Variance - RELAXED =====
            # Lửa nên có biến đổi cường độ (không phẳng)
            # Mở rộng range từ (24.39, 88.59) -> (15, 100) để chấp nhận nhiều hơn
            if v_std < 15:
                if debug:
                    print(f"❌ FAIL T5: Quá đồng nhất (không có biến đổi)")
                    print(f"   Đo được: V_std={v_std:.1f}")
                    print(f"   Yêu cầu: V_std>=15")
                return False
            
            if v_std > 100 and not is_thermal_camera:
                if debug:
                    print(f"⚠️  WARNING T5: Biến đổi quá cao ({v_std:.1f} > 100), có thể là nhiễu")
                # Không reject, chỉ cảnh báo
            
            if debug:
                print(f"✅ PASS T5: Value variance OK")
                print(f"   Đo được: V_std={v_std:.1f}, yêu cầu 15-100")
            
            # ===== TIÊU CHÍ 6: Kiểm tra chuyển động (nếu được bật) =====
            if settings.camera.rgb.check_motion:
                motion_threshold = settings.camera.rgb.motion_threshold
                motion_std_min = settings.camera.rgb.motion_std_min
                
                has_motion = self._check_motion_infrared(frame, bbox, motion_threshold, motion_std_min, debug)
                if not has_motion:
                    if debug:
                        print(f"❌ FAIL T6: Không có chuyển động đặc trưng của lửa")
                    return False
                
                if debug:
                    print(f"✅ PASS T6: Có chuyển động")
            
            # ✅ PASS TẤT CẢ CÁC TIÊU CHÍ
            if debug:
                camera_label = "THERMAL" if is_thermal_camera else "RGB"
                print(f"🔥 ✅ FIRE VALIDATED ({camera_label}): hue={hue_ratio:.2f}, bright={v_mean:.0f}, sat={s_mean:.0f}, var={v_std:.0f}\n")
            
            return True
            
        except Exception as e:
            if debug:
                print(f"⚠️  Color filter error: {e}")
            # Khi có lỗi, chấp nhận detection từ YOLO (thay vì reject)
            return True

    def _handle_fire_detections(
        self,
        detections: list,
        scale_x: float,
        scale_y: float,
        now: float,
        frame: np.ndarray
    ):
        """Xử lý các phát hiện cháy và xác định mức độ cảnh báo"""
        DEBUG = True
    
        valid_detections = []
        for d in detections:
            result = self._is_valid_fire_color(frame, d['bbox'], debug=self._debug_fire_detection)
            if result:
                valid_detections.append(d)
        
        if not valid_detections:
            return

        # Cập nhật các hộp lửa hiện tại để hiển thị
        for det in valid_detections:
            x1, y1, x2, y2 = det['bbox']
            x1_orig = int(x1 * scale_x)
            y1_orig = int(y1 * scale_y)
            x2_orig = int(x2 * scale_x)
            y2_orig = int(y2 * scale_y)
            self.current_fire_boxes.append((x1_orig, y1_orig, x2_orig, y2_orig))
        # Thêm vào lịch sử
        for det in valid_detections:
            det['timestamp'] = now
            self.recent_fire_detections.append(det)
        
        # Kiểm tra xem khóa cảnh báo đỏ đã hết hạn chưa
        if self.red_alert_mode_active and now > self.red_alert_mode_until:
            self.red_alert_mode_active = False
            print("INFO: Red alert lockdown expired")
            print("INFO: Chế độ khóa Cảnh báo Đỏ đã hết hạn.")
        
        if not self.recent_fire_detections:
            return
        
        # --- LOGIC CẢNH BÁO ĐỎ (KHẨN CẤP) ---
        
        # Xác định mức độ cảnh báo
        is_red_alert = False
        
        if self.red_alert_mode_active and valid_detections:
            is_red_alert = True
        else:
            # Kiểm tra sự phát triển của đám cháy
            # Nếu có nhiều phát hiện trong thời gian ngắn -> Cảnh báo đỏ
            recent_count = len(self.recent_fire_detections)
            
            # Ngưỡng số lượng phát hiện để kích hoạt cảnh báo đỏ
            # Giảm ngưỡng nếu ở chế độ IR (vì đã lọc kỹ)
            count_threshold = settings.fire_logic.confirmation_count
            if self._is_infrared_mode:
                count_threshold = max(1, int(count_threshold * 0.7))
            
            if recent_count >= count_threshold:
                is_red_alert = True
                # Kích hoạt chế độ khóa cảnh báo đỏ
                self.red_alert_mode_active = True
                self.red_alert_mode_until = now + settings.fire_logic.red_alert_lockdown_duration
                print(f"INFO: Red alert activated (count={recent_count})")
        
        if is_red_alert:
            if self.on_fire_alert:
                # Gửi cảnh báo đỏ
                # Chuẩn bị metadata
                metadata = {
                    "confidence": max([d.get('conf', 0) for d in valid_detections]),
                    "box_count": len(valid_detections),
                    "is_infrared": self._is_infrared_mode
                }
                
                # Tạo frame chú thích
                alert_frame = frame.copy()
                self._draw_visualizations(alert_frame, self._last_person_detection_enabled)
                
                self.on_fire_alert(self.source_id, alert_frame, AlertType.FIRE_RED, metadata)
            return

        # --- LOGIC CẢNH BÁO VÀNG (CẢNH BÁO SỚM) ---
        
        # Nếu có phát hiện nhưng chưa đủ để kích hoạt đỏ -> Cảnh báo vàng
        # Kiểm tra bộ lọc vàng (lỏng hơn)
        is_yellow_alert = False
        
        # Kiểm tra xem khóa cảnh báo vàng đã hết hạn chưa
        if self.yellow_alert_mode_active and now > self.yellow_alert_mode_until:
            self.yellow_alert_mode_active = False
        
        # Chỉ kích hoạt vàng nếu chưa active hoặc đã hết hạn
        if not self.yellow_alert_mode_active:
            # Kiểm tra xem có phát hiện nào thỏa mãn bộ lọc vàng không
            valid_yellow_detections = []
            for d in detections: # Check all detections, not just valid_detections (which passed red filter)
                 # Nếu đã pass red filter thì chắc chắn pass yellow
                if d in valid_detections:
                    valid_yellow_detections.append(d)
                    continue
                
                # Nếu fail red filter, check yellow filter
                if self._is_infrared_mode:
                    if self._is_valid_yellow_alert_infrared(frame, d['bbox'], debug=False):
                        valid_yellow_detections.append(d)
                else:
                    # Logic vàng cho RGB (tạm thời dùng lại logic đỏ nhưng chấp nhận confidence thấp hơn từ model)
                    # Ở đây ta giả định model đã filter confidence thấp rồi
                    # Nên nếu fail red filter màu sắc thì có thể vẫn là khói hoặc lửa mới
                    pass 

            if valid_yellow_detections:
                is_yellow_alert = True
                self.yellow_alert_mode_active = True
                self.yellow_alert_mode_until = now + settings.fire_logic.yellow_alert_lockdown_duration
                print(f"INFO: Yellow alert activated")

        if is_yellow_alert:
            if self.on_fire_alert:
                metadata = {
                    "confidence": max([d.get('conf', 0) for d in detections]), # Use raw detections max conf
                    "box_count": len(detections),
                    "is_infrared": self._is_infrared_mode
                }
                
                alert_frame = frame.copy()
                self._draw_visualizations(alert_frame, self._last_person_detection_enabled)
                
                self.on_fire_alert(self.source_id, alert_frame, AlertType.FIRE_YELLOW, metadata)

    def get_infrared_status(self) -> bool:
        """Trả về trạng thái chế độ hồng ngoại hiện tại"""
        return self._is_infrared_mode

    def _draw_visualizations(self, frame, person_detection_enabled):
        """Vẽ các hộp giới hạn và thông tin lên khung hình"""
        # Vẽ vùng cháy
        for box in self.current_fire_boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(frame, "FIRE", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        # Vẽ người nếu được bật
        if person_detection_enabled and self.person_tracker:
            self.person_tracker.draw_tracks(frame)
            
        # Vẽ trạng thái IR
        if self._is_infrared_mode:
            cv2.putText(frame, "IR MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if self.ir_enhancement_enabled:
                cv2.putText(frame, "ENHANCED", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Vẽ behavior analysis (nếu có)
        if self.behavior_analyzer and self.current_pose and self.current_pose.is_valid:
            try:
                # Scale keypoints từ process_size về frame size hiện tại
                h, w = frame.shape[:2]
                proc_w, proc_h = settings.camera.process_size
                
                scale_x = w / proc_w
                scale_y = h / proc_h
                
                scaled_kps = self.current_pose.keypoints.copy()
                scaled_kps[:, 0] *= scale_x
                scaled_kps[:, 1] *= scale_y
                
                color = self.behavior_analyzer.visualizer.get_color(self.current_anomaly_score)
                self.behavior_analyzer.visualizer.draw_skeleton(frame, scaled_kps, self.current_pose.confidence, color)
                self.behavior_analyzer.visualizer.draw_info(frame, self.current_anomaly_score)
            except Exception as e:
                print(f"ERROR: Visualization error: {e}")

    def _update_frames(self, processed_frame, raw_frame):
        """Cập nhật bộ đệm khung hình một cách an toàn"""
        with self._frame_lock:
            self._last_frame = processed_frame.copy()
            self._raw_frame = raw_frame.copy()
            
    def get_connection_status(self) -> bool:
        """Kiểm tra xem camera có đang kết nối và hoạt động không"""
        if not self.cap or not self.cap.isOpened():
            return False
        
        # Kiểm tra thời gian khung hình cuối cùng
        if time.time() - self._last_successful_frame > self._connection_timeout:
            return False
            
        return True
        
    def force_reconnect(self):
        """Buộc kết nối lại camera"""
        print(f"INFO: Forcing reconnection for camera {self.source_id}")
        self._reconnect()
        
    def reset_fire_state(self):
        """Đặt lại trạng thái phát hiện cháy"""
        self.recent_fire_detections.clear()
        self.current_fire_boxes = []
        self.red_alert_mode_active = False
        self.yellow_alert_mode_active = False
        print(f"INFO: Fire state reset for camera {self.source_id}")

    def set_ir_enhancement(self, enabled: bool):
        """Bật/tắt tính năng tăng cường ảnh hồng ngoại"""
        self.ir_enhancement_enabled = enabled
        print(f"INFO: IR enhancement for camera {self.source_id} set to {enabled}")

    def release(self):
        """Giải phóng tài nguyên"""
        self.quit = True
        if self.cap:
            self.cap.release()
        print(f"INFO: Camera released: {self.source}")

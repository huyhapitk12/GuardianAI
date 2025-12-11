# core/camera.py
# =============================================================================
# MODULE XỬ LÝ CAMERA
# =============================================================================
# Module này xử lý video từ camera và chạy các bộ phát hiện:
# - Phát hiện người + nhận diện khuôn mặt
# - Phát hiện cháy/khói
# - Phân tích hành vi bất thường
# =============================================================================

import cv2
import time
import queue
import platform
import threading
import numpy as np
from collections import deque

from config import settings, AlertType
from core.detection import PersonTracker, FireFilter, BehaviorAnalyzer, FireTracker
from core.motion_detector import MotionDetector


# =============================================================================
# CLASS CAMERA - XỬ LÝ VIDEO TỪ MỘT CAMERA
# =============================================================================
class Camera:
    
    def __init__(self, source, person_alert_callback=None, fire_alert_callback=None, shared_model=None):
        """
        Khởi tạo camera
        
        source: URL camera, đường dẫn video, hoặc số (webcam ID)
        person_alert_callback: Callback khi phát hiện người
        fire_alert_callback: Callback khi phát hiện cháy
        shared_model: Model YOLO dùng chung (tiết kiệm RAM)
        """
        # Nguồn video
        self.source = source
        self.source_id = str(source)
        
        # Đối tượng VideoCapture của OpenCV
        self.cap = None
        
        # Cờ báo hiệu tắt
        self.quit = False
        
        # ----- Quản lý frame -----
        # Lock để tránh xung đột khi nhiều thread đọc/ghi frame
        self.frame_lock = threading.Lock()
        self.last_frame = None      # Frame đã xử lý (có vẽ box)
        self.raw_frame = None       # Frame gốc (không vẽ gì)
        self.frame_idx = 0          # Đếm số frame
        
        # ----- Quản lý kết nối -----
        self.reconnect_attempts = 0
        self.last_frame_time = time.time()
        self.ai_active_until = 0    # Thời điểm AI tắt nếu không có chuyển động
        
        # ----- Phát hiện chế độ IR (hồng ngoại/ban đêm) -----
        self.is_ir = False
        self.ir_history = deque(maxlen=30)  # Lưu lịch sử 30 frame
        
        # ----- Phát hiện cháy -----
        debug_fire = settings.get('camera.debug_fire_detection', False)
        self.fire_filter = FireFilter(debug=debug_fire)
        self.fire_boxes = []        # Vị trí các đám cháy
        self.fire_history = deque(maxlen=150)
        self.fire_tracker = FireTracker()
        
        # ----- Phát hiện người -----
        self.person_tracker = PersonTracker(shared_model=shared_model)
        
        # ----- Phân tích hành vi -----
        self.behavior_analyzer = None
        self.last_pose = None           # Lưu pose cuối cùng
        self.last_pose_time = 0         # Thời điểm pose cuối
        self.pose_hold_time = 0.3       # Giữ pose trong 0.3 giây để tránh nhấp nháy
        
        # ----- Phát hiện chuyển động -----
        # Dùng để tiết kiệm CPU: không có chuyển động = không cần chạy AI
        self.motion_detector = MotionDetector(
            motion_threshold=settings.get('camera.motion_threshold', 25.0),
            min_area=settings.get('camera.motion_min_area', 500)
        )
        
        # ----- Callback functions -----
        self.person_alert_callback = person_alert_callback
        self.fire_alert_callback = fire_alert_callback
        
        # ----- Queue cho xử lý đa luồng -----
        # maxsize=2: Tối đa 2 frame trong queue, tránh tồn đọng
        self.fire_queue = queue.Queue(maxsize=2)
        self.behavior_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=16)
        
        # Trạng thái detection
        self.last_detection_enabled = False
        
        # Kết nối camera
        self.init_capture()
    
    def init_capture(self):
        """Kết nối với camera"""
        try:
            # Nếu là webcam (số), thử nhiều backend
            if isinstance(self.source, int):
                backends = self.get_backends()
                for backend in backends:
                    self.cap = cv2.VideoCapture(self.source, backend)
                    if self.cap.isOpened():
                        break
            else:
                # URL hoặc file video
                self.cap = cv2.VideoCapture(self.source)
            
            # Cấu hình camera
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm độ trễ
                print(f"✅ Camera {self.source_id} đã kết nối!")
                
        except Exception as e:
            print(f"❌ Camera {self.source_id} kết nối thất bại: {e}")
    
    def get_backends(self):
        """Lấy danh sách backend phù hợp với hệ điều hành"""
        if platform.system() == 'Windows':
            return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
        elif platform.system() == 'Linux':
            return [cv2.CAP_V4L2, cv2.CAP_ANY]
        return [cv2.CAP_ANY]
    
    def read(self):
        """Đọc frame đã xử lý (có vẽ box, label)"""
        with self.frame_lock:
            if self.last_frame is not None:
                return True, self.last_frame.copy()
            return False, None
    
    def read_raw(self):
        """Đọc frame gốc (không xử lý)"""
        with self.frame_lock:
            if self.raw_frame is not None:
                return True, self.raw_frame.copy()
            return False, None
    
    def start_workers(self, fire_detector, face_detector, behavior_analyzer=None):
        """
        Khởi động các worker xử lý
        Worker = Thread chạy nền để xử lý từng tác vụ
        """
        # Gắn face detector vào person tracker
        self.person_tracker.set_face_detector(face_detector)
        self.person_tracker.initialize()
        
        # Gắn behavior analyzer
        self.behavior_analyzer = behavior_analyzer
        
        # Thread phát hiện cháy
        threading.Thread(
            target=self.fire_worker,
            args=(fire_detector,),
            daemon=True
        ).start()
        
        # Thread phân tích hành vi
        if self.behavior_analyzer:
            threading.Thread(
                target=self.behavior_worker,
                daemon=True
            ).start()
            print(f"✅ Behavior worker đã chạy cho camera {self.source_id}")
    
    def fire_worker(self, detector):
        """
        Worker phát hiện cháy
        Chạy trong thread riêng, lấy frame từ queue và phát hiện
        """
        while not self.quit:
            try:
                frame = self.fire_queue.get(timeout=1.0)
                detections = detector.detect(frame)
                if detections:
                    self.result_queue.put(('fire', detections))
            except queue.Empty:
                continue
    
    def behavior_worker(self):
        """
        Worker phân tích hành vi
        Phát hiện hành vi bất thường như: ngã, đánh nhau,...
        """
        skip_counter = 0
        skip_n = settings.get('behavior.process_every_n_frames', 3)
        
        while not self.quit:
            try:
                frame = self.behavior_queue.get(timeout=1.0)
                
                # Bỏ qua một số frame để giảm tải
                skip_counter += 1
                if skip_counter % skip_n != 0:
                    continue
                
                # Phân tích
                result = self.behavior_analyzer.process_frame(frame)
                
                # Kiểm tra có bất thường không
                if result.is_anomaly and self.behavior_analyzer.should_alert():
                    self.result_queue.put(('behavior', result, frame.copy()))
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Lỗi behavior worker: {e}")
    
    # =========================================================================
    # VÒNG LẶP XỬ LÝ CHÍNH
    # =========================================================================
    def process_loop(self, state_manager):
        """
        Vòng lặp chính xử lý video
        Chạy liên tục cho đến khi self.quit = True
        """
        # Tính interval giữa các frame dựa trên FPS mong muốn
        interval = 1.0 / settings.camera.target_fps
        last_time = 0
        cleanup_counter = 0
        
        while not self.quit:
            now = time.time()
            
            # Điều khiển tốc độ xử lý
            if now - last_time < interval:
                time.sleep(0.001)
                continue
            last_time = now
            
            # ----- Kiểm tra kết nối -----
            if not self.cap or not self.cap.isOpened():
                if not self.reconnect():
                    time.sleep(2.0)
                    continue
            
            # ----- Đọc frame -----
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if not self.check_health():
                    self.reconnect()
                continue
            
            self.last_frame_time = time.time()
            self.frame_idx += 1
            
            # ----- Dọn dẹp định kỳ -----
            cleanup_counter += 1
            if cleanup_counter >= 100:
                self.fire_filter.cleanup()
                cleanup_counter = 0
            
            # ----- Phát hiện chế độ IR (mỗi 10 frame) -----
            if self.frame_idx % 10 == 0:
                self.detect_ir(frame)
            
            # ----- Áp dụng bộ lọc màu -----
            frame = self.apply_color_filter(frame)
            
            # ----- Resize frame để xử lý -----
            # Frame nhỏ hơn = xử lý nhanh hơn
            proc_size = settings.camera.process_size
            small = cv2.resize(frame, tuple(proc_size))
            
            # Tính tỉ lệ scale để chuyển đổi tọa độ
            h, w = frame.shape[:2]
            scale_x = w / proc_size[0]
            scale_y = h / proc_size[1]
            
            # ----- Kiểm tra detection có bật không -----
            detection_enabled = state_manager.is_detection_enabled(self.source_id)
            self.last_detection_enabled = detection_enabled
            
            # ----- Phát hiện chuyển động -----
            has_motion = self.motion_detector.detect(small)
            
            # ===== LOGIC THÔNG MINH: Tiết kiệm CPU =====
            # 1. Có chuyển động → Bật AI 5 giây
            if has_motion:
                self.ai_active_until = now + 5.0
            
            # 2. Chỉ chạy AI khi cần
            should_run_ai = detection_enabled and (
                now < self.ai_active_until or self.frame_idx < 30
            )
            
            if should_run_ai:
                self.process_persons(small, frame, scale_x, scale_y)
                
                # 3. Nếu có người, giữ AI hoạt động (tránh mất track khi đứng yên)
                if self.person_tracker.has_tracks():
                    self.ai_active_until = now + 5.0
            
            # ----- Phát hiện cháy (luôn chạy vì quan trọng) -----
            if not self.fire_queue.full():
                self.fire_queue.put(small.copy())
                self.fire_queue.put(small.copy())
            
            # ----- Phân tích hành vi -----
            # [LOGIC] Chỉ chạy AI hành vi khi ĐÃ phát hiện người
            has_people = self.person_tracker.has_tracks()
            if detection_enabled and self.behavior_analyzer and not self.behavior_queue.full() and has_people:
                self.behavior_queue.put(small.copy())
            
            # ----- Xử lý kết quả từ các worker -----
            self.process_results(frame, scale_x, scale_y)
            
            # ----- Cập nhật frame hiển thị -----
            display = frame.copy()
            self.draw_overlays(display, detection_enabled)
            
            with self.frame_lock:
                self.last_frame = display
                self.raw_frame = frame.copy()
        
        # Dọn dẹp khi thoát
        self.release()
    
    def process_persons(self, small, full, scale_x, scale_y):
        """Xử lý phát hiện và tracking người"""
        try:
            # Lấy ngưỡng tin cậy
            threshold = settings.get('detection.person_confidence', 0.5)
            if self.is_ir:
                # IR mode: ngưỡng thấp hơn vì ảnh khó hơn
                threshold = settings.get('camera.infrared.person_detection_threshold', 0.45)
            
            # Phát hiện người
            detections = self.person_tracker.detect(small, threshold)
            
            # Cập nhật tracking
            if self.is_ir:
                # IR: Bỏ qua nhận diện khuôn mặt (không có màu)
                self.person_tracker.update(detections, full, scale_x, scale_y, skip_face_check=True)
            else:
                self.person_tracker.update(detections, full, scale_x, scale_y)
            
            # Kiểm tra cảnh báo
            for tid, alert_type, metadata in self.person_tracker.check_alerts():
                if self.person_alert_callback:
                    alert_frame = full.copy()
                    self.draw_overlays(alert_frame, True)
                    self.person_alert_callback(self.source_id, alert_frame, alert_type, metadata)
                    
        except Exception as e:
            print(f"Lỗi xử lý người: {e}")
    
    def process_results(self, frame, scale_x, scale_y):
        """Xử lý kết quả từ các worker queue"""
        self.fire_boxes = []
        
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                result_type = result[0]
                
                if result_type == 'fire':
                    detections = result[1]
                    self.handle_fire_detections(detections, frame, scale_x, scale_y)
                
                elif result_type == 'behavior':
                    behavior_result = result[1]
                    alert_frame = result[2]
                    self.handle_behavior_alert(behavior_result, alert_frame)
                    
        except queue.Empty:
            pass
    
    def handle_fire_detections(self, detections, frame, scale_x, scale_y):
        """
        Xử lý phát hiện cháy với hệ thống Red Alert Mode
        
        Yellow Alert: Nghi ngờ có cháy (cần xác nhận thêm)
        Red Alert: Chắc chắn có cháy (nguy hiểm!)
        """
        validated_dets = []
        
        for det in detections:
            bbox = det['bbox']
            
            # Validate với bộ lọc (loại bỏ false positive)
            if not self.fire_filter.validate(frame, bbox, self.is_ir):
                continue
            
            # Scale tọa độ về kích thước gốc
            x1, y1, x2, y2 = bbox
            scaled_bbox = (
                int(x1 * scale_x), int(y1 * scale_y),
                int(x2 * scale_x), int(y2 * scale_y)
            )
            
            self.fire_boxes.append(scaled_bbox)
            self.fire_history.append({'time': time.time(), **det})
            validated_dets.append(det)
        
        # Cập nhật fire tracker và kiểm tra điều kiện cảnh báo
        now = time.time()
        should_alert, is_yellow, is_red = self.fire_tracker.update(validated_dets, now)
        
        # Gửi cảnh báo nếu cần
        if should_alert and self.fire_alert_callback:
            alert_frame = frame.copy()
            self.draw_overlays(alert_frame, True)
            
            # Red = CRITICAL, Yellow = WARNING
            alert_type = AlertType.FIRE_CRITICAL if is_red else AlertType.FIRE_WARNING
            
            if is_red:
                print(f"🔴 RED ALERT - Camera {self.source_id}")
            elif is_yellow:
                print(f"🟡 Yellow Alert - Camera {self.source_id}")
            
            self.fire_alert_callback(self.source_id, alert_frame, alert_type)
    
    def handle_behavior_alert(self, result, frame):
        """Xử lý cảnh báo hành vi bất thường"""
        if self.person_alert_callback:
            # Vẽ visualization
            if self.behavior_analyzer:
                self.behavior_analyzer.draw_on_frame(frame, result)
            
            metadata = {
                'score': result.score,
                'timestamp': result.timestamp
            }
            self.person_alert_callback(
                self.source_id,
                frame,
                AlertType.ANOMALOUS_BEHAVIOR,
                metadata
            )
    
    def draw_overlays(self, frame, detection_enabled):
        """Vẽ các thông tin lên frame"""
        
        # ----- Vẽ box cháy (đỏ) -----
        for box in self.fire_boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
            cv2.putText(frame, "🔥 FIRE", (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # ----- Vẽ box người -----
        if detection_enabled:
            self.draw_persons_with_behavior(frame)
        
        # ----- Vẽ box chuyển động (Cyan) -----
        if hasattr(self.motion_detector, 'motion_boxes'):
            dh, dw = frame.shape[:2]
            ph, pw = settings.camera.process_size[1], settings.camera.process_size[0]
            sx = dw / pw
            sy = dh / ph
            
            for (mx1, my1, mx2, my2) in self.motion_detector.motion_boxes:
                final_x1 = int(mx1 * sx)
                final_y1 = int(my1 * sy)
                final_x2 = int(mx2 * sx)
                final_y2 = int(my2 * sy)
                
                cv2.rectangle(frame, (final_x1, final_y1), (final_x2, final_y2), (255, 255, 0), 1)
                cv2.putText(frame, "Motion", (final_x1, final_y1 - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # ----- Hiển thị chế độ IR -----
        if self.is_ir:
            cv2.putText(frame, "IR MODE", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def draw_persons_with_behavior(self, frame):
        """Vẽ box người kèm trạng thái hành vi"""
        tracks = self.person_tracker.tracks
        
        # Lấy behavior score
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
            
            # ===== XÁC ĐỊNH MÀU BOX =====
            if is_anomaly:
                color = (0, 0, 255)      # Đỏ - Bất thường
                status = "BAT THUONG"
            elif is_stranger:
                color = (0, 165, 255)    # Cam - Người lạ
                status = "Chua xac dinh"
            else:
                color = (0, 255, 0)      # Xanh lá - Người quen
                status = "Binh thuong"
            
            # ===== VẼ BOX =====
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # ===== TẠO LABEL =====
            if self.behavior_analyzer and self.behavior_analyzer.loaded:
                label = f"ID:{tid} {name} | {status} ({behavior_score:.2f})"
            else:
                label = f"ID:{tid} {name}"
            
            # ===== VẼ LABEL =====
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            label_y1 = max(0, y1 - text_h - 10)
            label_y2 = y1 - 2
            label_x2 = min(frame.shape[1], x1 + text_w + 8)
            
            cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
            cv2.putText(frame, label, (x1 + 4, label_y2 - 4), font, font_scale, (255, 255, 255), thickness)
        
        # ===== VẼ SKELETON =====
        # Lấy pose hiện tại từ analyzer
        current_pose = self.behavior_analyzer.current_pose if self.behavior_analyzer else None
        now = time.time()
        
        # Cập nhật last_pose nếu có pose mới
        if current_pose and current_pose.is_valid:
            self.last_pose = current_pose
            self.last_pose_time = now
        
        # Vẽ skeleton nếu có pose và chưa quá thời gian hold
        if self.last_pose and self.last_pose.bbox and (now - self.last_pose_time < self.pose_hold_time):
            # [LOGIC MỚI] Chỉ vẽ nếu skeleton nằm trong vùng của người đã phát hiện
            # Điều này giúp đồng bộ giữa Person Detection và Behavior Analysis
            should_draw = False
            
            # 1. Tính toán tọa độ skeleton trên frame hiển thị
            h, w = frame.shape[:2]
            proc_w, proc_h = settings.camera.process_size
            scale_x = w / proc_w
            scale_y = h / proc_h
            
            px1, py1, px2, py2 = self.last_pose.bbox
            # Box của skeleton (đã scale)
            sk_x1 = px1 * scale_x
            sk_y1 = py1 * scale_y
            sk_x2 = px2 * scale_x
            sk_y2 = py2 * scale_y
            
            # Tâm của skeleton
            sk_cx = (sk_x1 + sk_x2) / 2
            sk_cy = (sk_y1 + sk_y2) / 2
            
            # 2. Kiểm tra có trùng với người nào không
            for tid, track in tracks.items():
                tx1, ty1, tx2, ty2 = track.bbox
                
                # Kiểm tra tâm skeleton nằm trong box người
                # Mở rộng box người một chút (margin) để bắt dính tốt hơn
                margin = 50 
                if (tx1 - margin <= sk_cx <= tx2 + margin) and \
                   (ty1 - margin <= sk_cy <= ty2 + margin):
                    should_draw = True
                    break
            
            if should_draw:
                self.draw_skeleton_only(frame, is_anomaly, self.last_pose)

    def draw_skeleton_only(self, frame, is_anomaly, pose):
        """
        Vẽ skeleton (bộ xương) của người
        
        pose: PoseResult chứa keypoints đã ở tọa độ process_size
        """
        if not pose or not pose.is_valid:
            return
        
        # Scale keypoints từ process_size về kích thước frame hiển thị
        h, w = frame.shape[:2]
        proc_w, proc_h = settings.camera.process_size
        
        scale_x = w / proc_w
        scale_y = h / proc_h
        
        # Copy và scale keypoints
        scaled_kps = pose.keypoints.copy()
        scaled_kps[:, 0] *= scale_x
        scaled_kps[:, 1] *= scale_y
        
        # Màu theo trạng thái
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)
        
        # Các đường nối skeleton (theo format COCO)
        SKELETON = [
            (0, 1), (0, 2), (1, 3), (2, 4),      # Đầu
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Tay
            (5, 11), (6, 12), (11, 12),          # Thân
            (11, 13), (13, 15), (12, 14), (14, 16)   # Chân
        ]
        
        # Vẽ xương
        for i, j in SKELETON:
            if i < len(pose.confidence) and j < len(pose.confidence):
                if pose.confidence[i] > 0.3 and pose.confidence[j] > 0.3:
                    pt1 = tuple(scaled_kps[i].astype(int))
                    pt2 = tuple(scaled_kps[j].astype(int))
                    cv2.line(frame, pt1, pt2, color, 2)
        
        # Vẽ khớp
        for pt, conf in zip(scaled_kps, pose.confidence):
            if conf > 0.3:
                center = tuple(pt.astype(int))
                cv2.circle(frame, center, 5, color, -1)
                cv2.circle(frame, center, 5, (255, 255, 255), 1)
    
    def detect_ir(self, frame):
        """
        Phát hiện chế độ IR (hồng ngoại/ban đêm)
        
        Camera IR chỉ có đen trắng, không có màu.
        Khi camera chuyển sang ban đêm, cần điều chỉnh các ngưỡng.
        """
        # Lấy mẫu (sample) để tính nhanh
        sample = frame[::10, ::10]
        
        # Tách kênh màu
        b, g, r = cv2.split(sample.astype(np.float32))
        
        # Tính trung bình và độ lệch chuẩn
        means = [np.mean(r), np.mean(g), np.mean(b)]
        std = np.std(means)
        ratio = min(means) / max(means) if max(means) > 0 else 1.0
        
        # Tính độ bão hòa
        hsv = cv2.cvtColor(sample.astype(np.uint8), cv2.COLOR_BGR2HSV)
        sat = np.mean(hsv[:, :, 1])
        
        # IR: Các kênh màu gần bằng nhau + độ bão hòa thấp
        is_ir = std < 2.0 and ratio > 0.98 and sat < 10
        self.ir_history.append(is_ir)
        
        # Cần đủ lịch sử để quyết định
        if len(self.ir_history) >= 10:
            ir_ratio = sum(self.ir_history) / len(self.ir_history)
            new_mode = ir_ratio >= 0.7
            
            # Thông báo khi chuyển chế độ
            if new_mode != self.is_ir:
                self.is_ir = new_mode
                mode_name = 'IR (Ban đêm)' if new_mode else 'RGB (Ban ngày)'
                print(f"📷 Camera {self.source_id}: Chuyển sang chế độ {mode_name}")
                if new_mode:
                    print(f"   → Tắt nhận diện khuôn mặt (ảnh đen trắng)")
    
    def apply_color_filter(self, frame):
        """Áp dụng bộ lọc màu theo chế độ"""
        if self.is_ir:
            # Chuyển sang grayscale để xử lý thống nhất
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Trả về frame gốc
        return frame
    
    def reconnect(self):
        """Thử kết nối lại camera"""
        self.reconnect_attempts += 1
        
        max_attempts = settings.get('camera.max_reconnect_attempts', 10)
        if self.reconnect_attempts > max_attempts:
            print(f"❌ Camera {self.source_id}: Đã thử {max_attempts} lần, dừng kết nối lại")
            return False
        
        print(f"Đang kết nối lại camera {self.source_id}... (lần {self.reconnect_attempts}/{max_attempts})")
        
        if self.cap:
            self.cap.release()
        
        time.sleep(2.0)
        self.init_capture()
        
        if self.cap and self.cap.isOpened():
            self.reconnect_attempts = 0
            return True
        
        return False
    
    def check_health(self):
        """Kiểm tra camera còn hoạt động không"""
        return time.time() - self.last_frame_time < 10
    
    def get_connection_status(self):
        """Lấy trạng thái kết nối"""
        return self.cap is not None and self.cap.isOpened() and self.check_health()
    
    def has_active_threat(self):
        """
        Kiểm tra có mối nguy hiểm đang hoạt động không
        Dùng để quyết định có kéo dài thời gian ghi video không
        """
        # 1. Kiểm tra cháy
        if self.fire_tracker.is_red_alert or self.fire_tracker.is_yellow_alert:
            return True
        
        # 2. Kiểm tra người lạ
        if self.person_tracker.has_active_threats():
            return True
        
        # 3. Kiểm tra hành vi bất thường
        if self.behavior_analyzer:
            if self.behavior_analyzer.current_score >= self.behavior_analyzer.threshold:
                return True
        
        return False
    
    def get_infrared_status(self):
        """Lấy trạng thái IR"""
        return self.is_ir
    
    def force_reconnect(self):
        """Bắt buộc kết nối lại"""
        self.reconnect_attempts = 0
        self.reconnect()
    
    def release(self):
        """Giải phóng tài nguyên"""
        self.quit = True
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.behavior_analyzer:
            self.behavior_analyzer.close()
            self.behavior_analyzer = None
        print(f"Camera {self.source_id} đã giải phóng")

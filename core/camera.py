# Xử lý video từ camera & chạy các bộ phát hiện (người, cháy, khói)
import cv2
import time
import queue
import platform
import threading
import numpy as np
from collections import deque

from config import settings, AlertType
from core.detection import PersonTracker, FireFilter, FireTracker, FallDetector
from core.motion_detector import MotionDetector


# Class xử lý video từ 1 camera
class Camera:
    
    def __init__(self, source, person_alert_callback=None, fire_alert_callback=None, fall_alert_callback=None, shared_model=None):
        # Nguồn video
        self.source = source
        self.source_id = str(source)
        
        # Đối tượng VideoCapture
        self.cap = None
        
        # Cờ báo hiệu tắt
        self.quit = False
        
        # Quản lý frame và lock
        self.frame_lock = threading.Lock()
        self.last_frame = None      # Frame đã xử lý
        self.raw_frame = None       # Frame gốc
        self.frame_idx = 0
        
        self.reconnect_attempts = 0
        self.last_frame_time = time.time()
        self.ai_active_until = 0    # Thời điểm AI tắt nếu không có chuyển động
        
        # Chế độ IR
        self.is_ir = False
        self.ir_history = deque(maxlen=30)  # Lịch sử 30 frame
        self.ir_manual_override = None  # None = auto, True/False = manual
        
        # Phát hiện cháy
        debug_fire = settings.get('camera.debug_fire_detection', False)
        self.fire_filter = FireFilter(debug=debug_fire)
        self.fire_boxes = []        # Vị trí đám cháy
        self.fire_history = deque(maxlen=150)
        self.fire_tracker = FireTracker()
        
        # Phát hiện người
        self.person_tracker = PersonTracker(shared_model=shared_model)
        
        # Phát hiện chuyển động để tiết kiệm CPU
        self.motion_detector = MotionDetector(
            motion_threshold=settings.get('camera.motion_threshold', 25.0),
            min_area=settings.get('camera.motion_min_area', 500)
        )
        
        # Callback cảnh báo
        self.person_alert_callback = person_alert_callback
        self.fire_alert_callback = fire_alert_callback
        self.fall_alert_callback = fall_alert_callback
        
        # Phát hiện té ngã
        self.fall_detector = None
        self.is_fall_detected = False
        self.fall_prob = 0.0
        
        # Queue xử lý đa luồng
        self.fire_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=16)
        
        # Trạng thái detection
        self.last_detection_enabled = False
        
        # Ghi đè cài đặt riêng cho camera (None = dùng cài đặt chung)
        self.face_enabled = None
        
        # Kết nối camera
        self.init_capture()
    
    # Kết nối camera
    def init_capture(self):
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
            print(f"[OK] Camera {self.source_id} đã kết nối!")
    
    # Lấy backend video phù hợp OS
    def get_backends(self):
        if platform.system() == 'Windows':
            return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
        elif platform.system() == 'Linux':
            return [cv2.CAP_V4L2, cv2.CAP_ANY]
        return [cv2.CAP_ANY]
    
    # Đọc frame đã xử lý
    def read(self):
        with self.frame_lock:
            if self.last_frame is not None:
                return True, self.last_frame.copy()
            return False, None
    
    # Đọc frame gốc
    def read_raw(self):
        with self.frame_lock:
            if self.raw_frame is not None:
                return True, self.raw_frame.copy()
            return False, None
    
    # Khởi động các worker thread
    def start_workers(self, fire_detector, face_detector):
        # Gắn face detector vào person tracker
        self.person_tracker.set_face_detector(face_detector)
        self.person_tracker.initialize()
        
        # Khởi tạo Fall Detector
        self.fall_detector = FallDetector()
        print(f"[OK] Camera {self.source_id}: Fall Detector đã sẵn sàng")
        
        # Thread phát hiện cháy
        threading.Thread(
            target=self.fire_worker,
            args=(fire_detector,),
            daemon=True
        ).start()
    
    # Worker phát hiện cháy (chạy thread riêng)
    def fire_worker(self, detector):
        while not self.quit:
            if not self.fire_queue.empty():
                frame = self.fire_queue.get()
                detections = detector.detect(frame)
                if detections:
                    self.result_queue.put(('fire', detections))
            else:
                time.sleep(0.1)
    
    # Vòng lặp chính xử lý video
    def process_loop(self, state_manager):
        # Tính interval theo FPS mục tiêu
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
            
            # Kiểm tra kết nối
            if not self.cap or not self.cap.isOpened():
                if not self.reconnect():
                    time.sleep(2.0)
                    continue
            
            # Đọc frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if not self.check_health():
                    self.reconnect()
                continue
            
            self.last_frame_time = time.time()
            self.frame_idx += 1
            
            # Dọn dẹp định kỳ
            cleanup_counter += 1
            if cleanup_counter >= 100:
                self.fire_filter.cleanup()
                cleanup_counter = 0
            
            # Phát hiện chế độ IR (mỗi 10 frame)
            if self.frame_idx % 10 == 0:
                self.detect_ir(frame)
            
            # Áp dụng bộ lọc màu
            frame = self.apply_color_filter(frame)
            
            # Thay đổi kích thước giữ tỷ lệ khung hình để xử lý nhanh hơn
            proc_w, proc_h = settings.camera.process_size
            h, w = frame.shape[:2]
            
            # Tính tỷ lệ để vừa với kích thước xử lý
            scale = min(proc_w / w, proc_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            small = cv2.resize(frame, (new_w, new_h))
            
            # Tỷ lệ cho toạ độ
            scale_x = w / new_w
            scale_y = h / new_h
            
            # Kiểm tra xem phát hiện có bật không
            detection_enabled = state_manager.is_detection_enabled(self.source_id)
            self.last_detection_enabled = detection_enabled
            
            # Phát hiện chuyển động
            has_motion = self.motion_detector.detect(small)
            
            # Logic tiết kiệm CPU: Có chuyển động -> chạy AI 5s
            if has_motion:
                self.ai_active_until = now + 5.0
            
            # Chỉ chạy AI khi cần (Bật nhận diện khuôn mặt hoặc có chuyển động)
            face_enabled = self.face_enabled if self.face_enabled is not None else settings.get('detection.face_recognition_enabled', True)
            
            should_run_ai = detection_enabled and face_enabled and (
                now < self.ai_active_until or self.frame_idx < 30
            )
            
            if should_run_ai:
                self.process_persons(small, frame, scale_x, scale_y, face_enabled)
                
                # Duy trì AI hoạt động nếu đang có người
                if self.person_tracker.has_tracks():
                    self.ai_active_until = now + 5.0
                    self.process_fall(frame, scale_x, scale_y)
            
            # Phát hiện cháy (luôn chạy)
            if not self.fire_queue.full():
                self.fire_queue.put(small.copy())
            
            # Xử lý kết quả từ các worker
            self.process_results(frame, scale_x, scale_y)
            
            # Cập nhật frame hiển thị
            display = frame.copy()
            self.draw_overlays(display, detection_enabled, scale_x, scale_y)
            
            with self.frame_lock:
                self.last_frame = display
                self.raw_frame = frame.copy()
        
        # Dọn dẹp khi thoát
        self.release()
    
    # Xử lý phát hiện và tracking người
    def process_persons(self, small, full, scale_x, scale_y, face_enabled=True):
        # Lấy ngưỡng tin cậy
        threshold = settings.get('detection.person_confidence_threshold', 0.5)
        if self.is_ir:
            # IR mode: ngưỡng thấp hơn vì ảnh khó hơn
            threshold = settings.get('camera.infrared.person_detection_threshold', 0.45)
        
        # Phát hiện người
        detections = self.person_tracker.detect(small, threshold)
        
        # Cập nhật tracking
        # Bỏ qua kiểm tra khuôn mặt nếu: Chế độ IR HOẶC Nhận diện khuôn mặt bị tắt
        skip_face = self.is_ir or not face_enabled
        
        self.person_tracker.update(detections, full, scale_x, scale_y, skip_face_check=skip_face)
        
        # Kiểm tra cảnh báo
        for tid, alert_type, metadata in self.person_tracker.check_alerts():
            if self.person_alert_callback:
                alert_frame = full.copy()
                self.draw_overlays(alert_frame, True, scale_x, scale_y)
                self.person_alert_callback(self.source_id, alert_frame, alert_type, metadata)
    
    # Xử lý té ngã (khi có người)
    def process_fall(self, frame, scale_x, scale_y):
        if not self.fall_detector:
            return
        
        # Lấy bbox của người đầu tiên từ person tracker (tránh chạy YOLOX lần 2)
        tracks = self.person_tracker.tracks
        bbox = None
        if tracks:
            first_track = next(iter(tracks.values()))
            bbox = first_track.bbox  # (x1, y1, x2, y2)
        
        # Cập nhật bộ phát hiện té ngã với khung hình và hộp giới hạn
        self.fall_detector.update(frame, bbox=bbox)
        
        # Kiểm tra trạng thái té ngã
        is_fall, prob = self.fall_detector.check_fall()
        self.is_fall_detected = is_fall
        self.fall_prob = prob
        
        # Gửi cảnh báo nếu phát hiện té ngã
        if is_fall and self.fall_alert_callback:
            alert_frame = frame.copy()
            self.draw_overlays(alert_frame, True, scale_x, scale_y)
            self.fall_alert_callback(self.source_id, alert_frame, AlertType.FALL)
            print(f"[ALERT] FALL DETECTED - Camera {self.source_id} (prob={prob:.2f})")
            
            # Reset để không gửi liên tục
            self.fall_detector.reset()
    
    # Xử lý kết quả từ các worker queue
    def process_results(self, frame, scale_x, scale_y):
        self.fire_boxes = []
        
        while not self.result_queue.empty():
            result = self.result_queue.get_nowait()
            result_type = result[0]
            
            if result_type == 'fire':
                detections = result[1]
                self.handle_fire_detections(detections, frame, scale_x, scale_y)
    
    # Xử lý kết quả cháy (Cảnh báo Đỏ/Vàng)
    def handle_fire_detections(self, detections, frame, scale_x, scale_y):
        
        validated_dets = []
        
        for det in detections:
            bbox = det['bbox']
            
            # Xác thực với bộ lọc (loại bỏ dương tính giả)
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
            self.draw_overlays(alert_frame, True, scale_x, scale_y)
            
            # Red = CRITICAL, Yellow = WARNING
            alert_type = AlertType.FIRE_CRITICAL if is_red else AlertType.FIRE_WARNING
            
            if is_red:
                print(f"[CRITICAL] RED ALERT - Camera {self.source_id}")
            elif is_yellow:
                print(f"[WARN] Yellow Alert - Camera {self.source_id}")
            
            self.fire_alert_callback(self.source_id, alert_frame, alert_type)
    
    # Vẽ thông tin lên khung hình
    def draw_overlays(self, frame, detection_enabled, scale_x=1.0, scale_y=1.0):
        
        # Vẽ box cháy (đỏ)
        for box in self.fire_boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
            
            # Label chính
            cv2.putText(frame, "FIRE", (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Hiển thị tốc độ phát triển (Growth)
            growth = self.fire_tracker.current_growth_rate
            if growth > 1.05: # Chỉ hiện khi tăng > 5%
                text = f"Growth: +{(growth-1)*100:.0f}%"
                cv2.putText(frame, text, (box[0], box[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Vẽ box người
        if detection_enabled:
            self.draw_persons(frame, scale_x, scale_y)
        
        # Vẽ box chuyển động
        if hasattr(self.motion_detector, 'motion_boxes'):
            sx = scale_x
            sy = scale_y
            
            for (mx1, my1, mx2, my2) in self.motion_detector.motion_boxes:
                final_x1 = int(mx1 * sx)
                final_y1 = int(my1 * sy)
                final_x2 = int(mx2 * sx)
                final_y2 = int(my2 * sy)
                
                cv2.rectangle(frame, (final_x1, final_y1), (final_x2, final_y2), (255, 255, 0), 1)
                cv2.putText(frame, "Motion", (final_x1, final_y1 - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Hiển thị chế độ IR
        if self.is_ir:
            cv2.putText(frame, "IR MODE", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Vẽ khung xương nếu có bộ phát hiện té ngã
        if self.fall_detector and self.fall_detector.last_kps is not None:
            self.fall_detector.draw_skeleton_overlay(frame)
        
        # Hiển thị phát hiện té ngã
        if self.is_fall_detected:
            # Vẽ chữ FALL DETECTED lớn ở giữa màn hình
            h, w = frame.shape[:2]
            text = "FALL DETECTED!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            x = (w - text_w) // 2
            y = 60
            
            # Vẽ nền đỏ
            cv2.rectangle(frame, (x - 10, y - text_h - 10), (x + text_w + 10, y + 10), (0, 0, 200), -1)
            cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)
            
            # Hiển thị xác suất
            prob_text = f"Prob: {self.fall_prob:.2f}"
            cv2.putText(frame, prob_text, (x, y + 35), font, 0.7, (0, 0, 255), 2)
    
    # Vẽ box người và tên
    def draw_persons(self, frame, scale_x=1.0, scale_y=1.0):
        tracks = self.person_tracker.tracks
        
        for tid, track in tracks.items():
            x1, y1, x2, y2 = map(int, track.bbox)
            
            # Xác định tên hiển thị
            face_enabled = self.face_enabled if self.face_enabled is not None else settings.get('detection.face_recognition_enabled', True)
            
            # Xác định tên và trạng thái
            if face_enabled:
                name = track.confirmed_name or track.name
                is_stranger = (name == "Stranger")
            else:
                name = "Person"
                is_stranger = False
            
            # Xác định màu box
            if is_stranger:
                color = (0, 165, 255)    # Cam - Người lạ
            else:
                color = (0, 255, 0)      # Xanh lá - Người quen
            
            # Vẽ box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Tạo label
            label = f"ID:{tid} {name}"
            
            # Vẽ label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            label_y1 = max(0, y1 - text_h - 10)
            label_y2 = y1 - 2
            label_x2 = min(frame.shape[1], x1 + text_w + 8)
            
            cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
            cv2.putText(frame, label, (x1 + 4, label_y2 - 4), font, font_scale, (255, 255, 255), thickness)
    
    # Phát hiện IR (camera đen trắng/ban đêm)
    def detect_ir(self, frame):
        if self.ir_manual_override is not None:
             return
        
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
                print(f"[INFO] Camera {self.source_id}: Chuyển sang chế độ {mode_name}")
                if new_mode:
                    print(f"   → Tắt nhận diện khuôn mặt (ảnh đen trắng)")
    
    # Áp dụng bộ lọc màu theo chế độ
    def apply_color_filter(self, frame):
        if self.is_ir:
            # Chuyển sang grayscale để xử lý thống nhất
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Trả về frame gốc
        return frame
    
    # Thử kết nối lại camera
    def reconnect(self):
        self.reconnect_attempts += 1
        
        max_attempts = settings.get('camera.max_reconnect_attempts', 10)
        if self.reconnect_attempts > max_attempts:
            print(f"[ERR] Camera {self.source_id}: Đã thử {max_attempts} lần, dừng kết nối lại")
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
    
    # Kiểm tra camera còn hoạt động không
    def check_health(self):
        return time.time() - self.last_frame_time < 10
    
    # Lấy trạng thái kết nối
    def get_connection_status(self):
        return self.cap is not None and self.cap.isOpened() and self.check_health()
    
    # Kiểm tra mối nguy (cháy, người lạ) để quyết định ghi video
    def has_active_threat(self):
        if self.fire_tracker.get_is_red_alert() or self.fire_tracker.get_is_yellow_alert():
            return True
        
        # 2. Kiểm tra người lạ
        if self.person_tracker.has_active_threats():
            return True
        
        return False
    
    # Lấy trạng thái IR
    def get_infrared_status(self):
        return self.is_ir
    
    # Bật/tắt chế độ IR thủ công (tắt tự động phát hiện)
    def set_ir_enhancement(self, enabled):
        self.ir_manual_override = enabled  # Đánh dấu đang dùng manual
        self.is_ir = enabled
        mode = "IR (Ban đêm)" if enabled else "RGB (Ban ngày)"
        print(f"[INFO] Camera {self.source_id}: Chuyển sang chế độ {mode} (manual)")
    
    # Đặt lại về tự động phát hiện IR
    def reset_ir_auto(self):
        self.ir_manual_override = None
        self.ir_history.clear()
        print(f"[INFO] Camera {self.source_id}: Reset về auto detect IR")
    
    # Bắt buộc kết nối lại
    def force_reconnect(self):
        self.reconnect_attempts = 0
        self.reconnect()
    
    # Giải phóng tài nguyên
    def release(self):
        self.quit = True
        
        # Giải phóng fall detector
        if self.fall_detector:
            self.fall_detector.close()
            self.fall_detector = None
        
        if self.cap:
            self.cap.release()
            self.cap = None

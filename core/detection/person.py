# core/detection/person.py
# =============================================================================
# MODULE PHÁT HIỆN VÀ THEO DÕI NGƯỜI - PERSON DETECTION & TRACKING
# =============================================================================
# Module này thực hiện 3 nhiệm vụ chính:
# 1. Phát hiện người trong video (dùng YOLO)
# 2. Theo dõi người qua các frame (tracking)
# 3. Nhận diện khuôn mặt để xác định người quen/lạ
# =============================================================================

import time
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from ultralytics import YOLO
import supervision as sv
from trackers import SORTTracker
from config import settings, AlertType


# =============================================================================
# CLASS TRACK - LƯU THÔNG TIN MỘT NGƯỜI ĐANG THEO DÕI
# =============================================================================
# Mỗi người trong camera được gán một Track
# Track lưu: vị trí, tên, trạng thái nhận diện,...
# =============================================================================
class Track:
    
    def __init__(self, bbox):
        # bbox = bounding box = hình chữ nhật bao quanh người
        # Dạng (x1, y1, x2, y2): góc trên-trái và góc dưới-phải
        self.bbox = bbox
        
        # Tên người (mặc định là "Stranger" = người lạ)
        self.name = "Stranger"
        
        # Khoảng cách so với khuôn mặt đã biết (càng nhỏ càng giống)
        self.distance = float('inf')  # inf = vô cực
        
        # Thời điểm nhìn thấy lần cuối
        self.last_seen = 0
        
        # Số lần phát hiện khuôn mặt khớp
        self.face_hits = 0
        
        # Thời điểm kiểm tra khuôn mặt lần cuối
        self.last_face_check = 0
        
        # Tên đã xác nhận (sau khi đủ số lần face_hits)
        self.confirmed_name = None
        
        # Đã gửi cảnh báo người quen chưa
        self.alert_sent = False
        
        # Đã gửi cảnh báo người lạ chưa
        self.stranger_alert_sent = False
        
        # Số frame không nhận diện được
        self.frames_unidentified = 0
        
        # ID nhận diện lại (ReID) - giúp nhớ người khi họ đi ra và vào lại
        self.reid_id = None
        self.reid_embedding = None


# =============================================================================
# CLASS PERSON TRACKER - THEO DÕI VÀ NHẬN DIỆN NGƯỜI
# =============================================================================
# Đây là class chính để phát hiện và theo dõi người
# =============================================================================
class PersonTracker:
    
    def __init__(self, face_detector=None, shared_model=None):
        # Model YOLO để phát hiện người
        self.model = shared_model
        self.owns_model = shared_model is None
        
        # Bộ theo dõi SORT (Simple Online and Realtime Tracking)
        self.sort = None
        
        # Bộ nhận diện khuôn mặt
        self.face_detector = face_detector
        
        # Dictionary lưu các Track đang theo dõi
        # Key: ID của track, Value: đối tượng Track
        self.tracks = {}
        
        # ID tiếp theo sẽ gán cho track mới
        self.next_id = 0
        
        # ----- ReID: Nhận diện lại người -----
        # Khi một người đi ra khỏi camera rồi quay lại
        # ReID giúp nhận ra đó là cùng một người
        self.reid_memory = {}
        self.next_reid = 1
        self.alerted_reids = set()

    def initialize(self):
        """
        Khởi tạo tracker
        Trả về: True nếu thành công
        """
        # Nếu đã có model từ bên ngoài
        if self.model is not None:
            if SORTTracker:
                self.sort = SORTTracker()
            print("✅ Person Tracker đã sẵn sàng!")
            return True
        
        # Tải model YOLO mới
        if not YOLO:
            return False
        
        try:
            # Lấy cấu hình model
            yolo_size = settings.get('models.yolo_size', 'medium').lower()
            yolo_format = settings.get('models.yolo_format', 'openvino')
            path = settings.get_yolo_model_path('person', yolo_size, yolo_format)
            
            # Tải model
            self.model = YOLO(str(path), verbose=False)
            
            # Khởi tạo SORT tracker
            if SORTTracker:
                self.sort = SORTTracker()
            
            print("✅ Person Tracker đã khởi tạo!")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khởi tạo Person Tracker: {e}")
            return False
    
    def get_face_detector(self):
        """Lấy bộ nhận diện khuôn mặt"""
        return self.face_detector
    
    def set_face_detector(self, detector):
        """Đặt bộ nhận diện khuôn mặt"""
        self.face_detector = detector
    
    def detect(self, frame, conf=None):
        """
        Phát hiện người trong frame
        
        frame: Hình ảnh cần kiểm tra
        conf: Ngưỡng tin cậy (tùy chọn)
        
        Trả về: Danh sách bounding box của người phát hiện được
        """
        if not self.model:
            return []
        
        # Lấy ngưỡng tin cậy
        threshold = conf or settings.get('detection.person_confidence', 0.5)
        yolo_format = settings.get('models.yolo_format', 'openvino')
        
        try:
            # Chạy YOLO, classes=0 nghĩa là chỉ tìm "person" (class 0 trong COCO)
            if yolo_format == 'openvino':
                results = self.model(frame, conf=threshold, classes=0, verbose=False)[0]
            else:
                results = self.model(frame, conf=threshold, classes=0, verbose=False, device='cpu')[0]
            
            # Trích xuất bounding box
            if hasattr(results, 'boxes'):
                return [tuple(map(float, b.xyxy[0].tolist())) for b in results.boxes]
            return []
            
        except Exception as e:
            print(f"⚠️ Lỗi phát hiện người: {e}")
            return []
    
    def update(self, detections, frame, scale_x=1.0, scale_y=1.0, skip_face_check=False):
        """
        Cập nhật tracker với các detection mới
        
        detections: Danh sách bounding box người phát hiện được
        frame: Frame gốc (kích thước lớn)
        scale_x, scale_y: Tỉ lệ scale từ frame nhỏ lên frame lớn
        skip_face_check: Bỏ qua kiểm tra khuôn mặt (dùng khi camera IR)
        
        Trả về: Dictionary các track đang active
        """
        now = time.time()
        
        # Scale bounding box từ frame xử lý nhỏ lên frame gốc lớn
        scaled = []
        for x1, y1, x2, y2 in detections:
            scaled.append((
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y)
            ))
        
        # Cập nhật tracker
        # Ưu tiên dùng SORT nếu có, nếu không thì dùng IOU matching
        if self.sort and scaled and sv:
            self.update_sort(scaled, frame, now)
        else:
            self.update_iou(scaled, now)
        
        # Kiểm tra khuôn mặt (nếu cho phép)
        if self.face_detector and not skip_face_check:
            for tid, track in self.tracks.items():
                self.check_face(tid, track, frame, now)
        
        # Xóa các track quá cũ (timeout)
        timeout = settings.get('tracker.timeout_seconds', 30)
        self.tracks = {k: v for k, v in self.tracks.items() 
                        if now - v.last_seen < timeout}
        
        return self.tracks
    
    def update_sort(self, detections, frame, now):
        """
        Cập nhật bằng SORT tracker
        SORT: Simple Online and Realtime Tracking
        Hoạt động: Dùng Kalman Filter để dự đoán vị trí + Hungarian Algorithm để match
        """
        try:
            # Chuyển sang định dạng supervision cần
            xyxy = np.array(detections, dtype=float)
            sv_dets = sv.Detections(xyxy=xyxy, confidence=np.ones(len(detections)))
            
            # Chạy SORT
            tracked = self.sort.update(sv_dets)
            
            # Cập nhật tracks
            if hasattr(tracked, 'xyxy') and hasattr(tracked, 'tracker_id'):
                for i, tid in enumerate(tracked.tracker_id):
                    bbox = tuple(map(int, tracked.xyxy[i]))
                    self.update_track(tid, bbox, now)
                    
        except Exception:
            # Nếu SORT lỗi, fallback về IOU matching
            self.update_iou(detections, now)
    
    def update_iou(self, detections, now):
        """
        Cập nhật bằng IOU matching (phương pháp đơn giản)
        
        IOU = Intersection over Union (Tỉ lệ giao / hợp)
        Giá trị từ 0 đến 1:
        - 0 = không trùng chút nào
        - 1 = trùng hoàn toàn
        
        Ý tưởng: Box ở frame hiện tại khớp với box nào ở frame trước có IOU cao nhất
        """
        # Nếu chưa có track nào, tạo mới hết
        if not self.tracks:
            for bbox in detections:
                track = Track(bbox)
                track.last_seen = now
                self.tracks[self.next_id] = track
                self.next_id += 1
            return
        
        # Match các detection mới với track cũ
        track_ids = list(self.tracks.keys())
        matched_dets = set()  # Các detection đã được match
        
        for tid in track_ids:
            track = self.tracks[tid]
            best_iou, best_idx = 0, -1
            
            # Tìm detection khớp nhất với track này
            for i, det in enumerate(detections):
                if i in matched_dets:
                    continue  # Đã match rồi, bỏ qua
                    
                iou = self.calc_iou(track.bbox, det)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            
            # Nếu IOU đủ cao, cập nhật track
            iou_thresh = settings.get('detection.iou_threshold', 0.3)
            if best_iou > iou_thresh:
                track.bbox = detections[best_idx]
                track.last_seen = now
                matched_dets.add(best_idx)
        
        # Tạo track mới cho các detection không match được
        for i, bbox in enumerate(detections):
            if i not in matched_dets:
                track = Track(bbox)
                track.last_seen = now
                self.tracks[self.next_id] = track
                self.next_id += 1
    
    def update_track(self, tid, bbox, now):
        """Cập nhật hoặc tạo mới một track"""
        if tid not in self.tracks:
            track = Track(bbox)
            track.last_seen = now
            self.tracks[tid] = track
        else:
            self.tracks[tid].bbox = bbox
            self.tracks[tid].last_seen = now
    
    def check_face(self, tid, track, frame, now):
        """
        Kiểm tra khuôn mặt của một track
        Nếu khớp với người đã biết -> cập nhật tên
        """
        # Đã xác nhận rồi, không cần kiểm tra nữa
        if track.confirmed_name:
            return
        
        # Chờ cooldown giữa các lần kiểm tra
        cooldown = settings.get('tracker.face_recognition_cooldown', 1.0)
        if now - track.last_face_check < cooldown:
            return
        
        track.last_face_check = now
        
        # Cắt vùng chứa người từ frame
        x1, y1, x2, y2 = track.bbox
        h, w = frame.shape[:2]
        
        # Mở rộng thêm 10% để đảm bảo lấy hết khuôn mặt
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)
        
        crop = frame[int(crop_y1):int(crop_y2), int(crop_x1):int(crop_x2)]
        
        # Kiểm tra crop có đủ lớn không
        if crop.size == 0 or (crop.shape[0] < 20 or crop.shape[1] < 20):
            return
        
        # Tìm khuôn mặt trong vùng crop
        faces = self.face_detector.detect_faces(crop)
        if not faces:
            track.face_hits = max(0, track.face_hits - 1)
            return
        
        # Lấy khuôn mặt lớn nhất
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        # Cập nhật ReID
        self.update_reid(track, face.embedding, now)
        
        # Nhận diện khuôn mặt
        name, dist = self.face_detector.recognize(face.embedding)
        
        if name:
            track.face_hits += 1
            track.name = name
            track.distance = dist
        else:
            track.face_hits = max(0, track.face_hits - 1)
            if track.face_hits == 0:
                track.name = "Stranger"
    
    def update_reid(self, track, embedding, now):
        """
        Cập nhật ReID (Re-Identification)
        
        ReID giúp nhận ra cùng một người khi họ:
        - Đi ra khỏi camera rồi quay lại
        - Bị che khuất tạm thời
        
        Dùng embedding (vector đặc trưng) của khuôn mặt để so sánh
        """
        ttl = settings.get('reid.ttl_seconds', 30)  # Thời gian sống
        threshold = settings.get('reid.distance_threshold', 0.35)  # Ngưỡng khoảng cách
        
        # Xóa entry cũ
        self.reid_memory = {k: v for k, v in self.reid_memory.items() 
                            if now - v['last_seen'] < ttl}
        
        # Tìm match trong memory
        best_rid, best_dist = None, float('inf')
        for rid, info in self.reid_memory.items():
            # Tính khoảng cách cosine giữa 2 embedding
            dist = cosine(embedding, info['embedding'])
            if dist < best_dist:
                best_dist, best_rid = dist, rid
        
        if best_dist <= threshold:
            # Tìm thấy match -> cập nhật
            # Cập nhật embedding bằng trung bình có trọng số
            old = self.reid_memory[best_rid]['embedding']
            self.reid_memory[best_rid]['embedding'] = 0.8 * old + 0.2 * embedding
            self.reid_memory[best_rid]['last_seen'] = now
            track.reid_id = best_rid
        else:
            # Không tìm thấy -> tạo mới
            rid = self.next_reid
            self.next_reid += 1
            self.reid_memory[rid] = {'embedding': embedding, 'last_seen': now}
            track.reid_id = rid
    
    def check_alerts(self):
        """
        Kiểm tra xem có track nào cần gửi cảnh báo không
        
        Trả về: List các tuple (track_id, alert_type, metadata)
        """
        alerts = []
        
        # Lấy config
        known_confirm = settings.get('tracker.known_person_confirm_frames', 3)
        stranger_confirm = settings.get('tracker.stranger_confirm_frames', 30)
        
        for tid, track in self.tracks.items():
            # ----- Cảnh báo người quen -----
            # Điều kiện: nhận diện đủ số lần + chưa xác nhận + chưa gửi cảnh báo
            if (track.face_hits >= known_confirm and 
                not track.confirmed_name and 
                track.name != "Stranger" and 
                not track.alert_sent):
                
                track.confirmed_name = track.name
                track.alert_sent = True
                alerts.append((tid, AlertType.KNOWN_PERSON, 
                              {'name': track.name, 'distance': track.distance}))
            
            # ----- Cảnh báo người lạ -----
            # Điều kiện: chưa nhận diện được + đủ số frame + chưa gửi cảnh báo
            if not track.confirmed_name:
                track.frames_unidentified += 1
                
                if (track.frames_unidentified > stranger_confirm and 
                    not track.stranger_alert_sent):
                    
                    rid = track.reid_id
                    # Kiểm tra ReID để tránh gửi trùng cho cùng một người
                    if rid is None or rid not in self.alerted_reids:
                        track.stranger_alert_sent = True
                        if rid:
                            self.alerted_reids.add(rid)
                        alerts.append((tid, AlertType.STRANGER, {}))
        
        return alerts
    
    def draw(self, frame):
        """
        Vẽ các track lên frame
        
        Màu:
        - Xanh lá: Người quen đã nhận diện
        - Đỏ: Người lạ (đã gửi cảnh báo)
        - Vàng: Đang theo dõi (chưa xác định)
        """
        for tid, track in self.tracks.items():
            x1, y1, x2, y2 = map(int, track.bbox)
            
            # Xác định màu và label
            if track.confirmed_name and track.name != "Stranger":
                color = (0, 255, 0)  # Xanh lá
                label = track.name
            elif track.stranger_alert_sent:
                color = (0, 0, 255)  # Đỏ
                label = "STRANGER"
            else:
                color = (255, 255, 0)  # Vàng
                label = f"ID-{tid}"
            
            # Vẽ box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def calc_iou(self, box1, box2):
        """
        Tính IOU (Intersection over Union) giữa 2 box
        
        IOU = Diện tích giao / Diện tích hợp
        
        Ví dụ minh họa:
        ┌─────────┐
        │  Box1   │
        │    ┌────┼───┐
        └────┼────┘   │
            │  Box2  │
            └────────┘
        Phần giao là vùng chồng lấn giữa 2 box
        """
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        
        # Tọa độ vùng giao
        xi1 = max(x1, x1_)
        yi1 = max(y1, y1_)
        xi2 = min(x2, x2_)
        yi2 = min(y2, y2_)
        
        # Diện tích giao
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Diện tích từng box
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)
        
        # Diện tích hợp = Tổng 2 box - Phần giao (vì phần giao bị tính 2 lần)
        union = box1_area + box2_area - inter
        
        # IOU = Giao / Hợp
        return inter / union if union > 0 else 0
    
    def has_active_threats(self):
        """
        Kiểm tra có mối nguy hiểm đang hoạt động không
        (Người lạ chưa xử lý)
        """
        now = time.time()
        timeout = settings.get('tracker.timeout_seconds', 30)
        
        for track in self.tracks.values():
            # Bỏ qua track quá cũ
            if now - track.last_seen > timeout:
                continue
            
            # Có người lạ đã xác nhận
            if track.stranger_alert_sent:
                return True
        
        return False
    
    def has_tracks(self):
        """Kiểm tra có track nào đang active không"""
        return bool(self.tracks)

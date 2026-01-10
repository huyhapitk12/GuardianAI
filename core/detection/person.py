# Module phát hiện và theo dõi người
# 1. Detect người (YOLO)
# 2. Tracking (SORT/IOU)
# 3. Face ID

import time
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from ultralytics import YOLO
import supervision as sv
from trackers import SORTTracker
from config import settings, AlertType


# Class Track: Lưu thông tin người được theo dõi
class Track:
    # bbox format: (x1, y1, x2, y2)
    def __init__(self, bbox):
        self.bbox = bbox
        
        self.name = "Stranger"
        
    
        self.distance = float('inf')  # Khoảng cách so với khuôn mặt đã biết (càng nhỏ càng giống)
        self.last_seen = 0            # Thời điểm nhìn thấy lần cuối
        self.face_hits = 0            # Số lần phát hiện khuôn mặt khớp
        self.last_face_check = 0      # Thời điểm kiểm tra khuôn mặt lần cuối
        self.confirmed_name = None    # Tên đã xác nhận (sau khi đủ số lần face_hits)
        self.alert_sent = False       # Đã gửi cảnh báo người quen chưa
        self.stranger_alert_sent = False
        self.frames_unidentified = 0  # Số frame không nhận diện được
        
        # ID nhận diện lại (ReID) - giúp nhớ người khi họ đi ra và vào lại
        self.reid_id = None
        self.reid_embedding = None


# Class PersonTracker: Quản lý tracking và nhận diện
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
        
        # ReID memory: nhận diện lại người đã khuất bóng
        self.reid_memory = {}
        self.next_reid = 1
        self.alerted_reids = set()

    # Khởi tạo tracker
    def initialize(self):
        # Nếu đã có model từ bên ngoài
        if self.model is not None:
            if SORTTracker:
                self.sort = SORTTracker()
            print("[OK] Person Tracker đã sẵn sàng!")
            return True
        
        # Tải model YOLO mới
        if not YOLO:
            return False
        
        # Lấy cấu hình model (format tự động dựa trên device)
        yolo_size = settings.get('models.mode', 'Medium').lower()
        yolo_format = settings.get_optimal_yolo_format()
        
        # Đảm bảo model tồn tại (tự export nếu cần)
        path = settings.ensure_yolo_model('person', yolo_size)
        if not path:
            print("[ERR] Không thể tải model Person!")
            return False
        
        print(f"[INFO] Đang tải model Person: {path.name} (format: {yolo_format})")
        
        # Tải model
        self.model = YOLO(str(path), verbose=False)
        
        # Khởi tạo SORT tracker
        if SORTTracker:
            self.sort = SORTTracker()
        
        print("[OK] Person Tracker đã khởi tạo!")
        return True
    
    # Lấy bộ nhận diện khuôn mặt
    def get_face_detector(self):
        return self.face_detector
    
    # Đặt bộ nhận diện khuôn mặt
    def set_face_detector(self, detector):
        self.face_detector = detector
    
    # Detect người trong frame
    # Return: List of bboxes
    def detect(self, frame, conf=None):
        if not self.model:
            return []
        
        # Lấy ngưỡng tin cậy
        threshold = conf or settings.get('detection.person_confidence_threshold', 0.5)
        
        # Lấy format và device tự động
        yolo_format = settings.get_optimal_yolo_format()
        device = settings.get('models.device', 'cpu')
        
        # Chạy YOLO, classes=0 nghĩa là chỉ tìm "person" (class 0 trong COCO)
        if yolo_format == 'openvino':
            results = self.model(frame, conf=threshold, classes=0, verbose=False)[0]
        else:
            results = self.model(frame, conf=threshold, classes=0, verbose=False, device=device)[0]
        
        # Trích xuất bounding box
        if hasattr(results, 'boxes'):
            return [tuple(map(float, b.xyxy[0].tolist())) for b in results.boxes]
        return []
    
    # Cập nhật tracker với danh sách detection mới
    def update(self, detections, frame, scale_x=1.0, scale_y=1.0, skip_face_check=False):
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
    
    # Cập nhật dùng SORT (Kalman Filter + Hungarian)
    def update_sort(self, detections, frame, now):
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
    
    # Cập nhật bằng IOU matching (phương pháp đơn giản)
    # IOU = Intersection over Union (Tỉ lệ giao / hợp)
    # Giá trị từ 0 đến 1:
    # - 0 = không trùng chút nào
    # - 1 = trùng hoàn toàn
    def update_iou(self, detections, now):
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
    
    # Cập nhật hoặc tạo mới một track
    def update_track(self, tid, bbox, now):
        if tid not in self.tracks:
            track = Track(bbox)
            track.last_seen = now
            self.tracks[tid] = track
        else:
            self.tracks[tid].bbox = bbox
            self.tracks[tid].last_seen = now
    
    # Nhận diện khuôn mặt cho track nếu cần
    def check_face(self, tid, track, frame, now):
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
            
            # Xác nhận tên nếu đủ số lần nhận diện
            confirm_threshold = settings.get('tracker.known_person_confirm_frames', 4)
            if track.face_hits >= confirm_threshold and not track.confirmed_name:
                track.confirmed_name = track.name
                print(f"[OK] Đã xác nhận: {track.name}")
        else:
            # Giảm face_hits chậm hơn (chỉ giảm 0.5 thay vì 1)
            # Và chỉ reset về Stranger khi xuống âm nhiều
            track.face_hits = max(-3, track.face_hits - 0.5)
            if track.face_hits <= -3:
                track.name = "Stranger"
                track.face_hits = 0
    
    # Cập nhật ReID (Re-Identification)
    def update_reid(self, track, embedding, now):
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
    
    # Kiểm tra và tạo alert nếu cần
    def check_alerts(self):
        alerts = []
        
        # Lấy config
        known_confirm = settings.get('tracker.known_person_confirm_frames', 3)
        stranger_confirm = settings.get('tracker.stranger_confirm_frames', 30)
        
        for tid, track in self.tracks.items():
            # Cảnh báo người quen
            # Điều kiện: đã xác nhận được người quen + chưa gửi cảnh báo
            if (track.confirmed_name and 
                track.confirmed_name != "Stranger" and 
                not track.alert_sent):
                
                track.alert_sent = True
                alerts.append((tid, AlertType.KNOWN_PERSON, 
                              {'name': track.confirmed_name, 'distance': track.distance}))
            
            # Cảnh báo người lạ
            # Điều kiện: chưa nhận diện được + đủ số frame + chưa gửi cảnh báo
            if not track.confirmed_name:
                # CHỈ tăng counter nếu THỰC SỰ không có dấu hiệu nhận diện
                # Nếu đang nhận diện được (face_hits > 0 hoặc tên khác Stranger) -> không tăng
                if track.face_hits <= 0 and track.name == "Stranger":
                    track.frames_unidentified += 1
                else:
                    # Đang nhận diện được -> reset counter
                    track.frames_unidentified = 0
                
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
    
    # Vẽ bbox lên frame (Xanh: Quen, Đỏ: Lạ, Vàng: Chưa biết)
    def draw(self, frame):
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
    
    # Tính diện tích phần giao của 2 box
    def calc_iou(self, box1, box2):
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
    
    # Kiểm tra có mối nguy (Người lạ)
    def has_active_threats(self):
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
    
    # Kiểm tra có track nào đang active không
    def has_tracks(self):
        return bool(self.tracks)

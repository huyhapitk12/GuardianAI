# detection_core.py
import os
import time
import threading
import queue
import pickle
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from ultralytics import YOLO
from Lib.insightface.app import FaceAnalysis
from config import TARGET_FPS

from config import (
    EMBEDDING_FILE, NAMES_FILE, YOLO_MODEL_PATH, YOLO_PERSON_MODEL_PATH,
    MODEL_NAME, RECOGNITION_THRESHOLD, DATA_DIR, FRAMES_REQUIRED,
    PROCESS_EVERY_N_FRAMES, PROCESS_SIZE,
    FIRE_CONFIDENCE_THRESHOLD, FIRE_WINDOW_SECONDS, FIRE_REQUIRED,
    MODEL_DIR
)
from shared_state import state as sm

# --- Hook để main.py có thể bind hàm callback vào ---
on_alert_callback = None

# --- Helper Functions ---
def _colorfulness_score(img):
    """Đo độ 'màu sắc' của ảnh để tự động phát hiện định dạng video."""
    if img is None or img.ndim < 3 or img.shape[2] != 3:
        return 0.0
    # Tổng độ lệch chuẩn của các kênh màu
    return float(img[..., 0].std() + img[..., 1].std() + img[..., 2].std())

def _apply_conversion(frame, mode):
    """Áp dụng các chuyển đổi màu sắc phổ biến cho frame từ camera."""
    try:
        if mode == 'NONE': return frame
        if mode == 'BAYER_BG': return cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
        if mode == 'BAYER_RG': return cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2BGR)
        if mode == 'BAYER_GB': return cv2.cvtColor(frame, cv2.COLOR_BAYER_GB2BGR)
        if mode == 'BAYER_GR': return cv2.cvtColor(frame, cv2.COLOR_BAYER_GR2BGR)
        if mode == 'YUY2': return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
        if mode == 'NV12': return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
        if mode == 'I420': return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
    except Exception:
        return frame
    return frame

# --- Tải các model và dữ liệu một lần khi khởi động ---
print("[detection_core] Bắt đầu tải các model...")
try:
    model = YOLO(YOLO_MODEL_PATH, task='detect')
    print(f"[detection_core] Đã tải model YOLO Cháy/Khói từ: {YOLO_MODEL_PATH}")
except Exception as e:
    print(f"[detection_core] CẢNH BÁO: Không thể tải model YOLO Cháy/Khói: {e}")
    model = None

try:
    model_person = YOLO(YOLO_PERSON_MODEL_PATH)
    print(f"[detection_core] Đã tải model YOLO Người từ: {YOLO_PERSON_MODEL_PATH}")
except Exception as e:
    print(f"[detection_core] CẢNH BÁO: Không thể tải model YOLO Người: {e}")
    model_person = None

app = FaceAnalysis(name=MODEL_NAME, root="Data/Model", allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("[detection_core] InsightFace đã sẵn sàng.")

# --- Tải dữ liệu khuôn mặt đã biết ---
known_embeddings, known_names = [], []
if os.path.exists(EMBEDDING_FILE) and os.path.exists(NAMES_FILE):
    print("[detection_core] Đang tải dữ liệu khuôn mặt đã biết từ cache...")
    try:
        with open(EMBEDDING_FILE, 'rb') as f: known_embeddings = pickle.load(f)
        with open(NAMES_FILE, 'rb') as f: known_names = pickle.load(f)
        print(f"[detection_core] Đã tải {len(known_names)} khuôn mặt.")
    except Exception as e:
        print(f"[detection_core] Lỗi khi tải dữ liệu khuôn mặt: {e}")
else:
    print("[detection_core] Không tìm thấy file cache, sẽ tạo khi có dữ liệu mới.")

# --- Các hàm cập nhật động (gọi từ GUI) ---
def match_face(embedding):
    if not known_embeddings:
        return None, float('inf')
    distances = [float(cosine(embedding, k_emb)) for k_emb in known_embeddings]
    best_match_index = np.argmin(distances)
    return known_names[best_match_index], distances[best_match_index]

def update_known_data():
    global known_embeddings, known_names
    try:
        with open(EMBEDDING_FILE, 'rb') as f: new_embeddings = pickle.load(f)
        with open(NAMES_FILE, 'rb') as f: new_names = pickle.load(f)
        known_embeddings, known_names = new_embeddings, new_names
        print(f"[detection_core] Dữ liệu khuôn mặt đã được cập nhật. Tổng số: {len(known_names)}")
    except Exception as e:
        known_embeddings, known_names = [], []
        print(f"[detection_core] Không thể tải dữ liệu khuôn mặt, đã reset: {e}")

def update_model(selected_model_name):
    global app
    print(f"[detection_core] Đang chuyển đổi model nhận diện mặt sang: {selected_model_name}")
    app = FaceAnalysis(name=selected_model_name, root="Data/Model", allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("[detection_core] Chuyển đổi model nhận diện mặt thành công.")

def update_yolo_model(size: str):
    global model, model_person
    print(f"[detection_core] Yêu cầu chuyển đổi model YOLO sang kích thước: '{size}'")
    if size not in ["small", "medium"]:
        print(f"[detection_core] Lỗi: Kích thước model không hợp lệ: {size}.")
        return

    ### THAY ĐỔI 1: Sửa lại cách xây dựng đường dẫn cho đúng với định dạng `_openvino_model` ###
    fire_path = os.path.join(MODEL_DIR, size.capitalize(), "Fire", f"{size}_openvino_model")
    person_path = os.path.join(MODEL_DIR, size.capitalize(), "Person", f"{size}_openvino_model")

    try:
        model = YOLO(fire_path, task='detect')
        print(f"Tải model Lửa/Khói '{size}' thành công từ '{fire_path}'.")
    except Exception as e:
        print(f"Lỗi khi tải model Lửa/Khói từ '{fire_path}': {e}")
        model = None
    try:
        model_person = YOLO(person_path)
        print(f"Tải model Người '{size}' thành công từ '{person_path}'.")
    except Exception as e:
        print(f"Lỗi khi tải model Người từ '{person_path}': {e}")
        model_person = None

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea > 0 else 0

def create_tracker_prefer_csrt():
    try:
        # Ưu tiên tracker CSRT mới nhất nếu có
        if hasattr(cv2, "TrackerCSRT_create"): return cv2.TrackerCSRT_create()
        # Fallback cho các phiên bản OpenCV cũ hơn
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"): return cv2.legacy.TrackerCSRT_create()
    except Exception: pass
    return None

# --- Lớp Camera chính ---
class Camera:
    def __init__(self, src=0, show_window=False):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # chỉ giữ frame mới nhất, bỏ backlog
        if not self.cap.isOpened():
            raise RuntimeError(f"Không thể mở camera/nguồn: {src}")
        self.quit = False
        self.show_window = show_window

        # Queues để xử lý bất đồng bộ
        self.face_queue = queue.Queue(maxsize=2)
        self.fire_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=16)

        # State cho việc nhận diện
        self.fire_detection_timestamps = []
        self.current_fire_boxes = []
        self.tracked_objects = {}
        self.next_object_id = 0
        self.person_detection_was_on = True

        # Cấu hình
        self.IOU_THRESHOLD = 0.3
        self.TRACKER_TIMEOUT_SECONDS = 2.0
        self.PROC_W, self.PROC_H = PROCESS_SIZE

        # Worker threads
        threading.Thread(target=self.face_worker, daemon=True).start()
        threading.Thread(target=self.fire_worker, daemon=True).start()

        # State cho frame
        self._frame_lock = threading.Lock()
        self._last_frame = None
        self._raw_frame = None
        self._frame_idx = 0
        self._conversion_mode = None # Sẽ được tự động phát hiện

    def face_worker(self):
        """Thread xử lý nhận diện khuôn mặt."""
        while not self.quit:
            try:
                frame = self.face_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            if not sm.is_person_detection_enabled():
                time.sleep(0.1)
                continue

            try:
                faces = app.get(frame)
                face_results = []
                for face in faces:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    best_name, best_d = match_face(face.embedding)
                    if best_name is None or best_d > RECOGNITION_THRESHOLD:
                        best_name = "Nguoi la"
                    face_results.append({"bbox": (x1, y1, x2, y2), "name": best_name, "distance": best_d})
                if face_results:
                    self.result_queue.put(("face", face_results))
            except Exception as e:
                print(f"[Face Worker Error] {e}")

    def fire_worker(self):
        """Thread xử lý nhận diện cháy."""
        while not self.quit:
            try:
                frame = self.fire_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            ### THAY ĐỔI 2: Thêm lớp bảo vệ để worker không bị crash ###
            # Kiểm tra model mỗi lần lặp để đảm bảo nó không bị `None` do lỗi tải lại
            if model is None:
                time.sleep(1.0) # Chờ một chút trước khi thử lại
                continue
            
            try:
                results = model(frame, verbose=False)
                fire_results = []
                if results and hasattr(results[0], "boxes"):
                    for box in results[0].boxes:
                        if float(box.conf[0]) >= FIRE_CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cls_name = results[0].names.get(int(box.cls[0]), "unknown")
                            if cls_name.lower() in ("fire", "smoke", "flame"):
                                fire_results.append((x1, y1, x2, y2))
                if fire_results:
                    self.result_queue.put(("fire", fire_results))
            except Exception as e:
                print(f"[Fire Worker Error] {e}")

    def _update_last_frame(self, frame_with_drawings, raw_frame):
        with self._frame_lock:
            self._last_frame = frame_with_drawings.copy()
            self._raw_frame = raw_frame.copy()

    def read(self):
        with self._frame_lock:
            return (True, self._last_frame.copy()) if self._last_frame is not None else (False, None)

    def read_raw(self):
        with self._frame_lock:
            return (True, self._raw_frame.copy()) if self._raw_frame is not None else (False, None)

    def _auto_detect_format(self, frame):
        """Tự động phát hiện và chọn định dạng màu tốt nhất cho camera."""
        if self._frame_idx < 30: # Chỉ kiểm tra trong 30 frame đầu
            modes = ['NONE', 'BAYER_BG', 'BAYER_RG', 'BAYER_GB', 'BAYER_GR', 'YUY2', 'NV12', 'I420']
            candidates = []
            for mode in modes:
                try:
                    cand_frame = _apply_conversion(frame, mode)
                    score = _colorfulness_score(cand_frame)
                    if score > 0: candidates.append((mode, score))
                except Exception:
                    pass
            
            if candidates:
                best_mode, best_score = max(candidates, key=lambda x: x[1])
                if best_score > 10: # Ngưỡng để tránh chọn sai
                    self._conversion_mode = best_mode
                    print(f"[Camera Format] Tự động chọn định dạng: {best_mode} (score={best_score:.2f})")
        
        return _apply_conversion(frame, self._conversion_mode) if self._conversion_mode else frame

    def _process_face_results(self, results, scale_x, scale_y, now, frame):
        # Chuyển đổi tọa độ về kích thước frame gốc
        for res in results:
            x1, y1, x2, y2 = res['bbox']
            res['bbox'] = (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))

        matched_obj_ids = set()
        for res in results:
            best_iou, best_obj_id = 0, None
            for obj_id, data in self.tracked_objects.items():
                if obj_id in matched_obj_ids: continue
                iou = calculate_iou(res['bbox'], data['bbox'])
                if iou > self.IOU_THRESHOLD and iou > best_iou:
                    best_iou, best_obj_id = iou, obj_id
            
            if best_obj_id is not None: # Nếu khớp với tracker đã có
                data = self.tracked_objects[best_obj_id]
                data.update({'bbox': res['bbox'], 'last_seen_by_detector': now, 'hits': data['hits'] + 1})
                if data['name'] == "Nguoi la": # Cập nhật tên nếu trước đó là 'Nguoi la'
                    data.update({'name': res['name'], 'distance': res['distance']})
                
                if data['hits'] >= FRAMES_REQUIRED and not data['alert_sent']:
                    reason = "nguoi_quen" if data['name'] != "Nguoi la" else "nguoi_la"
                    name_param = data['name'] if reason == "nguoi_quen" else None
                    if on_alert_callback:
                        on_alert_callback(frame.copy(), reason, name_param, {"distance": data['distance']})
                    data['alert_sent'] = True
                matched_obj_ids.add(best_obj_id)
            else: # Nếu là đối tượng mới
                tracker = create_tracker_prefer_csrt()
                if tracker is None: continue
                x, y, w, h = res['bbox'][0], res['bbox'][1], res['bbox'][2]-res['bbox'][0], res['bbox'][3]-res['bbox'][1]
                tracker.init(frame, (x, y, w, h))
                
                new_id = self.next_object_id
                self.next_object_id += 1
                self.tracked_objects[new_id] = {
                    'tracker': tracker, 'bbox': res['bbox'], 'name': res['name'],
                    'distance': res['distance'], 'last_updated': now,
                    'last_seen_by_detector': now, 'hits': 1, 'alert_sent': False
                }

    def _process_fire_results(self, results, scale_x, scale_y, now, frame):
        self.fire_detection_timestamps.extend([now] * len(results))
        self.fire_detection_timestamps = [t for t in self.fire_detection_timestamps if now - t < FIRE_WINDOW_SECONDS]

        if len(self.fire_detection_timestamps) >= FIRE_REQUIRED:
            if on_alert_callback:
                frame_with_box = frame.copy()
                self.current_fire_boxes.clear()
                for box in results:
                    x1, y1, x2, y2 = box
                    x1_orig, y1_orig = int(x1 * scale_x), int(y1 * scale_y)
                    x2_orig, y2_orig = int(x2 * scale_x), int(y2 * scale_y)
                    self.current_fire_boxes.append((x1_orig, y1_orig, x2_orig, y2_orig))
                    cv2.rectangle(frame_with_box, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 0, 255), 2)
                
                on_alert_callback(frame_with_box, "lua_chay", None, {})
            self.fire_detection_timestamps.clear() # Reset sau khi gửi cảnh báo

    def detect(self):
        frame_interval = 1.0 / TARGET_FPS
        last_time = 0
        
        while not self.quit:
            now = time.time()
            if now - last_time < frame_interval:
                time.sleep(0.001)
                continue
            last_time = now
            
            # bỏ bớt frame cũ trong buffer (nếu có)
            for _ in range(2):  
                self.cap.grab()
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            self._frame_idx += 1
            now = time.time()
            
            # Tự động sửa định dạng màu nếu cần
            current_raw_frame = self._auto_detect_format(frame)
            display_frame = current_raw_frame.copy()
            orig_h, orig_w = current_raw_frame.shape[:2]

            # Xử lý bật/tắt nhận diện người
            person_detection_is_on = sm.is_person_detection_enabled()
            if not person_detection_is_on and self.person_detection_was_on:
                self.tracked_objects.clear()
                print("[Detection] Nhận diện người đã tắt. Đã xóa các tracker.")
            self.person_detection_was_on = person_detection_is_on

            # Cập nhật trackers
            if person_detection_is_on:
                ids_to_delete = []
                for obj_id, data in self.tracked_objects.items():
                    success, bbox = data['tracker'].update(current_raw_frame)
                    if success:
                        data['bbox'] = tuple(map(int, (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])))
                        data['last_updated'] = now
                    if not success or (now - data['last_seen_by_detector'] > self.TRACKER_TIMEOUT_SECONDS):
                        ids_to_delete.append(obj_id)
                for obj_id in ids_to_delete:
                    if obj_id in self.tracked_objects: del self.tracked_objects[obj_id]

            # Gửi frame cho workers xử lý
            if (self._frame_idx % PROCESS_EVERY_N_FRAMES) == 0:
                try:
                    small_frame = cv2.resize(current_raw_frame, (self.PROC_W, self.PROC_H), interpolation=cv2.INTER_AREA)
                    if person_detection_is_on and not self.face_queue.full(): self.face_queue.put_nowait(small_frame)
                    if not self.fire_queue.full(): self.fire_queue.put_nowait(small_frame)
                except Exception as e:
                    print(f"[Detect Loop] Lỗi khi gửi frame cho worker: {e}")

            # Xử lý kết quả từ workers
            scale_x, scale_y = orig_w / float(self.PROC_W), orig_h / float(self.PROC_H)
            try:
                while not self.result_queue.empty():
                    result_type, results = self.result_queue.get_nowait()
                    if result_type == "face" and person_detection_is_on:
                        self._process_face_results(results, scale_x, scale_y, now, current_raw_frame)
                    elif result_type == "fire":
                        self._process_fire_results(results, scale_x, scale_y, now, current_raw_frame)
            except queue.Empty:
                pass

            # Vẽ kết quả lên frame hiển thị
            if person_detection_is_on:
                for data in self.tracked_objects.values():
                    x1, y1, x2, y2 = data['bbox']
                    color = (0, 255, 0) if data['name'] != "Nguoi la" else (0, 0, 255)
                    label = f"{data['name']} ({data.get('distance', 0.0):.2f})"
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            for (x1, y1, x2, y2) in self.current_fire_boxes:
                 cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            self._update_last_frame(display_frame, current_raw_frame)

            if self.show_window:
                cv2.imshow("Guardian Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.quit = True
            else:
                time.sleep(0.001) # Sleep nhẹ để tránh 100% CPU

        self.delete()

    def delete(self):
        print("[Camera] Đang dọn dẹp và thoát...")
        self.quit = True
        if self.cap: self.cap.release()
        if self.show_window: cv2.destroyAllWindows()

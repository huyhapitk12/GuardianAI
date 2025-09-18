# detection_core.py
import os
import time
import uuid
import threading
import queue
import pickle
import cv2
from collections import defaultdict

from ultralytics import YOLO
from Lib.insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

from config import (
    EMBEDDING_FILE, NAMES_FILE, YOLO_MODEL_PATH, YOLO_PERSON_MODEL_PATH, 
    MODEL_NAME, RECOGNITION_THRESHOLD, DATA_DIR, FRAMES_REQUIRED, 
    PROCESS_EVERY_N_FRAMES, PROCESS_SIZE, DEBOUNCE_SECONDS, 
    FIRE_CONFIDENCE_THRESHOLD, FIRE_WINDOW_SECONDS, FIRE_REQUIRED,
    MODEL_DIR # <--- THÊM DÒNG NÀY
)
from shared_state import state as sm

# --- Hook để main bind ---
on_alert_callback = None

# --- Global loaded models / data (load once) ---
print("[detection_core] Loading models...")
try:
    # Model này dùng cho phát hiện lửa/khói
    model = YOLO(YOLO_MODEL_PATH, task='detect')
    print(f"[detection_core] Loaded Fire/Smoke YOLO model from: {YOLO_MODEL_PATH}")
except Exception as e:
    print("[detection_core] Warning: cannot load YOLO model from", YOLO_MODEL_PATH, e)
    model = None

try:
    # Model này dùng cho phát hiện người (thay thế cho face detection khi cần)
    model_person = YOLO(YOLO_PERSON_MODEL_PATH)
    print(f"[detection_core] Loaded Person YOLO model from: {YOLO_PERSON_MODEL_PATH}")
except Exception as e:
    print("[detection_core] Warning: cannot load YOLO person model from", YOLO_PERSON_MODEL_PATH, e)
    model_person = None

app = FaceAnalysis(name=MODEL_NAME, root="Data/Model", allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("[detection_core] InsightFace ready.")

# --- Load known embeddings/names ---
if os.path.exists(EMBEDDING_FILE) and os.path.exists(NAMES_FILE):
    print("Đang tải dữ liệu khuôn mặt đã biết từ bộ nhớ cache...")
    try:
        with open(EMBEDDING_FILE, 'rb') as f:
            known_embeddings = pickle.load(f)
        with open(NAMES_FILE, 'rb') as f:
            known_names = pickle.load(f)
        print(f"Đã tải {len(known_names)} khuôn mặt đã biết.")
    except Exception as e:
        print("[detection_core] Failed to load known data:", e)
        known_embeddings, known_names = [], []
else:
    print("Không tìm thấy file dữ liệu, khởi tạo danh sách rỗng.")
    known_embeddings, known_names = [], []


# --- Utility functions ---
def match_face(embedding):
    if not known_embeddings:
        return None, float('inf')
    best_d, best_name = float('inf'), None
    for k_emb, k_name in zip(known_embeddings, known_names):
        d = float(cosine(embedding, k_emb))
        if d < best_d:
            best_d, best_name = d, k_name
    return best_name, best_d

def update_known_data():
    """
    Tải lại dữ liệu khuôn mặt đã biết từ file một cách an toàn.
    Nếu không tải được, sẽ reset dữ liệu trong bộ nhớ về rỗng.
    """
    global known_embeddings, known_names
    new_embeddings, new_names = [], []
    try:
        if os.path.exists(EMBEDDING_FILE) and os.path.exists(NAMES_FILE):
            with open(EMBEDDING_FILE, 'rb') as f:
                new_embeddings = pickle.load(f)
            with open(NAMES_FILE, 'rb') as f:
                new_names = pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        print(f"[detection_core] Không thể tải dữ liệu đã biết ({type(e).__name__}), đang reset.")
    except Exception as e:
        print(f"[detection_core] Lỗi không mong muốn khi tải dữ liệu: {e}")
    
    known_embeddings = new_embeddings
    known_names = new_names
    print(f"[detection_core] Dữ liệu đã biết được cập nhật. Tổng số khuôn mặt: {len(known_names)}")

def update_model(selected):
    global app
    app = FaceAnalysis(name=selected, root="Data/Model", allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=-1, det_size=(640, 640))

# <--- THÊM HÀM MỚI DƯỚI ĐÂY --->
def update_yolo_model(size: str):
    """Tải lại các model YOLO với kích thước được chỉ định ('small' hoặc 'medium')."""
    global model, model_person
    print(f"[detection_core] Yêu cầu chuyển đổi model YOLO sang kích thước: '{size}'")
    
    if size not in ["small", "medium"]:
        print(f"[detection_core] Lỗi: Kích thước model không hợp lệ: {size}. Phải là 'small' hoặc 'medium'.")
        return

    fire_path = os.path.join(MODEL_DIR, size.capitalize(), "Fire", f"{size}")
    person_path = os.path.join(MODEL_DIR, size.capitalize(), "Person", f"{size}")

    try:
        print(f"Đang tải model Lửa/Khói mới từ: {fire_path}")
        model = YOLO(fire_path, task='detect')
        print("Tải model Lửa/Khói thành công.")
    except Exception as e:
        print(f"Lỗi khi tải model Lửa/Khói từ '{fire_path}': {e}")
        model = None

    try:
        print(f"Đang tải model Người mới từ: {person_path}")
        model_person = YOLO(person_path)
        print("Tải model Người thành công.")
    except Exception as e:
        print(f"Lỗi khi tải model Người từ '{person_path}': {e}")
        model_person = None
# <--- KẾT THÚC PHẦN THÊM MỚI --->


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

def create_tracker_prefer_csrt():
    try:
        if hasattr(cv2, "TrackerCSRT_create"): return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"): return cv2.legacy.TrackerCSRT_create()
    except Exception: pass
    return None

# --- Camera class ---
class Camera:
    def __init__(self, src=0, show_window=True):
        try: src_param = int(src)
        except Exception: src_param = src
        self.cap = cv2.VideoCapture(src_param, cv2.CAP_DSHOW) if isinstance(src_param, int) else cv2.VideoCapture(src_param)
        if not self.cap.isOpened(): raise RuntimeError(f"Cannot open camera/source: {src}")
        self.quit = False

        self.face_queue = queue.Queue(maxsize=2)
        self.fire_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=16)
        self.fire_detection_timestamps = []

        self.tracked_objects = {}
        self.next_object_id = 0
        self.IOU_THRESHOLD = 0.3
        self.TRACKER_TIMEOUT_SECONDS = 2.0

        self.FRAMES_REQUIRED = FRAMES_REQUIRED
        self.PROCESS_EVERY_N_FRAMES = PROCESS_EVERY_N_FRAMES
        if len(PROCESS_SIZE) == 2: self.PROC_W, self.PROC_H = PROCESS_SIZE[0], PROCESS_SIZE[1]
        else: self.PROC_W, self.PROC_H = 320, 240

        self.face_thread = threading.Thread(target=self.face_worker, daemon=True)
        self.fire_thread = threading.Thread(target=self.fire_worker, daemon=True)
        self.face_thread.start()
        self.fire_thread.start()

        self.show_window = show_window
        self._frame_lock = threading.Lock()
        self._last_frame = None
        self._frame_idx = 0
        
        self.person_detection_was_on = True

    def face_worker(self):
        while not self.quit:
            try: frame = self.face_queue.get(timeout=1.0)
            except queue.Empty: continue
            
            if not sm.is_person_detection_enabled():
                time.sleep(0.1)
                continue

            try:
                faces = app.get(frame)
                face_results = []
                for face in faces:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    emb = face.embedding
                    best_name, best_d = match_face(emb)
                    if best_name is None or best_d > RECOGNITION_THRESHOLD: best_name = "Nguoi la"
                    face_results.append({"bbox": (x1, y1, x2, y2), "name": best_name, "distance": best_d})
                if face_results: self.result_queue.put(("face", face_results), timeout=0.5)
            except Exception: pass

    def fire_worker(self):
        if model is None:
            print("[Fire Worker] Model is None, worker will sleep.")
            while not self.quit: 
                if model is not None:
                    print("[Fire Worker] Model has been loaded, starting work.")
                    break
                time.sleep(1.0)

        while not self.quit:
            if model is None: # Kiểm tra lại trong vòng lặp
                time.sleep(1.0)
                continue
            try:
                frame = self.fire_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                results = model(frame, verbose=False)
                fire_results = []
                if len(results) > 0 and hasattr(results[0], "boxes"):
                    for box in results[0].boxes:
                        conf = float(box.conf[0]) if hasattr(box, "conf") else 1.0
                        if conf < FIRE_CONFIDENCE_THRESHOLD:
                            continue
                        coords = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, coords)
                        cls_id = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
                        cls_name = results[0].names.get(cls_id, str(cls_id)) if hasattr(results[0], "names") else str(cls_id)
                        if cls_name.lower() in ("fire", "smoke", "flame"):
                            fire_results.append((x1, y1, x2, y2, "lua_chay"))
                if fire_results: self.result_queue.put(("fire", fire_results), timeout=0.5)
            except Exception: pass

    def _update_last_frame(self, frame):
        with self._frame_lock: self._last_frame = frame.copy()

    def read(self):
        with self._frame_lock:
            if self._last_frame is None: return False, None
            return True, self._last_frame.copy()

    def detect(self):
        while not self.quit:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            self._update_last_frame(frame)
            orig_h, orig_w = frame.shape[:2]
            self._frame_idx += 1
            now = time.time()

            person_detection_is_on = sm.is_person_detection_enabled()

            if not person_detection_is_on and self.person_detection_was_on:
                self.tracked_objects.clear()
                print("[Detection] Person detection disabled. Clearing trackers.")
            
            self.person_detection_was_on = person_detection_is_on
            
            if person_detection_is_on:
                objects_to_delete = []
                for obj_id, data in self.tracked_objects.items():
                    success, bbox = data['tracker'].update(frame)
                    if success:
                        data['bbox'] = tuple(map(int, (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])))
                        data['last_updated'] = now
                    else:
                        objects_to_delete.append(obj_id)
                    
                    if now - data['last_seen_by_detector'] > self.TRACKER_TIMEOUT_SECONDS:
                        objects_to_delete.append(obj_id)

                for obj_id in set(objects_to_delete):
                    if obj_id in self.tracked_objects: del self.tracked_objects[obj_id]

            if (self._frame_idx % self.PROCESS_EVERY_N_FRAMES) == 0:
                try:
                    small_frame = cv2.resize(frame, (self.PROC_W, self.PROC_H), interpolation=cv2.INTER_AREA)
                    if person_detection_is_on and not self.face_queue.full(): 
                        self.face_queue.put_nowait(small_frame)
                    if not self.fire_queue.full(): 
                        self.fire_queue.put_nowait(small_frame)
                except Exception: pass

            try:
                while not self.result_queue.empty():
                    typ, results = self.result_queue.get_nowait()
                    if typ == "face" and person_detection_is_on:
                        scale_x = orig_w / float(self.PROC_W)
                        scale_y = orig_h / float(self.PROC_H)
                        
                        for res in results:
                            x1, y1, x2, y2 = res['bbox']
                            res['bbox'] = (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))

                        iou_matches = []
                        for i, res in enumerate(results):
                            for obj_id, data in self.tracked_objects.items():
                                iou = calculate_iou(res['bbox'], data['bbox'])
                                if iou > self.IOU_THRESHOLD:
                                    iou_matches.append((iou, i, obj_id))
                        
                        iou_matches.sort(key=lambda x: x[0], reverse=True)

                        matched_det_indices = set()
                        matched_obj_ids = set()

                        for iou, det_idx, obj_id in iou_matches:
                            if det_idx in matched_det_indices or obj_id in matched_obj_ids:
                                continue
                            
                            res = results[det_idx]
                            data = self.tracked_objects[obj_id]
                            
                            tracker = create_tracker_prefer_csrt()
                            x, y, w, h = res['bbox'][0], res['bbox'][1], res['bbox'][2]-res['bbox'][0], res['bbox'][3]-res['bbox'][1]
                            tracker.init(frame, (x, y, w, h))
                            data['tracker'] = tracker
                            data['bbox'] = res['bbox']
                            data['last_seen_by_detector'] = now
                            data['hits'] += 1
                            
                            if data['name'] == "Nguoi la":
                                data['name'] = res['name']
                                data['distance'] = res['distance']

                            if data['hits'] >= self.FRAMES_REQUIRED and not data['alert_sent']:
                                reason = "nguoi_quen" if data['name'] != "Nguoi la" else "nguoi_la"
                                name_param = data['name'] if reason == "nguoi_quen" else None
                                if on_alert_callback:
                                    on_alert_callback(frame.copy(), reason, name_param, {"distance": data['distance']})
                                data['alert_sent'] = True

                            matched_det_indices.add(det_idx)
                            matched_obj_ids.add(obj_id)

                        for i, res in enumerate(results):
                            if i not in matched_det_indices:
                                tracker = create_tracker_prefer_csrt()
                                if tracker is None: continue
                                x, y, w, h = res['bbox'][0], res['bbox'][1], res['bbox'][2]-res['bbox'][0], res['bbox'][3]-res['bbox'][1]
                                tracker.init(frame, (x, y, w, h))
                                
                                new_id = self.next_object_id
                                self.next_object_id += 1
                                self.tracked_objects[new_id] = {
                                    'tracker': tracker,
                                    'bbox': res['bbox'],
                                    'name': res['name'],
                                    'distance': res['distance'],
                                    'last_updated': now,
                                    'last_seen_by_detector': now,
                                    'hits': 1,
                                    'alert_sent': False
                                }

                    elif typ == "fire":
                        # ... (giữ nguyên) ...
                        pass
            except queue.Empty:
                pass

            if person_detection_is_on:
                for obj_id, data in self.tracked_objects.items():
                    x1, y1, x2, y2 = data['bbox']
                    name = data['name']
                    dist = data.get('distance', 0.0)
                    color = (0, 255, 0) if name != "Nguoi la" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{obj_id} {name} ({dist:.2f})"
                    cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if self.show_window:
                display_frame = cv2.resize(frame, (self.PROC_W, self.PROC_H), interpolation=cv2.INTER_AREA)
                cv2.imshow("Detection (press q to quit)", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.quit = True
                    break
        self.delete()

    def delete(self):
        self.quit = True
        try: self.cap.release()
        except: pass
        try: cv2.destroyAllWindows()
        except: pass

def get_known_data():
    return known_embeddings, known_names

__all__ = ['Camera', 'app', 'match_face', 'update_known_data', 'update_yolo_model'] # <--- THÊM 'update_yolo_model'
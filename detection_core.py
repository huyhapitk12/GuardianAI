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
from Lib.trackers import SORTTracker
import supervision as sv
from collections import deque

from config import (
    EMBEDDING_FILE, NAMES_FILE, YOLO_MODEL_PATH, YOLO_PERSON_MODEL_PATH,
    MODEL_NAME, RECOGNITION_THRESHOLD, DATA_DIR, FRAMES_REQUIRED,
    PROCESS_EVERY_N_FRAMES, PROCESS_SIZE,
    FIRE_CONFIDENCE_THRESHOLD, FIRE_WINDOW_SECONDS, FIRE_REQUIRED,
    MODEL_DIR, TARGET_FPS, INSIGHTFACE_CTX_ID, INSIGHTFACE_DET_SIZE,
    IOU_THRESHOLD, TRACKER_TIMEOUT_SECONDS, FACE_RECOG_COOLDOWN,
    FIRE_YELLOW_ALERT_FRAMES, FIRE_RED_ALERT_GROWTH_THRESHOLD,
    FIRE_RED_ALERT_GROWTH_WINDOW, FIRE_RED_ALERT_LOCKDOWN_SECONDS,
    STRANGER_CONFIRM_FRAMES, YOLO_PERSON_CONFIDENCE
)
from shared_state import state as sm

# --- Hook để main.py có thể bind hàm callback vào ---
on_alert_callback = None

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

# InsightFace app
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
        if hasattr(cv2, "TrackerCSRT_create"): return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"): return cv2.legacy.TrackerCSRT_create()
    except Exception: pass
    return None

# safe IoU and robust parse_tracked used by SORT adapter
def safe_iou(boxA, boxB):
    try:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0.0, xB - xA)
        interH = max(0.0, yB - yA)
        interArea = interW * interH
        boxAArea = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
        boxBArea = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
        union = boxAArea + boxBArea - interArea
        if union <= 0.0:
            return 0.0
        return interArea / union
    except Exception:
        return 0.0

def parse_tracked(tracked):
    """
    Trả về (xyxy_array, id_list) hoặc (None, None) nếu không parse được.
    Hỗ trợ một số định dạng phổ biến (supervision.Detections, np.ndarray, list).
    """
    try:
        # supervision.Detections-like
        if hasattr(tracked, "xyxy"):
            xyxy = np.asarray(getattr(tracked, "xyxy"))
            ids = None
            for n in ("tracker_id","ids","track_id","id"):
                if hasattr(tracked, n):
                    try:
                        ids = list(getattr(tracked, n))
                        break
                    except Exception:
                        ids = None
            if ids is None and xyxy.ndim == 2 and xyxy.shape[1] >= 5:
                lastcol = xyxy[:, -1]
                if np.all(np.isfinite(lastcol)) and np.allclose(lastcol, np.round(lastcol)):
                    ids = lastcol.astype(int).tolist()
                    xyxy = xyxy[:, :4]
            if ids is None:
                ids = [None]*len(xyxy)
            if len(ids) != len(xyxy):
                ids = (list(ids) + [None]*len(xyxy))[:len(xyxy)]
            return np.asarray(xyxy, dtype=float), ids

        # numpy array Nx4/5/6: if last col int -> ids
        if isinstance(tracked, np.ndarray):
            if tracked.ndim == 2 and tracked.shape[1] >= 4:
                if tracked.shape[1] == 4:
                    return tracked.astype(float), [None]*tracked.shape[0]
                last = tracked[:, -1]
                if np.all(np.isfinite(last)) and np.allclose(last, np.round(last)):
                    return tracked[:, :4].astype(float), last.astype(int).tolist()
                else:
                    return tracked[:, :4].astype(float), [None]*tracked.shape[0]

        # list-like: tuples (x1,y1,x2,y2,id) or objects with bbox attrs
        if isinstance(tracked, (list, tuple)):
            if len(tracked) == 0:
                return np.zeros((0,4)), []
            first = tracked[0]
            # tuple-like
            if isinstance(first, (list, tuple)) and len(first) >= 4:
                xy = []
                ids = []
                for item in tracked:
                    if len(item) >= 4 and all(isinstance(v, (int,float,np.integer,np.floating)) for v in item[:4]):
                        xy.append(tuple(item[:4]))
                        ids.append(item[4] if len(item) >= 5 else None)
                    elif len(item) == 2 and isinstance(item[0], (list,tuple,np.ndarray)):
                        bbox = item[0]
                        xy.append(tuple(bbox[:4])); ids.append(item[1])
                if len(xy) == 0:
                    return None, None
                return np.asarray(xy, dtype=float), ids

            # object-like: try to extract tlbr/tlwh and id
            if hasattr(first, "to_tlbr") or hasattr(first, "tlbr") or hasattr(first, "tlwh"):
                xy = []; ids = []
                for obj in tracked:
                    try:
                        if hasattr(obj, "to_tlbr"):
                            tb = obj.to_tlbr()
                        elif hasattr(obj, "tlbr"):
                            tb = obj.tlbr if not callable(obj.tlbr) else obj.tlbr()
                        elif hasattr(obj, "tlwh"):
                            t = obj.tlwh if not callable(obj.tlwh) else obj.tlwh()
                            tb = (t[0], t[1], t[0]+t[2], t[1]+t[3])
                        else:
                            tb = (0,0,0,0)
                        tid = None
                        for n in ("track_id","id","tracker_id"):
                            if hasattr(obj, n):
                                try:
                                    tid = getattr(obj, n)
                                    break
                                except Exception:
                                    tid = None
                        xy.append(tb); ids.append(tid)
                    except Exception:
                        xy.append((0,0,0,0)); ids.append(None)
                return np.asarray(xy, dtype=float), ids

        # dict-like
        if isinstance(tracked, dict):
            for key in ("tracks","detections","results"):
                if key in tracked:
                    return parse_tracked(tracked[key])

    except Exception as e:
        print(f"[parse_tracked] error: {e}")
        return None, None

    return None, None

def _extract_face_from_person_box(frame, person_box, expand_ratio=0.4):
    """
    person_box: (x1,y1,x2,y2) trên frame gốc
    Trả về crop image khu vực đầu/face ước lượng
    """
    h, w = frame.shape[:2]
    x1,y1,x2,y2 = person_box
    ph = y2 - y1
    expand_h = int(ph * expand_ratio)
    cy1 = max(0, y1 - expand_h)
    cy2 = min(h, int(y1 + ph * 0.45))
    cx1 = max(0, int(x1 - expand_h))
    cx2 = min(w, int(x2 + expand_h))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return frame[cy1:cy2, cx1:cx2].copy()

# --- Lớp Camera chính ---
class Camera:
    def __init__(self, src=0, show_window=False):
        if isinstance(src, int):
            # Nếu src là một số (như 0), đây là camera -> dùng DSHOW để tối ưu
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            # Nếu src là một chuỗi (đường dẫn file/URL) -> để OpenCV tự chọn backend tốt nhất
            self.cap = cv2.VideoCapture(src)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không thể mở camera/nguồn: {src}")
        self.quit = False
        self.show_window = show_window

        # queues
        self.fire_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=16)

        # state
        self.recent_fire_detections = deque(maxlen=int(TARGET_FPS * FIRE_WINDOW_SECONDS))
        self.current_fire_boxes_on_display = []
        self.last_fire_alert_time = 0
        self.tracked_objects = {}
        self.next_object_id = 0
        self.person_detection_was_on = True

        # config
        self.IOU_THRESHOLD = IOU_THRESHOLD
        self.TRACKER_TIMEOUT_SECONDS = TRACKER_TIMEOUT_SECONDS
        self.PROC_W, self.PROC_H = PROCESS_SIZE
        # face recognition params
        self.FACE_RECOG_COOLDOWN = FACE_RECOG_COOLDOWN
        self.FACE_CONFIRM_HITS = FRAMES_REQUIRED
        # Cấu hình cho logic cảnh báo cháy phân cấp
        self.YELLOW_ALERT_FRAMES = FIRE_YELLOW_ALERT_FRAMES
        self.RED_ALERT_GROWTH_THRESHOLD = FIRE_RED_ALERT_GROWTH_THRESHOLD
        self.RED_ALERT_GROWTH_WINDOW = FIRE_RED_ALERT_GROWTH_WINDOW
        # Cấu hình cho chế độ "khóa" cảnh báo Đỏ
        self.RED_ALERT_LOCKDOWN_SECONDS = FIRE_RED_ALERT_LOCKDOWN_SECONDS
        self.red_alert_mode_active = False
        self.red_alert_mode_until = 0
        # <--- THÊM MỚI: Cấu hình cho cảnh báo người lạ ---
        self.STRANGER_CONFIRM_FRAMES = STRANGER_CONFIRM_FRAMES
        # --- KẾT THÚC THÊM MỚI ---

        # SORT tracker
        try:
            self.sort = SORTTracker()
            self.use_sort = True
            print("[Camera] SORT tracker khởi tạo thành công.")
        except Exception as e:
            self.sort = None
            self.use_sort = False
            print(f"[Camera] Không thể khởi tạo SORT: {e}. Fallback sang CSRT/IOU.")

        # worker threads
        threading.Thread(target=self.fire_worker, daemon=True).start()
        # Note: face worker kept for compatibility but not used in person->face pipeline
        threading.Thread(target=self.face_worker, daemon=True).start()

        # frame state
        self._frame_lock = threading.Lock()
        self._last_frame = None
        self._raw_frame = None
        self._frame_idx = 0
        self._conversion_mode = None

    def face_worker(self):
        """(Còn giữ để tương thích) worker xử lý face queue — ít dùng khi pipeline person->face."""
        while not self.quit:
            try:
                frame = None
                time.sleep(1.0)
            except Exception:
                time.sleep(0.1)

    def fire_worker(self):
        """Thread xử lý nhận diện cháy."""
        while not self.quit:
            try:
                frame = self.fire_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if model is None:
                time.sleep(1.0)
                continue

            try:
                results = model(frame, verbose=False)
                fire_results = []
                if results and hasattr(results[0], "boxes"):
                    for box in results[0].boxes:
                        conf = float(box.conf[0])
                        if conf >= FIRE_CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cls_name = results[0].names.get(int(box.cls[0]), "unknown").lower()
                            if cls_name in ("fire", "smoke", "flame"):
                                area = (x2 - x1) * (y2 - y1)
                                fire_results.append({
                                    "bbox": (x1, y1, x2, y2),
                                    "class": "smoke" if cls_name == "smoke" else "fire",
                                    "area": area,
                                    "conf": conf
                                })
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

    def _process_person_results(self, person_results, scale_x, scale_y, now, frame):
        """
        Xử lý person detection -> gọi SORT -> crop face -> face recognition -> cập nhật tracked_objects.
        person_results: list of (x1,y1,x2,y2) relative to small_frame (PROC_W x PROC_H).
        """
        # convert coordinates to original frame size
        converted = []
        for box in person_results:
            x1,y1,x2,y2 = box
            converted.append((int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)))

        if len(converted) > 0:
            xyxy_in = np.array(converted, dtype=float)
            scores = np.ones((xyxy_in.shape[0],), dtype=float)
            dets_array = np.hstack([xyxy_in, scores.reshape(-1,1)])
        else:
            xyxy_in = np.zeros((0, 4), dtype=float)
            scores = np.array([], dtype=float)
            dets_array = np.zeros((0, 5), dtype=float)

        # call SORT
        tracked_xyxy, tracked_ids = None, None
        if self.use_sort and self.sort is not None:
            try:
                try:
                    tracked = self.sort.update(dets_array)
                except Exception:
                    dets = sv.Detections(xyxy=xyxy_in, confidence=scores, class_id=np.zeros(len(scores), dtype=int))
                    tracked = self.sort.update(dets)
                tracked_xyxy, tracked_ids = parse_tracked(tracked)
            except Exception as e:
                print("[Person->SORT] error:", e)
                tracked_xyxy, tracked_ids = None, None

        # Fallback simple IOU matching if SORT failed
        if tracked_xyxy is None or tracked_ids is None:
            # match each detected person to existing tracked_objects by IOU else create new
            for box in converted:
                best_iou=0; best_id=None
                for tid, d in self.tracked_objects.items():
                    iou = safe_iou(box, d['bbox'])
                    if iou>best_iou:
                        best_iou, best_id = iou, tid
                if best_id is not None and best_iou >= self.IOU_THRESHOLD:
                    d = self.tracked_objects[best_id]
                    d['bbox']=box; d['last_seen_by_detector']=now; d['last_updated']=now
                else:
                    nid = self.next_object_id; self.next_object_id+=1
                    # <--- THAY ĐỔI: Thêm các trường mới cho logic người lạ ---
                    self.tracked_objects[nid] = {'bbox': box, 'name': "Nguoi la", 'distance': float('inf'),
                                                 'last_seen_by_detector': now, 'last_updated': now,
                                                 'face_hits':0, 'last_face_rec':0.0, 'confirmed_name': None, 'alert_sent':False,
                                                 'frames_unidentified': 0, 'stranger_alert_sent': False}
                    # --- KẾT THÚC THAY ĐỔI ---
            return

        # SORT returned something -> use it
        tracked_xyxy = np.asarray(tracked_xyxy, dtype=int)
        for i, tid in enumerate(tracked_ids):
            tb = tuple(map(int, tracked_xyxy[i]))
            if tid is None:
                tid = self.next_object_id
                self.next_object_id += 1

            # ensure entry
            if tid not in self.tracked_objects:
                # <--- THAY ĐỔI: Thêm các trường mới cho logic người lạ ---
                self.tracked_objects[tid] = {'bbox': tb, 'name': "Nguoi la", 'distance': float('inf'),
                                             'last_seen_by_detector': now, 'last_updated': now,
                                             'face_hits':0, 'last_face_rec':0.0, 'confirmed_name': None, 'alert_sent':False,
                                             'frames_unidentified': 0, 'stranger_alert_sent': False}
                # --- KẾT THÚC THAY ĐỔI ---
            else:
                data = self.tracked_objects[tid]
                data['bbox'] = tb
                data['last_seen_by_detector'] = now
                data['last_updated'] = now

            # Face recognition decision
            data = self.tracked_objects[tid]
            cooldown = getattr(self, "FACE_RECOG_COOLDOWN", 1.0)
            need_recog = (now - data.get('last_face_rec', 0.0) >= cooldown) and (data.get('confirmed_name') is None)

            if need_recog:
                data['last_face_rec'] = now
                
                x1, y1, x2, y2 = tb
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                if x2 > x1 and y2 > y1:
                    person_crop = frame[y1:y2, x1:x2]                    
                    try:
                        faces = app.get(person_crop)
                        
                        if faces and len(faces) > 0:
                            best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                            emb = best_face.embedding
                            name, dist = match_face(emb)
                            
                            if name is not None and dist <= RECOGNITION_THRESHOLD:
                                data['face_hits'] = data.get('face_hits', 0) + 1
                                data['name'] = name
                                data['distance'] = dist
                            else:
                                data['face_hits'] = max(0, data.get('face_hits', 0) - 1)
                                if data['face_hits'] == 0:
                                    data['name'] = "Nguoi la"
                                    data['distance'] = float('inf')
                        else:
                            data['face_hits'] = max(0, data.get('face_hits', 0) - 1)
                            
                    except Exception as e:
                        print(f"[Face recog error on person_crop] {e}")
                        data['face_hits'] = max(0, data.get('face_hits', 0) - 1)

            # <--- THAY ĐỔI: Toàn bộ logic cảnh báo người quen và người lạ ---
            # 1. Cảnh báo NGƯỜI QUEN (logic cũ)
            if data.get('face_hits',0) >= self.FACE_CONFIRM_HITS and data.get('confirmed_name') != data.get('name'):
                data['confirmed_name'] = data['name']
                if data['confirmed_name'] != "Nguoi la" and on_alert_callback and not data.get('alert_sent', False):
                    on_alert_callback(frame.copy(), "nguoi_quen", data['confirmed_name'], {"distance": data['distance']})
                    data['alert_sent'] = True

            # 2. Cảnh báo NGƯỜI LẠ (logic mới)
            # Nếu chưa xác nhận được danh tính, tăng bộ đếm
            if data.get('confirmed_name') is None:
                data['frames_unidentified'] = data.get('frames_unidentified', 0) + 1
            
            # Kiểm tra nếu bộ đếm vượt ngưỡng và chưa gửi cảnh báo
            if (data.get('frames_unidentified', 0) > self.STRANGER_CONFIRM_FRAMES and
                data.get('confirmed_name') is None and
                not data.get('stranger_alert_sent', False) and
                on_alert_callback):
                
                print(f"[Alert] Kích hoạt cảnh báo NGƯỜI LẠ cho track ID {tid}")
                on_alert_callback(frame.copy(), "nguoi_la", None, {})
                data['stranger_alert_sent'] = True # Đánh dấu đã gửi để không gửi lại
            # --- KẾT THÚC THAY ĐỔI ---

        # prune
        ids_to_remove = [oid for oid, d in list(self.tracked_objects.items()) if now - d.get('last_seen_by_detector', now) > self.TRACKER_TIMEOUT_SECONDS]
        for oid in ids_to_remove:
            self.tracked_objects.pop(oid, None)

    def _process_fire_results(self, results, scale_x, scale_y, now, frame):
        """
        Xử lý kết quả nhận diện cháy/khói và quyết định gửi cảnh báo phân cấp.
        Các hộp bao quanh chỉ hiển thị cho các phát hiện trong khung hình hiện tại.
        """
        # 1. Cập nhật các hộp hiển thị CHỈ với các kết quả của khung hình hiện tại
        for res in results:
            x1, y1, x2, y2 = res['bbox']
            x1_orig, y1_orig = int(x1 * scale_x), int(y1 * scale_y)
            x2_orig, y2_orig = int(x2 * scale_x), int(y2 * scale_y)
            self.current_fire_boxes_on_display.append((x1_orig, y1_orig, x2_orig, y2_orig))

        # 2. Thêm các phát hiện mới vào hàng đợi lịch sử để quyết định cảnh báo
        for res in results:
            res['timestamp'] = now
            self.recent_fire_detections.append(res)

        # Kiểm tra và reset chế độ khóa cảnh báo Đỏ nếu đã hết hạn
        if self.red_alert_mode_active and now > self.red_alert_mode_until:
            self.red_alert_mode_active = False
            self.red_alert_mode_until = 0
            print("[Fire Alert] Chế độ khóa cảnh báo Đỏ đã kết thúc.")

        # Nếu không có phát hiện nào trong lịch sử (và cả hiện tại), thì không làm gì cả
        if not self.recent_fire_detections:
            return

        # --- Logic quyết định cảnh báo (dựa trên lịch sử) ---
        is_red_alert = False
        
        # Ưu tiên cảnh báo Đỏ nếu đang trong chế độ khóa
        if self.red_alert_mode_active and results: # Chỉ kích hoạt nếu có phát hiện mới
            is_red_alert = True
            print("[Fire Alert] Cảnh báo ĐỎ được kích hoạt do đang trong chế độ khóa.")
        else:
            # Kiểm tra điều kiện cảnh báo ĐỎ (Khẩn cấp) lần đầu
            fire_detections = [d for d in self.recent_fire_detections if d['class'] == 'fire']
            if len(fire_detections) > 2:
                current_fires = [d for d in fire_detections if now - d['timestamp'] < 0.5]
                past_fires = [d for d in fire_detections if now - d['timestamp'] > self.RED_ALERT_GROWTH_WINDOW - 0.5 and now - d['timestamp'] < self.RED_ALERT_GROWTH_WINDOW]
                
                if current_fires and past_fires:
                    avg_current_area = sum(d['area'] for d in current_fires) / len(current_fires)
                    avg_past_area = sum(d['area'] for d in past_fires) / len(past_fires)
                    if avg_current_area > avg_past_area * self.RED_ALERT_GROWTH_THRESHOLD:
                        is_red_alert = True
                        print("[Fire Alert] RED ALERT triggered by fire growth.")

            if not is_red_alert:
                classes_present = {d['class'] for d in self.recent_fire_detections}
                if 'fire' in classes_present and 'smoke' in classes_present:
                    is_red_alert = True
                    print("[Fire Alert] RED ALERT triggered by simultaneous fire and smoke.")

        # Nếu là cảnh báo ĐỎ, gửi và xử lý
        if is_red_alert:
            if not self.red_alert_mode_active:
                self.red_alert_mode_active = True
                self.red_alert_mode_until = now + self.RED_ALERT_LOCKDOWN_SECONDS
                print(f"[Fire Alert] Kích hoạt chế độ khóa cảnh báo Đỏ trong {self.RED_ALERT_LOCKDOWN_SECONDS} giây.")

            if on_alert_callback:
                frame_with_box = frame.copy()
                # Vẽ tất cả các hộp trong lịch sử lên ảnh cảnh báo để cung cấp ngữ cảnh
                for d in self.recent_fire_detections:
                    x1, y1, x2, y2 = d['bbox']
                    x1_orig, y1_orig = int(x1 * scale_x), int(y1 * scale_y)
                    x2_orig, y2_orig = int(x2 * scale_x), int(y2 * scale_y)
                    cv2.rectangle(frame_with_box, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 0, 255), 3)
                
                on_alert_callback(frame_with_box, "lua_chay_khan_cap", None, {})
                self.recent_fire_detections.clear() # Xóa lịch sử để tránh gửi lại ngay lập tức
            return

        # Nếu không phải ĐỎ, kiểm tra điều kiện cảnh báo VÀNG (Nghi ngờ)
        if len(self.recent_fire_detections) >= self.YELLOW_ALERT_FRAMES:
            print(f"[Fire Alert] YELLOW ALERT triggered by sustained detection ({len(self.recent_fire_detections)} frames).")
            if on_alert_callback:
                frame_with_box = frame.copy()
                # Vẽ tất cả các hộp trong lịch sử lên ảnh cảnh báo
                for d in self.recent_fire_detections:
                    x1, y1, x2, y2 = d['bbox']
                    x1_orig, y1_orig = int(x1 * scale_x), int(y1 * scale_y)
                    x2_orig, y2_orig = int(x2 * scale_x), int(y2 * scale_y)
                    cv2.rectangle(frame_with_box, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 255), 2)
                
                on_alert_callback(frame_with_box, "lua_chay_nghi_ngo", None, {})
                self.recent_fire_detections.clear() # Xóa lịch sử để tránh gửi lại ngay lập tức
            return

    def detect(self):
        frame_interval = 1.0 / TARGET_FPS
        last_time = 0

        while not self.quit:
            now = time.time()
            if now - last_time < frame_interval:
                time.sleep(0.005)
                continue
            
            last_time = now

            for _ in range(2):
                self.cap.grab()

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            self._frame_idx += 1
            now = time.time()
            
            display_frame = frame.copy()
            orig_h, orig_w = frame.shape[:2]

            person_detection_is_on = sm.is_person_detection_enabled()
            if not person_detection_is_on and self.person_detection_was_on:
                self.tracked_objects.clear()
                print("[Detection] Nhận diện người đã tắt. Đã xóa các tracker.")
            self.person_detection_was_on = person_detection_is_on

            # prune stale tracks
            if person_detection_is_on:
                ids_to_delete = []
                for obj_id, data in list(self.tracked_objects.items()):
                    if now - data.get('last_seen_by_detector', now) > self.TRACKER_TIMEOUT_SECONDS:
                        ids_to_delete.append(obj_id)
                for obj_id in ids_to_delete:
                    self.tracked_objects.pop(obj_id, None)

            # Prepare small_frame for processing
            try:
                small_frame = cv2.resize(frame, (self.PROC_W, self.PROC_H), interpolation=cv2.INTER_AREA)
            except Exception as e:
                small_frame = frame.copy()

            # compute scaling factors from small_frame -> orig frame
            scale_x, scale_y = orig_w / float(self.PROC_W), orig_h / float(self.PROC_H)

            # Person detection + processing on a schedule
            if (self._frame_idx % PROCESS_EVERY_N_FRAMES) == 0 and person_detection_is_on:
                try:
                    if model_person is not None:
                        per_res = model_person(small_frame, conf=YOLO_PERSON_CONFIDENCE, classes=0, verbose=False)[0]
                        person_boxes = []
                        if hasattr(per_res, "boxes"):
                            for b in per_res.boxes:
                                try:
                                    x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
                                except Exception:
                                    x1,y1,x2,y2 = float(b.x1), float(b.y1), float(b.x2), float(b.y2)
                                person_boxes.append((x1, y1, x2, y2))
                        self._process_person_results(person_boxes, scale_x, scale_y, now, frame)
                except Exception as e:
                    print("[Detect person error]", e)

            # send to fire worker (non-blocking)
            if not self.fire_queue.full():
                try:
                    self.fire_queue.put_nowait(small_frame)
                except Exception:
                    pass

            self.current_fire_boxes_on_display = []

            # handle fire results
            try:
                while not self.result_queue.empty():
                    result_type, results = self.result_queue.get_nowait()
                    if result_type == "fire":
                        self._process_fire_results(results, scale_x, scale_y, now, frame)
            except queue.Empty:
                pass

            # draw tracked people
            if person_detection_is_on:
                for data in self.tracked_objects.values():
                    x1, y1, x2, y2 = data['bbox']
                    color = (0, 255, 0) if data.get('confirmed_name') and data.get('confirmed_name') != "Nguoi la" else (0, 0, 255)
                    name = data.get('confirmed_name') or data.get('name') or "Nguoi la"
                    label = f"{name} ({data.get('distance', 0.0):.2f})"
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Vẽ các hộp cháy/khói từ state (chỉ chứa các phát hiện của frame hiện tại)
            for (x1, y1, x2, y2) in self.current_fire_boxes_on_display:
                 cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

            self._update_last_frame(display_frame, frame)

            if self.show_window:
                cv2.imshow("Guardian Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.quit = True
            else:
                time.sleep(0.001)

        self.delete()

    def delete(self):
        print("[Camera] Đang dọn dẹp và thoát...")
        self.quit = True
        if self.cap: self.cap.release()
        if self.show_window: cv2.destroyAllWindows()
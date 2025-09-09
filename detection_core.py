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
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

from config import EMBEDDING_FILE, NAMES_FILE, YOLO_MODEL_PATH, YOLO_PERSON_MODEL_PATH, MODEL_NAME, RECOGNITION_THRESHOLD, DATA_DIR, FRAMES_REQUIRED, PROCESS_EVERY_N_FRAMES, PROCESS_SIZE, DEBOUNCE_SECONDS, FIRE_CONFIDENCE_THRESHOLD

# --- Hook để main bind ---
# signature: on_alert_callback(frame, reason, name, meta)
# reason in: "nguoi_la", "nguoi_quen", "lua_chay"
on_alert_callback = None

# --- Global loaded models / data (load once) ---
print("[detection_core] Loading models...")
try:
    model = YOLO(YOLO_MODEL_PATH, task='detect')
except Exception as e:
    print("[detection_core] Warning: cannot load YOLO model from", YOLO_MODEL_PATH, e)
    model = None

try:
    model_person = YOLO(YOLO_PERSON_MODEL_PATH)
except Exception as e:
    print("[detection_core] Warning: cannot load YOLO person model from", YOLO_PERSON_MODEL_PATH, e)
    model_person = None

# InsightFace
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
        known_embeddings = []
        known_names = []
else:
    print("Đang mã hóa các khuôn mặt đã biết. Quá trình này có thể mất một lúc...")
    known_embeddings = []
    known_names = []
    for person_name in os.listdir(DATA_DIR):
        person_path = os.path.join(DATA_DIR, person_name)
        if not os.path.isdir(person_path):
            continue
        for filename in os.listdir(person_path):
            if not (filename.endswith(".jpg") or filename.endswith(".png")):
                continue
            filepath = os.path.join(person_path, filename)
            img = cv2.imread(filepath)
            if img is None:
                print(f"Lỗi: không thể đọc ảnh {filepath}")
                continue
            faces = app.get(img)
            if faces:
                embedding = faces[0].embedding
                known_embeddings.append(embedding)
                known_names.append(person_name)
                print(f"Đã mã hóa: {person_name}/{filename}")
            else:
                print(f"Cảnh báo: Không tìm thấy khuôn mặt nào trong ảnh {filepath}")
    if known_embeddings:
        try:
            with open(EMBEDDING_FILE, 'wb') as f:
                pickle.dump(known_embeddings, f)
            with open(NAMES_FILE, 'wb') as f:
                pickle.dump(known_names, f)
            print(f"Đã lưu {len(known_names)} vector đặc trưng vào bộ nhớ cache.")
        except Exception as e:
            print("[detection_core] Failed to save cache:", e)

# --- Utility functions ---
def match_face(embedding):
    """
    So sánh embedding với known_embeddings, trả về (best_name, best_distance)
    Nếu không có known embeddings, trả về (None, inf)
    """
    if not known_embeddings:
        return None, float('inf')
    best_d = float('inf')
    best_name = None
    for k_emb, k_name in zip(known_embeddings, known_names):
        d = float(cosine(embedding, k_emb))
        if d < best_d:
            best_d = d
            best_name = k_name
    return best_name, best_d

def update_known_data():
    global known_embeddings, known_names
    try:
        with open(EMBEDDING_FILE, 'rb') as f:
            known_embeddings = pickle.load(f)
        with open(NAMES_FILE, 'rb') as f:
            known_names = pickle.load(f)
        print(f"[detection_core] Known data updated. Total faces: {len(known_names)}")
    except Exception as e:
        print(f"[detection_core] Failed to update known data: {e}")

def update_model(selected):
    global app
    app = FaceAnalysis(name=selected, root="Data/Model", allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=-1, det_size=(640, 640))

# Tracker helper
def create_tracker_prefer_csrt():
    """
    Try multiple ways to create a tracker. Return tracker object or None.
    """
    # Try legacy CSRT
    try:
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
    except Exception:
        pass
    # Try top-level CSRT
    try:
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
    except Exception:
        pass
    # Fallback to other tracker types
    for name in ("MOSSE", "KCF", "MIL", "TLD", "BOOSTING"):
        try:
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, f"Tracker{name}_create"):
                return getattr(cv2.legacy, f"Tracker{name}_create")()
            if hasattr(cv2, f"Tracker{name}_create"):
                return getattr(cv2, f"Tracker{name}_create")()
        except Exception:
            continue
    return None

# --- Camera class ---
class Camera:
    def __init__(self, src=0, show_window=True):
        # Use src param properly
        # If src is int-like, pass as int; else pass as string (e.g., rtsp)
        try:
            src_param = int(src)
        except Exception:
            src_param = src
        # try different API preference could be added; keep default
        self.cap = cv2.VideoCapture(src_param, cv2.CAP_DSHOW) if isinstance(src_param, int) else cv2.VideoCapture(src_param)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera/source: {src}")
        self.quit = False

        # queues
        self.face_queue = queue.Queue(maxsize=2)
        self.fire_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=16)  # sẽ chứa tuples ("face"/"fire", results)

        # tracking & debounce
        self.last_alert_time = defaultdict(lambda: 0.0)
        self.recognized_counts = {}

        # parameters
        self.FRAMES_REQUIRED = FRAMES_REQUIRED
        self.PROCESS_EVERY_N_FRAMES = PROCESS_EVERY_N_FRAMES
        # Ensure PROCESS_SIZE is (width, height)
        if len(PROCESS_SIZE) == 2:
            self.PROC_W, self.PROC_H = PROCESS_SIZE[0], PROCESS_SIZE[1]
        else:
            self.PROC_W, self.PROC_H = 320, 240

        # start workers
        self.face_thread = threading.Thread(target=self.face_worker, daemon=True)
        self.fire_thread = threading.Thread(target=self.fire_worker, daemon=True)
        self.face_thread.start()
        self.fire_thread.start()

        self.show_window = show_window
        self._frame_lock = threading.Lock()
        self._last_frame = None

        # tracker
        self.tracker = None
        self.tracking_bbox = None

        # frame counter for sampling
        self._frame_idx = 0

    def face_worker(self):
        """Worker thực hiện detect bằng insightface và so sánh embedding"""
        while not self.quit:
            try:
                frame = self.face_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                faces = app.get(frame)
                face_results = []
                for face in faces:
                    try:
                        x1, y1, x2, y2 = face.bbox.astype(int)
                    except Exception:
                        x1, y1, x2, y2 = 0, 0, 0, 0
                    emb = face.embedding
                    best_name, best_d = match_face(emb)
                    if best_name is None or best_d > RECOGNITION_THRESHOLD:
                        best_name = "Nguoi la"
                    face_results.append((x1, y1, x2, y2, best_name, best_d, uuid.uuid4().hex[:8]))
                try:
                    self.result_queue.put(("face", face_results), timeout=0.5)
                except queue.Full:
                    # best-effort drop if congested
                    pass
            except Exception as e:
                print("[detection_core.face_worker] Exception:", e)
                continue

    def fire_worker(self):
        """Worker dùng YOLO model chính để detect fire/smoke (hoặc các class khác), có ngưỡng confidence."""
        if model is None:
            while not self.quit:
                time.sleep(1.0)
            return
        while not self.quit:
            try:
                frame = self.fire_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                results = model(frame, verbose=False)
                fire_results = []
                if len(results) > 0 and hasattr(results[0], "boxes"):
                    for box in results[0].boxes:
                        # Lấy confidence score
                        conf = float(box.conf[0]) if hasattr(box, "conf") else 1.0
                        if conf < FIRE_CONFIDENCE_THRESHOLD:
                            continue
                        coords = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, coords)
                        cls_id = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
                        cls_name = results[0].names.get(cls_id, str(cls_id)) if hasattr(results[0], "names") else str(cls_id)
                        # Chỉ giữ các class có tên liên quan tới fire/smoke
                        if cls_name.lower() in ("fire", "smoke", "flame"):
                            fire_results.append((x1, y1, x2, y2, "lua_chay"))
                try:
                    self.result_queue.put(("fire", fire_results), timeout=0.5)
                except queue.Full:
                    pass
            except Exception as e:
                print("[detection_core.fire_worker] Exception:", e)
                continue

    def _should_alert(self, key):
        last = self.last_alert_time.get(key, 0.0)
        if time.time() - last >= DEBOUNCE_SECONDS:
            self.last_alert_time[key] = time.time()
            return True
        return False

    def _update_last_frame(self, frame):
        with self._frame_lock:
            try:
                self._last_frame = frame.copy()
            except Exception:
                self._last_frame = frame

    def read(self):
        with self._frame_lock:
            if self._last_frame is None:
                return False, None
            try:
                return True, self._last_frame.copy()
            except Exception:
                return True, self._last_frame

    def detect_object(self, frame):
        # # Use YOLO person model to detect person for tracker initialization.
        # if model_person is not None:
        #     try:
        #         results = model_person(frame, verbose=False)
        #         if results and len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
        #             box = results[0].boxes[0]
        #             coords = box.xyxy[0].tolist()
        #             x1, y1, x2, y2 = map(int, coords)
        #             return (x1, y1, x2 - x1, y2 - y1)
        #     except Exception as e:
        #         # model_person may fail on some frames; swallow and return None
        #         print("[detection_core.detect_object] person model exception:", e)
        return None

    def detect(self):
        """
        Vòng lặp chính: đọc camera, gửi frames cho worker, lấy kết quả từ result_queue và gọi on_alert_callback khi cần.
        """
        while not self.quit:
            ret, frame = self.cap.read()
            if not ret:
                # no frame, small sleep to avoid hot loop
                time.sleep(0.01)
                continue

            # update last frame buffer
            self._update_last_frame(frame)

            orig_h, orig_w = frame.shape[:2]
            self._frame_idx += 1

            # enqueue small frames for workers every N frames
            if (self._frame_idx % max(1, self.PROCESS_EVERY_N_FRAMES)) == 0:
                try:
                    small_frame = cv2.resize(frame, (self.PROC_W, self.PROC_H), interpolation=cv2.INTER_AREA)  # resize with INTER_AREA for better quality when downscaling
                except Exception:
                    # fallback to copy if resize fails
                    small_frame = frame.copy()
                if not self.face_queue.full():
                    try:
                        self.face_queue.put_nowait(small_frame)
                    except Exception:
                        pass
                if not self.fire_queue.full():
                    try:
                        self.fire_queue.put_nowait(small_frame)
                    except Exception:
                        pass

            # Tracker init/update
            if self.tracker is None:
                bbox = self.detect_object(frame)
                if bbox is not None:
                    tracker = create_tracker_prefer_csrt()
                    if tracker is not None:
                        try:
                            tracker.init(frame, tuple(map(float, bbox)))
                            self.tracker = tracker
                            self.tracking_bbox = bbox
                        except Exception as e:
                            print("[detection_core] Tracker init failed:", e)
                            self.tracker = None
                    else:
                        print("[detection_core] No tracker available in this OpenCV build.")
            else:
                try:
                    success, bbox = self.tracker.update(frame)
                except Exception as e:
                    success = False
                    bbox = None
                    print("[detection_core] Tracker update exception:", e)
                if success and bbox is not None:
                    self.tracking_bbox = bbox
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                else:
                    self.tracker = None

            # consume results
            try:
                while not self.result_queue.empty():
                    typ, results = self.result_queue.get_nowait()
                    if typ == "face":
                        # results given in small_frame coordinates (PROC_W, PROC_H)
                        scale_x = orig_w / float(self.PROC_W) if self.PROC_W else 1.0
                        scale_y = orig_h / float(self.PROC_H) if self.PROC_H else 1.0
                        scaled_faces = []
                        for (x1, y1, x2, y2, best_name, best_d, fid) in results:
                            x1s = int(max(0, min(orig_w, round(x1 * scale_x))))
                            y1s = int(max(0, min(orig_h, round(y1 * scale_y))))
                            x2s = int(max(0, min(orig_w, round(x2 * scale_x))))
                            y2s = int(max(0, min(orig_h, round(y2 * scale_y))))
                            scaled_faces.append((x1s, y1s, x2s, y2s, best_name, best_d, fid))

                            color = (0, 255, 0) if best_name != "Nguoi la" else (0, 0, 255)
                            cv2.rectangle(frame, (x1s, y1s), (x2s, y2s), color, 2)
                            cv2.putText(frame, f"{best_name} ({best_d:.2f})", (x1s, max(0, y1s - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            print(f"[Face] {best_name} - distance={best_d:.2f}")

                        for x1s, y1s, x2s, y2s, best_name, best_d, fid in scaled_faces:
                            if best_name == "Nguoi la":
                                key = ("nguoi_la",)
                                if self._should_alert(key) and on_alert_callback:
                                    try:
                                        on_alert_callback(frame.copy(), "nguoi_la", None, {"distance": best_d})
                                    except Exception as e:
                                        print("[detection_core] on_alert_callback exception:", e)
                            else:
                                cnt = self.recognized_counts.get(best_name, 0) + 1
                                self.recognized_counts[best_name] = cnt
                                if cnt >= self.FRAMES_REQUIRED:
                                    key = ("nguoi_quen", best_name)
                                    if self._should_alert(key) and on_alert_callback:
                                        try:
                                            on_alert_callback(frame.copy(), "nguoi_quen", best_name, {"distance": best_d})
                                        except Exception as e:
                                            print("[detection_core] on_alert_callback exception:", e)
                                    self.recognized_counts[best_name] = 0

                    elif typ == "fire":
                        for (x1, y1, x2, y2, label) in results:
                            if label == "lua_chay":
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # cam đỏ
                                cv2.putText(frame, "FIRE", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                                key = ("lua_chay",)
                                if self._should_alert(key) and on_alert_callback:
                                    try:
                                        on_alert_callback(frame.copy(), "lua_chay", None, {"label": label})
                                    except Exception as e:
                                        print("[detection_core] on_alert_callback exception:", e)
            except queue.Empty:
                pass
            except Exception as e:
                print("[detection_core] Exception while processing results:", e)

            if self.show_window:
                display_frame = cv2.resize(frame, (self.PROC_W, self.PROC_H), interpolation=cv2.INTER_AREA)  # resize for display
                cv2.imshow("Detection (press q to quit)", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.quit = True
                    break

        self.delete()

    def delete(self):
        self.quit = True
        try:
            self.cap.release()
        except:
            pass
        try:
            cv2.destroyAllWindows()
        except:
            pass

# convenience for main
def get_known_data():
    return known_embeddings, known_names

__all__ = ['Camera', 'app', 'match_face', 'update_known_data']

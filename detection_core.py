# detection_core.py
import os
import time
import uuid
import threading
import queue
import pickle
import copy
from collections import defaultdict

import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

# --- FILES / CONFIG (cập nhật nếu cần) ---
EMBEDDING_FILE = "Data/known_embeddings.pkl"
NAMES_FILE = "Data/known_names.pkl"
DATA_DIR = "Data/Image"
MODEL_NAME = 'buffalo_s'
RECOGNITION_THRESHOLD = 0.45   # ngưỡng so sánh embedding (càng nhỏ càng strict)
FRAMES_REQUIRED = 3            # số frame liên tiếp để xác nhận một face
PROCESS_EVERY_N_FRAMES = 3
PROCESS_SIZE = (416, 416)
DEBOUNCE_SECONDS = 30         # thời gian debounce cho cùng 1 alert (theo type/name)

# --- Hook để main bind ---
# signature: on_alert_callback(frame, reason, name, meta)
# reason in: "nguoi_la", "nguoi_quen", "lua_chay"
on_alert_callback = None

# --- Global loaded models / data (load once) ---
print("[detection_core] Loading models...")
# YOLO model dùng để detect fire/smoke/person
# Nếu bạn chỉ dùng 1 model, đặt path tương ứng
YOLO_MODEL_PATH = "Data/Model/model_openvino_model"   # model detect fire/smoke (and possibly person)
YOLO_PERSON_MODEL_PATH = "Data/Model/yolo11n_openvino_model"  # optional separate person detector

# NẾU bạn chỉ có 1 model, có thể dùng same path for both variables
try:
    model = YOLO(YOLO_MODEL_PATH, task='detect')
except Exception as e:
    print("[detection_core] Warning: cannot load YOLO model from", YOLO_MODEL_PATH, e)
    model = None

try:
    model_person = YOLO(YOLO_PERSON_MODEL_PATH)
except Exception as e:
    # nếu không có model_person, ta vẫn có thể rely vào insightface detection trực tiếp
    print("[detection_core] Warning: cannot load YOLO person model from", YOLO_PERSON_MODEL_PATH, e)
    model_person = None

# InsightFace
app = FaceAnalysis(name=MODEL_NAME, root="Data/Model", allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 480))
print("[detection_core] InsightFace ready.")

# --- MÃ HÓA CÁC KHUÔN MẶT ĐÃ BIẾT ---
if os.path.exists(EMBEDDING_FILE) and os.path.exists(NAMES_FILE):
    print("Đang tải dữ liệu khuôn mặt đã biết từ bộ nhớ cache...")
    with open(EMBEDDING_FILE, 'rb') as f:
        known_embeddings = pickle.load(f)
    with open(NAMES_FILE, 'rb') as f:
        known_names = pickle.load(f)
    print(f"Đã tải {len(known_names)} khuôn mặt đã biết.")
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
        print(f"Đã lưu {len(known_names)} vector đặc trưng vào bộ nhớ cache.")
        with open(EMBEDDING_FILE, 'wb') as f:
            pickle.dump(known_embeddings, f)
        with open(NAMES_FILE, 'wb') as f:
            pickle.dump(known_names, f)

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

# --- Camera class ---
class Camera:
    """
    Camera wrapper: dùng cv2.VideoCapture nội bộ, 2 worker threads:
     - face_worker: nhận frames từ face_queue, chạy InsightFace, trả kết quả về result_queue
     - fire_worker: nhận frames từ fire_queue, chạy YOLO model (fire/smoke), trả result_queue
    detect() là vòng lặp chính: đọc camera, gửi frames cho worker, nhận kết quả và gọi on_alert_callback khi cần.
    """

    def __init__(self, src=0, show_window=True):
        # Sử dụng composition thay vì subclassing cv2.VideoCapture để linh hoạt
        # self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap= cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        self.quit = False

        # queues
        self.face_queue = queue.Queue(maxsize=2)
        self.fire_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=8)  # sẽ chứa tuples ("face"/"fire", results)

        # tracking & debounce
        self.last_alert_time = defaultdict(lambda: 0.0)  # keys: ("nguoi_la","nguoi_quen","lua_chay") or ("nguoi_quen", name)
        self.recognized_counts = {}  # name -> consecutive frames count (reset if not seen)

        # parameters
        self.FRAMES_REQUIRED = FRAMES_REQUIRED
        self.PROCESS_EVERY_N_FRAMES = PROCESS_EVERY_N_FRAMES
        self.PROCESS_SIZE = PROCESS_SIZE

        # start workers
        self.face_thread = threading.Thread(target=self.face_worker, daemon=True)
        self.fire_thread = threading.Thread(target=self.fire_worker, daemon=True)
        self.face_thread.start()
        self.fire_thread.start()

        self.show_window = show_window

        self._frame_lock = threading.Lock()
        self._last_frame = None
        self.quit = False

    def face_worker(self):
        """Worker thực hiện detect bằng insightface và so sánh embedding"""
        while not self.quit:
            try:
                frame = self.face_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                # Chạy insightface trên frame (frame có thể là crop hoặc toàn bộ frame tuỳ caller)
                faces = app.get(frame)
                face_results = []
                for face in faces:
                    try:
                        x1, y1, x2, y2 = face.bbox.astype(int)
                    except Exception:
                        x1, y1, x2, y2 = 0, 0, 0, 0
                    emb = face.embedding
                    best_name, best_d = match_face(emb)
                    # Nếu không có known embeddings thì best_name=None, treat as unknown
                    if best_name is None:
                        best_name = "Nguoi la"
                        best_d = float('inf')
                    else:
                        # nếu distance lớn hơn threshold => không nhận diện
                        if best_d > RECOGNITION_THRESHOLD:
                            best_name = "Nguoi la"
                    face_results.append((x1, y1, x2, y2, best_name, best_d, uuid.uuid4().hex[:8]))
                # gửi kết quả (dù rỗng)
                self.result_queue.put(("face", face_results))
            except Exception as e:
                # không crash worker
                print("[detection_core.face_worker] Exception:", e)
                continue

    def fire_worker(self):
        """Worker dùng YOLO model chính để detect fire/smoke (hoặc các class khác)"""
        if model is None:
            # không có model => worker chỉ sleep
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
                # results[0].boxes có thể rỗng
                if len(results) > 0 and hasattr(results[0], "boxes"):
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
                        cls_name = results[0].names.get(cls_id, str(cls_id)) if hasattr(results[0], "names") else str(cls_id)
                        fire_results.append((x1, y1, x2, y2, cls_name))
                # push fire results
                self.result_queue.put(("fire", fire_results))
            except Exception as e:
                print("[detection_core.fire_worker] Exception:", e)
                continue

    def _should_alert(self, key):
        """Kiểm tra debounce; key có thể là ('lua_chay'), ('nguoi_quen', name), ('nguoi_la')"""
        last = self.last_alert_time.get(key, 0.0)
        if time.time() - last >= DEBOUNCE_SECONDS:
            self.last_alert_time[key] = time.time()
            return True
        return False
    
    def _update_last_frame(self, frame):
        """Gọi từ detect() mỗi khi có frame mới để lưu lại (thread-safe)."""
        with self._frame_lock:
            try:
                self._last_frame = frame.copy()
            except Exception:
                self._last_frame = frame

    def read(self):
        """API giống cv2.VideoCapture.read(): trả về (ret, frame)."""
        with self._frame_lock:
            if self._last_frame is None:
                return False, None
            try:
                return True, self._last_frame.copy()
            except Exception:
                return True, self._last_frame


    def detect(self):
        """
        Vòng lặp chính: đọc camera, gửi frames cho worker, lấy kết quả từ result_queue và gọi on_alert_callback khi cần.
        Lưu ý: để giảm latency, face_queue và fire_queue sẽ nhận 'small_frame' (resize) nhưng khi gửi alert ta sẽ dùng frame gốc (full) để lưu ảnh rõ nét.
        """
        frame_counter = 0

        while not self.quit:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("[detection_core] Cannot read frame from camera")
                time.sleep(0.1)
                continue

            if ret:
                # cập nhật last_frame để thread recorder có thể lấy
                try:
                    self._update_last_frame(frame)
                except Exception:
                    pass

            frame = cv2.resize(frame, (1280, 720))

            original_h, original_w = frame.shape[:2]
            small_frame = cv2.resize(frame, self.PROCESS_SIZE)
            frame_counter += 1

            # gửi frame cho worker mỗi N frame
            if frame_counter % self.PROCESS_EVERY_N_FRAMES == 0:
                # push small_frame for faster processing
                try:
                    if not self.face_queue.full():
                        self.face_queue.put_nowait(small_frame.copy())
                except queue.Full:
                    pass
                try:
                    if not self.fire_queue.full():
                        self.fire_queue.put_nowait(small_frame.copy())
                except queue.Full:
                    pass

            # consume results (nhiều kết quả có thể được trả về)
            try:
                # process all available results
                while not self.result_queue.empty():
                    typ, results = self.result_queue.get_nowait()
                    if typ == "face":
                        # results: list of (x1,y1,x2,y2,best_name,best_d,face_id)
                        # Map coordinates from small_frame->original_frame scale
                        scaled_faces = []
                        for (x1, y1, x2, y2, best_name, best_d, fid) in results:
                            # scale coordinates (PROCESS_SIZE -> original)
                            x1s = int(x1 * original_w / self.PROCESS_SIZE[0])
                            y1s = int(y1 * original_h / self.PROCESS_SIZE[1])
                            x2s = int(x2 * original_w / self.PROCESS_SIZE[0])
                            y2s = int(y2 * original_h / self.PROCESS_SIZE[1])
                            scaled_faces.append((x1s, y1s, x2s, y2s, best_name, best_d, fid))

                            # --- Vẽ box & in ra console ---
                            color = (0, 255, 0) if best_name != "Nguoi la" else (0, 0, 255)
                            cv2.rectangle(frame, (x1s, y1s), (x2s, y2s), color, 2)
                            cv2.putText(frame, f"{best_name} ({best_d:.2f})",
                                        (x1s, y1s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            print(f"[Face] {best_name} - distance={best_d:.2f}")

                        # xử lý từng face: gọi callback nếu đủ điều kiện (debounce, frames required)
                        for x1s, y1s, x2s, y2s, best_name, best_d, fid in scaled_faces:
                            if best_name == "Nguoi la":
                                key = ("nguoi_la",)  # chung cho unknown
                                if self._should_alert(key):
                                    if on_alert_callback:
                                        # gửi frame gốc (full res) cho main để save/send
                                        try:
                                            on_alert_callback(frame.copy(), "nguoi_la", None, {"distance": best_d})
                                        except Exception as e:
                                            print("[detection_core] on_alert_callback exception:", e)
                            else:
                                # known person: chúng ta có thể require nhiều frames liên tiếp trước khi alert known
                                cnt = self.recognized_counts.get(best_name, 0) + 1
                                self.recognized_counts[best_name] = cnt
                                if cnt >= self.FRAMES_REQUIRED:
                                    key = ("nguoi_quen", best_name)
                                    if self._should_alert(key):
                                        if on_alert_callback:
                                            try:
                                                on_alert_callback(frame.copy(), "nguoi_quen", best_name, {"distance": best_d})
                                            except Exception as e:
                                                print("[detection_core] on_alert_callback exception:", e)
                                    # reset count so not spam
                                    self.recognized_counts[best_name] = 0

                    elif typ == "fire":
                        # results: list of (x1,y1,x2,y2,label)
                        # nếu có label 'fire' hoặc 'smoke' thì alert; để giảm nhầm lẫn, ta gọi _should_alert per label
                        for (x1, y1, x2, y2, label) in results:
                            labl = str(label).lower()
                            if "fire" in labl or "smoke" in labl or "lava" in labl:
                                key = ("lua_chay",)
                                if self._should_alert(key):
                                    if on_alert_callback:
                                        try:
                                            on_alert_callback(frame.copy(), "lua_chay", None, {"label": label})
                                        except Exception as e:
                                            print("[detection_core] on_alert_callback exception:", e)
                            else:
                                # bạn có thể xử lý thêm label khác nếu muốn
                                pass
            except queue.Empty:
                pass
            except Exception as e:
                print("[detection_core] Exception while processing results:", e)

            # (TÙY CHỌN) hiển thị frame để debug — nếu môi trường có GUI
            if self.show_window:
                # VẼ nhanh các label từ last processed results (nếu muốn), 
                # nhưng để đơn giản ở đây chúng ta không vẽ (main có thể vẽ nếu cần)
                cv2.imshow("Detection (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.quit = True
                    break

        # cleanup
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

if __name__ == "__main__":
    cam = Camera(show_window=True)
    cam.detect()

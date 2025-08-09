import cv2, pickle, os
from PIL import Image
import threading
import queue
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import time
import uuid

# --- YOUR ORIGINAL CONSTANTS (keep as-is) ---
EMBEDDING_FILE = "Data/known_embeddings.pkl"
NAMES_FILE = "Data/known_names.pkl"
DATA_DIR = "Data/Image"
MODEL_NAME = 'buffalo_s'
RECOGNITION_THRESHOLD = 0.4

# --- ALERT CALLBACK (set by main) ---
# signature: on_alert_callback(frame, reason, name, alert_meta_dict)
# reason in: "nguoi_la", "nguoi_quen", "lua_chay"
on_alert_callback = None

frame_count = 0

print("Load model...")
# --- KHỞI TẠO MÔ HÌNH ---
model = YOLO("Data/Model\\model_openvino_model", task='detect')
model_person = YOLO("Data/Model\\yolo11n_openvino_model")
app = FaceAnalysis(name=MODEL_NAME, root="Data/Model", allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 480))  # ctx_id=-1 để dùng CPU
print("Load data...")

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

class Camera(cv2.VideoCapture):
    def __init__(self):
        super().__init__(0, cv2.CAP_DSHOW)
        self.quit = False
        self.face_queue = queue.Queue(maxsize=1)
        self.fire_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)

        self.alerting_stranger = False
        self.last_stranger_alert_time = 0
        
        # Added for face recognition tracking
        self.recognized_faces = {}  # key: recognized name, value: {id, frame_count, confirmed}
        self.FRAMES_REQUIRED = 5  # số frame liên tục để xác nhận khuôn mặt

    def start_stranger_alert_loop(self, frame):
        def alert_loop():
            start_time = time.time()
            while time.time() - start_time <= 10:  # Gửi trong 10s
                # call callback if set
                if on_alert_callback:
                    on_alert_callback(frame.copy(), "nguoi_la", None, {"note":"looping"})
                time.sleep(10)
            self.alerting_stranger = False

        self.alerting_stranger = True
        threading.Thread(target=alert_loop, daemon=True).start()

    def face_recognition_thread(self):
        while not self.quit:
            try:
                frame = self.face_queue.get(timeout=1)
                # 1. Sử dụng model_person để phát hiện người trong khung hình
                person_results = model_person(frame, verbose=False)
                if person_results and len(person_results[0].boxes) > 0:
                    # 2. Nếu phát hiện có người, tiến hành nhận diện khuôn mặt
                    faces = app.get(frame)
                    face_results = []   # Danh sách chứa thông tin nhận diện khuôn mặt để theo dõi và cảnh báo
                    for face in faces:
                        embedding = face.embedding
                        best_distance = float('inf')
                        best_name = None
                        for known_emb, known_name in zip(known_embeddings, known_names):
                            dist = cosine(embedding, known_emb)
                            if dist < best_distance:
                                best_distance = dist
                                best_name = known_name
                        if best_distance < RECOGNITION_THRESHOLD:
                            # Lấy bounding box từ đối tượng face (nếu có thuộc tính bbox)
                            bbox = face.bbox if hasattr(face, 'bbox') else [0, 0, 0, 0]
                            try:
                                bbox = [int(x) for x in bbox]
                            except Exception:
                                bbox = [0, 0, 0, 0]
                            
                            # Cập nhật thông tin nhận diện cho khuôn mặt
                            rec = self.recognized_faces.get(best_name, {'id': None, 'frame_count': 0, 'confirmed': False})
                            rec['frame_count'] += 1
                            if rec['frame_count'] >= self.FRAMES_REQUIRED and not rec['confirmed']:
                                rec['confirmed'] = True
                                rec['id'] = uuid.uuid4().hex
                            self.recognized_faces[best_name] = rec
                            
                            # Nếu đã xác nhận, thêm thông tin nhận diện vào danh sách để theo dõi và cảnh báo
                            if rec['confirmed']:
                                face_results.append((bbox[0], bbox[1], bbox[2], bbox[3], best_name, best_distance, rec['id']))
                    if face_results:
                        self.result_queue.put(("face", face_results))
            except queue.Empty:
                pass

    def fire_smoke_thread(self):
        while not self.quit:
            try:
                frame = self.fire_queue.get(timeout=1)
                results = model(frame, verbose=False)
                fire_results = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = results[0].names[int(box.cls)]
                    fire_results.append((x1, y1, x2, y2, label))
                self.result_queue.put(("fire", fire_results))
            except queue.Empty:
                continue

    def detect(self):
        self.last_face_results = []
        self.last_fire_results = []

        frame_counter = 0
        PROCESS_EVERY_N_FRAMES = 3
        PROCESS_SIZE = (416, 416)

        t1 = threading.Thread(target=self.face_recognition_thread, daemon=True)
        t2 = threading.Thread(target=self.fire_smoke_thread, daemon=True)
        t1.start()
        t2.start()

        while not self.quit:
            ret, frame = self.read()
            if not ret:
                print("Không thể đọc khung hình từ camera")
                break

            original_h, original_w, _ = frame.shape
            small_frame = cv2.resize(frame, PROCESS_SIZE)
            frame_counter += 1

            person_results = model_person(frame, verbose=False)
            if person_results and len(person_results[0].boxes) > 0:
                for box in person_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                if not self.face_queue.full():
                    self.face_queue.put(small_frame.copy())
                if not self.fire_queue.full():
                    self.fire_queue.put(small_frame.copy())

            try:
                while not self.result_queue.empty():
                    typ, results = self.result_queue.get_nowait()
                    if typ == "face":
                        self.last_face_results = results
                        # If unknown found -> call callback
                        for x1, y1, x2, y2, best_name, best_distance, rec_id in results:
                            if best_name == "Nguoi la":
                                if on_alert_callback:
                                    on_alert_callback(frame.copy(), "nguoi_la", None, {"distance": best_distance})
                                break
                            else:
                                if on_alert_callback:
                                    on_alert_callback(frame.copy(), "nguoi_quen", best_name, {"distance": best_distance})
                                break
                    elif typ == "fire":
                        self.last_fire_results = results
                        for x1, y1, x2, y2, label in results:
                            if label.lower() in ["fire", "smoke"]:
                                if on_alert_callback:
                                    on_alert_callback(frame.copy(), "lua_chay", None, {"label": label})
                                break
            except queue.Empty:
                pass

            # draw face boxes
            for x1, y1, x2, y2, best_name, best_distance, rec_id in self.last_face_results:
                x1_s = int(x1 * original_w / PROCESS_SIZE[0])
                y1_s = int(y1 * original_h / PROCESS_SIZE[1])
                x2_s = int(x2 * original_w / PROCESS_SIZE[0])
                y2_s = int(y2 * original_h / PROCESS_SIZE[1])
                color = (0, 255, 0) if best_name != "Nguoi la" else (0, 0, 255)
                cv2.rectangle(frame, (x1_s, y1_s), (x2_s, y2_s), color, 2)
                text = f"{best_name} ({best_distance:.2f}) | ID: {rec_id}"
                cv2.putText(frame, text, (x1_s, y1_s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # draw fire boxes
            for x1, y1, x2, y2, label in self.last_fire_results:
                x1_s = int(x1 * original_w / PROCESS_SIZE[0])
                y1_s = int(y1 * original_h / PROCESS_SIZE[1])
                x2_s = int(x2 * original_w / PROCESS_SIZE[0])
                y2_s = int(y2 * original_h / PROCESS_SIZE[1])
                color = (0, 0, 255)
                cv2.rectangle(frame, (x1_s, y1_s), (x2_s, y2_s), color, 2)
                cv2.putText(frame, label, (x1_s, y1_s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.quit = True
                break

        self.delete()

    def delete(self):
        self.quit = True
        self.release()
        cv2.destroyAllWindows()

def get_known_data():
    return known_embeddings, known_names
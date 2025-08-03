
import cv2, pickle, os#, utils
from PIL import Image
import threading
import queue
from ultralytics import YOLO
from Lib.insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine


EMBEDDING_FILE = "Data/known_embeddings.pkl"
NAMES_FILE = "Data/known_names.pkl"
DATA_DIR = "Data/Image"
MODEL_NAME = 'buffalo_s'
RECOGNITION_THRESHOLD = 0.4

frame_count = 0

print("Load model...")
# --- KHỞI TẠO MÔ HÌNH ---
model = YOLO("Data\Model\model_openvino_model", task='detect')
app = FaceAnalysis(name=MODEL_NAME, root="Data/Model", allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 480))  # ctx_id=-1 để dùng CPU
print("Load data...")


# --- MÃ HÓA CÁC KHUÔN MẶT ĐÃ BIẾT ---
# Kiểm tra xem các tệp embedding đã tồn tại chưa
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

    # Duyệt qua từng người trong thư mục dữ liệu
    for person_name in os.listdir(DATA_DIR):
        person_path = os.path.join(DATA_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        # Duyệt qua từng ảnh của người đó
        for filename in os.listdir(person_path):
            if not (filename.endswith(".jpg") or filename.endswith(".png")):
                continue

            filepath = os.path.join(person_path, filename)
            img = cv2.imread(filepath)

            if img is None:
                print(f"Lỗi: không thể đọc ảnh {filepath}")
                continue

            # Sử dụng insightface để phát hiện khuôn mặt
            faces = app.get(img)

            # Chỉ lấy khuôn mặt lớn nhất trong ảnh
            if faces:
                # Lấy vector đặc trưng trực tiếp từ đối tượng face
                embedding = faces[0].embedding
                known_embeddings.append(embedding)
                known_names.append(person_name)
                print(f"Đã mã hóa: {person_name}/{filename}")
            else:
                print(f"Cảnh báo: Không tìm thấy khuôn mặt nào trong ảnh {filepath}")

    # Lưu lại để lần sau chạy nhanh hơn
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

    # def get_image_camera(self):
    #     ret, frame = self.read()
    #     if not ret:
    #         print("Không thể đọc khung hình từ camera")
    #         return None
    #     frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    #     image = Image.fromarray(frame)
    #     base_64_image = utils.p2b(image)
    #     return base_64_image

    def face_recognition_thread(self):
        while not self.quit:
            try:
                frame = self.face_queue.get(timeout=1)
                # (Bạn có thể giữ lại logic LIFO từ câu trả lời trước nếu muốn)
                # while not self.face_queue.empty():
                #     try: frame = self.face_queue.get_nowait()
                #     except queue.Empty: break

                faces = app.get(frame)
                face_results = []
                for face in faces:
                    # ... (code nhận diện khuôn mặt của bạn giữ nguyên) ...
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    current_embedding = face.embedding
                    best_distance = 1.0
                    best_name = "Nguoi la"
                    for i, known_embedding in enumerate(known_embeddings):
                        distance = cosine(current_embedding, known_embedding)
                        if distance < best_distance:
                            best_distance = distance
                            best_name = known_names[i] if distance <= RECOGNITION_THRESHOLD else "Nguoi la"
                    face_results.append((x1, y1, x2, y2, best_name, best_distance))
                
                # --- SỬA ĐỔI QUAN TRỌNG ---
                # Luôn gửi kết quả, kể cả khi nó rỗng
                self.result_queue.put(("face", face_results))

            except queue.Empty:
                continue

    def fire_smoke_thread(self):
        while not self.quit:
            try:
                frame = self.fire_queue.get(timeout=1)
                # (Bạn có thể giữ lại logic LIFO)
                # while not self.fire_queue.empty():
                #     try: frame = self.fire_queue.get_nowait()
                #     except queue.Empty: break

                # Gọi model cho 1 frame
                results = model(frame, verbose=False) # Thêm verbose=False để tắt log của YOLO
                fire_results = []

                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Bạn cần lấy tên class từ model, ở đây tôi giả sử class 0 là "Fire"
                    label = results[0].names[int(box.cls)]
                    fire_results.append((x1, y1, x2, y2, label))
                
                # --- SỬA ĐỔI QUAN TRỌNG ---
                # Luôn gửi kết quả, kể cả khi nó rỗng
                self.result_queue.put(("fire", fire_results))

            except queue.Empty:
                continue

    def detect(self):
        # --- BIẾN LƯU TRỮ KẾT QUẢ GẦN NHẤT ---
        self.last_face_results = []
        self.last_fire_results = []
        
        # --- CÁC THAM SỐ TỐI ƯU HÓA ---
        frame_counter = 0
        PROCESS_EVERY_N_FRAMES = 3  # Chỉ xử lý mỗi 3 frame
        PROCESS_SIZE = (416, 416)   # Resize ảnh về kích thước này để xử lý nhanh hơn

        # Start threads (sử dụng các hàm thread đã được sửa để luôn trả về kết quả)
        t1 = threading.Thread(target=self.face_recognition_thread, daemon=True)
        t2 = threading.Thread(target=self.fire_smoke_thread, daemon=True)
        t1.start()
        t2.start()

        while not self.quit:
            ret, frame = self.read()
            if not ret:
                print("Không thể đọc khung hình từ camera")
                break
            
            # Lấy kích thước frame gốc để scale kết quả sau này
            original_h, original_w, _ = frame.shape
            
            # Resize ảnh về kích thước nhỏ để xử lý
            small_frame = cv2.resize(frame, PROCESS_SIZE)
            
            frame_counter += 1
            
            # Chỉ gửi frame đi xử lý mỗi N frame
            if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                if not self.face_queue.full():
                    self.face_queue.put(small_frame.copy())
                if not self.fire_queue.full():
                    self.fire_queue.put(small_frame.copy())

            # Lấy kết quả mới (nếu có) mà không chờ đợi
            try:
                while not self.result_queue.empty():
                    typ, results = self.result_queue.get_nowait()
                    if typ == "face":
                        self.last_face_results = results
                    elif typ == "fire":
                        self.last_fire_results = results
            except queue.Empty:
                pass

            # --- VẼ KẾT QUẢ ĐÃ SCALE LÊN FRAME GỐC ---

            # Vẽ kết quả nhận diện khuôn mặt
            for x1, y1, x2, y2, best_name, best_distance in self.last_face_results:
                # Scale tọa độ từ PROCESS_SIZE về kích thước frame gốc
                x1_s = int(x1 * original_w / PROCESS_SIZE[0])
                y1_s = int(y1 * original_h / PROCESS_SIZE[1])
                x2_s = int(x2 * original_w / PROCESS_SIZE[0])
                y2_s = int(y2 * original_h / PROCESS_SIZE[1])
                
                # <<< SỬA LỖI Ở ĐÂY: Điền đầy đủ tham số vẽ >>>
                color = (0, 255, 0) if best_name != "Nguoi la" else (0, 0, 255)
                cv2.rectangle(frame, (x1_s, y1_s), (x2_s, y2_s), color, 2)
                text = f"{best_name} ({best_distance:.2f})"
                cv2.putText(frame, text, (x1_s, y1_s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Vẽ kết quả phát hiện cháy
            for x1, y1, x2, y2, label in self.last_fire_results:
                # Scale tọa độ
                x1_s = int(x1 * original_w / PROCESS_SIZE[0])
                y1_s = int(y1 * original_h / PROCESS_SIZE[1])
                x2_s = int(x2 * original_w / PROCESS_SIZE[0])
                y2_s = int(y2 * original_h / PROCESS_SIZE[1])

                # <<< SỬA LỖI Ở ĐÂY: Điền đầy đủ tham số vẽ >>>
                color = (0, 0, 255) # Màu đỏ cho lửa
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

if __name__ == "__main__":
    a = Camera()
    a.detect()
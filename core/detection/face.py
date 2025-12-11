# core/detection/face.py
# =============================================================================
# MODULE NHẬN DIỆN KHUÔN MẶT - FACE RECOGNITION
# =============================================================================
# Module này dùng để:
# 1. Phát hiện khuôn mặt trong ảnh
# 2. Nhận diện đó là ai (so sánh với dữ liệu đã lưu)
# Sử dụng thư viện InsightFace (rất mạnh và chính xác)
# =============================================================================

import os
import pickle
import cv2
import numpy as np
from pathlib import Path

# Import thư viện nhận diện khuôn mặt
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from config import settings


# =============================================================================
# CLASS NHẬN DIỆN KHUÔN MẶT
# =============================================================================
class FaceDetector:
    
    # __slots__ giúp tiết kiệm bộ nhớ ram
    __slots__ = ('app', 'embeddings', 'names', 'det_path', 'rec_path')
    
    def __init__(self):
        self.app = None            # Ứng dụng InsightFace
        self.embeddings = []       # Danh sách đặc trưng mặt (vectors)
        self.names = []            # Danh sách tên tương ứng
        self.det_path = None
        self.rec_path = None
    
    def initialize(self, detector="Small", recognizer="Small"):
        """
        Khởi tạo hệ thống nhận diện khuôn mặt
        detector: Tên model phát hiện (Small/Medium/Large)
        recognizer: Tên model nhận diện
        
        Trả về: True nếu thành công
        """
        try:
            # Tối ưu hóa cho CPU
            os.environ['OMP_NUM_THREADS'] = '4'
            
            # Chọn thiết bị chạy (ưu tiên GPU nếu có)
            providers = ['CPUExecutionProvider']
            try:
                import onnxruntime as ort
                available = set(ort.get_available_providers())
                # Kiểm tra các loại GPU
                for p in ['CUDAExecutionProvider', 'OpenVINOExecutionProvider', 'DmlExecutionProvider']:
                    if p in available:
                        providers.insert(0, p)
                        break
            except ImportError:
                pass
            
            # Tìm file model trong thư mục
            model_dir = settings.paths.model_dir
            det_path = self.find_model(model_dir / detector, "detect")
            rec_path = self.find_model(model_dir / recognizer, "recog")
            
            if not det_path or not rec_path:
                print("❌ Không tìm thấy file model khuôn mặt!")
                return False
            
            # Tạo ứng dụng nhận diện
            self.app = self.create_app(det_path, rec_path, providers)
            
            # Chuẩn bị model (kích thước ảnh 640x640)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            
            print(f"✅ Đã khởi tạo nhận diện khuôn mặt ({detector}/{recognizer})")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khởi tạo nhận diện khuôn mặt: {e}")
            return False
    
    def find_model(self, directory, keyword):
        """Tìm file model ONNX trong thư mục"""
        if not directory.exists():
            return None
        
        # Duyệt qua các file .onnx
        for f in directory.glob("*.onnx"):
            if keyword in f.name.lower():
                return f
        return None
    
    def create_app(self, det_path, rec_path, providers):
        """
        Tạo ứng dụng FaceAnalysis tùy chỉnh
        Chúng ta cần class này để nạp model từ đường dẫn riêng
        """
        class CustomFaceAnalysis(FaceAnalysis):
            def __init__(self, det, rec, prov):
                self.models = {}
                # Nạp model detection (phát hiện)
                self.models['detection'] = get_model(str(det), providers=prov)
                # Nạp model recognition (nhận diện)
                self.models['recognition'] = get_model(str(rec), providers=prov)
                self.det_model = self.models['detection']
            
            def prepare(self, ctx_id, det_size):
                self.det_size = det_size
                for name, model in self.models.items():
                    if name == 'detection':
                        model.prepare(ctx_id, input_size=det_size)
                    else:
                        model.prepare(ctx_id)
        
        return CustomFaceAnalysis(det_path, rec_path, providers)
    
    def load_known_faces(self):
        """
        Tải dữ liệu khuôn mặt đã học từ file
        File này chứa:
        - embeddings: Vector đặc trưng của khuôn mặt
        - names: Tên người tương ứng
        """
        try:
            emb_file = settings.paths.data_dir / "known_embeddings.pkl"
            names_file = settings.paths.data_dir / "known_names.pkl"
            
            if not emb_file.exists() or not names_file.exists():
                return False
            
            # Đọc file binary
            with open(emb_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            with open(names_file, 'rb') as f:
                self.names = pickle.load(f)
            
            print(f"✅ Đã tải dữ liệu của {len(self.names)} người")
            return True
        except Exception:
            return False
    
    def detect_faces(self, image):
        """
        Phát hiện khuôn mặt trong ảnh
        Trả về danh sách các khuôn mặt tìm thấy
        Mỗi khuôn mặt có: bbox (vị trí), embedding (đặc trưng),...
        """
        if not self.app:
            return []
        
        # InsightFace tự động làm hết việc khó :)
        return self.app.get(image)
    
    def recognize(self, embedding):
        """
        Nhận diện xem khuôn mặt này là ai
        
        embedding: Vector đặc trưng của khuôn mặt mới phát hiện
        Trả về: (Tên người, Khoảng cách)
        """
        # Nếu chưa học ai cả thì chịu
        if not self.embeddings:
            return None, float('inf')
        
        try:
            # Chuyển về dạng numpy array để tính toán
            emb = np.array(embedding, dtype=np.float32)
            known = np.array(self.embeddings, dtype=np.float32)
            
            # ----- Chuẩn hóa vector -----
            # Để độ dài vector = 1, giúp so sánh chính xác hơn
            emb_norm = emb / np.linalg.norm(emb)
            known_norm = known / np.linalg.norm(known, axis=1, keepdims=True)
            
            # ----- Tính khoảng cách Cosine -----
            # Khoảng cách càng nhỏ = càng giống nhau
            # distance = 1 - độ tương đồng (dot product)
            distances = 1 - np.dot(known_norm, emb_norm)
            
            # Tìm người giống nhất (khoảng cách nhỏ nhất)
            idx = np.argmin(distances)
            dist = float(distances[idx])
            
            # Lấy ngưỡng chấp nhận từ cài đặt
            threshold = settings.detection.face_recognition_threshold
            
            # Nếu khoảng cách nhỏ hơn ngưỡng -> Là người đó!
            if dist <= threshold:
                return self.names[idx], dist
            
            # Nếu lớn hơn -> Không biết là ai
            return None, dist
            
        except Exception as e:
            print(f"Lỗi nhận diện: {e}")
            return None, float('inf')
    
    def rebuild_embeddings(self):
        """
        Học lại khuôn mặt từ thư mục ảnh (Data/Faces)
        Dùng khi bạn thêm ảnh mới vào thư mục
        """
        from utils import security
        
        faces_dir = settings.paths.faces_dir
        if not faces_dir.exists():
            return 0
        
        embeddings, names = [], []
        
        print("Đang học lại khuôn mặt từ ảnh...")
        
        # Duyệt qua từng thư mục tên người
        for person_dir in faces_dir.iterdir():
            if not person_dir.is_dir():
                continue
            
            # Duyệt qua từng ảnh của người đó
            for img_file in person_dir.glob("*.*"):
                if img_file.suffix.lower() not in ('.jpg', '.png', '.jpeg'):
                    continue
                
                try:
                    # Thử đọc ảnh (hỗ trợ ảnh mã hóa)
                    img = security.load_image(img_file)
                    if img is None:
                        img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    
                    # Phát hiện khuôn mặt trong ảnh
                    faces = self.detect_faces(img)
                    if faces:
                        # Lấy đặc trưng (embedding) của mặt đầu tiên tìm thấy
                        embeddings.append(faces[0].embedding)
                        names.append(person_dir.name)
                        print(f"  + Đã học: {person_dir.name}/{img_file.name}")
                except Exception as e:
                    print(f"  - Lỗi ảnh {img_file.name}: {e}")
        
        # Lưu lại vào file để lần sau dùng
        try:
            with open(settings.paths.data_dir / "known_embeddings.pkl", 'wb') as f:
                pickle.dump(embeddings, f)
            with open(settings.paths.data_dir / "known_names.pkl", 'wb') as f:
                pickle.dump(names, f)
            
            self.embeddings = embeddings
            self.names = names
            print(f"✅ Đã học xong {len(names)} khuôn mặt!")
            return len(names)
        except Exception:
            return 0
    
    # Các hàm property để lấy dữ liệu
    @property
    def known_names(self):
        return self.names
    
    @property
    def known_embeddings(self):
        return self.embeddings

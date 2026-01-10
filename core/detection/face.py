# Nhận diện khuôn mặt
import os
import pickle
import cv2
import numpy as np
import contextlib
from pathlib import Path

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from config import settings


# Class nhận diện người quen (FaceRec)
class FaceDetector:
    
    def __init__(self):
        self.app = None
        self.embeddings = []
        self.names = []
        self.det_path = None
        self.rec_path = None
        print("face detector init")
    
    # Khởi tạo model InsightFace
    def initialize(self):
        # Đọc config
        mode = settings.get('models.mode', 'Small')
        device = settings.get('models.device', 'cpu')
        
        # Xác định provider dựa trên device
        if device.lower() == 'gpu':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # tìm model
        model_dir = settings.paths.model_dir
        det_path = self.find_model(model_dir / mode, "detect")
        rec_path = self.find_model(model_dir / mode, "recog")
        
        if not det_path or not rec_path:
            print("khong thay model face")
            return False
        
        # tạo app
        self.app = self.create_app(det_path, rec_path, providers)
        
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        print(f"init face model: {mode} | device: {device}")
        return True
    
    # Tìm model file trong thư mục
    def find_model(self, directory, keyword):
        if not directory.exists():
            return None
        
        for f in directory.glob("*.onnx"):
            if keyword in f.name.lower():
                return f
        return None
    
    # Tạo app InsightFace
    def create_app(self, det_path, rec_path, providers):
        class CustomFaceAnalysis(FaceAnalysis):
            def __init__(self, det, rec, prov):
                self.models = {}
                # Tắt log khó chịu của onnxruntime
                with open(os.devnull, 'w') as fnull:
                    with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                        self.models['detection'] = get_model(str(det), providers=prov)
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
    
    # Load database khuôn mặt đã học
    def load_known_faces(self):
        emb_file = settings.paths.data_dir / "known_embeddings.pkl"
        names_file = settings.paths.data_dir / "known_names.pkl"
        
        if not emb_file.exists() or not names_file.exists():
            return False
        
        with open(emb_file, 'rb') as f:
            self.embeddings = pickle.load(f)
        with open(names_file, 'rb') as f:
            self.names = pickle.load(f)
        
        print(f"load dc {len(self.names)} nguoi")
        return True
    
    # Detect khuôn mặt trong ảnh
    def detect_faces(self, image):
        if not self.app:
            return []
        
        return self.app.get(image)
    
    # So sánh embedding để nhận diện
    def recognize(self, embedding):
        if not self.embeddings:
            return None, float('inf')
        
        emb = np.array(embedding, dtype=np.float32)
        known = np.array(self.embeddings, dtype=np.float32)
        
        # chuẩn hóa
        emb_norm = emb / np.linalg.norm(emb)
        known_norm = known / np.linalg.norm(known, axis=1, keepdims=True)
        
        # tính cosin
        distances = 1 - np.dot(known_norm, emb_norm)
        
        idx = np.argmin(distances)
        dist = float(distances[idx])
        
        threshold = settings.detection.face_recognition_threshold
        
        if dist <= threshold:
            print(f"nhan ra: {self.names[idx]} ({dist:.2f})")
            return self.names[idx], dist
        
        # print("khong biet la ai")
        return None, dist
    
    # Học lại khuôn mặt từ thư mục ảnh
    def rebuild_embeddings(self):
        from utils import security
        
        faces_dir = settings.paths.faces_dir
        if not faces_dir.exists():
            return 0
        
        embeddings, names = [], []
        
        print("bat dau hoc lai mat...")
        
        for person_dir in faces_dir.iterdir():
            if not person_dir.is_dir():
                continue
            
            for img_file in person_dir.glob("*.*"):
                if img_file.suffix.lower() not in ('.jpg', '.png', '.jpeg'):
                    continue
                
                img = security.load_image(img_file)
                if img is None:
                    img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                faces = self.detect_faces(img)
                if faces:
                    embeddings.append(faces[0].embedding)
                    names.append(person_dir.name)
                    print(f"Learn: {person_dir.name}")
        
        with open(settings.paths.data_dir / "known_embeddings.pkl", 'wb') as f:
            pickle.dump(embeddings, f)
        with open(settings.paths.data_dir / "known_names.pkl", 'wb') as f:
            pickle.dump(names, f)
        
        self.embeddings = embeddings
        self.names = names
        print("[OK] Hoàn tất học lại khuôn mặt")
        return len(names)

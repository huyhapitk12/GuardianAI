# Nhận diện khuôn mặt
import os
import pickle
import cv2
import numpy as np
from pathlib import Path

# from insightface.app import FaceAnalysis
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from config import settings


# Class này để nhận diện người quen
class FaceDetector:
    
    def __init__(self):
        self.app = None
        self.embeddings = []
        self.names = []
        self.det_path = None
        self.rec_path = None
        print("face detector init")
    
    # khởi tạo model
    def initialize(self, detector="Small", recognizer="Small"):
        try:
            # os.environ['OMP_NUM_THREADS'] = '4' # old code
            
            providers = ['CPUExecutionProvider']
            try:
                import onnxruntime as ort
                # print("onnx check...")
                available = set(ort.get_available_providers())
                for p in ['CUDAExecutionProvider', 'OpenVINOExecutionProvider', 'DmlExecutionProvider']:
                    if p in available:
                        providers.insert(0, p)
                        break
            except Exception:
                pass
            
            # tìm model
            model_dir = settings.paths.model_dir
            det_path = self.find_model(model_dir / detector, "detect")
            rec_path = self.find_model(model_dir / recognizer, "recog")
            
            if not det_path or not rec_path:
                print("khong thay model face")
                return False
            
            # tạo app
            self.app = self.create_app(det_path, rec_path, providers)
            
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            
            print(f"init face model: {detector}/{recognizer}")
            return True
            
        except Exception as e:
            print(f"loi init face: {e}")
            return False
    
    # hàm tìm file
    def find_model(self, directory, keyword):
        if not directory.exists():
            return None
        
        for f in directory.glob("*.onnx"):
            if keyword in f.name.lower():
                return f
        return None
    
    # tạo app từ class con
    def create_app(self, det_path, rec_path, providers):
        class CustomFaceAnalysis(FaceAnalysis):
            def __init__(self, det, rec, prov):
                self.models = {}
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
    
    # load data mặt đã học
    def load_known_faces(self):
        try:
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
        except Exception:
            return False
    
    # tìm mặt
    def detect_faces(self, image):
        if not self.app:
            return []
        
        return self.app.get(image)
    
    # check xem là ai
    def recognize(self, embedding):
        if not self.embeddings:
            return None, float('inf')
        
        try:
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
            
        except Exception as e:
            print(e)
            return None, float('inf')
    
    # học lại mặt
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
                
                try:
                    img = security.load_image(img_file)
                    if img is None:
                        img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    
                    faces = self.detect_faces(img)
                    if faces:
                        embeddings.append(faces[0].embedding)
                        names.append(person_dir.name)
                        print(f"hoc: {person_dir.name}")
                except Exception as e:
                    print(f"loi: {e}")
        
        try:
            with open(settings.paths.data_dir / "known_embeddings.pkl", 'wb') as f:
                pickle.dump(embeddings, f)
            with open(settings.paths.data_dir / "known_names.pkl", 'wb') as f:
                pickle.dump(names, f)
            
            self.embeddings = embeddings
            self.names = names
            print("hoc xong")
            return len(names)
        except Exception:
            return 0

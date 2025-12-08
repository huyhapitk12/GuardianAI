"""Face detection and recognition"""

from __future__ import annotations
import os
import pickle
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from config import settings


class FaceDetector:
    """InsightFace-based face detector and recognizer"""
    
    __slots__ = ('_app', '_embeddings', '_names', '_det_path', '_rec_path')
    
    def __init__(self):
        self._app = None
        self._embeddings: List[np.ndarray] = []
        self._names: List[str] = []
        self._det_path = None
        self._rec_path = None
    
    def initialize(self, detector: str = "Small", recognizer: str = "Small") -> bool:
        """Initialize face analysis models"""
        try:
            os.environ['OMP_NUM_THREADS'] = '4'
            
            providers = ['CPUExecutionProvider']
            try:
                import onnxruntime as ort
                available = set(ort.get_available_providers())
                for p in ['CUDAExecutionProvider', 'OpenVINOExecutionProvider', 'DmlExecutionProvider']:
                    if p in available:
                        providers.insert(0, p)
                        break
            except ImportError:
                pass
            
            # Find models
            model_dir = settings.paths.model_dir
            det_path = self._find_model(model_dir / detector, "detect")
            rec_path = self._find_model(model_dir / recognizer, "recog")
            
            if not det_path or not rec_path:
                raise FileNotFoundError("Face models not found")
            
            # Create custom face analysis
            self._app = self._create_app(det_path, rec_path, providers)
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            
            print(f"✅ Face detector initialized: {detector}/{recognizer}")
            return True
            
        except Exception as e:
            print(f"❌ Face detector init failed: {e}")
            return False
    
    def _find_model(self, directory: Path, keyword: str) -> Optional[Path]:
        """Find ONNX model by keyword"""
        if not directory.exists():
            return None
        
        for f in directory.glob("*.onnx"):
            if keyword in f.name.lower():
                return f
        return None
    
    def _create_app(self, det_path: Path, rec_path: Path, providers: list):
        """Create custom FaceAnalysis instance"""
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
    
    def load_known_faces(self) -> bool:
        """Load saved embeddings"""
        try:
            emb_file = settings.paths.data_dir / "known_embeddings.pkl"
            names_file = settings.paths.data_dir / "known_names.pkl"
            
            if not emb_file.exists() or not names_file.exists():
                return False
            
            with open(emb_file, 'rb') as f:
                self._embeddings = pickle.load(f)
            with open(names_file, 'rb') as f:
                self._names = pickle.load(f)
            
            print(f"✅ Loaded {len(self._names)} known faces")
            return True
        except Exception:
            return False
    
    def detect_faces(self, image: np.ndarray) -> list:
        """Detect faces in image"""
        if not self._app:
            return []
        return self._app.get(image)
    
    def recognize(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize face by embedding"""
        if not self._embeddings:
            return None, float('inf')
        
        try:
            emb = np.array(embedding, dtype=np.float32)
            known = np.array(self._embeddings, dtype=np.float32)
            
            # Normalize
            emb_norm = emb / np.linalg.norm(emb)
            known_norm = known / np.linalg.norm(known, axis=1, keepdims=True)
            
            # Cosine distance
            distances = 1 - np.dot(known_norm, emb_norm)
            idx = np.argmin(distances)
            dist = float(distances[idx])
            
            threshold = settings.detection.face_recognition_threshold
            # print(f"DEBUG: Distance: {dist:.4f}, Threshold: {threshold}")
            if dist <= threshold:
                return self._names[idx], dist
            return None, dist
            
        except Exception as e:
            print(f"Recognition error: {e}")
            return None, float('inf')
    
    def rebuild_embeddings(self) -> int:
        """Rebuild all embeddings from faces directory"""
        from utils import security
        
        # Use faces_dir for reading images
        faces_dir = settings.paths.faces_dir
        if not faces_dir.exists():
            return 0
        
        embeddings, names = [], []
        
        for person_dir in faces_dir.iterdir():
            if not person_dir.is_dir():
                continue
            
            for img_file in person_dir.glob("*.*"):
                if img_file.suffix.lower() not in ('.jpg', '.png', '.jpeg'):
                    continue
                
                try:
                    # Try encrypted first
                    img = security.load_image(img_file)
                    if img is None:
                        img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    
                    faces = self.detect_faces(img)
                    if faces:
                        embeddings.append(faces[0].embedding)
                        names.append(person_dir.name)
                        print(f"  Encoded: {person_dir.name}/{img_file.name}")
                except Exception as e:
                    print(f"  Error: {img_file}: {e}")
        
        # Save to data_dir
        try:
            with open(settings.paths.data_dir / "known_embeddings.pkl", 'wb') as f:
                pickle.dump(embeddings, f)
            with open(settings.paths.data_dir / "known_names.pkl", 'wb') as f:
                pickle.dump(names, f)
            
            self._embeddings = embeddings
            self._names = names
            print(f"✅ Rebuilt {len(names)} embeddings")
            return len(names)
        except Exception:
            return 0
    
    @property
    def known_names(self) -> List[str]:
        return self._names
    
    @property
    def known_embeddings(self) -> List[np.ndarray]:
        return self._embeddings
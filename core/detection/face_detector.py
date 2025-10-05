"""Face detection and recognition wrapper"""
import pickle
import logging
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from scipy.spatial.distance import cosine

from core.lib.insightface.app import FaceAnalysis
from config.settings import settings
from config.constants import FACE_RECOGNITION_THRESHOLD

logger = logging.getLogger(__name__)

class FaceDetector:
    """Wrapper for InsightFace face detection and recognition"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.models.face_model_name
        self.app = None
        self.known_embeddings: List[np.ndarray] = []
        self.known_names: List[str] = []
        
    def initialize(self) -> bool:
        """Initialize the face detection model"""
        try:
            self.app = FaceAnalysis(
                name=self.model_name,
                root=str(settings.paths.model_dir),
                allowed_modules=['detection', 'recognition']
            )
            self.app.prepare(
                ctx_id=settings.models.insightface_ctx_id,
                det_size=settings.models.insightface_det_size
            )
            logger.info(f"Face detector initialized with model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            return False
    
    def load_known_faces(self) -> bool:
        """Load known face embeddings from cache"""
        embedding_file = settings.paths.embedding_file
        names_file = settings.paths.names_file
        
        if not embedding_file.exists() or not names_file.exists():
            logger.warning("No cached face data found")
            return False
        
        try:
            with open(embedding_file, 'rb') as f:
                self.known_embeddings = pickle.load(f)
            with open(names_file, 'rb') as f:
                self.known_names = pickle.load(f)
            logger.info(f"Loaded {len(self.known_names)} known faces")
            return True
        except Exception as e:
            logger.error(f"Failed to load known faces: {e}")
            return False
    
    def detect_faces(self, image: np.ndarray) -> list:
        """Detect faces in an image"""
        if self.app is None:
            return []
        try:
            return self.app.get(image)
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []
    
    def recognize_face(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Match a face embedding against known faces
        Returns: (name, distance) or (None, inf) if no match
        """
        if not self.known_embeddings:
            return None, float('inf')
        
        try:
            distances = [
                float(cosine(embedding, known_emb)) 
                for known_emb in self.known_embeddings
            ]
            best_idx = np.argmin(distances)
            best_distance = distances[best_idx]
            
            if best_distance <= FACE_RECOGNITION_THRESHOLD:
                return self.known_names[best_idx], best_distance
            return None, best_distance
        except Exception as e:
            logger.error(f"Face recognition error: {e}")
            return None, float('inf')
    
    def update_model(self, model_name: str) -> bool:
        """Switch to a different face detection model"""
        try:
            self.model_name = model_name
            return self.initialize()
        except Exception as e:
            logger.error(f"Failed to update model: {e}")
            return False
    
    def rebuild_embeddings(self) -> int:
        """Rebuild face embeddings from image directory"""
        data_dir = settings.paths.data_dir
        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            return 0
        
        embeddings = []
        names = []
        
        for person_dir in data_dir.iterdir():
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            
            for img_file in person_dir.glob("*.*"):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                try:
                    import cv2
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    
                    faces = self.detect_faces(img)
                    if faces:
                        embedding = faces[0].embedding
                        embeddings.append(embedding)
                        names.append(person_name)
                        logger.debug(f"Encoded: {person_name}/{img_file.name}")
                except Exception as e:
                    logger.error(f"Failed to process {img_file}: {e}")
        
        # Save to cache
        try:
            with open(settings.paths.embedding_file, 'wb') as f:
                pickle.dump(embeddings, f)
            with open(settings.paths.names_file, 'wb') as f:
                pickle.dump(names, f)
            
            self.known_embeddings = embeddings
            self.known_names = names
            
            logger.info(f"Rebuilt {len(names)} face embeddings")
            return len(names)
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            return 0
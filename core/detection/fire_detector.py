"""Fire and smoke detection wrapper"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO

from config.settings import settings
from config.constants import FIRE_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

class FireDetector:
    """Wrapper for YOLO fire/smoke detection"""
    
    def __init__(self):
        self.model: Optional[YOLO] = None
        self.model_size = settings.models.yolo_size
    
    def initialize(self) -> bool:
        """Initialize the fire detection model"""
        model_path = settings.models.get_yolo_fire_path(settings.paths)
        
        try:
            self.model = YOLO(str(model_path), task='detect')
            logger.info(f"Fire detector initialized: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize fire detector: {e}")
            return False
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect fire/smoke in an image
        Returns: List of detections with bbox, class, area, confidence
        """
        if self.model is None:
            return []
        
        try:
            results = self.model(image, verbose=False)
            detections = []
            
            if results and hasattr(results[0], "boxes"):
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    
                    if conf < FIRE_CONFIDENCE_THRESHOLD:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls[0])
                    cls_name = results[0].names.get(cls_id, "unknown").lower()
                    
                    if cls_name in ("fire", "smoke", "flame"):
                        area = (x2 - x1) * (y2 - y1)
                        detections.append({
                            "bbox": (x1, y1, x2, y2),
                            "class": "smoke" if cls_name == "smoke" else "fire",
                            "area": area,
                            "conf": conf
                        })
            
            return detections
        except Exception as e:
            logger.error(f"Fire detection error: {e}")
            return []
    
    def update_model(self, size: str) -> bool:
        """Switch to a different YOLO model size"""
        if size not in ["small", "medium"]:
            logger.error(f"Invalid model size: {size}")
            return False
        
        try:
            self.model_size = size
            settings.models.yolo_size = size
            return self.initialize()
        except Exception as e:
            logger.error(f"Failed to update fire model: {e}")
            return False
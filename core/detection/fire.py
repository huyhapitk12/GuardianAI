# core/detection/fire.py
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Deque
from collections import deque

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from config import settings


class FireConfig:
    """Fire filter configuration"""
    
    def __init__(self):
        self.min_roi_size = 8
        self.flicker_history = 15
        self.flicker_min_frames = 5
        
        # RGB thresholds
        self.rgb_hue_max = 35
        self.rgb_saturation_min = 80
        self.rgb_brightness_min = 100
        self.rgb_white_ratio_max = 0.88
        self.rgb_entropy_min = 4.0
        self.rgb_flicker_min = 5.0
        
        # IR thresholds
        self.ir_brightness_min = 120
        self.ir_brightness_std_min = 25
        self.ir_hot_ratio_min = 0.08
        self.ir_hot_ratio_max = 0.70
        self.ir_irregularity_min = 0.3
        self.ir_flicker_min = 3.0


class FireFilter:
    """Fire detection validation filter"""
    
    def __init__(self, config: Optional[FireConfig] = None, debug: bool = False):
        self.config = config or FireConfig()
        self._history: Dict[str, Deque[float]] = {}
        self._debug = debug
    
    def validate(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                 is_ir: bool = False) -> bool:
        """Validate if detection is real fire"""
        roi = self._get_roi(frame, bbox)
        if roi is None:
            return False
        
        return self._validate_ir(roi, bbox) if is_ir else self._validate_rgb(roi, bbox)
    
    def _get_roi(self, frame: np.ndarray, bbox: Tuple) -> Optional[np.ndarray]:
        """Extract and validate ROI"""
        x1, y1, x2, y2 = map(int, bbox)
        min_size = self.config.min_roi_size
        
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            return None
        
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        roi = frame[y1:y2, x1:x2]
        return roi if roi.size > 0 else None
    
    def _validate_rgb(self, roi: np.ndarray, bbox: Tuple) -> bool:
        """Validate RGB mode detection"""
        cfg = self.config
        
        # Convert to HSV and LAB
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        # 1. Check for pure white reflection (High V, Low S) -> NOT FIRE
        # Reflections often have V > 220 and S < 30
        reflection_mask = (v > 220) & (s < 40)
        reflection_ratio = np.mean(reflection_mask)
        if reflection_ratio > 0.3: # If >30% of ROI is reflection
           return self._fail("reflection")

        # 2. Strict Fire Hue (Red-Orange-Yellow)
        # Hue is 0-179 in OpenCV. Red is ~0-10 & 170-180. Yellow/Orange is ~10-30.
        # We exclude green/blue/purple tints strictly
        fire_hue_mask = ((h >= 0) & (h <= 30)) | ((h >= 165) & (h <= 180))
        valid_hue_ratio = np.mean(fire_hue_mask)
        
        if valid_hue_ratio < 0.4: # At least 40% of pixels must be correct fire color
            return self._fail("hue")
        
        # 3. Saturation Check (Fire is vibrant, not pale)
        # Real fire usuall has saturation > 50 (approx 20% of 255)
        if np.mean(s) < 50: 
            return self._fail("saturation")
        
        # 4. Brightness Check (Fire is bright source)
        if np.max(v) < 120:
            return self._fail("too_dark")
        
        # 5. Texture/Entropy check (Fire is chaotic)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist[hist > 0] / hist.sum()
        entropy = -np.sum(hist * np.log2(hist))
        
        if entropy < 3.5: # Fire is textured due to flickering/movement
             return self._fail("texture")
        
        return True
    
    def _validate_ir(self, roi: np.ndarray, bbox: Tuple) -> bool:
        """Validate IR mode detection"""
        cfg = self.config
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # 1. Brightness
        if np.mean(gray) < cfg.ir_brightness_min and np.max(gray) < 180:
            return self._fail("brightness")
        
        # 2. Variation
        if np.std(gray) < cfg.ir_brightness_std_min:
            return self._fail("variation")
        
        # 3. Hot core ratio
        hot_ratio = np.sum(gray > 200) / gray.size
        if not (cfg.ir_hot_ratio_min <= hot_ratio <= cfg.ir_hot_ratio_max):
            return self._fail("hot_core")
        
        # 4. Shape irregularity
        _, thresh = cv2.threshold(gray.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 10:
                peri = cv2.arcLength(c, True)
                circ = 4 * np.pi * area / (peri ** 2) if peri > 0 else 0
                if (1.0 - circ) < cfg.ir_irregularity_min:
                    return self._fail("shape")
        
        # 5. Flickering
        if not self._check_flicker(gray.astype(np.uint8), bbox, cfg.ir_flicker_min):
            return self._fail("flicker")
        
        return True
    
    def _check_flicker(self, gray: np.ndarray, bbox: Tuple, threshold: float) -> bool:
        """Check for flickering characteristic of fire"""
        key = f"{bbox[0]//20}_{bbox[1]//20}"
        
        if key not in self._history:
            self._history[key] = deque(maxlen=self.config.flicker_history)
        
        hist = self._history[key]
        hist.append(float(np.mean(gray)))
        
        if len(hist) < self.config.flicker_min_frames:
            return True
        
        return np.std(list(hist)) > threshold
    
    def _fail(self, reason: str) -> bool:
        if self._debug:
            print(f"❌ Fire filter FAIL: {reason}")
        return False
    
    def cleanup(self):
        """Remove stale history entries"""
        if len(self._history) > 50:
            keys = list(self._history.keys())[:-30]
            for k in keys:
                del self._history[k]


class FireDetector:
    """YOLO-based fire detector with filtering"""
    
    def __init__(self, debug: bool = False):
        self._model: Optional[YOLO] = None
        self._filter = FireFilter(debug=debug)
        self._frame_count = 0
        self._skip = settings.get('camera.process_every_n_frames', 3)
    
    def initialize(self) -> bool:
        """Initialize fire detector"""
        if not YOLO:
            print("⚠️ ultralytics not installed - fire detection disabled")
            return False
        
        try:
            # Get model config from settings
            yolo_size = settings.get('models.yolo_size', 'medium').lower()
            yolo_format = settings.get('models.yolo_format', 'openvino')
            
            # Use get_yolo_model_path from old settings
            model_path = settings.get_yolo_model_path('fire', yolo_size, yolo_format)
            
            if not model_path.exists():
                print(f"⚠️ Fire model not found: {model_path}")
                return False
            
            # Load model
            print(f"🔥 Loading fire detector from: {model_path}")
            self._model = YOLO(str(model_path), task='detect', verbose=False)
            print(f"✅ Fire detector initialized: {yolo_size}/{yolo_format}")
            
            # Warm-up GPU with dummy inference (for OpenVINO GPU initialization)
            if yolo_format == 'openvino':
                import numpy as np
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                self._model(dummy_frame, verbose=False)
                print("✅ Fire detector GPU warm-up complete")
            
            return True
            
        except Exception as e:
            print(f"❌ Fire detector init failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def detect(self, frame: np.ndarray, skip: bool = True) -> list:
        """Detect fire in frame"""
        if not self._model:
            return []
        
        if skip:
            self._frame_count += 1
            if self._frame_count % self._skip != 0:
                return []
        
        # For OpenVINO, device selection is handled via environment variable
        yolo_format = settings.get('models.yolo_format', 'openvino')
        
        try:
            if yolo_format == 'openvino':
                # OpenVINO: No device parameter - uses env var or auto-selects GPU
                results = self._model(frame, verbose=False)
            else:
                # PyTorch/ONNX: Use device parameter
                device = 'cpu'
                results = self._model(frame, verbose=False, device=device)
            
            detections = []
            
            if results and hasattr(results[0], 'boxes'):
                h, w = frame.shape[:2]
                total_area = w * h
                
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    cls = results[0].names.get(int(box.cls[0]), '').lower()
                    
                    if cls not in ('fire', 'flame', 'smoke'):
                        continue
                    
                    threshold = settings.get('detection.smoke_confidence', 0.7) if cls == 'smoke' else settings.get('detection.fire_confidence', 0.6)
                    if conf < threshold:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    area = (x2 - x1) * (y2 - y1) / total_area
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'class': cls,
                        'conf': conf,
                        'area': area
                    })
            
            return detections
        except Exception as e:
            print(f"⚠️ Fire detection error: {e}")
            return []
    
    def validate(self, frame: np.ndarray, bbox: Tuple, is_ir: bool = False) -> bool:
        """Validate detection with filter"""
        return self._filter.validate(frame, bbox, is_ir)
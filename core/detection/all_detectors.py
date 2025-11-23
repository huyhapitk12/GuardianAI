import os
import time
import cv2
import pickle
import datetime as dt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import deque
from scipy.spatial.distance import cosine

# Third-party imports
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("WARNING: ultralytics not found")

try:
    import supervision as sv
except ImportError:
    sv = None

# Local imports
from config import settings
from insightface.app import FaceAnalysis
from trackers import SORTTracker

# ============================================================================
# LOGGING HELPERS
# ============================================================================

def _log(level: str, module: str, message: str):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} - {module} - {level} - {message}")

def print_info(module, msg): _log("INFO", module, msg)
def print_warning(module, msg): _log("WARNING", module, msg)
def print_error(module, msg): _log("ERROR", module, msg)

# ============================================================================
# FACE DETECTION
# ============================================================================

class FaceDetector:
    def __init__(self, detector_name: Optional[str] = None, recognizer_name: Optional[str] = None):
        self.detector_name = detector_name or settings.models.face.detector_name
        self.recognizer_name = recognizer_name or settings.models.face.recognizer_name
        self.app = None
        self.known_embeddings: List[np.ndarray] = []
        self.known_names: List[str] = []
        
    def initialize(self) -> bool:
        try:
            os.environ['OMP_NUM_THREADS'] = '4'
            os.environ['MKL_NUM_THREADS'] = '4'
            
            providers = ['CPUExecutionProvider']
            try:
                import onnxruntime as ort
                avail = set(ort.get_available_providers())
                prio = ['CUDAExecutionProvider', 'OpenVINOExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
                providers = [p for p in prio if p in avail]
            except: pass

            model_root = str(settings.paths.model_dir)
            det_path = Path(model_root) / self.detector_name / f"det_{'10g' if self.detector_name == 'buffalo_l' else '500m'}.onnx"
            rec_path = Path(model_root) / self.recognizer_name / f"w600k_{'r50' if self.recognizer_name == 'buffalo_l' else 'mbf'}.onnx"
            
            if not det_path.exists() or not rec_path.exists():
                raise FileNotFoundError("Face models not found")

            if FaceAnalysis:
                self.app = FaceAnalysis(det_model=str(det_path), rec_model=str(rec_path), root=model_root, allowed_modules=['detection', 'recognition'], providers=providers)
                self.app.prepare(ctx_id=settings.models.insightface_ctx_id, det_size=settings.models.insightface_det_size)
                print_info("face", f"Initialized {self.detector_name}/{self.recognizer_name}")
                return True
            return False
        except Exception as e:
            print_error("face", f"Init failed: {e}")
            return False
    
    def load_known_faces(self) -> bool:
        try:
            if not settings.paths.embedding_file.exists(): return False
            with open(settings.paths.embedding_file, 'rb') as f: self.known_embeddings = pickle.load(f)
            with open(settings.paths.names_file, 'rb') as f: self.known_names = pickle.load(f)
            print_info("face", f"Loaded {len(self.known_names)} faces")
            return True
        except: return False
    
    def detect_faces(self, image: np.ndarray) -> list:
        return self.app.get(image) if self.app else []
    
    def recognize_face(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        if not self.known_embeddings: return None, float('inf')
        try:
            emb = np.array(embedding, dtype=np.float32)
            known = np.array(self.known_embeddings, dtype=np.float32)
            emb_norm = emb / np.linalg.norm(emb)
            known_norm = known / np.linalg.norm(known, axis=1, keepdims=True)
            dists = 1 - np.dot(known_norm, emb_norm)
            idx = np.argmin(dists)
            dist = float(dists[idx])
            return (self.known_names[idx], dist) if dist <= settings.detection.face_recognition_threshold else (None, dist)
        except: return None, float('inf')

    def rebuild_embeddings(self) -> int:
        data_dir = settings.paths.data_dir
        if not data_dir.exists(): return 0
        
        embeddings, names = [], []
        print_info("face", "Rebuilding embeddings...")
        
        for p_dir in data_dir.iterdir():
            if not p_dir.is_dir(): continue
            for img_file in p_dir.glob("*.*"):
                if img_file.suffix.lower() not in ['.jpg', '.png', '.jpeg']: continue
                try:
                    from utils.security import security_manager
                    img = security_manager.decrypt_image(img_file)
                    if img is None: img = cv2.imread(str(img_file))
                    if img is None: continue
                    
                    faces = self.detect_faces(img)
                    if faces:
                        embeddings.append(faces[0].embedding)
                        names.append(p_dir.name)
                        print_info("face", f"Encoded {p_dir.name}/{img_file.name}")
                except Exception as e: print_error("face", f"Error {img_file}: {e}")
        
        try:
            with open(settings.paths.embedding_file, 'wb') as f: pickle.dump(embeddings, f)
            with open(settings.paths.names_file, 'wb') as f: pickle.dump(names, f)
            self.known_embeddings, self.known_names = embeddings, names
            return len(names)
        except: return 0

class OptimizedFaceDetector(FaceDetector):
    def __init__(self, detector_name=None, recognizer_name=None):
        super().__init__(detector_name, recognizer_name)
        self.detection_count = 0
        self.recognition_count = 0

    def detect_and_recognize(self, frame: np.ndarray, use_cache: bool = True) -> List[Dict]:
        if not self.app or frame is None: return []
        try:
            self.detection_count += 1
            faces = self.app.get(frame)
            results = []
            for face in faces:
                bbox = face.bbox.astype(int)
                name, conf = 'Unknown', 0.0
                if self.known_embeddings and use_cache:
                    n, dist = self.recognize_face(face.embedding)
                    if n: name, conf = n, 1.0 - min(dist, 1.0)
                    self.recognition_count += 1
                results.append({'bbox': bbox, 'embedding': face.embedding, 'name': name, 'confidence': conf, 'face': face})
            return results
        except Exception as e:
            print_error("face", f"Detect error: {e}")
            return []

# ============================================================================
# FIRE DETECTION
# ============================================================================

class FireDetector:
    def __init__(self):
        self.model: Optional[YOLO] = None
        self.model_size = settings.models.yolo_size
        self.model_format = settings.models.yolo_format
    
    def initialize(self) -> bool:
        try:
            os.environ['OMP_NUM_THREADS'] = '4'
            path = settings.get_yolo_model_path('fire', self.model_size, self.model_format)
            if self.model_format == 'openvino' and not path.exists(): return False
            
            if YOLO:
                self.model = YOLO(str(path), task='detect')
                # Only call .to() for PyTorch models, not for exported formats
                if self.model_format not in ['openvino', 'onnx', 'tensorrt', 'tflite'] and hasattr(self.model, 'to'):
                    self.model.to('cpu')
                print_info("fire", f"Initialized {self.model_format}")
                return True
            return False
        except Exception as e:
            print_error("fire", f"Init failed: {e}")
            return False

    def detect(self, image: np.ndarray) -> List[Dict]:
        if not self.model: return []
        try:
            results = self.model(image, verbose=False, device='cpu')
            detections = []
            h, w = image.shape[:2]
            area_total = w * h
            
            if results and hasattr(results[0], "boxes"):
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    name = results[0].names.get(cls_id, "unknown").lower()
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    if name in ["smoke", "fire", "flame"]:
                        thresh = settings.detection.smoke_confidence_threshold if name == "smoke" else settings.detection.fire_confidence_threshold
                        if conf >= thresh:
                            area = ((x2-x1)*(y2-y1)) / area_total if area_total > 0 else 0
                            detections.append({"bbox": (x1, y1, x2, y2), "class": name, "area": area, "conf": conf})
            return detections
        except: return []

class OptimizedFireDetector(FireDetector):
    def __init__(self):
        super().__init__()
        self.frame_skip = settings.camera.process_every_n_frames
        self.frame_count = 0
        self.stats = deque(maxlen=100)

    def detect(self, image: np.ndarray, skip_frames: bool = True) -> List[Dict]:
        if skip_frames:
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0: return []
        
        dets = super().detect(image)
        self.stats.append(len(dets))
        return dets

# ============================================================================
# PERSON TRACKER
# ============================================================================

class PersonTracker:
    def __init__(self, face_detector=None):
        self.person_model: Optional[YOLO] = None
        self.sort_tracker = None
        self.face_detector = face_detector
        self.model_size = settings.models.yolo_size
        self.tracked_objects = {}
        self.next_object_id = 0
        self.reid_memory = {}
        self.next_reid_id = 1
        self.alerted_reid_ids = set()

    def initialize(self) -> bool:
        try:
            os.environ['OMP_NUM_THREADS'] = '4'
            path = settings.get_yolo_model_path('person', self.model_size, settings.models.yolo_format)
            if YOLO:
                self.person_model = YOLO(str(path))
                # Only call .to() for PyTorch models, not for exported formats
                if settings.models.yolo_format not in ['openvino', 'onnx', 'tensorrt', 'tflite'] and hasattr(self.person_model, 'to'):
                    self.person_model.to('cpu')
                print_info("tracker", "Person detector initialized")
            
            if SORTTracker:
                self.sort_tracker = SORTTracker()
                print_info("tracker", "SORT initialized")
            return True
        except Exception as e:
            print_error("tracker", f"Init failed: {e}")
            return False

    def detect_persons(self, image: np.ndarray, conf_threshold: float = None) -> List[Tuple]:
        if not self.person_model: return []
        thresh = conf_threshold or settings.detection.person_confidence_threshold
        try:
            res = self.person_model(image, conf=thresh, classes=0, verbose=False, device='cpu')[0]
            return [tuple(map(float, b.xyxy[0].tolist())) for b in res.boxes] if hasattr(res, "boxes") else []
        except: return []

    def update_tracks(self, detections: List[Tuple], frame: np.ndarray, scale_x=1.0, scale_y=1.0) -> Dict:
        now = time.time()
        scaled = [(int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)) for x1,y1,x2,y2 in detections]
        
        if self.sort_tracker and scaled and sv:
            self._update_sort(scaled, frame, now)
        else:
            self._update_iou(scaled, now)
            
        self._prune_stale(now)
        return self.tracked_objects

    def _update_sort(self, detections, frame, now):
        try:
            xyxy = np.array(detections, dtype=float)
            sv_dets = sv.Detections(xyxy=xyxy, confidence=np.ones(len(detections)))
            tracked = self.sort_tracker.update(sv_dets)
            
            # Parse output
            t_xyxy, t_ids = None, None
            if hasattr(tracked, "xyxy"):
                t_xyxy = tracked.xyxy
                t_ids = tracked.tracker_id if hasattr(tracked, "tracker_id") else (tracked.track_id if hasattr(tracked, "track_id") else None)
            elif isinstance(tracked, np.ndarray) and tracked.shape[1] >= 5:
                t_xyxy = tracked[:, :4]
                t_ids = tracked[:, 4].astype(int)

            if t_xyxy is not None and t_ids is not None:
                for i, tid in enumerate(t_ids):
                    bbox = tuple(map(int, t_xyxy[i]))
                    if tid not in self.tracked_objects:
                        self.tracked_objects[tid] = self._create_track(bbox, now)
                    else:
                        self._update_track(tid, bbox, frame, now)
            else:
                self._update_iou(detections, now)
        except: self._update_iou(detections, now)

    def _update_iou(self, detections, now):
        # Simple IOU matching logic (simplified for brevity)
        if not self.tracked_objects:
            for bbox in detections:
                self.tracked_objects[self.next_object_id] = self._create_track(bbox, now)
                self.next_object_id += 1
            return

        # Match existing
        track_ids = list(self.tracked_objects.keys())
        track_boxes = [self.tracked_objects[i]['bbox'] for i in track_ids]
        
        used_dets = set()
        for i, t_box in enumerate(track_boxes):
            best_iou = 0
            best_det_idx = -1
            for j, d_box in enumerate(detections):
                if j in used_dets: continue
                iou = self._iou(t_box, d_box)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = j
            
            if best_iou > settings.detection.iou_threshold:
                self._update_track(track_ids[i], detections[best_det_idx], None, now)
                used_dets.add(best_det_idx)
        
        # New tracks
        for j, bbox in enumerate(detections):
            if j not in used_dets:
                self.tracked_objects[self.next_object_id] = self._create_track(bbox, now)
                self.next_object_id += 1

    def _create_track(self, bbox, now):
        return {
            'bbox': bbox, 'name': "Nguoi la", 'distance': float('inf'),
            'last_seen_by_detector': now, 'last_updated': now,
            'face_hits': 0, 'last_face_rec': 0.0, 'confirmed_name': None,
            'alert_sent': False, 'frames_unidentified': 0, 'stranger_alert_sent': False,
            'reid_id': None, 'reid_embedding': None
        }

    def _update_track(self, tid, bbox, frame, now):
        data = self.tracked_objects[tid]
        data['bbox'] = bbox
        data['last_seen_by_detector'] = now
        data['last_updated'] = now
        if frame is not None and self.face_detector:
            self._try_face_rec(tid, bbox, frame, now)

    def _try_face_rec(self, tid, bbox, frame, now):
        data = self.tracked_objects[tid]
        if (now - data['last_face_rec'] < settings.tracker.face_recognition_cooldown) or data['confirmed_name']: return
        
        data['last_face_rec'] = now
        x1, y1, x2, y2 = bbox
        crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
        if crop.size == 0: return

        try:
            faces = self.face_detector.detect_faces(crop)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                
                # ReID
                try:
                    emb = face.embedding
                    rid = self._assign_reid(emb, now)
                    data['reid_id'] = rid
                    data['reid_embedding'] = emb if data['reid_embedding'] is None else 0.7*data['reid_embedding'] + 0.3*emb
                except: pass

                name, dist = self.face_detector.recognize_face(face.embedding)
                if name:
                    data['face_hits'] += 1
                    data['name'] = name
                    data['distance'] = dist
                else:
                    data['face_hits'] = max(0, data['face_hits'] - 1)
                    if data['face_hits'] == 0: data['name'] = "Nguoi la"
            else:
                data['face_hits'] = max(0, data['face_hits'] - 1)
        except: pass

    def _assign_reid(self, emb, now):
        # Cleanup
        for r in [k for k,v in self.reid_memory.items() if now - v['last_seen'] > settings.reid.ttl_seconds]:
            del self.reid_memory[r]
            self.alerted_reid_ids.discard(r)
            
        if not self.reid_memory:
            rid = self.next_reid_id
            self.next_reid_id += 1
            self.reid_memory[rid] = {'embedding': emb, 'last_seen': now}
            return rid

        # Match
        best_rid, best_dist = None, float('inf')
        for rid, info in self.reid_memory.items():
            dist = cosine(emb, info['embedding'])
            if dist < best_dist: best_dist, best_rid = dist, rid
            
        if best_dist <= settings.reid.distance_threshold:
            self.reid_memory[best_rid]['embedding'] = 0.8*self.reid_memory[best_rid]['embedding'] + 0.2*emb
            self.reid_memory[best_rid]['last_seen'] = now
            return best_rid
            
        rid = self.next_reid_id
        self.next_reid_id += 1
        self.reid_memory[rid] = {'embedding': emb, 'last_seen': now}
        return rid

    def check_confirmations(self) -> List[Tuple]:
        alerts = []
        for tid, data in self.tracked_objects.items():
            # Known
            if data['face_hits'] >= settings.tracker.known_person_confirm_frames and not data['confirmed_name'] and data['name'] != "Nguoi la" and not data['alert_sent']:
                data['confirmed_name'] = data['name']
                data['alert_sent'] = True
                alerts.append((tid, 'nguoi_quen', {'name': data['name'], 'distance': data['distance']}))
            # Stranger
            if not data['confirmed_name']:
                data['frames_unidentified'] += 1
                if data['frames_unidentified'] > settings.tracker.stranger_confirm_frames and not data['stranger_alert_sent']:
                    rid = data.get('reid_id')
                    if rid is None or rid not in self.alerted_reid_ids:
                        data['stranger_alert_sent'] = True
                        if rid: self.alerted_reid_ids.add(rid)
                        alerts.append((tid, 'nguoi_la', {}))
        return alerts

    def _prune_stale(self, now):
        stale = [t for t, d in self.tracked_objects.items() if now - d['last_seen_by_detector'] > settings.tracker.timeout_seconds]
        for t in stale: del self.tracked_objects[t]

    @staticmethod
    def _iou(boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0

    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        """Draw bounding boxes and labels for tracked persons"""
        for tid, data in self.tracked_objects.items():
            x1, y1, x2, y2 = map(int, data['bbox'])
            name = data.get('confirmed_name') or data.get('name', 'Unknown')
            
            # Choose color based on identity status
            color = (0, 255, 0) if data.get('confirmed_name') else (0, 165, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"ID:{tid} {name}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        return frame

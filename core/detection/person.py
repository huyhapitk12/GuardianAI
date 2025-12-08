"""Person detection and tracking"""

from __future__ import annotations
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.spatial.distance import cosine

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import supervision as sv
    from trackers import SORTTracker
except ImportError:
    sv = None
    SORTTracker = None

from config import settings, AlertType


class Track:
    """Person track data"""
    
    def __init__(self, bbox):
        self.bbox = bbox
        self.name = "Stranger"
        self.distance = float('inf')
        self.last_seen = 0
        self.face_hits = 0
        self.last_face_check = 0
        self.confirmed_name = None
        self.alert_sent = False
        self.stranger_alert_sent = False
        self.frames_unidentified = 0
        self.reid_id = None
        self.reid_embedding = None


class PersonTracker:
    """Person detection and tracking with face recognition"""
    
    def __init__(self, face_detector=None, shared_model=None):
        self._model: Optional[YOLO] = shared_model
        self._owns_model = shared_model is None  # Chỉ tự quản lý model nếu không có shared
        self._sort = None
        self._face_detector = face_detector
        self._tracks: Dict[int, Track] = {}
        self._next_id = 0
        self._reid_memory: Dict[int, dict] = {}
        self._next_reid = 1
        self._alerted_reids: set = set()

    def initialize(self) -> bool:
        """Initialize person tracker"""
        # Nếu đã có model (shared), chỉ cần init SORT
        if self._model is not None:
            if SORTTracker:
                self._sort = SORTTracker()
            print("✅ Person Tracker initialized (shared model)")
            return True
        
        # Fallback: tự load model nếu không có shared
        if not YOLO:
            return False
        
        try:
            yolo_size = settings.get('models.yolo_size', 'medium').lower()
            yolo_format = settings.get('models.yolo_format', 'openvino')
            path = settings.get_yolo_model_path('person', yolo_size, yolo_format)
            
            self._model = YOLO(str(path), verbose=False)
            
            if SORTTracker:
                self._sort = SORTTracker()
            
            print("✅ Person Tracker initialized")
            return True
        except Exception as e:
            print(f"❌ Person tracker init failed: {e}")
            return False
    
    def get_face_detector(self):
        return self._face_detector
    
    def set_face_detector(self, detector):
        self._face_detector = detector
    
    def detect(self, frame: np.ndarray, conf: Optional[float] = None) -> List[Tuple]:
        """Detect persons in frame"""
        if not self._model:
            return []
        
        # Use get() with default for old settings compatibility
        threshold = conf or settings.get('detection.person_confidence', 0.5)
        
        # For OpenVINO, device selection is handled via environment variable
        # Don't pass device parameter - OpenVINO will auto-select GPU if available
        yolo_format = settings.get('models.yolo_format', 'openvino')
        
        try:
            if yolo_format == 'openvino':
                # OpenVINO: No device parameter needed - uses env var or auto-selects
                results = self._model(frame, conf=threshold, classes=0, verbose=False)[0]
            else:
                # PyTorch/ONNX: Use device parameter
                device = 'cpu'  # Can be 'cuda' for NVIDIA GPU
                results = self._model(frame, conf=threshold, classes=0, verbose=False, device=device)[0]
            
            return [tuple(map(float, b.xyxy[0].tolist())) for b in results.boxes] if hasattr(results, 'boxes') else []
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            return []
    
    def update(self, detections: List[Tuple], frame: np.ndarray, 
               scale_x: float = 1.0, scale_y: float = 1.0, skip_face_check: bool = False) -> Dict[int, Track]:
        """Update tracks with new detections"""
        now = time.time()
        
        # Scale detections
        scaled = [(int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)) 
                  for x1, y1, x2, y2 in detections]
        
        if self._sort and scaled and sv:
            self._update_sort(scaled, frame, now)
        else:
            self._update_iou(scaled, now)
        
        # Face recognition (skip in IR mode)
        if self._face_detector and not skip_face_check:
            for tid, track in self._tracks.items():
                self._check_face(tid, track, frame, now)
        
        # Remove stale tracks
        timeout = settings.get('tracker.timeout_seconds', 30)
        self._tracks = {k: v for k, v in self._tracks.items() 
                        if now - v.last_seen < timeout}
        
        return self._tracks
    
    def _update_sort(self, detections: List[Tuple], frame: np.ndarray, now: float):
        """Update using SORT tracker"""
        try:
            xyxy = np.array(detections, dtype=float)
            sv_dets = sv.Detections(xyxy=xyxy, confidence=np.ones(len(detections)))
            tracked = self._sort.update(sv_dets)
            
            if hasattr(tracked, 'xyxy') and hasattr(tracked, 'tracker_id'):
                for i, tid in enumerate(tracked.tracker_id):
                    bbox = tuple(map(int, tracked.xyxy[i]))
                    self._update_track(tid, bbox, now)
        except Exception:
            self._update_iou(detections, now)
    
    def _update_iou(self, detections: List[Tuple], now: float):
        """Update using IOU matching"""
        if not self._tracks:
            for bbox in detections:
                track = Track(bbox)
                track.last_seen = now
                self._tracks[self._next_id] = track
                self._next_id += 1
            return
        
        # Match existing tracks
        track_ids = list(self._tracks.keys())
        matched_dets = set()
        
        for tid in track_ids:
            track = self._tracks[tid]
            best_iou, best_idx = 0, -1
            
            for i, det in enumerate(detections):
                if i in matched_dets:
                    continue
                iou = self._calc_iou(track.bbox, det)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            
            iou_thresh = settings.get('detection.iou_threshold', 0.3)
            if best_iou > iou_thresh:
                track.bbox = detections[best_idx]
                track.last_seen = now
                matched_dets.add(best_idx)
        
        # Create new tracks
        for i, bbox in enumerate(detections):
            if i not in matched_dets:
                track = Track(bbox)
                track.last_seen = now
                self._tracks[self._next_id] = track
                self._next_id += 1
    
    def _update_track(self, tid: int, bbox: Tuple, now: float):
        """Update single track"""
        if tid not in self._tracks:
            track = Track(bbox)
            track.last_seen = now
            self._tracks[tid] = track
        else:
            self._tracks[tid].bbox = bbox
            self._tracks[tid].last_seen = now
    
    def _check_face(self, tid: int, track: Track, frame: np.ndarray, now: float):
        """Check face recognition for track"""
        # OPTIMIZATION: If already confirmed, NEVER check again until track is lost
        if track.confirmed_name:
            return

        cooldown = settings.get('tracker.face_recognition_cooldown', 1.0)
        
        if now - track.last_face_check < cooldown:
            return
        
        track.last_face_check = now
        
        x1, y1, x2, y2 = track.bbox
        h, w = frame.shape[:2]
        
        # Add padding to face crop for better recognition
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y) 
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)
        
        crop = frame[int(crop_y1):int(crop_y2), int(crop_x1):int(crop_x2)]
        
        if crop.size == 0 or (crop.shape[0] < 20 or crop.shape[1] < 20):
            return
        
        # print(f"DEBUG: Checking face for track {tid}")
        faces = self._face_detector.detect_faces(crop)
        if not faces:
            # print(f"DEBUG: No faces found in crop for track {tid}")
            track.face_hits = max(0, track.face_hits - 1)
            return
        
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        # print(f"DEBUG: Face detected for track {tid}, embedding len: {len(face.embedding)}")
        
        # ReID
        self._update_reid(track, face.embedding, now)
        
        # Recognition
        name, dist = self._face_detector.recognize(face.embedding)
        # print(f"DEBUG: Recognize result for track {tid}: name={name}, dist={dist}")
        if name:
            track.face_hits += 1
            track.name = name
            track.distance = dist
        else:
            track.face_hits = max(0, track.face_hits - 1)
            if track.face_hits == 0:
                track.name = "Stranger"
    
    def _update_reid(self, track: Track, embedding: np.ndarray, now: float):
        """Update ReID for track"""
        ttl = settings.get('reid.ttl_seconds', 30)
        threshold = settings.get('reid.distance_threshold', 0.35)
        
        # Cleanup old entries
        self._reid_memory = {k: v for k, v in self._reid_memory.items() 
                            if now - v['last_seen'] < ttl}
        
        # Find match
        best_rid, best_dist = None, float('inf')
        for rid, info in self._reid_memory.items():
            dist = cosine(embedding, info['embedding'])
            if dist < best_dist:
                best_dist, best_rid = dist, rid
        
        if best_dist <= threshold:
            # Update existing
            self._reid_memory[best_rid]['embedding'] = 0.8 * self._reid_memory[best_rid]['embedding'] + 0.2 * embedding
            self._reid_memory[best_rid]['last_seen'] = now
            track.reid_id = best_rid
        else:
            # Create new
            rid = self._next_reid
            self._next_reid += 1
            self._reid_memory[rid] = {'embedding': embedding, 'last_seen': now}
            track.reid_id = rid
    
    def check_alerts(self) -> List[Tuple[int, str, dict]]:
        alerts = []
        
        known_confirm = settings.get('tracker.known_person_confirm_frames', 3)
        stranger_confirm = settings.get('tracker.stranger_confirm_frames', 30)
        
        for tid, track in self._tracks.items():
            # Known person alert
            if (track.face_hits >= known_confirm and 
                not track.confirmed_name and 
                track.name != "Stranger" and 
                not track.alert_sent):
                
                track.confirmed_name = track.name
                track.alert_sent = True
                alerts.append((tid, AlertType.KNOWN_PERSON, 
                              {'name': track.name, 'distance': track.distance}))
            
            # Stranger alert
            if not track.confirmed_name:
                track.frames_unidentified += 1
                
                if (track.frames_unidentified > stranger_confirm and 
                    not track.stranger_alert_sent):
                    
                    rid = track.reid_id
                    if rid is None or rid not in self._alerted_reids:
                        track.stranger_alert_sent = True
                        if rid:
                            self._alerted_reids.add(rid)
                        alerts.append((tid, AlertType.STRANGER, {}))
        
        return alerts
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracks on frame"""
        for tid, track in self._tracks.items():
            x1, y1, x2, y2 = map(int, track.bbox)
            
            # Color based on recognition
            if track.confirmed_name and track.name != "Stranger":
                color = (0, 255, 0)  # Green for known
                label = track.name
            elif track.stranger_alert_sent:
                color = (0, 0, 255)  # Red for stranger
                label = "STRANGER"
            else:
                color = (255, 255, 0)  # Yellow for tracking
                label = f"ID-{tid}"
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def _calc_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IOU between boxes"""
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        
        xi1 = max(x1, x1_)
        yi1 = max(y1, y1_)
        xi2 = min(x2, x2_)
        yi2 = min(y2, y2_)
        
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)
        union = box1_area + box2_area - inter
        
        return inter / union if union > 0 else 0
    
    def has_active_threats(self) -> bool:
        """Check if there are any active threats (Strangers or Anomalies)"""
        now = time.time()
        timeout = settings.get('tracker.timeout_seconds', 30)
        
        for track in self._tracks.values():
            if now - track.last_seen > timeout:
                continue
                
            # Threat 1: Stranger (confirmed stranger logic)
            # track.stranger_alert_sent mean we already decided it's a stranger
            if track.stranger_alert_sent:
                return True
                
            # Threat 2: Anomaly (requires external setting/logic but we can check name/flags if set)
            # Currently behavior is separate but visualized on track.
            # We will rely on Camera class to check behavior analyzer, 
            # OR if we want to add behavior status to Track dataclass more explicitly.
            # For now, let's just return True for strangers.
            # Behavior anomalies are usually transient, handled by main app or Camera behavior status.
            
        return False
        
    def has_tracks(self) -> bool:
        """Check if there are any active tracks"""
        return bool(self._tracks)
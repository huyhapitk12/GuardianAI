"""Person detection and tracking"""
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv

from core.lib.trackers import SORTTracker
from config.settings import settings
from config.constants import (
    PERSON_CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    TRACKER_TIMEOUT_SECONDS,
    FACE_RECOGNITION_COOLDOWN,
    FRAMES_REQUIRED_FOR_CONFIRMATION,
    STRANGER_CONFIRM_FRAMES
)

logger = logging.getLogger(__name__)

class PersonTracker:
    """Handles person detection and tracking"""
    
    def __init__(self, face_detector=None):
        self.person_model: Optional[YOLO] = None
        self.sort_tracker: Optional[SORTTracker] = None
        self.face_detector = face_detector
        
        # Tracking state
        self.tracked_objects: Dict[int, Dict[str, Any]] = {}
        self.next_object_id = 0
        
    def initialize(self) -> bool:
        """Initialize person detection model and tracker"""
        # Load YOLO person detection model
        model_path = settings.models.get_yolo_person_path(settings.paths)
        try:
            self.person_model = YOLO(str(model_path))
            logger.info(f"Person detector initialized: {model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize person detector: {e}")
            return False
        
        # Initialize SORT tracker
        try:
            self.sort_tracker = SORTTracker()
            logger.info("SORT tracker initialized")
        except Exception as e:
            logger.warning(f"SORT tracker unavailable: {e}")
            self.sort_tracker = None
        
        return True
    
    def detect_persons(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect persons in an image, returns list of bounding boxes"""
        if self.person_model is None:
            return []
        
        try:
            results = self.person_model(
                image,
                conf=PERSON_CONFIDENCE_THRESHOLD,
                classes=0,  # person class
                verbose=False
            )[0]
            
            boxes = []
            if hasattr(results, "boxes"):
                for box in results.boxes:
                    try:
                        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                        boxes.append((x1, y1, x2, y2))
                    except Exception:
                        continue
            
            return boxes
        except Exception as e:
            logger.error(f"Person detection error: {e}")
            return []
    
    def update_tracks(
        self,
        detections: List[Tuple[int, int, int, int]],
        frame: np.ndarray,
        scale_x: float = 1.0,
        scale_y: float = 1.0
    ) -> Dict[int, Dict[str, Any]]:
        """
        Update tracking with new detections
        Returns: Dictionary of tracked objects with IDs
        """
        now = time.time()
        
        # Scale detections to original frame size
        scaled_detections = [
            (int(x1 * scale_x), int(y1 * scale_y), 
             int(x2 * scale_x), int(y2 * scale_y))
            for x1, y1, x2, y2 in detections
        ]
        
        # Use SORT tracker if available
        if self.sort_tracker and scaled_detections:
            tracked_ids = self._update_with_sort(scaled_detections, frame, now)
        else:
            tracked_ids = self._update_with_iou(scaled_detections, now)
        
        # Remove stale tracks
        self._prune_stale_tracks(now)
        
        return self.tracked_objects
    
    def _update_with_sort(
        self,
        detections: List[Tuple],
        frame: np.ndarray,
        now: float
    ) -> Dict[int, Dict]:
        """Update tracking using SORT algorithm"""
        # Prepare detections for SORT as supervision.Detections
        xyxy = np.array(detections, dtype=float)
        scores = np.ones((len(detections),), dtype=float)
        sv_dets = sv.Detections(xyxy=xyxy, confidence=scores)
        
        try:
            # Update SORT tracker
            tracked = self.sort_tracker.update(sv_dets)
            tracked_xyxy, tracked_ids = self._parse_tracked_output(tracked)
            
            if tracked_xyxy is None or tracked_ids is None:
                return self._update_with_iou(detections, now)
            
            # Update tracked objects
            for i, track_id in enumerate(tracked_ids):
                bbox = tuple(map(int, tracked_xyxy[i]))
                
                if track_id is None:
                    track_id = self.next_object_id
                    self.next_object_id += 1
                
                if track_id not in self.tracked_objects:
                    self.tracked_objects[track_id] = self._create_track_data(bbox, now)
                else:
                    self._update_track_data(track_id, bbox, frame, now)
            
            return self.tracked_objects
        except Exception as e:
            logger.error(f"SORT tracking error: {e}")
            return self._update_with_iou(detections, now)
    
    def _update_with_iou(
        self,
        detections: List[Tuple],
        now: float
    ) -> Dict[int, Dict]:
        """Fallback: Update tracking using simple IOU matching"""
        for bbox in detections:
            best_iou = 0
            best_id = None
            
            # Find best matching existing track
            for track_id, track_data in self.tracked_objects.items():
                iou = self._calculate_iou(bbox, track_data['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_id = track_id
            
            # Update existing or create new track
            if best_id is not None and best_iou >= IOU_THRESHOLD:
                self._update_track_data(best_id, bbox, None, now)
            else:
                new_id = self.next_object_id
                self.next_object_id += 1
                self.tracked_objects[new_id] = self._create_track_data(bbox, now)
        
        return self.tracked_objects
    
    def _create_track_data(self, bbox: Tuple, now: float) -> Dict:
        """Create initial track data"""
        return {
            'bbox': bbox,
            'name': "Nguoi la",
            'distance': float('inf'),
            'last_seen_by_detector': now,
            'last_updated': now,
            'face_hits': 0,
            'last_face_rec': 0.0,
            'confirmed_name': None,
            'alert_sent': False,
            'frames_unidentified': 0,
            'stranger_alert_sent': False
        }
    
    def _update_track_data(
        self,
        track_id: int,
        bbox: Tuple,
        frame: Optional[np.ndarray],
        now: float
    ) -> None:
        """Update existing track data"""
        data = self.tracked_objects[track_id]
        data['bbox'] = bbox
        data['last_seen_by_detector'] = now
        data['last_updated'] = now
        
        # Perform face recognition if needed and frame provided
        if frame is not None and self.face_detector:
            self._try_face_recognition(track_id, bbox, frame, now)
    
    def _try_face_recognition(
        self,
        track_id: int,
        bbox: Tuple,
        frame: np.ndarray,
        now: float
    ) -> None:
        """Attempt face recognition on tracked person"""
        data = self.tracked_objects[track_id]
        
        # Check if we should do face recognition
        cooldown_passed = (now - data['last_face_rec']) >= FACE_RECOGNITION_COOLDOWN
        not_confirmed = data['confirmed_name'] is None
        
        if not (cooldown_passed and not_confirmed):
            return
        
        data['last_face_rec'] = now
        
        # Extract person crop
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return
        
        person_crop = frame[y1:y2, x1:x2]
        
        try:
            faces = self.face_detector.detect_faces(person_crop)
            
            if faces:
                # Get largest face
                best_face = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                )
                
                name, dist = self.face_detector.recognize_face(best_face.embedding)
                
                if name is not None:
                    data['face_hits'] += 1
                    data['name'] = name
                    data['distance'] = dist
                else:
                    data['face_hits'] = max(0, data['face_hits'] - 1)
                    if data['face_hits'] == 0:
                        data['name'] = "Nguoi la"
                        data['distance'] = float('inf')
            else:
                data['face_hits'] = max(0, data['face_hits'] - 1)
        except Exception as e:
            logger.error(f"Face recognition error for track {track_id}: {e}")
            data['face_hits'] = max(0, data['face_hits'] - 1)
    
    def check_confirmations(self) -> List[Tuple[int, str, Dict]]:
        """
        Check for confirmed identities (known person or stranger)
        Returns: List of (track_id, alert_type, metadata)
        """
        alerts = []
        
        for track_id, data in self.tracked_objects.items():
            # Check for known person confirmation
            if (data['face_hits'] >= FRAMES_REQUIRED_FOR_CONFIRMATION and
                data['confirmed_name'] != data['name'] and
                data['name'] != "Nguoi la" and
                not data['alert_sent']):
                
                data['confirmed_name'] = data['name']
                data['alert_sent'] = True
                alerts.append((track_id, 'nguoi_quen', {
                    'name': data['name'],
                    'distance': data['distance']
                }))
            
            # Check for stranger confirmation
            if (data['confirmed_name'] is None):
                data['frames_unidentified'] += 1
                
                if (data['frames_unidentified'] > STRANGER_CONFIRM_FRAMES and
                    not data['stranger_alert_sent']):
                    
                    data['stranger_alert_sent'] = True
                    alerts.append((track_id, 'nguoi_la', {}))
        
        return alerts
    
    def _prune_stale_tracks(self, now: float) -> None:
        """Remove tracks that haven't been seen recently"""
        stale_ids = [
            track_id for track_id, data in self.tracked_objects.items()
            if now - data['last_seen_by_detector'] > TRACKER_TIMEOUT_SECONDS
        ]
        for track_id in stale_ids:
            del self.tracked_objects[track_id]
    
    def clear_all_tracks(self) -> None:
        """Clear all tracked objects"""
        self.tracked_objects.clear()
        logger.info("All tracks cleared")
    
    @staticmethod
    def _calculate_iou(boxA: Tuple, boxB: Tuple) -> float:
        """Calculate Intersection over Union between two boxes"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union_area = float(boxA_area + boxB_area - inter_area)
        
        return inter_area / union_area if union_area > 0 else 0
    
    @staticmethod
    def _parse_tracked_output(tracked) -> Tuple[Optional[np.ndarray], Optional[List]]:
        """Parse SORT tracker output into (xyxy_array, id_list)"""
        try:
            # Handle supervision.Detections-like objects
            if hasattr(tracked, "xyxy"):
                xyxy = np.asarray(getattr(tracked, "xyxy"))
                ids = None
                
                for attr_name in ("tracker_id", "ids", "track_id", "id"):
                    if hasattr(tracked, attr_name):
                        try:
                            raw_ids = np.asarray(getattr(tracked, attr_name)).ravel().tolist()
                            # Map negative IDs (e.g., -1 for immature) to None
                            ids = [int(i) if (i is not None and int(i) >= 0) else None for i in raw_ids]
                            break
                        except Exception:
                            continue
                
                if ids is None:
                    ids = [None] * len(xyxy)
                
                return xyxy, ids
            
            # Handle numpy arrays
            if isinstance(tracked, np.ndarray):
                if tracked.ndim == 2 and tracked.shape[1] >= 4:
                    if tracked.shape[1] == 4:
                        return tracked.astype(float), [None] * tracked.shape[0]
                    
                    # Check if last column is IDs
                    last_col = tracked[:, -1]
                    if np.all(np.isfinite(last_col)) and np.allclose(last_col, np.round(last_col)):
                        ids = last_col.astype(int).tolist()
                        ids = [i if i >= 0 else None for i in ids]
                        return tracked[:, :4].astype(float), ids
                    else:
                        return tracked[:, :4].astype(float), [None] * tracked.shape[0]
            
            return None, None
        except Exception as e:
            logger.error(f"Error parsing tracked output: {e}")
            return None, None

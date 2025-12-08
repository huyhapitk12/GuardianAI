# core/detection/fire_tracking.py
"""Fire object tracking with Red Alert Mode"""

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

from config import settings


@dataclass
class TrackedFireObject:
    """Single tracked fire object"""
    id: int
    bbox: Tuple[int, int, int, int]
    area: float
    first_seen: float
    last_seen: float
    age: int = 0  # frames
    stability_score: float = 0.0
    matched_count: int = 0
    
    def update(self, bbox: Tuple, area: float, now: float):
        """Update object with new detection"""
        self.bbox = bbox
        self.area = area
        self.last_seen = now
        self.age += 1
        self.matched_count += 1
        
        # Calculate stability (how consistent the detections are)
        if self.age > 0:
            self.stability_score = min(1.0, self.matched_count / self.age)


class RedAlertMode:
    """Red Alert Mode manager"""
    
    __slots__ = ('active', 'until', 'lockdown_duration')
    
    def __init__(self):
        self.active = False
        self.until = 0.0
        self.lockdown_duration = settings.get('fire_logic.lockdown_seconds', 300)
    
    def activate(self, now: float):
        """Activate Red Alert Mode"""
        if not self.active:
            print(f"🔴 RED ALERT ACTIVATED - Lockdown for {self.lockdown_duration}s")
        self.active = True
        self.until = now + self.lockdown_duration
    
    def is_active(self, now: float) -> bool:
        """Check if Red Alert is currently active"""
        if self.active and now > self.until:
            print("🟢 Red Alert expired")
            self.active = False
            self.until = 0.0
        return self.active
    
    def reset(self):
        """Reset Red Alert Mode"""
        self.active = False
        self.until = 0.0


class FireTracker:
    """Advanced fire tracking with growth monitoring and Red Alert"""
    
    __slots__ = ('_objects', '_next_id', '_red_alert', '_recent_detections',
                 '_yellow_frames', '_config')
    
    def __init__(self):
        self._objects: Dict[int, TrackedFireObject] = {}
        self._next_id = 1
        self._red_alert = RedAlertMode()
        self._recent_detections: deque = deque(maxlen=150)
        self._yellow_frames = deque(maxlen=20)
        
        # Load config
        self._config = {
            'yellow_alert_frames': settings.get('fire_logic.yellow_alert_frames', 8),
            'growth_threshold': settings.get('fire_logic.red_alert_growth_threshold', 1.3),
            'growth_window': settings.get('fire_logic.red_alert_growth_window', 10),
            'area_threshold': settings.get('fire_logic.red_alert_area_threshold', 0.05),
            'iou_threshold': settings.get('fire_logic.object_analysis.iou_threshold', 0.4),
            'min_age_warning': settings.get('fire_logic.object_analysis.min_age_for_warning', 10),
            'min_stability': settings.get('fire_logic.object_analysis.min_stability_for_warning', 0.8),
            'max_age': settings.get('fire_logic.object_analysis.max_age', 20),
        }
    
    def update(self, detections: List[dict], now: float) -> Tuple[bool, bool, bool]:
        """
        Update tracked objects and check alert conditions
        
        Returns:
            (should_alert, is_yellow, is_red_alert)
        """
        # Update objects with IOU matching
        self._match_and_update(detections, now)
        
        # Clean up old objects
        self._cleanup(now)
        
        # Track recent detections for growth analysis
        if detections:
            total_area = sum(d['area'] for d in detections)
            self._recent_detections.append({'time': now, 'area': total_area})
        
        # Check Red Alert conditions
        is_red = self._check_red_alert(detections, now)
        
        # Yellow alert: need consecutive frames
        has_fire = len(detections) > 0
        self._yellow_frames.append(has_fire)
        
        consecutive_count = sum(1 for x in self._yellow_frames if x)
        is_yellow = consecutive_count >= self._config['yellow_alert_frames']
        
        # Determine if should send alert
        should_alert = is_red or (is_yellow and not self._red_alert.active)
        
        return should_alert, is_yellow, is_red
    
    def _match_and_update(self, detections: List[dict], now: float):
        """Match detections with tracked objects using IOU"""
        if not detections:
            return
        
        # Calculate IOU matrix
        matched_dets = set()
        matched_objs = set()
        
        for obj_id, obj in self._objects.items():
            best_iou, best_idx = 0, -1
            
            for i, det in enumerate(detections):
                if i in matched_dets:
                    continue
                iou = self._calc_iou(obj.bbox, det['bbox'])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            
            if best_iou > self._config['iou_threshold']:
                det = detections[best_idx]
                obj.update(det['bbox'], det['area'], now)
                matched_dets.add(best_idx)
                matched_objs.add(obj_id)
        
        # Create new objects for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_dets:
                self._objects[self._next_id] = TrackedFireObject(
                    id=self._next_id,
                    bbox=det['bbox'],
                    area=det['area'],
                    first_seen=now,
                    last_seen=now
                )
                self._next_id += 1
    
    def _cleanup(self, now: float):
        """Remove stale objects"""
        max_age = self._config['max_age']
        self._objects = {
            k: v for k, v in self._objects.items()
            if (now - v.last_seen) < 3.0 or v.age < max_age
        }
    
    def _check_red_alert(self, detections: List[dict], now: float) -> bool:
        """Check if Red Alert should be triggered"""
        # Check if already in Red Alert
        if self._red_alert.is_active(now):
            # Stay in Red Alert if still detecting fire
            if detections:
                self._red_alert.activate(now)  # Extend lockdown
                return True
            return False
        
        # Condition 1: Large fire area
        if detections:
            total_area = sum(d['area'] for d in detections)
            if total_area > self._config['area_threshold']:
                self._red_alert.activate(now)
                return True
        
        # Condition 2: Rapid growth
        if self._check_fire_growth():
            self._red_alert.activate(now)
            return True
        
        # Condition 3: Multiple stable fire objects
        stable_objects = [
            obj for obj in self._objects.values()
            if obj.age >= self._config['min_age_warning']
            and obj.stability_score >= self._config['min_stability']
        ]
        if len(stable_objects) >= 2:
            print(f"🔥 {len(stable_objects)} stable fire objects detected")
            self._red_alert.activate(now)
            return True
        
        return False
    
    def _check_fire_growth(self) -> bool:
        """Check if fire is growing rapidly"""
        if len(self._recent_detections) < 5:
            return False
        
        now = time.time()
        window = self._config['growth_window']
        threshold = self._config['growth_threshold']
        
        # Get current detections (last 1s)
        recent = [d for d in self._recent_detections if now - d['time'] < 1.0]
        if not recent:
            return False
        avg_current = np.mean([d['area'] for d in recent])
        
        # Get past detections (around growth_window seconds ago)
        past = [d for d in self._recent_detections 
                if window - 1.0 < now - d['time'] < window + 1.0]
        if not past:
            return False
        avg_past = np.mean([d['area'] for d in past])
        
        # Check growth rate
        if avg_past > 0 and avg_current > avg_past * threshold:
            print(f"🔥 RAPID GROWTH: {avg_past:.4f} → {avg_current:.4f} ({avg_current/avg_past:.2f}x)")
            return True
        
        return False
    
    @staticmethod
    def _calc_iou(box1: Tuple, box2: Tuple) -> float:
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
    
    @property
    def is_red_alert(self) -> bool:
        """Check if currently in Red Alert"""
        return self._red_alert.is_active(time.time())
    
    @property
    def tracked_objects(self) -> List[TrackedFireObject]:
        return list(self._objects.values())

    @property
    def is_yellow_alert(self) -> bool:
        """Check if currently in Yellow Alert"""
        consecutive_count = sum(1 for x in self._yellow_frames if x)
        return consecutive_count >= self._config['yellow_alert_frames']

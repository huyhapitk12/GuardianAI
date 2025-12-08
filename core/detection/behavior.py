"""Behavior/Anomaly detection with pose estimation"""

from __future__ import annotations
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False

from config import settings


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PoseResult:
    """Pose extraction result"""
    keypoints: Optional[np.ndarray]  # (17, 2) COCO format
    confidence: Optional[np.ndarray]  # (17,)
    bbox: Optional[Tuple[int, int, int, int]]
    
    @property
    def is_valid(self) -> bool:
        return self.keypoints is not None and not np.isnan(self.keypoints).any()


@dataclass
class BehaviorResult:
    """Behavior analysis result"""
    score: float
    is_anomaly: bool
    pose: Optional[PoseResult]
    segment_scores: Optional[np.ndarray] = None
    timestamp: float = 0.0


# ============================================================================
# MODEL
# ============================================================================

class TemporalAttention(nn.Module):
    """Temporal attention mechanism"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = np.sqrt(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attn = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        return torch.bmm(attn, V)


class DeepMIL(nn.Module):
    """Deep Multiple Instance Learning for anomaly detection"""
    
    def __init__(self, input_dim: int = 68, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.attention = TemporalAttention(hidden_dim * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x, _ = self.lstm(x)
        x = self.attention(x)
        return self.classifier(x).squeeze(-1)


# ============================================================================
# POSE EXTRACTION
# ============================================================================

class PoseExtractor:
    """Extract COCO-format keypoints using MediaPipe"""
    
    # MediaPipe to COCO keypoint mapping
    MP_TO_COCO = {
        0: 0,    # nose
        2: 1,    # left_eye
        5: 2,    # right_eye
        7: 3,    # left_ear
        8: 4,    # right_ear
        11: 5,   # left_shoulder
        12: 6,   # right_shoulder
        13: 7,   # left_elbow
        14: 8,   # right_elbow
        15: 9,   # left_wrist
        16: 10,  # right_wrist
        23: 11,  # left_hip
        24: 12,  # right_hip
        25: 13,  # left_knee
        26: 14,  # right_knee
        27: 15,  # left_ankle
        28: 16,  # right_ankle
    }
    
    def __init__(self, model_complexity: int = 1):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe not available")
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract(self, frame: np.ndarray) -> PoseResult:
        """Extract pose from frame"""
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if not results.pose_landmarks:
            return PoseResult(None, None, None)
        
        landmarks = results.pose_landmarks.landmark
        keypoints = np.zeros((17, 2), dtype=np.float32)
        confidence = np.zeros(17, dtype=np.float32)
        
        for mp_idx, coco_idx in self.MP_TO_COCO.items():
            lm = landmarks[mp_idx]
            keypoints[coco_idx] = [lm.x * w, lm.y * h]
            confidence[coco_idx] = lm.visibility
        
        # Compute bbox
        valid_pts = keypoints[confidence > 0.5]
        if len(valid_pts) > 0:
            margin = 20
            x_min, y_min = valid_pts.min(axis=0) - margin
            x_max, y_max = valid_pts.max(axis=0) + margin
            bbox = (
                int(max(0, x_min)),
                int(max(0, y_min)),
                int(min(w, x_max)),
                int(min(h, y_max))
            )
        else:
            bbox = None
        
        return PoseResult(keypoints, confidence, bbox)
    
    def close(self):
        self.pose.close()


# ============================================================================
# FEATURE PROCESSING
# ============================================================================

class TrajectoryProcessor:
    """Convert keypoint sequences to model features"""
    
    def __init__(self, num_segments: int = 32, feature_dim: int = 68):
        self.num_segments = num_segments
        self.feature_dim = feature_dim
    
    def process(self, keypoints_seq: np.ndarray) -> Optional[np.ndarray]:
        """Process keypoint sequence into features"""
        if keypoints_seq is None or len(keypoints_seq) < 2:
            return None
        
        try:
            # Normalize by hip center
            hip_center = (keypoints_seq[:, 11, :] + keypoints_seq[:, 12, :]) / 2.0
            normalized = keypoints_seq - hip_center[:, None, :]
            
            # Scale by shoulder distance
            shoulder_dist = np.linalg.norm(
                keypoints_seq[:, 5, :] - keypoints_seq[:, 6, :],
                axis=1, keepdims=True
            )
            scale = np.clip(np.mean(shoulder_dist), 1e-6, None)
            normalized = normalized / scale
            
            # Compute velocity
            velocity = np.zeros_like(normalized)
            velocity[1:] = normalized[1:] - normalized[:-1]
            
            # Concatenate: (T, 34) + (T, 34) = (T, 68)
            features = np.concatenate([
                normalized.reshape(-1, 34),
                velocity.reshape(-1, 34)
            ], axis=1)
            
            # Segment into fixed length
            segments = self._segment(features)
            
            if np.isnan(segments).any() or np.isinf(segments).any():
                return None
            
            return segments
            
        except Exception:
            return None
    
    def _segment(self, features: np.ndarray) -> np.ndarray:
        """Segment features into fixed number of segments"""
        T = features.shape[0]
        segments = np.zeros((self.num_segments, self.feature_dim), dtype=np.float32)
        
        if T <= self.num_segments:
            indices = np.linspace(0, T - 1, self.num_segments)
            for i, idx in enumerate(indices):
                segments[i] = features[int(idx)]
        else:
            splits = np.array_split(np.arange(T), self.num_segments)
            for i, split in enumerate(splits):
                if len(split) > 0:
                    segments[i] = np.mean(features[split], axis=0)
        
        return segments


class SlidingWindowBuffer:
    """Buffer for sliding window analysis"""
    
    def __init__(self, window_size: int = 64, stride: int = 16, num_segments: int = 32):
        self.window_size = window_size
        self.stride = stride
        self.buffer = deque(maxlen=window_size)
        self.frame_count = 0
        self.processor = TrajectoryProcessor(num_segments)
    
    def add(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """Add keypoints and return features if ready"""
        self.buffer.append(keypoints)
        self.frame_count += 1
        
        if len(self.buffer) >= self.window_size and self.frame_count % self.stride == 0:
            keypoints_array = np.array(list(self.buffer))
            return self.processor.process(keypoints_array)
        
        return None
    
    def reset(self):
        self.buffer.clear()
        self.frame_count = 0


# ============================================================================
# VISUALIZATION
# ============================================================================

class BehaviorVisualizer:
    """Draw pose and anomaly visualization"""
    
    # COCO skeleton connections
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def get_color(self, score: float) -> Tuple[int, int, int]:
        """Get color based on anomaly score"""
        if score >= self.threshold:
            return (0, 0, 255)  # Red - anomaly
        elif score >= 0.3:
            return (0, 165, 255)  # Orange - suspicious
        return (0, 255, 0)  # Green - normal
    
    def draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0)
    ):
        """Draw skeleton on frame"""
        # Draw bones
        for i, j in self.SKELETON:
            if confidence[i] > 0.3 and confidence[j] > 0.3:
                pt1 = tuple(keypoints[i].astype(int))
                pt2 = tuple(keypoints[j].astype(int))
                cv2.line(frame, pt1, pt2, color, 2)
        
        # Draw joints
        for pt, conf in zip(keypoints, confidence):
            if conf > 0.3:
                center = tuple(pt.astype(int))
                cv2.circle(frame, center, 4, color, -1)
                cv2.circle(frame, center, 4, (255, 255, 255), 1)
    
    def draw_score(self, frame: np.ndarray, score: float, is_anomaly: bool):
        """Draw anomaly score on frame"""
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)
        
        # Score text
        text = f"Behavior: {score:.2f}"
        cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Warning if anomaly
        if is_anomaly:
            cv2.putText(frame, "⚠ ANOMALY DETECTED", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def draw_timeline(self, frame: np.ndarray, segment_scores: np.ndarray):
        """Draw temporal score timeline"""
        if segment_scores is None:
            return
        
        h, w = frame.shape[:2]
        n = len(segment_scores)
        seg_width = (w - 40) // n
        
        y1 = h - 50
        y2 = h - 20
        
        for i, score in enumerate(segment_scores):
            x1 = 20 + i * seg_width
            x2 = x1 + seg_width - 1
            color = self.get_color(score)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        
        # Border
        cv2.rectangle(frame, (20, y1), (w - 20, y2), (255, 255, 255), 1)


# ============================================================================
# MAIN DETECTOR
# ============================================================================

class BehaviorAnalyzer:
    """Complete behavior analysis pipeline"""
    
    __slots__ = (
        'device', 'threshold', 'model', 'loaded',
        'pose_extractor', 'buffer', 'visualizer',
        'current_score', 'current_segments', 'current_pose',
        '_last_alert_time', '_alert_cooldown'
    )
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        threshold: float = 0.5,
        window_size: int = 64,
        stride: int = 16
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.loaded = False
        
        self.current_score = 0.0
        self.current_segments = None
        self.current_pose = None
        
        self._last_alert_time = 0
        self._alert_cooldown = settings.get('behavior.alert_cooldown', 30)
        
        # Load model
        self.model = DeepMIL(input_dim=68, hidden_dim=256, dropout=0.3)
        self._load_model(model_path)
        
        # Components
        if MEDIAPIPE_AVAILABLE:
            self.pose_extractor = PoseExtractor(model_complexity=1)
        else:
            self.pose_extractor = None
            print("⚠️ MediaPipe not available - behavior analysis disabled")
        
        self.buffer = SlidingWindowBuffer(window_size, stride, num_segments=32)
        self.visualizer = BehaviorVisualizer(threshold)
    
    def _load_model(self, model_path: str):
        """Load pre-trained model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            print(f"✅ Behavior model loaded from {model_path}")
            
        except Exception as e:
            print(f"❌ Behavior model load failed: {e}")
            self.loaded = False
    
    def process_frame(self, frame: np.ndarray) -> BehaviorResult:
        """Process single frame
        
        Returns:
            BehaviorResult with score and pose info
        """
        if not self.loaded or self.pose_extractor is None:
            return BehaviorResult(0.0, False, None)
        
        # Extract pose
        pose = self.pose_extractor.extract(frame)
        self.current_pose = pose
        
        if not pose.is_valid:
            return BehaviorResult(self.current_score, False, pose)
        
        # Add to buffer
        features = self.buffer.add(pose.keypoints)
        
        # Run inference if ready
        if features is not None:
            with torch.no_grad():
                x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                self.current_segments = self.model(x).squeeze(0).cpu().numpy()
                self.current_score = float(self.current_segments.max())
        
        is_anomaly = self.current_score >= self.threshold
        
        return BehaviorResult(
            score=self.current_score,
            is_anomaly=is_anomaly,
            pose=pose,
            segment_scores=self.current_segments,
            timestamp=time.time()
        )
    
    def draw_on_frame(self, frame: np.ndarray, result: BehaviorResult = None):
        """Draw visualization on frame"""
        if result is None:
            result = BehaviorResult(self.current_score, self.current_score >= self.threshold, self.current_pose)
        
        # Draw skeleton
        if result.pose and result.pose.is_valid:
            color = self.visualizer.get_color(result.score)
            self.visualizer.draw_skeleton(
                frame,
                result.pose.keypoints,
                result.pose.confidence,
                color
            )
        
        # Draw score
        self.visualizer.draw_score(frame, result.score, result.is_anomaly)
        
        # Draw timeline
        if result.segment_scores is not None:
            self.visualizer.draw_timeline(frame, result.segment_scores)
    
    def should_alert(self) -> bool:
        """Check if should trigger alert (respects cooldown)"""
        now = time.time()
        if now - self._last_alert_time >= self._alert_cooldown:
            if self.current_score >= self.threshold:
                self._last_alert_time = now
                return True
        return False
    
    def reset(self):
        """Reset state"""
        self.buffer.reset()
        self.current_score = 0.0
        self.current_segments = None
        self.current_pose = None
    
    def close(self):
        """Cleanup resources"""
        if self.pose_extractor:
            self.pose_extractor.close()
        self.reset()
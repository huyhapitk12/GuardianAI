# core/detection/behavior_analyzer.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import time


# ==================== DATA CLASSES ====================

@dataclass
class PoseResult:
    """Result from pose extraction"""
    keypoints: np.ndarray  # (17, 2) COCO format
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    is_valid: bool = True


@dataclass
class BehaviorResult:
    """Result from behavior analysis"""
    score: float
    is_anomaly: bool
    segment_scores: np.ndarray
    attention_weights: np.ndarray
    pose: Optional[PoseResult] = None
    timestamp: float = 0.0


# ==================== POSE EXTRACTOR ====================

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
    
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def extract(self, frame: np.ndarray) -> PoseResult:
        """Extract COCO keypoints from frame"""
        
        # Convert BGR to RGB
        rgb_frame = frame[:, :, ::-1] if frame.shape[2] == 3 else frame
        
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return PoseResult(
                keypoints=np.zeros((17, 2)),
                confidence=0.0,
                is_valid=False
            )
        
        h, w = frame.shape[:2]
        landmarks = results.pose_landmarks.landmark
        
        # Extract COCO keypoints
        keypoints = np.zeros((17, 2), dtype=np.float32)
        confidences = []
        
        for mp_idx, coco_idx in self.MP_TO_COCO.items():
            lm = landmarks[mp_idx]
            keypoints[coco_idx] = [lm.x * w, lm.y * h]
            confidences.append(lm.visibility)
        
        avg_confidence = np.mean(confidences)
        
        # Compute bounding box
        valid_kpts = keypoints[keypoints.sum(axis=1) > 0]
        if len(valid_kpts) > 0:
            x_min, y_min = valid_kpts.min(axis=0).astype(int)
            x_max, y_max = valid_kpts.max(axis=0).astype(int)
            bbox = (x_min, y_min, x_max, y_max)
        else:
            bbox = None
        
        return PoseResult(
            keypoints=keypoints,
            confidence=avg_confidence,
            bbox=bbox,
            is_valid=avg_confidence > 0.3
        )
    
    def close(self):
        self.pose.close()


# ==================== TRAJECTORY PROCESSOR ====================

class TrajectoryProcessor:
    """Process pose sequences into model features"""
    
    # Bone connections for COCO format
    BONES = [
        (5, 6),   # shoulders
        (5, 7),   # left upper arm
        (7, 9),   # left forearm
        (6, 8),   # right upper arm
        (8, 10),  # right forearm
        (11, 12), # hips
        (5, 11),  # left torso
        (6, 12),  # right torso
        (11, 13), # left thigh
        (13, 15), # left calf
        (12, 14), # right thigh
        (14, 16), # right calf
        (0, 5),   # nose to left shoulder
        (0, 6),   # nose to right shoulder
    ]
    
    # Joint angles
    ANGLES = [
        (5, 7, 9),    # left elbow
        (6, 8, 10),   # right elbow
        (11, 13, 15), # left knee
        (12, 14, 16), # right knee
        (7, 5, 11),   # left shoulder
        (8, 6, 12),   # right shoulder
    ]
    
    def __init__(self, num_segments: int = 32):
        self.num_segments = num_segments
        self.feature_dim = 91  # Must match training
    
    def compute_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute features from trajectory
        
        Args:
            trajectory: (num_frames, 17, 2) keypoints
            
        Returns:
            features: (num_frames, 91)
        """
        if trajectory is None or len(trajectory) == 0:
            return None
        
        num_frames = len(trajectory)
        features_list = []
        
        for i in range(num_frames):
            frame_features = []
            kpts = trajectory[i]  # (17, 2)
            
            # Hip center
            hip_center = (kpts[11] + kpts[12]) / 2
            
            # 1. Normalized coordinates (34)
            norm_kpts = kpts - hip_center
            frame_features.extend(norm_kpts.flatten())
            
            # 2. Velocity (34)
            if i > 0:
                velocity = kpts - trajectory[i-1]
                frame_features.extend(velocity.flatten())
            else:
                frame_features.extend(np.zeros(34))
            
            # 3. Bone lengths (14)
            for (j1, j2) in self.BONES:
                bone_len = np.linalg.norm(kpts[j2] - kpts[j1])
                frame_features.append(bone_len)
            
            # 4. Joint angles (6)
            for (j1, j2, j3) in self.ANGLES:
                v1 = kpts[j1] - kpts[j2]
                v2 = kpts[j3] - kpts[j2]
                cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angle = np.arccos(np.clip(cos_ang, -1, 1))
                frame_features.append(angle)
            
            # 5. Body metrics (3)
            shoulder_w = np.linalg.norm(kpts[5] - kpts[6])
            hip_w = np.linalg.norm(kpts[11] - kpts[12])
            torso_l = np.linalg.norm(hip_center - (kpts[5] + kpts[6]) / 2)
            frame_features.extend([shoulder_w, hip_w, torso_l])
            
            features_list.append(frame_features)
        
        features = np.array(features_list, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std
        
        return features
    
    def segment(self, features: np.ndarray) -> np.ndarray:
        """Segment features into fixed number of segments"""
        if features is None or len(features) == 0:
            return np.zeros((self.num_segments, self.feature_dim), dtype=np.float32)
        
        num_frames = len(features)
        indices = np.linspace(0, num_frames, self.num_segments + 1, dtype=int)
        
        segments = []
        for i in range(self.num_segments):
            start, end = indices[i], indices[i+1]
            if start >= end:
                end = start + 1
            end = min(end, num_frames)
            start = min(start, num_frames - 1)
            
            segments.append(features[start:end].mean(axis=0))
        
        return np.array(segments, dtype=np.float32)


# ==================== MODEL ====================

class SkeletonEncoder(nn.Module):
    """LSTM-Attention encoder for skeleton sequences"""
    
    def __init__(
        self,
        input_dim: int = 91,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_out = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.attention = nn.Sequential(
            nn.Linear(lstm_out, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended = lstm_out * attn_weights
        
        scores = self.classifier(attended).squeeze(-1)
        return scores, attn_weights.squeeze(-1)


class DeepMIL(nn.Module):
    """Deep Multiple Instance Learning model"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = SkeletonEncoder(
            input_dim=config.get('input_dim', 91),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.5),
            bidirectional=config.get('bidirectional', True)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)


# ==================== SLIDING WINDOW BUFFER ====================

class SlidingWindowBuffer:
    """Buffer for sliding window analysis"""
    
    def __init__(self, window_size: int = 64, stride: int = 16):
        self.window_size = window_size
        self.stride = stride
        self.buffer = deque(maxlen=window_size * 2)  # Prevent memory leak
        self.frame_count = 0
    
    def add(self, keypoints: np.ndarray) -> bool:
        """Add keypoints and return True if ready for analysis"""
        self.buffer.append(keypoints)
        self.frame_count += 1
        
        # Ready when we have enough frames and at stride interval
        return (len(self.buffer) >= self.window_size and 
                self.frame_count % self.stride == 0)
    
    def get_window(self) -> np.ndarray:
        """Get current window"""
        if len(self.buffer) < self.window_size:
            return None
        
        # Get last window_size frames
        window = list(self.buffer)[-self.window_size:]
        return np.array(window, dtype=np.float32)
    
    def clear(self):
        self.buffer.clear()
        self.frame_count = 0


# ==================== MAIN ANALYZER ====================

class BehaviorAnalyzer:
    """Main behavior analysis class"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        threshold: float = 0.5,
        window_size: int = 64,
        stride: int = 16,
        num_segments: int = 32,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.device = torch.device(device)
        self.threshold = threshold
        
        # Initialize components
        self.pose_extractor = PoseExtractor(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.trajectory_processor = TrajectoryProcessor(num_segments=num_segments)
        
        self.buffer = SlidingWindowBuffer(
            window_size=window_size,
            stride=stride
        )
        
        # Load model
        self._load_model(model_path)
        
        # State
        self.last_result: Optional[BehaviorResult] = None
        self._last_valid_pose: Optional[np.ndarray] = None
    
    def _load_model(self, model_path: str):
        """Load pre-trained model"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get config from checkpoint or use defaults
        model_config = checkpoint.get('config', {})
        model_config.setdefault('input_dim', 91)
        model_config.setdefault('hidden_dim', 256)
        
        self.model = DeepMIL(model_config).to(self.device)
        
        # Load weights
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"INFO: Behavior model loaded from {model_path}")
    
    def process_frame(self, frame: np.ndarray) -> Optional[BehaviorResult]:
        """
        Process a single frame
        
        Args:
            frame: BGR image (H, W, 3)
            
        Returns:
            BehaviorResult if analysis ready, None otherwise
        """
        # Extract pose
        pose = self.pose_extractor.extract(frame)
        
        # Use last valid pose if current extraction failed
        if pose.is_valid:
            keypoints = pose.keypoints
            self._last_valid_pose = keypoints
        elif self._last_valid_pose is not None:
            keypoints = self._last_valid_pose
        else:
            return None
        
        # Add to buffer
        ready = self.buffer.add(keypoints)
        
        if not ready:
            return None
        
        # Get trajectory window
        trajectory = self.buffer.get_window()
        
        # Compute features
        features = self.trajectory_processor.compute_features(trajectory)
        if features is None:
            return None
        
        # Segment
        segments = self.trajectory_processor.segment(features)
        
        # Run inference
        with torch.no_grad():
            x = torch.FloatTensor(segments).unsqueeze(0).to(self.device)
            scores, attention = self.model(x)
            
            scores = scores.squeeze(0).cpu().numpy()
            attention = attention.squeeze(0).cpu().numpy()
        
        max_score = float(np.max(scores))
        
        result = BehaviorResult(
            score=max_score,
            is_anomaly=max_score >= self.threshold,
            segment_scores=scores,
            attention_weights=attention,
            pose=pose,
            timestamp=time.time()
        )
        
        self.last_result = result
        return result
    
    def get_last_result(self) -> Optional[BehaviorResult]:
        """Get last analysis result"""
        return self.last_result
    
    def reset(self):
        """Reset buffer and state"""
        self.buffer.clear()
        self.last_result = None
        self._last_valid_pose = None
    
    def close(self):
        """Cleanup resources"""
        self.pose_extractor.close()
        self.reset()


# ==================== VISUALIZATION ====================

def draw_pose(frame: np.ndarray, pose: PoseResult, color=(0, 255, 0)) -> np.ndarray:
    """Draw pose skeleton on frame"""
    import cv2
    
    if not pose.is_valid:
        return frame
    
    frame = frame.copy()
    kpts = pose.keypoints.astype(int)
    
    # Draw keypoints
    for i, (x, y) in enumerate(kpts):
        if x > 0 and y > 0:
            cv2.circle(frame, (x, y), 4, color, -1)
    
    # Draw skeleton
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
        (5, 11), (6, 12), (11, 12),  # torso
        (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    ]
    
    for (i, j) in skeleton:
        if i < len(kpts) and j < len(kpts):
            x1, y1 = kpts[i]
            x2, y2 = kpts[j]
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
    
    return frame


def draw_behavior_result(
    frame: np.ndarray, 
    result: BehaviorResult,
    show_score: bool = True
) -> np.ndarray:
    """Draw behavior analysis result on frame"""
    import cv2
    
    frame = frame.copy()
    
    # Draw pose
    if result.pose and result.pose.is_valid:
        color = (0, 0, 255) if result.is_anomaly else (0, 255, 0)
        frame = draw_pose(frame, result.pose, color)
    
    # Draw score
    if show_score:
        text = f"Anomaly: {result.score:.2f}"
        color = (0, 0, 255) if result.is_anomaly else (0, 255, 0)
        
        cv2.putText(
            frame, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )
        
        if result.is_anomaly:
            cv2.putText(
                frame, "WARNING!", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
    
    return frame

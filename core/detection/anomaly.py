import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = np.sqrt(dim)

    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        attn = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        return torch.bmm(attn, V)


class DeepMIL(nn.Module):
    """Deep MIL for Anomaly Detection"""
    
    def __init__(self, input_dim=68, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=dropout)
        
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

    def forward(self, x):
        x = self.encoder(x)
        x, _ = self.lstm(x)
        x = self.attention(x)
        return self.classifier(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# MEDIAPIPE POSE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PoseResult:
    keypoints: np.ndarray  # (17, 2)
    confidence: np.ndarray  # (17,)
    bbox: Optional[Tuple]
    
    @property
    def is_valid(self):
        return self.keypoints is not None and not np.isnan(self.keypoints).any()


class MediaPipePoseExtractor:
    """Extract COCO 17 keypoints from MediaPipe 33 keypoints"""
    
    MP_TO_COCO = {
        0: 0, 2: 1, 5: 2, 7: 3, 8: 4,           # Head
        11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10,  # Arms
        23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16  # Legs
    }
    
    def __init__(self, model_complexity=1, static_image_mode=False):
        # Reduced model complexity to 1 for better realtime performance
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract(self, frame: np.ndarray) -> PoseResult:
        h, w = frame.shape[:2]
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
            x1, y1 = valid_pts.min(axis=0) - 20
            x2, y2 = valid_pts.max(axis=0) + 20
            bbox = (int(max(0, x1)), int(max(0, y1)), 
                   int(min(w, x2)), int(min(h, y2)))
        else:
            bbox = None
            
        return PoseResult(keypoints, confidence, bbox)
    
    def close(self):
        self.pose.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class TrajectoryProcessor:
    """Convert keypoints → model features"""
    
    def __init__(self, num_segments=32, feature_dim=68):
        self.num_segments = num_segments
        self.feature_dim = feature_dim
        
    def process(self, keypoints_sequence: np.ndarray) -> Optional[np.ndarray]:
        if keypoints_sequence is None or len(keypoints_sequence) < 2:
            return None
            
        try:
            # Normalize by hip center
            hip_center = (keypoints_sequence[:, 11, :] + keypoints_sequence[:, 12, :]) / 2.0
            normalized = keypoints_sequence - hip_center[:, None, :]
            
            # Scale by shoulder distance
            shoulder_dist = np.linalg.norm(
                keypoints_sequence[:, 5, :] - keypoints_sequence[:, 6, :], 
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
            
            if np.isnan(segments).any() or np.isinf(segments).any():
                return None
                
            return segments
            
        except Exception as e:
            logger.error(f"Feature processing error: {e}")
            return None


class SlidingWindowBuffer:
    """Buffer for realtime inference"""
    
    def __init__(self, window_size=64, stride=16, num_segments=32):
        self.window_size = window_size
        self.stride = stride
        self.buffer = deque(maxlen=window_size)
        self.frame_count = 0
        self.processor = TrajectoryProcessor(num_segments)
        
    def add_frame(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        self.buffer.append(keypoints)
        self.frame_count += 1
        
        if len(self.buffer) >= self.window_size and self.frame_count % self.stride == 0:
            keypoints_array = np.array(list(self.buffer))
            return self.processor.process(keypoints_array)
            
        return None
    
    def reset(self):
        self.buffer.clear()
        self.frame_count = 0


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class Visualizer:
    """Draw pose and anomaly scores"""
    
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        
    def get_color(self, score):
        if score >= self.threshold:
            return (0, 0, 255)  # Red
        elif score >= 0.3:
            return (0, 165, 255)  # Orange
        return (0, 255, 0)  # Green
    
    def draw_skeleton(self, frame, keypoints, confidence, color=(255, 255, 0)):
        """Draw skeleton"""
        # frame = frame.copy() # Avoid copy for performance if possible, but safe for now
        
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
                
        return frame
    
    def draw_info(self, frame, score, fps=None):
        """Draw score and status"""
        h, w = frame.shape[:2]
        
        # FPS
        if fps is not None:
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_timeline(self, frame, segment_scores):
        """Draw temporal timeline"""
        h, w = frame.shape[:2]
        n = len(segment_scores)
        seg_width = (w - 40) // n
        
        y1 = h - 60
        y2 = h - 20
        
        for i, score in enumerate(segment_scores):
            x1 = 20 + i * seg_width
            x2 = x1 + seg_width - 1
            color = self.get_color(score)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        
        cv2.rectangle(frame, (20, y1), (w - 20, y2), (255, 255, 255), 1)
        cv2.putText(frame, "Timeline", (20, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """Complete detection pipeline"""
    
    def __init__(self, model_path, device='cuda', threshold=0.5):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # Load model
        logger.info(f"Loading anomaly model from {model_path}")
        self.model = DeepMIL(input_dim=68, hidden_dim=256, dropout=0.3)
        
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
        except Exception as e:
            logger.error(f"Failed to load anomaly model: {e}")
            self.loaded = False
        
        # Components
        # Use model_complexity=1 for better performance in realtime app
        self.pose_extractor = MediaPipePoseExtractor(model_complexity=1)
        self.buffer = SlidingWindowBuffer(window_size=64, stride=16, num_segments=32)
        self.visualizer = Visualizer(threshold=threshold)
        
        self.current_score = 0.0
        self.current_segments = None
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Process a single frame.
        Returns:
            frame: Annotated frame
            score: Current anomaly score
            is_anomaly: Boolean indicating if anomaly is detected
        """
        if not self.loaded:
            return frame, 0.0, False
            
        # Extract pose
        pose = self.pose_extractor.extract(frame)
        
        if pose.is_valid:
            # Add to buffer
            features = self.buffer.add_frame(pose.keypoints)
            
            # Inference if buffer ready
            if features is not None:
                with torch.no_grad():
                    x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    self.current_segments = self.model(x).squeeze(0).cpu().numpy()
                    self.current_score = float(self.current_segments.max())
            
            # Draw skeleton
            color = self.visualizer.get_color(self.current_score)
            frame = self.visualizer.draw_skeleton(
                frame, pose.keypoints, pose.confidence, color
            )
        
        # Draw info
        frame = self.visualizer.draw_info(frame, self.current_score)
        
        if self.current_segments is not None:
            frame = self.visualizer.draw_timeline(frame, self.current_segments)
            
        is_anomaly = self.current_score >= self.threshold
        
        return frame, self.current_score, is_anomaly, pose
    
    def reset(self):
        self.buffer.reset()
        self.current_score = 0.0
        self.current_segments = None
    
    def close(self):
        self.pose_extractor.close()

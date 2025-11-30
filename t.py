"""
═══════════════════════════════════════════════════════════════════════════════
HR-CRIME ANOMALY DETECTION - COMPLETE DEMO (Single File)
═══════════════════════════════════════════════════════════════════════════════

Pipeline: Video/Webcam → MediaPipe Pose → Features → DeepMIL → Anomaly Score

Usage:
    # Video file
    python hr_crime_demo.py --video path/to/video.mp4 --output result.mp4
    
    # Webcam realtime
    python hr_crime_demo.py --realtime --camera 0
    
    # Just predict (no visualization)
    python hr_crime_demo.py --video test.mp4 --no-viz

Requirements:
    pip install torch opencv-python mediapipe numpy tqdm

Author: HR-Crime MediaPipe Implementation
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import time
from pathlib import Path
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import mediapipe as mp
except ImportError:
    print("❌ MediaPipe not installed. Run: pip install mediapipe")
    exit(1)


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
    
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    def __init__(self, model_complexity=2, static_image_mode=False):
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
        """
        Args:
            keypoints_sequence: (T, 17, 2)
        Returns:
            features: (num_segments, 68) or None
        """
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
        frame = frame.copy()
        
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
        color = self.get_color(score)
        
        # Status
        if score >= self.threshold:
            status = "⚠️ ANOMALY"
        elif score >= 0.3:
            status = "⚡ SUSPICIOUS"
        else:
            status = "✓ NORMAL"
        
        # Background rectangle
        cv2.rectangle(frame, (10, 10), (300, 90), (0, 0, 0), -1)
        
        # Score bar
        bar_width = 280
        fill_width = int(bar_width * min(score, 1.0))
        cv2.rectangle(frame, (10, 30), (10 + bar_width, 50), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 30), (10 + fill_width, 50), color, -1)
        cv2.rectangle(frame, (10, 30), (10 + bar_width, 50), (255, 255, 255), 1)
        
        # Text
        cv2.putText(frame, f"Score: {score:.3f}", (15, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, status, (15, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
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
        print(f"📦 Loading model from {model_path}")
        self.model = DeepMIL(input_dim=68, hidden_dim=256, dropout=0.3)
        
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
        
        # Components
        self.pose_extractor = MediaPipePoseExtractor(model_complexity=2)
        self.processor = TrajectoryProcessor(num_segments=32)
        self.visualizer = Visualizer(threshold=threshold)
        
        print(f"✅ Detector initialized on {self.device}")
        
    def predict_video(self, video_path, output_path=None, show=False):
        """Predict on video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\n📹 Processing: {video_path}")
        print(f"   Frames: {total_frames}, FPS: {fps:.1f}")
        
        # Phase 1: Extract poses
        keypoints_list = []
        frames_list = []
        
        frame_idx = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames_list.append(frame)
            pose = self.pose_extractor.extract(frame)
            
            if pose.is_valid:
                keypoints_list.append(pose.keypoints)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                eta = (total_frames - frame_idx) * elapsed / frame_idx
                print(f"   Extracting poses: {frame_idx}/{total_frames} "
                      f"(ETA: {eta:.1f}s)")
        
        cap.release()
        
        if len(keypoints_list) < 10:
            print("⚠️ Not enough poses detected")
            return {'error': 'insufficient_poses'}
        
        # Phase 2: Process trajectory
        print(f"   Valid poses: {len(keypoints_list)}/{total_frames} "
              f"({len(keypoints_list)/total_frames*100:.1f}%)")
        
        keypoints_array = np.array(keypoints_list)
        features = self.processor.process(keypoints_array)
        
        if features is None:
            print("❌ Feature processing failed")
            return {'error': 'feature_processing_failed'}
        
        # Phase 3: Inference
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            segment_scores = self.model(x).squeeze(0).cpu().numpy()
            video_score = float(segment_scores.max())
        
        is_anomaly = video_score >= self.threshold
        
        print(f"\n📊 Results:")
        print(f"   Video Score: {video_score:.4f}")
        print(f"   Status: {'⚠️ ANOMALY' if is_anomaly else '✓ NORMAL'}")
        
        # Phase 4: Generate output
        if output_path or show:
            print(f"\n🎬 Generating annotated video...")
            
            for i, frame in enumerate(frames_list):
                pose = self.pose_extractor.extract(frame)
                
                if pose.is_valid:
                    color = self.visualizer.get_color(video_score)
                    frame = self.visualizer.draw_skeleton(
                        frame, pose.keypoints, pose.confidence, color
                    )
                
                frame = self.visualizer.draw_info(frame, video_score, fps)
                frame = self.visualizer.draw_timeline(frame, segment_scores)
                
                if writer:
                    writer.write(frame)
                
                if show:
                    cv2.imshow('Anomaly Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if (i + 1) % 100 == 0:
                    print(f"   Rendering: {i+1}/{len(frames_list)}")
            
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()
            
            if output_path:
                print(f"✅ Saved to: {output_path}")
        
        return {
            'video_score': video_score,
            'segment_scores': segment_scores,
            'is_anomaly': is_anomaly,
            'num_frames': total_frames,
            'num_poses': len(keypoints_list)
        }
    
    def run_realtime(self, camera_id=0, window_size=64, stride=16):
        """Run on webcam"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
        
        buffer = SlidingWindowBuffer(window_size, stride, num_segments=32)
        
        print("\n🎥 Realtime Detection Started")
        print("   Press 'q' to quit, 'r' to reset buffer")
        
        current_score = 0.0
        current_segments = None
        fps_counter = 0
        fps_start = time.time()
        fps = 0.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract pose
            pose = self.pose_extractor.extract(frame)
            
            if pose.is_valid:
                # Add to buffer
                features = buffer.add_frame(pose.keypoints)
                
                # Inference if buffer ready
                if features is not None:
                    with torch.no_grad():
                        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                        current_segments = self.model(x).squeeze(0).cpu().numpy()
                        current_score = float(current_segments.max())
                
                # Draw skeleton
                color = self.visualizer.get_color(current_score)
                frame = self.visualizer.draw_skeleton(
                    frame, pose.keypoints, pose.confidence, color
                )
            
            # Draw info
            frame = self.visualizer.draw_info(frame, current_score, fps)
            
            if current_segments is not None:
                frame = self.visualizer.draw_timeline(frame, current_segments)
            
            # FPS calculation
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start = time.time()
            
            cv2.imshow('Realtime Anomaly Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                buffer.reset()
                current_score = 0.0
                print("🔄 Buffer reset")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def close(self):
        self.pose_extractor.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='HR-Crime Anomaly Detection with MediaPipe',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Video file with visualization
  python hr_crime_demo.py --video test.mp4 --output result.mp4
  
  # Video file without visualization (faster)
  python hr_crime_demo.py --video test.mp4 --no-viz
  
  # Realtime webcam
  python hr_crime_demo.py --realtime --camera 0
  
  # Custom threshold
  python hr_crime_demo.py --video test.mp4 --threshold 0.3
        """
    )
    
    # Mode
    parser.add_argument('--video', '-v', help='Input video path')
    parser.add_argument('--realtime', '-r', action='store_true', 
                       help='Run on webcam')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera ID for realtime mode')
    
    # Model
    parser.add_argument('--model', '-m', required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='Anomaly threshold (0-1)')
    
    # Output
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--show', '-s', action='store_true',
                       help='Show video while processing')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization (faster)')
    
    # Realtime params
    parser.add_argument('--window', '-w', type=int, default=64,
                       help='Window size for realtime (frames)')
    parser.add_argument('--stride', type=int, default=16,
                       help='Stride for realtime (frames)')
    
    args = parser.parse_args()
    
    # Validate
    if not args.video and not args.realtime:
        parser.error("Must specify --video or --realtime")
    
    if args.video and args.realtime:
        parser.error("Cannot use both --video and --realtime")
    
    # Banner
    print("═" * 70)
    print("  HR-CRIME ANOMALY DETECTION - MediaPipe Implementation")
    print("═" * 70)
    
    # Initialize detector
    detector = AnomalyDetector(
        model_path=args.model,
        device=args.device,
        threshold=args.threshold
    )
    
    try:
        if args.realtime:
            # Realtime mode
            detector.run_realtime(
                camera_id=args.camera,
                window_size=args.window,
                stride=args.stride
            )
        else:
            # Video mode
            show = args.show and not args.no_viz
            output = None if args.no_viz else args.output
            
            result = detector.predict_video(
                video_path=args.video,
                output_path=output,
                show=show
            )
            
            # Print summary
            if 'error' not in result:
                print("\n" + "═" * 70)
                print("  DETECTION SUMMARY")
                print("═" * 70)
                print(f"  Video: {args.video}")
                print(f"  Score: {result['video_score']:.4f}")
                print(f"  Status: {'⚠️ ANOMALY DETECTED' if result['is_anomaly'] else '✓ NORMAL'}")
                print(f"  Frames: {result['num_frames']}")
                print(f"  Valid Poses: {result['num_poses']} "
                      f"({result['num_poses']/result['num_frames']*100:.1f}%)")
                
                # Top suspicious segments
                if result['segment_scores'] is not None:
                    sorted_idx = result['segment_scores'].argsort()[::-1][:5]
                    print(f"\n  Top Suspicious Segments:")
                    for i, idx in enumerate(sorted_idx):
                        score = result['segment_scores'][idx]
                        print(f"    {i+1}. Segment {idx:2d}: {score:.4f}")
                
                print("═" * 70)
    
    finally:
        detector.close()
        print("\n✅ Done!")


if __name__ == '__main__':
    main()
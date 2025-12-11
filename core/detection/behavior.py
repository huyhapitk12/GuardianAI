# core/detection/behavior.py
# =============================================================================
# MODULE PHÂN TÍCH HÀNH VI - BEHAVIOR ANALYSIS
# =============================================================================
# Module này dùng AI để phát hiện hành vi bất thường (như ngã, đánh nhau...)
# Cách hoạt động:
# 1. Trích xuất bộ xương (skeleton/pose) từ video bằng MediaPipe
# 2. Đưa chuỗi chuyển động của bộ xương vào mạng AI (LSTM)
# 3. AI sẽ chấm điểm "bất thường" (0.0 - 1.0)
# =============================================================================

import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass

# Import MediaPipe (thư viện trích xuất bộ xương)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False

from config import settings


# =============================================================================
# CẤU TRÚC DỮ LIỆU
# =============================================================================

@dataclass
class PoseResult:
    """Kết quả trích xuất bộ xương"""
    keypoints: any       # Tọa độ các khớp (mũi, vai, khuỷu tay...)
    confidence: any      # Độ tin cậy của từng khớp
    bbox: any            # Hình chữ nhật bao quanh người
    
    @property
    def is_valid(self):
        # Kiểm tra dữ liệu có hợp lệ không
        return self.keypoints is not None and not np.isnan(self.keypoints).any()


@dataclass
class BehaviorResult:
    """Kết quả phân tích hành vi"""
    score: float         # Điểm bất thường (0.0 - 1.0)
    is_anomaly: bool     # Có phải bất thường không?
    pose: any            # Dữ liệu bộ xương (để vẽ lên hình)
    segment_scores: any = None
    timestamp: float = 0.0


# =============================================================================
# MÔ HÌNH AI (NEURAL NETWORK)
# =============================================================================
# Phần này định nghĩa kiến trúc mạng AI
# Sử dụng LSTM (Long Short-Term Memory) để hiểu chuỗi thời gian
# =============================================================================

class TemporalAttention(nn.Module):
    """Cơ chế chú ý (Attention) - giúp AI tập trung vào đoạn quan trọng"""
    
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = np.sqrt(dim)
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Tính toán độ chú ý
        attn = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        return torch.bmm(attn, V)


class DeepMIL(nn.Module):
    """Mạng chính để phát hiện bất thường"""
    
    def __init__(self, input_dim=68, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        # Mã hóa thông tin đầu vào
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM để học chuỗi thời gian (chuyển động)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,  # Học cả 2 chiều (quá khứ <-> tương lai trong cửa sổ)
            dropout=dropout
        )
        
        # Cơ chế chú ý
        self.attention = TemporalAttention(hidden_dim * 2)
        
        # Bộ phân loại đưa ra điểm số cuối cùng
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Đưa về đoạn [0, 1]
        )
        
        self.init_weights()
    
    def init_weights(self):
        """Khởi tạo trọng số ngẫu nhiên"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Tính toán kết quả"""
        x = self.encoder(x)
        x, _ = self.lstm(x)
        x = self.attention(x)
        return self.classifier(x).squeeze(-1)


# =============================================================================
# TRÍCH XUẤT BỘ XƯƠNG (POSE EXTRACTION)
# =============================================================================

class PoseExtractor:
    """Sử dụng MediaPipe để lấy tọa độ các khớp xương"""
    
    # Bản đồ ánh xạ từ MediaPipe sang chuẩn COCO (17 điểm)
    MP_TO_COCO = {
        0: 0,    # Mũi
        2: 1,    # Mắt trái
        5: 2,    # Mắt phải
        7: 3,    # Tai trái
        8: 4,    # Tai phải
        11: 5,   # Vai trái
        12: 6,   # Vai phải
        13: 7,   # Khuỷu tay trái
        14: 8,   # Khuỷu tay phải
        15: 9,   # Cổ tay trái
        16: 10,  # Cổ tay phải
        23: 11,  # Hông trái
        24: 12,  # Hông phải
        25: 13,  # Đầu gối trái
        26: 14,  # Đầu gối phải
        27: 15,  # Cổ chân trái
        28: 16,  # Cổ chân phải
    }
    
    def __init__(self, model_complexity=1):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("Chưa cài đặt MediaPipe!")
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract(self, frame):
        """Trích xuất bộ xương từ ảnh"""
        h, w = frame.shape[:2]
        
        # MediaPipe cần ảnh RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if not results.pose_landmarks:
            return PoseResult(None, None, None)
        
        landmarks = results.pose_landmarks.landmark
        keypoints = np.zeros((17, 2), dtype=np.float32)
        confidence = np.zeros(17, dtype=np.float32)
        
        # Chuyển đổi sang chuẩn COCO
        for mp_idx, coco_idx in self.MP_TO_COCO.items():
            lm = landmarks[mp_idx]
            # MediaPipe trả về tọa độ chuẩn hóa (0-1), cần nhân với kích thước ảnh
            keypoints[coco_idx] = [lm.x * w, lm.y * h]
            confidence[coco_idx] = lm.visibility
        
        # Tính bounding box bao quanh người
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
        try:
            self.pose.close()
        except:
            pass


# =============================================================================
# XỬ LÝ ĐẶC TRƯNG (FEATURE PROCESSING)
# =============================================================================

class TrajectoryProcessor:
    """Chuyển đổi chuỗi chuyển động thành đặc trưng để đưa vào AI"""
    
    def __init__(self, num_segments=32, feature_dim=68):
        self.num_segments = num_segments
        self.feature_dim = feature_dim
    
    def process(self, keypoints_seq):
        """Xử lý chuỗi keypoints"""
        if keypoints_seq is None or len(keypoints_seq) < 2:
            return None
        
        try:
            # 1. Chuẩn hóa vị trí (đưa về gốc tọa độ là hông)
            hip_center = (keypoints_seq[:, 11, :] + keypoints_seq[:, 12, :]) / 2.0
            normalized = keypoints_seq - hip_center[:, None, :]
            
            # 2. Chuẩn hóa kích thước (chia cho chiều rộng vai)
            # Để người to hay nhỏ thì AI đều hiểu như nhau
            shoulder_dist = np.linalg.norm(
                keypoints_seq[:, 5, :] - keypoints_seq[:, 6, :],
                axis=1, keepdims=True
            )
            scale = np.clip(np.mean(shoulder_dist), 1e-6, None)
            normalized = normalized / scale
            
            # 3. Tính vận tốc (độ thay đổi vị trí giữa các frame)
            velocity = np.zeros_like(normalized)
            velocity[1:] = normalized[1:] - normalized[:-1]
            
            # 4. Gộp vị trí và vận tốc lại
            # (Thời gian, 34 điểm) + (Thời gian, 34 điểm) = (Thời gian, 68 điểm)
            features = np.concatenate([
                normalized.reshape(-1, 34),
                velocity.reshape(-1, 34)
            ], axis=1)
            
            # 5. Chia thành các đoạn cố định (Segment)
            segments = self.segment_features(features)
            
            if np.isnan(segments).any() or np.isinf(segments).any():
                return None
            
            return segments
            
        except Exception:
            return None
    
    def segment_features(self, features):
        """Chia features thành số lượng đoạn cố định (để input vào AI có kích thước cố định)"""
        T = features.shape[0]
        segments = np.zeros((self.num_segments, self.feature_dim), dtype=np.float32)
        
        if T <= self.num_segments:
            # Nếu ít frame, lấy mẫu đều
            indices = np.linspace(0, T - 1, self.num_segments)
            for i, idx in enumerate(indices):
                segments[i] = features[int(idx)]
        else:
            # Nếu nhiều frame, chia đều và tính trung bình mỗi đoạn
            splits = np.array_split(np.arange(T), self.num_segments)
            for i, split in enumerate(splits):
                if len(split) > 0:
                    segments[i] = np.mean(features[split], axis=0)
        
        return segments


class SlidingWindowBuffer:
    """Bộ đệm trượt - Lưu các frame gần nhất để phân tích"""
    
    def __init__(self, window_size=64, stride=16, num_segments=32):
        self.window_size = window_size  # Kích thước cửa sổ (số frame)
        self.stride = stride            # Bước nhảy
        self.buffer = deque(maxlen=window_size)
        self.frame_count = 0
        self.processor = TrajectoryProcessor(num_segments)
    
    def add(self, keypoints):
        """Thêm keypoints mới và trả về features nếu đủ dữ liệu"""
        self.buffer.append(keypoints)
        self.frame_count += 1
        
        # Nếu đủ frame và đến lúc kiểm tra
        if len(self.buffer) >= self.window_size and self.frame_count % self.stride == 0:
            keypoints_array = np.array(list(self.buffer))
            return self.processor.process(keypoints_array)
        
        return None
    
    def reset(self):
        self.buffer.clear()
        self.frame_count = 0


# =============================================================================
# TRỰC QUAN HÓA (VISUALIZATION)
# =============================================================================

class BehaviorVisualizer:
    """Vẽ bộ xương và điểm số lên màn hình"""
    
    # Các đường nối bộ xương (để vẽ line)
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),      # Đầu
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Tay
        (5, 11), (6, 12), (11, 12),          # Thân
        (11, 13), (13, 15), (12, 14), (14, 16)   # Chân
    ]
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def get_color(self, score):
        """Lấy màu dựa trên điểm bất thường"""
        if score >= self.threshold:
            return (0, 0, 255)    # Đỏ - Bất thường
        elif score >= 0.3:
            return (0, 165, 255)  # Cam - Nghi ngờ
        return (0, 255, 0)        # Xanh - Bình thường
    
    def draw_skeleton(self, frame, keypoints, confidence, color=(0, 255, 0)):
        """Vẽ bộ xương"""
        # Vẽ xương
        for i, j in self.SKELETON:
            if confidence[i] > 0.3 and confidence[j] > 0.3:
                pt1 = tuple(keypoints[i].astype(int))
                pt2 = tuple(keypoints[j].astype(int))
                cv2.line(frame, pt1, pt2, color, 2)
        
        # Vẽ khớp
        for pt, conf in zip(keypoints, confidence):
            if conf > 0.3:
                center = tuple(pt.astype(int))
                cv2.circle(frame, center, 4, color, -1)
                cv2.circle(frame, center, 4, (255, 255, 255), 1)
    
    def draw_score(self, frame, score, is_anomaly):
        """Vẽ điểm số"""
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)
        
        # Vẽ điểm
        text = f"Hanh vi: {score:.2f}"
        cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Cảnh báo nếu bất thường
        if is_anomaly:
            cv2.putText(frame, "⚠ PHAT HIEN BAT THUONG", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def draw_timeline(self, frame, segment_scores):
        """Vẽ biểu đồ timeline ở dưới đáy màn hình"""
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
        
        # Viền
        cv2.rectangle(frame, (20, y1), (w - 20, y2), (255, 255, 255), 1)


# =============================================================================
# CLASS CHÍNH: BehaviorAnalyzer
# =============================================================================
# Class này kết hợp tất cả các thành phần trên
# 1. PoseExtractor -> Lấy xương
# 2. SlidingWindowBuffer -> Lấy chuỗi chuyển động
# 3. Model -> Chấm điểm
# 4. Visualizer -> Vẽ kết quả
# =============================================================================

class BehaviorAnalyzer:
    
    # Tiết kiệm RAM
    __slots__ = (
        'device', 'threshold', 'model', 'loaded',
        'pose_extractor', 'buffer', 'visualizer',
        'current_score', 'current_segments', 'current_pose',
        'last_alert_time', 'alert_cooldown'
    )
    
    def __init__(self, model_path, device='cpu', threshold=0.5, window_size=64, stride=16):
        # Chọn thiết bị chạy (CPU/CUDA)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.loaded = False
        
        self.current_score = 0.0
        self.current_segments = None
        self.current_pose = None
        
        self.last_alert_time = 0
        self.alert_cooldown = settings.get('behavior.alert_cooldown', 30)
        
        # Tạo và nạp model
        self.model = DeepMIL(input_dim=68, hidden_dim=256, dropout=0.3)
        self.load_model(model_path)
        
        # Khởi tạo bộ trích xuất xương
        if MEDIAPIPE_AVAILABLE:
            self.pose_extractor = PoseExtractor(model_complexity=1)
        else:
            self.pose_extractor = None
            print("⚠️ Không có MediaPipe - Tắt tính năng phân tích hành vi")
        
        self.buffer = SlidingWindowBuffer(window_size, stride, num_segments=32)
        self.visualizer = BehaviorVisualizer(threshold)
    
    def load_model(self, model_path):
        """Nạp model đã train sẵn"""
        try:
            # Tải checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Xử lý các định dạng checkpoint khác nhau
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
            self.model.eval()  # Chuyển sang chế độ đánh giá (evaluation)
            self.loaded = True
            print(f"✅ Đã tải model hành vi: {model_path}")
            
        except Exception as e:
            print(f"❌ Lỗi tải model hành vi: {e}")
            self.loaded = False
    
    def process_frame(self, frame):
        """
        Xử lý một frame video
        Trả về kết quả phân tích
        """
        if not self.loaded or self.pose_extractor is None:
            return BehaviorResult(0.0, False, None)
        
        # 1. Trích xuất xương
        pose = self.pose_extractor.extract(frame)
        self.current_pose = pose
        
        if not pose.is_valid:
            # Nếu không thấy người, giữ nguyên điểm cũ hoặc trả về 0
            return BehaviorResult(self.current_score, False, pose)
        
        # 2. Thêm vào bộ đệm và lấy features
        features = self.buffer.add(pose.keypoints)
        
        # 3. Nếu đủ dữ liệu, chạy model AI
        if features is not None:
            with torch.no_grad():
                x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                results = self.model(x).squeeze(0).cpu().numpy()
                self.current_segments = results
                # Lấy điểm cao nhất trong chuỗi
                self.current_score = float(self.current_segments.max())
        
        is_anomaly = self.current_score >= self.threshold
        
        return BehaviorResult(
            score=self.current_score,
            is_anomaly=is_anomaly,
            pose=pose,
            segment_scores=self.current_segments,
            timestamp=time.time()
        )
    
    def draw_on_frame(self, frame, result=None):
        """Vẽ kết quả lên màn hình"""
        if result is None:
            result = BehaviorResult(self.current_score, self.current_score >= self.threshold, self.current_pose)
        
        # Vẽ xương
        if result.pose and result.pose.is_valid:
            color = self.visualizer.get_color(result.score)
            self.visualizer.draw_skeleton(
                frame,
                result.pose.keypoints,
                result.pose.confidence,
                color
            )
        
        # Vẽ điểm số
        self.visualizer.draw_score(frame, result.score, result.is_anomaly)
        
        # Vẽ timeline
        if result.segment_scores is not None:
            self.visualizer.draw_timeline(frame, result.segment_scores)
    
    def should_alert(self):
        """Kiểm tra có nên gửi cảnh báo không (dựa trên cooldown)"""
        now = time.time()
        if now - self.last_alert_time >= self.alert_cooldown:
            if self.current_score >= self.threshold:
                self.last_alert_time = now
                return True
        return False
    
    def reset(self):
        """Reset trạng thái"""
        self.buffer.reset()
        self.current_score = 0.0
        self.current_segments = None
        self.current_pose = None
    
    def close(self):
        """Giải phóng tài nguyên"""
        if self.pose_extractor:
            self.pose_extractor.close()
            self.pose_extractor = None
        self.reset()

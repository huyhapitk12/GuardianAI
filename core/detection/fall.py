# core/detection/fall.py
# =============================================================================
# PHÁT HIỆN TÉ NGÃ (FALL DETECTION)
# =============================================================================
# Module này phát hiện té ngã sử dụng model ONNX và MediaPipe Pose
# Logic: Thu thập keypoints theo thời gian -> Tính features -> Dự đoán
# =============================================================================

import numpy as np
from collections import deque

import cv2
import mediapipe as mp
import onnxruntime as ort

from config import settings


# Mapping từ MediaPipe (33 keypoints) sang COCO format (17 keypoints)
MP_TO_COCO = {
    0: 0,    # nose
    2: 1,    # left eye
    5: 2,    # right eye
    7: 3,    # left ear
    8: 4,    # right ear
    11: 5,   # left shoulder
    12: 6,   # right shoulder
    13: 7,   # left elbow
    14: 8,   # right elbow
    15: 9,   # left wrist
    16: 10,  # right wrist
    23: 11,  # left hip
    24: 12,  # right hip
    25: 13,  # left knee
    26: 14,  # right knee
    27: 15,  # left ankle
    28: 16,  # right ankle
}


# =============================================================================
# FEATURE EXTRACTION (Giống logic training)
# =============================================================================

# Chuẩn hóa keypoints về tâm hông (hip center)
# keypoints_T_17_2: (T, 17, 2) - T frames, 17 keypoints, x/y
# return: (T, 17, 2) đã chuẩn hóa
# Tính tâm hông = trung bình left_hip và right_hip
def normalize_keypoints(keypoints_T_17_2):
    hip_center = (keypoints_T_17_2[:, 11] + keypoints_T_17_2[:, 12]) / 2.0  # (T,2)
    normalized = keypoints_T_17_2 - hip_center[:, None, :]  # (T,17,2)
    
    # Scale theo khoảng cách vai
    shoulder_dist = np.linalg.norm(
        keypoints_T_17_2[:, 5] - keypoints_T_17_2[:, 6], axis=1
    )  # (T,)
    scale = float(np.mean(shoulder_dist))
    if scale > 1e-6:
        normalized = normalized / scale
    
    return normalized


# Tính features từ keypoints: position + velocity
# keypoints_T_17_2: (T, 17, 2)
# return: (T, 68) - 34 pos + 34 vel
def compute_features(keypoints_T_17_2):
    T = keypoints_T_17_2.shape[0]
    pos = keypoints_T_17_2.reshape(T, -1)  # (T, 34)
    
    # Velocity = frame[t] - frame[t-1]
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]  # (T, 34)
    
    feats = np.concatenate([pos, vel], axis=1).astype(np.float32)  # (T, 68)
    return feats


# Padding hoặc truncate về số segment cố định
# features_T_D: (T, D)
# num_segments: số segment mong muốn
# return: (num_segments, D)
def to_num_segments(features_T_D, num_segments):
    T = features_T_D.shape[0]
    
    if T > num_segments:
        # Truncate: lấy đều từ đầu đến cuối
        idx = np.linspace(0, T - 1, num_segments, dtype=int)
        return features_T_D[idx]
    
    if T < num_segments:
        # Padding: lặp lại frame cuối
        pad = np.tile(features_T_D[-1:], (num_segments - T, 1))
        return np.concatenate([features_T_D, pad], axis=0)
    
    return features_T_D


# =============================================================================
# ONNX HELPERS
# =============================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Tính mean của top-k elements
def topk_mean_np(logits_T, ratio):
    T = logits_T.shape[0]
    k = max(1, int(T * ratio))
    topk = np.partition(logits_T, -k)[-k:]
    return float(np.mean(topk))


# =============================================================================
# FALL ONNX MODEL
# =============================================================================

# Wrapper cho model ONNX phát hiện té ngã
# Hỗ trợ 2 kiểu output:
# - Kiểu A: Output là fall_prob (B,) hoặc (B,1) -> dùng luôn
# - Kiểu B: Output là segment_logits (B,T) -> cần topk_mean + sigmoid
class FallONNX:
    
    def __init__(self, onnx_path):
        # Sử dụng CUDA nếu có
        providers = ort.get_available_providers()
        use_cuda = "CUDAExecutionProvider" in providers
        
        self.sess = ort.InferenceSession(
            onnx_path,
            providers=(
                ["CUDAExecutionProvider", "CPUExecutionProvider"] 
                if use_cuda else ["CPUExecutionProvider"]
            ),
        )
        
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.topk_ratio = 0.2  # Top-K ratio cho segment logits
    
    # Dự đoán xác suất té ngã
    # x_1_T_D: (1, T, D) - 1 batch, T frames, D features
    # return: float - xác suất té ngã [0, 1]
    def predict_prob(self, x_1_T_D):
        x_1_T_D = np.asarray(x_1_T_D, dtype=np.float32)
        out = self.sess.run([self.out_name], {self.in_name: x_1_T_D})[0]
        out = np.asarray(out)
        
        # Case A: output là prob
        if out.ndim == 2 and out.shape[1] == 1:
            return float(out[0, 0])
        if out.ndim == 1 and out.shape[0] == x_1_T_D.shape[0]:
            return float(out[0])
        
        # Case B: segment logits (B, T)
        if out.ndim == 2 and out.shape[1] == x_1_T_D.shape[1]:
            seg_logits = out[0].astype(np.float32)  # (T,)
            win_logit = topk_mean_np(seg_logits, self.topk_ratio)
            return float(sigmoid(np.array(win_logit, dtype=np.float32)))
        
        raise RuntimeError(f"Không nhận dạng được output shape của ONNX: {out.shape}")


# =============================================================================
# POSE EXTRACTOR
# =============================================================================

# Trích xuất pose từ frame sử dụng MediaPipe
class PoseExtractor:
    
    def __init__(self, confidence_threshold=0.8):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.confidence_threshold = confidence_threshold
    
    # Trích xuất 17 keypoints từ frame
    # frame_bgr: Frame BGR từ OpenCV
    # return: (keypoints (17, 2), confidence) hoặc (None, 0) nếu không detect được
    def extract_17xy(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        
        if not res.pose_landmarks:
            return None, 0.0
        
        lms = res.pose_landmarks.landmark
        kps = np.zeros((17, 2), dtype=np.float32)
        confs = []
        
        for mp_idx, coco_idx in MP_TO_COCO.items():
            lm = lms[mp_idx]
            kps[coco_idx] = [lm.x, lm.y]  # normalized 0..1
            confs.append(lm.visibility)
        
        return kps, float(np.mean(confs))
    
    def close(self):
        try:
            self.pose.close()
        except:
            pass


# =============================================================================
# FALL DETECTOR - CLASS CHÍNH
# =============================================================================

# Phát hiện té ngã cho hệ thống GuardianAI
# Cách dùng:
# 1. Khởi tạo: detector = FallDetector()
# 2. Mỗi frame: detector.update(frame)
# 3. Kiểm tra: is_fall, prob = detector.check_fall()
# 4. Khi xong: detector.close()
class FallDetector:
    
    def __init__(self):
        # Lấy config
        self.enabled = settings.get('fall.enabled', True)
        
        if not self.enabled:
            print("⚠️ Fall detection bị tắt trong config")
            return
        
        # Config
        model_path = settings.get('fall.model_path', 'Data/Model/Fall/fall.onnx')
        self.threshold = settings.get('fall.threshold', 0.50)
        self.n_consecutive = settings.get('fall.n_consecutive', 3)
        self.window_size = settings.get('fall.window_size', 30)
        self.stride = settings.get('fall.stride', 3)
        self.pose_confidence = settings.get('fall.pose_confidence', 0.8)
        self.num_segments = 30  # Giữ nguyên theo model training
        self.miss_threshold = 10  # Số frame mất track để reset
        
        # Khởi tạo model
        try:
            from pathlib import Path
            full_path = Path(settings.base_dir) / model_path
            self.model = FallONNX(str(full_path))
            print(f"✅ Đã tải model Fall Detection: {model_path}")
        except Exception as e:
            print(f"❌ Lỗi tải model Fall Detection: {e}")
            self.model = None
            self.enabled = False
            return
        
        # Khởi tạo pose extractor
        self.pose_extractor = PoseExtractor(self.pose_confidence)
        
        # Buffer lưu keypoints
        self.buffer = deque(maxlen=self.window_size)
        self.last_kps = None
        self.miss_count = 0
        self.frame_count = 0
        
        # Trạng thái detection
        self.hit_run = 0  # Số window liên tiếp vượt ngưỡng
        self.last_prob = None
        self.is_fall_detected = False
    
    # Cập nhật với frame mới
    # frame: Frame BGR từ OpenCV
    # return: None (kết quả lấy qua check_fall())
    def update(self, frame):
        if not self.enabled or self.model is None:
            return
        
        self.frame_count += 1
        
        # Trích xuất pose
        kps, conf = self.pose_extractor.extract_17xy(frame)
        
        if kps is not None and conf > self.pose_confidence:
            self.last_kps = kps
            self.buffer.append(kps)
            self.miss_count = 0
        else:
            self.miss_count += 1
            
            if self.miss_count > self.miss_threshold:
                # Mất track quá lâu -> Reset
                self.buffer.clear()
                self.last_kps = None
                self.hit_run = 0
                self.last_prob = None
                self.is_fall_detected = False
            elif self.last_kps is not None:
                # Mới mất track -> dùng keypoints cũ
                self.buffer.append(self.last_kps)
        
        # Chỉ infer theo stride
        if len(self.buffer) == self.window_size and (self.frame_count % self.stride == 0):
            self._run_inference()
    
    # Chạy inference với buffer hiện tại
    def _run_inference(self):
        try:
            # Chuẩn bị features
            win_kps = np.stack(self.buffer, axis=0)  # (30, 17, 2)
            win_norm = normalize_keypoints(win_kps)  # (30, 17, 2)
            feats = compute_features(win_norm)       # (30, 68)
            feats = to_num_segments(feats, self.num_segments)  # (30, 68)
            
            # Inference
            x = feats[None, :, :]  # (1, 30, 68)
            prob = self.model.predict_prob(x)
            self.last_prob = prob
            
            # Kiểm tra ngưỡng
            if prob >= self.threshold:
                self.hit_run += 1
            else:
                self.hit_run = 0
            
            # Cập nhật trạng thái
            self.is_fall_detected = (self.hit_run >= self.n_consecutive)
            
        except Exception as e:
            print(f"Lỗi fall inference: {e}")
    
    # Kiểm tra trạng thái té ngã hiện tại
    # return: (is_fall: bool, probability: float)
    def check_fall(self):
        if not self.enabled:
            return False, 0.0
        
        return self.is_fall_detected, self.last_prob or 0.0
    
    # Reset trạng thái detection
    def reset(self):
        self.buffer.clear()
        self.last_kps = None
        self.miss_count = 0
        self.hit_run = 0
        self.last_prob = None
        self.is_fall_detected = False
    
    # Giải phóng resources
    def close(self):
        if hasattr(self, 'pose_extractor') and self.pose_extractor:
            self.pose_extractor.close()
            self.pose_extractor = None
        self.model = None

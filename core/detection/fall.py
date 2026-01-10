# Module phát hiện té ngã (RTMPose + ONNX)
import numpy as np
from collections import deque
import cv2
import onnxruntime as ort
from config import settings


# COCO skeleton connections (các cặp điểm nối với nhau)
COCO_SKELETON = [
    (0, 1), (0, 2),           # mũi -> mắt
    (1, 3), (2, 4),           # mắt -> tai
    (5, 6),                   # vai
    (5, 7), (7, 9),           # cánh tay
    (6, 8), (8, 10),          # cánh tay
    (5, 11), (6, 12),         # xương sống
    (11, 12),                 # xương sống
    (11, 13), (13, 15),       # chân
    (12, 14), (14, 16),       # chân
]

# Màu sắc cho skeleton (BGR)
SKELETON_COLOR = (0, 255, 255)    # Vàng cho bones
KEYPOINT_COLOR = (0, 255, 0)      # Xanh lá cho keypoints
FALL_SKELETON_COLOR = (0, 0, 255)  # Đỏ khi fall detected

# Hàm vẽ xương lên frame (debug)
def draw_skeleton(frame, kps_normalized, is_fall=False):
    h, w = frame.shape[:2]
    
    # Chọn màu dựa vào trạng thái fall
    bone_color = FALL_SKELETON_COLOR if is_fall else SKELETON_COLOR
    point_color = FALL_SKELETON_COLOR if is_fall else KEYPOINT_COLOR
    
    # Chuyển tọa độ chuẩn hóa thành tọa độ pixel
    kps_pixel = np.zeros_like(kps_normalized)
    kps_pixel[:, 0] = kps_normalized[:, 0] * w
    kps_pixel[:, 1] = kps_normalized[:, 1] * h
    kps_pixel = kps_pixel.astype(int)
    
    # Vẽ xương
    for (i, j) in COCO_SKELETON:
        pt1 = tuple(kps_pixel[i])
        pt2 = tuple(kps_pixel[j])
        # Chỉ vẽ nếu cả 2 điểm đều valid (không ở góc 0,0)
        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
            cv2.line(frame, pt1, pt2, bone_color, 2, cv2.LINE_AA)
    
    # Vẽ keypoints
    for i, (x, y) in enumerate(kps_pixel):
        if x > 0 and y > 0:
            cv2.circle(frame, (x, y), 4, point_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 4, (0, 0, 0), 1, cv2.LINE_AA)  # viền đen
    
    return frame


# Chuẩn hóa keypoints (vị trí tương đối so với hông)
def normalize_keypoints(keypoints_T_17_2):
    hip_center = (keypoints_T_17_2[:, 11] + keypoints_T_17_2[:, 12]) / 2.0  # Tâm hông = trung bình hông trái và hông phải
    normalized = keypoints_T_17_2 - hip_center[:, None, :]  # Tính khoảng cách từ tâm hông (Đặt trục tại tâm hông)
    
    # Scale theo khoảng cách vai
    shoulder_dist = np.linalg.norm(
        keypoints_T_17_2[:, 5] - keypoints_T_17_2[:, 6], axis=1
    )  # Khoảng cách giữa vai trái và vai phải
    scale = float(np.mean(shoulder_dist))  # Trung bình khoảng cách vai các frame
    if scale > 1e-6: # Tránh chia cho 0
        normalized = normalized / scale
    
    return normalized


# Feature extraction: Location + Velocity
def compute_features(keypoints_T_17_2):
    T = keypoints_T_17_2.shape[0] # Số frame
    pos = keypoints_T_17_2.reshape(T, -1)  # Trải phẳng thành một hàng vs len = 34 * T
    
    vel = np.zeros_like(pos)      # Tạo vector vận tốc với cùng kích thước và giá trị = 0
    vel[1:] = pos[1:] - pos[:-1]  # Tính vector vận tốc
    
    feats = np.concatenate([pos, vel], axis=1).astype(np.float32)  # pos + vel
    return feats


# Resize/Pad vector tính năng
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

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Tính mean của top-k elements
def topk_mean_np(logits_T, ratio):
    T = logits_T.shape[0]
    k = max(1, int(T * ratio))
    topk = np.partition(logits_T, -k)[-k:]
    return float(np.mean(topk))


# Wrapper cho model ONNX
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
    
    # Predict probability
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


# POSE EXTRACTOR
class PoseExtractor:
    
    def __init__(self, confidence_threshold=0.5, mode='balanced', device='cpu', backend='onnxruntime'):
        self.confidence_threshold = confidence_threshold
        
        # Dùng Body wrapper của rtmlib
        from rtmlib import Body
        
        # Mode: 'lightweight' (Small), 'balanced' (Medium), 'performance' (Large)
        self.body = Body(
            to_openpose=False,  # COCO 17 keypoints
            mode=mode,
            backend=backend,
            device=device
        )
        # Lấy pose model để dùng trực tiếp với bbox (bỏ qua YOLOX khi inference)
        self.pose_model = self.body.pose_model
        self.img_shape = None
    
    # Extract keypoints dùng bbox có sẵn (Nhanh)
    def extract_with_bbox(self, frame_bgr, bbox):
        h, w = frame_bgr.shape[:2]
        self.img_shape = (h, w)
        
        # Chuyển bbox sang format rtmlib [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, bbox)
        bboxes = [[x1, y1, x2, y2]]
        
        # Chạy pose model trực tiếp với bbox (không cần YOLOX)
        keypoints, scores = self.pose_model(frame_bgr, bboxes=bboxes)
        
        if keypoints is None or len(keypoints) == 0:
            return None, 0.0
        
        kps = keypoints[0]  # (17, 2)
        confs = scores[0]   # (17,)
        
        # Normalize về 0..1
        kps_normalized = np.zeros((17, 2), dtype=np.float32)
        kps_normalized[:, 0] = kps[:, 0] / w
        kps_normalized[:, 1] = kps[:, 1] / h
        
        avg_conf = float(np.mean(confs))
        return kps_normalized, avg_conf
    
    # Extract keypoints full frame (Chậm hơn, tự detect)
    def extract_17xy(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        self.img_shape = (h, w)
        
        # RTMPose inference (có chạy YOLOX)
        keypoints, scores = self.body(frame_bgr)
        
        if keypoints is None or len(keypoints) == 0:
            return None, 0.0
        
        kps = keypoints[0]  # (17, 2)
        confs = scores[0]   # (17,)
        
        # Normalize về 0..1
        kps_normalized = np.zeros((17, 2), dtype=np.float32)
        kps_normalized[:, 0] = kps[:, 0] / w
        kps_normalized[:, 1] = kps[:, 1] / h
        
        avg_conf = float(np.mean(confs))
        return kps_normalized, avg_conf
    
    def close(self):
        pass


# Class FallDetector chính
class FallDetector:
    
    def __init__(self):
        # Lấy config
        self.enabled = settings.get('fall.enabled', True)
        
        if not self.enabled:
            print("[WARN] Fall detection bị tắt trong config")
            return
        
        # Config
        model_path = settings.get('fall.model_path', 'Data/Model/Fall/fall.onnx')
        self.threshold = settings.get('fall.threshold', 0.80)
        self.n_consecutive = settings.get('fall.n_consecutive', 3)
        self.window_size = settings.get('fall.window_size', 30)
        self.stride = settings.get('fall.stride', 3)
        self.pose_confidence = settings.get('fall.pose_confidence', 0.5)
        self.num_segments = 30  # Giữ nguyên theo model training
        self.miss_threshold = 10  # Số frame mất track để reset
        
        # Đọc config model
        model_mode = settings.get('models.mode', 'Medium')
        device = settings.get('models.device', 'cpu')
        pose_backend = settings.get('models.pose_backend', 'onnxruntime')
        
        # Mapping Small/Medium : lightweight/balanced
        mode_map = {
            'Small': 'lightweight',
            'Medium': 'balanced'
        }
        self.pose_mode = mode_map.get(model_mode, 'balanced')

        # Khởi tạo model
        from pathlib import Path
        full_path = Path(settings.base_dir) / model_path
        self.model = FallONNX(str(full_path))
        print(f"[OK] Đã tải model Fall Detection: {model_path}")
        
        # Khởi tạo pose extractor (RTMPose)
        # rtmlib cần 'cuda' thay vì 'gpu'
        rtml_device = 'cuda' if device.lower() == 'gpu' else 'cpu'
        
        # Khởi tạo pose extractor (RTMPose)
        self.pose_extractor = PoseExtractor(
            confidence_threshold=self.pose_confidence,
            mode=self.pose_mode,
            device=rtml_device,
            backend=pose_backend
        )
        print(f"[INFO] RTMPose mode: {self.pose_mode} ({model_mode}) | Device: {rtml_device} | Backend: {pose_backend}")
        
        # Buffer lưu keypoints
        self.buffer = deque(maxlen=self.window_size)
        self.last_kps = None
        self.miss_count = 0
        self.frame_count = 0
        
        # Trạng thái detection
        self.hit_run = 0  # Số window liên tiếp vượt ngưỡng
        self.last_prob = None
        self.is_fall_detected = False
    
    # Cập nhật frame mới
    def update(self, frame, bbox=None):
        if not self.enabled or self.model is None:
            return
        
        self.frame_count += 1
        
        # Trích xuất pose
        if bbox is not None:
            # Có bbox từ person detector : dùng trực tiếp, bỏ qua YOLOX (nhanh hơn)
            kps, conf = self.pose_extractor.extract_with_bbox(frame, bbox)
        else:
            # Không có bbox : phải tự detect bằng YOLOX (chậm hơn)
            kps, conf = self.pose_extractor.extract_17xy(frame)
        
        if kps is not None and conf > self.pose_confidence:
            self.last_kps = kps
            self.buffer.append(kps)
            self.miss_count = 0
        else:
            self.miss_count += 1
            
            if self.miss_count > self.miss_threshold:
                # Mất track quá lâu : Reset
                self.buffer.clear()
                self.last_kps = None
                self.hit_run = 0
                self.last_prob = None
                self.is_fall_detected = False
            elif self.last_kps is not None:
                # Mới mất track : dùng keypoints cũ
                self.buffer.append(self.last_kps)
        
        # Chỉ infer theo stride
        if len(self.buffer) == self.window_size and (self.frame_count % self.stride == 0):
            self._run_inference()
    
    # Chạy inference
    def _run_inference(self):
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
    
    # Check kết quả (True/False, prob)
    def check_fall(self):
        if not self.enabled:
            return False, 0.0
        
        return self.is_fall_detected, self.last_prob or 0.0
    
    # Vẽ overlay lên frame
    def draw_skeleton_overlay(self, frame):
        if not self.enabled or self.last_kps is None:
            return frame
        
        return draw_skeleton(frame, self.last_kps, is_fall=self.is_fall_detected)
    
    # Lấy keypoints hiện tại
    def get_keypoints(self):
        return self.last_kps
    
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

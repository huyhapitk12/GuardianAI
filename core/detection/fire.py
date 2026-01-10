# Module phát hiện cháy (YOLO + Bộ lọc màu)

import cv2
import numpy as np
from collections import deque

# Import thư viện YOLO (mạng nhận diện vật thể)
from ultralytics import YOLO

from config import settings


# Class cấu hình ngưỡng phát hiện cháy
class FireConfig:
    
    def __init__(self):
        # Load config from settings if available
        ff = settings.camera.fire_filter
        self.min_roi_size = 8
        self.flicker_history = 15
        self.flicker_min_frames = 5
            
        # RGB
        self.rgb_hue_max = getattr(ff.rgb, 'hue_orange_max', 35)
        self.rgb_saturation_min = getattr(ff.rgb, 'saturation_min', 80)
        self.rgb_brightness_min = getattr(ff.rgb, 'brightness_min', 100)
        self.rgb_white_ratio_max = getattr(ff.rgb, 'rgb_white_threshold', 0.88)
        self.rgb_entropy_min = getattr(ff.rgb, 'texture_entropy_min', 2)
        self.rgb_flicker_min = getattr(ff.rgb, 'flickering_std_min', 5.0)
            
        # IR
        self.ir_brightness_min = getattr(ff.infrared, 'brightness_mean_min', 120)
        self.ir_brightness_std_min = getattr(ff.infrared, 'brightness_std_min', 25)
        self.ir_hot_ratio_min = getattr(ff.infrared, 'bright_core_ratio_min', 0.08)
        self.ir_hot_ratio_max = getattr(ff.infrared, 'hot_spot_ratio_max', 0.70)
        self.ir_irregularity_min = getattr(ff.infrared, 'edge_irregularity_min', 0.3)
        self.ir_flicker_min = getattr(ff.infrared, 'flickering_std_min', 3.0)
        
    


# Bộ lọc giảm False Positive (đèn, áo cam, v.v.)
class FireFilter:
    
    # Dùng config mặc định nếu không truyền vào
    def __init__(self, config=None, debug=True):
        self.config = config or FireConfig()
        
        # Lưu lịch sử độ sáng để phân tích nhấp nháy
        # deque: giống list nhưng tự động xóa phần tử cũ khi đầy
        self.history = {}
        
        # Chế độ debug: in ra lý do loại bỏ
        self.debug = True
    
    # Kiểm tra vùng phát hiện
    def validate(self, frame, bbox, is_ir=False):
        # Cắt vùng cần kiểm tra
        roi = self.get_roi(frame, bbox)
        if roi is None:
            return False
        
        # Gọi hàm kiểm tra phù hợp với loại camera
        if is_ir:
            return self.validate_ir(roi, bbox)
        else:
            return self.validate_rgb(roi, bbox)
    
    # Cắt vùng khung vực cần check từ frame
    def get_roi(self, frame, bbox):
        # Làm tròn tọa độ
        x1, y1, x2, y2 = map(int, bbox)
        min_size = self.config.min_roi_size
        
        # Kiểm tra kích thước tối thiểu
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            return None
        
        # Đảm bảo tọa độ nằm trong frame
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Cắt và trả về
        roi = frame[y1:y2, x1:x2]
        return roi if roi.size > 0 else None
    
    # Kiểm tra với camera màu
    # Lửa thật có đặc điểm:
    # - Màu cam/đỏ/vàng (Hue thấp trong HSV)
    # - Độ bão hòa cao
    # - Kết cấu phức tạp (không đồng đều như đèn)
    def validate_rgb(self, roi, bbox):
        cfg = self.config
        
        # 1. Chuyển sang không gian màu HSV
        # HSV: Hue (màu sắc), Saturation (độ đậm), Value (độ sáng)
        # OpenCV HSV: H=[0,179], S=[0,255], V=[0,255]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)  # Tách 3 kênh
        
        # 2. Tạo mask pixel ứng viên lửa (Hue + Saturation + Brightness)
        # Hue lửa: đỏ/cam (0-30) hoặc đỏ wrap-around (165-179)
        fire_hue_mask = ((h >= 0) & (h <= 30)) | ((h >= 165) & (h <= 179))
        
        # Saturation: lửa có màu đậm (S >= 50)
        fire_sat_mask = s >= 50
        
        # Brightness: lửa sáng (V >= 80)
        fire_val_mask = v >= 80
        
        # Kết hợp: pixel phải thỏa cả 3 điều kiện
        fire_candidate_mask = fire_hue_mask & fire_sat_mask & fire_val_mask
        fire_ratio = np.mean(fire_candidate_mask)
        
        # 3. Tạo mask vùng trắng/lõi sáng (V cao, S thấp)
        white_core_mask = (v > 220) & (s < 40)
        white_ratio = np.mean(white_core_mask)
        
        # 4. Logic loại reflection:
        # Chỉ loại nếu vùng trắng NHIỀU nhưng vùng màu lửa ÍT
        # (tức là sáng trắng nhưng không có viền cam/đỏ)
        if white_ratio > 0.3 and fire_ratio < 0.08:
            return self.fail("reflection")  # Phản chiếu, không có viền lửa
        
        # 5. Check tỉ lệ pixel màu lửa (chỉ dựa trên Hue)
        hue_only_ratio = np.mean(fire_hue_mask)
        if hue_only_ratio < 0.05:
            return self.fail(f"hue ({hue_only_ratio:.2f}<0.05)")  # Không đủ pixel màu lửa
        
        # 6. Kiểm tra độ bão hòa bằng tỉ lệ pixel
        # Thay vì dùng mean(s), đếm tỉ lệ pixel có S cao
        sat_ratio = np.mean(s >= 50)
        if sat_ratio < 0.10:
            return self.fail(f"saturation ({sat_ratio:.2f}<0.10)")  # Màu quá nhạt
        
        # 7. Check độ sáng bằng percentile 95 (tránh bị 1 pixel sáng cứu cả ROI)
        brightness_p95 = np.percentile(v, 95)
        if brightness_p95 < 100:
            return self.fail(f"brightness (p95={brightness_p95:.0f}<100)")  # Quá tối
        
        # 8. Check texture (Entropy) chỉ trên vùng ứng viên lửa
        # Lửa có kết cấu phức tạp, LED thì đồng đều
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Tạo mask uint8 cho calcHist
        fire_mask_uint8 = (fire_hue_mask.astype(np.uint8)) * 255
        
        # Tính histogram chỉ trên vùng màu lửa
        hist = cv2.calcHist([gray], [0], fire_mask_uint8, [256], [0, 256]).flatten()
        hist = hist[hist > 0]
        
        if hist.sum() > 0:
            hist = hist / hist.sum()  # Chuẩn hóa
            entropy = -np.sum(hist * np.log2(hist))
        else:
            entropy = 0
        
        if entropy < 1.5:
            return self.fail(f"texture ({entropy:.2f}<1.5)")  # Kết cấu quá đơn giản
        
        # Qua tất cả bước kiểm tra : Là lửa thật!
        return True
    
    # Validate cho camera IR (hồng ngoại)
    def validate_ir(self, roi, bbox):
        cfg = self.config
        
        # Chuyển sang grayscale (handle cả ảnh 1 kênh và 3 kênh)
        if roi.ndim == 2:
            gray = roi.astype(np.float32)  # Đã là grayscale
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Kiểm tra độ sáng
        if np.mean(gray) < cfg.ir_brightness_min and np.max(gray) < 180:
            return self.fail("brightness")
        
        # Kiểm tra độ biến thiên
        # Lửa không đồng đều : độ lệch chuẩn cao
        if np.std(gray) < cfg.ir_brightness_std_min:
            return self.fail("variation")
        
        # Kiểm tra tỉ lệ điểm nóng (dùng ngưỡng tương đối để ổn định hơn với auto-gain)
        # Điểm nóng: pixel sáng hơn mean + 2*std
        mean_gray = np.mean(gray)
        std_gray = np.std(gray)
        hot_threshold = max(200, mean_gray + 2 * std_gray)  # Ít nhất 200, hoặc mean+2std
        hot_ratio = np.sum(gray > hot_threshold) / gray.size
        
        if not (cfg.ir_hot_ratio_min <= hot_ratio <= cfg.ir_hot_ratio_max):
            return self.fail("hot_core")
        
        # Kiểm tra hình dạng
        # Lửa có hình dạng bất quy tắc (không tròn như đèn)
        _, thresh = cv2.threshold(gray.astype(np.uint8), 0, 255, 
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)  # Contour lớn nhất
            area = cv2.contourArea(c)
            
            if area > 10:
                peri = cv2.arcLength(c, True)
                # Tính độ tròn (circularity): 1.0 = tròn hoàn hảo
                circ = 4 * np.pi * area / (peri ** 2) if peri > 0 else 0
                
                # Độ bất quy tắc = 1 - độ tròn
                if (1.0 - circ) < cfg.ir_irregularity_min:
                    return self.fail("shape")  # Quá tròn : đèn
        
        # Kiểm tra nhấp nháy
        if not self.check_flicker(gray.astype(np.uint8), bbox, cfg.ir_flicker_min):
            return self.fail("flicker")
        
        return True
    
    # Check nhấp nháy (flicker) theo thời gian
    def check_flicker(self, gray, bbox, threshold):
        # Tạo key dựa trên vị trí (chia ô để gộp các vị trí gần nhau)
        key = f"{bbox[0]//20}_{bbox[1]//20}"
        
        # Tạo history nếu chưa có
        if key not in self.history:
            self.history[key] = deque(maxlen=self.config.flicker_history)
        
        # Thêm độ sáng trung bình vào history
        hist = self.history[key]
        hist.append(float(np.mean(gray)))
        
        # Cần đủ frame để phân tích
        if len(hist) < self.config.flicker_min_frames:
            return True  # Chấp nhận tạm
        
        # Tính độ lệch chuẩn của độ sáng qua các frame
        # Lửa nhấp nháy : độ lệch chuẩn cao
        return np.std(list(hist)) > threshold
    
    # In lý do thất bại (nếu đang debug)
    def fail(self, reason):
        if self.debug:
            print(f"❌ Lọc phát hiện cháy - Loại: {reason}")
        return False
    
    # Dọn dẹp history cũ để tiết kiệm bộ nhớ
    def cleanup(self):
        if len(self.history) > 50:
            keys = list(self.history.keys())[:-30]
            for k in keys:
                del self.history[k]


# Class FireDetector: Dùng YOLO detect lửa/khói
class FireDetector:
    
    def __init__(self, debug=False):
        self.model = None                  # Model YOLO
        self.fire_filter = FireFilter(debug=debug)   # Bộ lọc
        self.frame_count = 0               # Đếm frame
        
        # Xử lý mỗi N frame để giảm tải CPU/GPU
        self.skip_interval = settings.get('camera.process_every_n_frames', 3)
    
    # Init model YOLO
    def initialize(self):
        # Kiểm tra đã cài YOLO chưa
        if not YOLO:
            print("⚠️ Thư viện ultralytics chưa được cài đặt!")
            return False
        
        # Lấy cấu hình (format tự động dựa trên device)
        yolo_size = settings.get('models.mode', 'Medium').lower()
        yolo_format = settings.get_optimal_yolo_format()
        
        # Đảm bảo model tồn tại (tự export nếu cần)
        model_path = settings.ensure_yolo_model('fire', yolo_size)
        if not model_path:
            print("❌ Không thể tải model Fire!")
            return False
        
        # Tải model
        print(f"🔥 Loading Fire model: {model_path.name} (format: {yolo_format})")
        self.model = YOLO(str(model_path), task='detect', verbose=False)
        print(f"✅ Model phát hiện cháy đã sẵn sàng!")
        
        # Chạy thử với ảnh giả để "khởi động" model (OpenVINO cần)
        if yolo_format == 'openvino':
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_frame, verbose=False)
        
        return True
    
    # Detect lửa/khói
    # Return: list các vùng phát hiện
    def detect(self, frame, skip=True):
        # Kiểm tra model đã tải chưa
        if not self.model:
            return []
        
        # Bỏ qua một số frame để giảm tải
        if skip:
            self.frame_count += 1
            if self.frame_count % self.skip_interval != 0:
                return []
        
        # Lấy format và device tự động
        yolo_format = settings.get_optimal_yolo_format()
        device = settings.get('models.device', 'cpu')
        
        # Chạy model YOLO
        if yolo_format == 'openvino':
            results = self.model(frame, verbose=False)
        else:
            results = self.model(frame, verbose=False, device=device)
        
        detections = []  # Danh sách kết quả
        
        # Xử lý kết quả từ YOLO
        if results and hasattr(results[0], 'boxes'):
            h, w = frame.shape[:2]
            total_area = w * h
            
            for box in results[0].boxes:
                # Lấy độ tin cậy (0.0 - 1.0)
                conf = float(box.conf[0])
                
                # Lấy tên class (fire, flame, smoke)
                cls = results[0].names.get(int(box.cls[0]), '').lower()
                
                # Chỉ quan tâm fire, flame, smoke
                if cls not in ('fire', 'flame', 'smoke'):
                    continue
                
                # Lấy ngưỡng tin cậy từ config
                if cls == 'smoke':
                    threshold = settings.get('detection.smoke_confidence_threshold', 0.7)
                else:
                    threshold = settings.get('detection.fire_confidence_threshold', 0.6)
                
                # Bỏ qua nếu độ tin cậy thấp
                if conf < threshold:
                    continue
                
                # Lấy tọa độ bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Tính diện tích tương đối
                area = (x2 - x1) * (y2 - y1) / total_area
                
                # Thêm vào kết quả
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class': cls,
                    'conf': conf,
                    'area': area
                })
        
        return detections
    
    # Kiểm tra vùng phát hiện có phải lửa thật không
    def validate(self, frame, bbox, is_ir=False):
        return self.fire_filter.validate(frame, bbox, is_ir)

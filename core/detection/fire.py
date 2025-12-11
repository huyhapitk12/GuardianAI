# core/detection/fire.py
# =============================================================================
# MODULE PHÁT HIỆN CHÁY - FIRE DETECTION
# =============================================================================
# Module này phát hiện lửa và khói trong video từ camera
# Sử dụng 2 phương pháp:
# 1. YOLO: Mạng neural network để nhận diện vật thể
# 2. Bộ lọc màu: Phân tích màu sắc đặc trưng của lửa
# =============================================================================

import cv2
import numpy as np
from collections import deque

# Import thư viện YOLO (mạng nhận diện vật thể)
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("⚠️ Không tìm thấy thư viện ultralytics/YOLO!")

from config import settings


# =============================================================================
# CLASS CẤU HÌNH PHÁT HIỆN CHÁY
# =============================================================================
# Class này chứa các ngưỡng (threshold) để phát hiện lửa
# Ngưỡng = giá trị dùng để so sánh, quyết định "có" hay "không"
# =============================================================================
class FireConfig:
    
    def __init__(self):
        # ----- Cấu hình chung -----
        self.min_roi_size = 8           # Kích thước tối thiểu vùng cần kiểm tra (pixel)
        self.flicker_history = 15       # Số frame lưu lại để phân tích nhấp nháy
        self.flicker_min_frames = 5     # Số frame tối thiểu để phân tích
        
        # ----- Ngưỡng cho camera MÀU (RGB) -----
        # Lửa có đặc điểm: màu cam/đỏ/vàng, sáng, nhấp nháy
        self.rgb_hue_max = 35               # Màu sắc tối đa (Hue trong HSV, 0-180)
        self.rgb_saturation_min = 80        # Độ bão hòa màu tối thiểu
        self.rgb_brightness_min = 100       # Độ sáng tối thiểu
        self.rgb_white_ratio_max = 0.88     # Tỉ lệ pixel trắng tối đa (để loại phản chiếu)
        self.rgb_entropy_min = 4.0          # Entropy tối thiểu (độ phức tạp kết cấu)
        self.rgb_flicker_min = 5.0          # Độ nhấp nháy tối thiểu
        
        # ----- Ngưỡng cho camera HỒNG NGOẠI (IR) -----
        # Camera IR không có màu, chỉ có độ sáng
        self.ir_brightness_min = 120        # Độ sáng tối thiểu
        self.ir_brightness_std_min = 25     # Độ lệch chuẩn sáng (lửa không đều)
        self.ir_hot_ratio_min = 0.08        # Tỉ lệ điểm nóng tối thiểu
        self.ir_hot_ratio_max = 0.70        # Tỉ lệ điểm nóng tối đa
        self.ir_irregularity_min = 0.3      # Độ bất quy tắc tối thiểu (lửa không tròn đều)
        self.ir_flicker_min = 3.0           # Độ nhấp nháy tối thiểu


# =============================================================================
# CLASS BỘ LỌC PHÁT HIỆN CHÁY
# =============================================================================
# Class này lọc bớt các phát hiện sai (false positive)
# Ví dụ: đèn đỏ, áo cam, TV có hình lửa -> không phải cháy thật
# =============================================================================
class FireFilter:
    
    def __init__(self, config=None, debug=False):
        # Dùng config mặc định nếu không truyền vào
        self.config = config or FireConfig()
        
        # Lưu lịch sử độ sáng để phân tích nhấp nháy
        # deque: giống list nhưng tự động xóa phần tử cũ khi đầy
        self.history = {}
        
        # Chế độ debug: in ra lý do loại bỏ
        self.debug = debug
    
    def validate(self, frame, bbox, is_ir=False):
        """
        Kiểm tra xem vùng được phát hiện có phải lửa thật không
        
        frame: Hình ảnh gốc
        bbox: Tọa độ vùng nghi ngờ (x1, y1, x2, y2)
        is_ir: Camera hồng ngoại hay không
        
        Trả về: True nếu là lửa thật, False nếu không
        """
        # Cắt vùng cần kiểm tra
        roi = self.get_roi(frame, bbox)
        if roi is None:
            return False
        
        # Gọi hàm kiểm tra phù hợp với loại camera
        if is_ir:
            return self.validate_ir(roi, bbox)
        else:
            return self.validate_rgb(roi, bbox)
    
    def get_roi(self, frame, bbox):
        """Cắt vùng ROI (Region of Interest) từ frame"""
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
    
    def validate_rgb(self, roi, bbox):
        """
        Kiểm tra với camera màu
        Lửa thật có đặc điểm:
        - Màu cam/đỏ/vàng (Hue thấp trong HSV)
        - Độ bão hòa cao
        - Kết cấu phức tạp (không đồng đều như đèn LED)
        """
        cfg = self.config
        
        # ----- Bước 1: Chuyển sang không gian màu HSV -----
        # HSV: Hue (màu sắc), Saturation (độ đậm), Value (độ sáng)
        # Dễ phân tích màu hơn RGB
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)  # Tách 3 kênh
        
        # ----- Bước 2: Loại bỏ phản chiếu (reflection) -----
        # Phản chiếu: sáng + ít màu = (v > 220) và (s < 40)
        reflection_mask = (v > 220) & (s < 40)
        reflection_ratio = np.mean(reflection_mask)
        
        if reflection_ratio > 0.3:
            return self.fail("reflection")  # Quá nhiều phản chiếu -> không phải lửa
        
        # ----- Bước 3: Kiểm tra màu sắc -----
        # Lửa có màu cam/đỏ: Hue từ 0-30 hoặc 165-180
        fire_hue_mask = ((h >= 0) & (h <= 30)) | ((h >= 165) & (h <= 180))
        valid_hue_ratio = np.mean(fire_hue_mask)
        
        if valid_hue_ratio < 0.4:
            return self.fail("hue")  # Không đủ pixel màu lửa
        
        # ----- Bước 4: Kiểm tra độ bão hòa -----
        if np.mean(s) < 50:
            return self.fail("saturation")  # Màu quá nhạt
        
        # ----- Bước 5: Kiểm tra độ sáng -----
        if np.max(v) < 120:
            return self.fail("too_dark")  # Quá tối
        
        # ----- Bước 6: Kiểm tra kết cấu (texture) -----
        # Dùng Entropy để đo độ phức tạp
        # Lửa có kết cấu phức tạp, đèn LED đồng đều
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Tính histogram (phân bố độ sáng)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist[hist > 0] / hist.sum()  # Chuẩn hóa
        
        # Tính entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        if entropy < 3.5:
            return self.fail("texture")  # Kết cấu quá đơn giản
        
        # Qua tất cả bước kiểm tra -> Là lửa thật!
        return True
    
    def validate_ir(self, roi, bbox):
        """
        Kiểm tra với camera hồng ngoại
        Camera IR chỉ có độ sáng, không có màu
        Lửa trong IR: sáng, không đều, nhấp nháy
        """
        cfg = self.config
        
        # Chuyển sang grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # ----- Kiểm tra độ sáng -----
        if np.mean(gray) < cfg.ir_brightness_min and np.max(gray) < 180:
            return self.fail("brightness")
        
        # ----- Kiểm tra độ biến thiên -----
        # Lửa không đồng đều -> độ lệch chuẩn cao
        if np.std(gray) < cfg.ir_brightness_std_min:
            return self.fail("variation")
        
        # ----- Kiểm tra tỉ lệ điểm nóng -----
        # Điểm nóng: pixel có giá trị > 200
        hot_ratio = np.sum(gray > 200) / gray.size
        
        if not (cfg.ir_hot_ratio_min <= hot_ratio <= cfg.ir_hot_ratio_max):
            return self.fail("hot_core")
        
        # ----- Kiểm tra hình dạng -----
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
                    return self.fail("shape")  # Quá tròn -> đèn
        
        # ----- Kiểm tra nhấp nháy -----
        if not self.check_flicker(gray.astype(np.uint8), bbox, cfg.ir_flicker_min):
            return self.fail("flicker")
        
        return True
    
    def check_flicker(self, gray, bbox, threshold):
        """
        Kiểm tra độ nhấp nháy theo thời gian
        Lửa thật nhấp nháy, đèn LED không
        """
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
        # Lửa nhấp nháy -> độ lệch chuẩn cao
        return np.std(list(hist)) > threshold
    
    def fail(self, reason):
        """In lý do thất bại (nếu đang debug)"""
        if self.debug:
            print(f"❌ Lọc phát hiện cháy - Loại: {reason}")
        return False
    
    def cleanup(self):
        """Dọn dẹp history cũ để tiết kiệm bộ nhớ"""
        if len(self.history) > 50:
            keys = list(self.history.keys())[:-30]
            for k in keys:
                del self.history[k]


# =============================================================================
# CLASS CHÍNH: FireDetector
# =============================================================================
# Class này sử dụng YOLO để phát hiện lửa/khói
# YOLO = You Only Look Once - Mạng neural nhận diện vật thể nhanh
# =============================================================================
class FireDetector:
    
    def __init__(self, debug=False):
        self.model = None                  # Model YOLO
        self.fire_filter = FireFilter(debug=debug)   # Bộ lọc
        self.frame_count = 0               # Đếm frame
        
        # Xử lý mỗi N frame để giảm tải CPU/GPU
        self.skip_interval = settings.get('camera.process_every_n_frames', 3)
    
    def initialize(self):
        """
        Khởi tạo model phát hiện cháy
        Trả về: True nếu thành công, False nếu thất bại
        """
        # Kiểm tra đã cài YOLO chưa
        if not YOLO:
            print("⚠️ Thư viện ultralytics chưa được cài đặt!")
            return False
        
        try:
            # Lấy cấu hình từ settings
            yolo_size = settings.get('models.yolo_size', 'medium').lower()
            yolo_format = settings.get('models.yolo_format', 'openvino')
            
            # Lấy đường dẫn model
            model_path = settings.get_yolo_model_path('fire', yolo_size, yolo_format)
            
            # Kiểm tra file model tồn tại
            if not model_path.exists():
                print(f"⚠️ Không tìm thấy model phát hiện cháy: {model_path}")
                return False
            
            # Tải model
            print(f"🔥 Đang tải model phát hiện cháy: {model_path}")
            self.model = YOLO(str(model_path), task='detect', verbose=False)
            print(f"✅ Model phát hiện cháy đã sẵn sàng!")
            
            # Chạy thử với ảnh giả để "khởi động" model (OpenVINO cần)
            if yolo_format == 'openvino':
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                self.model(dummy_frame, verbose=False)
            
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khởi tạo phát hiện cháy: {e}")
            return False
    
    def detect(self, frame, skip=True):
        """
        Phát hiện lửa/khói trong frame
        
        frame: Hình ảnh cần kiểm tra
        skip: Có bỏ qua một số frame để giảm tải không
        
        Trả về: Danh sách các vùng phát hiện được
        """
        # Kiểm tra model đã tải chưa
        if not self.model:
            return []
        
        # Bỏ qua một số frame để giảm tải
        if skip:
            self.frame_count += 1
            if self.frame_count % self.skip_interval != 0:
                return []
        
        yolo_format = settings.get('models.yolo_format', 'openvino')
        
        try:
            # Chạy model YOLO
            if yolo_format == 'openvino':
                results = self.model(frame, verbose=False)
            else:
                results = self.model(frame, verbose=False, device='cpu')
            
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
                        threshold = settings.get('detection.smoke_confidence', 0.7)
                    else:
                        threshold = settings.get('detection.fire_confidence', 0.6)
                    
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
            
        except Exception as e:
            print(f"⚠️ Lỗi phát hiện cháy: {e}")
            return []
    
    def validate(self, frame, bbox, is_ir=False):
        """Kiểm tra vùng phát hiện có phải lửa thật không"""
        return self.fire_filter.validate(frame, bbox, is_ir)

"""Motion detection để filter trước khi chạy YOLO"""

import cv2
import numpy as np
from collections import deque


class MotionDetector:
    """Phát hiện chuyển động đơn giản, nhanh"""
    
    __slots__ = ('_prev_gray', '_motion_threshold', '_min_area', 
                 '_motion_history', '_active', '_motion_boxes')
    
    def __init__(self, motion_threshold: float = 25.0, min_area: int = 500):
        """
        Args:
            motion_threshold: Ngưỡng độ sáng thay đổi (0-255)
            min_area: Diện tích tối thiểu vùng chuyển động (pixels)
        """
        self._prev_gray = None
        self._motion_threshold = motion_threshold
        self._min_area = min_area
        self._motion_history = deque(maxlen=5)  # Lưu 5 frame gần nhất
        self._active = False
        self._motion_boxes = []
    
    def detect(self, frame: np.ndarray) -> bool:
        """
        Kiểm tra có chuyển động không và lưu boxes
        """
        self._motion_boxes = [] # Reset boxes
        
        # Chuyển sang grayscale và giảm kích thước để nhanh hơn
        # Small size logic must match config used
        h, w = frame.shape[:2]
        small_h, small_w = 180, 320
        scale_x = w / small_w
        scale_y = h / small_h
        
        small = cv2.resize(frame, (small_w, small_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0) # Tăng kernel để giảm nhiễu
        
        # Frame đầu tiên
        if self._prev_gray is None:
            self._prev_gray = gray
            return True
        
        # Tính frame difference
        diff = cv2.absdiff(self._prev_gray, gray)
        _, thresh = cv2.threshold(diff, self._motion_threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        has_motion = False
        for c in contours:
            if cv2.contourArea(c) < self._min_area: # min_area này tính trên ảnh nhỏ (320x180)
                continue
            
            has_motion = True
            (x, y, bw, bh) = cv2.boundingRect(c)
            
            # Scale back to original size
            orig_box = (
                int(x * scale_x), 
                int(y * scale_y), 
                int((x + bw) * scale_x), 
                int((y + bh) * scale_y)
            )
            self._motion_boxes.append(orig_box)

        # Cập nhật prev frame
        self._prev_gray = gray
        
        # Lưu lịch sử
        self._motion_history.append(has_motion)
        
        # Trả về True nếu có motion trong 5 frame gần nhất
        recent_motion = sum(self._motion_history) >= 2
        
        if recent_motion:
            self._active = True
        elif len(self._motion_history) >= 5 and sum(self._motion_history) == 0:
            self._active = False
        
        return self._active

    @property
    def motion_boxes(self):
        return self._motion_boxes
    
    def reset(self):
        """Reset detector"""
        self._prev_gray = None
        self._motion_history.clear()
        self._active = False

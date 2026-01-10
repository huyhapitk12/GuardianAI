# Phát hiện chuyển động đơn giản
import cv2
import numpy as np
from collections import deque


# Tìm chuyển động trong khung hình
class MotionDetector:
    
    # init
    def __init__(self, motion_threshold=25.0, min_area=500):
        self._prev_gray = None
        self.nguong = motion_threshold
        self.dien_tich_min = min_area
        self.lich_su = deque(maxlen=5) # Lưu 5 frame gần nhất
        self._active = False
        self.motion_boxes = []
        print("motion detector init")
        # print(self.nguong)

    def detect(self, frame):
        # print("detecting...")
        self.motion_boxes = [] 
        
        # resize
        h, w = frame.shape[:2]
        small_h, small_w = 180, 320
        scale_x = w / small_w
        scale_y = h / small_h
        
        # small = cv2.resize(frame, (small_w, small_h))
        small = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0) # lọc nhiễu
        
        
        # print(gray.shape)

        if self._prev_gray is None:
            self._prev_gray = gray
            return True # Lần đầu luôn trả về true
        
        # Trừ ảnh
        diff = cv2.absdiff(self._prev_gray, gray)
        _, thresh = cv2.threshold(diff, self.nguong, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Tìm contour
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        co_chuyen_dong = False
        for c in contours:
            if cv2.contourArea(c) < self.dien_tich_min: 
                continue
            
            co_chuyen_dong = True
            # print("thay chuyen dong roi")
            (x, y, bw, bh) = cv2.boundingRect(c)
            
            # scale lại về kích thước cũ
            orig_box = (
                int(x * scale_x), 
                int(y * scale_y), 
                int((x + bw) * scale_x), 
                int((y + bh) * scale_y)
            )
            self.motion_boxes.append(orig_box)


        self._prev_gray = gray
        self.lich_su.append(co_chuyen_dong)
        
        # Check xem 5 frame gần nhất
        recent = sum(self.lich_su) >= 2
        
        if recent:
            self._active = True
        elif len(self.lich_su) >= 5 and sum(self.lich_su) == 0:
            self._active = False
        
        # testing
        # if self._active:
        #     print("active!")
            
        return self._active

    def reset(self):
        print("reset motion")
        self._prev_gray = None
        self.lich_su.clear()
        self._active = False

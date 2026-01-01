# Fire Tracking Logic
import time
import numpy as np
from collections import deque
from config import settings


class TrackedFireObject:
    # Thông tin đám cháy đang theo dõi
    def __init__(self, id, bbox, area, first_seen, last_seen):
        self.id = id
        self.bbox = bbox
        self.area = area
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.age = 0  # frames
        self.stability_score = 0.0
        self.matched_count = 0
    
    def update(self, bbox, area, now):
        # Cập nhật vị trí & tính ổn định
        self.bbox = bbox
        self.area = area
        self.last_seen = now
        self.age += 1
        self.matched_count += 1
        
        # Tính độ ổn định
        if self.age > 0:
            self.stability_score = min(1.0, self.matched_count / self.age)


class RedAlertMode:
    # Quản lý trạng thái Báo động Đỏ (Red Alert)
    
    def __init__(self):
        self.active = False
        self.until = 0.0
        self.lockdown_duration = settings.get('fire_logic.lockdown_seconds', 300)
        # print("init red alert")
    
    def activate(self, now):
        # Kích hoạt báo động
        if not self.active:
            print(f"BAO DONG DO!!! - Lockdown {self.lockdown_duration}s")
        self.active = True
        self.until = now + self.lockdown_duration
    
    def is_active(self, now):
        # Kiểm tra trạng thái active
        if self.active and now > self.until:
            print("Het bao dong do")
            self.active = False
            self.until = 0.0
        return self.active
    
    def reset(self):
        # Reset 
        self.active = False
        self.until = 0.0


# Theo dõi & phân tích đám cháy
class FireTracker:
    
    def __init__(self):
        self.objects = {}
        self.next_id = 1
        # self.red_alert = None
        self.red_alert = RedAlertMode()
        self.recent_detections = deque(maxlen=150)
        self.yellow_frames = deque(maxlen=20)
        
        # Load config
        self.config = {
            'yellow_alert_frames': settings.get('fire_logic.yellow_alert_frames', 8),
            'growth_threshold': settings.get('fire_logic.red_alert_growth_threshold', 1.3),
            'growth_window': settings.get('fire_logic.red_alert_growth_window', 10),
            'area_threshold': settings.get('fire_logic.red_alert_area_threshold', 0.05),
            'iou_threshold': settings.get('fire_logic.object_analysis.iou_threshold', 0.4),
            'min_age_warning': settings.get('fire_logic.object_analysis.min_age_for_warning', 10),
            'min_stability': settings.get('fire_logic.object_analysis.min_stability_for_warning', 0.8),
            'max_age': settings.get('fire_logic.object_analysis.max_age', 20),
        }
        self.current_growth_rate = 0.0 # Tỉ lệ tăng trưởng (1.0 = không đổi)
        print("khoi tao fire tracker")
    
    def update(self, detections, now):
        # Cập nhật state & check alerts
        
        # Update objects
        self.match_and_update(detections, now)
        
        # Dọn dẹp
        self.cleanup(now)
        
        # Track lịch xử để phân tích sự phát triển của lửa
        if detections:
            total_area = sum(d['area'] for d in detections)
            self.recent_detections.append({'time': now, 'area': total_area})
        
        # Check Red Alert
        is_red = self.check_red_alert(detections, now)
        
        # Yellow alert: cần số frame liên tục
        has_fire = len(detections) > 0
        self.yellow_frames.append(has_fire)
        
        consecutive_count = sum(1 for x in self.yellow_frames if x)
        is_yellow = consecutive_count >= self.config['yellow_alert_frames']
        
        
        # Quyết định
        should_alert = is_red or (is_yellow and not self.red_alert.active)
        
        return should_alert, is_yellow, is_red
    
    def match_and_update(self, detections, now):
        # Match detections với objects bằng IOU Matching
        if not detections:
            return
        
        # Calculate IOU matrix
        matched_dets = set()
        # matched_objs = set() # unused
        
        for obj_id, obj in self.objects.items():
            best_iou, best_idx = 0, -1
            
            for i, det in enumerate(detections):
                if i in matched_dets:
                    continue
                iou = self.calc_iou(obj.bbox, det['bbox'])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            
            if best_iou > self.config['iou_threshold']:
                det = detections[best_idx]
                obj.update(det['bbox'], det['area'], now)
                matched_dets.add(best_idx)
                # matched_objs.add(obj_id)
        
        # Tạo mới nếu không match được
        for i, det in enumerate(detections):
            if i not in matched_dets:
                print(f"phat hien dam chay moi id={self.next_id}")
                self.objects[self.next_id] = TrackedFireObject(
                    id=self.next_id,
                    bbox=det['bbox'],
                    area=det['area'],
                    first_seen=now,
                    last_seen=now
                )
                self.next_id += 1
    
    def cleanup(self, now):
        # Xóa các object cũ/hết hạn
        max_age = self.config['max_age']
        
        # Dùng dict comprehension
        self.objects = {
            k: v for k, v in self.objects.items()
            if (now - v.last_seen) < 3.0 or v.age < max_age
        }
    
    def check_red_alert(self, detections, now):
        # Logic báo động đỏ (Cháy to / Lan nhanh / Nhiều điểm cháy)
        
        # Check if already in Red Alert
        if self.red_alert.is_active(now):
            # Extend lockdown nếu vẫn đang cháy
            if detections:
                self.red_alert.activate(now) 
                return True
            return False
        
        # DK 1: Cháy to
        if detections:
            total_area = sum(d['area'] for d in detections)
            if total_area > self.config['area_threshold']:
                print("chay to qua -> red alert")
                self.red_alert.activate(now)
                return True
        
        # DK 2: Phát triển nhanh
        if self.check_fire_growth():
            print("lua lan nhanh -> red alert")
            self.red_alert.activate(now)
            return True
        
        # DK 3: Nhiều đám cháy ổn định
        stable_objects = [
            obj for obj in self.objects.values()
            if obj.age >= self.config['min_age_warning']
            and obj.stability_score >= self.config['min_stability']
        ]
        if len(stable_objects) >= 2:
            print(f"co {len(stable_objects)} dam chay on dinh -> red alert")
            self.red_alert.activate(now)
            return True
        
        return False
    
    def check_fire_growth(self):
        # Phân tích tốc độ lan của lửa
        if len(self.recent_detections) < 5:
            return False
        
        now = time.time()
        window = self.config['growth_window']
        threshold = self.config['growth_threshold']
        
        # Lấy detection hiện tại (1s)
        recent = [d for d in self.recent_detections if now - d['time'] < 1.0]
        if not recent:
            return False
        avg_current = np.mean([d['area'] for d in recent])
        
        # Lấy detection quá khứ
        past = [d for d in self.recent_detections 
                if window - 1.0 < now - d['time'] < window + 1.0]
        if not past:
            return False
        avg_past = np.mean([d['area'] for d in past])
        
        # So sánh
        if avg_past > 0:
            rate = avg_current / avg_past
            self.current_growth_rate = rate
            
            if rate > threshold:
                print(f"tang truong nhanh: {avg_past:.4f} : {avg_current:.4f} (x{rate:.2f})")
                return True
        else:
             self.current_growth_rate = 0.0
        
        return False
    
    
    def calc_iou(self, box1, box2):
        # Tính IOU score
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        
        xi1 = max(x1, x1_)
        yi1 = max(y1, y1_)
        xi2 = min(x2, x2_)
        yi2 = min(y2, y2_)
        
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)
        union = box1_area + box2_area - inter
        
        return inter / union if union > 0 else 0
    
    def get_is_red_alert(self):
        # Getter
        return self.red_alert.is_active(time.time())
    
    
    def get_tracked_objects(self):
        return list(self.objects.values())


    def get_is_yellow_alert(self):
        # Getter
        consecutive_count = sum(1 for x in self.yellow_frames if x)
        return consecutive_count >= self.config['yellow_alert_frames']

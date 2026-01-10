# Ghi video
import cv2
import time
import uuid
import threading
import numpy as np
from pathlib import Path

from config import settings
from utils import security


# Class ghi hình video
class Recorder:
    
    def __init__(self):
        self.lock = threading.Lock()
        self.dang_ghi = None
        self.out_dir = settings.paths.tmp_dir
        self.fps = settings.recorder.fps
        
        self.fourcc = cv2.VideoWriter.fourcc(*settings.recorder.fourcc)
        print("[OK] Khởi tạo recorder xong")


    # Bắt đầu phiên ghi hình
    def start(self, source_id, reason="alert", duration=None, wait_for_user=False):
        print("Bắt đầu ghi video...")
        print(source_id)
        
        duration = duration or settings.recorder.duration
        
        
        with self.lock:
            if self.dang_ghi is not None:
                print("Đang ghi rồi, bỏ qua")
                return None
            
            # Tạo tên file
            filename = f"rec_{reason}_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
            path = self.out_dir / filename
            
            print("Tên file:", filename)

            self.dang_ghi = {
                'path': path,
                'writer': None,
                'end_time': time.time() + duration,
                'source_id': source_id,
                'reason': reason,
                'alert_ids': [], # Danh sách cảnh báo
                'wait_for_user': wait_for_user,
            }
            
            return self.dang_ghi
    
    # Ghi frame
    def write(self, frame):
        with self.lock:
            if self.dang_ghi is None:
                return False
            
            # Init writer nếu chưa có
            if self.dang_ghi['writer'] is None:
                print("Tạo writer mới")
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(self.dang_ghi['path']),
                    self.fourcc,
                    self.fps,
                    (w, h)
                )
                if not writer.isOpened():
                    print("Lỗi: Không mở được file video")
                    self.dang_ghi = None
                    return False
                self.dang_ghi['writer'] = writer
            
            self.dang_ghi['writer'].write(frame)
            return True

    # Kiểm tra đã xong chưa
    def check_finalize(self):
        with self.lock:
            if self.dang_ghi is None:
                return None
            
            
            if time.time() < self.dang_ghi['end_time']:
                return None
            
            if self.dang_ghi.get('wait_for_user'):
                # print("Đợi người dùng...")
                return None
            
            print("Xong video")
            return self.finalize()
    
    # Kết thúc và mã hóa file
    def finalize(self):
        rec = self.dang_ghi
        self.dang_ghi = None
        
        if rec['writer']:
            rec['writer'].release()
        
        # Mã hóa file
        if rec['path'].exists():
            print("Mã hóa file...")
            security.encrypt_file(rec['path'])
        
        return {
            'path': rec['path'],
            'source_id': rec['source_id'],
            'alert_ids': rec['alert_ids'],
        }


    # Dừng ghi (bắt buộc dừng)
    def stop(self):
        print("Dừng ghi")
        with self.lock:
            if self.dang_ghi:
                self.dang_ghi['end_time'] = time.time()
                self.dang_ghi['wait_for_user'] = False
    
    # Hủy bỏ (xóa file)
    def discard(self):
        print("Hủy bỏ video")
        with self.lock:
            if self.dang_ghi is None:
                return False
            
            if self.dang_ghi['writer']:
                self.dang_ghi['writer'].release()
            
            if self.dang_ghi['path'].exists():
                self.dang_ghi['path'].unlink()
            
            self.dang_ghi = None
            return True
    
    # Gia hạn thời gian
    def extend(self, seconds):
        print(f"Gia hạn thêm {seconds}s")
        with self.lock:
            if self.dang_ghi:
                self.dang_ghi['end_time'] += seconds
    
    # Dừng chờ user
    def resolve_user_wait(self):
        print("Người dùng đã phản hồi")
        with self.lock:
            if self.dang_ghi:
                self.dang_ghi['wait_for_user'] = False
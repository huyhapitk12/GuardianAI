# Ghi video
import cv2
import time
import uuid
import threading
import numpy as np
from pathlib import Path

from config import settings
from utils import security


# Ghi video
class Recorder:
    
    def __init__(self):
        self.lock = threading.Lock()
        self.dang_ghi = None
        self.out_dir = settings.paths.tmp_dir
        self.fps = settings.recorder.fps
        
        self.fourcc = cv2.VideoWriter.fourcc(*settings.recorder.fourcc)
        print("khoi tao recorder xong")


    # Bắt đầu ghi video
    def start(self, source_id, reason="alert", duration=None, wait_for_user=False):
        print("bat dau ghi video...")
        print(source_id)
        
        duration = duration or settings.recorder.duration
        
        
        with self.lock:
            if self.dang_ghi is not None:
                print("dang ghi roi bo qua")
                return None
            
            # Tạo tên file
            filename = f"rec_{reason}_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
            path = self.out_dir / filename
            
            print("file name:", filename)

            self.dang_ghi = {
                'path': path,
                'writer': None,
                'end_time': time.time() + duration,
                'source_id': source_id,
                'reason': reason,
                'alert_ids': [], # Danh sách alert
                'wait_for_user': wait_for_user,
            }
            
            return self.dang_ghi
    
    # Ghi frame vào file
    def write(self, frame):
        with self.lock:
            if self.dang_ghi is None:
                return False
            
            # Init writer nếu chưa có
            if self.dang_ghi['writer'] is None:
                print("tao writer moi")
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(self.dang_ghi['path']),
                    self.fourcc,
                    self.fps,
                    (w, h)
                )
                if not writer.isOpened():
                    print("loi khong mo duoc file video")
                    self.dang_ghi = None
                    return False
                self.dang_ghi['writer'] = writer
            
            self.dang_ghi['writer'].write(frame)
            return True

    # Check xem xong chưa để lưu
    def check_finalize(self):
        with self.lock:
            if self.dang_ghi is None:
                return None
            
            
            if time.time() < self.dang_ghi['end_time']:
                return None
            
            if self.dang_ghi.get('wait_for_user'):
                # print("doi user...")
                return None
            
            print("xong video")
            return self.finalize()
    
    # Lưu và mã hóa
    def finalize(self):
        rec = self.dang_ghi
        self.dang_ghi = None
        
        if rec['writer']:
            rec['writer'].release()
        
        # Mã hóa file
        if rec['path'].exists():
            print("ma hoa file...")
            security.encrypt_file(rec['path'])
        
        return {
            'path': rec['path'],
            'source_id': rec['source_id'],
            'alert_ids': rec['alert_ids'],
        }


    # Dừng ghi
    def stop(self):
        print("stop ghi")
        with self.lock:
            if self.dang_ghi:
                self.dang_ghi['end_time'] = time.time()
                self.dang_ghi['wait_for_user'] = False
    
    # Hủy ghi
    def discard(self):
        print("huy bo video")
        with self.lock:
            if self.dang_ghi is None:
                return False
            
            if self.dang_ghi['writer']:
                self.dang_ghi['writer'].release()
            
            if self.dang_ghi['path'].exists():
                self.dang_ghi['path'].unlink()
            
            self.dang_ghi = None
            return True
    
    # Gia hạn thời gian ghi
    def extend(self, seconds):
        print(f"gia han them {seconds}s")
        with self.lock:
            if self.dang_ghi:
                self.dang_ghi['end_time'] += seconds
    
    # Chờ user phản hồi
    def resolve_user_wait(self):
        print("user da phan hoi")
        with self.lock:
            if self.dang_ghi:
                self.dang_ghi['wait_for_user'] = False
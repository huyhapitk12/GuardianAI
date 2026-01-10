# Quản lý danh sách camera & đa luồng

import threading
import numpy as np

from config import settings
from core.camera import Camera
from core.detection import PersonTracker


class CameraManager:
    def __init__(self, person_alert=None, fire_alert=None, fall_alert=None):
        # Dictionary camera {id: camera_obj}
        self.cameras = {}
        self.threads = {}
        
        self.lock = threading.Lock()        # Khóa để tránh xung đột luồng
        
        # Các bộ phát hiện (dùng cho camera mới)
        self.fire_detector = None
        self.face_detector = None
        self.state = None
        
        self.person_alert = person_alert
        self.fire_alert = fire_alert
        self.fall_alert = fall_alert
        
        # Tạo camera từ cấu hình
        self.create_cameras()
    
    # Tạo camera từ cấu hình
    def create_cameras(self):
        sources = settings.camera.sources
        if not isinstance(sources, (list, tuple)):
            sources = [sources]
            
        for source in sources:
            # Nếu source là số (webcam ID), chuyển về int
            if isinstance(source, int):
                src = source
            elif isinstance(source, str) and source.isdigit():
                src = int(source)
            else:
                src = source
            
            # Tạo camera
            cam = Camera(src, self.person_alert, self.fire_alert, self.fall_alert, shared_model=None)
            
            # Thêm vào dictionary
            self.cameras[str(source)] = cam
            print(f"[OK] Đã tạo camera: {source}")
    
    # Khởi động toàn bộ camera
    def start(self, fire_detector, face_detector, state_manager):
        # Lưu lại để dùng khi thêm camera mới
        self.fire_detector = fire_detector
        self.face_detector = face_detector
        self.state = state_manager
        
        # Sử dụng khóa để tránh xung đột
        with self.lock:
            for source, cam in self.cameras.items():
                # Khởi tạo các worker trong camera
                cam.start_workers(fire_detector, face_detector)
                
                # Tạo thread riêng cho mỗi camera
                thread = threading.Thread(
                    target=cam.process_loop,
                    args=(state_manager,),
                    daemon=True
                )
                
                self.threads[source] = thread
                thread.start()
                print(f"[OK] Camera {source} đang chạy!")

    # Dừng toàn bộ camera  
    def stop(self):
        print("Đang dừng camera...")
        
        # Báo hiệu tất cả camera dừng
        with self.lock:
            for cam in self.cameras.values():
                cam.quit = True
        
        # Chờ kết thúc các luồng
        for thread in self.threads.values():
            thread.join(timeout=5.0)
        
        print("[OK] Đã dừng tất cả camera!")
    
    # Lấy camera theo ID
    def get_camera(self, source):
        with self.lock:
            return self.cameras.get(source)
    
    # Lấy frame từ tất cả camera
    def get_all_frames(self):
        frames = {}
        with self.lock:
            for source, cam in self.cameras.items():
                frames[source] = cam.read()
        return frames
    
    # Lấy trạng thái kết nối của tất cả camera
    def get_status(self):
        status = {}
        with self.lock:
            for source, cam in self.cameras.items():
                status[source] = cam.get_connection_status()
        return status
    
    # Thêm camera mới vào hệ thống
    def add_camera(self, source): # source: URL hoặc ID của camera
        # Kiểm tra đã tồn tại chưa
        with self.lock:
            if source in self.cameras:
                return False, "Camera đã tồn tại!"
        
        # Chuyển đổi source
        if isinstance(source, int):
            src = source
        elif isinstance(source, str) and source.isdigit():
            src = int(source)
        else:
            src = source
        
        # Tạo camera mới
        cam = Camera(src, self.person_alert, self.fire_alert, self.fall_alert, shared_model=None)
        
        # Khởi động ngay nếu hệ thống đang chạy
        if self.fire_detector and self.face_detector:
            cam.start_workers(self.fire_detector, self.face_detector)
            
            # Tạo thread
            thread = threading.Thread(
                target=cam.process_loop,
                args=(self.state,),
                daemon=True
            )
            
            # Thêm vào danh sách
            with self.lock:
                self.cameras[source] = cam
                self.threads[source] = thread
            
            thread.start()
            
            # Lưu config
            from config.settings import add_camera_source_to_config
            save_ok, save_msg = add_camera_source_to_config(source)
            
            if not save_ok:
                print(f"[WARN] Camera đã thêm nhưng chưa lưu vào config: {save_msg}")
        
        return True, f"Đã thêm camera {source}!"

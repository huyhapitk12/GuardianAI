# core/camera_manager.py
# =============================================================================
# Cho phép:
# - Thêm/xóa camera
# - Lấy hình ảnh từ tất cả camera
# - Kiểm tra trạng thái kết nối
# =============================================================================

import threading
import numpy as np

from config import settings
from core.camera import Camera
from core.detection import PersonTracker


class CameraManager:
    def __init__(self, person_alert=None, fire_alert=None, fall_alert=None):
        # Lưu các camera
        self.cameras = {} # Key: ID camera, Value: Object Camera
        
        # Lưu các thread xử lý
        self.threads = {}
        
        # Lock để tránh xung đột khi nhiều thread truy cập cùng lúc
        self.lock = threading.Lock()
        
        # Lưu các detector để dùng khi thêm camera mới
        self.fire_detector = None
        self.face_detector = None
        self.state = None
        
        self.person_alert = person_alert
        self.fire_alert = fire_alert
        self.fall_alert = fall_alert
        
        # Tạo camera từ config
        self.create_cameras()
    
    # Tạo camera từ config
    def create_cameras(self):
        sources = settings.camera.sources
        if not isinstance(sources, (list, tuple)):
            sources = [sources]
            
        for source in sources:
            try:
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
                print(f"✅ Đã tạo camera: {source}")
                
            except Exception as e:
                print(f"❌ Lỗi tạo camera {source}: {e}")
    
    # Chạy tất cả camera
    def start(self, fire_detector, face_detector, state_manager):
        # Lưu lại để dùng khi thêm camera mới
        self.fire_detector = fire_detector
        self.face_detector = face_detector
        self.state = state_manager
        
        # Dùng lock để tránh xung đột
        with self.lock:
            for source, cam in self.cameras.items():
                # Khởi tạo các worker trong camera
                cam.start_workers(fire_detector, face_detector)
                
                # Tạo thread riêng cho mỗi camera
                # daemon=True: thread tự tắt khi chương trình main tắt
                thread = threading.Thread(
                    target=cam.process_loop,
                    args=(state_manager,),
                    daemon=True
                )
                
                self.threads[source] = thread
                thread.start()
                print(f"✅ Camera {source} đang chạy!")

    # Dừng tất cả camera    
    def stop(self):
        print("Đang dừng camera...")
        
        # Báo hiệu tất cả camera dừng
        with self.lock:
            for cam in self.cameras.values():
                cam.quit = True
        
        # Chờ các thread kết thúc (tối đa 5 giây mỗi thread)
        for thread in self.threads.values():
            thread.join(timeout=5.0)
        
        print("✅ Đã dừng tất cả camera!")
    
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
        
        try:
            # Chuyển đổi source
            if isinstance(source, int):
                src = source
            elif isinstance(source, str) and source.isdigit():
                src = int(source)
            else:
                src = source
            
            # Tạo camera mới
            cam = Camera(src, self.person_alert, self.fire_alert, self.fall_alert, shared_model=None)
            
            # Nếu hệ thống đã chạy, start camera luôn
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
                
                # Lưu vào config để lần sau còn nhớ
                from config.settings import add_camera_source_to_config
                save_ok, save_msg = add_camera_source_to_config(source)
                
                if not save_ok:
                    print(f"⚠️ Camera đã thêm nhưng chưa lưu vào config: {save_msg}")
            
            return True, f"Đã thêm camera {source}!"
            
        except Exception as e:
            return False, str(e)

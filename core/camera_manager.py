# core/camera_manager.py
import threading
from typing import List, Dict, Optional, Tuple
import numpy as np
from core.camera import Camera
from config.settings import settings, add_camera_source_to_config, Settings

def print_info(message):
    """Simple print function for info level"""
    import datetime as dt
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - camera_manager - INFO - {message}")

def print_warning(message):
    """Simple print function for warning level"""
    import datetime as dt
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - camera_manager - WARNING - {message}")

def print_error(message):
    """Simple print function for error level"""
    import datetime as dt
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - camera_manager - ERROR - {message}")

class CameraManager:
    """Quản lý việc tạo và xử lý nhiều camera."""

    def __init__(
        self,
        show_windows: bool = False,
        on_person_alert: Optional[callable] = None,
        on_fire_alert: Optional[callable] = None,
    ):
        self.camera_sources = settings.camera.sources
        self.show_windows = show_windows
        self.on_person_alert = on_person_alert
        self.on_fire_alert = on_fire_alert

        self.cameras: Dict[str, Camera] = {}
        self.camera_threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

        self.fire_detector = None
        self.face_detector = None
        self.state_manager = None

        self._create_cameras()

    def _create_cameras(self):
        """Khởi tạo các đối tượng Camera cho mỗi nguồn."""
        for source in self.camera_sources:
            try:
                # Chuyển đổi nguồn thành số nguyên nếu nó là một chuỗi số
                processed_source = int(source) if source.isdigit() else source
                
                # Tạo một PersonTracker riêng cho mỗi camera
                from core.detection.all_detectors import PersonTracker
                person_tracker = PersonTracker(self.face_detector) # face_detector được truyền vào sau
                
                cam = Camera(
                    source=processed_source,
                    show_window=self.show_windows,
                    on_person_alert=self.on_person_alert,
                    on_fire_alert=self.on_fire_alert,
                    person_tracker=person_tracker
                )
                self.cameras[str(source)] = cam
                print_info(f"Đối tượng camera được tạo cho nguồn: {source}")
            except (RuntimeError, ValueError) as e:
                print_error(f"Không thể tạo camera cho nguồn {source}: {e}")

    def start_workers(self, fire_detector, face_detector, state_manager):
        """Bắt đầu các luồng công nhân cho tất cả các camera."""
        self.fire_detector = fire_detector
        self.face_detector = face_detector
        self.state_manager = state_manager

        self.state_manager = state_manager

        with self._lock:
            cameras_items = list(self.cameras.items())

        for source, cam in cameras_items:
            # Truyền face_detector vào person_tracker của camera
            if cam.person_tracker:
                cam.person_tracker.face_detector = face_detector
                cam.person_tracker.initialize() # Khởi tạo model YOLO bên trong tracker

            cam.start_workers(fire_detector, face_detector)
            thread = threading.Thread(
                target=cam.process_frames,
                args=(state_manager,),
                daemon=True
            )
            self.camera_threads[source] = thread
            thread.start()
            print_info(f"Luồng xử lý đã bắt đầu cho camera: {source}")

    def stop_all(self):
        """Dừng tất cả các luồng camera."""
        print_info("Bắt đầu dừng tất cả các camera...")
        with self._lock:
            cameras_values = list(self.cameras.values())
            
        for cam in cameras_values:
            cam.release()  # Đặt cờ thoát và giải phóng tài nguyên
        
        # Đợi các luồng kết thúc
        for source, thread in self.camera_threads.items():
            thread.join(timeout=5.0)
            if thread.is_alive():
                print_warning(f"Luồng camera {source} không dừng kịp thời.")
        
        print_info("Tất cả các luồng camera đã được dừng.")

    def get_all_frames(self) -> Dict[str, Tuple[bool, Optional[np.ndarray]]]:
        """Lấy khung hình mới nhất từ tất cả các camera."""
        frames = {}
        with self._lock:
            cameras_items = list(self.cameras.items())
            
        for source, cam in cameras_items:
            frames[source] = cam.read()
        return frames

    def get_all_raw_frames(self) -> Dict[str, Tuple[bool, Optional[np.ndarray]]]:
        """Lấy khung hình thô mới nhất từ tất cả các camera."""
        raw_frames = {}
        with self._lock:
            cameras_items = list(self.cameras.items())
            
        for source, cam in cameras_items:
            raw_frames[source] = cam.read_raw()
        return raw_frames

    def get_camera(self, source: str) -> Optional[Camera]:
        """Lấy một đối tượng camera cụ thể bằng nguồn của nó."""
        with self._lock:
            return self.cameras.get(source)

    def get_all_connection_statuses(self) -> Dict[str, dict]:
        """Lấy trạng thái kết nối của tất cả các camera."""
        statuses = {}
        with self._lock:
            cameras_items = list(self.cameras.items())
            
        for source, cam in cameras_items:
            statuses[source] = cam.get_connection_status()
        return statuses
        
    def reset_fire_state_for_source(self, source: str):
        """Reset trạng thái phát hiện cháy cho một camera cụ thể."""
        cam = self.get_camera(source)
        if cam:
            cam.reset_fire_state()

    def add_new_camera(self, new_source):
        """
        Thêm một camera mới lúc đang chạy và lưu vào config.
        Trả về (True, "Thông báo") nếu thành công, (False, "Lỗi") nếu thất bại.
        """
        source_str = str(new_source).strip()
        
        # 1. Kiểm tra xem camera đã chạy trong session này chưa
        with self._lock:
            if source_str in self.cameras:
                print_warning(f"Camera {source_str} đã được quản lý.")
                return False, "Camera này đã đang chạy."

        # 2. Kiểm tra xem các worker (detectors) đã sẵn sàng chưa
        if not all([self.fire_detector, self.face_detector, self.state_manager]):
            print_error("Chưa gọi start_workers. Không thể thêm camera mới.")
            return False, "Trình quản lý các worker chưa sẵn sàng."

        try:
            # 3. Thử tạo đối tượng camera
            print_info(f"Đang thử thêm camera mới: {source_str}")
            processed_source = int(source_str) if source_str.isdigit() else source_str
            
            # Tạo một PersonTracker riêng cho camera mới
            from core.detection.all_detectors import PersonTracker
            person_tracker = PersonTracker(self.face_detector)
            person_tracker.initialize()

            cam = Camera(
                source=processed_source,
                show_window=self.show_windows,
                on_person_alert=self.on_person_alert,
                on_fire_alert=self.on_fire_alert,
                person_tracker=person_tracker
            )
            
            # (Nên có một hàm cam.is_opened() trong class Camera
            # để kiểm tra kết nối ở đây)

            # 4. Lưu vào file config.yaml
            # (Hàm này giờ đã nằm trong settings.py)
            success, message = add_camera_source_to_config(source_str) 
            if not success:
                # (Trường hợp này có thể là camera đã tồn tại trong config
                # nhưng chưa chạy trong session này)
                print_warning(f"Không thể lưu camera vào config: {message}")
                # Vẫn có thể tiếp tục chạy session này
                pass

            # 5. Khởi động worker cho camera mới
            cam.start_workers(self.fire_detector, self.face_detector)
            thread = threading.Thread(
                target=cam.process_frames,
                args=(self.state_manager,),
                daemon=True
            )
            
            # 6. Thêm vào danh sách quản lý VÀ bắt đầu luồng
            # 6. Thêm vào danh sách quản lý VÀ bắt đầu luồng
            with self._lock:
                self.cameras[source_str] = cam
                self.camera_threads[source_str] = thread
            
            thread.start()
            
            print_info(f"Đã thêm và khởi động camera mới: {source_str}")
            return True, f"Thêm camera {source_str} thành công."
            
        except (RuntimeError, ValueError, Exception) as e:
            print_error(f"Không thể tạo camera cho nguồn {new_source}: {e}")
            return False, f"Lỗi khi thêm camera: {e}"
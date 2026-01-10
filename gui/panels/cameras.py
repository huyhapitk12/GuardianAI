# Bảng quản lý Camera

import cv2
import uuid
import threading
from typing import Optional

from PIL import Image, ImageEnhance
import customtkinter as ctk
from customtkinter import CTkImage, StringVar

from config import settings
from utils import security
from gui.styles import Colors, Fonts, Sizes, create_button, create_card
from gui.widgets import CameraList, log_activity


# Xem camera trực tiếp và các điều khiển
class CamerasPanel(ctk.CTkFrame):
    
    
    def __init__(self, parent, camera_manager, state_manager, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        
        self.camera_manager = camera_manager
        self.state = state_manager
        self.selected_camera = StringVar()
        self.brightness = 1.0
        self.running = True
        
        # Lấy danh sách nguồn camera
        sources = list(camera_manager.cameras.keys()) if camera_manager else []
        if sources:
            self.selected_camera.set(sources[0])
        
        self.build_ui()
        self.start_video_loop()
    
    def build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Video panel
        video_panel = create_card(self)
        video_panel.grid(row=0, column=0, sticky="nsew", padx=Sizes.SM, pady=Sizes.SM)
        
        self.video_label = ctk.CTkLabel(
            video_panel,
            text="Chọn camera",
            font=Fonts.TITLE_MD,
            text_color=Colors.TEXT_MUTED
        )
        self.video_label.pack(expand=True, fill="both", padx=Sizes.LG, pady=Sizes.LG)
        
        # Control panel
        control_panel = ctk.CTkFrame(self, fg_color="transparent")
        control_panel.grid(row=0, column=1, sticky="nsew", padx=(0, Sizes.SM), pady=Sizes.SM)
        
        # Quick actions
        actions_card = create_card(control_panel)
        actions_card.pack(fill="x", pady=(0, Sizes.SM))
        
        ctk.CTkLabel(
            actions_card, text="Thao tác nhanh",
            font=Fonts.BODY_BOLD, text_color=Colors.TEXT_SECONDARY
        ).pack(anchor="w", padx=Sizes.MD, pady=(Sizes.MD, Sizes.SM))
        
        btn_frame = ctk.CTkFrame(actions_card, fg_color="transparent")
        btn_frame.pack(fill="x", padx=Sizes.MD, pady=(0, Sizes.MD))
        btn_frame.grid_columnconfigure((0, 1), weight=1)
        
        create_button(
            btn_frame, "Ghi hình", "success", "small",
            command=self.toggle_record
        ).grid(row=0, column=0, padx=(0, Sizes.XS), sticky="ew")
        
        create_button(
            btn_frame, "Chụp ảnh", "secondary", "small",
            command=self.take_snapshot
        ).grid(row=0, column=1, padx=(Sizes.XS, 0), sticky="ew")
        
        # Camera list
        list_card = create_card(control_panel)
        list_card.pack(fill="both", expand=True)
        
        self.camera_list = CameraList(
            list_card,
            self.camera_manager,
            self.state,
            on_view=self.select_camera,
            on_add=self.add_camera
        )
        self.camera_list.pack(fill="both", expand=True, padx=Sizes.SM, pady=Sizes.SM)
    
    # Bắt đầu vòng lặp cập nhật video
    def start_video_loop(self):
        self.update_video()
    
    # Cập nhật khung hình video
    def update_video(self):
        if not self.running:
            return
        
        selected = self.selected_camera.get()
        if not selected:
            self.after(100, self.update_video)
            return
        
        cam = self.camera_manager.get_camera(selected) if self.camera_manager else None
        if not cam:
            self.video_label.configure(text=f"[ERR] Không tìm thấy camera: {selected}", image=None)
            self.after(1000, self.update_video)
            return
        
        ret, frame = cam.read()
        
        if ret and frame is not None:
            # Đổi kích thước với phép nội suy chất lượng cao
            target_w, target_h = Sizes.VIDEO_WIDTH, Sizes.VIDEO_HEIGHT
            frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Chuyển đổi định dạng
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Độ sáng
            if self.brightness != 1.0:
                pil_img = ImageEnhance.Brightness(pil_img).enhance(self.brightness)
            
            ctk_img = CTkImage(pil_img, size=(target_w, target_h))
            
            if self.video_label.winfo_exists():
                self.video_label.configure(text="", image=ctk_img)
                self.video_label.image = ctk_img
        else:
            if self.video_label.winfo_exists():
                self.video_label.configure(text="Đang kết nối lại...", image=None)
        
        # Lên lịch cập nhật tiếp theo
        refresh_rate = 33 if self.state.is_detection_enabled() else 50
        self.after(refresh_rate, self.update_video)
    
    # Chọn camera để xem
    def select_camera(self, source: str):
        self.selected_camera.set(source)
        log_activity(f"Đang xem camera: {source}", "info")
    
    # Hiển thị dialog thêm camera
    def add_camera(self):
        from gui.dialogs import AddCameraDialog
        
        def on_success(source_id):
            # Gọi khi camera được thêm thành công
            camera = self.camera_manager.get_camera(source_id)
            if camera and self.camera_list:
                self.camera_list.add_camera(source_id, camera)
                # Tự động chọn camera mới
                self.select_camera(source_id)
        
        dialog = AddCameraDialog(self, self.camera_manager, on_success=on_success)
        dialog.grab_set()
    
    # Chụp ảnh từ camera hiện tại
    def take_snapshot(self):
        selected = self.selected_camera.get()
        if not selected:
            return
        
        cam = self.camera_manager.get_camera(selected) if self.camera_manager else None
        if not cam:
            return
        
        ret, frame = cam.read_raw()
        if not ret or frame is None:
            return
        
        img_path = settings.paths.tmp_dir / f"snap_{uuid.uuid4().hex}.jpg"
        security.save_image(img_path, frame)
        
        log_activity(f"Đã lưu ảnh: {img_path.name}", "success")
    
    # Bật/tắt ghi hình
    def toggle_record(self):
        log_activity("Đã bật/tắt ghi hình", "info")
    
    # Dừng vòng lặp video
    def stop(self):
        self.running = False
    
    def destroy(self):
        self.stop()
        super().destroy()
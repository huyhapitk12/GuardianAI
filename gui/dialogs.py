# Các hộp thoại GUI cho ứng dụng Guardian

import customtkinter as ctk

from gui.styles import Colors, Fonts, Sizes, create_button
from gui.widgets import log_activity


class AddCameraDialog(ctk.CTkToplevel):
    # Hộp thoại thêm camera mới
    
    def __init__(self, parent, camera_manager, on_success=None):
        super().__init__(parent)
        
        self.camera_manager = camera_manager
        self.on_success = on_success
        self.result = None
        
        self.setup_window()
        self.build_ui()
    
    def setup_window(self):
        # Cấu hình cửa sổ hộp thoại
        self.title("Thêm Camera")
        self.geometry("500x400")
        self.resizable(False, False)
        
        # Canh giữa cửa sổ cha
        self.transient(self.master)
        self.update_idletasks()
        
        parent_x = self.master.winfo_x()
        parent_y = self.master.winfo_y()
        parent_w = self.master.winfo_width()
        parent_h = self.master.winfo_height()
        
        x = parent_x + (parent_w - 500) // 2
        y = parent_y + (parent_h - 400) // 2
        self.geometry(f"+{x}+{y}")
        
        self.configure(fg_color=Colors.BG_PRIMARY)
    
    def build_ui(self):
        # Xây dựng giao diện hộp thoại
        # Container chính
        container = ctk.CTkFrame(self, fg_color=Colors.BG_SECONDARY, corner_radius=Sizes.RADIUS_LG)
        container.pack(fill="both", expand=True, padx=Sizes.LG, pady=Sizes.LG)
        
        # Tiêu đề
        ctk.CTkLabel(
            container,
            text="Thêm Camera Mới",
            font=Fonts.TITLE_LG,
            text_color=Colors.TEXT_PRIMARY
        ).pack(pady=(Sizes.LG, Sizes.MD))
        
        # Mô tả
        ctk.CTkLabel(
            container,
            text="Nhập nguồn camera (ID webcam hoặc URL RTSP)",
            font=Fonts.BODY,
            text_color=Colors.TEXT_SECONDARY
        ).pack(pady=(0, Sizes.LG))
        
        # Phần nhập liệu
        input_frame = ctk.CTkFrame(container, fg_color="transparent")
        input_frame.pack(fill="x", padx=Sizes.LG, pady=Sizes.MD)
        
        # Nhãn nguồn
        ctk.CTkLabel(
            input_frame,
            text="Nguồn Camera:",
            font=Fonts.BODY_BOLD,
            text_color=Colors.TEXT_PRIMARY,
            anchor="w"
        ).pack(anchor="w", pady=(0, Sizes.XS))
        
        # Ô nhập nguồn
        self.source_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="VD: 0, 1, hoặc rtsp://...",
            font=Fonts.BODY,
            fg_color=Colors.BG_TERTIARY,
            border_color=Colors.PRIMARY,
            text_color=Colors.TEXT_PRIMARY,
            height=40
        )
        self.source_entry.pack(fill="x", pady=(0, Sizes.SM))
        
        # Nhãn trạng thái
        self.status_label = ctk.CTkLabel(
            container,
            text="",
            font=Fonts.BODY,
            text_color=Colors.SUCCESS
        )
        self.status_label.pack(pady=Sizes.SM)
        
        # Các nút
        btn_frame = ctk.CTkFrame(container, fg_color="transparent")
        btn_frame.pack(pady=(Sizes.LG, Sizes.MD), padx=Sizes.LG)
        btn_frame.grid_columnconfigure((0, 1), weight=1)
        
        create_button(
            btn_frame, "Hủy", "secondary", "medium",
            command=self.on_cancel
        ).grid(row=0, column=0, padx=(0, Sizes.SM), sticky="ew")
        
        create_button(
            btn_frame, "Thêm Camera", "success", "medium",
            command=self.on_add
        ).grid(row=0, column=1, padx=(Sizes.SM, 0), sticky="ew")
        
        # Gán phím Enter
        self.source_entry.bind("<Return>", lambda e: self.on_add())
        self.source_entry.focus()
    
    def on_add(self):
        # Xử lý thêm camera
        source_str = self.source_entry.get().strip()
        
        if not source_str:
            self.show_status("Vui lòng nhập nguồn camera", Colors.ERROR)
            return
        
        # Thêm camera bằng phương thức có sẵn của CameraManager
        self.show_status("Đang thêm camera...", Colors.PRIMARY)
        self.update()
        
        # Sử dụng phương thức add_camera
        success, message = self.camera_manager.add_camera(source_str)
        
        if success:
            log_activity(message, "success")
            self.show_status(f"[OK] {message}", Colors.SUCCESS)
            
            # Gọi callback làm mới nếu có
            if self.on_success:
                self.on_success(source_str)
            
            # Đóng sau một khoảng trễ
            self.after(1000, self.destroy)
        else:
            self.show_status(f"[ERR] Lỗi: {message}", Colors.ERROR)
    
    def on_cancel(self):
        # Xử lý hủy
        self.destroy()
    
    def show_status(self, message: str, color: str):
        # Hiển thị thông báo trạng thái
        self.status_label.configure(text=message, text_color=color)

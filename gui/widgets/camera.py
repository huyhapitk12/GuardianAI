# Các widget liên quan đến camera

import cv2
import threading
from typing import Callable, Dict, Optional

import customtkinter as ctk
from customtkinter import StringVar

from gui.styles import Colors, Fonts, Sizes, create_button, create_card, create_switch
from gui.widgets.activity import log_activity


# Thẻ điều khiển camera riêng lẻ
class CameraCard(ctk.CTkFrame):
    
    
    def __init__(
        self, parent,
        source: str,
        camera,
        state_manager,
        on_view: Callable = None,
        **kwargs
    ):
        super().__init__(parent, fg_color=Colors.BG_SECONDARY, corner_radius=Sizes.RADIUS_LG, **kwargs)
        
        self.source = source
        self.camera = camera
        self.state = state_manager
        self.on_view = on_view
        self.labels = {}
        self.switches = {}
        
        self.build_ui()
    
    def build_ui(self):
        # Header row
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=Sizes.MD, pady=(Sizes.MD, Sizes.SM))
        
        # Camera Name via header grid
        header.grid_columnconfigure(0, weight=1)
        
        # Row 0: Name and Status Dot
        top_frame = ctk.CTkFrame(header, fg_color="transparent")
        top_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        display_name = self.source
        # Slightly less truncation needed if on own row, but keep safety
        if len(display_name) > 30:
            display_name = display_name[:27] + "..."
            
        title = ctk.CTkLabel(
            top_frame, text=f"📹 Camera {display_name}",
            font=Fonts.BODY_BOLD, text_color=Colors.TEXT_PRIMARY,
            anchor="w"
        )
        title.pack(side="left")
        
        # Status dot (moved next to name)
        self.labels['dot'] = ctk.CTkFrame(top_frame, width=10, height=10, corner_radius=5)
        self.labels['dot'].pack(side="right", padx=Sizes.SM)

        # Row 1: Switches
        # Switch container
        sw_frame = ctk.CTkFrame(header, fg_color="transparent")
        sw_frame.grid(row=1, column=0, sticky="ew")
        # Face Switch
        face_var = StringVar(value="on") # Default, will sync later
        self.switches['face'] = face_var
        
        face_sw = ctk.CTkSwitch(
            sw_frame, text="Khuôn mặt", variable=face_var,
            onvalue="on", offvalue="off",
            width=80, height=20, font=Fonts.TINY,
            command=lambda: self.toggle_feature('face')
        )
        face_sw.pack(side="right", padx=5) # Right aligned
        
        # Sync initial state
        if self.camera:
            # Read from camera logic (Global + Override)
            from config import settings
            f_val = self.camera.face_enabled if self.camera.face_enabled is not None else settings.get('detection.face_recognition_enabled', True)
            
            face_var.set("on" if f_val else "off")

        
        # Stats row
        stats = ctk.CTkFrame(self, fg_color="transparent")
        stats.pack(fill="x", padx=Sizes.MD, pady=(0, Sizes.SM))
        
        self.labels['fps'] = ctk.CTkLabel(stats, text="FPS: 0", font=Fonts.TINY, text_color=Colors.TEXT_MUTED)
        self.labels['fps'].pack(side="left", padx=(0, Sizes.SM))
        
        self.labels['res'] = ctk.CTkLabel(stats, text="0x0", font=Fonts.TINY, text_color=Colors.TEXT_MUTED)
        self.labels['res'].pack(side="left")
        
        self.labels['status'] = ctk.CTkLabel(stats, text="...", font=Fonts.TINY)
        self.labels['status'].pack(side="right")
        
        # IR row
        ir_row = ctk.CTkFrame(self, fg_color="transparent")
        ir_row.pack(fill="x", padx=Sizes.MD, pady=(0, Sizes.SM))
        
        self.labels['ir'] = ctk.CTkLabel(ir_row, text="IR: OFF", font=Fonts.TINY, text_color=Colors.TEXT_MUTED)
        self.labels['ir'].pack(side="left")
        
        ir_var = StringVar(value="off")
        self.switches['ir'] = ir_var
        
        ctk.CTkSwitch(
            ir_row, text="Enhance", variable=ir_var,
            onvalue="on", offvalue="off",
            command=lambda: self.toggle_ir(),
            font=Fonts.TINY, progress_color=Colors.WARNING,
            width=70, height=20
        ).pack(side="right")
        
        # Actions row
        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=Sizes.MD, pady=(0, Sizes.MD))
        
        if self.on_view:
            create_button(
                actions, "View", "primary", "small",
                width=60, command=lambda: self.on_view(self.source)
            ).pack(side="left", padx=(0, Sizes.SM))
        
        create_button(
            actions, "⟳", "secondary", "small",
            width=32, command=self.reconnect
        ).pack(side="right")
        
        # Progress bar
        self.labels['progress'] = ctk.CTkProgressBar(self, height=4, progress_color=Colors.SUCCESS)
        self.labels['progress'].pack(fill="x", padx=Sizes.MD, pady=(0, Sizes.SM))
    
    # Bật/tắt tính năng nhận diện khuôn mặt
    def toggle_feature(self, feature: str):
        if not self.camera:
            return
            
        # Get current state
        face_on = self.switches['face'].get() == "on"
        
        # Update feature
        if feature == 'face':
            self.camera.face_enabled = face_on
            log_activity(f"Face Recognition {'enabled' if face_on else 'disabled'} for {self.source}", "info")
            
        # [LOGIC] Master Detection Control
        should_detect = face_on
        current_detect = self.state.is_detection_enabled(self.source)
        
        if should_detect and not current_detect:
            self.state.set_detection(True, self.source)
            log_activity(f"Auto-enabled detection for {self.source}", "success")
        elif not should_detect and current_detect:
            self.state.set_detection(False, self.source)
            log_activity(f"Auto-disabled detection for {self.source} (All features off)", "warning")


    
    def toggle_ir(self):
        if hasattr(self.camera, 'set_ir_enhancement'):
            enabled = self.switches['ir'].get() == "on"
            self.camera.set_ir_enhancement(enabled)
    
    def reconnect(self):
        if self.camera:
            self.camera.force_reconnect()
            log_activity(f"Reconnecting Camera {self.source}", "info")
    
    # Cập nhật hiển thị trạng thái
    def update_status(self):
        if not self.camera:
            return
        
        # Connection status
        connected = self.camera.get_connection_status()
        
        if connected:
            self.labels['status'].configure(text="Online", text_color=Colors.SUCCESS)
            self.labels['dot'].configure(fg_color=Colors.SUCCESS)
            self.labels['progress'].set(1.0)
        else:
            self.labels['status'].configure(text="Offline", text_color=Colors.DANGER)
            self.labels['dot'].configure(fg_color=Colors.DANGER)
            self.labels['progress'].set(0.0)
        
        # Camera info
        if hasattr(self.camera, 'cap') and self.camera.cap:
            fps = int(self.camera.cap.get(cv2.CAP_PROP_FPS))
            w = int(self.camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.labels['fps'].configure(text=f"FPS: {fps}")
            self.labels['res'].configure(text=f"{w}x{h}")
        
        # IR status
        if hasattr(self.camera, 'get_infrared_status'):
            is_ir = self.camera.get_infrared_status()
            self.labels['ir'].configure(
                text="IR: ON" if is_ir else "IR: OFF",
                text_color=Colors.WARNING if is_ir else Colors.TEXT_MUTED
            )


# Danh sách camera với các điều khiển
class CameraList(ctk.CTkFrame):
    
    
    def __init__(
        self, parent,
        camera_manager,
        state_manager,
        on_view: Callable = None,
        on_add: Callable = None,
        **kwargs
    ):
        super().__init__(parent, fg_color="transparent", **kwargs)
        
        self.camera_manager = camera_manager
        self.state = state_manager
        self.on_view = on_view
        self.on_add = on_add
        self.cards: Dict[str, CameraCard] = {}
        
        self.build_ui()
        self.start_monitor()
    
    def build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=Sizes.SM, pady=(Sizes.SM, 0))
        
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(side="left")
        
        ctk.CTkLabel(title_frame, text="🎥", font=("Segoe UI", 20), text_color=Colors.PRIMARY).pack(side="left", padx=(0, Sizes.SM))
        ctk.CTkLabel(title_frame, text="Cameras", font=Fonts.HEADING, text_color=Colors.TEXT_PRIMARY).pack(side="left")
        
        if self.on_add:
            create_button(header, "+ Add", "primary", "small", width=60, command=self.on_add).pack(side="right")
        
        # Scrollable list
        self.scroll = ctk.CTkScrollableFrame(self, fg_color=Colors.BG_PRIMARY, corner_radius=Sizes.RADIUS_MD)
        self.scroll.pack(fill="both", expand=True, padx=Sizes.SM, pady=Sizes.SM)
        
        # Create camera cards
        if self.camera_manager:
            for source, camera in self.camera_manager.cameras.items():
                card = CameraCard(
                    self.scroll, source, camera, self.state,
                    on_view=self.on_view
                )
                card.pack(fill="x", padx=Sizes.SM, pady=Sizes.XS)
                self.cards[source] = card
    
    # Bắt đầu vòng lặp theo dõi trạng thái
    def start_monitor(self):
        self.update_status()
    
    # Cập nhật trạng thái tất cả camera
    def update_status(self):
        try:
            for card in self.cards.values():
                card.update_status()
        except Exception as e:
            print(f"Status update error: {e}")
        
        self.after(2000, self.update_status)
    
    # Thêm thẻ camera mới
    def add_camera(self, source: str, camera):
        if source not in self.cards:
            card = CameraCard(
                self.scroll, source, camera, self.state,
                on_view=self.on_view
            )
            card.pack(fill="x", padx=Sizes.SM, pady=Sizes.XS)
            self.cards[source] = card

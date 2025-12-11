"""Camera-related widgets"""

from __future__ import annotations
import cv2
import threading
from typing import Callable, Dict, Optional

import customtkinter as ctk
from customtkinter import StringVar

from gui.styles import Colors, Fonts, Sizes, create_button, create_card, create_switch
from gui.widgets.activity import log_activity


class CameraCard(ctk.CTkFrame):
    """Individual camera control card"""
    
    __slots__ = ('source', 'camera', 'state', 'labels', 'switches', 'on_view')
    
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
        
        ctk.CTkLabel(
            header, text=f"📹 Camera {self.source}",
            font=Fonts.BODY_BOLD, text_color=Colors.TEXT_PRIMARY
        ).pack(side="left")
        
        # Detection switch
        det_var = StringVar(value="on" if self.state.is_detection_enabled(self.source) else "off")
        self.switches['detect'] = det_var
        
        create_switch(
            header, "Detect", det_var,
            command=lambda: self.toggle_detection()
        ).pack(side="right")
        
        # Status dot
        self.labels['dot'] = ctk.CTkFrame(header, width=10, height=10, corner_radius=5)
        self.labels['dot'].pack(side="right", padx=Sizes.MD)
        
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
    
    def toggle_detection(self):
        enabled = self.switches['detect'].get() == "on"
        self.state.set_detection(enabled, self.source)
        log_activity(f"Detection {'enabled' if enabled else 'disabled'} for Camera {self.source}",
                    "success" if enabled else "warning")
    
    def toggle_ir(self):
        if hasattr(self.camera, 'set_ir_enhancement'):
            enabled = self.switches['ir'].get() == "on"
            self.camera.set_ir_enhancement(enabled)
    
    def reconnect(self):
        if self.camera:
            self.camera.force_reconnect()
            log_activity(f"Reconnecting Camera {self.source}", "info")
    
    def update_status(self):
        """Update status display"""
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


class CameraList(ctk.CTkFrame):
    """Camera list with controls"""
    
    __slots__ = ('camera_manager', 'state', 'on_view', 'on_add', 'cards', 'scroll')
    
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
    
    def start_monitor(self):
        """Start status monitoring loop"""
        self.update_status()
    
    def update_status(self):
        """Update all camera statuses"""
        try:
            for card in self.cards.values():
                card.update_status()
        except Exception as e:
            print(f"Status update error: {e}")
        
        self.after(2000, self.update_status)
    
    def add_camera(self, source: str, camera):
        """Add new camera card"""
        if source not in self.cards:
            card = CameraCard(
                self.scroll, source, camera, self.state,
                on_view=self.on_view
            )
            card.pack(fill="x", padx=Sizes.SM, pady=Sizes.XS)
            self.cards[source] = card

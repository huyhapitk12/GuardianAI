"""Cameras panel"""

from __future__ import annotations
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


class CamerasPanel(ctk.CTkFrame):
    """Live camera view and controls"""
    
    __slots__ = (
        'camera_manager', 'state', 'selected_camera', 'video_label',
        'camera_list', 'brightness', '_running'
    )
    
    def __init__(self, parent, camera_manager, state_manager, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        
        self.camera_manager = camera_manager
        self.state = state_manager
        self.selected_camera = StringVar()
        self.brightness = 1.0
        self._running = True
        
        # Get camera sources
        sources = list(camera_manager.cameras.keys()) if camera_manager else []
        if sources:
            self.selected_camera.set(sources[0])
        
        self._build_ui()
        self._start_video_loop()
    
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Video panel
        video_panel = create_card(self)
        video_panel.grid(row=0, column=0, sticky="nsew", padx=Sizes.SM, pady=Sizes.SM)
        
        self.video_label = ctk.CTkLabel(
            video_panel,
            text="📹 Select a camera",
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
            actions_card, text="Quick Actions",
            font=Fonts.BODY_BOLD, text_color=Colors.TEXT_SECONDARY
        ).pack(anchor="w", padx=Sizes.MD, pady=(Sizes.MD, Sizes.SM))
        
        btn_frame = ctk.CTkFrame(actions_card, fg_color="transparent")
        btn_frame.pack(fill="x", padx=Sizes.MD, pady=(0, Sizes.MD))
        btn_frame.grid_columnconfigure((0, 1), weight=1)
        
        create_button(
            btn_frame, "⏺️ Record", "success", "small",
            command=self._toggle_record
        ).grid(row=0, column=0, padx=(0, Sizes.XS), sticky="ew")
        
        create_button(
            btn_frame, "📸 Snap", "secondary", "small",
            command=self._take_snapshot
        ).grid(row=0, column=1, padx=(Sizes.XS, 0), sticky="ew")
        
        # Camera list
        list_card = create_card(control_panel)
        list_card.pack(fill="both", expand=True)
        
        self.camera_list = CameraList(
            list_card,
            self.camera_manager,
            self.state,
            on_view=self._select_camera,
            on_add=self._add_camera
        )
        self.camera_list.pack(fill="both", expand=True, padx=Sizes.SM, pady=Sizes.SM)
    
    def _start_video_loop(self):
        """Start video update loop"""
        self._update_video()
    
    def _update_video(self):
        """Update video frame"""
        if not self._running:
            return
        
        selected = self.selected_camera.get()
        if not selected:
            self.after(100, self._update_video)
            return
        
        cam = self.camera_manager.get_camera(selected) if self.camera_manager else None
        if not cam:
            self.video_label.configure(text=f"❌ Camera not found: {selected}", image=None)
            self.after(1000, self._update_video)
            return
        
        try:
            ret, frame = cam.read()
            
            if ret and frame is not None:
                # Resize with high-quality interpolation
                target_w, target_h = Sizes.VIDEO_WIDTH, Sizes.VIDEO_HEIGHT
                frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Convert
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Brightness
                if self.brightness != 1.0:
                    pil_img = ImageEnhance.Brightness(pil_img).enhance(self.brightness)
                
                ctk_img = CTkImage(pil_img, size=(target_w, target_h))
                
                if self.video_label.winfo_exists():
                    self.video_label.configure(text="", image=ctk_img)
                    self.video_label.image = ctk_img
            else:
                if self.video_label.winfo_exists():
                    self.video_label.configure(text="🔄 Reconnecting...", image=None)
                    
        except Exception as e:
            print(f"Video error: {e}")
            if self.video_label.winfo_exists():
                self.video_label.configure(text="❌ Feed error", image=None)
        
        # Schedule next update
        refresh_rate = 33 if self.state.is_detection_enabled() else 50
        self.after(refresh_rate, self._update_video)
    
    def _select_camera(self, source: str):
        """Select camera to view"""
        self.selected_camera.set(source)
        log_activity(f"Viewing camera: {source}", "info")
    
    def _add_camera(self):
        """Show add camera dialog"""
        from gui.dialogs import AddCameraDialog
        
        def on_success(source_id):
            """Called when camera is added successfully"""
            camera = self.camera_manager.get_camera(source_id)
            if camera and self.camera_list:
                self.camera_list.add_camera(source_id, camera)
                # Auto-select the new camera
                self._select_camera(source_id)
        
        dialog = AddCameraDialog(self, self.camera_manager, on_success=on_success)
        dialog.grab_set()
    
    def _take_snapshot(self):
        """Take snapshot from current camera"""
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
        
        log_activity(f"Snapshot saved: {img_path.name}", "success")
    
    def _toggle_record(self):
        """Toggle recording"""
        log_activity("Recording toggled", "info")
    
    def stop(self):
        """Stop video loop"""
        self._running = False
    
    def destroy(self):
        self.stop()
        super().destroy()
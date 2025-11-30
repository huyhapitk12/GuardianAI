# gui/control_panels.py

from __future__ import annotations

import threading
from typing import Any, List, Dict, Optional
from pathlib import Path
from datetime import datetime
import os
import platform

from customtkinter import (
    CTkFrame,
    CTkLabel,
    CTkScrollableFrame,
    CTkSwitch,
    StringVar,
    IntVar,
    CTkSlider,
    CTkTextbox,
    CTkEntry,
    CTkProgressBar,
    CTkOptionMenu,
)
from CTkMessagebox import CTkMessagebox

from .styles import (
    Colors, 
    Fonts, 
    Sizes, 
    create_card_frame, 
    create_modern_button, 
)
from config.settings import settings, update_config_value
from core.recorder import Recorder

# ============================================================================
# SETTINGS PANEL
# ============================================================================

class SettingsPanel(CTkFrame):
    """Bảng cài đặt hệ thống hiện đại với Sidebar"""

    def __init__(self, parent, state_manager=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.state_manager = state_manager
        
        # Layout: Sidebar (Left) | Content (Right)
        self.grid_columnconfigure(0, weight=0) # Sidebar fixed width
        self.grid_columnconfigure(1, weight=1) # Content expands
        self.grid_rowconfigure(0, weight=1)

        self._create_sidebar()
        self._create_content_area()
        
        # Select first tab by default
        self.after(100, lambda: self._select_tab("ai_models"))

    def _create_sidebar(self):
        """Create sidebar with navigation buttons"""
        self.sidebar = CTkFrame(self, fg_color=Colors.BG_SECONDARY, width=200, corner_radius=Sizes.CORNER_RADIUS)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, Sizes.PADDING_SM), pady=0)
        self.sidebar.grid_propagate(False)
        
        CTkLabel(
            self.sidebar, 
            text="CÀI ĐẶT", 
            font=Fonts.BODY_BOLD, 
            text_color=Colors.TEXT_MUTED
        ).pack(anchor="w", padx=Sizes.PADDING_MD, pady=(Sizes.PADDING_MD, Sizes.PADDING_SM))

        self.nav_buttons = {}
        
        tabs = [
            ("ai_models", "🤖 AI & Models"),
            ("detection", "🎯 Detection"),
            ("security", "🛡️ Security"),
            ("alarm", "🔊 Alarm & Sound"),
            ("assistant", "🧠 AI Assistant"),
        ]
        
        for key, text in tabs:
            btn = create_modern_button(
                self.sidebar,
                text=text,
                variant="ghost",
                height=40,
                anchor="w",
                command=lambda k=key: self._select_tab(k)
            )
            btn.pack(fill="x", padx=Sizes.PADDING_SM, pady=2)
            self.nav_buttons[key] = btn

    def _create_content_area(self):
        """Create area to hold settings pages"""
        self.content = CTkFrame(self, fg_color="transparent")
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)
        
        self.pages = {}
        
        # Initialize all pages
        self.pages["ai_models"] = self._create_page_frame(self._build_models_page)
        self.pages["detection"] = self._create_page_frame(self._build_detection_page)
        self.pages["security"] = self._create_page_frame(self._build_security_page)
        self.pages["alarm"] = self._create_page_frame(self._build_alarm_page)
        self.pages["assistant"] = self._create_page_frame(self._build_assistant_page)

    def _create_page_frame(self, build_func):
        """Helper to create a scrollable page frame"""
        frame = CTkScrollableFrame(self.content, fg_color="transparent")
        build_func(frame)
        return frame

    def _select_tab(self, key):
        """Switch visible page"""
        # Update buttons style
        for k, btn in self.nav_buttons.items():
            if k == key:
                btn.configure(fg_color=Colors.BG_TERTIARY, text_color=Colors.PRIMARY)
            else:
                btn.configure(fg_color="transparent", text_color=Colors.TEXT_PRIMARY)
        
        # Show selected page
        for k, page in self.pages.items():
            if k == key:
                page.grid(row=0, column=0, sticky="nsew")
            else:
                page.grid_forget()

    # ================= PAGE BUILDERS =================

    def _build_models_page(self, parent):
        self._create_section_header(parent, "Cấu hình AI & Models", "Quản lý các mô hình nhận diện và xử lý ảnh.")
        
        fields = [
            {"label": "Mô hình khuôn mặt", "key": "models.face_model_name", "type": "str", "options": ["buffalo_l", "buffalo_s"]},
            {"label": "Bộ phát hiện (Detector)", "key": "models.face.detector_name", "type": "str", "options": ["buffalo_l", "buffalo_s"]},
            {"label": "Bộ nhận diện (Recognizer)", "key": "models.face.recognizer_name", "type": "str", "options": ["buffalo_l", "buffalo_s"]},
            {"label": "Kích thước YOLO", "key": "models.yolo_size", "type": "str", "options": ["small", "medium", "large"]},
            {"label": "Định dạng YOLO", "key": "models.yolo_format", "type": "str", "options": ["openvino", "onnx", "pytorch"]},
            {"label": "InsightFace Context ID", "key": "models.insightface_ctx_id", "type": "int", "min": -1, "max": 10},
        ]
        self._create_settings_group(parent, "Model Selection", fields)

    def _build_detection_page(self, parent):
        self._create_section_header(parent, "Cấu hình Detection", "Tinh chỉnh các tham số phát hiện.")
        
        fields = [
            {"label": "Ngưỡng tin cậy (Confidence)", "key": "detection.person_confidence_threshold", "type": "float", "min": 0.0, "max": 1.0},
            {"label": "Ngưỡng IOU (Tracking)", "key": "detection.iou_threshold", "type": "float", "min": 0.0, "max": 1.0},
        ]
        self._create_settings_group(parent, "Thresholds", fields)

    def _build_security_page(self, parent):
        self._create_section_header(parent, "Bảo mật & Riêng tư", "Quản lý các tính năng an ninh.")
        
        card = create_card_frame(parent, fg_color=Colors.BG_SECONDARY)
        card.pack(fill="x", pady=(0, Sizes.PADDING_MD))
        
        self._add_switch(card, "Phát hiện chuyển động", "security.motion_detection", "on")
        self._add_switch(card, "Làm mờ khuôn mặt (Privacy)", "security.face_blur", "off")
        self._add_switch(card, "Tự động khóa khi rảnh", "security.auto_lock", "on")
        self._add_switch(card, "Xác thực 2 lớp (2FA)", "security.2fa", "off")
        
        # Sensitivity Slider
        slider_frame = CTkFrame(card, fg_color="transparent")
        slider_frame.pack(fill="x", padx=Sizes.PADDING_MD, pady=(Sizes.PADDING_MD, Sizes.PADDING_MD))
        CTkLabel(slider_frame, text="Độ nhạy chuyển động", font=Fonts.BODY).pack(anchor="w")
        CTkSlider(slider_frame, from_=0, to=100, progress_color=Colors.PRIMARY).pack(fill="x", pady=(5, 0))

    def _build_alarm_page(self, parent):
        self._create_section_header(parent, "Cảnh báo & Âm thanh", "Cấu hình âm báo và phản ứng.")
        
        fields = [
            {"label": "Độ nhạy cảnh báo người", "key": "alarm.person_sensitivity", "type": "float", "min": 0.0, "max": 1.0},
            {"label": "Thời gian chờ (giây)", "key": "alarm.cooldown_seconds", "type": "int", "min": 0, "max": 3600},
            {"label": "Âm lượng tối đa (0.0 - 1.0)", "key": "alarm.max_volume", "type": "float", "min": 0.0, "max": 1.0},
        ]
        self._create_settings_group(parent, "Alarm Settings", fields)

    def _build_assistant_page(self, parent):
        self._create_section_header(parent, "Trợ lý ảo AI", "Cấu hình LLM và phản hồi thông minh.")
        
        fields = [
            {"label": "Dùng LLM phân loại", "key": "ai.use_llm_for_classification", "type": "bool"},
            {"label": "Nhiệt độ (Sáng tạo)", "key": "ai.temperature", "type": "float", "min": 0.0, "max": 2.0},
            {"label": "Số token tối đa", "key": "ai.max_tokens", "type": "int", "min": 16, "max": 8192},
        ]
        self._create_settings_group(parent, "LLM Configuration", fields)

    # ================= HELPERS =================

    def _create_section_header(self, parent, title, subtitle):
        frame = CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=(0, Sizes.PADDING_MD))
        CTkLabel(frame, text=title, font=Fonts.TITLE_MD, text_color=Colors.TEXT_PRIMARY).pack(anchor="w")
        CTkLabel(frame, text=subtitle, font=Fonts.BODY, text_color=Colors.TEXT_MUTED).pack(anchor="w")

    def _add_switch(self, parent, text, key, default):
        frame = CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=Sizes.PADDING_MD, pady=5)
        
        CTkLabel(frame, text=text, font=Fonts.BODY, text_color=Colors.TEXT_PRIMARY).pack(side="left")
        
        val = settings.get(key, default) == "on"
        var = StringVar(value="on" if val else "off")
        
        CTkSwitch(
            frame, text="", variable=var, onvalue="on", offvalue="off", 
            progress_color=Colors.PRIMARY,
            command=lambda k=key, v=var: update_config_value(k, v.get() == "on")
        ).pack(side="right")

    def _create_settings_group(self, parent, title, fields):
        card = create_card_frame(parent, fg_color=Colors.BG_SECONDARY)
        card.pack(fill="x", pady=(0, Sizes.PADDING_MD))
        
        CTkLabel(card, text=title, font=Fonts.BODY_BOLD, text_color=Colors.PRIMARY_LIGHT).pack(anchor="w", padx=Sizes.PADDING_MD, pady=(Sizes.PADDING_MD, Sizes.PADDING_SM))
        
        grid = CTkFrame(card, fg_color="transparent")
        grid.pack(fill="x", padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_MD))
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=0)
        
        prepared_fields = []
        
        for i, field in enumerate(fields):
            val = settings.get(field["key"])
            
            # Label
            CTkLabel(grid, text=field["label"], font=Fonts.BODY, anchor="w").grid(row=i, column=0, sticky="ew", pady=5)
            
            # Input
            widget = None
            info = {"key": field["key"], "type": field["type"], "min": field.get("min"), "max": field.get("max")}
            
            if field["type"] == "bool":
                var = StringVar(value="on" if bool(val) else "off")
                widget = CTkSwitch(grid, text="", variable=var, onvalue="on", offvalue="off", progress_color=Colors.PRIMARY, width=50)
                widget.grid(row=i, column=1, sticky="e")
                info["var"] = var
            
            elif field["type"] == "str" and "options" in field:
                var = StringVar(value=str(val) if val else field["options"][0])
                widget = CTkOptionMenu(grid, variable=var, values=field["options"], width=150, fg_color=Colors.BG_TERTIARY)
                widget.grid(row=i, column=1, sticky="e")
                info["var"] = var
                
            else:
                widget = CTkEntry(grid, width=150, fg_color=Colors.BG_TERTIARY, border_color=Colors.BORDER)
                if val is not None: widget.insert(0, str(val))
                widget.grid(row=i, column=1, sticky="e")
            
            info["widget"] = widget
            prepared_fields.append(info)
            
        # Save Button
        create_modern_button(
            card, text="Lưu thay đổi", width=120, height=30,
            command=lambda: self._save_group(prepared_fields, title)
        ).pack(anchor="e", padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_MD))

    def _save_group(self, items, group_name):
        try:
            for item in items:
                key = item["key"]
                ftype = item["type"]
                
                new_val = None
                if ftype == "bool":
                    new_val = item["var"].get() == "on"
                elif ftype == "str" and "var" in item:
                    new_val = item["var"].get()
                else:
                    raw = item["widget"].get().strip()
                    if not raw: continue
                    if ftype == "int": new_val = int(raw)
                    elif ftype == "float": new_val = float(raw)
                    else: new_val = raw
                    
                    # Validate
                    if item["min"] is not None and new_val < item["min"]: raise ValueError(f"{key} too small")
                    if item["max"] is not None and new_val > item["max"]: raise ValueError(f"{key} too large")
                
                update_config_value(key, new_val)
                
        except Exception as e:
            CTkMessagebox(title="Error", message=f"Failed to save settings: {e}", icon="cancel")
        else:
            CTkMessagebox(title="Success", message=f"{group_name} saved successfully!", icon="check")

class RecordingPanel(CTkFrame):
    """Panel điều khiển ghi hình"""
    
    def __init__(self, parent, camera_manager, recorder=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        
        self.camera_manager = camera_manager
        self.recorder = recorder if recorder is not None else Recorder()
        
        self.is_recording = False
        self.recording_start = None
        
        self.selected_camera = StringVar(value="")
        self.duration_var = IntVar(value=60)
        self.progress = IntVar(value=0)  # Progress as percentage (0-100)
        
        self._create_ui()
        self._update_camera_list()
        self._start_update_loop()
        
    def _create_ui(self):
        card = create_card_frame(self, fg_color=Colors.BG_SECONDARY)
        card.pack(fill="both", expand=True)
        
        # Header
        header = CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)
        
        CTkLabel(
            header, 
            text="🎥 Recording Control", 
            font=Fonts.TITLE_SM, 
            text_color=Colors.TEXT_PRIMARY
        ).pack(side="left")
        
        self.status_label = CTkLabel(
            header,
            text="⭕ Not Recording",
            font=Fonts.BODY_BOLD,
            text_color=Colors.TEXT_SECONDARY
        )
        self.status_label.pack(side="right")
        
        # Controls
        controls = CTkFrame(card, fg_color="transparent")
        controls.pack(fill="x", padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_MD))
        
        # Camera selection
        self.camera_menu = CTkOptionMenu(
            controls,
            variable=self.selected_camera,
            values=["No cameras"],
            font=Fonts.BODY,
            width=150
        )
        self.camera_menu.pack(side="left", padx=(0, Sizes.PADDING_SM))
        
        # Duration selection
        duration_menu = CTkOptionMenu(
            controls,
            values=["30s", "60s", "120s", "300s"],
            command=self._on_duration_changed,
            font=Fonts.BODY,
            width=100
        )
        duration_menu.set("60s")
        duration_menu.pack(side="left", padx=Sizes.PADDING_SM)
        
        # Buttons
        self.record_btn = create_modern_button(
            controls,
            text="Start Recording",
            variant="danger",
            icon="⏺️",
            command=self._toggle_recording
        )
        self.record_btn.pack(side="left", padx=Sizes.PADDING_SM)
        
        create_modern_button(
            controls,
            text="Open Folder",
            variant="secondary",
            icon="📁",
            command=self._open_recordings_folder
        ).pack(side="left", padx=Sizes.PADDING_SM)
        
        # Progress bar
        self.progress_bar = CTkProgressBar(card, variable=self.progress)
        self.progress_bar.pack(fill="x", padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_SM))
        self.progress_bar.set(0)
        
        # Info label
        self.info_label = CTkLabel(
            card,
            text="",
            font=Fonts.SMALL,
            text_color=Colors.TEXT_MUTED
        )
        self.info_label.pack(pady=(0, Sizes.PADDING_MD))

    def _update_camera_list(self):
        """Cập nhật danh sách camera"""
        if self.camera_manager and self.camera_manager.cameras:
            cameras = list(self.camera_manager.cameras.keys())
            self.camera_menu.configure(values=cameras)
            if cameras and not self.selected_camera.get():
                self.selected_camera.set(cameras[0])
        else:
            self.camera_menu.configure(values=["No cameras"])
    
    def _on_duration_changed(self, choice):
        """Handle duration change"""
        duration = int(choice.replace('s', ''))
        self.duration_var.set(duration)
    
    def _toggle_recording(self):
        """Toggle recording"""
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()
    
    def _start_recording(self):
        """Start recording"""
        camera_source = self.selected_camera.get()
        if not camera_source or camera_source == "No cameras":
            print("WARNING: No camera selected for recording")
            return
        
        duration = self.duration_var.get()
        
        rec = self.recorder.start(
            source_id=camera_source,
            reason="manual",
            duration=duration,
            wait_for_user=False
        )
        
        if rec:
            self.is_recording = True
            self.recording_start = datetime.now()
            self.record_btn.configure(text="Stop Recording", fg_color=Colors.BG_ELEVATED)
            self.status_label.configure(text="🔴 Recording...", text_color=Colors.DANGER)
            self.info_label.configure(text=f"Recording {camera_source} for {duration}s...")
    
    def _stop_recording(self):
        """Stop recording"""
        self.recorder.stop()
        self.is_recording = False
        self.recording_start = None
        self.record_btn.configure(text="Start Recording", fg_color=Colors.DANGER)
        self.status_label.configure(text="⭕ Not Recording", text_color=Colors.TEXT_SECONDARY)
        self.info_label.configure(text="Recording stopped")
        self.progress.set(0)
        
    def _open_recordings_folder(self):
        """Open recordings folder"""
        path = settings.paths.recordings_dir
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            os.system(f"open {path}")
        else:
            os.system(f"xdg-open {path}")

    def _start_update_loop(self):
        """Update progress loop"""
        if self.is_recording and self.recording_start:
            elapsed = (datetime.now() - self.recording_start).total_seconds()
            duration = self.duration_var.get()
            
            if elapsed >= duration:
                self._stop_recording()
            else:
                prog = elapsed / duration
                self.progress.set(prog)
                self.info_label.configure(text=f"Recording... {int(elapsed)}/{duration}s")
        
        self.after(100, self._start_update_loop)
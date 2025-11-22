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
    """Bảng cài đặt hệ thống"""

    def __init__(self, parent, state_manager=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.state_manager = state_manager

        # Grid layout: 1 cột chính
        self.grid_columnconfigure(0, weight=1)
        # Row 0: Header, Row 1: Summary, Row 2: Settings List
        self.grid_rowconfigure(2, weight=1)

        self.sections: List[List[Dict[str, Any]]] = []

        # Biến cho switch tổng
        self.global_detection_var = StringVar(
            value="on" if bool(settings.get("detection.global_enabled", True)) else "off"
        )
        self.global_ai_var = StringVar(
            value="on" if bool(settings.get("ai.enabled", False)) else "off"
        )

        self._build_header()
        self._build_summary_card()
        self._build_body()
        self._update_summary()

    # ------------------------------------------------------------------ UI: Header & Summary
    def _build_header(self) -> None:
        header_frame = CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 10))

        title = CTkLabel(
            header_frame,
            text="⚙️ Cài đặt hệ thống",
            font=Fonts.TITLE_SM,
            text_color=Colors.TEXT_PRIMARY,
        )
        title.pack(side="top", anchor="w")

        subtitle = CTkLabel(
            header_frame,
            text="Điều chỉnh ngưỡng phát hiện, cảnh báo và cấu hình AI.",
            font=Fonts.SMALL,
            text_color=Colors.TEXT_SECONDARY,
        )
        subtitle.pack(side="top", anchor="w")

    def _build_summary_card(self) -> None:
        card = create_card_frame(self, fg_color=Colors.BG_SECONDARY)
        card.grid(row=1, column=0, sticky="ew", padx=0, pady=(0, 10))
        
        card.grid_columnconfigure(0, weight=1)
        card.grid_columnconfigure(1, weight=0) # Cột cho switch

        # Info bên trái
        left_panel = CTkFrame(card, fg_color="transparent")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)

        self.summary_label = CTkLabel(
            left_panel,
            text="",
            font=Fonts.BODY_BOLD,
            text_color=Colors.TEXT_PRIMARY,
            justify="left",
            anchor="w"
        )
        self.summary_label.pack(anchor="w", fill="x")

        hint = CTkLabel(
            left_panel,
            text="Sử dụng các công tắc bên phải để bật/tắt nhanh các tính năng chính.",
            font=Fonts.SMALL,
            text_color=Colors.TEXT_SECONDARY,
            justify="left",
            anchor="w"
        )
        hint.pack(anchor="w", fill="x", pady=(5, 0))

        # Switches bên phải
        right_panel = CTkFrame(card, fg_color="transparent")
        right_panel.grid(row=0, column=1, sticky="e", padx=15, pady=15)

        self.global_detection_switch = CTkSwitch(
            right_panel,
            text="Nhận diện người",
            font=Fonts.BODY,
            variable=self.global_detection_var,
            onvalue="on",
            offvalue="off",
            progress_color=Colors.PRIMARY,
            command=self._toggle_global_detection,
        )
        self.global_detection_switch.pack(side="left", padx=(0, 15))

        self.global_ai_switch = CTkSwitch(
            right_panel,
            text="Trợ lý AI",
            font=Fonts.BODY,
            variable=self.global_ai_var,
            onvalue="on",
            offvalue="off",
            progress_color=Colors.PRIMARY,
            command=self._toggle_ai_enabled,
        )
        self.global_ai_switch.pack(side="left")

    # ------------------------------------------------------------------ UI: Body & Sections
    def _build_body(self) -> None:
        self.container = CTkScrollableFrame(
            self,
            fg_color="transparent", # Transparent để hòa vào nền
            corner_radius=0,
        )
        self.container.grid(row=2, column=0, sticky="nsew")
        self.container.grid_columnconfigure(0, weight=1)

        # Tạo các nhóm cài đặt
        self._create_models_section(self.container)
        self._create_detection_section(self.container)
        self._create_alarm_section(self.container)
        self._create_ai_section(self.container)

    # --- Các nhóm cài đặt (Giữ nguyên cấu trúc dữ liệu) ---
    def _create_models_section(self, parent) -> None:
        fields = [
            {"label": "Mô hình khuôn mặt", "key": "models.face_model_name", "type": "str", "options": ["buffalo_l", "buffalo_s"]},
            {"label": "Bộ phát hiện (Detector)", "key": "models.face.detector_name", "type": "str", "options": ["buffalo_l", "buffalo_s"]},
            {"label": "Bộ nhận diện (Recognizer)", "key": "models.face.recognizer_name", "type": "str", "options": ["buffalo_l", "buffalo_s"]},
            {"label": "Kích thước YOLO", "key": "models.yolo_size", "type": "str", "options": ["small", "medium", "large"]},
            {"label": "Định dạng YOLO", "key": "models.yolo_format", "type": "str", "options": ["openvino", "onnx", "pytorch"]},
            {"label": "InsightFace Context ID", "key": "models.insightface_ctx_id", "type": "int", "min": -1, "max": 10},
        ]
        self._create_section(parent, "🤖 Cài đặt Mô hình AI", fields)

    def _create_detection_section(self, parent) -> None:
        fields = [
            {"label": "Ngưỡng nhận diện người (0.0 - 1.0)", "key": "detection.person_confidence_threshold", "type": "float", "min": 0.0, "max": 1.0},
            {"label": "Ngưỡng phát hiện cháy (0.0 - 1.0)", "key": "detection.fire_confidence_threshold", "type": "float", "min": 0.0, "max": 1.0},
            {"label": "Ngưỡng khớp khuôn mặt (Thấp = Chặt)", "key": "detection.face_recognition_threshold", "type": "float", "min": 0.0, "max": 2.0},
            {"label": "Ngưỡng IOU (Tracking)", "key": "detection.iou_threshold", "type": "float", "min": 0.0, "max": 1.0},
        ]
        self._create_section(parent, "🎯 Cấu hình Phát hiện", fields)

    def _create_alarm_section(self, parent) -> None:
        fields = [
            {"label": "Độ nhạy cảnh báo người", "key": "alarm.person_sensitivity", "type": "float", "min": 0.0, "max": 1.0},
            {"label": "Thời gian chờ giữa các cảnh báo (giây)", "key": "alarm.cooldown_seconds", "type": "int", "min": 0, "max": 3600},
            {"label": "Âm lượng tối đa (0.0 - 1.0)", "key": "alarm.max_volume", "type": "float", "min": 0.0, "max": 1.0},
        ]
        self._create_section(parent, "🔊 Cảnh báo & Âm thanh", fields)

    def _create_ai_section(self, parent) -> None:
        fields = [
            {"label": "Dùng LLM phân loại phản hồi", "key": "ai.use_llm_for_classification", "type": "bool"},
            {"label": "Nhiệt độ (Sáng tạo)", "key": "ai.temperature", "type": "float", "min": 0.0, "max": 2.0},
            {"label": "Số token tối đa", "key": "ai.max_tokens", "type": "int", "min": 16, "max": 8192},
        ]
        self._create_section(parent, "🧠 Trợ lý ảo AI", fields)

    # ------------------------------------------------------------------ Helpers tạo UI
    def _create_section(self, parent: CTkFrame, title: str, fields: List[Dict[str, Any]]) -> None:
        # Card chứa section
        card = create_card_frame(parent, fg_color=Colors.BG_SECONDARY)
        card.pack(fill="x", padx=0, pady=(0, 15))
        
        # Tiêu đề section
        header = CTkLabel(
            card, 
            text=title, 
            font=Fonts.BODY_BOLD, 
            text_color=Colors.PRIMARY_LIGHT
        )
        header.pack(anchor="w", padx=15, pady=(15, 10))

        # Grid layout cho các fields: Cột 0 là Label, Cột 1 là Input
        grid_frame = CTkFrame(card, fg_color="transparent")
        grid_frame.pack(fill="x", padx=15, pady=(0, 15))
        grid_frame.grid_columnconfigure(0, weight=1) # Label chiếm phần lớn
        grid_frame.grid_columnconfigure(1, weight=0) # Input kích thước cố định

        prepared_fields = []
        for idx, field in enumerate(fields):
            value = settings.get(field["key"])
            prepared = self._create_field(grid_frame, idx, field, value)
            prepared_fields.append(prepared)

        # Nút lưu
        save_btn = create_modern_button(
            card,
            text="Lưu thay đổi nhóm này",
            command=lambda items=prepared_fields, s=title: self._save_fields(items, s),
            width=200,
            height=35
        )
        save_btn.pack(anchor="e", padx=15, pady=(0, 15))
        
        self.sections.append(prepared_fields)

    def _create_field(self, parent: CTkFrame, row: int, field: Dict[str, Any], value: Any) -> Dict[str, Any]:
        """Tạo một dòng cài đặt: Label bên trái, Input bên phải."""
        
        # Label
        label = CTkLabel(
            parent,
            text=field["label"],
            font=Fonts.BODY,
            text_color=Colors.TEXT_PRIMARY,
            anchor="w"
        )
        label.grid(row=row, column=0, sticky="ew", pady=8, padx=(0, 20))

        field_info = {
            "key": field["key"],
            "type": field["type"],
            "min": field.get("min"),
            "max": field.get("max"),
            "widget": None,
        }

        # Widget Input
        ftype = field["type"]
        widget = None

        if ftype == "bool":
            # Switch
            var = StringVar(value="on" if bool(value) else "off")
            widget = CTkSwitch(
                parent,
                text="",
                variable=var,
                onvalue="on",
                offvalue="off",
                progress_color=Colors.PRIMARY,
                width=50
            )
            widget.grid(row=row, column=1, sticky="e")
            field_info["var"] = var

        elif ftype == "str" and "options" in field:
            # Dropdown
            var = StringVar(value=str(value) if value else field["options"][0])
            widget = CTkOptionMenu(
                parent,
                variable=var,
                values=field["options"],
                font=Fonts.SMALL,
                fg_color=Colors.BG_TERTIARY,
                button_color=Colors.PRIMARY,
                button_hover_color=Colors.PRIMARY_HOVER,
                width=150,
                height=30
            )
            widget.grid(row=row, column=1, sticky="e")
            field_info["var"] = var

        else:
            # Text Entry (Int/Float)
            # Sử dụng CTkEntry trực tiếp để kiểm soát màu nền tốt hơn
            entry_bg = Colors.BG_TERTIARY
            if entry_bg == Colors.BG_SECONDARY: # Nếu màu trùng nhau thì làm sáng hơn chút
                entry_bg = "#2C3E50" # Fallback dark color

            widget = CTkEntry(
                parent,
                placeholder_text="Nhập giá trị...",
                font=Fonts.BODY,
                fg_color=entry_bg, 
                border_color=Colors.BORDER,
                text_color=Colors.TEXT_PRIMARY,
                width=150,
                height=30
            )
            if value is not None:
                widget.insert(0, str(value))
            
            widget.grid(row=row, column=1, sticky="e")

        field_info["widget"] = widget
        return field_info

    # ------------------------------------------------------------------ Logic Lưu & Cập nhật
    def _save_fields(self, items: List[Dict[str, Any]], section_name: str) -> None:
        updated_keys = []
        try:
            for item in items:
                key = item["key"]
                ftype = item["type"]
                widget = item["widget"]
                
                new_value = None

                if ftype == "bool":
                    var = item["var"]
                    new_value = (var.get() == "on")
                elif ftype == "str" and "var" in item: # OptionMenu
                    new_value = item["var"].get()
                else: # Entry
                    raw_text = widget.get().strip()
                    if raw_text == "": continue
                    
                    if ftype == "int":
                        new_value = int(raw_text)
                    elif ftype == "float":
                        new_value = float(raw_text)
                    else:
                        new_value = raw_text

                    # Validate min/max
                    vmin, vmax = item.get("min"), item.get("max")
                    if vmin is not None and new_value < vmin:
                        raise ValueError(f"{key}: Giá trị {new_value} nhỏ hơn mức tối thiểu {vmin}")
                    if vmax is not None and new_value > vmax:
                        raise ValueError(f"{key}: Giá trị {new_value} lớn hơn mức tối đa {vmax}")

                # Update setting
                update_config_value(key, new_value)
                updated_keys.append(key)

            CTkMessagebox(title="Đã lưu", message=f"Đã cập nhật thành công nhóm:\n{section_name}", icon="check")
            self._update_summary() # Cập nhật lại bảng tóm tắt phía trên

        except Exception as e:
            print(f"ERROR saving settings: {e}")
            CTkMessagebox(title="Lỗi", message=f"Lỗi khi lưu cài đặt:\n{str(e)}", icon="cancel")

    def _toggle_global_detection(self) -> None:
        enabled = (self.global_detection_var.get() == "on")
        update_config_value("detection.global_enabled", enabled)
        if self.state_manager:
            self.state_manager.set_person_detection_enabled(enabled)
        self._update_summary()

    def _toggle_ai_enabled(self) -> None:
        enabled = (self.global_ai_var.get() == "on")
        update_config_value("ai.enabled", enabled)
        self._update_summary()

    def _update_summary(self) -> None:
        # Lấy giá trị mới nhất từ settings
        det_on = bool(settings.get("detection.global_enabled", True))
        ai_on = bool(settings.get("ai.enabled", False))
        
        p_thr = settings.get("detection.person_confidence_threshold", 0.6)
        f_thr = settings.get("detection.fire_confidence_threshold", 0.6)
        temp = settings.get("ai.temperature", 0.7)

        # Sync switches
        self.global_detection_var.set("on" if det_on else "off")
        self.global_ai_var.set("on" if ai_on else "off")

        # Text summary
        text = f"• Phát hiện: {'ĐANG BẬT' if det_on else 'TẮT'} (Người > {p_thr} | Cháy > {f_thr})\n"
        text += f"• AI: {'ĐANG BẬT' if ai_on else 'TẮT'} (Nhiệt độ: {temp})\n"
        
        self.summary_label.configure(text=text)


# ============================================================================
# SECURITY PANEL
# ============================================================================

class SecurityPanel(CTkFrame):
    """Panel cài đặt bảo mật"""
    
    def __init__(self, parent, state_manager, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        
        self.state = state_manager
        
        # Variables
        self.motion_detection_var = StringVar(value="on")
        self.face_blur_var = StringVar(value="off")
        self.auto_lock_var = StringVar(value="on")
        self.two_factor_var = StringVar(value="off")
        self.sensitivity_var = IntVar(value=70)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        self._create_content()
        
    def _create_content(self):
        card = create_card_frame(self, fg_color=Colors.BG_SECONDARY)
        card.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        
        CTkLabel(card, text="🛡️ Security Settings", font=Fonts.TITLE_SM, text_color=Colors.TEXT_PRIMARY).pack(anchor="w", padx=20, pady=(20, 10))
        
        self._create_switch(card, "Motion Detection", self.motion_detection_var)
        self._create_switch(card, "Face Blurring (Privacy)", self.face_blur_var)
        self._create_switch(card, "Auto Lock on Idle", self.auto_lock_var)
        self._create_switch(card, "Two-Factor Auth", self.two_factor_var)
        
        CTkLabel(card, text="Motion Sensitivity", font=Fonts.BODY, text_color=Colors.TEXT_PRIMARY).pack(anchor="w", padx=20, pady=(10, 5))
        slider = CTkSlider(card, from_=0, to=100, variable=self.sensitivity_var, progress_color=Colors.PRIMARY)
        slider.pack(fill="x", padx=20, pady=5)

    def _create_switch(self, parent, text, variable):
        frame = CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=5)
        
        CTkLabel(frame, text=text, font=Fonts.BODY, text_color=Colors.TEXT_PRIMARY).pack(side="left")
        CTkSwitch(frame, text="", variable=variable, onvalue="on", offvalue="off", progress_color=Colors.PRIMARY).pack(side="right")


# ============================================================================
# RECORDING PANEL
# ============================================================================

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
            
            self.status_label.configure(
                text="🔴 Recording...",
                text_color=Colors.DANGER
            )
            self.record_btn.configure(text="Stop Recording")
            
            print(f"INFO: Recording started on camera {camera_source} for {duration}s")
    
    def _stop_recording(self):
        """Stop recording"""
        self.recorder.stop_and_discard()
        self.is_recording = False
        self.recording_start = None
        
        self.status_label.configure(
            text="⭕ Not Recording",
            text_color=Colors.TEXT_SECONDARY
        )
        self.record_btn.configure(text="Start Recording")
        self.progress.set(0)
        self.progress_bar.set(0)
        
        print("INFO: Recording stopped manually")
    
    def _open_recordings_folder(self):
        """Open recordings folder"""
        folder_path = self.recorder.out_dir
        
        if platform.system() == 'Windows':
            os.startfile(folder_path)
        elif platform.system() == 'Darwin':  # macOS
            os.system(f'open "{folder_path}"')
        else:  # Linux
            os.system(f'xdg-open "{folder_path}"')
    
    def _start_update_loop(self):
        """Start update loop"""
        def update():
            try:
                if self.is_recording and self.recording_start:
                    elapsed = (datetime.now() - self.recording_start).total_seconds()
                    duration = self.duration_var.get()
                    progress = min(1.0, elapsed / duration)
                    self.progress.set(int(progress * 100))  # Convert to percentage (0-100)
                    self.progress_bar.set(progress)
                    
                    remaining = max(0, duration - elapsed)
                    self.info_label.configure(
                        text=f"Recording: {int(elapsed)}s / {duration}s (Remaining: {int(remaining)}s)"
                    )
                    
                    if self.recorder.current is None:
                        self._stop_recording()
                else:
                    if self.recorder.current:
                        path = self.recorder.current.get('path', 'N/A')
                        self.info_label.configure(text=f"Current: {Path(path).name}")
                    else:
                        self.info_label.configure(text="")
            except Exception as e:
                print(f"ERROR: Recording update error: {e}")
            finally:
                self.after(1000, update)
        
        update()
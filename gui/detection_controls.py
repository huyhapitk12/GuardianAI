# gui/detection_controls.py
import logging
import threading
from customtkinter import CTkFrame, CTkLabel, CTkSwitch, StringVar, CTkScrollableFrame
from .styles import Colors, Fonts, Sizes, create_card_frame, create_modern_button

logger = logging.getLogger(__name__)

class DetectionControlsFrame(CTkFrame):
    """Frame điều khiển chức năng nhận diện và camera - Phiên bản hiện đại"""
    
    def __init__(self, parent, camera_manager, state_manager, **kwargs):
        super().__init__(
            parent,
            fg_color="transparent",
            **kwargs
        )
        self.camera_manager = camera_manager
        self.state = state_manager
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        self._create_header()
        self._create_controls()
        
        # Start update loop for status
        self._update_status_loop()
    
    def _create_header(self):
        """Tạo tiêu đề phần điều khiển"""
        header_frame = CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, Sizes.PADDING_MD))
        
        title = CTkLabel(
            header_frame,
            text="🎥 Điều Khiển Camera & Cảm Biến",
            font=Fonts.TITLE_SM,
            text_color=Colors.TEXT_PRIMARY
        )
        title.pack(side="left", anchor="w")
    
    def _create_controls(self):
        """Tạo các nút điều khiển camera"""
        self.controls_frame = CTkScrollableFrame(
            self,
            fg_color=Colors.BG_SECONDARY,
            border_width=1,
            border_color=Colors.BORDER,
            corner_radius=Sizes.CORNER_RADIUS,
            label_text="Danh Sách Camera",
            label_font=Fonts.BODY_BOLD,
            label_text_color=Colors.TEXT_PRIMARY
        )
        self.controls_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        self.controls_frame.grid_columnconfigure(0, weight=1)
        
        self.camera_widgets = {}
        self._populate_controls()
    
    def _populate_controls(self):
        """Điền các công tắc cho mỗi camera"""
        # Clear existing
        for widget in self.controls_frame.winfo_children():
            widget.destroy()
        self.camera_widgets = {}

        cameras = self.camera_manager.cameras
        
        if not cameras:
            no_camera_label = CTkLabel(
                self.controls_frame,
                text="📭 Không có camera nào",
                font=Fonts.BODY,
                text_color=Colors.TEXT_MUTED
            )
            no_camera_label.pack(pady=Sizes.PADDING_LG)
            return
        
        for idx, (source_id, cam) in enumerate(cameras.items()):
            self._create_camera_control(self.controls_frame, source_id, cam)
    
    def _create_camera_control(self, parent, source_id: str, camera):
        """Tạo điều khiển cho một camera"""
        cam_frame = create_card_frame(parent, fg_color=Colors.BG_TERTIARY)
        cam_frame.pack(fill="x", padx=Sizes.PADDING_SM, pady=Sizes.PADDING_SM)
        cam_frame.grid_columnconfigure(1, weight=1)
        
        # --- COL 0: Icon ---
        icon_label = CTkLabel(
            cam_frame,
            text="📹",
            font=Fonts.TITLE_MD,
            text_color=Colors.PRIMARY
        )
        icon_label.grid(row=0, column=0, rowspan=2, padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD, sticky="nw")
        
        # --- COL 1: Info & Status ---
        info_frame = CTkFrame(cam_frame, fg_color="transparent")
        info_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(0, Sizes.PADDING_MD), pady=Sizes.PADDING_SM)
        
        # Name
        CTkLabel(
            info_frame,
            text=f"Camera {source_id}",
            font=Fonts.BODY_BOLD,
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w")
        
        # Connection Status
        status_label = CTkLabel(
            info_frame,
            text="● Đang kiểm tra...",
            font=Fonts.SMALL,
            text_color=Colors.TEXT_MUTED
        )
        status_label.pack(anchor="w")
        
        # IR Status
        ir_status_label = CTkLabel(
            info_frame,
            text="🌙 IR: Tắt",
            font=Fonts.SMALL,
            text_color=Colors.TEXT_MUTED
        )
        ir_status_label.pack(anchor="w")

        # --- COL 2: Actions (Reconnect) ---
        action_frame = CTkFrame(cam_frame, fg_color="transparent")
        action_frame.grid(row=0, column=2, rowspan=2, sticky="e", padx=Sizes.PADDING_SM)
        
        create_modern_button(
            action_frame,
            text="Kết nối lại",
            width=100,
            height=30,
            variant="secondary",
            command=lambda c=camera: self._reconnect_camera(c)
        ).pack(pady=2)

        # --- COL 3: Toggles ---
        toggles_frame = CTkFrame(cam_frame, fg_color="transparent")
        toggles_frame.grid(row=0, column=3, rowspan=2, sticky="e", padx=Sizes.PADDING_MD)
        
        # Person Detection Switch
        person_var = StringVar(value="on" if self.state.is_person_detection_enabled(source_id) else "off")
        person_switch = CTkSwitch(
            toggles_frame,
            text="Phát hiện người",
            variable=person_var,
            onvalue="on",
            offvalue="off",
            font=Fonts.BODY,
            progress_color=Colors.PRIMARY,
            command=lambda sid=source_id, var=person_var: self._toggle_person_detection(sid, var)
        )
        person_switch.pack(anchor="e", pady=2)
        
        # IR Enhance Switch
        ir_enhance_var = StringVar(value="on" if getattr(camera, 'ir_enhancement_enabled', False) else "off")
        ir_enhance_switch = CTkSwitch(
            toggles_frame,
            text="Tăng cường IR",
            variable=ir_enhance_var,
            onvalue="on",
            offvalue="off",
            font=Fonts.BODY,
            progress_color=Colors.WARNING,
            command=lambda c=camera, var=ir_enhance_var: self._toggle_ir_enhancement(c, var)
        )
        ir_enhance_switch.pack(anchor="e", pady=2)
        
        self.camera_widgets[source_id] = {
            'camera': camera,
            'status_label': status_label,
            'ir_status_label': ir_status_label,
            'person_var': person_var,
            'ir_enhance_var': ir_enhance_var
        }

    def _toggle_person_detection(self, source_id: str, var: StringVar):
        is_on = var.get() == "on"
        self.state.set_person_detection_enabled(is_on, source_id)
        logger.info(f"Person detection for {source_id}: {is_on}")

    def _toggle_ir_enhancement(self, camera, var: StringVar):
        is_on = var.get() == "on"
        if hasattr(camera, 'set_ir_enhancement'):
            camera.set_ir_enhancement(is_on)
        logger.info(f"IR enhancement for {camera.source_id}: {is_on}")

    def _reconnect_camera(self, camera):
        """Trigger manual reconnect"""
        threading.Thread(target=camera.force_reconnect, daemon=True).start()

    def _update_status_loop(self):
        """Periodically update status labels"""
        try:
            for source_id, widgets in self.camera_widgets.items():
                camera = widgets['camera']
                
                # Update Connection Status
                is_connected = camera.get_connection_status()
                status_lbl = widgets['status_label']
                if is_connected:
                    status_lbl.configure(text="● Hoạt động", text_color=Colors.SUCCESS)
                else:
                    status_lbl.configure(text="● Mất kết nối", text_color=Colors.DANGER)
                
                # Update IR Status
                is_ir = camera.get_infrared_status()
                ir_lbl = widgets['ir_status_label']
                if is_ir:
                    ir_lbl.configure(text="🌙 IR Mode: ON", text_color=Colors.WARNING)
                else:
                    ir_lbl.configure(text="☀️ IR Mode: OFF", text_color=Colors.TEXT_MUTED)
                
                # Sync Person Switch (in case changed elsewhere)
                is_person_enabled = self.state.is_person_detection_enabled(source_id)
                person_var = widgets['person_var']
                if (person_var.get() == "on") != is_person_enabled:
                    person_var.set("on" if is_person_enabled else "off")

        except Exception as e:
            logger.error(f"Error updating status: {e}")
            
        # Schedule next update
        if self.winfo_exists():
            self.after(1000, self._update_status_loop)

    def sync_all_switches(self):
        """Called externally to force sync (optional now with loop)"""
        pass
# gui/detection_controls.py
import logging
from customtkinter import CTkFrame, CTkLabel, CTkSwitch, StringVar, CTkScrollableFrame
from .styles import Colors, Fonts, Sizes, create_card_frame

logger = logging.getLogger(__name__)

class DetectionControlsFrame(CTkFrame):
    """Frame điều khiển chức năng nhận diện người cho từng camera - Phiên bản hiện đại"""
    
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
    
    def _create_header(self):
        """Tạo tiêu đề phần điều khiển"""
        header_frame = CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, Sizes.PADDING_MD))
        
        title = CTkLabel(
            header_frame,
            text="🎥 Điều Khiển Camera",
            font=Fonts.TITLE_SM,
            text_color=Colors.TEXT_PRIMARY
        )
        title.pack(side="left", anchor="w")
    
    def _create_controls(self):
        """Tạo các nút điều khiển camera"""
        controls_frame = CTkScrollableFrame(
            self,
            fg_color=Colors.BG_SECONDARY,
            border_width=1,
            border_color=Colors.BORDER,
            corner_radius=Sizes.CORNER_RADIUS,
            label_text="Trạng Thái Nhận Diện",
            label_font=Fonts.BODY_BOLD,
            label_text_color=Colors.TEXT_PRIMARY
        )
        controls_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        controls_frame.grid_columnconfigure(0, weight=1)
        
        self.switches = {}
        self._populate_controls(controls_frame)
    
    def _populate_controls(self, container):
        """Điền các công tắc cho mỗi camera"""
        cameras = self.camera_manager.cameras
        
        if not cameras:
            no_camera_label = CTkLabel(
                container,
                text="📭 Không có camera nào",
                font=Fonts.BODY,
                text_color=Colors.TEXT_MUTED
            )
            no_camera_label.pack(pady=Sizes.PADDING_LG)
            return
        
        for idx, (source_id, cam) in enumerate(cameras.items()):
            self._create_camera_control(container, source_id, idx)
    
    def _create_camera_control(self, parent, source_id: str, index: int):
        """Tạo điều khiển cho một camera - Sử dụng GRID thay vì pack"""
        # Frame chứa camera - Dùng pack() vì parent là scrollable frame
        cam_frame = create_card_frame(parent, fg_color=Colors.BG_TERTIARY)
        cam_frame.pack(fill="x", padx=Sizes.PADDING_SM, pady=Sizes.PADDING_SM)
        cam_frame.grid_columnconfigure(1, weight=1)
        
        # Icon camera
        icon_label = CTkLabel(
            cam_frame,
            text="📹",
            font=Fonts.TITLE_MD,
            text_color=Colors.PRIMARY
        )
        icon_label.grid(row=0, column=0, padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD, sticky="w")
        
        # Tên camera + Status
        info_frame = CTkFrame(cam_frame, fg_color="transparent")
        info_frame.grid(row=0, column=1, sticky="ew", padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)
        info_frame.grid_columnconfigure(0, weight=1)
        
        camera_name = CTkLabel(
            info_frame,
            text=f"📷 Camera {source_id}",
            font=Fonts.BODY_BOLD,
            text_color=Colors.TEXT_PRIMARY
        )
        camera_name.grid(row=0, column=0, sticky="w")
        
        # Status indicator
        status_dot = CTkLabel(
            info_frame,
            text="● Sẵn sàng",
            font=Fonts.SMALL,
            text_color=Colors.SUCCESS
        )
        status_dot.grid(row=1, column=0, sticky="w", pady=(Sizes.PADDING_SM, 0))
        
        # Công tắc bật/tắt
        switch_var = StringVar(
            value="on" if self.state.is_person_detection_enabled(source_id) else "off"
        )
        
        switch = CTkSwitch(
            cam_frame,
            text="Kích Hoạt",
            variable=switch_var,
            onvalue="on",
            offvalue="off",
            font=Fonts.BODY,
            text_color=Colors.TEXT_PRIMARY,
            progress_color=Colors.PRIMARY,
            button_color=Colors.BORDER,
            button_hover_color=Colors.PRIMARY,
            command=lambda sid=source_id, var=switch_var: self._toggle_detection(sid, var)
        )
        switch.grid(row=0, column=2, rowspan=2, padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD, sticky="e")
        
        self.switches[source_id] = {
            'var': switch_var,
            'switch': switch,
            'status_label': status_dot
        }
    
    def _toggle_detection(self, source_id: str, switch_var: StringVar):
        """Bật/tắt nhận diện cho một camera cụ thể"""
        is_on = switch_var.get() == "on"
        self.state.set_person_detection_enabled(is_on, source_id)
        
        status_label = self.switches[source_id]['status_label']
        if is_on:
            status_label.configure(text="● Hoạt động", text_color=Colors.SUCCESS)
        else:
            status_label.configure(text="● Tắt", text_color=Colors.TEXT_MUTED)
        
        logger.info(f"Person detection for camera {source_id} set to: {is_on}")
    
    def sync_all_switches(self):
        """Đồng bộ trạng thái của tất cả công tắc"""
        for source_id, switch_data in self.switches.items():
            is_enabled = self.state.is_person_detection_enabled(source_id)
            state_str = "on" if is_enabled else "off"
            
            if switch_data['var'].get() != state_str:
                switch_data['var'].set(state_str)
                status_label = switch_data['status_label']
                if is_enabled:
                    status_label.configure(text="● Hoạt động", text_color=Colors.SUCCESS)
                else:
                    status_label.configure(text="● Tắt", text_color=Colors.TEXT_MUTED)
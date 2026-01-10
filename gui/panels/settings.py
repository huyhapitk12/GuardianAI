# Bảng cài đặt nâng cao với tùy chỉnh toàn diện

import json
from pathlib import Path

import customtkinter as ctk
from customtkinter import StringVar, BooleanVar
from CTkMessagebox import CTkMessagebox

from config import settings
from gui.styles import Colors, Fonts, Sizes, create_button, create_card, create_entry


# Bảng cài đặt toàn diện
class SettingsPanel(ctk.CTkFrame):
    
    def __init__(self, parent, state_manager=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        
        self.state = state_manager
        self.pages = {}
        self.nav_buttons = {}
        self.setting_vars = {}  # Lưu trữ tất cả biến cài đặt
        self.original_values = {}  # Cho chức năng đặt lại
        self.has_changes = BooleanVar(value=False)
        
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.build_sidebar()
        self.build_content()
        
        # Tải các giá trị hiện tại
        self.load_current_settings()
        
        # Chọn tab đầu tiên
        self.after(100, lambda: self.select_tab("detection"))
    
    # Xây dựng thanh điều hướng bên trái
    def build_sidebar(self):
        sidebar = ctk.CTkFrame(self, fg_color=Colors.BG_SECONDARY, width=220, corner_radius=Sizes.RADIUS_LG)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, Sizes.SM))
        sidebar.grid_propagate(False)
        
        # Tiêu đề
        header = ctk.CTkFrame(sidebar, fg_color="transparent")
        header.pack(fill="x", padx=Sizes.MD, pady=Sizes.MD)
        
        ctk.CTkLabel(
            header, text="CÀI ĐẶT",
            font=Fonts.TITLE_SM, text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w")
        
        ctk.CTkLabel(
            header, text="Tùy chỉnh hệ thống",
            font=Fonts.CAPTION, text_color=Colors.TEXT_MUTED
        ).pack(anchor="w")
        
        # Phân cách
        ctk.CTkFrame(sidebar, fg_color=Colors.BORDER, height=1).pack(fill="x", padx=Sizes.MD, pady=Sizes.SM)
        
        # Các tab điều hướng
        tabs = [
            ("detection", "", "Nhận diện"),
            ("camera", "", "Camera"),
            ("alerts", "", "Cảnh báo"),
            ("recording", "", "Ghi hình"),
            ("telegram", "", "Telegram"),
            ("ai", "", "AI Assistant"),
            ("appearance", "", "Giao diện"),
            ("system", "", "Hệ thống"),
        ]
        
        for key, icon, text in tabs:
            btn = ctk.CTkButton(
                sidebar,
                text=f"{text}",
                font=Fonts.BODY,
                fg_color="transparent",
                hover_color=Colors.BG_TERTIARY,
                text_color=Colors.TEXT_SECONDARY,
                anchor="w",
                height=40,
                command=lambda k=key: self.select_tab(k)
            )
            btn.pack(fill="x", padx=Sizes.SM, pady=2)
            self.nav_buttons[key] = btn
        
        # Các hành động phía dưới
        bottom = ctk.CTkFrame(sidebar, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=Sizes.SM, pady=Sizes.MD)
        
        create_button(
            bottom, "Lưu thay đổi", "success",
            command=self.save_settings
        ).pack(fill="x", pady=(0, Sizes.XS))
        
        create_button(
            bottom, "Khôi phục", "secondary",
            command=self.reset_settings
        ).pack(fill="x")
    
    # Xây dựng khu vực nội dung
    def build_content(self):
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)
        
        # Tạo tất cả các trang
        self.pages["detection"] = self.build_detection_page()
        self.pages["camera"] = self.build_camera_page()
        self.pages["alerts"] = self.build_alerts_page()
        self.pages["recording"] = self.build_recording_page()
        self.pages["telegram"] = self.build_telegram_page()
        self.pages["ai"] = self.build_ai_page()
        self.pages["appearance"] = self.build_appearance_page()
        self.pages["system"] = self.build_system_page()
    
    # Chuyển đến tab được chọn
    def select_tab(self, key):
        for k, btn in self.nav_buttons.items():
            if k == key:
                btn.configure(fg_color=Colors.PRIMARY, text_color=Colors.TEXT_PRIMARY)
            else:
                btn.configure(fg_color="transparent", text_color=Colors.TEXT_SECONDARY)
        
        for k, page in self.pages.items():
            if k == key:
                page.grid(row=0, column=0, sticky="nsew")
            else:
                page.grid_forget()
    
    # Trang cài đặt nhận diện
    def build_detection_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        
        self.add_header(page, "Cài đặt Nhận diện", 
                        "Điều chỉnh ngưỡng và độ chính xác của các thuật toán nhận diện")
        
        # Person Detection
        card1 = self.create_section(page, "Nhận diện Người")
        
        self.add_switch(card1, "detection.face_recognition_enabled",
                        "Bật nhận diện khuôn mặt", True,
                        "Xác định danh tính người trong khung hình")
        
        self.add_slider(card1, "detection.person_confidence_threshold", 
                        "Ngưỡng phát hiện người", 0.0, 1.0, 0.6,
                        "Độ tin cậy tối thiểu để xác nhận có người trong khung hình")
        
        self.add_slider(card1, "detection.face_recognition_threshold", 
                        "Ngưỡng nhận diện khuôn mặt", 0.0, 1.0, 0.55,
                        "Độ chính xác tối thiểu để nhận ra khuôn mặt đã đăng ký")
        
        self.add_slider(card1, "detection.iou_threshold", 
                        "Ngưỡng IOU (tracking)", 0.0, 1.0, 0.5,
                        "Độ trùng khớp box để theo dõi cùng một đối tượng")
        
        # Fire Detection
        card2 = self.create_section(page, "Phát hiện Cháy")
        
        self.add_slider(card2, "detection.fire_confidence_threshold", 
                        "Ngưỡng phát hiện cháy", 0.0, 1.0, 0.85,
                        "Độ tin cậy tối thiểu để cảnh báo cháy")
        
        self.add_slider(card2, "detection.smoke_confidence_threshold", 
                        "Ngưỡng phát hiện khói", 0.0, 1.0, 0.8,
                        "Độ tin cậy tối thiểu để phát hiện khói")
        
        # Fall Detection
        card3 = self.create_section(page, "Phát hiện Té ngã")
        
        self.add_switch(card3, "fall.enabled",
                        "Bật phát hiện té ngã", True,
                        "Phát hiện khi có người bị ngã")
        
        self.add_slider(card3, "fall.threshold", 
                        "Ngưỡng phát hiện", 0.5, 1.0, 0.8,
                        "Độ tin cậy tối thiểu để cảnh báo té ngã")
        
        self.add_slider(card3, "fall.pose_confidence", 
                        "Ngưỡng pose", 0.3, 1.0, 0.8,
                        "Độ tin cậy tối thiểu của pose estimation")
        
        self.add_slider(card3, "fall.n_consecutive", 
                        "Số frame liên tiếp", 1, 10, 3,
                        "Số frame liên tiếp cần vượt ngưỡng để cảnh báo")
        
        # Tracker Settings
        card4 = self.create_section(page, "Theo dõi (Tracker)")
        
        self.add_slider(card4, "tracker.timeout_seconds", 
                        "Timeout (giây)", 5.0, 60.0, 10.0,
                        "Thời gian để xóa track khi mất người")
        
        self.add_slider(card4, "tracker.stranger_confirm_frames", 
                        "Frame xác nhận người lạ", 5, 60, 20,
                        "Số frame không nhận diện được để xác nhận người lạ")
        
        self.add_slider(card4, "tracker.known_person_confirm_frames", 
                        "Frame xác nhận người quen", 1, 10, 4,
                        "Số frame nhận diện được để xác nhận người quen")
        
        self.add_slider(card4, "tracker.face_recognition_cooldown", 
                        "Cooldown nhận diện (giây)", 0.5, 5.0, 3.0,
                        "Thời gian chờ giữa các lần kiểm tra khuôn mặt")
        
        return page
    
    # Cài đặt camera
    def build_camera_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        
        self.add_header(page, "Cài đặt Camera", 
                        "Điều chỉnh các thông số camera và xử lý video")
        
        card1 = self.create_section(page, "Cài đặt chung")
        
        self.add_slider(card1, "camera.target_fps", 
                        "FPS mục tiêu", 5, 30, 10,
                        "Số khung hình xử lý mỗi giây")
        
        self.add_slider(card1, "camera.process_every_n_frames", 
                        "Xử lý mỗi N frame", 1, 10, 2,
                        "Bỏ qua frame để tối ưu CPU")
        
        self.add_slider(card1, "camera.max_reconnect_attempts", 
                        "Số lần kết nối lại", 1, 20, 10,
                        "Số lần thử kết nối lại khi mất kết nối")
        
        card2 = self.create_section(page, "Độ phân giải")
        
        self.add_option(card2, "camera.process_size",
                        "Kích thước xử lý", 
                        ["640x480", "960x540", "1280x720"],
                        "Độ phân giải để nhận diện (nhỏ = nhanh)")
        
        card3 = self.create_section(page, "Chế độ Hồng ngoại (IR)")
        
        self.add_slider(card3, "camera.infrared.person_detection_threshold", 
                        "Ngưỡng người (IR)", 0.3, 0.8, 0.45,
                        "Ngưỡng phát hiện người trong chế độ IR")
        
        return page
    
    # Cài đặt cảnh báo
    def build_alerts_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        
        self.add_header(page, "Cài đặt Cảnh báo", 
                        "Điều chỉnh tần suất và loại cảnh báo")
        
        card1 = self.create_section(page, "Chống spam")
        
        self.add_slider(card1, "spam_guard.debounce_seconds", 
                        "Thời gian debounce (giây)", 30, 600, 120,
                        "Khoảng thời gian tối thiểu giữa các cảnh báo cùng loại")
        
        self.add_slider(card1, "spam_guard.min_interval", 
                        "Khoảng cách tối thiểu (giây)", 5, 60, 15,
                        "Thời gian chờ giữa mọi cảnh báo")
        
        self.add_slider(card1, "spam_guard.max_per_minute", 
                        "Tối đa mỗi phút", 1, 20, 4,
                        "Số cảnh báo tối đa trong 1 phút")
        
        card2 = self.create_section(page, "Còi báo động")
        
        self.add_slider(card2, "alarm.max_volume", 
                        "Âm lượng tối đa", 1.0, 10.0, 5.0,
                        "Độ to của còi báo động")
        
        self.add_slider(card2, "telegram.user_response_window_seconds", 
                        "Thời gian chờ phản hồi (giây)", 10, 120, 30,
                        "Thời gian chờ trước khi tự động bật còi")
        
        return page
    
    # Cài đặt ghi hình
    def build_recording_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        
        self.add_header(page, "Cài đặt Ghi hình", 
                        "Cấu hình video ghi lại khi có sự kiện")
        
        card1 = self.create_section(page, "Cài đặt chung")
        
        self.add_slider(card1, "recorder.duration_seconds", 
                        "Thời lượng (giây)", 5, 60, 10,
                        "Độ dài video ghi lại")
        
        self.add_slider(card1, "recorder.extension_seconds", 
                        "Thời gian kéo dài (giây)", 5, 30, 10,
                        "Thời gian kéo dài nếu vẫn còn nguy hiểm")
        
        card2 = self.create_section(page, "Chất lượng")
        
        self.add_option(card2, "recorder.fourcc",
                        "Codec video", 
                        ["mp4v", "XVID", "H264", "avc1"],
                        "Định dạng nén video")
        
        self.add_slider(card2, "recorder.fps", 
                        "FPS ghi hình", 5.0, 30.0, 10.0,
                        "Số khung hình mỗi giây của video")
        
        return page
    
    # Cài đặt Telegram bot
    def build_telegram_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        
        self.add_header(page, "Cài đặt Telegram", 
                        "Cấu hình bot và thông báo Telegram")
        
        card1 = self.create_section(page, "Thông tin Bot")
        
        self.add_text_input(card1, "telegram.token",
                            "Bot Token", "Nhập token từ @BotFather",
                            is_password=True)
        
        self.add_text_input(card1, "telegram.chat_id",
                            "Chat ID", "ID của cuộc trò chuyện")
        
        card2 = self.create_section(page, "Gửi tin nhắn")
        
        self.add_slider(card2, "telegram.user_response_window_seconds", 
                        "Thời gian chờ phản hồi (giây)", 10, 120, 30,
                        "Thời gian chờ user phản hồi cảnh báo")
        
        self.add_slider(card2, "telegram.httpx_timeout", 
                        "Timeout kết nối (giây)", 30, 300, 180,
                        "Thời gian chờ kết nối tới Telegram")
        
        return page
    
    # Cài đặt trợ lý AI
    def build_ai_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        
        self.add_header(page, "AI Assistant", 
                        "Cấu hình trợ lý AI thông minh")
        
        card1 = self.create_section(page, "Cài đặt chung")
        
        self.add_switch(card1, "ai.enabled",
                        "Bật AI Assistant", False,
                        "Kích hoạt tính năng trả lời thông minh")
        
        self.add_text_input(card1, "ai.api_key",
                            "API Key", "Nhập API key",
                            is_password=True)
        
        self.add_text_input(card1, "ai.api_base",
                            "API Base URL", "URL của API (nếu dùng custom)")
        
        card2 = self.create_section(page, "Tham số mô hình")
        
        self.add_text_input(card2, "ai.model",
                            "Mô hình", "Tên model (ví dụ: gpt-4o-mini)")
        
        self.add_slider(card2, "ai.temperature", 
                        "Temperature", 0.0, 2.0, 0.5,
                        "Độ sáng tạo của AI (cao = ngẫu nhiên hơn)")
        
        self.add_slider(card2, "ai.max_tokens", 
                        "Max tokens", 64, 4096, 512,
                        "Độ dài tối đa của phản hồi")
        
        card3 = self.create_section(page, "Ngữ cảnh")
        
        self.add_slider(card3, "ai.max_history_turns", 
                        "Số tin nhắn ngữ cảnh", 1, 20, 5,
                        "Số tin nhắn gần nhất để AI nhớ")
        
        return page
    
    # Cài đặt giao diện
    def build_appearance_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        
        self.add_header(page, "Giao diện", 
                        "Tùy chỉnh giao diện hiển thị")
        
        card1 = self.create_section(page, "Thông tin")
        
        ctk.CTkLabel(card1, text="Cài đặt giao diện chưa được hỗ trợ trong phiên bản này.",
                    font=("Arial", 12), text_color="#888888").pack(padx=16, pady=16, anchor="w")
        
        return page
    
    # Cài đặt hệ thống
    def build_system_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        
        self.add_header(page, "Cài đặt Hệ thống", 
                        "Quản lý tài nguyên và dữ liệu")
        
        # Models settings
        card_models = self.create_section(page, "Cài đặt Models")
        
        self.add_option(card_models, "models.mode",
                        "Chế độ Model", 
                        ["Small", "Medium"],
                        "Small (nhanh, nhẹ) hoặc Medium (cân bằng, chính xác)")
        
        self.add_option(card_models, "models.device",
                        "Thiết bị xử lý", 
                        ["cpu", "gpu"],
                        "Chọn CPU hoặc GPU (cần CUDA)")
        
        card1 = self.create_section(page, "Đường dẫn")
        
        self.add_path_input(card1, "paths.data_dir",
                            "Thư mục dữ liệu", str(settings.paths.data_dir))
        
        self.add_path_input(card1, "paths.tmp_dir",
                            "Thư mục tạm", str(settings.paths.tmp_dir))
        
        self.add_path_input(card1, "paths.model_dir",
                            "Thư mục model", str(settings.paths.model_dir))
        
        # Action buttons
        actions = ctk.CTkFrame(page, fg_color="transparent")
        actions.pack(fill="x", pady=Sizes.MD)
        
        create_button(
            actions, "Xóa dữ liệu tạm", "danger",
            command=self.clear_temp_data
        ).pack(side="left", padx=(0, Sizes.SM))
        
        create_button(
            actions, "Rebuild Embeddings", "secondary",
            command=self.rebuild_embeddings
        ).pack(side="left")
        
        return page
    
    # Thêm tiêu đề trang
    def add_header(self, parent, title, subtitle):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=(0, Sizes.MD))
        
        ctk.CTkLabel(frame, text=title, font=Fonts.TITLE_MD, 
                    text_color=Colors.TEXT_PRIMARY).pack(anchor="w")
        ctk.CTkLabel(frame, text=subtitle, font=Fonts.BODY, 
                    text_color=Colors.TEXT_MUTED).pack(anchor="w")
    
    # Tạo thẻ section
    def create_section(self, parent, title):
        card = create_card(parent)
        card.pack(fill="x", pady=(0, Sizes.MD))
        
        ctk.CTkLabel(card, text=title, font=Fonts.BODY_BOLD,
                    text_color=Colors.TEXT_PRIMARY).pack(anchor="w", padx=Sizes.MD, pady=(Sizes.MD, Sizes.SM))
        
        return card
    
    # Thêm cài đặt slider
    def add_slider(self, parent, key, label, 
                   min_val, max_val, default,
                   description=""):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=Sizes.MD, pady=Sizes.SM)
        
        # Header row
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x")
        
        ctk.CTkLabel(header, text=label, font=Fonts.BODY,
                    text_color=Colors.TEXT_PRIMARY).pack(side="left")
        
        # Xác định nếu là số nguyên
        is_int = isinstance(default, int) and isinstance(min_val, int) and isinstance(max_val, int)
        current = settings.get(key, default)
        
        value_var = StringVar(value=str(int(current) if is_int else f"{current:.2f}"))
        self.setting_vars[key] = {"var": value_var, "type": "slider", "is_int": is_int}
        
        value_label = ctk.CTkLabel(header, textvariable=value_var, font=Fonts.BODY_BOLD,
                                  text_color=Colors.PRIMARY, width=60)
        value_label.pack(side="right")
        
        # Description
        if description:
            ctk.CTkLabel(frame, text=description, font=Fonts.CAPTION,
                        text_color=Colors.TEXT_MUTED).pack(anchor="w")
        
        # Slider
        slider = ctk.CTkSlider(
            frame, from_=min_val, to=max_val,
            progress_color=Colors.PRIMARY,
            button_color=Colors.PRIMARY,
            button_hover_color=Colors.PRIMARY_HOVER
        )
        slider.set(current)
        slider.pack(fill="x", pady=(Sizes.XS, 0))
        
        def on_change(val):
            value_var.set(str(int(val)) if is_int else f"{val:.2f}")
            self.has_changes.set(True)
        
        slider.configure(command=on_change)
        self.setting_vars[key]["widget"] = slider
    
    # Thêm cài đặt switch
    def add_switch(self, parent, key, label, default, description=""):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=Sizes.MD, pady=Sizes.SM)
        
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x")
        
        text_frame = ctk.CTkFrame(header, fg_color="transparent")
        text_frame.pack(side="left", fill="x", expand=True)
        
        ctk.CTkLabel(text_frame, text=label, font=Fonts.BODY,
                    text_color=Colors.TEXT_PRIMARY).pack(anchor="w")
        
        if description:
            ctk.CTkLabel(text_frame, text=description, font=Fonts.CAPTION,
                        text_color=Colors.TEXT_MUTED).pack(anchor="w")
        
        current = settings.get(key, default)
        var = StringVar(value="on" if current else "off")
        self.setting_vars[key] = {"var": var, "type": "switch"}
        
        switch = ctk.CTkSwitch(
            header, text="", variable=var,
            onvalue="on", offvalue="off",
            progress_color=Colors.SUCCESS,
            command=lambda: self.has_changes.set(True)
        )
        switch.pack(side="right")
        self.setting_vars[key]["widget"] = switch
    
    # Thêm menu lựa chọn
    def add_option(self, parent, key, label, options, description=""):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=Sizes.MD, pady=Sizes.SM)
        
        text_frame = ctk.CTkFrame(frame, fg_color="transparent")
        text_frame.pack(side="left", fill="x", expand=True)
        
        ctk.CTkLabel(text_frame, text=label, font=Fonts.BODY,
                    text_color=Colors.TEXT_PRIMARY).pack(anchor="w")
        
        if description:
            ctk.CTkLabel(text_frame, text=description, font=Fonts.CAPTION,
                        text_color=Colors.TEXT_MUTED).pack(anchor="w")
        
        current = settings.get(key, options[0])
        if isinstance(current, list):
            current = f"{current[0]}x{current[1]}"
        
        var = StringVar(value=str(current))
        self.setting_vars[key] = {"var": var, "type": "option", "options": options}
        
        menu = ctk.CTkOptionMenu(
            frame, values=options, variable=var,
            fg_color=Colors.BG_TERTIARY,
            button_color=Colors.BG_ELEVATED,
            button_hover_color=Colors.PRIMARY,
            dropdown_fg_color=Colors.BG_SECONDARY,
            width=150,
            command=lambda _: self.has_changes.set(True)
        )
        menu.pack(side="right")
        self.setting_vars[key]["widget"] = menu
    
    # Thêm ô nhập liệu text
    def add_text_input(self, parent, key, label, placeholder, is_password=False):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=Sizes.MD, pady=Sizes.SM)
        
        ctk.CTkLabel(frame, text=label, font=Fonts.BODY,
                    text_color=Colors.TEXT_PRIMARY).pack(anchor="w")
        
        current = settings.get(key, "")
        
        entry = ctk.CTkEntry(
            frame,
            placeholder_text=placeholder,
            show="•" if is_password else "",
            fg_color=Colors.BG_TERTIARY,
            border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY,
            height=40
        )
        entry.pack(fill="x", pady=(Sizes.XS, 0))
        
        if current:
            entry.insert(0, str(current))
        
        entry.bind("<KeyRelease>", lambda _: self.has_changes.set(True))
        
        self.setting_vars[key] = {"widget": entry, "type": "text"}
    
    # Thêm ô nhập đường dẫn với nút duyệt file
    def add_path_input(self, parent, key, label, current):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=Sizes.MD, pady=Sizes.SM)
        
        ctk.CTkLabel(frame, text=label, font=Fonts.BODY,
                    text_color=Colors.TEXT_PRIMARY).pack(anchor="w")
        
        input_frame = ctk.CTkFrame(frame, fg_color="transparent")
        input_frame.pack(fill="x", pady=(Sizes.XS, 0))
        
        entry = ctk.CTkEntry(
            input_frame,
            fg_color=Colors.BG_TERTIARY,
            border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY,
            height=40
        )
        entry.pack(side="left", fill="x", expand=True, padx=(0, Sizes.XS))
        entry.insert(0, current)
        
        def browse():
            from tkinter import filedialog
            path = filedialog.askdirectory(initialdir=current)
            if path:
                entry.delete(0, "end")
                entry.insert(0, path)
                self.has_changes.set(True)
        
        create_button(input_frame, "Browse", "secondary", "small", 
                      width=60, command=browse).pack(side="right")
        
        self.setting_vars[key] = {"widget": entry, "type": "path"}
    
    # Tải các giá trị cài đặt hiện tại
    def load_current_settings(self):
        for key, data in self.setting_vars.items():
            current = settings.get(key)
            if current is not None:
                self.original_values[key] = current
    
    # Lưu tất cả cài đặt
    def save_settings(self):
        try:
            changes = {}
            
            for key, data in self.setting_vars.items():
                if data["type"] == "slider":
                    widget = data.get("widget")
                    if widget:
                        val = widget.get()
                        changes[key] = int(val) if data.get("is_int") else float(val)
                
                elif data["type"] == "switch":
                    var = data.get("var")
                    if var:
                        changes[key] = (var.get() == "on")
                
                elif data["type"] == "option":
                    var = data.get("var")
                    if var:
                        val = var.get()
                        # Xử lý các trường hợp đặc biệt như độ phân giải
                        if "x" in val and key.endswith("_size"):
                            parts = val.split("x")
                            changes[key] = [int(parts[0]), int(parts[1])]
                        else:
                            changes[key] = val
                
                elif data["type"] == "text":
                    widget = data.get("widget")
                    if widget:
                        changes[key] = widget.get()
                
                elif data["type"] == "path":
                    widget = data.get("widget")
                    if widget:
                        changes[key] = widget.get()
            
            # Áp dụng thay đổi
            for key, value in changes.items():
                settings.set(key, value)
            
            # Lưu vào file
            settings.save()
            
            self.has_changes.set(False)
            
            CTkMessagebox(
                title="Thành công",
                message="Đã lưu cài đặt!",
                icon="check"
            )
            
            print("Cài đặt đã được cập nhật")
            
        except Exception as e:
            CTkMessagebox(
                title="Lỗi",
                message=f"Không thể lưu: {e}",
                icon="cancel"
            )
    
    def reset_settings(self):
        """Khôi phục về giá trị mặc định"""
        result = CTkMessagebox(
            title="Xác nhận",
            message="Khôi phục tất cả cài đặt về mặc định?",
            icon="question",
            option_1="Hủy",
            option_2="Khôi phục"
        ).get()
        
        if result != "Khôi phục":
            return
        
        # Khôi phục về mặc định
        settings.reset_to_defaults()
        
        # Cập nhật các widget UI
        for key, data in self.setting_vars.items():
            current = settings.get(key)
            if current is None:
                continue
            
            if data["type"] == "slider":
                widget = data.get("widget")
                if widget:
                    widget.set(current)
                    var = data.get("var")
                    if var:
                        is_int = data.get("is_int", False)
                        var.set(str(int(current)) if is_int else f"{current:.2f}")
            
            elif data["type"] == "switch":
                var = data.get("var")
                if var:
                    var.set("on" if current else "off")
            
            elif data["type"] == "option":
                var = data.get("var")
                if var:
                    if isinstance(current, list):
                        var.set(f"{current[0]}x{current[1]}")
                    else:
                        var.set(str(current))
            
            elif data["type"] in ("text", "path"):
                widget = data.get("widget")
                if widget:
                    widget.delete(0, "end")
                    widget.insert(0, str(current))
        
        self.has_changes.set(False)
        
        CTkMessagebox(
            title="Thành công",
            message="Đã khôi phục cài đặt mặc định!",
            icon="check"
        )
    
    def clear_temp_data(self):
        """Xóa dữ liệu tạm"""
        result = CTkMessagebox(
            title="Xác nhận",
            message="Xóa tất cả dữ liệu tạm (video, ảnh cache)?",
            icon="warning",
            option_1="Hủy",
            option_2="Xóa"
        ).get()
        
        if result != "Xóa":
            return
        
        try:
            import shutil
            tmp_dir = settings.paths.tmp_dir
            
            if tmp_dir.exists():
                for f in tmp_dir.iterdir():
                    try:
                        if f.is_file():
                            f.unlink()
                        elif f.is_dir():
                            shutil.rmtree(f)
                    except Exception:
                        pass
            
            CTkMessagebox(
                title="Thành công",
                message="Đã xóa dữ liệu tạm!",
                icon="check"
            )
            
            print("[INFO] Đã xóa dữ liệu tạm")
            
        except Exception as e:
            CTkMessagebox(
                title="Lỗi",
                message=f"Không thể lưu: {e}",
                icon="cancel"
            )
    
    def rebuild_embeddings(self):
        """Rebuild face embeddings"""
        CTkMessagebox(
            title="Thông báo",
            message="Chức năng này cần được gọi từ menu Persons",
            icon="info"
        )
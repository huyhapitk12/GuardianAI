# Panel cài đặt nâng cao với tùy chỉnh toàn diện

import json
from pathlib import Path

import customtkinter as ctk
from customtkinter import StringVar, BooleanVar
from CTkMessagebox import CTkMessagebox

from config import settings
from gui.styles import Colors, Fonts, Sizes, create_button, create_card, create_entry


# Panel cài đặt toàn diện
class SettingsPanel(ctk.CTkFrame):
    
    def __init__(self, parent, state_manager=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        
        self.state = state_manager
        self.pages = {}
        self.nav_buttons = {}
        self.setting_vars = {}  # Store all setting variables
        self.original_values = {}  # For reset functionality
        self.has_changes = BooleanVar(value=False)
        
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.build_sidebar()
        self.build_content()
        
        # Load current values
        self.load_current_settings()
        
        # Select first tab
        self.after(100, lambda: self.select_tab("detection"))
    
    # Xây dựng thanh điều hướng bên trái
    def build_sidebar(self):
        sidebar = ctk.CTkFrame(self, fg_color=Colors.BG_SECONDARY, width=220, corner_radius=Sizes.RADIUS_LG)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, Sizes.SM))
        sidebar.grid_propagate(False)
        
        # Header
        header = ctk.CTkFrame(sidebar, fg_color="transparent")
        header.pack(fill="x", padx=Sizes.MD, pady=Sizes.MD)
        
        ctk.CTkLabel(
            header, text="⚙️ CÀI ĐẶT",
            font=Fonts.TITLE_SM, text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w")
        
        ctk.CTkLabel(
            header, text="Tùy chỉnh hệ thống",
            font=Fonts.CAPTION, text_color=Colors.TEXT_MUTED
        ).pack(anchor="w")
        
        # Divider
        ctk.CTkFrame(sidebar, fg_color=Colors.BORDER, height=1).pack(fill="x", padx=Sizes.MD, pady=Sizes.SM)
        
        # Navigation tabs
        tabs = [
            ("detection", "🎯", "Nhận diện"),
            ("camera", "📹", "Camera"),
            ("alerts", "🔔", "Cảnh báo"),
            ("recording", "⏺️", "Ghi hình"),
            ("telegram", "📱", "Telegram"),
            ("ai", "🤖", "AI Assistant"),
            ("appearance", "🎨", "Giao diện"),
            ("system", "💻", "Hệ thống"),
        ]
        
        for key, icon, text in tabs:
            btn = ctk.CTkButton(
                sidebar,
                text=f"{icon}  {text}",
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
        
        # Bottom actions
        bottom = ctk.CTkFrame(sidebar, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=Sizes.SM, pady=Sizes.MD)
        
        create_button(
            bottom, "💾 Lưu thay đổi", "success",
            command=self.save_settings
        ).pack(fill="x", pady=(0, Sizes.XS))
        
        create_button(
            bottom, "↩️ Khôi phục", "secondary",
            command=self.reset_settings
        ).pack(fill="x")
    
    # Xây dựng khu vực nội dung
    def build_content(self):
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)
        
        # Create all pages
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
    
    # =========================================================================
    # PAGE BUILDERS
    # =========================================================================
    
    # Trang cài đặt nhận diện
    def build_detection_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        
        self.add_header(page, "Cài đặt Nhận diện", 
                        "Điều chỉnh ngưỡng và độ chính xác của các thuật toán nhận diện")
        
        # Person Detection
        card1 = self.create_section(page, "👤 Nhận diện Người")
        
        self.add_switch(card1, "detection.face_recognition_enabled",
                        "Bật nhận diện khuôn mặt", True,
                        "Xác định danh tính người trong khung hình")
        
        self.add_slider(card1, "detection.person_confidence", 
                        "Ngưỡng phát hiện người", 0.0, 1.0, 0.6,
                        "Độ tin cậy tối thiểu để xác nhận có người trong khung hình")
        
        self.add_slider(card1, "detection.face_recognition", 
                        "Ngưỡng nhận diện khuôn mặt", 0.0, 1.0, 0.45,
                        "Độ chính xác tối thiểu để nhận ra khuôn mặt đã đăng ký")
        
        self.add_slider(card1, "detection.face_confirmation_threshold", 
                        "Ngưỡng xác nhận danh tính", 0.0, 1.0, 0.5,
                        "Số lần nhận diện liên tiếp để xác nhận chắc chắn")
        
        self.add_slider(card1, "detection.iou_threshold", 
                        "Ngưỡng IOU (tracking)", 0.0, 1.0, 0.6,
                        "Độ trùng khớp box để theo dõi cùng một đối tượng")
        
        # Fire Detection
        card2 = self.create_section(page, "🔥 Phát hiện Cháy")
        
        self.add_slider(card2, "detection.fire_confidence", 
                        "Ngưỡng phát hiện cháy", 0.0, 1.0, 0.85,
                        "Độ tin cậy tối thiểu để cảnh báo cháy")
        
        self.add_slider(card2, "detection.smoke_confidence", 
                        "Ngưỡng phát hiện khói", 0.0, 1.0, 0.7,
                        "Độ tin cậy tối thiểu để phát hiện khói")
        
        self.add_switch(card2, "detection.fire_filter_enabled",
                        "Bộ lọc nhiễu cháy", True,
                        "Lọc các phát hiện sai do ánh sáng mạnh")
        
        return page
    
    # Cài đặt camera
    def build_camera_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        
        self.add_header(page, "Cài đặt Camera", 
                        "Điều chỉnh các thông số camera và xử lý video")
        
        card1 = self.create_section(page, "📹 Cài đặt chung")
        
        self.add_slider(card1, "camera.target_fps", 
                        "FPS mục tiêu", 5, 30, 10,
                        "Số khung hình xử lý mỗi giây")
        
        self.add_slider(card1, "camera.process_every_n_frames", 
                        "Xử lý mỗi N frame", 1, 10, 5,
                        "Bỏ qua frame để tối ưu CPU")
        
        self.add_slider(card1, "camera.buffer_size", 
                        "Kích thước buffer", 1, 10, 1,
                        "Số frame lưu đệm (thấp = ít delay)")
        
        card2 = self.create_section(page, "📐 Độ phân giải")
        
        self.add_option(card2, "camera.process_size",
                        "Kích thước xử lý", 
                        ["320x240", "640x480", "960x540", "1280x720"],
                        "Độ phân giải để nhận diện (nhỏ = nhanh)")
        
        self.add_switch(card2, "camera.auto_resize",
                        "Tự động resize", True,
                        "Tự động điều chỉnh kích thước video")
        
        card3 = self.create_section(page, "🌙 Chế độ Hồng ngoại (IR)")
        
        self.add_switch(card3, "camera.infrared.auto_detect",
                        "Tự động phát hiện IR", True,
                        "Tự động nhận biết khi camera chuyển sang chế độ đêm")
        
        self.add_slider(card3, "camera.infrared.detection_threshold", 
                        "Ngưỡng phát hiện IR", 0.5, 1.0, 0.98,
                        "Độ nhạy phát hiện chế độ IR")
        
        self.add_slider(card3, "camera.infrared.person_detection_threshold", 
                        "Ngưỡng người (IR)", 0.3, 0.8, 0.45,
                        "Ngưỡng phát hiện người trong chế độ IR")
        
        self.add_switch(card3, "camera.infrared.enhance_enabled",
                        "Tăng cường IR", False,
                        "Cải thiện chất lượng ảnh hồng ngoại")
        
        return page
    
    # Cài đặt cảnh báo
    def build_alerts_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        
        self.add_header(page, "Cài đặt Cảnh báo", 
                        "Điều chỉnh tần suất và loại cảnh báo")
        
        card1 = self.create_section(page, "🔔 Chống spam")
        
        self.add_slider(card1, "spam_guard.debounce_seconds", 
                        "Thời gian debounce (giây)", 30, 600, 120,
                        "Khoảng thời gian tối thiểu giữa các cảnh báo cùng loại")
        
        self.add_slider(card1, "spam_guard.min_interval", 
                        "Khoảng cách tối thiểu (giây)", 5, 60, 15,
                        "Thời gian chờ giữa mọi cảnh báo")
        
        self.add_slider(card1, "spam_guard.max_per_minute", 
                        "Tối đa mỗi phút", 1, 20, 4,
                        "Số cảnh báo tối đa trong 1 phút")
        
        card2 = self.create_section(page, "🚨 Loại cảnh báo")
        
        self.add_switch(card2, "alerts.stranger_enabled",
                        "Cảnh báo người lạ", True,
                        "Gửi thông báo khi phát hiện người không quen")
        
        self.add_switch(card2, "alerts.known_person_enabled",
                        "Thông báo người quen", True,
                        "Gửi thông báo khi nhận ra người đã đăng ký")
        
        self.add_switch(card2, "alerts.fire_enabled",
                        "Cảnh báo cháy", True,
                        "Gửi thông báo khi phát hiện cháy/khói")
        
        card3 = self.create_section(page, "🔊 Còi báo động")
        
        self.add_switch(card3, "alarm.auto_play_fire",
                        "Tự động bật còi khi cháy", True,
                        "Bật còi sau khi không có phản hồi")
        
        self.add_slider(card3, "alarm.volume", 
                        "Âm lượng còi", 0.0, 1.0, 0.8,
                        "Độ to của còi báo động")
        
        self.add_slider(card3, "alarm.response_timeout", 
                        "Thời gian chờ phản hồi (giây)", 10, 120, 30,
                        "Thời gian chờ trước khi tự động bật còi")
        
        return page
    
    # Cài đặt ghi hình
    def build_recording_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        
        self.add_header(page, "Cài đặt Ghi hình", 
                        "Cấu hình video ghi lại khi có sự kiện")
        
        card1 = self.create_section(page, "⏺️ Cài đặt chung")
        
        self.add_switch(card1, "recorder.enabled",
                        "Bật ghi hình", True,
                        "Tự động ghi video khi có cảnh báo")
        
        self.add_slider(card1, "recorder.duration", 
                        "Thời lượng (giây)", 5, 60, 15,
                        "Độ dài video ghi lại")
        
        self.add_slider(card1, "recorder.pre_buffer", 
                        "Ghi trước (giây)", 0, 10, 3,
                        "Số giây ghi trước khi sự kiện xảy ra")
        
        card2 = self.create_section(page, "🎬 Chất lượng")
        
        self.add_option(card2, "recorder.codec",
                        "Codec video", 
                        ["mp4v", "XVID", "H264", "avc1"],
                        "Định dạng nén video")
        
        self.add_slider(card2, "recorder.fps", 
                        "FPS ghi hình", 10, 30, 15,
                        "Số khung hình mỗi giây của video")
        
        self.add_option(card2, "recorder.quality",
                        "Chất lượng", 
                        ["low", "medium", "high", "original"],
                        "Độ phân giải video lưu")
        
        card3 = self.create_section(page, "💾 Lưu trữ")
        
        self.add_slider(card3, "recorder.max_files", 
                        "Số file tối đa", 10, 500, 100,
                        "Tự động xóa file cũ khi vượt quá")
        
        self.add_slider(card3, "recorder.max_size_mb", 
                        "Dung lượng tối đa (MB)", 100, 10000, 1000,
                        "Xóa file cũ khi vượt dung lượng")
        
        self.add_switch(card3, "recorder.encrypt",
                        "Mã hóa video", True,
                        "Mã hóa video để bảo mật")
        
        return page
    
    # Cài đặt Telegram bot
    def build_telegram_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        
        self.add_header(page, "Cài đặt Telegram", 
                        "Cấu hình bot và thông báo Telegram")
        
        card1 = self.create_section(page, "🤖 Thông tin Bot")
        
        self.add_text_input(card1, "telegram.bot_token",
                            "Bot Token", "Nhập token từ @BotFather",
                            is_password=True)
        
        self.add_text_input(card1, "telegram.chat_id",
                            "Chat ID", "ID của cuộc trò chuyện")
        
        card2 = self.create_section(page, "📤 Gửi tin nhắn")
        
        self.add_slider(card2, "telegram.response_timeout", 
                        "Thời gian chờ phản hồi (giây)", 10, 120, 30,
                        "Thời gian chờ user phản hồi cảnh báo")
        
        self.add_switch(card2, "telegram.send_video",
                        "Gửi video kèm", True,
                        "Gửi video clip cùng với ảnh cảnh báo")
        
        self.add_switch(card2, "telegram.silent_known_person",
                        "Im lặng với người quen", False,
                        "Không phát âm thanh khi thông báo người quen")
        
        card3 = self.create_section(page, "❤️ Heartbeat")
        
        self.add_switch(card3, "telegram.heartbeat_enabled",
                        "Bật heartbeat", True,
                        "Gửi tin nhắn định kỳ để xác nhận hệ thống hoạt động")
        
        self.add_slider(card3, "telegram.heartbeat_interval", 
                        "Khoảng cách (phút)", 5, 60, 30,
                        "Thời gian giữa các heartbeat")
        
        return page
    
    # Cài đặt trợ lý AI
    def build_ai_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        
        self.add_header(page, "AI Assistant", 
                        "Cấu hình trợ lý AI thông minh")
        
        card1 = self.create_section(page, "🤖 Cài đặt chung")
        
        self.add_switch(card1, "ai.enabled",
                        "Bật AI Assistant", True,
                        "Kích hoạt tính năng trả lời thông minh")
        
        self.add_option(card1, "ai.provider",
                        "Nhà cung cấp AI", 
                        ["openai", "anthropic", "google", "local"],
                        "Chọn API AI để sử dụng")
        
        self.add_text_input(card1, "ai.api_key",
                            "API Key", "Nhập API key",
                            is_password=True)
        
        card2 = self.create_section(page, "⚙️ Tham số mô hình")
        
        self.add_option(card2, "ai.model",
                        "Mô hình", 
                        ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "claude-3-sonnet", "gemini-pro"],
                        "Model AI sử dụng")
        
        self.add_slider(card2, "ai.temperature", 
                        "Temperature", 0.0, 2.0, 0.5,
                        "Độ sáng tạo của AI (cao = ngẫu nhiên hơn)")
        
        self.add_slider(card2, "ai.max_tokens", 
                        "Max tokens", 64, 4096, 512,
                        "Độ dài tối đa của phản hồi")
        
        card3 = self.create_section(page, "💬 Ngữ cảnh")
        
        self.add_slider(card3, "ai.context_messages", 
                        "Số tin nhắn ngữ cảnh", 1, 20, 10,
                        "Số tin nhắn gần nhất để AI nhớ")
        
        self.add_slider(card3, "ai.context_timeout", 
                        "Timeout ngữ cảnh (phút)", 5, 120, 30,
                        "Xóa ngữ cảnh sau thời gian không hoạt động")
        
        return page
    
    # Cài đặt giao diện
    def build_appearance_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        
        self.add_header(page, "Giao diện", 
                        "Tùy chỉnh giao diện hiển thị")
        
        card1 = self.create_section(page, "🎨 Theme")
        
        self.add_option(card1, "appearance.theme",
                        "Chủ đề", 
                        ["dark", "light", "system"],
                        "Chế độ màu của ứng dụng")
        
        self.add_option(card1, "appearance.accent_color",
                        "Màu nhấn", 
                        ["blue", "green", "purple", "orange", "red"],
                        "Màu chủ đạo của giao diện")
        
        card2 = self.create_section(page, "📹 Video Display")
        
        self.add_switch(card2, "appearance.show_fps",
                        "Hiển thị FPS", False,
                        "Hiện số khung hình/giây trên video")
        
        self.add_switch(card2, "appearance.show_timestamp",
                        "Hiển thị thời gian", True,
                        "Hiện timestamp trên video")
        
        self.add_switch(card2, "appearance.show_detection_info",
                        "Hiển thị thông tin nhận diện", True,
                        "Hiện box và label trên video")
        
        card3 = self.create_section(page, "📊 Dashboard")
        
        self.add_switch(card3, "appearance.show_activity_feed",
                        "Hiển thị Activity Feed", True,
                        "Hiện bảng hoạt động gần đây")
        
        self.add_slider(card3, "appearance.activity_max_items", 
                        "Số hoạt động hiển thị", 10, 100, 50,
                        "Số mục tối đa trong Activity Feed")
        
        return page
    
    # Cài đặt hệ thống
    def build_system_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        
        self.add_header(page, "Cài đặt Hệ thống", 
                        "Quản lý tài nguyên và dữ liệu")
        
        card1 = self.create_section(page, "💾 Bộ nhớ")
        
        self.add_slider(card1, "system.memory_limit_mb", 
                        "Giới hạn RAM (MB)", 512, 8192, 2048,
                        "Dung lượng RAM tối đa sử dụng")
        
        self.add_slider(card1, "system.cleanup_interval", 
                        "Dọn dẹp mỗi (phút)", 5, 60, 15,
                        "Tần suất giải phóng bộ nhớ")
        
        self.add_switch(card1, "system.auto_gc",
                        "Tự động dọn rác", True,
                        "Tự động thu gom bộ nhớ không dùng")
        
        card2 = self.create_section(page, "📁 Đường dẫn")
        
        self.add_path_input(card2, "paths.data_dir",
                            "Thư mục dữ liệu", str(settings.paths.data_dir))
        
        self.add_path_input(card2, "paths.tmp_dir",
                            "Thư mục tạm", str(settings.paths.tmp_dir))
        
        self.add_path_input(card2, "paths.model_dir",
                            "Thư mục model", str(settings.paths.model_dir))
        
        card3 = self.create_section(page, "🔧 Nâng cao")
        
        self.add_switch(card3, "system.debug_mode",
                        "Chế độ Debug", False,
                        "Hiển thị thông tin debug chi tiết")
        
        self.add_switch(card3, "system.log_to_file",
                        "Ghi log ra file", True,
                        "Lưu log vào file để kiểm tra sau")
        
        # Action buttons
        actions = ctk.CTkFrame(page, fg_color="transparent")
        actions.pack(fill="x", pady=Sizes.MD)
        
        create_button(
            actions, "🗑️ Xóa dữ liệu tạm", "danger",
            command=self.clear_temp_data
        ).pack(side="left", padx=(0, Sizes.SM))
        
        create_button(
            actions, "📊 Rebuild Embeddings", "secondary",
            command=self.rebuild_embeddings
        ).pack(side="left")
        
        return page
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
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
        
        # Determine if integer
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
        
        create_button(input_frame, "📂", "secondary", "small", 
                      width=40, command=browse).pack(side="right")
        
        self.setting_vars[key] = {"widget": entry, "type": "path"}
    
    # =========================================================================
    # ACTIONS
    # =========================================================================
    
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
                        # Handle special cases like resolution
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
            
            # Apply changes
            for key, value in changes.items():
                settings.set(key, value)
            
            # Save to file
            settings.save()
            
            self.has_changes.set(False)
            
            CTkMessagebox(
                title="Thành công",
                message="Đã lưu cài đặt!",
                icon="check"
            )
            
            print("✅ Settings saved")
            print("ℹ️ [SYSTEM] Configuration updated")
            
        except Exception as e:
            CTkMessagebox(
                title="Lỗi",
                message=f"Không thể lưu: {e}",
                icon="cancel"
            )
    
    def reset_settings(self):
        """Reset to default values"""
        result = CTkMessagebox(
            title="Xác nhận",
            message="Khôi phục tất cả cài đặt về mặc định?",
            icon="question",
            option_1="Hủy",
            option_2="Khôi phục"
        ).get()
        
        if result != "Khôi phục":
            return
        
        # Reset to defaults
        settings.reset_to_defaults()
        
        # Update UI widgets
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
        """Clear temporary data"""
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
            
            print("ℹ️ Temp data cleared")
            
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
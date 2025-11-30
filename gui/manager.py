# gui/manager.py

import logging
import os
import shutil
import cv2
import threading
import _tkinter
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageEnhance
from tkinter import filedialog

from customtkinter import (
    CTk,
    CTkFrame,
    CTkLabel,
    CTkButton,
    CTkEntry,
    CTkOptionMenu,
    CTkScrollableFrame,
    CTkTabview,
    CTkProgressBar,
    CTkSwitch,
    CTkImage,
    StringVar,
    set_appearance_mode,
    set_default_color_theme
)

from CTkMessagebox import CTkMessagebox

from config.settings import settings, update_config_value
from core.recorder import Recorder
from utils import security_manager

from .styles import (
    Colors, Fonts, Sizes,
    create_modern_button, create_card_frame, create_glass_card,
    create_modern_entry, create_stat_card
)
from .control_panels import SettingsPanel, RecordingPanel
from .analytics_panel import AnalyticsPanel
from .widgets import (
    GalleryPanel,
    ActivityFeedWidget, SystemLogsWidget, log_activity, log_system,
    CameraHealthWidget, FireHistoryWidget, UnifiedCameraList
)

logger = logging.getLogger(__name__)


class ModernFaceManagerGUI:
    """Giao diện quản lý khuôn mặt hiện đại với hiệu ứng đẹp"""
    
    def __init__(self, root, camera, face_detector, state_manager):
        self.root = root
        self.camera = camera
        self.face_detector = face_detector
        self.state = state_manager

        self.recorder = Recorder()
        self.current_person = None
        self.current_view = "dashboard"
        self.animation_running = False
        self.brightness_value = 1.0
        self.brightness_value = 1.0

        self._setup_modern_window()
        self._apply_theme()
        self._create_animated_layout()

        self.populate_person_list()
        self._show_dashboard()
        self._start_video_feed()
        self._start_animations()

        log_system("GUI initialized successfully", "success")
        log_activity("Face Manager started", "success")

    # ================================================================
    # SETUP & THEME
    # ================================================================

    def _setup_modern_window(self):
        """Setup window với style hiện đại"""
        self.root.title("Guardian Security • Hệ thống Giám sát Thông minh")
        self.root.geometry("1600x900")
        self.root.minsize(1400, 800)
        self.root.maxsize(1920, 1080)  # Giới hạn kích thước tối đa

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.configure(fg_color=Colors.BG_PRIMARY)

        try:
            if os.path.exists("assets/icon.ico"):
                self.root.iconbitmap("assets/icon.ico")
        except Exception:
            pass

    def _apply_theme(self):
        """Apply modern dark theme"""
        import customtkinter
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")

    # ================================================================
    # LAYOUT CREATION
    # ================================================================

    def _create_animated_layout(self):
        """Create layout với animations"""
        main_container = CTkFrame(self.root, fg_color=Colors.BG_PRIMARY)
        main_container.grid(row=0, column=0, sticky="nsew")
        
        main_container.grid_columnconfigure(0, weight=1) 
        # main_container.grid_columnconfigure(1, weight=0, minsize=Sizes.RIGHT_SIDEBAR_WIDTH) # Removed
        main_container.grid_rowconfigure(0, weight=1)

        self._create_content_area(main_container)
        
        # Right panel removed - features moved to tabs
        # if not self.simple_mode:
        #     self._create_right_panel(main_container)

    def _create_centered_container(self, parent):
        """Create a centered container with responsive margins"""
        wrapper = CTkFrame(parent, fg_color="transparent")
        wrapper.pack(fill="both", expand=True)
        
        # 1-3-1 ratio creates ~60% width content area centered (Standard Web Container)
        wrapper.grid_columnconfigure(0, weight=1)
        wrapper.grid_columnconfigure(1, weight=3)
        wrapper.grid_columnconfigure(2, weight=1)
        wrapper.grid_rowconfigure(0, weight=1)
        
        content = CTkFrame(wrapper, fg_color="transparent")
        content.grid(row=0, column=1, sticky="nsew")
        return content

    def _create_content_area(self, parent):
        """Create main content area với tabs"""
        self.content_container = CTkFrame(
            parent, 
            fg_color=Colors.BG_PRIMARY, 
            corner_radius=0
        )
        self.content_container.grid(row=0, column=0, sticky="nsew", padx=(0, 1))
                
        self.main_tabs = CTkTabview(
            self.content_container,
            fg_color=Colors.BG_PRIMARY,
            segmented_button_fg_color=Colors.BG_SECONDARY,
            segmented_button_selected_color=Colors.PRIMARY,
            segmented_button_selected_hover_color=Colors.PRIMARY_HOVER,
            segmented_button_unselected_color=Colors.BG_TERTIARY,
            segmented_button_unselected_hover_color=Colors.BG_ELEVATED,
            text_color=Colors.TEXT_PRIMARY,
            corner_radius=Sizes.CORNER_RADIUS_LG,
            command=self._on_tab_change
        )
        self.main_tabs.pack(fill="both", expand=True, padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)
        self.main_tabs._segmented_button.configure(font=Fonts.BODY_BOLD, height=40)

        # Add all tabs
        # Add all tabs
        self.dashboard_tab = self.main_tabs.add("📊 Tổng quan")
        self.cameras_tab = self.main_tabs.add("📹 Camera")
        self.persons_tab = self.main_tabs.add("👥 Khuôn mặt")
        self.analytics_tab = self.main_tabs.add("📈 Phân tích")
        self.gallery_tab = self.main_tabs.add("🎞️ Thư viện")
        self.settings_tab = self.main_tabs.add("⚙️ Cài đặt")
        
        # Initialize tab contents
        self._init_dashboard_tab()
        self._init_cameras_tab()
        self._init_persons_tab()
        self._init_analytics_tab()
        self._init_gallery_tab()
        self._init_settings_tab()

    def _create_right_panel(self, parent):
        """Create right control panel"""
        self.right_panel = CTkFrame(
            parent,
            width=Sizes.RIGHT_SIDEBAR_WIDTH,  # ✅ Fixed width
            fg_color=Colors.BG_SECONDARY,
            corner_radius=0
        )
        self.right_panel.grid(row=0, column=1, sticky="ns")  # ✅ CHỈ sticky ns (north-south), KHÔNG sticky "ew"
        self.right_panel.grid_propagate(False)  # ✅ QUAN TRỌNG: Không cho children làm thay đổi size
        
        # ✅ Không set grid_columnconfigure cho right_panel
        
        panel_tabs = CTkTabview(
            self.right_panel,
            fg_color=Colors.BG_SECONDARY,
            segmented_button_fg_color=Colors.BG_TERTIARY,
            segmented_button_selected_color=Colors.PRIMARY,
            segmented_button_selected_hover_color=Colors.PRIMARY_HOVER,
            segmented_button_unselected_color=Colors.BG_TERTIARY,
            segmented_button_unselected_hover_color=Colors.BG_ELEVATED,
        )
        panel_tabs.pack(fill="both", expand=True, padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)
        panel_tabs._segmented_button.configure(font=Fonts.BODY_BOLD)

        controls_tab = panel_tabs.add("🎛️ Điều khiển")
        settings_tab = panel_tabs.add("⚙️ Cài đặt")
        logs_tab = panel_tabs.add("📋 Nhật ký")

        # Controls tab with detection controls + quick actions
        controls_container = CTkFrame(controls_tab, fg_color="transparent")
        controls_container.pack(fill="both", expand=True)

        self.detection_controls = DetectionControlsFrame(controls_container, self.camera, self.state)
        self.detection_controls.pack(
            fill="both",
            expand=True,
            padx=Sizes.PADDING_SM,
            pady=(Sizes.PADDING_SM, Sizes.PADDING_SM),
        )

        # Quick Actions card
        quick_card = create_card_frame(controls_container, fg_color=Colors.BG_TERTIARY)
        quick_card.pack(fill="x", padx=Sizes.PADDING_SM, pady=(0, Sizes.PADDING_SM))

        CTkLabel(
            quick_card,
            text="⚡ Tác vụ nhanh",
            font=Fonts.BODY_BOLD,
            text_color=Colors.TEXT_PRIMARY,
        ).pack(anchor="w", padx=Sizes.PADDING_MD, pady=(Sizes.PADDING_MD, Sizes.PADDING_SM))

        btn_row = CTkFrame(quick_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_MD))
        btn_row.grid_columnconfigure((0, 1), weight=1, uniform="qa")

        create_modern_button(
            btn_row,
            text="Tạo lại Embeddings",
            variant="primary",
            icon="🔄",
            command=self.rebuild_all,
        ).grid(row=0, column=0, padx=(0, Sizes.PADDING_SM), pady=Sizes.PADDING_XS, sticky="ew")

        create_modern_button(
            btn_row,
            text="Xóa tất cả dữ liệu",
            variant="danger",
            icon="🗑️",
            command=self.delete_all,
        ).grid(row=0, column=1, padx=(Sizes.PADDING_SM, 0), pady=Sizes.PADDING_XS, sticky="ew")

        # Settings tab
        self.settings_panel = SettingsPanel(settings_tab, state_manager=self.state)
        self.settings_panel.pack(fill="both", expand=True, padx=Sizes.PADDING_SM)

        # Logs tab
        self._init_logs_panel(logs_tab)

    # ================================================================
    # TAB INITIALIZATION
    # ================================================================

    def _init_dashboard_tab(self):
        """Initialize dashboard with cards"""
        # Use centered container
        container = self._create_centered_container(self.dashboard_tab)
        
        stats_container = CTkFrame(container, fg_color="transparent")
        stats_container.pack(fill="x", pady=Sizes.PADDING_MD)
        for i in range(4):
            stats_container.grid_columnconfigure(i, weight=1)

        stats = [
            ("📹", "Camera hoạt động", len(self.camera.cameras) if self.camera else 0, Colors.PRIMARY, "đang online"),
            ("👤", "Khuôn mặt", self._count_persons(), Colors.SUCCESS, "đã đăng ký"),
            ("🔥", "Cảnh báo cháy", 0, Colors.WARNING, "tuần này"),
            ("🚨", "Xâm nhập", 0, Colors.DANGER, "hôm nay"),
        ]

        for i, (icon, title, value, color, subtitle) in enumerate(stats):
            card = self._create_stat_card(stats_container, icon, title, value, color, subtitle)
            card.grid(row=0, column=i, padx=Sizes.PADDING_SM, sticky="nsew")

            card.grid(row=0, column=i, padx=Sizes.PADDING_SM, sticky="nsew")

        content_frame = CTkFrame(container, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, pady=Sizes.PADDING_MD)
        content_frame.grid_columnconfigure(0, weight=2)
        content_frame.grid_columnconfigure(1, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)

        chart_card = create_glass_card(content_frame)
        chart_card.grid(row=0, column=0, padx=Sizes.PADDING_SM, sticky="nsew")
        self.fire_history = FireHistoryWidget(chart_card)
        self.fire_history.pack(fill="both", expand=True)

        activity_card = create_glass_card(content_frame)
        activity_card.grid(row=0, column=1, padx=Sizes.PADDING_SM, sticky="nsew")

        CTkLabel(
            activity_card,
            text="🔔 Hoạt động gần đây",
            font=Fonts.HEADING,
            text_color=Colors.TEXT_PRIMARY,
        ).pack(anchor="w", padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)

        self.activity_feed = ActivityFeedWidget(activity_card, height=400)
        self.activity_feed.pack(fill="both", expand=True, padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_MD))

        log_activity("System initialized", "success")
        log_activity("Face detector ready", "info")
        log_activity("All cameras connected", "success")

    def _create_stat_card(self, parent, icon, title, value, color, subtitle):
        """Create animated stat card"""
        card = create_glass_card(parent)
        card.configure(height=120)

        icon_frame = CTkFrame(card, width=50, height=50, fg_color=color, corner_radius=Sizes.CORNER_RADIUS)
        icon_frame.pack(anchor="w", padx=Sizes.PADDING_MD, pady=(Sizes.PADDING_MD, Sizes.PADDING_SM))
        icon_frame.pack_propagate(False)

        CTkLabel(icon_frame, text=icon, font=("Segoe UI", 24)).pack(expand=True)

        CTkLabel(card, text=str(value), font=Fonts.TITLE_LG, text_color=Colors.TEXT_PRIMARY).pack(
            anchor="w", padx=Sizes.PADDING_MD
        )
        CTkLabel(card, text=title, font=Fonts.CAPTION, text_color=Colors.TEXT_MUTED).pack(
            anchor="w", padx=Sizes.PADDING_MD, pady=(0, 2)
        )
        CTkLabel(card, text=subtitle, font=Fonts.TINY, text_color=Colors.TEXT_MUTED).pack(
            anchor="w", padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_MD)
        )

        return card

    def _init_cameras_tab(self):
        """Initialize cameras view with immersive layout"""
        # Main container with 2 columns: Video (Left, 3/4) | Controls (Right, 1/4)
        self.cameras_tab.grid_columnconfigure(0, weight=3)
        self.cameras_tab.grid_columnconfigure(1, weight=1)
        self.cameras_tab.grid_rowconfigure(0, weight=1)

        # ================= LEFT COLUMN: VIDEO FEED =================
        video_container = create_glass_card(self.cameras_tab)
        video_container.grid(row=0, column=0, sticky="nsew", padx=Sizes.PADDING_SM, pady=Sizes.PADDING_SM)
        
        # Video Label (Placeholder)
        self.video_label = CTkLabel(
            video_container,
            text="📹 Chọn camera để xem trực tiếp",
            font=Fonts.TITLE_MD,
            text_color=Colors.TEXT_MUTED,
        )
        self.video_label.pack(expand=True, fill="both", padx=Sizes.PADDING_LG, pady=Sizes.PADDING_LG)

        # ================= RIGHT COLUMN: CONTROLS =================
        # Use a normal Frame, let the internal lists scroll
        controls_panel = CTkFrame(self.cameras_tab, fg_color="transparent")
        controls_panel.grid(row=0, column=1, sticky="nsew", padx=(0, Sizes.PADDING_SM), pady=Sizes.PADDING_SM)
        
        # Initialize camera selection state (hidden but needed for logic)
        self.camera_sources = list(self.camera.cameras.keys()) if self.camera else []
        initial_cam = self.camera_sources[0] if self.camera_sources else "Không có camera"
        self.selected_camera = StringVar(value=initial_cam)
        
        # Auto-play first camera
        if self.camera_sources:
            self.root.after(500, lambda: self._on_camera_changed(initial_cam))

        # 1. Quick Actions Card
        act_card = create_card_frame(controls_panel, fg_color=Colors.BG_SECONDARY)
        act_card.pack(fill="x", pady=(0, Sizes.PADDING_SM))

        CTkLabel(act_card, text="Tác vụ nhanh", font=Fonts.BODY_BOLD, text_color=Colors.TEXT_SECONDARY).pack(anchor="w", padx=Sizes.PADDING_MD, pady=(Sizes.PADDING_MD, 5))

        btn_grid = CTkFrame(act_card, fg_color="transparent")
        btn_grid.pack(fill="x", padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_MD))
        btn_grid.grid_columnconfigure((0, 1), weight=1)

        create_modern_button(
            btn_grid, text="Ghi hình", variant="success", icon="⏺️", command=self._toggle_recording, height=35
        ).grid(row=0, column=0, padx=(0, 5), sticky="ew")

        create_modern_button(
            btn_grid, text="Chụp ảnh", variant="secondary", icon="📸", command=self._take_snapshot, height=35
        ).grid(row=0, column=1, padx=(5, 0), sticky="ew")
        
        # 2. Unified Camera Control & Health
        # Use a container to ensure it expands properly
        unified_card = create_card_frame(controls_panel, fg_color=Colors.BG_SECONDARY)
        unified_card.pack(fill="both", expand=True, pady=(0, Sizes.PADDING_SM))
        
        self.unified_controls = UnifiedCameraList(
            unified_card, 
            self.camera, 
            self.state,
            on_view_command=self._on_camera_changed,
            on_add_command=self.on_add_camera_clicked
        )
        self.unified_controls.pack(fill="both", expand=True, padx=Sizes.PADDING_SM, pady=Sizes.PADDING_SM)

    def _init_persons_tab(self):
        """Initialize persons management view"""
        container = self._create_centered_container(self.persons_tab)
        persons_container = CTkFrame(container, fg_color="transparent")
        persons_container.pack(fill="both", expand=True)
        persons_container.grid_columnconfigure(0, minsize=300)
        persons_container.grid_columnconfigure(1, weight=1)
        persons_container.grid_rowconfigure(0, weight=1)

        list_panel = create_glass_card(persons_container)
        list_panel.grid(row=0, column=0, sticky="nsew", padx=Sizes.PADDING_SM, pady=Sizes.PADDING_SM)

        search_frame, self.search_entry = create_modern_entry(list_panel, placeholder="Search persons...", icon="🔍")
        search_frame.pack(fill="x", padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)
        self.search_entry.bind("<KeyRelease>", self._on_search_changed)

        CTkLabel(
            list_panel,
            text="Registered Persons",
            font=Fonts.HEADING,
            text_color=Colors.TEXT_PRIMARY,
        ).pack(anchor="w", padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_SM))

        self.person_list_frame = CTkScrollableFrame(
            list_panel,
            fg_color=Colors.BG_PRIMARY,
            corner_radius=Sizes.CORNER_RADIUS
        )
        self.person_list_frame.pack(fill="both", expand=True, padx=Sizes.PADDING_SM, pady=(0, Sizes.PADDING_SM))

        create_modern_button(
            list_panel,
            text="Add New Person",
            variant="primary",
            icon="➕",
            command=self.add_person,
            width=None,
        ).pack(fill="x", padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)

        # Add Rebuild Embeddings to Persons Tab
        create_modern_button(
            list_panel,
            text="Tạo lại Embeddings",
            variant="secondary",
            icon="🔄",
            command=self.rebuild_embeddings,
            width=None,
        ).pack(fill="x", padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_MD))

        # Add Delete Face Data to Persons Tab
        create_modern_button(
            list_panel,
            text="Xóa dữ liệu khuôn mặt",
            variant="danger",
            icon="🗑️",
            command=self.delete_all,
            width=None,
        ).pack(fill="x", padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_MD))

        self.person_details_panel = create_glass_card(persons_container)
        self.person_details_panel.grid(row=0, column=1, sticky="nsew", padx=Sizes.PADDING_SM, pady=Sizes.PADDING_SM)

        self.person_placeholder = CTkLabel(
            self.person_details_panel,
            text="👤 Select a person to view details",
            font=Fonts.BODY,
            text_color=Colors.TEXT_MUTED,
        )
        self.person_placeholder.pack(expand=True)

    def _init_analytics_tab(self):
        """Initialize analytics view"""
        container = self._create_centered_container(self.analytics_tab)
        self.analytics_panel = AnalyticsPanel(
            container,
            state_manager=self.state,
            camera_manager=self.camera
        )
        self.analytics_panel.pack(fill="both", expand=True)

    def _init_gallery_tab(self):
        """Initialize gallery view"""
        container = self._create_centered_container(self.gallery_tab)
        self.gallery_panel = GalleryPanel(container)
        self.gallery_panel.pack(fill="both", expand=True, padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)



    def _init_logs_panel(self, parent):
        """Initialize system logs panel"""
        log_frame = CTkFrame(parent, fg_color="transparent")
        log_frame.pack(fill="both", expand=True)

        CTkLabel(
            log_frame,
            text="System Logs",
            font=Fonts.HEADING,
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w", pady=(0, Sizes.PADDING_SM))

        self.logs_widget = SystemLogsWidget(log_frame)
        self.logs_widget.pack(fill="both", expand=True)

        log_system("System started", "info")
        log_system("Face detector initialized", "success")
        log_system("Connected to camera 0", "info")
    
    def _init_settings_tab(self):
        """Initialize settings tab with system configuration"""
        wrapper = self._create_centered_container(self.settings_tab)
        container = CTkFrame(wrapper, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=Sizes.PADDING_LG, pady=Sizes.PADDING_LG)
        
        # Settings Panel Section
        self.settings_panel = SettingsPanel(container, state_manager=self.state)
        self.settings_panel.pack(fill="both", expand=True, padx=0, pady=(0, Sizes.PADDING_MD))

    # ================================================================
    # VIEW SWITCHING
    # ================================================================

    def rebuild_embeddings(self):
        """Rebuild face embeddings"""
        try:
            count = self.face_detector.rebuild_embeddings()
            CTkMessagebox(
                title="Success",
                message=f"Đã tạo lại embeddings cho {count} khuôn mặt",
                icon="check"
            )
            self.populate_person_list()
            log_activity("Rebuilt face embeddings", "success")
        except Exception as e:
            CTkMessagebox(
                title="Error",
                message=f"Lỗi khi tạo lại embeddings: {e}",
                icon="cancel"
            )
    
    def confirm_clear_data(self):
        """Confirm before clearing all data"""
        from CTkMessagebox import CTkMessagebox
        
        result = CTkMessagebox(
            title="Xác nhận",
            message="Bạn có chắc muốn xóa TẤT CẢ dữ liệu?\nHành động này KHÔNG thể hoàn tác!",
            icon="warning",
            option_1="Hủy",
            option_2="Xóa",
        )
        
        if result.get() == "Xóa":
            self.clear_all_data()
    
    def clear_all_data(self):
        """Clear all face data and recordings"""
        try:
            import shutil
            from config import settings
            
            # Clear face data
            data_dir = settings.paths.data_dir
            if data_dir.exists():
                shutil.rmtree(data_dir)
                data_dir.mkdir(parents=True, exist_ok=True)
            
            # Clear recordings
            tmp_dir = settings.paths.tmp_dir
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
                tmp_dir.mkdir(parents=True, exist_ok=True)
            
            # Rebuild embeddings
            self.face_detector.rebuild_embeddings()
            self.populate_person_list()
            
            CTkMessagebox(
                title="Success",
                message="Đã xóa tất cả dữ liệu thành công",
                icon="check"
            )
            log_activity("Cleared all data", "warning")
        except Exception as e:
            CTkMessagebox(
                title="Error",
                message=f"Lỗi khi xóa dữ liệu: {e}",
                icon="cancel"
            )

    def _show_dashboard(self):
        """Show dashboard tab"""
        try:
            self.current_view = "dashboard"
            self.main_tabs.set("📊 Tổng quan")
        except Exception as e:
            print(f"ERROR: Failed to show dashboard: {e}")
        log_activity("Switched to Dashboard view", "info")

    def _show_cameras(self):
        self.current_view = "cameras"
        self.main_tabs.set("📹 Camera")
        log_activity("Switched to Live Cameras view", "info")

    def _show_persons(self):
        self.current_view = "persons"
        self.main_tabs.set("👥 Face Database")
        self.populate_person_list()
        log_activity("Switched to Face Database view", "info")

    def _show_analytics(self):
        self.current_view = "analytics"
        self.main_tabs.set("📈 Analytics")
        if hasattr(self, "analytics_panel"):
            self.analytics_panel.refresh_data()
        log_activity("Switched to Analytics view", "info")

    def _show_security(self):
        self.current_view = "security"
        self.main_tabs.set("🔐 Security")
        log_activity("Switched to Security view", "info")

    # ================================================================
    # PERSONS MANAGEMENT
    # ================================================================

    def populate_person_list(self, search_term: str = ""):
        """Populate the persons list with saved faces"""
        # Check if person_list_frame hashas been created yet
        if not hasattr(self, 'person_list_frame') or self.person_list_frame is None:
            print("DEBUG: person_list_frame not created yet, skipping populate")
            return
            
        for widget in self.person_list_frame.winfo_children():
            widget.destroy()

        data_dir = settings.paths.data_dir
        print(f"DEBUG: data_dir = {data_dir}")
        if not data_dir.exists():
            print(f"DEBUG: data_dir does not exist!")
            CTkLabel(
                self.person_list_frame,
                text="No persons registered yet",
                font=Fonts.BODY,
                text_color=Colors.TEXT_MUTED,
            ).pack(pady=Sizes.PADDING_LG)
            return

        persons = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
        print(f"DEBUG: Found {len(persons)} persons: {persons}")

        if search_term:
            persons = [p for p in persons if search_term.lower() in p.lower()]

        if not persons:
            print(f"DEBUG: No persons to show (search_term={search_term})")
            CTkLabel(
                self.person_list_frame,
                text=f"No persons found for '{search_term}'" if search_term else "No persons registered yet",
                font=Fonts.BODY,
                text_color=Colors.TEXT_MUTED,
            ).pack(pady=Sizes.PADDING_LG)
            return

        print(f"DEBUG: Creating cards for {len(persons)} persons")
        for person_name in persons:
            card = self._create_person_card(person_name)
            card.pack(fill="x", pady=Sizes.PADDING_XS)

    def _create_person_card(self, name):
        """Create modern person card"""
        card = CTkFrame(
            self.person_list_frame,
            fg_color=Colors.BG_TERTIARY,
            corner_radius=Sizes.CORNER_RADIUS,
            height=60,
            cursor="hand2",
        )

        def on_enter(_):
            card.configure(fg_color=Colors.BG_ELEVATED)

        def on_leave(_):
            card.configure(fg_color=Colors.BG_TERTIARY)

        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)
        card.bind("<Button-1>", lambda _: self.select_person(name))

        avatar = CTkFrame(card, width=40, height=40, fg_color=Colors.PRIMARY, corner_radius=20)
        avatar.pack(side="left", padx=Sizes.PADDING_SM, pady=Sizes.PADDING_SM)
        avatar.pack_propagate(False)

        CTkLabel(avatar, text=name[0].upper(), font=Fonts.BODY_BOLD, text_color=Colors.TEXT_PRIMARY).pack(expand=True)

        CTkLabel(card, text=name, font=Fonts.BODY, text_color=Colors.TEXT_PRIMARY).pack(
            side="left", padx=Sizes.PADDING_SM
        )

        person_dir = settings.paths.data_dir / name
        image_count = len(list(person_dir.glob("*.jpg"))) + len(list(person_dir.glob("*.png")))

        CTkLabel(
            card,
            text=f"{image_count} images",
            font=Fonts.CAPTION,
            text_color=Colors.TEXT_MUTED,
        ).pack(side="right", padx=Sizes.PADDING_MD)

        return card

    def select_person(self, name):
        """Select and display person details"""
        self.current_person = name

        for widget in self.person_details_panel.winfo_children():
            widget.destroy()

        header = CTkFrame(self.person_details_panel, fg_color="transparent")
        header.pack(fill="x", padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)

        CTkLabel(header, text=name, font=Fonts.TITLE_MD, text_color=Colors.TEXT_PRIMARY).pack(side="left")

        actions = CTkFrame(header, fg_color="transparent")
        actions.pack(side="right")

        create_modern_button(
            actions,
            text="Add Photo",
            variant="primary",
            size="small",
            icon="➕",
            command=lambda: self.add_image_for_person(name),
        ).pack(side="left", padx=2)

        create_modern_button(
            actions,
            text="Delete",
            variant="danger",
            size="small",
            icon="🗑️",
            command=lambda: self.delete_person(name),
        ).pack(side="left", padx=2)

        self._load_person_gallery(name)

        log_activity(f"Viewed person: {name}", "info")
        log_system(f"Person profile accessed: {name}", "info")

    def _load_person_gallery(self, name):
        """Load person image gallery"""
        gallery_frame = CTkScrollableFrame(
            self.person_details_panel,
            fg_color=Colors.BG_PRIMARY,
            corner_radius=Sizes.CORNER_RADIUS,
        )
        gallery_frame.pack(fill="both", expand=True, padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_MD))

        person_dir = settings.paths.data_dir / name
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))

        if not images:
            CTkLabel(
                gallery_frame,
                text="No images yet",
                font=Fonts.BODY,
                text_color=Colors.TEXT_MUTED
            ).pack(pady=Sizes.PADDING_LG)
            return

        # Loading indicator
        loading_label = CTkLabel(gallery_frame, text="Loading images...", text_color=Colors.TEXT_MUTED)
        loading_label.pack(pady=10)

        def load_images_thread():
            columns = 3
            
            for i, img_path in enumerate(images):
                if not gallery_frame.winfo_exists():
                    return
                    
                row = i // columns
                col = i % columns
                
                try:
                    # Decrypt image first
                    decrypted_img = security_manager.decrypt_image(img_path)
                    if decrypted_img is not None:
                        # Convert from BGR to RGB for PIL
                        img_rgb = cv2.cvtColor(decrypted_img, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(img_rgb)
                    else:
                        # Fallback to normal loading if not encrypted
                        img = Image.open(img_path)
                    
                    img.thumbnail((150, 150))
                    ctk_img = CTkImage(img, size=(150, 150))

                    def add_image_to_ui(r=row, c=col, image=ctk_img):
                        if not gallery_frame.winfo_exists(): return
                        
                        # Remove loading label on first image
                        if loading_label.winfo_exists():
                            loading_label.destroy()

                        img_frame = CTkFrame(
                            gallery_frame,
                            fg_color=Colors.BG_TERTIARY,
                            corner_radius=Sizes.CORNER_RADIUS
                        )
                        img_frame.grid(row=r, column=c, padx=Sizes.PADDING_SM, pady=Sizes.PADDING_SM)

                        label = CTkLabel(img_frame, text="", image=image)
                        label.pack(padx=2, pady=2)
                        label.image = image  # Keep reference

                    # Schedule UI update
                    self.root.after(0, add_image_to_ui)
                    
                    # Small sleep to yield to UI thread if loading many images
                    if i % 5 == 0:
                        time.sleep(0.01)

                except Exception as e:
                    logger.error("Failed to load image %s: %s", img_path, e)

        threading.Thread(target=load_images_thread, daemon=True).start()

    def add_person(self):
        """Add new person với modern dialog"""
        dialog = CTk()
        dialog.title("Add New Person")
        dialog.geometry("400x250")  # Increased from 200
        dialog.resizable(False, False)  # Không cho phóng to
        dialog.configure(fg_color=Colors.BG_PRIMARY)

        try:
            dialog.transient(self.root)
            dialog.grab_set()
        except Exception:
            pass

        content = CTkFrame(dialog, fg_color=Colors.BG_SECONDARY, corner_radius=Sizes.CORNER_RADIUS_LG)
        content.pack(fill="both", expand=True, padx=20, pady=20)

        CTkLabel(
            content,
            text="Add New Person",
            font=Fonts.HEADING,
            text_color=Colors.TEXT_PRIMARY
        ).pack(pady=(20, 10))

        entry_frame, name_entry = create_modern_entry(content, placeholder="Enter person's name")
        entry_frame.pack(pady=10, padx=20)

        btn_frame = CTkFrame(content, fg_color="transparent")
        btn_frame.pack(pady=20)

        def on_add():
            name = name_entry.get().strip()
            if not name:
                CTkMessagebox(title="Error", message="Please enter a name", icon="warning")
                return

            person_dir = settings.paths.data_dir / name
            if person_dir.exists():
                CTkMessagebox(title="Error", message=f"Person '{name}' already exists", icon="cancel")
                return

            dialog.destroy()
            self.root.after(100, lambda: self._add_person_select_image(name))

        create_modern_button(btn_frame, text="Add", variant="primary", command=on_add).pack(side="left", padx=5)
        create_modern_button(btn_frame, text="Cancel", variant="secondary", command=dialog.destroy).pack(
            side="left", padx=5
        )

        name_entry.focus()
        dialog.mainloop()

    def _add_person_select_image(self, name):
        """Helper to select image after dialog closes"""
        img_paths = filedialog.askopenfilenames(
            title=f"Select photos for {name} (can select multiple)",
            filetypes=[("Image Files", "*.jpg *.png *.jpeg")],
        )

        if not img_paths:
            return

        person_dir = settings.paths.data_dir / name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        # Load, encrypt, and save all selected images
        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is not None:
                dest_path = person_dir / Path(img_path).name
                security_manager.save_encrypted_image(dest_path, img)
                saved_count += 1
            else:
                # Fallback: copy without encryption if cv2 fails
                try:
                    shutil.copy(img_path, person_dir)
                    saved_count += 1
                except Exception as e:
                    print(f"ERROR: Failed to copy {img_path}: {e}")

        if saved_count == 0:
            CTkMessagebox(title="Error", message="No images were saved successfully", icon="warning")
            return

        self.face_detector.rebuild_embeddings()
        self.populate_person_list()
        self.select_person(name)

        CTkMessagebox(
            title="Success", 
            message=f"Added '{name}' with {saved_count} image(s) successfully", 
            icon="check"
        )

        log_activity(f"Added new person: {name} with {saved_count} images", "success")
        log_system(f"New person registered: {name}", "success")
        self.log_detection_event("new_registration", "system")

    def add_image_for_person(self, name):
        """Add image for existing person"""
        img_path = filedialog.askopenfilenames(
            title=f"Select photo for {name}",
            filetypes=[("Image Files", "*.jpg *.png *.jpeg")],
        )

        if not img_path:
            return

        person_dir = settings.paths.data_dir / name
        
        # Load image, encrypt, and save
        img = cv2.imread(img_path)
        if img is not None:
            dest_path = person_dir / Path(img_path).name
            security_manager.save_encrypted_image(dest_path, img)
        else:
            shutil.copy(img_path, person_dir)

        self.face_detector.rebuild_embeddings()
        self._load_person_gallery(name)

        CTkMessagebox(title="Success", message=f"Added photo for '{name}'", icon="check")
        log_activity(f"Added photo for: {name}", "info")

    def delete_person(self, name):
        """Delete person với confirmation"""
        result = CTkMessagebox(
            title="Delete Person",
            message=f"Are you sure you want to delete '{name}' and all associated data?",
            icon="question",
            option_1="Cancel",
            option_2="Delete",
            option_3="Delete All Data",
        ).get()

        if result == "Cancel":
            return

        try:
            person_dir = settings.paths.data_dir / name
            shutil.rmtree(person_dir)

            self.face_detector.rebuild_embeddings()
            self.populate_person_list()

            for widget in self.person_details_panel.winfo_children():
                widget.destroy()
            self.person_placeholder.pack(expand=True)

            CTkMessagebox(title="Success", message=f"Deleted '{name}'", icon="check")

            log_activity(f"Deleted person: {name}", "warning")
            log_system(f"Person removed from database: {name}", "warning")
        except Exception as e:
            CTkMessagebox(title="Error", message=f"Failed to delete: {e}", icon="cancel")

    def rebuild_all(self):
        """Rebuild all embeddings với progress"""
        result = CTkMessagebox(
            title="Rebuild Database",
            message="This will re-encode all faces. Continue?",
            icon="question",
            option_1="Cancel",
            option_2="Rebuild",
        ).get()

        if result != "Rebuild":
            return

        progress_dialog = CTk()
        progress_dialog.title("Rebuilding...")
        progress_dialog.geometry("400x150")
        progress_dialog.configure(fg_color=Colors.BG_PRIMARY)

        try:
            progress_dialog.transient(self.root)
        except Exception:
            pass
        try:
            progress_dialog.grab_set()
        except Exception:
            pass

        content = CTkFrame(progress_dialog, fg_color=Colors.BG_SECONDARY, corner_radius=Sizes.CORNER_RADIUS_LG)
        content.pack(fill="both", expand=True, padx=20, pady=20)

        CTkLabel(
            content,
            text="Rebuilding face database...",
            font=Fonts.BODY,
            text_color=Colors.TEXT_PRIMARY
        ).pack(pady=20)

        progress = CTkProgressBar(
            content,
            width=300,
            height=20,
            corner_radius=10,
            progress_color=Colors.PRIMARY,
            fg_color=Colors.BG_TERTIARY,
        )
        progress.pack(pady=10)
        progress.set(0)

        def rebuild_thread():
            try:
                count = self.face_detector.rebuild_embeddings()
                
                def cleanup_and_show_result():
                    try:
                        # Update progress in main thread
                        progress.set(1)
                        if progress_dialog.winfo_exists():
                            progress_dialog.destroy()
                    except Exception as e:
                        logger.warning("Error destroying progress dialog: %s", e)

                    CTkMessagebox(
                        title="Success",
                        message=f"Rebuilt database with {count} faces",
                        icon="check",
                    )
                    log_activity(f"Rebuilt database: {count} faces", "success")

                self.root.after(500, cleanup_and_show_result)
            except Exception as e:
                logger.error("Rebuild error: %s", e)
                error_msg = str(e)

                def show_error():
                    try:
                        if progress_dialog.winfo_exists():
                            progress_dialog.destroy()
                    except Exception:
                        pass

                    CTkMessagebox(
                        title="Error",
                        message=f"Rebuild failed: {error_msg}",
                        icon="cancel",
                    )

                self.root.after(100, show_error)


        threading.Thread(target=rebuild_thread, daemon=True).start()

        try:
            progress_dialog.wait_window()
        except Exception as e:
            logger.warning("wait_window failed: %s", e)

    def delete_all(self):
        """Delete all data với multiple confirmations"""
        result1 = CTkMessagebox(
            title="⚠️ WARNING",
            message="Delete ALL face data?\nThis CANNOT be undone!",
            icon="warning",
            option_1="Cancel",
            option_2="Continue",
        ).get()

        if result1 != "Continue":
            return

        confirm_dialog = CTk()
        confirm_dialog.title("Final Confirmation")
        confirm_dialog.geometry("400x200")
        confirm_dialog.configure(fg_color=Colors.BG_PRIMARY)

        try:
            confirm_dialog.transient(self.root)
        except _tkinter.TclError:
            confirm_dialog.transient(None)
        confirm_dialog.grab_set()

        content = CTkFrame(confirm_dialog, fg_color=Colors.BG_SECONDARY, corner_radius=Sizes.CORNER_RADIUS_LG)
        content.pack(fill="both", expand=True, padx=20, pady=20)

        CTkLabel(
            content,
            text="Type 'DELETE ALL' to confirm:",
            font=Fonts.BODY,
            text_color=Colors.DANGER,
        ).pack(pady=20)

        entry_frame, confirm_entry = create_modern_entry(content, placeholder="Type here to confirm")
        entry_frame.pack(pady=10, padx=20)

        def on_confirm():
            if confirm_entry.get() != "DELETE ALL":
                CTkMessagebox(title="Cancelled", message="Delete operation cancelled", icon="info")
                confirm_dialog.destroy()
                return

            try:
                data_dir = settings.paths.data_dir
                if data_dir.exists():
                    shutil.rmtree(data_dir)
                    data_dir.mkdir(parents=True, exist_ok=True)

                for f in [settings.paths.embedding_file, settings.paths.names_file]:
                    if f.exists():
                        f.unlink()

                self.face_detector.known_embeddings = []
                self.face_detector.known_names = []
                self.populate_person_list()

                confirm_dialog.destroy()
                CTkMessagebox(title="Success", message="All data deleted", icon="check")

                log_activity("Deleted all face data", "error")
                log_system("All face data purged from system", "error")
            except Exception as e:
                CTkMessagebox(title="Error", message=f"Failed to delete: {e}", icon="cancel")
                confirm_dialog.destroy()

        btn_frame = CTkFrame(content, fg_color="transparent")
        btn_frame.pack(pady=20)

        create_modern_button(btn_frame, text="Delete All", variant="danger", command=on_confirm).pack(
            side="left", padx=5
        )
        create_modern_button(btn_frame, text="Cancel", variant="secondary", command=confirm_dialog.destroy).pack(
            side="left", padx=5
        )

        confirm_entry.focus()
        confirm_dialog.mainloop()

    # ================================================================
    # HELPERS & CAMERA OPS
    # ================================================================

    def _on_tab_change(self):
        """Handle tab change event"""
        try:
            selected_tab = self.main_tabs.get()
            if "Camera" in selected_tab:
                self._show_cameras()
            elif "Tổng quan" in selected_tab:
                self._show_dashboard()
            elif "Thư viện" in selected_tab:
                self.current_view = "gallery"
                log_activity("Switched to Gallery view", "info")
            elif "Database" in selected_tab or "Khuôn mặt" in selected_tab:
                self._show_persons()
            elif "Analytics" in selected_tab:
                self._show_analytics()
            elif "Cài đặt" in selected_tab:
                self.current_view = "settings"
                log_activity("Switched to Settings view", "info")
        except Exception as e:
            logger.error(f"Error handling tab change: {e}")

    def _count_persons(self):
        """Count registered persons"""
        data_dir = settings.paths.data_dir
        if not data_dir.exists():
            return 0
        return len([d for d in data_dir.iterdir() if d.is_dir()])

    def _take_snapshot(self):
        """Take camera snapshot"""
        selected_cam = self.selected_camera.get()
        if not selected_cam or selected_cam == "No cameras":
            CTkMessagebox(title="Error", message="No camera selected", icon="warning")
            return

        cam = self.camera.get_camera(selected_cam)
        if not cam:
            CTkMessagebox(title="Error", message="Camera not found", icon="cancel")
            return

        ret, frame = cam.read_raw()
        if not ret or frame is None:
            CTkMessagebox(title="Error", message="Failed to capture snapshot", icon="cancel")
            return

        import uuid
        img_path = settings.paths.tmp_dir / f"snapshot_{uuid.uuid4().hex}.jpg"
        # Save encrypted image
        security_manager.save_encrypted_image(img_path, frame)

        CTkMessagebox(title="Success", message=f"Snapshot saved to:\n{img_path}", icon="check")
        log_activity(f"Snapshot taken from {selected_cam}", "info")

    def _toggle_recording(self):
        """Toggle recording state"""
        dialog = CTk()
        dialog.title("Recording Control")
        dialog.geometry("600x500")
        dialog.configure(fg_color=Colors.BG_PRIMARY)

        try:
            dialog.transient(self.root)
            dialog.grab_set()
        except Exception:
            pass

        recording_panel = RecordingPanel(dialog, camera_manager=self.camera, recorder=self.recorder)
        recording_panel.pack(fill="both", expand=True)

        create_modern_button(dialog, text="Close", variant="secondary", command=dialog.destroy).pack(
            pady=Sizes.PADDING_MD
        )

        log_activity("Opened Recording Control", "info")
        dialog.mainloop()

    def _on_camera_changed(self, choice):
        """Handle camera selection change"""
        if hasattr(self, "selected_camera"):
            self.selected_camera.set(choice)
        logger.info("Camera changed to: %s", choice)
        log_activity(f"Switched to camera: {choice}", "info")

    def _on_search_changed(self, _event):
        """Handle search input change"""
        search_term = self.search_entry.get()
        self.populate_person_list(search_term)

    # ================================================================
    # ANALYTICS LOGGING HELPERS
    # ================================================================

    def log_detection_event(self, detection_type: str, source_id: str):
        """Log detection event to analytics"""
        if hasattr(self, "analytics_panel"):
            self.analytics_panel.log_detection(detection_type, source_id)
        log_activity(f"Detected {detection_type} on camera {source_id}", "detection")

    def log_alert_event(self, alert_type: str, source_id: str, resolved: bool = False):
        """Log alert event"""
        if hasattr(self, "analytics_panel"):
            self.analytics_panel.log_alert(alert_type, source_id, resolved)
        status = "resolved" if resolved else "triggered"
        log_activity(f"Alert {status}: {alert_type} on camera {source_id}", "alert")

    def log_fire_event(self, event_type: str, source_id: str, area: float, confidence: float):
        """Log fire detection event"""
        if hasattr(self, "fire_history"):
            self.fire_history.log_fire_event(event_type, source_id, area, confidence)
        log_activity(f"Fire detection: {event_type} at camera {source_id}", "alert")

    # ================================================================
    # VIDEO LOOP & SYNC
    # ================================================================

    def _start_animations(self):
        """Start UI animations"""
        self.animation_running = True
        self._sync_detection_switch()

    def _sync_detection_switch(self):
        """Sync detection switches"""
        if hasattr(self, "detection_controls"):
            self.detection_controls.sync_all_switches()
        self.root.after(1000, self._sync_detection_switch)


    def _start_video_feed(self):
        """Start video feed với smooth updates"""
        def update():
            if not hasattr(self, "video_label"):
                return
            
            # DEBUG LOGS
            # print(f"DEBUG: Video update loop. Current view: {self.current_view}")
            
            if self.current_view != "cameras":
                self.root.after(100, update)
                return

            selected_cam = self.selected_camera.get()
            # print(f"DEBUG: Selected camera: {selected_cam}")
            
            if not selected_cam or selected_cam == "No cameras":
                self.video_label.configure(text="📹 No camera available", image=None)
                self.root.after(1000, update)
                return

            cam = self.camera.get_camera(selected_cam)
            if not cam:
                print(f"DEBUG: Camera object not found for {selected_cam}")
                self.video_label.configure(text=f"❌ Camera not found: {selected_cam}", image=None)
                self.root.after(1000, update)
                return

            try:
                ret, frame = cam.read()
                
                if ret and frame is not None:
                    # Resize frame BEFORE converting to PIL for speed (OpenCV is faster at resizing)
                    # Calculate target size maintaining aspect ratio if needed, or just fit to box
                    # Here we just resize to display size for maximum performance
                    target_w, target_h = Sizes.VIDEO_FEED_WIDTH, Sizes.VIDEO_FEED_HEIGHT
                    
                    # Use INTER_LINEAR (Bilinear) for speed
                    frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    if self.brightness_value != 1.0:
                        pil_image = ImageEnhance.Brightness(pil_image).enhance(self.brightness_value)

                    # CTkImage needs size argument
                    ctk_image = CTkImage(pil_image, size=(target_w, target_h))
                    
                    if self.video_label.winfo_exists():
                        self.video_label.configure(text="", image=ctk_image)
                        self.video_label.image = ctk_image  # Keep reference
                    else:
                        return
                else:
                    if self.video_label.winfo_exists():
                        self.video_label.configure(text="🔄 Reconnecting...", image=None)
                    else:
                        return
            except Exception as e:
                logger.error("Video feed error: %s", e)
                if self.video_label.winfo_exists():
                    self.video_label.configure(text="❌ Feed error", image=None)
                else:
                    return

            # Adaptive refresh rate
            refresh_rate = 33 if self.state.is_person_detection_enabled() else 40
            try:
                self.root.after(refresh_rate, update)
            except Exception as e:
                logger.error("Could not schedule video update: %s", e)

        update()

    def show_video_view(self):
        """Show video view"""
        self._show_cameras()

    def on_add_camera_clicked(self):
        """Handle add camera button"""
        dialog = CTk()
        dialog.title("Add Camera")
        dialog.geometry("400x250")  # Increased from 200
        dialog.resizable(False, False)  # Không cho phóng to
        dialog.configure(fg_color=Colors.BG_PRIMARY)

        try:
            dialog.transient(self.root)
            dialog.grab_set()
        except Exception:
            pass

        content = CTkFrame(dialog, fg_color=Colors.BG_SECONDARY, corner_radius=Sizes.CORNER_RADIUS_LG)
        content.pack(fill="both", expand=True, padx=20, pady=20)

        CTkLabel(
            content,
            text="Add New Camera",
            font=Fonts.HEADING,
            text_color=Colors.TEXT_PRIMARY
        ).pack(pady=(20, 10))

        entry_frame, source_entry = create_modern_entry(content, placeholder="Camera ID or RTSP URL")
        entry_frame.pack(pady=10, padx=20)

        def on_add():
            source = source_entry.get().strip()
            if not source:
                CTkMessagebox(title="Error", message="Please enter camera source", icon="warning")
                return

            try:
                success, message = self.camera.add_new_camera(source)
                if success:
                    dialog.destroy()
                    self._update_camera_list(source)
                    CTkMessagebox(title="Success", message=message, icon="check")
                    log_activity(f"Added camera: {source}", "success")
                else:
                    CTkMessagebox(title="Error", message=message, icon="cancel")
            except Exception as e:
                CTkMessagebox(title="Error", message=str(e), icon="cancel")

        btn_frame = CTkFrame(content, fg_color="transparent")
        btn_frame.pack(pady=20)

        create_modern_button(btn_frame, text="Add", variant="primary", command=on_add).pack(side="left", padx=5)
        create_modern_button(btn_frame, text="Cancel", variant="secondary", command=dialog.destroy).pack(
            side="left", padx=5
        )

        source_entry.focus()
        dialog.mainloop()

    def _update_camera_list(self, newly_added_source=None):
        """Update camera list"""
        self.camera_sources = list(self.camera.cameras.keys())

        if hasattr(self, "camera_menu"):
            self.camera_menu.configure(values=self.camera_sources or ["No cameras"])

        if newly_added_source and newly_added_source in self.camera_sources:
            self.selected_camera.set(newly_added_source)

    # ================================================================
    # RUN GUI
    # ================================================================


# ================================================================
# RUN GUI
# ================================================================

def run_gui(camera, face_detector, state_manager):
    """Run modern GUI"""
    root = CTk()
    app = ModernFaceManagerGUI(root, camera, face_detector, state_manager)

    log_system("Guardian GUI fully loaded", "success")
    log_activity("System ready for operation", "success")

    root.mainloop()
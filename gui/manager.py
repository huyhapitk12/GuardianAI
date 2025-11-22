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
from .control_panels import SettingsPanel, SecurityPanel, RecordingPanel
from .analytics_panel import AnalyticsPanel
from .detection_controls import DetectionControlsFrame
from .widgets import (
    GalleryPanel,
    ActivityFeedWidget, SystemLogsWidget, log_activity, log_system,
    CameraHealthWidget, FireHistoryWidget
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
        self.simple_mode = True  # Default to Simple Mode for non-tech users

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
        main_container.grid_columnconfigure(1, weight=0, minsize=Sizes.RIGHT_SIDEBAR_WIDTH)
        main_container.grid_rowconfigure(0, weight=1)

        self._create_content_area(main_container)
        
        # Only show right panel in Advanced Mode
        if not self.simple_mode:
            self._create_right_panel(main_container)

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
        )
        self.main_tabs.pack(fill="both", expand=True, padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)
        self.main_tabs._segmented_button.configure(font=Fonts.BODY_BOLD, height=40)

        # Add all tabs
        # Add tabs based on mode
        if self.simple_mode:
            self.dashboard_tab = self.main_tabs.add("📊 Tổng quan")
            self.cameras_tab = self.main_tabs.add("📹 Camera")
            self.gallery_tab = self.main_tabs.add("🎞️ Thư viện")
            
            self._init_simple_dashboard_tab()
            self._init_cameras_tab()
            self._init_gallery_tab()
        else:
            self.dashboard_tab = self.main_tabs.add("📊 Tổng quan")
            self.cameras_tab = self.main_tabs.add("📹 Camera")
            self.persons_tab = self.main_tabs.add("👥 Khuôn mặt")
            self.analytics_tab = self.main_tabs.add("📈 Phân tích")
            self.gallery_tab = self.main_tabs.add("🎞️ Thư viện")
            self.security_tab = self.main_tabs.add("🔐 Bảo mật")

            # Initialize tab contents
            self._init_dashboard_tab()
            self._init_cameras_tab()
            self._init_persons_tab()
            self._init_analytics_tab()
            self._init_gallery_tab()
            self._init_security_tab()
            
        # Mode toggle button in header (using absolute positioning or pack in a frame)
        self._create_mode_toggle()

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
        stats_container = CTkFrame(self.dashboard_tab, fg_color="transparent")
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

        content_frame = CTkFrame(self.dashboard_tab, fg_color="transparent")
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
        """Initialize cameras view"""
        self.camera_grid = CTkFrame(self.cameras_tab, fg_color="transparent")
        self.camera_grid.pack(fill="both", expand=True, padx=Sizes.PADDING_SM)

        self.camera_grid.grid_columnconfigure(0, weight=2)
        self.camera_grid.grid_columnconfigure(1, weight=1)
        self.camera_grid.grid_rowconfigure(0, weight=1)

        video_container = create_glass_card(self.camera_grid)
        video_container.grid(row=0, column=0, sticky="nsew", pady=Sizes.PADDING_SM, padx=(0, Sizes.PADDING_SM))

        self.video_label = CTkLabel(
            video_container,
            text="📹 Chọn camera để xem trực tiếp",
            font=Fonts.BODY,
            text_color=Colors.TEXT_MUTED,
        )
        self.video_label.pack(expand=True, fill="both", padx=Sizes.PADDING_LG, pady=Sizes.PADDING_LG)

        health_container = CTkFrame(self.camera_grid, fg_color="transparent")
        health_container.grid(row=0, column=1, sticky="nsew", pady=Sizes.PADDING_SM, padx=(Sizes.PADDING_SM, 0))

        self.camera_health = CameraHealthWidget(health_container, self.camera)
        self.camera_health.pack(fill="both", expand=True)

        controls_frame = CTkFrame(self.cameras_tab, fg_color="transparent", height=60)
        controls_frame.pack(fill="x", padx=Sizes.PADDING_SM, pady=Sizes.PADDING_SM)

        self.camera_sources = list(self.camera.cameras.keys()) if self.camera else []
        self.selected_camera = StringVar(value=self.camera_sources[0] if self.camera_sources else "Không có camera")

        self.camera_menu = CTkOptionMenu(
            controls_frame,
            variable=self.selected_camera,
            values=self.camera_sources or ["Không có camera"],
            font=Fonts.BODY,
            text_color=Colors.TEXT_PRIMARY,
            fg_color=Colors.BG_TERTIARY,
            button_color=Colors.PRIMARY,
            button_hover_color=Colors.PRIMARY_HOVER,
            dropdown_fg_color=Colors.BG_SECONDARY,
            dropdown_hover_color=Colors.BG_TERTIARY,
            corner_radius=Sizes.CORNER_RADIUS,
            width=200,
            height=40,
            command=self._on_camera_changed,
        )
        self.camera_menu.pack(side="left", padx=Sizes.PADDING_SM)

        create_modern_button(
            controls_frame,
            text="Xem trực tiếp",
            variant="primary",
            icon="👁️",
            command=self.show_video_view,
        ).pack(side="left", padx=Sizes.PADDING_SM)

        create_modern_button(
            controls_frame,
            text="Chụp ảnh",
            variant="secondary",
            icon="📸",
            command=self._take_snapshot,
        ).pack(side="left", padx=Sizes.PADDING_SM)

        create_modern_button(
            controls_frame,
            text="Ghi hình",
            variant="success",
            icon="⏺️",
            command=self._toggle_recording,
        ).pack(side="left", padx=Sizes.PADDING_SM)

    def _init_persons_tab(self):
        """Initialize persons management view"""
        persons_container = CTkFrame(self.persons_tab, fg_color="transparent")
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
        self.analytics_panel = AnalyticsPanel(
            self.analytics_tab,
            state_manager=self.state,
            camera_manager=self.camera
        )
        self.analytics_panel.pack(fill="both", expand=True)

    def _init_gallery_tab(self):
        """Initialize gallery view"""
        self.gallery_panel = GalleryPanel(self.gallery_tab)
        self.gallery_panel.pack(fill="both", expand=True, padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)

    def _init_security_tab(self):
        """Initialize security tab"""
        self.security_panel = SecurityPanel(self.security_tab, state_manager=self.state)
        self.security_panel.pack(fill="both", expand=True, padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)

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

    # ================================================================
    # VIEW SWITCHING
    # ================================================================

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
            return
            
        for widget in self.person_list_frame.winfo_children():
            widget.destroy()

        data_dir = settings.paths.data_dir
        if not data_dir.exists():
            CTkLabel(
                self.person_list_frame,
                text="No persons registered yet",
                font=Fonts.BODY,
                text_color=Colors.TEXT_MUTED,
            ).pack(pady=Sizes.PADDING_LG)
            return

        persons = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

        if search_term:
            persons = [p for p in persons if search_term.lower() in p.lower()]

        if not persons:
            CTkLabel(
                self.person_list_frame,
                text=f"No persons found for '{search_term}'" if search_term else "No persons registered yet",
                font=Fonts.BODY,
                text_color=Colors.TEXT_MUTED,
            ).pack(pady=Sizes.PADDING_LG)
            return

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

        columns = 3
        for i, img_path in enumerate(images):
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

                img_frame = CTkFrame(
                    gallery_frame,
                    fg_color=Colors.BG_TERTIARY,
                    corner_radius=Sizes.CORNER_RADIUS
                )
                img_frame.grid(row=row, column=col, padx=Sizes.PADDING_SM, pady=Sizes.PADDING_SM)

                label = CTkLabel(img_frame, text="", image=ctk_img)
                label.pack(padx=2, pady=2)
                label.image = ctk_img  # Keep reference
            except Exception as e:
                logger.error("Failed to load image: %s", e)

    def add_person(self):
        """Add new person với modern dialog"""
        dialog = CTk()
        dialog.title("Add New Person")
        dialog.geometry("400x200")
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

        entry_frame, name_entry = create_modern_entry(content, placeholder="Enter person's name", icon="👤")
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
        img_path = filedialog.askopenfilename(
            title=f"Select first photo for {name}",
            filetypes=[("Image Files", "*.jpg *.png *.jpeg")],
        )

        if not img_path:
            return

        person_dir = settings.paths.data_dir / name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image, encrypt, and save
        img = cv2.imread(img_path)
        if img is not None:
            dest_path = person_dir / Path(img_path).name
            security_manager.save_encrypted_image(dest_path, img)
        else:
            shutil.copy(img_path, person_dir)

        self.face_detector.rebuild_embeddings()
        self.populate_person_list()
        self.select_person(name)

        CTkMessagebox(title="Success", message=f"Added '{name}' successfully", icon="check")

        log_activity(f"Added new person: {name}", "success")
        log_system(f"New person registered: {name}", "success")
        self.log_detection_event("new_registration", "system")

    def add_image_for_person(self, name):
        """Add image for existing person"""
        img_path = filedialog.askopenfilename(
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
            if self.current_view != "cameras":
                self.root.after(100, update)
                return

            selected_cam = self.selected_camera.get()
            if not selected_cam or selected_cam == "No cameras":
                self.video_label.configure(text="📹 No camera available", image=None)
                self.root.after(1000, update)
                return

            cam = self.camera.get_camera(selected_cam)
            if not cam:
                self.video_label.configure(text=f"❌ Camera not found: {selected_cam}", image=None)
                self.root.after(1000, update)
                return

            try:
                ret, frame = cam.read()
                if ret and frame is not None:

                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    if self.brightness_value != 1.0:
                        pil_image = ImageEnhance.Brightness(pil_image).enhance(self.brightness_value)

                    display_size = (Sizes.VIDEO_FEED_WIDTH, Sizes.VIDEO_FEED_HEIGHT)
                    pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)

                    ctk_image = CTkImage(pil_image, size=pil_image.size)
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

            refresh_rate = 33 if self.state.is_person_detection_enabled() else 50
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
        dialog.geometry("400x200")
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

        entry_frame, source_entry = create_modern_entry(content, placeholder="Camera ID or RTSP URL", icon="📹")
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
    # SIMPLE MODE
    # ================================================================

    def _create_mode_toggle(self):
        """Create toggle button for Simple/Advanced mode"""
        toggle_frame = CTkFrame(self.root, fg_color="transparent")
        toggle_frame.place(relx=0.98, rely=0.02, anchor="ne")
        
        mode_text = "Giao diện: Đơn giản" if self.simple_mode else "Mode: Advanced"
        
        self.mode_btn = create_modern_button(
            toggle_frame,
            text=mode_text,
            variant="secondary",
            size="small",
            icon="🔄",
            command=self._toggle_mode
        )
        self.mode_btn.pack()

    def _toggle_mode(self):
        """Switch between Simple and Advanced modes"""
        self.simple_mode = not self.simple_mode
        
        # Clear current layout
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Re-create layout
        self._setup_modern_window()
        self._create_animated_layout()
        
        log_activity(f"Switched to {'Simple' if self.simple_mode else 'Advanced'} Mode", "info")

    def _init_simple_dashboard_tab(self):
        """Initialize simplified dashboard for non-tech users"""
        # Main container with 2 columns
        container = CTkFrame(self.dashboard_tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=Sizes.PADDING_LG, pady=Sizes.PADDING_LG)
        
        container.grid_columnconfigure(0, weight=3) # Left: Status & Controls
        container.grid_columnconfigure(1, weight=2) # Right: Recent Activity
        container.grid_rowconfigure(0, weight=1)
        
        # LEFT COLUMN
        left_col = CTkFrame(container, fg_color="transparent")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, Sizes.PADDING_LG))
        
        # 1. System Status Card (Big & Clear)
        status_card = create_glass_card(left_col)
        status_card.pack(fill="x", pady=(0, Sizes.PADDING_LG))
        
        status_frame = CTkFrame(status_card, fg_color="transparent")
        status_frame.pack(fill="x", padx=Sizes.PADDING_LG, pady=Sizes.PADDING_LG)
        
        # Big Icon & Status Text
        icon_label = CTkLabel(status_frame, text="🛡️", font=("Segoe UI", 64))
        icon_label.pack(side="left", padx=(0, Sizes.PADDING_LG))
        
        text_frame = CTkFrame(status_frame, fg_color="transparent")
        text_frame.pack(side="left", fill="y")
        
        CTkLabel(
            text_frame, 
            text="Hệ thống đang hoạt động", 
            font=Fonts.TITLE_LG, 
            text_color=Colors.SUCCESS
        ).pack(anchor="w")
        
        CTkLabel(
            text_frame, 
            text="Tất cả camera đang giám sát an toàn", 
            font=Fonts.BODY, 
            text_color=Colors.TEXT_SECONDARY
        ).pack(anchor="w")
        
        # 2. Big Control Buttons
        controls_card = create_glass_card(left_col)
        controls_card.pack(fill="x", pady=(0, Sizes.PADDING_LG))
        
        CTkLabel(
            controls_card,
            text="Điều khiển nhanh",
            font=Fonts.HEADING,
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w", padx=Sizes.PADDING_LG, pady=(Sizes.PADDING_LG, Sizes.PADDING_MD))
        
        btn_grid = CTkFrame(controls_card, fg_color="transparent")
        btn_grid.pack(fill="x", padx=Sizes.PADDING_LG, pady=(0, Sizes.PADDING_LG))
        btn_grid.grid_columnconfigure((0, 1), weight=1)
        
        # Arm/Disarm Button
        self.arm_btn = create_modern_button(
            btn_grid,
            text="TẮT BÁO ĐỘNG",
            variant="danger",
            height=60,
            icon="🔕",
            command=self._simple_toggle_alarm
        )
        self.arm_btn.grid(row=0, column=0, padx=(0, Sizes.PADDING_MD), sticky="ew")
        
        # View Camera Button
        create_modern_button(
            btn_grid,
            text="XEM CAMERA",
            variant="primary",
            height=60,
            icon="📹",
            command=lambda: self.main_tabs.set("📹 Camera")
        ).grid(row=0, column=1, padx=(Sizes.PADDING_MD, 0), sticky="ew")

        # 3. Simple Settings (Toggle Switches)
        settings_card = create_glass_card(left_col)
        settings_card.pack(fill="x")
        
        CTkLabel(
            settings_card,
            text="Cài đặt nhanh",
            font=Fonts.HEADING,
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w", padx=Sizes.PADDING_LG, pady=(Sizes.PADDING_LG, Sizes.PADDING_MD))
        
        self._create_simple_setting_row(settings_card, "Phát hiện người lạ", "detection.global_enabled")
        self._create_simple_setting_row(settings_card, "Cảnh báo cháy", "detection.fire_confidence_threshold", is_threshold=True)
        self._create_simple_setting_row(settings_card, "Trợ lý ảo AI", "ai.enabled")
        
        # RIGHT COLUMN
        right_col = CTkFrame(container, fg_color="transparent")
        right_col.grid(row=0, column=1, sticky="nsew")
        
        activity_card = create_glass_card(right_col)
        activity_card.pack(fill="both", expand=True)
        
        CTkLabel(
            activity_card,
            text="🔔 Hoạt động gần đây",
            font=Fonts.HEADING,
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w", padx=Sizes.PADDING_LG, pady=Sizes.PADDING_LG)
        
        self.activity_feed = ActivityFeedWidget(activity_card)
        self.activity_feed.pack(fill="both", expand=True, padx=Sizes.PADDING_LG, pady=(0, Sizes.PADDING_LG))

    def _create_simple_setting_row(self, parent, label, config_key, is_threshold=False):
        """Create a simple setting row with label and switch"""
        row = CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=Sizes.PADDING_LG, pady=(0, Sizes.PADDING_MD))
        
        CTkLabel(
            row,
            text=label,
            font=Fonts.BODY,
            text_color=Colors.TEXT_PRIMARY
        ).pack(side="left")
        
        # Logic to get initial value
        if is_threshold:
            val = settings.get(config_key, 0.0) > 0
        else:
            val = settings.get(config_key, False)
            
        var = StringVar(value="on" if val else "off")
        
        switch = CTkSwitch(
            row,
            text="",
            variable=var,
            onvalue="on",
            offvalue="off",
            progress_color=Colors.PRIMARY,
            command=lambda k=config_key, v=var: self._update_simple_setting(k, v)
        )
        switch.pack(side="right")

    def _update_simple_setting(self, key, var):
        """Update setting from simple mode"""
        enabled = (var.get() == "on")
        update_config_value(key, enabled)
        log_activity(f"Đã {'bật' if enabled else 'tắt'} {key}", "info")

    def _simple_toggle_alarm(self):
        """Simple toggle for alarm system (Global Detection)"""
        current_state = settings.get("detection.global_enabled", True)
        new_state = not current_state
        
        update_config_value("detection.global_enabled", new_state)
        if self.state:
            self.state.set_person_detection_enabled(new_state)
            
        # Update button text/style
        if new_state:
            self.arm_btn.configure(text="TẮT BÁO ĐỘNG", fg_color=Colors.DANGER, hover_color=Colors.DANGER_HOVER)
            log_activity("Đã BẬT hệ thống báo động", "success")
        else:
            self.arm_btn.configure(text="BẬT BÁO ĐỘNG", fg_color=Colors.SUCCESS, hover_color=Colors.SUCCESS_HOVER)
            log_activity("Đã TẮT hệ thống báo động", "warning")


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
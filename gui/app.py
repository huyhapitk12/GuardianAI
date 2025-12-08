"""Main GUI application"""

from __future__ import annotations
import threading
from typing import Optional

import customtkinter as ctk
from customtkinter import set_appearance_mode

from config import settings
from gui.styles import Colors, Fonts, Sizes
from gui.widgets import log_activity, log_system
from gui.panels import CamerasPanel, PersonsPanel, SettingsPanel
from gui.widgets.gallery import GalleryPanel


class GuardianApp:
    """Main application window"""
    
    __slots__ = (
        'root', 'camera_manager', 'face_detector', 'state',
        'tabs', 'panels', 'current_tab'
    )
    
    def __init__(self, root: ctk.CTk, camera_manager, face_detector, state_manager):
        self.root = root
        self.camera_manager = camera_manager
        self.face_detector = face_detector
        self.state = state_manager
        
        self.panels = {}
        self.current_tab = "cameras"
        
        self._setup_window()
        self._build_ui()
        
        log_system("GUI initialized", "success")
        log_activity("Guardian started", "success")
    
    def _setup_window(self):
        """Configure window"""
        set_appearance_mode("dark")
        
        self.root.title("Guardian Security System")
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)
        self.root.configure(fg_color=Colors.BG_PRIMARY)
        
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
    
    def _build_ui(self):
        """Build main UI"""
        # Main container
        container = ctk.CTkFrame(self.root, fg_color=Colors.BG_PRIMARY)
        container.grid(row=0, column=0, sticky="nsew")
        
        # Tabview
        self.tabs = ctk.CTkTabview(
            container,
            fg_color=Colors.BG_PRIMARY,
            segmented_button_fg_color=Colors.BG_SECONDARY,
            segmented_button_selected_color=Colors.PRIMARY,
            segmented_button_selected_hover_color=Colors.PRIMARY_HOVER,
            segmented_button_unselected_color=Colors.BG_TERTIARY,
            segmented_button_unselected_hover_color=Colors.BG_ELEVATED,
            text_color=Colors.TEXT_PRIMARY,
            corner_radius=Sizes.RADIUS_LG,
            command=self._on_tab_change
        )
        self.tabs.pack(fill="both", expand=True, padx=Sizes.MD, pady=Sizes.MD)
        self.tabs._segmented_button.configure(font=Fonts.BODY_BOLD, height=40)
        
        # Add tabs
        cameras_tab = self.tabs.add("📹 Cameras")
        persons_tab = self.tabs.add("👥 Persons")
        gallery_tab = self.tabs.add("🎞️ Gallery")
        settings_tab = self.tabs.add("⚙️ Settings")
        
        # Create panels
        self.panels['cameras'] = CamerasPanel(
            cameras_tab, self.camera_manager, self.state
        )
        self.panels['cameras'].pack(fill="both", expand=True)
        
        self.panels['persons'] = PersonsPanel(
            persons_tab, self.face_detector
        )
        self.panels['persons'].pack(fill="both", expand=True)
        
        self.panels['gallery'] = GalleryPanel(gallery_tab)
        self.panels['gallery'].pack(fill="both", expand=True)
        
        self.panels['settings'] = SettingsPanel(
            settings_tab, self.state
        )
        self.panels['settings'].pack(fill="both", expand=True)
    
    def _on_tab_change(self):
        """Handle tab change"""
        selected = self.tabs.get()
        
        if "Cameras" in selected:
            self.current_tab = "cameras"
        elif "Persons" in selected:
            self.current_tab = "persons"
            if 'persons' in self.panels:
                self.panels['persons'].refresh_list()
        elif "Gallery" in selected:
            self.current_tab = "gallery"
            if 'gallery' in self.panels:
                self.panels['gallery'].refresh()
        elif "Settings" in selected:
            self.current_tab = "settings"
        
        log_activity(f"Switched to {self.current_tab}", "info")
    
    def shutdown(self):
        """Cleanup on shutdown"""
        if 'cameras' in self.panels:
            self.panels['cameras'].stop()


def run_gui(camera_manager, face_detector, state_manager, main_app=None):
    """Run the GUI application"""
    root = ctk.CTk()
    app = GuardianApp(root, camera_manager, face_detector, state_manager)
    
    def on_close():
        """Handle window close event"""
        try:
            # Stop GUI components first
            app.shutdown()
            
            # Call main app shutdown if available
            if main_app is not None:
                log_system("Shutting down main app...", "info")
                main_app.shutdown()
        except Exception as e:
            log_system(f"Shutdown error: {e}", "error")
        finally:
            # Always destroy the GUI window
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    
    log_system("GUI ready", "success")
    root.mainloop()
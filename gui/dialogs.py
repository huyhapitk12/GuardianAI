"""GUI dialogs for Guardian application"""

from __future__ import annotations
import customtkinter as ctk

from gui.styles import Colors, Fonts, Sizes, create_button
from gui.widgets import log_activity


class AddCameraDialog(ctk.CTkToplevel):
    """Dialog for adding a new camera"""
    
    def __init__(self, parent, camera_manager, on_success=None):
        super().__init__(parent)
        
        self.camera_manager = camera_manager
        self.on_success = on_success
        self.result = None
        
        self.setup_window()
        self.build_ui()
    
    def setup_window(self):
        """Configure dialog window"""
        self.title("Add Camera")
        self.geometry("500x400")
        self.resizable(False, False)
        
        # Center on parent
        self.transient(self.master)
        self.update_idletasks()
        
        parent_x = self.master.winfo_x()
        parent_y = self.master.winfo_y()
        parent_w = self.master.winfo_width()
        parent_h = self.master.winfo_height()
        
        x = parent_x + (parent_w - 500) // 2
        y = parent_y + (parent_h - 400) // 2
        self.geometry(f"+{x}+{y}")
        
        self.configure(fg_color=Colors.BG_PRIMARY)
    
    def build_ui(self):
        """Build dialog UI"""
        # Main container
        container = ctk.CTkFrame(self, fg_color=Colors.BG_SECONDARY, corner_radius=Sizes.RADIUS_LG)
        container.pack(fill="both", expand=True, padx=Sizes.LG, pady=Sizes.LG)
        
        # Title
        ctk.CTkLabel(
            container,
            text="📹 Add New Camera",
            font=Fonts.TITLE_LG,
            text_color=Colors.TEXT_PRIMARY
        ).pack(pady=(Sizes.LG, Sizes.MD))
        
        # Description
        ctk.CTkLabel(
            container,
            text="Enter camera source (webcam ID or RTSP URL)",
            font=Fonts.BODY,
            text_color=Colors.TEXT_SECONDARY
        ).pack(pady=(0, Sizes.LG))
        
        # Input section
        input_frame = ctk.CTkFrame(container, fg_color="transparent")
        input_frame.pack(fill="x", padx=Sizes.LG, pady=Sizes.MD)
        
        # Source label
        ctk.CTkLabel(
            input_frame,
            text="Camera Source:",
            font=Fonts.BODY_BOLD,
            text_color=Colors.TEXT_PRIMARY,
            anchor="w"
        ).pack(anchor="w", pady=(0, Sizes.XS))
        
        # Source entry
        self.source_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="e.g., 0, 1, or rtsp://...",
            font=Fonts.BODY,
            fg_color=Colors.BG_TERTIARY,
            border_color=Colors.PRIMARY,
            text_color=Colors.TEXT_PRIMARY,
            height=40
        )
        self.source_entry.pack(fill="x", pady=(0, Sizes.SM))
        
        # Examples
        examples_text = """Examples:
• Webcam: 0, 1, 2...
• RTSP: rtsp://username:password@ip:port/stream
• File: /path/to/video.mp4"""
        
        ctk.CTkLabel(
            input_frame,
            text=examples_text,
            font=Fonts.SMALL,
            text_color=Colors.TEXT_MUTED,
            anchor="w",
            justify="left"
        ).pack(anchor="w", pady=(Sizes.SM, 0))
        
        # Status label
        self.status_label = ctk.CTkLabel(
            container,
            text="",
            font=Fonts.BODY,
            text_color=Colors.SUCCESS
        )
        self.status_label.pack(pady=Sizes.SM)
        
        # Buttons
        btn_frame = ctk.CTkFrame(container, fg_color="transparent")
        btn_frame.pack(pady=(Sizes.LG, Sizes.MD), padx=Sizes.LG)
        btn_frame.grid_columnconfigure((0, 1), weight=1)
        
        create_button(
            btn_frame, "Cancel", "secondary", "medium",
            command=self.on_cancel
        ).grid(row=0, column=0, padx=(0, Sizes.SM), sticky="ew")
        
        create_button(
            btn_frame, "Add Camera", "success", "medium",
            command=self.on_add
        ).grid(row=0, column=1, padx=(Sizes.SM, 0), sticky="ew")
        
        # Bind Enter key
        self.source_entry.bind("<Return>", lambda e: self.on_add())
        self.source_entry.focus()
    
    def on_add(self):
        """Handle add camera"""
        source_str = self.source_entry.get().strip()
        
        if not source_str:
            self.show_status("Please enter a camera source", Colors.ERROR)
            return
        
        # Add camera using CameraManager's built-in method
        try:
            self.show_status("Adding camera...", Colors.PRIMARY)
            self.update()
            
            # Use CameraManager's add_camera method
            success, message = self.camera_manager.add_camera(source_str)
            
            if success:
                log_activity(message, "success")
                self.show_status(f"✅ {message}", Colors.SUCCESS)
                
                # Trigger refresh callback if provided
                if self.on_success:
                    self.on_success(source_str)
                
                # Close after delay
                self.after(1000, self.destroy)
            else:
                self.show_status(f"❌ Error: {message}", Colors.ERROR)
                
        except Exception as e:
            self.show_status(f"❌ Error: {str(e)}", Colors.ERROR)
    
    def on_cancel(self):
        """Handle cancel"""
        self.destroy()
    
    def show_status(self, message: str, color: str):
        """Show status message"""
        self.status_label.configure(text=message, text_color=color)

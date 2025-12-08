"""Common reusable widgets"""

from __future__ import annotations
import customtkinter as ctk
from gui.styles import Colors, Fonts, Sizes, create_card


class StatusBadge(ctk.CTkFrame):
    """Status indicator badge"""
    
    __slots__ = ('_label',)
    
    COLORS = {
        "success": Colors.SUCCESS,
        "warning": Colors.WARNING,
        "danger": Colors.DANGER,
        "info": Colors.INFO,
        "default": Colors.BG_ELEVATED,
    }
    
    def __init__(self, parent, text: str, status: str = "default", **kwargs):
        super().__init__(
            parent,
            fg_color=self.COLORS.get(status, self.COLORS["default"]),
            corner_radius=12,
            **kwargs
        )
        
        self._label = ctk.CTkLabel(
            self, text=text,
            font=Fonts.TINY,
            text_color=Colors.TEXT_PRIMARY
        )
        self._label.pack(padx=Sizes.SM, pady=2)
    
    def set_status(self, text: str, status: str):
        self.configure(fg_color=self.COLORS.get(status, self.COLORS["default"]))
        self._label.configure(text=text)


class StatCard(ctk.CTkFrame):
    """Statistics display card"""
    
    __slots__ = ('_value_label', '_subtitle_label')
    
    def __init__(
        self, parent,
        icon: str,
        title: str,
        value: str,
        subtitle: str = "",
        color: str = Colors.PRIMARY,
        **kwargs
    ):
        super().__init__(parent, fg_color=Colors.BG_SECONDARY, corner_radius=Sizes.RADIUS_LG, **kwargs)
        
        # Icon
        icon_frame = ctk.CTkFrame(self, width=50, height=50, fg_color=color, corner_radius=Sizes.RADIUS_MD)
        icon_frame.pack(anchor="w", padx=Sizes.MD, pady=(Sizes.MD, Sizes.SM))
        icon_frame.pack_propagate(False)
        ctk.CTkLabel(icon_frame, text=icon, font=("Segoe UI", 24)).pack(expand=True)
        
        # Value
        self._value_label = ctk.CTkLabel(self, text=str(value), font=Fonts.TITLE_LG, text_color=Colors.TEXT_PRIMARY)
        self._value_label.pack(anchor="w", padx=Sizes.MD)
        
        # Title
        ctk.CTkLabel(self, text=title, font=Fonts.CAPTION, text_color=Colors.TEXT_MUTED).pack(anchor="w", padx=Sizes.MD)
        
        # Subtitle
        if subtitle:
            self._subtitle_label = ctk.CTkLabel(self, text=subtitle, font=Fonts.TINY, text_color=Colors.TEXT_MUTED)
            self._subtitle_label.pack(anchor="w", padx=Sizes.MD, pady=(0, Sizes.MD))
    
    def set_value(self, value):
        self._value_label.configure(text=str(value))


class IconButton(ctk.CTkButton):
    """Icon-only button"""
    
    def __init__(self, parent, icon: str, command=None, size: int = 32, **kwargs):
        super().__init__(
            parent,
            text=icon,
            width=size,
            height=size,
            font=("Segoe UI", size // 2),
            fg_color=Colors.BG_TERTIARY,
            hover_color=Colors.BG_ELEVATED,
            corner_radius=size // 2,
            command=command,
            **kwargs
        )
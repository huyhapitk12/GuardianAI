# GUI styling constants and widget factories

from dataclasses import dataclass
from typing import Optional, Tuple
import customtkinter as ctk


@dataclass(frozen=True)
class Colors:
    # Color palette
    # Backgrounds
    BG_PRIMARY = "#0F172A"
    BG_SECONDARY = "#1E293B"
    BG_TERTIARY = "#334155"
    BG_ELEVATED = "#475569"
    
    # Primary
    PRIMARY = "#3B82F6"
    PRIMARY_HOVER = "#2563EB"
    PRIMARY_LIGHT = "#60A5FA"
    
    # Accent
    ACCENT = "#8B5CF6"
    
    # Status
    SUCCESS = "#10B981"
    SUCCESS_LIGHT = "#34D399"
    WARNING = "#F59E0B"
    WARNING_LIGHT = "#FBBF24"
    DANGER = "#EF4444"
    DANGER_LIGHT = "#F87171"
    INFO = "#06B6D4"
    
    # Text
    TEXT_PRIMARY = "#F1F5F9"
    TEXT_SECONDARY = "#CBD5E1"
    TEXT_MUTED = "#94A3B8"
    
    # Border
    BORDER = "#334155"
    BORDER_FOCUS = "#3B82F6"


@dataclass(frozen=True)
class Sizes:
    # Size constants
    # Spacing
    XS = 4
    SM = 8
    MD = 12
    LG = 16
    XL = 24
    
    # Components
    BUTTON_HEIGHT = 40
    BUTTON_HEIGHT_SM = 32
    INPUT_HEIGHT = 40
    
    # Layout
    SIDEBAR_WIDTH = 280
    VIDEO_WIDTH = 960
    VIDEO_HEIGHT = 720
    
    # Border radius
    RADIUS_SM = 4
    RADIUS_MD = 8
    RADIUS_LG = 12


class Fonts:
    # Font definitions
    FAMILY = "Segoe UI"
    
    # Sizes
    TITLE_XL = (FAMILY, 28, "bold")
    TITLE_LG = (FAMILY, 24, "bold")
    TITLE_MD = (FAMILY, 20, "bold")
    TITLE_SM = (FAMILY, 18, "bold")
    HEADING = (FAMILY, 16, "bold")
    BODY = (FAMILY, 14, "normal")
    BODY_BOLD = (FAMILY, 14, "bold")
    CAPTION = (FAMILY, 12, "normal")
    SMALL = (FAMILY, 11, "normal")
    TINY = (FAMILY, 10, "normal")


def create_button(parent, text: str, variant: str = "primary", size: str = "medium", icon: str = None, **kwargs):
    
    VARIANTS = {
        "primary": {
            "fg_color": Colors.PRIMARY,
            "hover_color": Colors.PRIMARY_HOVER,
            "text_color": Colors.TEXT_PRIMARY,
        },
        "secondary": {
            "fg_color": Colors.BG_TERTIARY,
            "hover_color": Colors.BG_ELEVATED,
            "text_color": Colors.TEXT_PRIMARY,
            "border_width": 1,
            "border_color": Colors.BORDER,
        },
        "ghost": {
            "fg_color": "transparent",
            "hover_color": Colors.BG_TERTIARY,
            "text_color": Colors.TEXT_PRIMARY,
        },
        "danger": {
            "fg_color": Colors.DANGER,
            "hover_color": Colors.DANGER_LIGHT,
            "text_color": Colors.TEXT_PRIMARY,
        },
        "success": {
            "fg_color": Colors.SUCCESS,
            "hover_color": Colors.SUCCESS_LIGHT,
            "text_color": Colors.TEXT_PRIMARY,
        },
    }
    
    SIZES = {
        "small": {"height": Sizes.BUTTON_HEIGHT_SM, "font": Fonts.SMALL},
        "medium": {"height": Sizes.BUTTON_HEIGHT, "font": Fonts.BODY_BOLD},
        "large": {"height": 48, "font": Fonts.HEADING},
    }
    
    style = VARIANTS.get(variant, VARIANTS["primary"]).copy()
    size_cfg = SIZES.get(size, SIZES["medium"]).copy()
    
    display_text = f"{icon} {text}" if icon else text
    
    # Build config - kwargs override defaults
    config = {
        "text": display_text,
        "corner_radius": Sizes.RADIUS_MD,
        **size_cfg,
        **style,
    }
    
    # Apply kwargs (override any existing keys)
    config.update(kwargs)
    
    return ctk.CTkButton(parent, **config)


def create_card(parent, **kwargs):
    # Create card frame
    defaults = {
        "fg_color": Colors.BG_SECONDARY,
        "corner_radius": Sizes.RADIUS_LG,
        "border_width": 1,
        "border_color": Colors.BORDER,
    }
    defaults.update(kwargs)
    return ctk.CTkFrame(parent, **defaults)


def create_entry(
    parent,
    placeholder: str = "",
    icon: str = None,
    **kwargs
):
    # Create styled entry with optional icon
    
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    
    if icon:
        ctk.CTkLabel(
            frame, text=icon, font=Fonts.BODY,
            text_color=Colors.TEXT_MUTED, width=30
        ).pack(side="left", padx=(Sizes.SM, 0))
    
    defaults = {
        "placeholder_text": placeholder,
        "fg_color": Colors.BG_TERTIARY,
        "border_color": Colors.BORDER,
        "text_color": Colors.TEXT_PRIMARY,
        "placeholder_text_color": Colors.TEXT_MUTED,
        "corner_radius": Sizes.RADIUS_MD,
        "height": Sizes.INPUT_HEIGHT,
        "font": Fonts.BODY,
    }
    defaults.update(kwargs)
    
    entry = ctk.CTkEntry(frame, **defaults)
    entry.pack(side="left", fill="x", expand=True, padx=(Sizes.XS if icon else 0, 0))
    
    # Focus styling
    entry.bind("<FocusIn>", lambda _: entry.configure(border_color=Colors.BORDER_FOCUS))
    entry.bind("<FocusOut>", lambda _: entry.configure(border_color=Colors.BORDER))
    
    return frame, entry


def create_switch(
    parent,
    text: str,
    variable: ctk.StringVar,
    command=None,
    **kwargs
):
    # Create styled switch
    defaults = {
        "text": text,
        "variable": variable,
        "onvalue": "on",
        "offvalue": "off",
        "command": command,
        "font": Fonts.BODY,
        "progress_color": Colors.PRIMARY,
    }
    defaults.update(kwargs)
    return ctk.CTkSwitch(parent, **defaults)


def create_label(
    parent,
    text: str,
    style: str = "body",
    color: str = None,
    **kwargs
):
    # Create styled label
    
    STYLES = {
        "title": {"font": Fonts.TITLE_MD, "text_color": Colors.TEXT_PRIMARY},
        "heading": {"font": Fonts.HEADING, "text_color": Colors.TEXT_PRIMARY},
        "body": {"font": Fonts.BODY, "text_color": Colors.TEXT_PRIMARY},
        "caption": {"font": Fonts.CAPTION, "text_color": Colors.TEXT_SECONDARY},
        "muted": {"font": Fonts.SMALL, "text_color": Colors.TEXT_MUTED},
    }
    
    cfg = STYLES.get(style, STYLES["body"]).copy()
    if color:
        cfg["text_color"] = color
    cfg.update(kwargs)
    
    return ctk.CTkLabel(parent, text=text, **cfg)


def create_slider(
    parent,
    from_: float,
    to: float,
    value: float = None,
    command=None,
    **kwargs
):
    # Create styled slider
    defaults = {
        "from_": from_,
        "to": to,
        "progress_color": Colors.PRIMARY,
        "button_color": Colors.PRIMARY,
        "button_hover_color": Colors.PRIMARY_HOVER,
    }
    defaults.update(kwargs)
    
    slider = ctk.CTkSlider(parent, **defaults)
    if value is not None:
        slider.set(value)
    if command:
        slider.configure(command=command)
    
    return slider


def create_option_menu(
    parent,
    values: list,
    variable: ctk.StringVar = None,
    command=None,
    **kwargs
):
    # Create styled option menu
    defaults = {
        "values": values,
        "fg_color": Colors.BG_TERTIARY,
        "button_color": Colors.BG_ELEVATED,
        "button_hover_color": Colors.PRIMARY,
        "dropdown_fg_color": Colors.BG_SECONDARY,
        "dropdown_hover_color": Colors.BG_TERTIARY,
        "font": Fonts.BODY,
    }
    defaults.update(kwargs)
    
    if variable:
        defaults["variable"] = variable
    if command:
        defaults["command"] = command
    
    return ctk.CTkOptionMenu(parent, **defaults)


def create_progress_bar(parent, **kwargs):
    # Create styled progress bar
    defaults = {
        "progress_color": Colors.PRIMARY,
        "fg_color": Colors.BG_TERTIARY,
        "height": 8,
    }
    defaults.update(kwargs)
    return ctk.CTkProgressBar(parent, **defaults)


def create_scrollable_frame(parent, **kwargs):
    # Create styled scrollable frame
    defaults = {
        "fg_color": Colors.BG_PRIMARY,
        "corner_radius": Sizes.RADIUS_MD,
    }
    defaults.update(kwargs)
    return ctk.CTkScrollableFrame(parent, **defaults)
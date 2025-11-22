import customtkinter as ctk
from typing import Optional

# ============ THEME CONSTANTS ============

class Colors:
    GRADIENT_START = "#0F172A"
    GRADIENT_END = "#1E293B"
    BG_PRIMARY = "#0F172A"
    BG_SECONDARY = "#1E293B"
    BG_TERTIARY = "#334155"
    BG_ELEVATED = "#475569"
    GLASS_BG = "#1E293BCC"
    GLASS_BORDER = "#4B5563"
    
    PRIMARY = "#3B82F6"
    PRIMARY_HOVER = "#2563EB"
    PRIMARY_DARK = "#1E40AF"
    PRIMARY_LIGHT = "#60A5FA"
    PRIMARY_GLOW = "#93C5FD"
    
    ACCENT = "#8B5CF6"
    ACCENT_HOVER = "#7C3AED"
    
    SUCCESS = "#10B981"
    SUCCESS_LIGHT = "#34D399"
    WARNING = "#F59E0B"
    WARNING_LIGHT = "#FBBF24"
    DANGER = "#EF4444"
    DANGER_LIGHT = "#F87171"
    INFO = "#06B6D4"
    
    TEXT_PRIMARY = "#F1F5F9"
    TEXT_SECONDARY = "#CBD5E1"
    TEXT_MUTED = "#94A3B8"
    TEXT_DISABLED = "#64748B"
    
    BORDER = "#334155"
    BORDER_LIGHT = "#475569"
    BORDER_FOCUS = "#3B82F6"
    
    ENTRY_BG = "#1E293B"
    ENTRY_BORDER = "#475569"
    ENTRY_PLACEHOLDER = "#64748B"

class Fonts:
    FAMILY_PRIMARY = "Inter"
    FAMILY_SECONDARY = "Segoe UI"
    _initialized = False

    HERO = ("Segoe UI", 36, "bold")
    TITLE_XL = ("Segoe UI", 28, "bold")
    TITLE_LG = ("Segoe UI", 24, "bold")
    TITLE_MD = ("Segoe UI", 20, "bold")
    TITLE_MEDIUM = ("Segoe UI", 20, "bold") # Alias for TITLE_MD
    TITLE_SM = ("Segoe UI", 18, "bold")
    HEADING = ("Segoe UI", 16, "bold")
    BODY = ("Segoe UI", 14, "normal")
    BODY_BOLD = ("Segoe UI", 14, "bold")
    CAPTION = ("Segoe UI", 12, "normal")
    SMALL = ("Segoe UI", 11, "normal")
    TINY = ("Segoe UI", 10, "normal")
    CODE = ("Consolas", 12, "normal")

    @classmethod
    def init_fonts(cls):
        if cls._initialized: return
        try:
            import tkinter.font as tkfont
            avail = tkfont.families()
            fam = next((f for f in [cls.FAMILY_PRIMARY, "SF Pro Display", cls.FAMILY_SECONDARY] if f in avail), "Segoe UI")
            cls.HERO = (fam, 36, "bold")
            cls.TITLE_XL = (fam, 28, "bold")
            cls.TITLE_LG = (fam, 24, "bold")
            cls.TITLE_MD = (fam, 20, "bold")
            cls.TITLE_MEDIUM = (fam, 20, "bold")
            cls.TITLE_SM = (fam, 18, "bold")
            cls.HEADING = (fam, 16, "bold")
            cls.BODY = (fam, 14, "normal")
            cls.BODY_BOLD = (fam, 14, "bold")
            cls.CAPTION = (fam, 12, "normal")
            cls.SMALL = (fam, 11, "normal")
            cls.TINY = (fam, 10, "normal")
            cls._initialized = True
        except: pass

class Sizes:
    UNIT = 8
    SPACE_XS = UNIT
    SPACE_SM = UNIT * 2
    SPACE_MD = UNIT * 3
    SPACE_LG = UNIT * 4
    
    SIDEBAR_WIDTH = 300
    RIGHT_SIDEBAR_WIDTH = 480
    HEADER_HEIGHT = 64
    
    BUTTON_HEIGHT = 40
    BUTTON_HEIGHT_SM = 32
    BUTTON_HEIGHT_LG = 48
    INPUT_HEIGHT = 40
    
    PADDING_XS = SPACE_XS
    PADDING_SM = SPACE_SM
    PADDING_MD = SPACE_MD
    PADDING_LG = SPACE_LG
    
    CORNER_RADIUS_SM = 6
    CORNER_RADIUS = 10
    CORNER_RADIUS_LG = 16
    CORNER_RADIUS_FULL = 999
    
    VIDEO_FEED_WIDTH = 640
    VIDEO_FEED_HEIGHT = 480
    
    BORDER_WIDTH = 1
    BORDER_WIDTH_FOCUS = 2

# ============ WIDGET FACTORIES ============

def create_modern_button(parent, text, variant="primary", size="medium", icon=None, **kwargs):
    variants = {
        "primary": {"fg_color": Colors.PRIMARY, "hover_color": Colors.PRIMARY_HOVER, "text_color": Colors.TEXT_PRIMARY, "border_width": 0},
        "secondary": {"fg_color": Colors.BG_TERTIARY, "hover_color": Colors.BG_ELEVATED, "text_color": Colors.TEXT_PRIMARY, "border_width": 1, "border_color": Colors.BORDER},
        "ghost": {"fg_color": "transparent", "hover_color": Colors.BG_TERTIARY, "text_color": Colors.TEXT_PRIMARY, "border_width": 1, "border_color": Colors.BORDER},
        "danger": {"fg_color": Colors.DANGER, "hover_color": Colors.DANGER_LIGHT, "text_color": Colors.TEXT_PRIMARY, "border_width": 0},
        "success": {"fg_color": Colors.SUCCESS, "hover_color": Colors.SUCCESS_LIGHT, "text_color": Colors.TEXT_PRIMARY, "border_width": 0},
    }
    sizes = {
        "small": {"height": Sizes.BUTTON_HEIGHT_SM, "font": Fonts.SMALL, "corner_radius": Sizes.CORNER_RADIUS_SM},
        "medium": {"height": Sizes.BUTTON_HEIGHT, "font": Fonts.BODY_BOLD, "corner_radius": Sizes.CORNER_RADIUS},
        "large": {"height": Sizes.BUTTON_HEIGHT_LG, "font": Fonts.HEADING, "corner_radius": Sizes.CORNER_RADIUS_LG},
    }
    
    style = variants.get(variant, variants["primary"])
    size_cfg = sizes.get(size, sizes["medium"])
    txt = f"{icon} {text}" if icon else text
    
    # Extract width/height from kwargs
    width = kwargs.pop('width', None)
    height = kwargs.pop('height', None)
    
    cfg = {"text": txt, "font": size_cfg["font"], "height": size_cfg["height"], "corner_radius": size_cfg["corner_radius"], **style, **kwargs}
    
    # Only add width/height if not None
    if width is not None:
        cfg["width"] = width
    if height is not None:
        cfg["height"] = height
        
    return ctk.CTkButton(parent, **cfg)

def create_glass_card(parent, **kwargs):
    kwargs.setdefault("fg_color", Colors.BG_SECONDARY)
    kwargs.setdefault("border_width", 1)
    kwargs.setdefault("border_color", Colors.BORDER)
    kwargs.setdefault("corner_radius", Sizes.CORNER_RADIUS_LG)
    return ctk.CTkFrame(parent, **kwargs)

def create_card_frame(parent, **kwargs):
    kwargs.setdefault("fg_color", Colors.BG_SECONDARY)
    kwargs.setdefault("corner_radius", Sizes.CORNER_RADIUS_LG)
    kwargs.setdefault("border_width", 1)
    kwargs.setdefault("border_color", Colors.BORDER)
    return ctk.CTkFrame(parent, **kwargs)

def create_modern_entry(parent, placeholder="", icon=None, **kwargs):
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    if icon:
        ctk.CTkLabel(frame, text=icon, font=Fonts.BODY, text_color=Colors.TEXT_MUTED, width=30).pack(side="left", padx=(10, 0))
    
    cfg = {
        "fg_color": Colors.ENTRY_BG, "border_color": Colors.ENTRY_BORDER, "border_width": Sizes.BORDER_WIDTH,
        "text_color": Colors.TEXT_PRIMARY, "placeholder_text_color": Colors.ENTRY_PLACEHOLDER,
        "corner_radius": Sizes.CORNER_RADIUS, "height": Sizes.INPUT_HEIGHT, "font": Fonts.BODY
    }
    if placeholder: cfg["placeholder_text"] = placeholder
    cfg.update(kwargs)
    
    entry = ctk.CTkEntry(frame, **cfg)
    entry.pack(side="left", fill="x", expand=True, padx=(5 if icon else 0, 0))
    
    entry.bind("<FocusIn>", lambda _: entry.configure(border_color=Colors.BORDER_FOCUS, border_width=Sizes.BORDER_WIDTH_FOCUS))
    entry.bind("<FocusOut>", lambda _: entry.configure(border_color=Colors.ENTRY_BORDER, border_width=Sizes.BORDER_WIDTH))
    return frame, entry

def create_status_badge(parent, text, status="info", **kwargs):
    colors = {
        "success": (Colors.SUCCESS, Colors.TEXT_PRIMARY), "warning": (Colors.WARNING, Colors.TEXT_PRIMARY),
        "danger": (Colors.DANGER, Colors.TEXT_PRIMARY), "info": (Colors.INFO, Colors.TEXT_PRIMARY),
        "default": (Colors.BG_ELEVATED, Colors.TEXT_SECONDARY)
    }
    bg, fg = colors.get(status, colors["default"])
    badge = ctk.CTkFrame(parent, fg_color=bg, corner_radius=Sizes.CORNER_RADIUS_FULL, **kwargs)
    ctk.CTkLabel(badge, text=text, font=Fonts.TINY, text_color=fg).pack(padx=8, pady=2)
    return badge

def create_stat_card(parent, icon, title, value, color, subtext=""):
    card = create_card_frame(parent)
    ctk.CTkLabel(card, text=icon, font=("Segoe UI", 32), text_color=color).pack(side="left", padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)
    content = ctk.CTkFrame(card, fg_color="transparent")
    content.pack(side="left", fill="both", expand=True, pady=Sizes.PADDING_SM)
    ctk.CTkLabel(content, text=title, font=Fonts.CAPTION, text_color=Colors.TEXT_SECONDARY).pack(anchor="w")
    ctk.CTkLabel(content, text=str(value), font=Fonts.TITLE_LG, text_color=Colors.TEXT_PRIMARY).pack(anchor="w")
    if subtext: ctk.CTkLabel(content, text=subtext, font=Fonts.TINY, text_color=Colors.TEXT_MUTED).pack(anchor="w")
    return card

# Helpers
create_button_primary = lambda p, t, **kw: create_modern_button(p, t, "primary", **kw)
create_button_secondary = lambda p, t, **kw: create_modern_button(p, t, "secondary", **kw)
create_button_danger = lambda p, t, **kw: create_modern_button(p, t, "danger", **kw)
create_entry = lambda p, **kw: create_modern_entry(p, **kw)[1]

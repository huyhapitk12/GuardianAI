# gui/__init__.py

from .manager import ModernFaceManagerGUI, run_gui
from .styles import (
    Colors, Fonts, Sizes,
    create_modern_button, create_glass_card,
    create_modern_entry, create_status_badge,
    create_card_frame, create_button_primary,
    create_button_secondary, create_button_danger,
    create_entry
)
from .detection_controls import DetectionControlsFrame
from .control_panels import SettingsPanel

__all__ = [
    'ModernFaceManagerGUI',
    'run_gui',
    'Colors',
    'Fonts', 
    'Sizes',
    'create_modern_button',
    'create_glass_card',
    'create_modern_entry',
    'create_status_badge',
    'create_card_frame',
    'create_button_primary',
    'create_button_secondary',
    'create_button_danger',
    'create_entry',
    'DetectionControlsFrame',
    'SettingsPanel'
]
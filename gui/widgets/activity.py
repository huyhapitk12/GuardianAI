# gui/widgets/activity.py
def log_activity(message: str, activity_type: str = "info"):
    # Log activity - just print
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "detection": "👁️",
        "alert": "🚨"
    }
    icon = icons.get(activity_type, "ℹ️")
    print(f"{icon} {message}")


def log_system(message: str, level: str = "info"):
    # Log system - just print
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    icon = icons.get(level, "ℹ️")
    print(f"{icon} [SYSTEM] {message}")
   
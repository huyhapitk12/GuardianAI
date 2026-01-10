# gui/widgets/activity.py
def log_activity(message: str, activity_type: str = "info"):
    # Log activity - just print
    prefixes = {
        "info": "[INFO]",
        "success": "[OK]",
        "warning": "[WARN]",
        "error": "[ERR]",
        "detection": "[DET]",
        "alert": "[ALERT]"
    }
    prefix = prefixes.get(activity_type, "[INFO]")
    print(f"{prefix} {message}")


def log_system(message: str, level: str = "info"):
    # Log system - just print
    prefixes = {
        "info": "[INFO]",
        "success": "[OK]",
        "warning": "[WARN]",
        "error": "[ERR]"
    }
    prefix = prefixes.get(level, "[INFO]")
    print(f"{prefix} [SYSTEM] {message}")
   
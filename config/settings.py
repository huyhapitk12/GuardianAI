# config/settings.py
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, List
import yaml
# Remove logging import - using print instead
class Settings:
    """Configuration management: loads from YAML and environment variables"""
    def __init__(self, config_path='config/config.yaml'):
        self.base_dir = Path(__file__).parent.parent
        config_file = self.base_dir / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        self._config = self._resolve_env_vars(config_data)
        self._resolve_paths()
        self._process_special_values()
        self._create_dynamic_enums()

    def _resolve_env_vars(self, data: Any) -> Any:
        """Recursively resolve environment variables in config"""
        if isinstance(data, dict):
            return {k: self._resolve_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._resolve_env_vars(i) for i in data]
        elif isinstance(data, str):
            match = re.match(r'^\$\{(.+?)(?::"(.+?)")?\}$', data)
            if match:
                var_name, default_val = match.groups()
                return os.getenv(var_name, default_val)
        return data

    def _resolve_paths(self):
        """Convert relative paths to absolute"""
        if 'paths' in self._config:
            for key, value in self._config['paths'].items():
                path = Path(value)
                if not path.is_absolute():
                    self._config['paths'][key] = self.base_dir / value
            tmp_dir = self._config['paths'].get('tmp_dir')
            if tmp_dir:
                tmp_dir.mkdir(exist_ok=True)

    def _process_special_values(self):
        """Process special config values (e.g., camera.sources)"""
        if 'camera' in self._config and 'sources' in self._config['camera']:
            sources_str = self._config['camera']['sources']
            if isinstance(sources_str, str):
                self._config['camera']['sources'] = [s.strip() for s in sources_str.split(',')]

    def _create_dynamic_enums(self):
        """Create dynamic Enums (kept in source for clarity)"""
        pass

    def get_yolo_model_path(self, model_type: str, size: str, format_type: str = None) -> Path:
        """Get YOLO model path based on type, size, and format"""
        if format_type is None:
            format_type = self.get('models.yolo_format', 'pytorch')
        size_capitalized = size.capitalize()
        type_capitalized = model_type.capitalize()
        base_path = self.paths.model_dir / size_capitalized / type_capitalized
        if format_type.lower() == 'pytorch':
            model_path = base_path / f"{size}.pt"
        elif format_type.lower() == 'onnx':
            model_path = base_path / f"{size}.onnx"
        elif format_type.lower() == 'openvino':
            model_path = base_path / f"{size}_openvino_model"
        else:
            print(f"WARNING: Unknown YOLO format '{format_type}', using pytorch")
            model_path = base_path / f"{size}.pt"
        return model_path

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to top-level config keys"""
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return _ConfigProxy(value)
            return value
        raise AttributeError(f"'Settings' object has no attribute '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style get method"""
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

class _ConfigProxy:
    """Proxy for accessing nested dictionaries as attributes"""
    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return _ConfigProxy(value)
            return value
        raise AttributeError(f"Configuration section has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

# --- Enums for type hints ---
class AlertType(Enum):
    KNOWN_PERSON = "nguoi_quen"
    STRANGER = "nguoi_la"
    FIRE_WARNING = "lua_chay_nghi_ngo"
    FIRE_CRITICAL = "lua_chay_khan_cap"

class ActionCode(Enum):
    TOGGLE_ON = "TOGGLE_ON"
    TOGGLE_OFF = "TOGGLE_OFF"
    GET_IMAGE = "GET_IMAGE"
    ALARM_ON = "ALARM_ON"
    ALARM_OFF = "ALARM_OFF"

def _get_config_path() -> Path:
    """Get config file path"""
    base_dir = Path(__file__).parent.parent
    return base_dir / 'config/config.yaml'

def load_raw_config() -> dict:
    """Load raw config file"""
    config_path = _get_config_path()
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Error loading config: {e}")
        return {}

def save_raw_config(data: dict) -> bool:
    """Save config to file"""
    config_path = _get_config_path()
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        return True
    except Exception as e:
        print(f"ERROR: Error saving config: {e}")
        return False

def update_config_value(key_path: str, value: Any) -> bool:
    """Update config value and save to file"""
    config = load_raw_config()
    if not config:
        return False
    keys = key_path.split('.')
    current = config
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    if save_raw_config(config):
        global settings
        current_settings = settings._config
        for key in keys[:-1]:
            if key not in current_settings:
                current_settings[key] = {}
            current_settings = current_settings[key]
        current_settings[keys[-1]] = value
        print(f"INFO: Updated config: {key_path} = {value}")
        return True
    return False

def add_camera_source_to_config(new_source) -> tuple[bool, str]:
    """Add new camera source to config.yaml"""
    config = load_raw_config()
    if not config:
        return False, "Could not load config"
    if 'camera' not in config:
        config['camera'] = {}
    if 'sources' not in config['camera']:
        config['camera']['sources'] = ""
    current_sources_str = str(config['camera'].get('sources', ''))
    sources_list = [s.strip() for s in current_sources_str.split(',') if s.strip()]
    str_new_source = str(new_source).strip()
    if str_new_source in sources_list:
        print(f"WARNING: Camera '{str_new_source}' already exists")
        return False, "Camera already exists"
    sources_list.append(str_new_source)
    config['camera']['sources'] = ", ".join(sources_list)
    if save_raw_config(config):
        print(f"INFO: Added camera '{str_new_source}' to config")
        global settings
        settings._config['camera']['sources'] = sources_list
        return True, "Camera added successfully"
    else:
        return False, "Error saving config"

# Phiên bản cài đặt toàn cục
settings = Settings()

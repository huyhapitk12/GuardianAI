# config/settings.py
import os
import re
from enum import Enum
from pathlib import Path
import yaml

# Quản lý cấu hình: tải từ file YAML và biến môi trường
class Settings:
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

    # Đệ quy xử lý biến môi trường trong config
    def _resolve_env_vars(self, data):
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

    # Chuyển đường dẫn tương đối thành tuyệt đối
    def _resolve_paths(self):
        if 'paths' in self._config:
            for key, value in self._config['paths'].items():
                path = Path(value)
                if not path.is_absolute():
                    self._config['paths'][key] = self.base_dir / value
            tmp_dir = self._config['paths'].get('tmp_dir')
            if tmp_dir:
                tmp_dir.mkdir(exist_ok=True)

    # Xử lý các giá trị đặc biệt (ví dụ: camera.sources)
    def _process_special_values(self):
        if 'camera' in self._config and 'sources' in self._config['camera']:
            sources_val = self._config['camera']['sources']
            if isinstance(sources_val, str):
                self._config['camera']['sources'] = [s.strip() for s in sources_val.split(',')]
            elif not isinstance(sources_val, list):
                # Handle int or other single values
                self._config['camera']['sources'] = [sources_val]

    # Lấy đường dẫn model YOLO dựa trên loại, kích thước và định dạng
    def get_yolo_model_path(self, model_type, size, format_type=None):
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

    # Cho phép truy cập config kiểu attribute
    def __getattr__(self, name):
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return _ConfigProxy(value)
            return value
        raise AttributeError(f"'Settings' object has no attribute '{name}'")

    # Lấy giá trị kiểu dict
    def get(self, key, default=None):
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    # Cập nhật giá trị setting theo key
    def set(self, key, value):
        keys = key.split('.')
        current = self._config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    # Lưu cấu hình hiện tại xuống file
    def save(self):
        data = self._prepare_for_save(self._config)
        save_raw_config(data)
        
    # Đệ quy chuyển đổi Path object thành chuỗi để lưu
    def _prepare_for_save(self, data):
        if isinstance(data, dict):
            return {k: self._prepare_for_save(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_save(i) for i in data]
        elif isinstance(data, Path):
            try:
                # Try to make path relative to project root
                return str(data.relative_to(self.base_dir))
            except ValueError:
                return str(data)
        return data

# Proxy để truy cập dictionary lồng nhau như attribute
class _ConfigProxy:
    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name):
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return _ConfigProxy(value)
            return value
        raise AttributeError(f"Configuration section has no attribute '{name}'")

    def __getitem__(self, key):
        return self._data[key]

# --- Enum cho type hints ---
class AlertType(Enum):
    KNOWN_PERSON = "nguoi_quen"
    STRANGER = "nguoi_la"
    FIRE_WARNING = "lua_chay_nghi_ngo"
    FIRE_CRITICAL = "lua_chay_khan_cap"
    ANOMALOUS_BEHAVIOR = "hanh_vi_bat_thuong"
    FALL = "te_nga"

# Mức độ ưu tiên cảnh báo
class AlertPriority(Enum):
    CRITICAL = 1  # Báo động cháy - cần xử lý ngay lập tức
    HIGH = 2      # Người lạ có hành vi bất thường - đe dọa an ninh
    MEDIUM = 3    # Chỉ phát hiện người lạ - giám sát an ninh
    LOW = 4       # Người quen - thông tin tham khảo

class ActionCode(Enum):
    TOGGLE_ON = "TOGGLE_ON"
    TOGGLE_OFF = "TOGGLE_OFF"
    GET_IMAGE = "GET_IMAGE"
    ALARM_ON = "ALARM_ON"
    ALARM_OFF = "ALARM_OFF"

# Lấy đường dẫn file config
def _get_config_path():
    base_dir = Path(__file__).parent.parent
    return base_dir / 'config/config.yaml'

# Tải file config thô
def load_raw_config():
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

# Lưu config thô
def save_raw_config(data):
    config_path = _get_config_path()
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        return True
    except Exception as e:
        print(f"ERROR: Error saving config: {e}")
        return False

# Cập nhật giá trị config và lưu
def update_config_value(key_path, value):
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

# Thêm camera mới vào config
def add_camera_source_to_config(new_source):
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

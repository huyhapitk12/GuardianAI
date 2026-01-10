# config/settings.py
import os
import re
import platform
from enum import Enum
from pathlib import Path
import yaml


# Kiểm tra CPU có phải Intel không
def is_intel_cpu():
    cpu_info = platform.processor().lower()
    return 'intel' in cpu_info or 'genuine' in cpu_info


# Kiểm tra GPU có phải NVIDIA không
def is_nvidia_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            return 'nvidia' in gpu_name or 'geforce' in gpu_name or 'rtx' in gpu_name or 'gtx' in gpu_name
    except:
        pass
    return False


# Quản lý cấu hình (YAML + ENV)
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

    # Xử lý biến môi trường (đệ quy)
    def _resolve_env_vars(self, data):
        if isinstance(data, dict):
            return {k: self._resolve_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._resolve_env_vars(i) for i in data]
        elif isinstance(data, str):
            match = re.match(r'^\$\{(.+?)(?:"(.+?)")?\}$', data)
            if match:
                var_name, default_val = match.groups()
                return os.getenv(var_name, default_val)
        return data

    # Đường dẫn tương đối -> Tuyệt đối
    def _resolve_paths(self):
        if 'paths' in self._config:
            for key, value in self._config['paths'].items():
                path = Path(value)
                if not path.is_absolute():
                    self._config['paths'][key] = self.base_dir / value
                else:
                    self._config['paths'][key] = path  # Đảm bảo là Path object
            tmp_dir = self._config['paths'].get('tmp_dir')
            if tmp_dir:
                Path(tmp_dir).mkdir(exist_ok=True)

    # Xử lý các giá trị đặc biệt
    def _process_special_values(self):
        if 'camera' in self._config and 'sources' in self._config['camera']:
            sources_val = self._config['camera']['sources']
            if isinstance(sources_val, str):
                # Phân tích chuỗi thành danh sách, lọc bỏ giá trị None/rỗng
                sources_list = []
                for s in sources_val.split(','):
                    s = s.strip()
                    # Bỏ qua giá trị None, rỗng
                    if s and s.lower() != 'none':
                        # Chuyển thành int nếu là số
                        if s.isdigit():
                            sources_list.append(int(s))
                        else:
                            sources_list.append(s)
                self._config['camera']['sources'] = sources_list
            elif isinstance(sources_val, list):
                # Lọc bỏ None trong list
                self._config['camera']['sources'] = [s for s in sources_val if s is not None]
            elif sources_val is not None:
                # Xử lý số nguyên hoặc các giá trị đơn lẻ khác
                self._config['camera']['sources'] = [sources_val]
            else:
                self._config['camera']['sources'] = []

    # Tự động chọn format YOLO tối ưu dựa trên device
    def get_optimal_yolo_format(self):
        device = self.get('models.device', 'cpu').lower()
        
        # GPU
        if device == 'gpu':
            # NVIDIA GPU: ưu tiên TensorRT (nhanh nhất)
            if is_nvidia_gpu():
                return 'tensorrt'
            # GPU khác (AMD, Intel Arc): dùng pytorch
            return 'pytorch'
        
        # CPU Intel: dùng OpenVINO (tối ưu cho Intel)
        if is_intel_cpu():
            return 'openvino'
        
        # CPU khác (AMD, ARM): dùng ONNX
        return 'onnx'

    # Lấy path model YOLO
    def get_yolo_model_path(self, model_type, size, format_type=None):
        # Nếu không chỉ định format, tự động chọn tối ưu
        if format_type is None:
            format_type = self.get_optimal_yolo_format()
        
        size_capitalized = size.capitalize()
        type_capitalized = model_type.capitalize()
        base_path = self.paths.model_dir / size_capitalized / type_capitalized
        if format_type.lower() == 'pytorch':
            model_path = base_path / f"{size}.pt"
        elif format_type.lower() == 'onnx':
            model_path = base_path / f"{size}.onnx"
        elif format_type.lower() == 'openvino':
            model_path = base_path / f"{size}_openvino_model"
        elif format_type.lower() == 'tensorrt':
            model_path = base_path / f"{size}.engine"
        else:
            print(f"WARNING: Unknown YOLO format '{format_type}', using pytorch")
            model_path = base_path / f"{size}.pt"
        return model_path
    
    # Đảm bảo model tồn tại, nếu chưa có thì export từ .pt
    def ensure_yolo_model(self, model_type, size, format_type=None):
        if format_type is None:
            format_type = self.get_optimal_yolo_format()
        
        model_path = self.get_yolo_model_path(model_type, size, format_type)
        
        # Đã có file rồi
        if model_path.exists():
            return model_path
        
        # Tìm file .pt gốc để export
        pt_path = self.get_yolo_model_path(model_type, size, 'pytorch')
        use_int8 = False  # Flag để quantize INT8
        
        # Nếu không có file .pt ở size hiện tại, thử fallback sang size lớn hơn
        if not pt_path.exists():
            fallback_order = ['small', 'medium', 'large']
            current_idx = fallback_order.index(size.lower()) if size.lower() in fallback_order else -1
            
            for fallback_size in fallback_order[current_idx + 1:]:
                fallback_pt = self.get_yolo_model_path(model_type, fallback_size, 'pytorch')
                if fallback_pt.exists():
                    print(f"[WARN] Không có {size}.pt, dùng {fallback_size}.pt + INT8 quantization")
                    pt__path = fallback_pt
                    useint8 = True  # Bật quantization để bù đắp tốc độ
                    break
        
        if not pt_path.exists():
            print(f"[ERR] Không tìm thấy model gốc: {pt_path}")
            return None
        
        # Export sang format cần thiết
        quant_str = " (INT8)" if use_int8 else ""
        print(f"[INFO] Đang export model {model_type} sang {format_type}{quant_str}...")
        print(f"   Nguồn: {pt_path}")
        print(f"   Đích: {model_path}")
        
        try:
            from ultralytics import YOLO
            
            # Tự động cài đặt thư viện cần thiết
            if format_type.lower() == 'tensorrt':
                self._ensure_tensorrt()
            elif format_type.lower() == 'openvino':
                self._ensure_openvino()
            
            model = YOLO(str(pt_path))
            
            if format_type.lower() == 'onnx':
                model.export(format='onnx', simplify=True, int8=use_int8)
            elif format_type.lower() == 'openvino':
                model.export(format='openvino', int8=use_int8)
            elif format_type.lower() == 'tensorrt':
                # TensorRT cần GPU NVIDIA
                model.export(format='engine', device=0, int8=use_int8)
            
            print(f"[OK] Export thành công: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"[ERR] Export thất bại: {e}")
            # Fallback về pytorch
            print(f"[WARN] Sử dụng model pytorch thay thế")
            return pt_path
    
    # Đảm bảo TensorRT đã cài đặt
    def _ensure_tensorrt(self):
        try:
            import tensorrt
        except ImportError:
            print("[INFO] Đang cài đặt TensorRT...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorrt', '-q'])
            print("[OK] Đã cài TensorRT!")
    
    # Đảm bảo OpenVINO đã cài đặt
    def _ensure_openvino(self):
        try:
            import openvino
        except ImportError:
            print("[INFO] Đang cài đặt OpenVINO...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openvino', '-q'])
            print("[OK] Đã cài OpenVINO!")

    # Lấy thuộc tính: giá trị cấu hình
    def __getattr__(self, name):
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return _ConfigProxy(value)
            return value
        raise AttributeError(f"'Settings' object has no attribute '{name}'")

    # Lấy giá trị (hỗ trợ dấu chấm)
    def get(self, key, default=None):
        keys = key.split('.')
        value = self._config
        for k in keys:
            value = value[k]
        return value

    # Cập nhật cấu hình
    def set(self, key, value):
        keys = key.split('.')
        current = self._config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    # Lưu lại config
    def save(self):
        data = self._prepare_for_save(self._config)
        save_raw_config(data)
        
    # Hỗ trợ: Chuẩn bị cho YAML
    def _prepare_for_save(self, data):
        if isinstance(data, dict):
            return {k: self._prepare_for_save(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_save(i) for i in data]
        elif isinstance(data, Path):
            # Cố gắng tạo đường dẫn tương đối so với gốc dự án
            return str(data.relative_to(self.base_dir))
        return data

# Hỗ trợ: Truy cập dict như object
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

# Loại Alert
class AlertType(Enum):
    KNOWN_PERSON = "nguoi_quen"
    STRANGER = "nguoi_la"
    FIRE_WARNING = "lua_chay_nghi_ngo"
    FIRE_CRITICAL = "lua_chay_khan_cap"
    ANOMALOUS_BEHAVIOR = "hanh_vi_bat_thuong"
    FALL = "te_nga"

# Mức độ ưu tiên
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

# Lấy path file config
def _get_config_path():
    base_dir = Path(__file__).parent.parent
    return base_dir / 'config/config.yaml'

# Tải config raw
def load_raw_config():
    config_path = _get_config_path()
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Save config raw
def save_raw_config(data):
    config_path = _get_config_path()
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return True

# Cập nhật và lưu config
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

# Thêm camera (nhanh)
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

settings = Settings()

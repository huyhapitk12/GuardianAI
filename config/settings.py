"""Application settings loaded from environment or defaults"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PathConfig:
    """File paths configuration"""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    @property
    def data_dir(self) -> Path:
        return self.base_dir / "Data" / "Image"
    
    @property
    def model_dir(self) -> Path:
        return self.base_dir / "Data" / "Model"
    
    @property
    def tmp_dir(self) -> Path:
        path = self.base_dir / "tmp"
        path.mkdir(exist_ok=True)
        return path
    
    @property
    def embedding_file(self) -> Path:
        return self.base_dir / "Data" / "known_embeddings.pkl"
    
    @property
    def names_file(self) -> Path:
        return self.base_dir / "Data" / "known_names.pkl"
    
    @property
    def log_csv(self) -> Path:
        return self.base_dir / "events_log.csv"
    
    @property
    def alarm_sound(self) -> Path:
        return self.base_dir / "Data" / "Audio" / "alarm.mp3"

@dataclass
class TelegramConfig:
    """Telegram bot configuration"""
    token: str = os.getenv("TELEGRAM_TOKEN", "7874716410:AAFKDHbXiyeaZZzaJGyA2_Qr6r-5mxf3K-g")
    chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "-4901296113")

@dataclass
class ModelConfig:
    """AI/ML model configuration"""
    face_model_name: str = "buffalo_l"
    yolo_size: str = "medium"  # "medium" or "small"
    insightface_ctx_id: int = -1  # -1 for CPU, 0+ for GPU
    insightface_det_size: tuple = (640, 640)
    
    def get_yolo_fire_path(self, paths: PathConfig) -> Path:
        return paths.model_dir / self.yolo_size.capitalize() / "Fire" / f"{self.yolo_size}_openvino_model"
    
    def get_yolo_person_path(self, paths: PathConfig) -> Path:
        return paths.model_dir / self.yolo_size.capitalize() / "Person" / f"{self.yolo_size}_openvino_model"

@dataclass
class AIConfig:
    """AI assistant configuration"""
    api_base: Optional[str] = "https://generativelanguage.googleapis.com/v1beta/openai/"#os.getenv("API_BASE")
    api_key: Optional[str] = "AIzaSyBvBkXirUSiTAqXNykZjfoHWwdPqZDZYnA"#os.getenv("API_KEY")
    model: Optional[str] = "gemini-2.5-flash"#.getenv("AI_MODEL")
    
    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.api_base)

@dataclass
class CameraConfig:
    """Camera configuration"""
    ip_camera_url: Optional[str] = os.getenv("IP_CAMERA_URL")
    
    @property
    def source(self):
        """Return camera source (URL or device index)"""
        return self.ip_camera_url if self.ip_camera_url else 0

@dataclass
class Settings:
    """Main application settings"""
    paths: PathConfig = field(default_factory=PathConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)

# Global settings instance
settings = Settings()
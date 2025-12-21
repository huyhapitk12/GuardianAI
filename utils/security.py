# Các tiện ích bảo mật - mã hóa/giải mã
import cv2
import numpy as np
from pathlib import Path
from cryptography.fernet import Fernet
from config import settings


# Xử lý mã hóa/giải mã file
class SecurityManager:
    
    
    def __init__(self):
        self._key_file = settings.base_dir / 'secret.key'
        self._key = self._load_or_create_key()
        self._cipher = Fernet(self._key)
    
    def _load_or_create_key(self):
        if self._key_file.exists():
            return self._key_file.read_bytes()
        
        key = Fernet.generate_key()
        self._key_file.write_bytes(key)
        return key
    
    def encrypt(self, data):
        return self._cipher.encrypt(data)
    
    def decrypt(self, data):
        return self._cipher.decrypt(data)
    
    # Mã hóa file tại chỗ (ghi đè file gốc)
    def encrypt_file(self, path):
        path = Path(path)
        if not path.exists():
            return
        
        data = path.read_bytes()
        path.write_bytes(self.encrypt(data))
    
    # Giải mã file và trả về bytes
    def decrypt_file(self, path):
        try:
            return self.decrypt(Path(path).read_bytes())
        except Exception:
            return None
    
    # Lưu ảnh có mã hóa
    def save_image(self, path, image):
        try:
            _, encoded = cv2.imencode('.jpg', image)
            encrypted = self.encrypt(encoded.tobytes())
            Path(path).write_bytes(encrypted)
            return True
        except Exception:
            return False
    
    # Tải và giải mã ảnh
    def load_image(self, path):
        try:
            decrypted = self.decrypt_file(path)
            if decrypted:
                arr = np.frombuffer(decrypted, np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
            
            # Fallback to unencrypted
            return cv2.imread(str(path))
        except Exception:
            return None


# Biến toàn cục
security = SecurityManager()
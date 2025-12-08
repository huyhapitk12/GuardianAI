"""Security utilities - encryption/decryption"""

from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union
from cryptography.fernet import Fernet

from config import settings


class SecurityManager:
    """Handle file encryption/decryption"""
    
    __slots__ = ('_key', '_cipher', '_key_file')
    
    def __init__(self):
        self._key_file = settings.base_dir / 'secret.key'
        self._key = self._load_or_create_key()
        self._cipher = Fernet(self._key)
    
    def _load_or_create_key(self) -> bytes:
        if self._key_file.exists():
            return self._key_file.read_bytes()
        
        key = Fernet.generate_key()
        self._key_file.write_bytes(key)
        return key
    
    def encrypt(self, data: bytes) -> bytes:
        return self._cipher.encrypt(data)
    
    def decrypt(self, data: bytes) -> bytes:
        return self._cipher.decrypt(data)
    
    def encrypt_file(self, path: Union[str, Path]):
        """Encrypt file in place"""
        path = Path(path)
        if not path.exists():
            return
        
        data = path.read_bytes()
        path.write_bytes(self.encrypt(data))
    
    def decrypt_file(self, path: Union[str, Path]) -> Optional[bytes]:
        """Decrypt file and return bytes"""
        try:
            return self.decrypt(Path(path).read_bytes())
        except Exception:
            return None
    
    def save_image(self, path: Union[str, Path], image: np.ndarray) -> bool:
        """Save image with encryption"""
        try:
            _, encoded = cv2.imencode('.jpg', image)
            encrypted = self.encrypt(encoded.tobytes())
            Path(path).write_bytes(encrypted)
            return True
        except Exception:
            return False
    
    def load_image(self, path: Union[str, Path]) -> Optional[np.ndarray]:
        """Load and decrypt image"""
        try:
            decrypted = self.decrypt_file(path)
            if decrypted:
                arr = np.frombuffer(decrypted, np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
            
            # Fallback to unencrypted
            return cv2.imread(str(path))
        except Exception:
            return None


# Global instance
security = SecurityManager()
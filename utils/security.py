import os
import cv2
import numpy as np
from pathlib import Path
from cryptography.fernet import Fernet
from config import settings

class SecurityManager:
    def __init__(self):
        self.key_file = settings.base_dir / 'secret.key'
        self.key = self._load_or_generate_key()
        self.cipher = Fernet(self.key)

    def _load_or_generate_key(self):
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            return key

    def encrypt_data(self, data: bytes) -> bytes:
        return self.cipher.encrypt(data)

    def decrypt_data(self, data: bytes) -> bytes:
        return self.cipher.decrypt(data)

    def encrypt_file(self, file_path: str | Path):
        path = Path(file_path)
        if not path.exists():
            return
        
        with open(path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.encrypt_data(data)
        
        with open(path, 'wb') as f:
            f.write(encrypted_data)

    def decrypt_file(self, file_path: str | Path) -> bytes:
        path = Path(file_path)
        with open(path, 'rb') as f:
            data = f.read()
        return self.decrypt_data(data)

    def try_decrypt_file(self, file_path: str | Path) -> bytes | None:
        """Try to decrypt file, return None if not encrypted or fails"""
        try:
            return self.decrypt_file(file_path)
        except Exception:
            return None

    def decrypt_image(self, file_path: str | Path) -> np.ndarray:
        """Decrypt image and return as numpy array (OpenCV format)"""
        try:
            decrypted_data = self.decrypt_file(file_path)
            nparr = np.frombuffer(decrypted_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"ERROR: Failed to decrypt image {file_path}: {e}")
            return None

    def save_encrypted_image(self, file_path: str | Path, image: np.ndarray):
        """Save image directly as encrypted file"""
        success, encoded_img = cv2.imencode('.jpg', image)
        if success:
            encrypted_data = self.encrypt_data(encoded_img.tobytes())
            with open(file_path, 'wb') as f:
                f.write(encrypted_data)
            return True
        return False

# Global instance
security_manager = SecurityManager()

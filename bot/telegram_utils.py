"""Telegram utility functions for sending media"""
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

def send_photo(token: str, chat_id: str, photo_path: str, caption: str = "") -> bool:
    """
    Send a photo to a Telegram chat
    
    Args:
        token: Telegram bot token
        chat_id: Chat ID to send to
        photo_path: Path to the photo file
        caption: Optional caption for the photo
    
    Returns:
        True if successful, False otherwise
    """
    try:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        
        with open(photo_path, 'rb') as photo_file:
            files = {'photo': photo_file}
            data = {
                'chat_id': chat_id,
                'caption': caption
            }
            
            response = requests.post(url, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"Photo sent successfully to chat {chat_id}")
                return True
            else:
                logger.error(f"Failed to send photo: {response.status_code} - {response.text}")
                return False
                
    except FileNotFoundError:
        logger.error(f"Photo file not found: {photo_path}")
        return False
    except Exception as e:
        logger.error(f"Error sending photo: {e}")
        return False


def send_video_or_document(token: str, chat_id: str, file_path: str, caption: str = "") -> bool:
    """
    Send a video or document to a Telegram chat
    Automatically chooses between video and document based on file size
    
    Args:
        token: Telegram bot token
        chat_id: Chat ID to send to
        file_path: Path to the file
        caption: Optional caption for the file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        # Get file size in MB
        file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
        
        # Use document for files larger than 50MB (Telegram video limit is 50MB)
        # or if file is not a video format
        is_video = file_path_obj.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        use_video = is_video and file_size_mb < 50
        
        if use_video:
            return _send_video(token, chat_id, file_path, caption)
        else:
            return _send_document(token, chat_id, file_path, caption)
            
    except Exception as e:
        logger.error(f"Error sending file: {e}")
        return False


def _send_video(token: str, chat_id: str, video_path: str, caption: str = "") -> bool:
    """Send a video file to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{token}/sendVideo"
        
        with open(video_path, 'rb') as video_file:
            files = {'video': video_file}
            data = {
                'chat_id': chat_id,
                'caption': caption,
                'supports_streaming': True
            }
            
            response = requests.post(url, files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                logger.info(f"Video sent successfully to chat {chat_id}")
                return True
            else:
                logger.error(f"Failed to send video: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"Error sending video: {e}")
        return False


def _send_document(token: str, chat_id: str, document_path: str, caption: str = "") -> bool:
    """Send a document file to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{token}/sendDocument"
        
        with open(document_path, 'rb') as document_file:
            files = {'document': document_file}
            data = {
                'chat_id': chat_id,
                'caption': caption
            }
            
            response = requests.post(url, files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                logger.info(f"Document sent successfully to chat {chat_id}")
                return True
            else:
                logger.error(f"Failed to send document: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"Error sending document: {e}")
        return False

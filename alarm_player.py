# alarm_player.py
import pygame
import os
import logging
import time
import threading
from config import ALARM_SOUND_FILE, ALARM_FADE_IN_DURATION, ALARM_START_VOLUME, ALARM_MAX_VOLUME

log = logging.getLogger("alarm_player")

# --- THÊM MỚI: Biến toàn cục để quản lý luồng tăng âm lượng ---
_alarm_thread = None
_stop_thread_flag = threading.Event()
# --- KẾT THÚC THÊM MỚI ---

def init_alarm():
    """Khởi tạo pygame mixer để sẵn sàng phát âm thanh."""
    try:
        pygame.mixer.init()
        log.info("Trình phát âm thanh (pygame.mixer) đã được khởi tạo thành công.")
        return True
    except Exception as e:
        log.error(f"Lỗi khi khởi tạo trình phát âm thanh: {e}")
        return False

def _volume_fade_in_worker():
    """
    Worker chạy trong một luồng riêng để tăng âm lượng từ từ.
    """
    start_time = time.time()
    duration = ALARM_FADE_IN_DURATION
    start_vol = ALARM_START_VOLUME
    max_vol = ALARM_MAX_VOLUME
    
    log.info(f"Bắt đầu tăng âm lượng trong {duration} giây...")
    
    while not _stop_thread_flag.is_set():
        elapsed = time.time() - start_time
        if elapsed >= duration:
            break
            
        # Tính toán âm lượng hiện tại dựa trên thời gian đã trôi qua
        progress = elapsed / duration
        current_volume = start_vol + (max_vol - start_vol) * progress
        
        # Đảm bảo âm lượng không vượt quá giới hạn
        final_volume = min(current_volume, max_vol)
        
        try:
            pygame.mixer.music.set_volume(final_volume)
        except pygame.error as e:
            log.error(f"Lỗi khi đặt âm lượng: {e}")
            break # Thoát khỏi vòng lặp nếu có lỗi
            
        time.sleep(0.1) # Cập nhật âm lượng 10 lần mỗi giây

    # Sau khi vòng lặp kết thúc, đảm bảo âm lượng được đặt ở mức tối đa (nếu không bị dừng)
    if not _stop_thread_flag.is_set():
        try:
            pygame.mixer.music.set_volume(max_vol)
            log.info(f"Âm lượng đã đạt mức tối đa: {max_vol}")
        except pygame.error:
            pass # Bỏ qua nếu có lỗi ở bước cuối

def play_alarm():
    """
    Phát file âm thanh báo động với hiệu ứng tăng dần âm lượng.
    """
    global _alarm_thread
    
    if not pygame.mixer.get_init():
        log.warning("Trình phát âm thanh chưa được khởi tạo. Bỏ qua phát báo động.")
        return

    if not os.path.exists(ALARM_SOUND_FILE):
        log.error(f"Không tìm thấy file âm thanh báo động tại: {ALARM_SOUND_FILE}")
        return

    try:
        if not pygame.mixer.music.get_busy():
            # Đặt cờ dừng về trạng thái chưa được thiết lập
            _stop_thread_flag.clear()

            pygame.mixer.music.load(ALARM_SOUND_FILE)
            
            # Bắt đầu phát ở âm lượng thấp
            pygame.mixer.music.set_volume(ALARM_START_VOLUME)
            pygame.mixer.music.play(loops=-1)
            log.info(f"Đang phát còi báo động từ file: {ALARM_SOUND_FILE} (bắt đầu ở âm lượng {ALARM_START_VOLUME})")

            # Bắt đầu luồng để tăng âm lượng
            _alarm_thread = threading.Thread(target=_volume_fade_in_worker, daemon=True)
            _alarm_thread.start()
        else:
            log.info("Còi báo động đã đang được phát.")
    except Exception as e:
        log.error(f"Lỗi khi phát âm thanh báo động: {e}")

def stop_alarm():
    """Dừng phát âm thanh báo động và luồng tăng âm lượng."""
    global _alarm_thread
    
    if not pygame.mixer.get_init():
        return
        
    try:
        if pygame.mixer.music.get_busy():
            # Gửi tín hiệu dừng cho luồng tăng âm lượng
            _stop_thread_flag.set()
            
            # Chờ luồng kết thúc (với timeout nhỏ để không bị treo)
            if _alarm_thread and _alarm_thread.is_alive():
                _alarm_thread.join(timeout=0.5)
            
            pygame.mixer.music.stop()
            log.info("Đã dừng còi báo động.")
            
            _alarm_thread = None # Reset biến luồng
    except Exception as e:
        log.error(f"Lỗi khi dừng âm thanh báo động: {e}")
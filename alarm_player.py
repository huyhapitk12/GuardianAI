# alarm_player.py
import logging
import os
import threading
import time

import pygame

from config import (ALARM_FADE_IN_DURATION, ALARM_MAX_VOLUME,
                    ALARM_SOUND_FILE, ALARM_START_VOLUME)

log = logging.getLogger("alarm_player")

# Biến toàn cục để quản lý luồng tăng âm lượng
_alarm_thread = None
_stop_thread_flag = threading.Event()

def init_alarm():
    """Khởi tạo pygame mixer để sẵn sàng phát âm thanh."""
    try:
        pygame.mixer.init()
        log.info("Trình phát âm thanh (pygame.mixer) đã khởi tạo thành công.")
        return True
    except Exception as e:
        log.error(f"Lỗi khi khởi tạo trình phát âm thanh: {e}")
        return False

def _volume_fade_in_worker():
    """Worker chạy trong luồng riêng để tăng âm lượng từ từ."""
    start_time = time.time()
    log.info(f"Bắt đầu tăng âm lượng trong {ALARM_FADE_IN_DURATION} giây...")

    while not _stop_thread_flag.is_set():
        elapsed = time.time() - start_time
        if elapsed >= ALARM_FADE_IN_DURATION:
            break

        progress = elapsed / ALARM_FADE_IN_DURATION
        current_volume = ALARM_START_VOLUME + (ALARM_MAX_VOLUME - ALARM_START_VOLUME) * progress
        final_volume = min(current_volume, ALARM_MAX_VOLUME)

        try:
            pygame.mixer.music.set_volume(final_volume)
        except pygame.error as e:
            log.error(f"Lỗi khi đặt âm lượng: {e}")
            break
        time.sleep(0.1)

    if not _stop_thread_flag.is_set():
        try:
            pygame.mixer.music.set_volume(ALARM_MAX_VOLUME)
            log.info(f"Âm lượng đã đạt mức tối đa: {ALARM_MAX_VOLUME}")
        except pygame.error:
            pass

def play_alarm():
    """Phát file âm thanh báo động với hiệu ứng tăng dần âm lượng."""
    global _alarm_thread

    if not pygame.mixer.get_init():
        log.warning("Trình phát âm thanh chưa khởi tạo. Bỏ qua phát báo động.")
        return
    if not os.path.exists(ALARM_SOUND_FILE):
        log.error(f"Không tìm thấy file âm thanh báo động: {ALARM_SOUND_FILE}")
        return

    try:
        if not pygame.mixer.music.get_busy():
            _stop_thread_flag.clear()
            pygame.mixer.music.load(ALARM_SOUND_FILE)
            pygame.mixer.music.set_volume(ALARM_START_VOLUME)
            pygame.mixer.music.play(loops=-1)
            log.info(f"Đang phát còi báo động từ file: {ALARM_SOUND_FILE}")

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
            _stop_thread_flag.set()
            if _alarm_thread and _alarm_thread.is_alive():
                _alarm_thread.join(timeout=0.5)

            pygame.mixer.music.stop()
            log.info("Đã dừng còi báo động.")
            _alarm_thread = None
    except Exception as e:
        log.error(f"Lỗi khi dừng âm thanh báo động: {e}")
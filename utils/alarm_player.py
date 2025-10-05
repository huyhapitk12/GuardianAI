"""Alarm sound player with fade-in effect"""
import pygame
import os
import logging
import time
import threading
from typing import Optional
from pathlib import Path
from config.constants import (
    ALARM_FADE_IN_DURATION,
    ALARM_START_VOLUME,
    ALARM_MAX_VOLUME
)

logger = logging.getLogger(__name__)

class AlarmPlayer:
    """Handles alarm sound playback"""
    
    def __init__(self, sound_file: Path):
        self.sound_file = sound_file
        self._alarm_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize pygame mixer"""
        try:
            pygame.mixer.init()
            self._initialized = True
            logger.info("Alarm player initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize alarm player: {e}")
            return False

    def play(self) -> None:
        """Play alarm sound with fade-in effect"""
        if not self._initialized:
            logger.warning("Alarm player not initialized")
            return

        if not self.sound_file.exists():
            logger.error(f"Alarm sound file not found: {self.sound_file}")
            return

        try:
            if not pygame.mixer.music.get_busy():
                self._stop_flag.clear()
                
                pygame.mixer.music.load(str(self.sound_file))
                pygame.mixer.music.set_volume(ALARM_START_VOLUME)
                pygame.mixer.music.play(loops=-1)
                
                logger.info(f"Playing alarm from: {self.sound_file}")
                
                # Start fade-in thread
                self._alarm_thread = threading.Thread(
                    target=self._fade_in_worker,
                    daemon=True
                )
                self._alarm_thread.start()
            else:
                logger.info("Alarm already playing")
        except Exception as e:
            logger.error(f"Error playing alarm: {e}")

    def stop(self) -> None:
        """Stop alarm playback"""
        if not self._initialized:
            return
        
        try:
            if pygame.mixer.music.get_busy():
                self._stop_flag.set()
                
                if self._alarm_thread and self._alarm_thread.is_alive():
                    self._alarm_thread.join(timeout=0.5)
                
                pygame.mixer.music.stop()
                logger.info("Alarm stopped")
                
                self._alarm_thread = None
        except Exception as e:
            logger.error(f"Error stopping alarm: {e}")

    def _fade_in_worker(self) -> None:
        """Worker thread to gradually increase volume"""
        start_time = time.time()
        duration = ALARM_FADE_IN_DURATION
        start_vol = ALARM_START_VOLUME
        max_vol = ALARM_MAX_VOLUME
        
        while not self._stop_flag.is_set():
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
            
            progress = elapsed / duration
            current_volume = start_vol + (max_vol - start_vol) * progress
            final_volume = min(current_volume, max_vol)
            
            try:
                pygame.mixer.music.set_volume(final_volume)
            except pygame.error as e:
                logger.error(f"Error setting volume: {e}")
                break
            
            time.sleep(0.1)
        
        # Set final volume
        if not self._stop_flag.is_set():
            try:
                pygame.mixer.music.set_volume(max_vol)
                logger.info(f"Volume reached maximum: {max_vol}")
            except pygame.error:
                pass

# Global alarm player instance
_alarm_player: Optional[AlarmPlayer] = None

def init_alarm(sound_file: Path) -> bool:
    """Initialize global alarm player"""
    global _alarm_player
    _alarm_player = AlarmPlayer(sound_file)
    return _alarm_player.initialize()

def play_alarm() -> None:
    """Play alarm"""
    if _alarm_player:
        _alarm_player.play()

def stop_alarm() -> None:
    """Stop alarm"""
    if _alarm_player:
        _alarm_player.stop()
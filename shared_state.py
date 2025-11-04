# shared_state.py
import queue
from state_manager import StateManager
from video_recorder import Recorder
from spam_guard import SpamGuard
from config import DEBOUNCE_SECONDS, SPAM_GUARD_MIN_INTERVAL, SPAM_GUARD_MAX_PER_MINUTE, RECORDER_FPS, RECORDER_FOURCC

# Đây là "nguồn chân lý" duy nhất cho các đối tượng chia sẻ
state = StateManager()
response_queue = queue.Queue()
recorder = Recorder(fps=RECORDER_FPS, fourcc_str=RECORDER_FOURCC)
guard = SpamGuard(
    debounce_seconds=DEBOUNCE_SECONDS, 
    min_interval=SPAM_GUARD_MIN_INTERVAL, 
    max_per_minute=SPAM_GUARD_MAX_PER_MINUTE
)

camera_instance = None
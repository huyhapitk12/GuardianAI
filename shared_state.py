# shared_state.py
import queue

from config import (DEBOUNCE_SECONDS, RECORDER_FPS, RECORDER_FOURCC,
                    SPAM_GUARD_MAX_PER_MINUTE, SPAM_GUARD_MIN_INTERVAL)
from spam_guard import SpamGuard
from state_manager import StateManager
from video_recorder import Recorder

# Đây là "nguồn chân lý" duy nhất cho các đối tượng chia sẻ trong toàn bộ ứng dụng.
state = StateManager()
response_queue = queue.Queue()
recorder = Recorder(fps=RECORDER_FPS, fourcc_str=RECORDER_FOURCC)
guard = SpamGuard(
    debounce_seconds=DEBOUNCE_SECONDS,
    min_interval=SPAM_GUARD_MIN_INTERVAL,
    max_per_minute=SPAM_GUARD_MAX_PER_MINUTE
)

# Lưu trữ các instance camera đang hoạt động
active_cameras = {}
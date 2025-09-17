# shared_state.py
import queue
from state_manager import StateManager
from video_recorder import Recorder
from spam_guard import SpamGuard
from config import DEBOUNCE_SECONDS

# Đây là "nguồn chân lý" duy nhất cho các đối tượng chia sẻ
state = StateManager()
response_queue = queue.Queue()
recorder = Recorder()
guard = SpamGuard(debounce_seconds=DEBOUNCE_SECONDS, min_interval=10, max_per_minute=4)
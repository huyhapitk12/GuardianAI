# state_manager.py
import time, threading, uuid
from config import DEBOUNCE_SECONDS  # added debounce config

class StateManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = {}  # alert_id -> info

    def create_alert(self, typ, chat_id, asked_for=None):
        import time
        current_time = time.time()
        # Check for an existing unresolved alert within the debounce period
        for alert in self.active.values():
            if (alert['type'] == typ and str(alert['chat_id']) == str(chat_id) and 
                not alert['resolved'] and (current_time - alert['ts'] < DEBOUNCE_SECONDS)):
                return alert['id']
        aid = uuid.uuid4().hex
        info = {"id": aid, "type": typ, "chat_id": chat_id, "asked_for": asked_for, "ts": current_time, "resolved": False, "reply": None}
        with self.lock:
            self.active[aid] = info
        return aid

    def resolve_alert(self, aid, reply_text):
        with self.lock:
            if aid in self.active:
                self.active[aid]["resolved"] = True
                self.active[aid]["reply"] = reply_text

    def latest_unresolved_for_chat(self, chat_id):
        with self.lock:
            cands = [a for a in self.active.values() if (not a["resolved"]) and str(a["chat_id"])==str(chat_id)]
            cands.sort(key=lambda x: x["ts"], reverse=True)
            return cands[0] if cands else None

    def list_alerts(self):
        with self.lock:
            return list(self.active.values())

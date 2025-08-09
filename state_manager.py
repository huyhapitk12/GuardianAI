# state_manager.py
import time, threading, uuid

class StateManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = {}  # alert_id -> info

    def create_alert(self, typ, chat_id, asked_for=None):
        aid = uuid.uuid4().hex
        info = {"id":aid, "type":typ, "chat_id":chat_id, "asked_for":asked_for, "ts":time.time(), "resolved":False, "reply":None}
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

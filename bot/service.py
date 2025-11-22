import asyncio
import time
import threading
import json
import re
import requests
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Tuple, List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Config imports
from config import settings, ActionCode
from utils.security import security_manager

# ============================================================================
# TELEGRAM LIBRARY CHECK
# ============================================================================
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
    from telegram.constants import ParseMode
    TELEGRAM_AVAILABLE = True
except ImportError:
    # Dummy classes for when telegram is not available
    class Update:
        ALL_TYPES = []
        message = None
        callback_query = None
        effective_user = None
        effective_chat = None
    
    class InlineKeyboardButton:
        def __init__(self, text, callback_data=None):
            self.text = text
            self.callback_data = callback_data
    
    class InlineKeyboardMarkup:
        def __init__(self, keyboard):
            self.keyboard = keyboard
        def to_dict(self):
            return {"inline_keyboard": self.keyboard}
    
    class Application:
        @classmethod
        def builder(cls): return cls()
        def token(self, token): return self
        def build(self): return DummyApp()
    
    class CommandHandler:
        def __init__(self, command, handler): pass
        def add_to_app(self, app): pass
    
    class MessageHandler:
        def __init__(self, *args, **kwargs): pass
        def add_to_app(self, app): pass
    
    class CallbackQueryHandler:
        def __init__(self, handler): pass
        def add_to_app(self, app): pass
    
    class ContextTypes:
        DEFAULT_TYPE = None
    
    class filters:
        class _BaseFilter:
            def __and__(self, other): return self
            def __or__(self, other): return self
            def __invert__(self): return self
        TEXT = _BaseFilter()
        COMMAND = _BaseFilter()
    
    class ParseMode:
        MARKDOWN = 'Markdown'
        HTML = 'HTML'
    
    TELEGRAM_AVAILABLE = False
    print("WARNING: python-telegram-bot not available, bot functionality will be disabled")

class DummyApp:
    def add_handler(self, handler): pass
    async def initialize(self): pass
    async def start(self): pass
    async def stop(self): pass
    async def shutdown(self): pass
    @property
    def updater(self): return DummyUpdater()
    @property
    def running(self): return False

class DummyUpdater:
    async def start_polling(self, **kwargs): pass
    @property
    def running(self): return False
    async def stop(self): pass

# ============================================================================
# TELEGRAM UTILS
# ============================================================================

def _create_session_with_retry():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def send_photo(token: str, chat_id: str, photo_path: str, caption: str = "", reply_markup: Optional[dict] = None) -> bool:
    for attempt in range(3):
        try:
            url = f"https://api.telegram.org/bot{token}/sendPhoto"
            session = _create_session_with_retry()
            decrypted_data = security_manager.try_decrypt_file(photo_path)
            
            if decrypted_data:
                files = {'photo': ('image.jpg', decrypted_data)}
            else:
                files = {'photo': open(photo_path, 'rb')}
                
            data = {'chat_id': chat_id, 'caption': caption}
            if reply_markup: data['reply_markup'] = json.dumps(reply_markup)
            
            response = session.post(url, files=files, data=data, timeout=30)
            if not decrypted_data: files['photo'].close()
            
            if response.status_code == 200: return True
            time.sleep(2)
        except Exception as e:
            print(f"ERROR: Error sending photo: {e}")
    return False

def send_video_or_document(token: str, chat_id: str, file_path: str, caption: str = "") -> bool:
    try:
        path = Path(file_path)
        if not path.exists(): return False
        size_mb = path.stat().st_size / (1024 * 1024)
        is_video = path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        if is_video and size_mb < 50:
            return _send_media(token, chat_id, file_path, caption, 'video')
        else:
            return _send_media(token, chat_id, file_path, caption, 'document')
    except: return False

def _send_media(token: str, chat_id: str, file_path: str, caption: str, media_type: str) -> bool:
    endpoint = "sendVideo" if media_type == 'video' else "sendDocument"
    for attempt in range(3):
        try:
            url = f"https://api.telegram.org/bot{token}/{endpoint}"
            session = _create_session_with_retry()
            decrypted_data = security_manager.try_decrypt_file(file_path)
            
            filename = f"file.{'mp4' if media_type == 'video' else 'dat'}"
            if decrypted_data:
                files = {media_type: (filename, decrypted_data)}
            else:
                files = {media_type: open(file_path, 'rb')}
                
            data = {'chat_id': chat_id, 'caption': caption}
            response = session.post(url, files=files, data=data, timeout=30)
            if not decrypted_data: files[media_type].close()
            
            if response.status_code == 200: return True
            time.sleep(2)
        except: pass
    return False

# ============================================================================
# AI ASSISTANT
# ============================================================================

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None
    OPENAI_AVAILABLE = False

AI_SYSTEM_INSTRUCTION = """
You are Guardian Bot - an intelligent, friendly security assistant that responds concisely in Vietnamese.
Embed codes: [ACTION:TOGGLE_ON], [ACTION:TOGGLE_OFF], [ACTION:GET_IMAGE], [ACTION:ALARM_ON], [ACTION:ALARM_OFF]
""".strip()

class AIAssistant:
    def __init__(self):
        self.enabled = settings.ai.enabled
        self.use_llm = settings.ai.use_llm_for_classification
        self.client = None
        self.history: Dict[str, list] = {}
        
        if self.enabled and OPENAI_AVAILABLE and AsyncOpenAI:
            try:
                self.client = AsyncOpenAI(base_url=settings.ai.api_base, api_key=settings.ai.api_key, timeout=settings.telegram.httpx_timeout)
            except: self.enabled = False
        elif self.enabled: self.enabled = False

    async def process_message(self, chat_id: str, message: str, user_info: Optional[Dict] = None) -> Tuple[str, Optional[str]]:
        if not self.enabled: return f"[AI Đã tắt] {message}", None
        if not self.client: return self._simple_response(message), None
        try: return await self._process_with_llm(chat_id, message, user_info)
        except Exception as e: return f"Lỗi: {e}", None

    def _simple_response(self, message: str) -> str:
        msg = message.lower()
        if any(w in msg for w in ['bật', 'on']): return "Đã bật. [ACTION:TOGGLE_ON]"
        if any(w in msg for w in ['tắt', 'off']): return "Đã tắt. [ACTION:TOGGLE_OFF]"
        if any(w in msg for w in ['camera', 'ảnh']): return "Ảnh camera. [ACTION:GET_IMAGE]"
        if any(w in msg for w in ['báo động', 'alarm']): return "Báo động! [ACTION:ALARM_ON]"
        return "AI không khả dụng."

    async def _process_with_llm(self, chat_id: str, message: str, user_info: Optional[Dict]) -> Tuple[str, Optional[str]]:
        hist = self.history.get(chat_id, [])
        reply = await self._call_llm(message, hist)
        
        action = None
        match = re.search(r'\[ACTION:([^\]]+)\]', reply)
        if match:
            action = match.group(1)
            reply = re.sub(r'\s*\[ACTION:[^\]]+\]\s*', '', reply).strip()
            
        hist.append({"role": "user", "parts": [{"text": message}]})
        hist.append({"role": "model", "parts": [{"text": reply}]})
        if len(hist) > 20: hist = hist[-20:]
        self.history[chat_id] = hist
        return reply, action

    async def _call_llm(self, prompt: str, history: List[Dict]) -> str:
        if not self.client: return "Unavailable"
        msgs = [{"role": "system", "content": AI_SYSTEM_INSTRUCTION}]
        for item in history:
            role = "assistant" if item["role"] == "model" else item["role"]
            msgs.append({"role": role, "content": item["parts"][0]["text"]})
        msgs.append({"role": "user", "content": prompt})
        
        resp = await self.client.chat.completions.create(
            model=settings.ai.model, messages=msgs, max_tokens=settings.ai.max_tokens, temperature=settings.ai.temperature
        )
        return resp.choices[0].message.content.strip() if resp.choices else ""

    def add_system_message(self, chat_id: str, message: str):
        """Adds a system-generated message to the chat history for context."""
        if not self.enabled or not self.client:
            return
        
        hist = self.history.get(chat_id, [])
        hist.append({"role": "system", "parts": [{"text": f"System Alert: {message}"}]})
        if len(hist) > 20: hist = hist[-20:]
        self.history[chat_id] = hist

    def clear_history(self, chat_id: str):
        if chat_id in self.history: del self.history[chat_id]

# ============================================================================
# HANDLERS & BOT
# ============================================================================

def create_fire_alert_keyboard(alert_id: str):
    if not TELEGRAM_AVAILABLE: return None
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Cháy thật", callback_data=f"fire_real:{alert_id}"),
         InlineKeyboardButton("❌ Giả", callback_data=f"fire_false:{alert_id}")],
        [InlineKeyboardButton("📞 114", callback_data=f"fire_call:{alert_id}")]
    ])

def create_person_alert_keyboard(alert_id: str):
    if not TELEGRAM_AVAILABLE: return None
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Quen", callback_data=f"person_yes:{alert_id}"),
        InlineKeyboardButton("❌ Lạ", callback_data=f"person_no:{alert_id}")
    ]])

class TelegramHandlers:
    def __init__(self, state_manager, ai_assistant, spam_guard, alarm_player, camera_snapshot_func, camera_manager):
        self.state = state_manager
        self.ai = ai_assistant
        self.spam_guard = spam_guard
        self.alarm_player = alarm_player
        self.camera_manager = camera_manager
        self.get_snapshot = camera_snapshot_func
        self.response_queue = None

    async def start_cmd(self, update: Update, context: Any):
        if update.message: await update.message.reply_text("🛡️ Guardian Bot Sẵn sàng. /status, /get_image, /detect")

    async def status_cmd(self, update: Update, context: Any):
        if not update.message: return
        alerts = self.state.list_alerts()
        status = (f"📊 *Trạng thái*\nPhát hiện: {'🟢' if self.state.is_person_detection_enabled() else '🔴'}\n"
                  f"Cảnh báo: {len(alerts)}")
        await update.message.reply_text(status, parse_mode='Markdown')

    async def toggle_detection_cmd(self, update: Update, context: Any):
        if not update.message: return
        if not context.args:
            msg = "📸 *Camera:*\nToàn cục: " + ("🟢" if self.state.is_person_detection_enabled() else "🔴") + "\n"
            for cam in self.camera_manager.cameras:
                msg += f"Cam {cam}: {'🟢' if self.state.is_person_detection_enabled(cam) else '🔴'}\n"
            await update.message.reply_text(msg + "\nDùng /detect <id>", parse_mode='Markdown')
            return
            
        cam_id = context.args[0]
        if cam_id in self.camera_manager.cameras:
            new_state = not self.state.is_person_detection_enabled(cam_id)
            self.state.set_person_detection_enabled(new_state, cam_id)
            await update.message.reply_text(f"Cam {cam_id}: {'🟢' if new_state else '🔴'}")
        else:
            await update.message.reply_text("❌ ID Camera không hợp lệ")

    async def get_image_cmd(self, update: Update, context: Any):
        if update.message:
            source = context.args[0] if context.args else None
            await update.message.reply_text("📸 Đang chụp...")
            self.get_snapshot(str(update.message.chat_id), source)

    async def toggle_alarm_cmd(self, update: Update, context: Any):
        if self.alarm_player.is_alarm_playing:
            self.alarm_player.stop()
            await update.message.reply_text("✅ Đã tắt báo động")
        else:
            self.alarm_player.play()
            await update.message.reply_text("🚨 Đã bật báo động")

    async def message_listener(self, update: Update, context: Any):
        if not update.message or not update.message.text: return
        text = update.message.text.strip()
        chat_id = str(update.message.chat_id)
        
        reply, action = await self.ai.process_message(chat_id, text, self._extract_user(update.effective_user))
        if action: await self._execute_action(action, chat_id)
        if reply: await update.message.reply_text(reply)

    async def _execute_action(self, code: str, chat_id: str):
        try:
            act = ActionCode[code]
            if act == ActionCode.TOGGLE_ON: self.state.set_person_detection_enabled(True)
            elif act == ActionCode.TOGGLE_OFF: self.state.set_person_detection_enabled(False)
            elif act == ActionCode.GET_IMAGE: self.get_snapshot(chat_id)
            elif act == ActionCode.ALARM_ON: self.alarm_player.play()
            elif act == ActionCode.ALARM_OFF: self.alarm_player.stop()
        except: pass

    async def button_callback_handler(self, update: Update, context: Any):
        query = update.callback_query
        await query.answer()
        action, alert_id = query.data.split(":", 1)
        self.alarm_player.stop()
        self.state.resolve_alert(alert_id, f"user:{action}")
        
        caption = query.message.caption or ""
        if "fire" in action:
            if "real" in action: 
                self.alarm_player.play()
                caption += "\n✅ XÁC NHẬN CÓ CHÁY!"
            elif "false" in action:
                self.spam_guard.mute("lua_chay", 120)
                caption += "\n❌ Báo động giả"
        elif "person" in action:
            if "yes" in action: caption += "\n✅ Người quen"
            elif "no" in action:
                self.alarm_player.play()
                caption += "\n❌ NGƯỜI LẠ!"
                
        await query.edit_message_caption(caption=caption)

    @staticmethod
    def _extract_user(user):
        return {"id": user.id, "name": user.first_name} if user else {}

class GuardianBot:
    def __init__(self, state_manager, ai_assistant, spam_guard, alarm_player, response_queue, camera_snapshot_func, camera_manager):
        self.state = state_manager
        self.ai = ai_assistant
        self.spam_guard = spam_guard
        self.alarm_player = alarm_player
        self.response_queue = response_queue
        self.camera_manager = camera_manager
        self.quit = False
        
        if not TELEGRAM_AVAILABLE: return
        
        self.handlers = TelegramHandlers(state_manager, ai_assistant, spam_guard, alarm_player, camera_snapshot_func, camera_manager)
        self.handlers.response_queue = response_queue
        self.app = Application.builder().token(settings.telegram.token).build()
        self._setup_handlers()
        print("INFO: GuardianBot initialized")

    def _setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.handlers.start_cmd))
        self.app.add_handler(CommandHandler("status", self.handlers.status_cmd))
        self.app.add_handler(CommandHandler("detect", self.handlers.toggle_detection_cmd))
        self.app.add_handler(CommandHandler("alarm", self.handlers.toggle_alarm_cmd))
        self.app.add_handler(CommandHandler("get_image", self.handlers.get_image_cmd))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handlers.message_listener))
        self.app.add_handler(CallbackQueryHandler(self.handlers.button_callback_handler))

    def schedule_alert(self, chat_id: str, image_path: str, caption: str, alert_id: str, is_fire: bool = False):
        if not TELEGRAM_AVAILABLE: return
        kb = create_fire_alert_keyboard(alert_id) if is_fire else create_person_alert_keyboard(alert_id)
        markup = kb.to_dict() if kb else None
        threading.Thread(target=lambda: send_photo(settings.telegram.token, chat_id, image_path, caption, reply_markup=markup), daemon=True).start()

    def schedule_person_alert(self, chat_id: str, image_path: str, caption: str, alert_id: str):
        """Schedules a person alert. Alias for schedule_alert with is_fire=False."""
        self.schedule_alert(chat_id, image_path, caption, alert_id, is_fire=False)

    def send_heartbeat(self):
        """Sends a heartbeat message to the configured chat ID."""
        if not TELEGRAM_AVAILABLE or not settings.telegram.chat_id:
            return

        status_text = f"❤️ Guardian đang hoạt động.\n- Phát hiện: {'🟢' if self.state.is_person_detection_enabled() else '🔴'}"
        
        # This needs to run in a separate thread to not block the caller
        threading.Thread(
            target=self._send_message_sync,
            args=(settings.telegram.chat_id, status_text),
            daemon=True
        ).start()

    def run(self):
        if not TELEGRAM_AVAILABLE: return
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._run_async())
        except Exception as e: print(f"Bot error: {e}")

    def _send_message_sync(self, chat_id: str, text: str):
        """Synchronously sends a message. For use in threads."""
        try:
            url = f"https://api.telegram.org/bot{settings.telegram.token}/sendMessage"
            data = {'chat_id': chat_id, 'text': text}
            session = _create_session_with_retry()
            response = session.post(url, data=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"ERROR: Failed to send message synchronously: {e}")

    async def _run_async(self):
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
        while not self.quit: await asyncio.sleep(1)
        await self.app.stop()
        await self.app.shutdown()

    def stop(self):
        self.quit = True

"""Telegram bot service"""

from __future__ import annotations
import asyncio
import json
import re
import time
import threading
from typing import Any, Callable, Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import settings, ActionCode
from utils import security, state_manager

# Optional telegram imports
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = None

# Optional OpenAI
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None
    OPENAI_AVAILABLE = False


def get_session() -> requests.Session:
    """Create session with retry"""
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def send_photo(chat_id: str, photo_path: str, caption: str = "", 
               reply_markup: dict = None, silent: bool = False) -> bool:
    """Send photo via Telegram API"""
    for _ in range(3):
        try:
            url = f"https://api.telegram.org/bot{settings.telegram.token}/sendPhoto"
            
            # Try decrypt
            data = security.decrypt_file(photo_path)
            if data:
                files = {'photo': ('image.jpg', data)}
            else:
                files = {'photo': open(photo_path, 'rb')}
            
            payload = {
                'chat_id': chat_id,
                'caption': caption,
                'disable_notification': silent,
            }
            if reply_markup:
                payload['reply_markup'] = json.dumps(reply_markup)
            
            response = get_session().post(url, files=files, data=payload, timeout=30)
            
            if not isinstance(files['photo'], tuple):
                files['photo'].close()
            
            if response.status_code == 200:
                return True
                
        except Exception as e:
            print(f"Send photo error: {e}")
        
        time.sleep(2)
    
    return False


def send_video(chat_id: str, video_path: str, caption: str = "") -> bool:
    """Send video via Telegram API"""
    try:
        from pathlib import Path
        path = Path(video_path)
        
        if not path.exists():
            return False
        
        size_mb = path.stat().st_size / (1024 * 1024)
        endpoint = "sendVideo" if size_mb < 50 else "sendDocument"
        
        url = f"https://api.telegram.org/bot{settings.telegram.token}/{endpoint}"
        
        data = security.decrypt_file(video_path)
        if data:
            files = {'video' if endpoint == 'sendVideo' else 'document': ('video.mp4', data)}
        else:
            files = {'video' if endpoint == 'sendVideo' else 'document': open(video_path, 'rb')}
        
        payload = {'chat_id': chat_id, 'caption': caption}
        response = get_session().post(url, files=files, data=payload, timeout=60)
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Send video error: {e}")
        return False


class AIAssistant:
    """AI-powered chat assistant"""
    
    __slots__ = ('client', 'history', 'enabled')
    
    SYSTEM_PROMPT = """Bạn là Guardian Bot - trợ lý bảo mật thông minh.
Trả lời ngắn gọn bằng tiếng Việt.
Có thể dùng: [ACTION:TOGGLE_ON], [ACTION:TOGGLE_OFF], [ACTION:GET_IMAGE], [ACTION:ALARM_ON], [ACTION:ALARM_OFF]"""
    
    def __init__(self):
        self.enabled = settings.ai.enabled
        self.client = None
        self.history: Dict[str, list] = {}
        
        if self.enabled and OPENAI_AVAILABLE and AsyncOpenAI:
            try:
                self.client = AsyncOpenAI(
                    base_url=settings.ai.api_base,
                    api_key=settings.ai.api_key,
                    timeout=settings.telegram.httpx_timeout
                )
            except Exception:
                self.enabled = False
    
    async def process(self, chat_id: str, message: str) -> Tuple[str, Optional[str]]:
        """Process message and return (reply, action)"""
        if not self.enabled or not self.client:
            return self.simple_response(message), None
        
        try:
            return await self.llm_response(chat_id, message)
        except Exception as e:
            return f"Lỗi: {e}", None
    
    def simple_response(self, message: str) -> str:
        """Simple keyword-based response"""
        msg = message.lower()
        
        if any(w in msg for w in ['bật', 'on', 'enable']):
            return "Đã bật. [ACTION:TOGGLE_ON]"
        if any(w in msg for w in ['tắt', 'off', 'disable']):
            return "Đã tắt. [ACTION:TOGGLE_OFF]"
        if any(w in msg for w in ['camera', 'ảnh', 'image']):
            return "Đang chụp. [ACTION:GET_IMAGE]"
        if any(w in msg for w in ['báo động', 'alarm']):
            return "Báo động! [ACTION:ALARM_ON]"
        
        return "AI không khả dụng."
    
    async def llm_response(self, chat_id: str, message: str) -> Tuple[str, Optional[str]]:
        """LLM-based response"""
        history = self.history.get(chat_id, [])
        
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        for item in history[-10:]:
            role = "assistant" if item["role"] == "model" else item["role"]
            messages.append({"role": role, "content": item["content"]})
        messages.append({"role": "user", "content": message})
        
        response = await self.client.chat.completions.create(
            model=settings.ai.model,
            messages=messages,
            max_tokens=settings.ai.max_tokens,
            temperature=settings.ai.temperature
        )
        
        reply = response.choices[0].message.content.strip() if response.choices else ""
        
        # Extract action
        action = None
        match = re.search(r'\[ACTION:([^\]]+)\]', reply)
        if match:
            action = match.group(1)
            reply = re.sub(r'\s*\[ACTION:[^\]]+\]\s*', '', reply).strip()
        
        # Update history
        history.append({"role": "user", "content": message})
        history.append({"role": "model", "content": reply})
        self.history[chat_id] = history[-20:]
        
        return reply, action
    
    def add_context(self, chat_id: str, message: str):
        """Add system context to history"""
        if chat_id not in self.history:
            self.history[chat_id] = []
        self.history[chat_id].append({"role": "system", "content": message})
    
    def clear(self, chat_id: str):
        """Clear chat history"""
        self.history.pop(chat_id, None)


def create_alert_keyboard(alert_id: str, is_fire: bool = False):
    """Create inline keyboard for alerts"""
    if not TELEGRAM_AVAILABLE:
        return None
    
    if is_fire:
        keyboard = [
            [InlineKeyboardButton("✅ Cháy thật", callback_data=f"fire_real:{alert_id}"),
             InlineKeyboardButton("❌ Giả", callback_data=f"fire_false:{alert_id}")],
            [InlineKeyboardButton("📞 114", callback_data=f"fire_call:{alert_id}")]
        ]
    else:
        keyboard = [[
            InlineKeyboardButton("✅ Quen", callback_data=f"person_yes:{alert_id}"),
            InlineKeyboardButton("❌ Lạ", callback_data=f"person_no:{alert_id}")
        ]]
    
    return InlineKeyboardMarkup(keyboard)


class GuardianBot:
    """Telegram bot for Guardian system"""
    
    __slots__ = ('app', 'ai', 'alarm', 'snapshot_fn', 'camera_mgr',
                 'response_queue', 'quit')
    
    def __init__(self, ai_assistant: AIAssistant, alarm_player, 
                 snapshot_fn: Callable, camera_manager, response_queue):
        self.ai = ai_assistant
        self.alarm = alarm_player
        self.snapshot_fn = snapshot_fn
        self.camera_mgr = camera_manager
        self.response_queue = response_queue
        self.quit = False
        self.app = None
        
        if not TELEGRAM_AVAILABLE:
            print("⚠️ Telegram not available")
            return
        
        self.app = Application.builder().token(settings.telegram.token).build()
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup command handlers"""
        handlers = [
            CommandHandler("start", self.cmd_start),
            CommandHandler("status", self.cmd_status),
            CommandHandler("detect", self.cmd_detect),
            CommandHandler("alarm", self.cmd_alarm),
            CommandHandler("get_image", self.cmd_image),
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_message),
            CallbackQueryHandler(self.on_callback),
        ]
        for h in handlers:
            self.app.add_handler(h)
    
    async def cmd_start(self, update: Update, context: Any):
        await update.message.reply_text("🛡️ Guardian Bot sẵn sàng!\n/status, /detect, /get_image, /alarm")
    
    async def cmd_status(self, update: Update, context: Any):
        enabled = "🟢" if state_manager.is_detection_enabled() else "🔴"
        alerts = len(state_manager.list_alerts())
        await update.message.reply_text(f"📊 Trạng thái\nPhát hiện: {enabled}\nCảnh báo: {alerts}")
    
    async def cmd_detect(self, update: Update, context: Any):
        if context.args:
            cam_id = context.args[0]
            current = state_manager.is_detection_enabled(cam_id)
            state_manager.set_detection(not current, cam_id)
            status = "🟢" if not current else "🔴"
            await update.message.reply_text(f"Camera {cam_id}: {status}")
        else:
            status = "🟢" if state_manager.is_detection_enabled() else "🔴"
            await update.message.reply_text(f"Phát hiện: {status}\nDùng /detect <id> để bật/tắt camera")
    
    async def cmd_alarm(self, update: Update, context: Any):
        if self.alarm and hasattr(self.alarm, 'is_alarm_playing') and self.alarm.is_alarm_playing:
            self.alarm.stop()
            await update.message.reply_text("✅ Đã tắt báo động")
        else:
            if self.alarm:
                self.alarm.play()
            await update.message.reply_text("🚨 Đã bật báo động")
    
    async def cmd_image(self, update: Update, context: Any):
        source = context.args[0] if context.args else None
        await update.message.reply_text("📸 Đang chụp...")
        self.snapshot_fn(str(update.message.chat_id), source)
    
    async def on_message(self, update: Update, context: Any):
        text = update.message.text.strip()
        chat_id = str(update.message.chat_id)
        
        reply, action = await self.ai.process(chat_id, text)
        
        if action:
            await self.execute_action(action, chat_id)
        
        if reply:
            await update.message.reply_text(reply)
    
    async def execute_action(self, action: str, chat_id: str):
        try:
            code = ActionCode[action]
            if code == ActionCode.TOGGLE_ON:
                state_manager.set_detection(True)
            elif code == ActionCode.TOGGLE_OFF:
                state_manager.set_detection(False)
            elif code == ActionCode.GET_IMAGE:
                self.snapshot_fn(chat_id)
            elif code == ActionCode.ALARM_ON and self.alarm:
                self.alarm.play()
            elif code == ActionCode.ALARM_OFF and self.alarm:
                self.alarm.stop()
        except Exception:
            pass
    
    async def on_callback(self, update: Update, context: Any):
        query = update.callback_query
        await query.answer()
        
        data = query.data
        action, alert_id = data.split(":", 1)
        
        # Stop alarm
        if self.alarm:
            self.alarm.stop()
        
        # Resolve alert
        state_manager.resolve_alert(alert_id, f"user:{action}")
        
        # Update caption
        caption = query.message.caption or ""
        
        if "fire_real" in action:
            if self.alarm:
                self.alarm.play()
            caption += "\n✅ XÁC NHẬN CÓ CHÁY!"
        elif "fire_false" in action:
            caption += "\n❌ Báo động giả"
        elif "person_yes" in action:
            caption += "\n✅ Người quen"
            # Put response in queue
            self.response_queue.put({"alert_id": alert_id, "decision": "yes"})
        elif "person_no" in action:
            if self.alarm:
                self.alarm.play()
            caption += "\n❌ NGƯỜI LẠ!"
            self.response_queue.put({"alert_id": alert_id, "decision": "no"})
        
        await query.edit_message_caption(caption=caption)
    
    def schedule_alert(self, chat_id: str, image_path: str, caption: str, 
                       alert_id: str, is_fire: bool = False, silent: bool = False):
        """Schedule alert to be sent"""
        if not TELEGRAM_AVAILABLE:
            return
        
        kb = create_alert_keyboard(alert_id, is_fire)
        markup = kb.to_dict() if kb else None
        
        threading.Thread(
            target=lambda: send_photo(chat_id, image_path, caption, markup, silent),
            daemon=True
        ).start()
    
    def send_heartbeat(self):
        """Send heartbeat message"""
        if not TELEGRAM_AVAILABLE:
            return
        
        status = "🟢" if state_manager.is_detection_enabled() else "🔴"
        text = f"❤️ Guardian hoạt động\nPhát hiện: {status}"
        
        threading.Thread(
            target=self.send_text,
            args=(settings.telegram.chat_id, text),
            daemon=True
        ).start()
    
    def send_text(self, chat_id: str, text: str):
        try:
            url = f"https://api.telegram.org/bot{settings.telegram.token}/sendMessage"
            get_session().post(url, data={'chat_id': chat_id, 'text': text}, timeout=10)
        except Exception:
            pass
    
    def run(self):
        """Run bot"""
        if not TELEGRAM_AVAILABLE or not self.app:
            return
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.run_async())
    
    async def run_async(self):
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        
        while not self.quit:
            await asyncio.sleep(1)
        
        # Proper shutdown sequence: stop updater first, then stop app, then shutdown
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()
    
    def stop(self):
        self.quit = True
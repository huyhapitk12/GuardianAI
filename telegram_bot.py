# telegram_bot.py
import threading
import queue
import httpx
import os
import asyncio
import logging
from typing import Optional
 
from telegram import Update, Bot
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    filters,
    CommandHandler,
    ContextTypes,
)
 
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, HTTPX_TIMEOUT, OPENAI_API_KEY, AI_ENABLED
from state_manager import StateManager
 
# --- state & queues (exported) ---
state = StateManager()
response_queue = queue.Queue()
 
# --- logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telegram_bot")
 
# --- helper: create Bot instance for direct sends (used by other modules if needed) ---
_bot_instance: Optional[Bot] = None
def get_bot():
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = Bot(token=TELEGRAM_TOKEN)
    return _bot_instance
 
# --- AI chat helper (async) ---
# Uses OpenAI Chat Completions REST endpoint via httpx.
# You can replace body/headers to match Gemini if you prefer.
AI_MODEL = os.getenv("AI_MODEL", "gpt-3.5-turbo")
 
async def ai_chat_async(prompt: str, user_info: dict = None) -> str:
    """
    Call OpenAI chat completions (REST). If OPENAI_API_KEY not set, returns a helpful message.
    """
    if not OPENAI_API_KEY:
        # fallback: short echo/notice
        return "AI chưa được cấu hình (OPENAI_API_KEY missing)."
 
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": AI_MODEL,
        "messages": [
            {"role": "system", "content": "Bạn là Guardian Bot - trợ lý AI thân thiện, trả lời ngắn gọn bằng tiếng Việt khi có thể."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.6
    }
 
    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            r = await client.post("https://a...content-available-to-author-only...i.com/v1/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            # defensive access
            if "choices" in data and len(data["choices"]) > 0:
                msg = data["choices"][0].get("message", {}).get("content", "")
                return (msg or "").strip()
            else:
                logger.warning("AI response missing choices: %s", data)
                return "AI không trả lời được (không có choices)."
    except Exception as e:
        logger.exception("ai_chat_async exception")
        return f"Lỗi gọi AI: {e}"
 
# --- Bot command handlers ---
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        await update.message.reply_text("Guardian bot active. Gửi tin nhắn để trò chuyện hoặc trả lời cảnh báo khi có.")
 
async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    alerts = state.list_alerts()
    if update.message:
        await update.message.reply_text(f"Alerts total: {len(alerts)}")
 
# --- message listener: handles alert replies OR AI chat (fallback) ---
async def message_listener(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # normalize text
    text_raw = update.message.text if update.message and update.message.text else ""
    text = text_raw.strip()
    chat_id = str(update.message.chat_id) if update.message else None
    user = update.effective_user
    user_info = {"id": user.id, "username": user.username, "name": f"{user.first_name} {user.last_name or ''}".strip()} if user else {}
 
    # check for unresolved alert in this chat
    matched = state.latest_unresolved_for_chat(chat_id)
 
    if matched:
        # handle alert replies using same simple rule-based logic as before
        txt_lower = text.lower()
        neg = ["không", "ko", "k", "no", "not"]
        pos = ["có", "co", "yes", "đúng", "ok", "đúng rồi", "cos"]
        left_tokens = ["đã ra khỏi nhà", "đã đi", "ra khỏi", "đi rồi", "đã ra"]
 
        decision = None
        if any(tok in txt_lower for tok in left_tokens):
            decision = "left"
        elif any(tok in txt_lower for tok in pos) and not any(tok in txt_lower for tok in neg):
            decision = "yes"
        elif any(tok in txt_lower for tok in neg):
            decision = "no"
 
        # resolve and push to response queue
        state.resolve_alert(matched["id"], text)
        response_queue.put({
            "alert_id": matched["id"],
            "decision": decision,
            "raw_text": text,
            "user": user_info
        })
 
        if update.message:
            reply_text = f"Đã ghi nhận: {text} (xử lý={decision})"
            await update.message.reply_text(reply_text)
        return
 
    # --- No alert found -> handle as AI chat (if AI enabled) ---
    # Note: keep short-circuit if empty text (ignore)
    if not text:
        return
 
    # Indicate typing action (optional)
    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    except Exception:
        pass
 
    # Call AI (await)
    if OPENAI_API_KEY:
        ai_reply = await ai_chat_async(text, user_info=user_info)
    else:
        # fallback: simple echo or helpful message
        ai_reply = f"[AI chưa cấu hình] Bạn vừa gửi: {text}\nThiết lập OPENAI_API_KEY để bot trả lời thông minh hơn."
 
    # reply back
    if update.message:
        # split long reply into multiple messages if needed
        MAX_LEN = 4000
        if len(ai_reply) <= MAX_LEN:
            await update.message.reply_text(ai_reply)
        else:
            # chunk and send
            for i in range(0, len(ai_reply), MAX_LEN):
                await update.message.reply_text(ai_reply[i:i+MAX_LEN])
 
# --- utility run function (same pattern as your project) ---
def run_bot():
    """
    Start the telegram bot polling. This is blocking (so call inside a thread if needed).
    """
    asyncio.set_event_loop(asyncio.new_event_loop())
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
 
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message_listener))
 
    logger.info("[telegram_bot] start polling...")
    app.run_polling()
# telegram_bot.py
import threading
# import queue # <--- XÓA DÒNG NÀY
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
# from state_manager import StateManager # <--- XÓA DÒNG NÀY
from shared_state import state, response_queue # <--- THAY ĐỔI DÒNG NÀY
 
# --- state & queues (exported) ---
# state = StateManager() # <--- XÓA DÒNG NÀY
# response_queue = queue.Queue() # <--- XÓA DÒNG NÀY
 
# --- logging ---
# Sửa đổi logging để đảm bảo nó hoạt động nhất quán
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
 
# --- helper: create Bot instance for direct sends (used by other modules if needed) ---
_bot_instance: Optional[Bot] = None
def get_bot():
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = Bot(token=TELEGRAM_TOKEN)
    return _bot_instance
 
# --- AI chat helper (async) ---
AI_MODEL = os.getenv("AI_MODEL", "gpt-3.5-turbo")
 
async def ai_chat_async(prompt: str, user_info: dict = None) -> str:
    if not OPENAI_API_KEY:
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
            r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
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

# <--- THÊM HÀM MỚI DƯỚI ĐÂY --->
async def toggle_detection_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Bật/tắt tính năng nhận diện người."""
    if not update.message:
        return
    
    current_state = state.is_person_detection_enabled()
    new_state = not current_state
    state.set_person_detection_enabled(new_state)
    
    status_text = "🟢 BẬT" if new_state else "🔴 TẮT"
    await update.message.reply_text(f"✅ Đã cập nhật: Nhận diện người hiện đang {status_text}.")
# <--- KẾT THÚC PHẦN THÊM MỚI --->

# --- message listener: handles alert replies OR AI chat (fallback) ---
async def message_listener(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # <--- THAY ĐỔI: Thêm logging chẩn đoán ngay từ đầu --->
    logger.info("--- message_listener triggered ---")
    if not update or not update.message:
        logger.warning("message_listener: update hoặc update.message is None.")
        return

    text_raw = update.message.text if update.message.text else ""
    chat_id = str(update.message.chat_id)
    logger.info(f"Received message in chat_id: {chat_id}, text: '{text_raw}'")
    
    text = text_raw.strip()
    user = update.effective_user
    user_info = {"id": user.id, "username": user.username, "name": f"{user.first_name} {user.last_name or ''}".strip()} if user else {}
 
    # check for unresolved alert in this chat
    matched = state.latest_unresolved_for_chat(chat_id)
    
    if matched:
        logger.info(f"Found matching unresolved alert: {matched['id']} (type: {matched['type']})")
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
 
        logger.info(f"Decision based on text: '{decision}'")
        
        state.resolve_alert(matched["id"], text)
        response_queue.put({
            "alert_id": matched["id"],
            "decision": decision,
            "raw_text": text,
            "user": user_info
        })
 
        try:
            reply_text = f"✅ Đã ghi nhận: '{text}' (xử lý={decision})"
            await update.message.reply_text(reply_text)
            logger.info("Confirmation reply sent successfully.")
        except Exception as e:
            logger.error(f"Failed to send confirmation reply: {e}")
        return
 
    logger.info("No unresolved alert found for this chat. Treating as normal message/AI chat.")
    if not text:
        return
 
    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    except Exception:
        pass
 
    if AI_ENABLED:
        ai_reply = await ai_chat_async(text, user_info=user_info)
    else:
        ai_reply = f"[AI chưa cấu hình] Bạn vừa gửi: {text}\nThiết lập OPENAI_API_KEY để bot trả lời thông minh hơn."
 
    MAX_LEN = 4000
    if len(ai_reply) <= MAX_LEN:
        await update.message.reply_text(ai_reply)
    else:
        for i in range(0, len(ai_reply), MAX_LEN):
            await update.message.reply_text(ai_reply[i:i+MAX_LEN])
 
def run_bot():
    asyncio.set_event_loop(asyncio.new_event_loop())
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
 
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("detect", toggle_detection_cmd)) # <-- THÊM DÒNG NÀY
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message_listener))
 
    logger.info("Telegram bot starting polling...")
    app.run_polling()
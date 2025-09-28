# telegram_bot.py
import threading
import httpx
import os
import asyncio
import logging
import shutil
from typing import Optional
 
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, HTTPX_TIMEOUT, OPENAI_API_KEY, AI_ENABLED, FALSE_POSITIVES_DIR, AI_MODEL, AI_MAX_TOKENS, AI_TEMPERATURE
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    Application,
    MessageHandler,
    filters,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
)
 
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, HTTPX_TIMEOUT, OPENAI_API_KEY, AI_ENABLED, FALSE_POSITIVES_DIR
from shared_state import state, response_queue
 
# --- logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- BIẾN TOÀN CỤC ĐỂ LƯU INSTANCE CỦA BOT APPLICATION VÀ EVENT LOOP ---
_app_instance: Optional[Application] = None
_app_loop: Optional[asyncio.AbstractEventLoop] = None
# --- KẾT THÚC PHẦN THÊM MỚI ---
 
# --- helper: create Bot instance for direct sends (used by other modules if needed) ---
_bot_instance: Optional[Bot] = None
def get_bot():
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = Bot(token=TELEGRAM_TOKEN)
    return _bot_instance

# --- Gửi cảnh báo với nút bấm (hàm async gốc) ---
async def send_alert_with_buttons_async(chat_id: str, image_path: str, caption: str, reply_markup: InlineKeyboardMarkup):
    """Gửi ảnh cảnh báo kèm theo các nút bấm."""
    bot = get_bot()
    try:
        with open(image_path, "rb") as photo_file:
            await bot.send_photo(
                chat_id=chat_id,
                photo=photo_file,
                caption=caption,
                reply_markup=reply_markup
            )
        logger.info(f"Sent alert with buttons to {chat_id}")
    except Exception as e:
        logger.exception(f"Failed to send alert with buttons to {chat_id}")

# --- HÀM "CẦU NỐI" AN TOÀN TỪ LUỒNG KHÁC ---
def schedule_send_alert(chat_id: str, image_path: str, caption: str, reply_markup: InlineKeyboardMarkup):
    """
    Lên lịch gửi tin nhắn cảnh báo trên event loop của bot một cách an toàn từ một thread khác.
    """
    global _app_loop
    if _app_loop:
        # Sử dụng asyncio.run_coroutine_threadsafe để gửi một coroutine
        # từ một thread khác vào event loop của bot.
        # Đây là cách làm chuẩn và an toàn nhất.
        asyncio.run_coroutine_threadsafe(
            send_alert_with_buttons_async(chat_id, image_path, caption, reply_markup),
            _app_loop
        )
    else:
        logger.error("Telegram bot application loop not available. Cannot schedule message.")
# --- KẾT THÚC PHẦN THAY ĐỔI ---

# --- AI chat helper (async) ---
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
        "max_tokens": AI_MAX_TOKENS,
        "temperature": AI_TEMPERATURE
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

async def toggle_detection_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Bật/tắt tính năng nhận diện người."""
    if not update.message:
        return
    
    current_state = state.is_person_detection_enabled()
    new_state = not current_state
    state.set_person_detection_enabled(new_state)
    
    status_text = "🟢 BẬT" if new_state else "🔴 TẮT"
    await update.message.reply_text(f"✅ Đã cập nhật: Nhận diện người hiện đang {status_text}.")

# --- Xử lý nút bấm ---
async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý các sự kiện bấm nút từ Inline Keyboard."""
    query = update.callback_query
    await query.answer()

    try:
        action, alert_id = query.data.split(":", 1)
        logger.info(f"Button clicked: action='{action}', alert_id='{alert_id}'")

        original_caption = query.message.caption if query.message else ""
        new_caption = original_caption

        if action == "fire_real":
            new_caption += "\n\n✅ Đã xác nhận: CHÁY THẬT. Hãy hành động ngay!"
        elif action == "fire_false":
            new_caption += "\n\n❌ Đã xác nhận: Báo động giả."
            alert_info = state.get_alert_by_id(alert_id)
            if alert_info and alert_info.get("image_path"):
                img_path = alert_info["image_path"]
                if os.path.exists(img_path):
                    os.makedirs(FALSE_POSITIVES_DIR, exist_ok=True)
                    filename = os.path.basename(img_path)
                    dest_path = os.path.join(FALSE_POSITIVES_DIR, f"false_{filename}")
                    shutil.copy(img_path, dest_path)
                    logger.info(f"Saved false positive image to: {dest_path}")
                    new_caption += "\n(Đã lưu ảnh để cải thiện hệ thống)"
                else:
                    logger.warning(f"Image path for false positive not found: {img_path}")
            else:
                logger.warning(f"Could not find alert info or image_path for alert_id: {alert_id}")
        elif action == "fire_call":
            new_caption += "\n\n📞 Yêu cầu gọi PCCC đã được ghi nhận."

        await query.edit_message_caption(caption=new_caption, reply_markup=None)

    except Exception as e:
        logger.exception("Error in button_callback_handler")
        try:
            await query.edit_message_caption(caption=f"{query.message.caption}\n\n⚠️ Đã xảy ra lỗi khi xử lý lựa chọn của bạn.", reply_markup=None)
        except Exception:
            pass

# --- message listener ---
async def message_listener(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
 
    matched = state.latest_unresolved_for_chat(chat_id)
    
    if matched and matched['type'] in ('nguoi_quen', 'nguoi_la'):
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
    global _app_instance, _app_loop

    # Tạo và thiết lập event loop cho thread này
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Lưu lại loop để các thread khác có thể truy cập
    _app_loop = loop

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    _app_instance = app

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("detect", toggle_detection_cmd))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message_listener))
    app.add_handler(CallbackQueryHandler(button_callback_handler))
 
    logger.info("Telegram bot starting polling...")
    app.run_polling()
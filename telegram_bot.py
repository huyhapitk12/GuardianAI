# telegram_bot.py
import asyncio
import logging
import os
import re
import threading
import uuid
from typing import Optional

import cv2
import openai
from openai import AsyncOpenAI
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (Application, ApplicationBuilder,
                          CallbackQueryHandler, CommandHandler, ContextTypes,
                          MessageHandler, filters)

from alarm_player import play_alarm, stop_alarm
from config import (AI_ENABLED, AI_MAX_TOKENS, AI_MODEL, AI_SYSTEM_INSTRUCTION,
                    AI_TEMPERATURE, API_BASE, API_KEY, HTTPX_TIMEOUT,
                    TELEGRAM_TOKEN, TMP_DIR)
from shared_state import guard, response_queue, state
import shared_state
from video_recorder import send_photo

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- State & Instances ---
_app_instance: Optional[Application] = None
_app_loop: Optional[asyncio.AbstractEventLoop] = None
_bot_instance: Optional[Bot] = None
conversation_history = {}
MAX_HISTORY_TURNS = 5

def get_bot():
    """Lấy instance của bot, khởi tạo nếu cần."""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = Bot(token=TELEGRAM_TOKEN)
    return _bot_instance

# --- Asynchronous Helpers ---
async def send_alert_with_buttons_async(chat_id: str, image_path: str, caption: str, reply_markup: InlineKeyboardMarkup):
    """Gửi cảnh báo kèm nút bấm một cách bất đồng bộ."""
    bot = get_bot()
    try:
        with open(image_path, "rb") as photo_file:
            await bot.send_photo(chat_id=chat_id, photo=photo_file, caption=caption, reply_markup=reply_markup)
        logger.info(f"Đã gửi cảnh báo có nút bấm tới {chat_id}")
    except Exception as e:
        logger.exception(f"Không thể gửi cảnh báo có nút bấm tới {chat_id}: {e}")

def schedule_send_alert(chat_id: str, image_path: str, caption: str, reply_markup: InlineKeyboardMarkup):
    """Lên lịch gửi tin nhắn từ một luồng khác vào event loop của bot."""
    if _app_loop:
        asyncio.run_coroutine_threadsafe(
            send_alert_with_buttons_async(chat_id, image_path, caption, reply_markup), _app_loop
        )
    else:
        logger.error("Event loop của bot không khả dụng.")

def send_current_camera_snapshot(chat_id, camera_name=None):
    """Lấy khung hình hiện tại từ camera, lưu và gửi qua Telegram."""
    if not shared_state.active_cameras:
        logger.error("Không có camera nào đang hoạt động.")
        return

    cam_name = camera_name or list(shared_state.active_cameras.keys())[0]
    cam_obj = shared_state.active_cameras.get(cam_name)
    if not cam_obj:
        logger.error(f"Không tìm thấy camera có tên '{cam_name}'.")
        return

    try:
        ret, frame = cam_obj.read_raw()
        if not ret or frame is None:
            logger.error(f"Không thể đọc khung hình từ camera '{cam_name}'.")
            return

        img_path = os.path.join(TMP_DIR, f"snapshot_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(img_path, frame)
        caption = f"📸 Ảnh chụp nhanh từ camera: {cam_name}."
        threading.Thread(target=lambda: send_photo(TELEGRAM_TOKEN, chat_id, img_path, caption), daemon=True).start()
    except Exception as e:
        logger.exception(f"Lỗi khi gửi ảnh chụp nhanh từ '{cam_name}': {e}")

def add_system_message_to_history(chat_id: str, text: str):
    """Thêm một tin nhắn hệ thống (cảnh báo) vào lịch sử chat cho AI."""
    chat_id_str = str(chat_id)
    chat_history = conversation_history.get(chat_id_str, [])
    chat_history.append({"role": "model", "parts": [{"text": f"Thông báo hệ thống: {text}"}]})
    if len(chat_history) > MAX_HISTORY_TURNS * 2:
        chat_history = chat_history[-(MAX_HISTORY_TURNS * 2):]
    conversation_history[chat_id_str] = chat_history

# --- AI Integration ---
async def ai_chat_async(prompt: str, history: list, user_info: dict = None, system_instruction: str = AI_SYSTEM_INSTRUCTION) -> str:
    """Gửi yêu cầu đến AI và nhận phản hồi."""
    if not API_BASE: return "AI chưa được cấu hình."
    try:
        client = AsyncOpenAI(base_url=API_BASE, api_key=API_KEY, timeout=HTTPX_TIMEOUT)
        messages = [{"role": "system", "content": system_instruction}] if system_instruction else []
        for item in history:
            role = "assistant" if item.get("role") == "model" else item.get("role")
            try:
                content = item.get("parts", [{}])[0].get("text", "")
                if content and role in ["user", "assistant"]:
                    messages.append({"role": role, "content": content})
            except (IndexError, AttributeError): continue
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(model=AI_MODEL, messages=messages, max_tokens=AI_MAX_TOKENS, temperature=AI_TEMPERATURE)
        return response.choices[0].message.content.strip() if response.choices else "AI không trả lời được."
    except Exception as e:
        logger.exception("Lỗi không xác định trong hàm ai_chat_async")
        return f"Lỗi khi gọi AI: {e}"

async def ai_classify_response_async(user_response: str) -> str:
    """Sử dụng AI để phân loại câu trả lời của người dùng thành 'yes', 'no', hoặc 'chat'."""
    prompt = f"""Phân tích câu trả lời của người dùng cho câu hỏi "Bạn có nhận ra người này không?". Phân loại vào một trong ba loại và chỉ trả lời bằng một từ duy nhất: 'yes', 'no', hoặc 'chat'.
- 'yes': nếu người dùng xác nhận (ví dụ: "có", "đúng rồi", "người quen").
- 'no': nếu người dùng phủ nhận (ví dụ: "không", "không phải", "người lạ").
- 'chat': cho bất kỳ trường hợp nào khác (câu hỏi, mệnh lệnh, không liên quan).
Câu trả lời của người dùng: "{user_response}" """
    decision = await ai_chat_async(prompt.strip(), history=[], system_instruction="")
    clean_decision = decision.lower().strip()
    return clean_decision if clean_decision in ('yes', 'no', 'chat') else 'chat'

# --- Bot Handlers ---
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message: await update.message.reply_text("Guardian bot đã hoạt động.")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message: await update.message.reply_text(f"Tổng số cảnh báo: {len(state.list_alerts())}")

async def toggle_detection_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message: return
    new_state = not state.is_person_detection_enabled()
    state.set_person_detection_enabled(new_state)
    status_text = "🟢 BẬT" if new_state else "🔴 TẮT"
    await update.message.reply_text(f"✅ Đã cập nhật: Nhận diện người hiện đang {status_text}.")

async def alarm_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        play_alarm()
        await update.message.reply_text("🚨 Đã kích hoạt còi báo động!")

async def alarm_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        stop_alarm()
        await update.message.reply_text("✅ Đã tắt còi báo động.")

async def get_image_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message: return
    camera_name = context.args[0] if context.args else None
    await update.message.reply_text(f"Đang lấy ảnh từ camera '{camera_name or 'mặc định'}', vui lòng chờ...")
    send_current_camera_snapshot(update.message.chat_id, camera_name)

async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message: return
    chat_id = str(update.message.chat_id)
    if chat_id in conversation_history:
        del conversation_history[chat_id]
        await update.message.reply_text("✅ Đã xóa lịch sử trò chuyện.")
    else:
        await update.message.reply_text("🤔 Không có lịch sử nào để xóa.")

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query or not query.data: return
    await query.answer()
    try:
        action, alert_id = query.data.split(":", 1)
        logger.info(f"Nút đã được bấm: action='{action}', alert_id='{alert_id}'")
        stop_alarm()
        state.resolve_alert(alert_id, f"user_response:{action}")
        new_caption = query.message.caption if query.message else ""
        if action == "fire_real":
            play_alarm()
            new_caption += "\n\n✅ ĐÃ XÁC NHẬN CHÁY THẬT. KÍCH HOẠT CÒI BÁO ĐỘNG!"
        elif action == "fire_false":
            guard.mute("lua_chay", 120)
            new_caption += "\n\n❌ Đã xác nhận: Báo động giả. (Tạm dừng cảnh báo cháy trong 2 phút)"
        elif action == "fire_call":
            new_caption += "\n\n📞 Yêu cầu gọi PCCC đã được ghi nhận."
        await query.edit_message_caption(caption=new_caption, reply_markup=None)
    except Exception as e:
        logger.exception("Lỗi trong hàm button_callback_handler: %s", e)

async def message_listener(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update or not update.message or not update.message.text: return
    text = update.message.text.strip()
    chat_id = str(update.message.chat_id)
    user = update.effective_user
    user_info = {"id": user.id, "username": user.username, "name": f"{user.first_name} {user.last_name or ''}".strip()} if user else {}

    unresolved_alert = state.latest_unresolved_for_chat(chat_id)
    is_alert_response = False

    if unresolved_alert and unresolved_alert['type'] in ('nguoi_quen', 'nguoi_la') and AI_ENABLED:
        classification = await ai_classify_response_async(text)
        logger.info(f"AI phân loại phản hồi cho cảnh báo {unresolved_alert['id']} là: '{classification}'")
        if classification in ("yes", "no"):
            is_alert_response = True
            if classification == "no": play_alarm()
            state.resolve_alert(unresolved_alert["id"], text)
            response_queue.put({"alert_id": unresolved_alert["id"], "decision": classification, "raw_text": text, "user": user_info})
            reply_text = f"✅ AI đã ghi nhận: '{text}' (phân loại là '{classification}')."
            if classification == "no": reply_text += " Đã bật còi báo động!"
            await update.message.reply_text(reply_text)

    if not is_alert_response:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        if AI_ENABLED:
            chat_history = conversation_history.get(chat_id, [])
            ai_reply = await ai_chat_async(text, history=chat_history, user_info=user_info)

            action_match = re.search(r'\[ACTION:([^\]]+)\]', ai_reply)
            if action_match:
                action = action_match.group(1)
                ai_reply = re.sub(r'\s*\[ACTION:[^\]]+\]\s*', '', ai_reply).strip()
                if action == "TOGGLE_ON": state.set_person_detection_enabled(True)
                elif action == "TOGGLE_OFF": state.set_person_detection_enabled(False)
                elif action == "GET_IMAGE": send_current_camera_snapshot(chat_id)
                elif action == "ALARM_ON": play_alarm()
                elif action == "ALARM_OFF": stop_alarm()

            chat_history.append({"role": "user", "parts": [{"text": text}]})
            chat_history.append({"role": "model", "parts": [{"text": ai_reply}]})
            conversation_history[chat_id] = chat_history[-MAX_HISTORY_TURNS*2:]
        else:
            ai_reply = f"[AI chưa cấu hình] Bạn vừa gửi: {text}"
        if ai_reply: await update.message.reply_text(ai_reply)

# --- Bot Runner ---
def run_bot():
    """Khởi tạo và chạy bot Telegram trong một event loop riêng."""
    global _app_instance, _app_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _app_loop = loop
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    _app_instance = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("detect", toggle_detection_cmd))
    app.add_handler(CommandHandler("clear", clear_cmd))
    app.add_handler(CommandHandler("alarm_on", alarm_on_cmd))
    app.add_handler(CommandHandler("alarm_off", alarm_off_cmd))
    app.add_handler(CommandHandler("get_image", get_image_cmd))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message_listener))
    app.add_handler(CallbackQueryHandler(button_callback_handler))
    logger.info("Telegram bot bắt đầu chạy...")
    app.run_polling()
# telegram_bot.py
import threading
import os
import asyncio
import logging
import uuid
import cv2
import re
import openai
from video_recorder import send_photo
from typing import Optional
from openai import AsyncOpenAI
 
from config import TELEGRAM_TOKEN, HTTPX_TIMEOUT, API_KEY, API_BASE, AI_ENABLED, AI_MODEL, AI_MAX_TOKENS, AI_TEMPERATURE, TMP_DIR
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
 
from shared_state import state, response_queue, guard
import shared_state
from alarm_player import stop_alarm, play_alarm
 
# --- logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

conversation_history = {}
MAX_HISTORY_TURNS = 5

AI_SYSTEM_INSTRUCTION = """
Bạn là Guardian Bot - trợ lý AI an ninh thông minh, thân thiện, trả lời ngắn gọn bằng tiếng Việt.
Bạn có thể thực hiện một số hành động đặc biệt. Khi người dùng yêu cầu, hãy nhúng một trong các mã sau vào cuối câu trả lời của bạn:
- Bật nhận diện: `[ACTION:TOGGLE_ON]`
- Tắt nhận diện: `[ACTION:TOGGLE_OFF]`
- Gửi ảnh camera: `[ACTION:GET_IMAGE]`
- Bật còi báo động: `[ACTION:ALARM_ON]`
- Tắt còi báo động: `[ACTION:ALARM_OFF]`

Ví dụ:
- User: "bật hệ thống lên" -> AI Reply: "Đã bật hệ thống nhận diện. [ACTION:TOGGLE_ON]"
- User: "tắt cảnh báo đi" -> AI Reply: "Ok, tôi đã tạm tắt cảnh báo. [ACTION:TOGGLE_OFF]"
- User: "cho xem camera" -> AI Reply: "Đây là hình ảnh từ camera. [ACTION:GET_IMAGE]"
- User: "bật báo động ngay" -> AI Reply: "Đã kích hoạt còi báo động! [ACTION:ALARM_ON]"
- User: "tắt chuông đi" -> AI Reply: "Đã tắt còi báo động. [ACTION:ALARM_OFF]"
Nếu không phải lệnh, chỉ cần trả lời bình thường.
""".strip()

_app_instance: Optional[Application] = None
_app_loop: Optional[asyncio.AbstractEventLoop] = None
 
_bot_instance: Optional[Bot] = None
def get_bot():
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = Bot(token=TELEGRAM_TOKEN)
    return _bot_instance

async def send_alert_with_buttons_async(chat_id: str, image_path: str, caption: str, reply_markup: InlineKeyboardMarkup):
    bot = get_bot()
    try:
        with open(image_path, "rb") as photo_file:
            await bot.send_photo(
                chat_id=chat_id,
                photo=photo_file,
                caption=caption,
                reply_markup=reply_markup
            )
        logger.info(f"Đã gửi cảnh báo có nút bấm tới {chat_id}")
    except Exception as e:
        logger.exception(f"Không thể gửi cảnh báo có nút bấm tới {chat_id}")

def schedule_send_alert(chat_id: str, image_path: str, caption: str, reply_markup: InlineKeyboardMarkup):
    global _app_loop
    if _app_loop:
        asyncio.run_coroutine_threadsafe(
            send_alert_with_buttons_async(chat_id, image_path, caption, reply_markup),
            _app_loop
        )
    else:
        logger.error("Event loop của bot không khả dụng. Không thể lên lịch gửi tin nhắn.")

def send_current_camera_snapshot(chat_id):
    """
    Lấy khung hình hiện tại từ camera (thông qua shared_state), lưu và gửi nó qua Telegram.
    """
    cam_obj = shared_state.camera_instance
    if not cam_obj:
        logger.error("Không tìm thấy đối tượng camera trong shared_state để chụp ảnh.")
        # Có thể gửi tin nhắn báo lỗi cho người dùng nếu muốn
        return False

    try:
        ret, frame = cam_obj.read_raw()
        if not ret or frame is None:
            logger.error("Không thể đọc khung hình từ camera để gửi.")
            return False

        img_path = os.path.join(TMP_DIR, f"snapshot_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(img_path, frame)

        caption = "📸 Đây là ảnh chụp nhanh từ camera."
        threading.Thread(
            target=lambda: send_photo(TELEGRAM_TOKEN, chat_id, img_path, caption),
            daemon=True
        ).start()
        logger.info(f"Đã lên lịch gửi ảnh chụp nhanh tới chat_id {chat_id}")
        return True
    except Exception as e:
        logger.exception(f"Lỗi khi gửi ảnh chụp nhanh: {e}")
        return False

def add_system_message_to_history(chat_id: str, text: str):
    global conversation_history
    chat_id_str = str(chat_id)
    chat_history = conversation_history.get(chat_id_str, [])
    chat_history.append({"role": "model", "parts": [{"text": f"Thông báo hệ thống: {text}"}]})
    if len(chat_history) > MAX_HISTORY_TURNS * 2:
        chat_history = chat_history[-(MAX_HISTORY_TURNS * 2):]
    conversation_history[chat_id_str] = chat_history
    logger.info(f"Đã thêm cảnh báo vào lịch sử trò chuyện cho chat_id {chat_id_str}")

async def ai_chat_async(prompt: str, history: list, user_info: dict = None, system_instruction: str = AI_SYSTEM_INSTRUCTION) -> str:
    if not API_BASE:
        return "AI chưa được cấu hình (thiếu API_BASE)."

    try:
        client = AsyncOpenAI(
            base_url=API_BASE,
            api_key=API_KEY,  # Nhiều server local không cần key, có thể điền "not-needed"
            timeout=HTTPX_TIMEOUT,
        )

        # Chuyển đổi định dạng history từ Google sang OpenAI
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        
        for item in history:
            role = item.get("role")
            if role == "model":
                role = "assistant" # Chuyển 'model' thành 'assistant'
            try:
                content = item.get("parts", [{}])[0].get("text", "")
                if content and role in ["user", "assistant"]:
                    messages.append({"role": role, "content": content})
            except (IndexError, AttributeError):
                continue

        messages.append({"role": "user", "content": prompt})

        # Gọi API một cách gọn gàng
        response = await client.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            max_tokens=AI_MAX_TOKENS,
            temperature=AI_TEMPERATURE
        )

        # Lấy kết quả trả về
        if response.choices:
            return response.choices[0].message.content.strip()
        else:
            return "AI không trả lời được (không có lựa chọn nào)."

    except openai.APIConnectionError as e:
        logger.error(f"Không thể kết nối đến server AI: {e.__cause__}")
        return "Lỗi: Không thể kết nối đến server AI."
    except openai.APIStatusError as e:
        logger.error(f"Lỗi API từ server: {e.status_code} - {e.response}")
        return f"Lỗi từ server AI: {e.status_code}"
    except Exception as e:
        logger.exception("Lỗi không xác định trong hàm ai_chat_async")
        return f"Lỗi không xác định khi gọi AI: {e}"

async def ai_confirm_stranger_async(user_response: str) -> str:
    """Sử dụng AI để phân loại câu trả lời của người dùng thành 'yes', 'no', hoặc 'unknown'."""
    prompt = f"""
    Phân tích câu trả lời của người dùng cho câu hỏi "Bạn có nhận ra người này không?".
    Chỉ trả lời bằng một trong ba từ sau: 'yes', 'no', hoặc 'unknown'.

    Ví dụ:
    - "có, người quen đó" -> yes
    - "không phải" -> no
    - "tôi không biết" -> no
    - "đúng rồi" -> yes
    - "chắc là không" -> no
    - "hôm nay trời đẹp quá" -> unknown

    Câu trả lời của người dùng: "{user_response}"
    """
    decision = await ai_chat_async(prompt.strip(), history=[], system_instruction="")
    return decision.lower().strip()

# --- Bot command handlers (không đổi) ---
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
    """Xử lý lệnh /alarm_on để bật còi báo động."""
    if not update.message: return
    play_alarm()
    logger.info(f"Còi báo động được bật thủ công bởi người dùng {update.effective_user.name}")
    await update.message.reply_text("🚨 Đã kích hoạt còi báo động!")

async def alarm_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý lệnh /alarm_off để tắt còi báo động."""
    if not update.message: return
    stop_alarm()
    logger.info(f"Còi báo động được tắt thủ công bởi người dùng {update.effective_user.name}")
    await update.message.reply_text("✅ Đã tắt còi báo động.")

async def get_image_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý lệnh /get_image để yêu cầu ảnh chụp nhanh."""
    if not update.message: return
    await update.message.reply_text("Đang lấy ảnh từ camera, vui lòng chờ...")
    send_current_camera_snapshot(update.message.chat_id)

# --- Xử lý nút bấm (không đổi) ---
async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    try:
        action, alert_id = query.data.split(":", 1)
        logger.info(f"Nút đã được bấm: action='{action}', alert_id='{alert_id}'")
        stop_alarm()
        state.resolve_alert(alert_id, f"user_response:{action}")
        original_caption = query.message.caption if query.message else ""
        new_caption = original_caption
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
        logger.exception("Lỗi trong hàm button_callback_handler")

# --- message listener (thay đổi lớn) ---
async def message_listener(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update or not update.message: return
    text_raw = update.message.text or ""
    chat_id = str(update.message.chat_id)
    logger.info(f"Đã nhận tin nhắn trong chat_id: {chat_id}, nội dung: '{text_raw}'")
    text = text_raw.strip()
    user = update.effective_user
    user_info = {"id": user.id, "username": user.username, "name": f"{user.first_name} {user.last_name or ''}".strip()} if user else {}
 
    matched = state.latest_unresolved_for_chat(chat_id)
    
    if matched and matched['type'] in ('nguoi_quen', 'nguoi_la') and AI_ENABLED:
        logger.info(f"Dùng AI để phân tích phản hồi cho cảnh báo {matched['id']}")
        decision = await ai_confirm_stranger_async(text)
        logger.info(f"AI đưa ra quyết định: '{decision}'")
        
        if decision in ("yes", "no"):
            if decision == "no":
                play_alarm()
                logger.warning(f"AI xác nhận người lạ, KÍCH HOẠT CÒI BÁO ĐỘNG!")
            
            state.resolve_alert(matched["id"], text)
            response_queue.put({"alert_id": matched["id"], "decision": decision, "raw_text": text, "user": user_info})
            
            reply_text = f"✅ AI đã ghi nhận: '{text}' (phân loại là '{decision}')."
            if decision == "no":
                reply_text += " Đã bật còi báo động!"
            await update.message.reply_text(reply_text)
        else: 
            await update.message.reply_text("🤔 AI không chắc về câu trả lời của bạn. Vui lòng trả lời rõ hơn là 'có' hoặc 'không'.")
        return
 
    if not text: return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
 
    if AI_ENABLED:
        chat_history = conversation_history.get(chat_id, [])
        ai_reply = await ai_chat_async(text, history=chat_history, user_info=user_info)
        
        action_match = re.search(r'\[ACTION:([^\]]+)\]', ai_reply)
        if action_match:
            action = action_match.group(1)
            logger.info(f"AI đã xác định một hành động: {action}")
            ai_reply = re.sub(r'\s*\[ACTION:[^\]]+\]\s*', '', ai_reply).strip()

            if action == "TOGGLE_ON":
                state.set_person_detection_enabled(True)
            elif action == "TOGGLE_OFF":
                state.set_person_detection_enabled(False)
            elif action == "GET_IMAGE":
                send_current_camera_snapshot(chat_id)
            elif action == "ALARM_ON":
                play_alarm()
            elif action == "ALARM_OFF":
                stop_alarm()

        chat_history.append({"role": "user", "parts": [{"text": text}]})
        chat_history.append({"role": "model", "parts": [{"text": ai_reply}]})
        if len(chat_history) > MAX_HISTORY_TURNS * 2:
            chat_history = chat_history[-(MAX_HISTORY_TURNS * 2):]
        conversation_history[chat_id] = chat_history
    else:
        ai_reply = f"[AI chưa cấu hình] Bạn vừa gửi: {text}"
 
    if ai_reply:
        await update.message.reply_text(ai_reply)

async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message: return
    chat_id = str(update.message.chat_id)
    if chat_id in conversation_history:
        del conversation_history[chat_id]
        await update.message.reply_text("✅ Đã xóa lịch sử trò chuyện.")
    else:
        await update.message.reply_text("🤔 Không có lịch sử nào để xóa.")
 
def run_bot():
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
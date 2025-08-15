# telegram_bot.py
import threading, queue, httpx
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, CommandHandler, ContextTypes
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, HTTPX_TIMEOUT
from state_manager import StateManager
import os, asyncio

state = StateManager()
response_queue = queue.Queue()

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        await update.message.reply_text("Guardian bot active.")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    alerts = state.list_alerts()
    if update.message:
        await update.message.reply_text(f"Alerts total: {len(alerts)}")

async def message_listener(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text.strip().lower() if update.message and update.message.text else "")
    chat_id = str(update.message.chat_id) if update.message else None
    user = update.effective_user
    matched = state.latest_unresolved_for_chat(chat_id)
    if not matched:
        # no open alert -> ignore or you can forward to AI chat
        return
    # simple rules
    neg = ["không","ko","k","no","not"]
    pos = ["có","co","yes","đúng","ok","đúng rồi"]
    left_tokens = ["đã ra khỏi nhà","đã đi","ra khỏi","đi rồi","đã ra"]
    decision = None
    if any(tok in text for tok in left_tokens):
        decision = "left"
    elif any(tok in text for tok in pos) and not any(tok in text for tok in neg):
        decision = "yes"
    elif any(tok in text for tok in neg):
        decision = "no"
    # resolve and push to response queue
    state.resolve_alert(matched["id"], text)
    response_queue.put({"alert_id": matched["id"], "decision": decision, "raw_text": text, "user": user.to_dict() if user else {}})
    if update.message:
        await update.message.reply_text(f"Đã ghi nhận: {text} (xử lý={decision})")

def run_bot():
    asyncio.set_event_loop(asyncio.new_event_loop())
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message_listener))
    print("[telegram_bot] start polling...")
    app.run_polling()

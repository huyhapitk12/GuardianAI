"""Telegram bot main class and utilities"""
import logging
import asyncio
from typing import Optional, Callable
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)

from config.settings import settings
from config.constants import ActionCode, USER_RESPONSE_WINDOW_SECONDS

logger = logging.getLogger(__name__)

def create_fire_alert_keyboard(alert_id: str) -> InlineKeyboardMarkup:
    """Create inline keyboard for fire alerts"""
    keyboard = [
        [
            InlineKeyboardButton("✅ Cháy thật", callback_data=f"fire_real:{alert_id}"),
            InlineKeyboardButton("❌ Báo động giả", callback_data=f"fire_false:{alert_id}"),
        ],
        [InlineKeyboardButton("📞 Gọi PCCC (114)", callback_data=f"fire_call:{alert_id}")],
    ]
    return InlineKeyboardMarkup(keyboard)

class GuardianBot:
    """Main Telegram bot class"""
    
    def __init__(
        self,
        state_manager,
        ai_assistant,
        spam_guard,
        alarm_player,
        response_queue,
        camera_snapshot_func: Callable
    ):
        self.state = state_manager
        self.ai = ai_assistant
        self.spam_guard = spam_guard
        self.alarm_player = alarm_player
        self.response_queue = response_queue
        self.get_snapshot = camera_snapshot_func
        
        # Import handlers here to avoid circular imports
        from .handlers import TelegramHandlers
        
        # Create handlers instance
        self.handlers = TelegramHandlers(
            state_manager=state_manager,
            ai_assistant=ai_assistant,
            spam_guard=spam_guard,
            alarm_player=alarm_player,
            camera_snapshot_func=camera_snapshot_func
        )
        self.handlers.response_queue = response_queue
        
        # Create application
        self.app = Application.builder().token(settings.telegram.token).build()
        self._setup_handlers()
        
        logger.info("GuardianBot initialized")
    
    def _setup_handlers(self):
        """Setup all command and message handlers"""
        # Command handlers
        self.app.add_handler(CommandHandler("start", self.handlers.start_cmd))
        self.app.add_handler(CommandHandler("status", self.handlers.status_cmd))
        self.app.add_handler(CommandHandler("detect", self.handlers.toggle_detection_cmd))
        self.app.add_handler(CommandHandler("alarm_on", self.handlers.alarm_on_cmd))
        self.app.add_handler(CommandHandler("alarm_off", self.handlers.alarm_off_cmd))
        self.app.add_handler(CommandHandler("get_image", self.handlers.get_image_cmd))
        self.app.add_handler(CommandHandler("clear", self.handlers.clear_cmd))
        
        # Message handler (for AI chat)
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handlers.message_listener)
        )
        
        # Callback query handler (for inline buttons)
        self.app.add_handler(CallbackQueryHandler(self.handlers.button_callback_handler))
        
        logger.info("Bot handlers registered")
    
    def schedule_alert(self, chat_id: str, image_path: str, caption: str, alert_id: str):
        """Schedule a fire alert to be sent with inline keyboard"""
        async def send_alert():
            try:
                keyboard = create_fire_alert_keyboard(alert_id)
                with open(image_path, 'rb') as photo:
                    await self.app.bot.send_photo(
                        chat_id=chat_id,
                        photo=photo,
                        caption=caption,
                        reply_markup=keyboard
                    )
                logger.info(f"Fire alert sent for alert_id: {alert_id}")
            except Exception as e:
                logger.error(f"Failed to send fire alert: {e}")
        
        # Schedule the coroutine
        asyncio.create_task(send_alert())
    
    def run(self):
        """Run the bot (blocking)"""
        logger.info("Starting Telegram bot...")
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)

class TelegramHandlers:
    """Handlers for Telegram bot commands and callbacks"""
    
    def __init__(
        self,
        state_manager,
        ai_assistant,
        spam_guard,
        alarm_player,
        camera_snapshot_func
    ):
        self.state = state_manager
        self.ai = ai_assistant
        self.spam_guard = spam_guard
        self.alarm_player = alarm_player
        self.get_snapshot = camera_snapshot_func
        self.response_queue = None  # Will be set externally
    
    # ==================== COMMAND HANDLERS ====================
    
    async def start_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        if update.message:
            await update.message.reply_text(
                "🛡️ Guardian bot đã hoạt động.\n\n"
                "Sử dụng /status để xem trạng thái hệ thống."
            )
    
    async def status_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        if not update.message:
            return
        
        alerts = self.state.list_alerts()
        detection_status = "🟢 BẬT" if self.state.is_person_detection_enabled() else "🔴 TẮT"
        unresolved_count = sum(1 for a in alerts if not a.resolved)
        
        status_text = (
            f"📊 *Trạng thái hệ thống Guardian*\n\n"
            f"🔍 Nhận diện người: {detection_status}\n"
            f"📋 Tổng số cảnh báo: {len(alerts)}\n"
            f"⚠️ Cảnh báo chưa xử lý: {unresolved_count}\n\n"
            f"Sử dụng /detect để bật/tắt nhận diện"
        )
        
        await update.message.reply_text(status_text, parse_mode='Markdown')
    
    async def toggle_detection_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /detect command to toggle person detection"""
        if not update.message:
            return
        
        new_state = not self.state.is_person_detection_enabled()
        self.state.set_person_detection_enabled(new_state)
        
        status_text = "🟢 BẬT" if new_state else "🔴 TẮT"
        await update.message.reply_text(
            f"✅ Đã cập nhật: Nhận diện người hiện đang *{status_text}*.",
            parse_mode='Markdown'
        )
        
        logger.info(f"Person detection toggled to: {new_state}")
    
    async def alarm_on_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /alarm_on command"""
        if not update.message:
            return
        
        try:
            self.alarm_player.play()
            user_name = update.effective_user.name if update.effective_user else "Unknown"
            logger.info(f"Alarm activated manually by {user_name}")
            await update.message.reply_text("🚨 Đã kích hoạt còi báo động!")
        except Exception as e:
            logger.error(f"Failed to activate alarm: {e}")
            await update.message.reply_text("❌ Lỗi khi kích hoạt còi báo động.")
    
    async def alarm_off_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /alarm_off command"""
        if not update.message:
            return
        
        try:
            self.alarm_player.stop()
            user_name = update.effective_user.name if update.effective_user else "Unknown"
            logger.info(f"Alarm stopped manually by {user_name}")
            await update.message.reply_text("✅ Đã tắt còi báo động.")
        except Exception as e:
            logger.error(f"Failed to stop alarm: {e}")
            await update.message.reply_text("❌ Lỗi khi tắt còi báo động.")
    
    async def get_image_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /get_image command"""
        if not update.message:
            return
        
        await update.message.reply_text("📸 Đang lấy ảnh từ camera, vui lòng chờ...")
        
        try:
            self.get_snapshot(str(update.message.chat_id))
        except Exception as e:
            logger.error(f"Failed to get snapshot: {e}")
            await update.message.reply_text("❌ Không thể lấy ảnh từ camera.")
    
    async def clear_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command to clear AI conversation history"""
        if not update.message:
            return
        
        chat_id = str(update.message.chat_id)
        self.ai.clear_history(chat_id)
        await update.message.reply_text("✅ Đã xóa lịch sử trò chuyện với AI.")
        logger.info(f"Conversation history cleared for chat {chat_id}")
    
    # ==================== MESSAGE HANDLER ====================
    
    async def message_listener(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages"""
        if not update or not update.message:
            return
        
        text = (update.message.text or "").strip()
        chat_id = str(update.message.chat_id)
        user = update.effective_user
        
        logger.info(f"Message received from chat {chat_id}: '{text}'")
        
        # Check for pending alerts requiring user response
        matched = self.state.get_latest_unresolved_for_chat(chat_id)
        
        if matched and matched.type in ('nguoi_quen', 'nguoi_la'):
            await self._handle_alert_response(update, matched, text, user)
            return
        
        if not text:
            return
        
        # Show typing indicator
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )
        
        # Process with AI
        user_info = self._extract_user_info(user)
        reply_text, action_code = await self.ai.process_message(chat_id, text, user_info)
        
        # Execute action if present
        if action_code:
            await self._execute_action(action_code, chat_id)
        
        # Send reply
        if reply_text:
            await update.message.reply_text(reply_text)
    
    async def _handle_alert_response(self, update, alert, text, user):
        """Handle user response to person alert"""
        logger.info(f"Processing alert response: '{text}' for alert {alert.id}")
        
        decision = await self.ai.classify_stranger_response(text)
        logger.info(f"AI classification decision: '{decision}'")
        
        if decision in ("yes", "no"):
            # Resolve alert
            self.state.resolve_alert(alert.id, text)
            
            # Activate alarm for stranger
            if decision == "no":
                self.alarm_player.play()
                logger.warning("AI confirmed stranger, ALARM ACTIVATED!")
            
            # Put response in queue
            if self.response_queue:
                self.response_queue.put({
                    "alert_id": alert.id,
                    "decision": decision,
                    "raw_text": text,
                    "user": self._extract_user_info(user)
                })
            
            reply = f"✅ AI đã ghi nhận: '{text}' (phân loại là '{decision}')."
            if decision == "no":
                reply += " 🚨 Đã bật còi báo động!"
            
            await update.message.reply_text(reply)
        else:
            await update.message.reply_text(
                "🤔 AI không chắc về câu trả lời của bạn. "
                "Vui lòng trả lời rõ hơn là 'có' hoặc 'không'."
            )
    
    async def _execute_action(self, action_code: str, chat_id: str):
        """Execute system action based on action code"""
        try:
            action = ActionCode[action_code]
            
            if action == ActionCode.TOGGLE_ON:
                self.state.set_person_detection_enabled(True)
                logger.info("Person detection enabled via AI command")
                
            elif action == ActionCode.TOGGLE_OFF:
                self.state.set_person_detection_enabled(False)
                logger.info("Person detection disabled via AI command")
                
            elif action == ActionCode.GET_IMAGE:
                self.get_snapshot(chat_id)
                logger.info(f"Camera snapshot requested via AI command for chat {chat_id}")
                
            elif action == ActionCode.ALARM_ON:
                self.alarm_player.play()
                logger.info("Alarm activated via AI command")
                
            elif action == ActionCode.ALARM_OFF:
                self.alarm_player.stop()
                logger.info("Alarm stopped via AI command")
            
        except KeyError:
            logger.warning(f"Unknown action code: {action_code}")
        except Exception as e:
            logger.error(f"Error executing action {action_code}: {e}")
    
    # ==================== CALLBACK QUERY HANDLER ====================
    
    async def button_callback_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button presses (for fire alerts)"""
        query = update.callback_query
        await query.answer()
        
        try:
            action, alert_id = query.data.split(":", 1)
            logger.info(f"Button pressed: action='{action}', alert_id='{alert_id}'")
            
            # Stop alarm first
            self.alarm_player.stop()
            
            # Resolve alert
            self.state.resolve_alert(alert_id, f"user_response:{action}")
            
            # Get original caption
            original_caption = query.message.caption if query.message else ""
            new_caption = original_caption
            
            # Handle different actions
            if action == "fire_real":
                self.alarm_player.play()
                new_caption += "\n\n✅ ĐÃ XÁC NHẬN CHÁY THẬT. KÍCH HOẠT CÒI BÁO ĐỘNG!"
                logger.warning(f"Fire confirmed as REAL for alert {alert_id}")
                
            elif action == "fire_false":
                # Mute fire alerts for 2 minutes
                self.spam_guard.mute("lua_chay", 120)
                new_caption += "\n\n❌ Đã xác nhận: Báo động giả. (Tạm dừng cảnh báo cháy trong 2 phút)"
                logger.info(f"Fire confirmed as FALSE for alert {alert_id}")
                
            elif action == "fire_call":
                new_caption += "\n\n📞 Yêu cầu gọi PCCC đã được ghi nhận."
                logger.info(f"Fire department call requested for alert {alert_id}")
            
            # Update message
            await query.edit_message_caption(caption=new_caption, reply_markup=None)
            
        except ValueError as e:
            logger.error(f"Invalid callback data format: {query.data}")
        except Exception as e:
            logger.error(f"Error in button callback handler: {e}")
    
    # ==================== HELPER METHODS ====================
    
    @staticmethod
    def _extract_user_info(user) -> dict:
        """Extract user information from Telegram user object"""
        if not user:
            return {}
        
        return {
            "id": user.id,
            "username": user.username,
            "name": f"{user.first_name} {user.last_name or ''}".strip()
        }
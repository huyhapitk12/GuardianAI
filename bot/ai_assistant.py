"""AI assistant for natural language command processing"""
import re
import logging
from typing import Optional, Tuple, List, Dict
from openai import AsyncOpenAI

from config.settings import settings
from config.constants import (
    ActionCode,
    MAX_HISTORY_TURNS,
    AI_MAX_TOKENS,
    AI_TEMPERATURE,
    HTTPX_TIMEOUT
)

logger = logging.getLogger(__name__)

AI_SYSTEM_INSTRUCTION = """
You are Guardian Bot - an intelligent, friendly security assistant that responds concisely in Vietnamese.
You can perform special actions. When users request them, embed one of these codes at the end of your response:
- Turn ON detection: [ACTION:TOGGLE_ON]
- Turn OFF detection: [ACTION:TOGGLE_OFF]
- Send camera image: [ACTION:GET_IMAGE]
- Turn ON alarm: [ACTION:ALARM_ON]
- Turn OFF alarm: [ACTION:ALARM_OFF]

Examples:
- User: "bật hệ thống lên" -> AI: "Đã bật hệ thống nhận diện. [ACTION:TOGGLE_ON]"
- User: "tắt cảnh báo đi" -> AI: "Ok, tôi đã tạm tắt cảnh báo. [ACTION:TOGGLE_OFF]"
- User: "cho xem camera" -> AI: "Đây là hình ảnh từ camera. [ACTION:GET_IMAGE]"
- User: "bật báo động ngay" -> AI: "Đã kích hoạt còi báo động! [ACTION:ALARM_ON]"
- User: "tắt chuông đi" -> AI: "Đã tắt còi báo động. [ACTION:ALARM_OFF]"

If not a command, just respond normally.
""".strip()

class AIAssistant:
    """AI-powered conversational assistant"""
    
    def __init__(self):
        self.enabled = settings.ai.enabled
        self.client: Optional[AsyncOpenAI] = None
        self.conversation_history: Dict[str, List[Dict]] = {}
        
        if self.enabled:
            self.client = AsyncOpenAI(
                base_url=settings.ai.api_base,
                api_key=settings.ai.api_key,
                timeout=HTTPX_TIMEOUT
            )
    
    async def process_message(
        self,
        chat_id: str,
        message: str,
        user_info: Optional[Dict] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Process a user message
        
        Returns: (reply_text, action_code or None)
        """
        if not self.enabled:
            return f"[AI chưa cấu hình] Bạn vừa gửi: {message}", None
        
        try:
            # Get chat history
            history = self.conversation_history.get(chat_id, [])
            
            # Call AI
            ai_reply = await self._call_ai(message, history, user_info)
            
            # Extract action code if present
            action_match = re.search(r'\[ACTION:([^\]]+)\]', ai_reply)
            action_code = None
            
            if action_match:
                action_code = action_match.group(1)
                # Remove action code from reply
                ai_reply = re.sub(r'\s*\[ACTION:[^\]]+\]\s*', '', ai_reply).strip()
            
            # Update history
            history.append({"role": "user", "parts": [{"text": message}]})
            history.append({"role": "model", "parts": [{"text": ai_reply}]})
            
            # Keep only recent history
            if len(history) > MAX_HISTORY_TURNS * 2:
                history = history[-(MAX_HISTORY_TURNS * 2):]
            
            self.conversation_history[chat_id] = history
            
            return ai_reply, action_code
        except Exception as e:
            logger.error(f"AI processing error: {e}")
            return f"Lỗi khi xử lý tin nhắn: {str(e)}", None
    
    async def classify_stranger_response(self, response: str) -> str:
        """
        Classify user response to stranger alert as 'yes', 'no', or 'unknown'
        """
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

        Câu trả lời của người dùng: "{response}"
        """
        
        try:
            decision = await self._call_ai(prompt.strip(), history=[])
            return decision.lower().strip()
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return "unknown"
    
    def add_system_message(self, chat_id: str, message: str):
        """Add a system message to conversation history"""
        history = self.conversation_history.get(chat_id, [])
        history.append({
            "role": "model",
            "parts": [{"text": f"Thông báo hệ thống: {message}"}]
        })
        self.conversation_history[chat_id] = history
    
    def clear_history(self, chat_id: str):
        """Clear conversation history for a chat"""
        if chat_id in self.conversation_history:
            del self.conversation_history[chat_id]
    
    async def _call_ai(
        self,
        prompt: str,
        history: List[Dict],
        user_info: Optional[Dict] = None
    ) -> str:
        """Make API call to AI model"""
        if not self.client:
            return "AI không khả dụng"
        
        try:
            # Convert history format
            messages = [{"role": "system", "content": AI_SYSTEM_INSTRUCTION}]
            
            for item in history:
                role = item.get("role")
                if role == "model":
                    role = "assistant"
                
                try:
                    content = item.get("parts", [{}])[0].get("text", "")
                    if content and role in ["user", "assistant"]:
                        messages.append({"role": role, "content": content})
                except (IndexError, AttributeError):
                    continue
            
            messages.append({"role": "user", "content": prompt})
            
            # Call API
            response = await self.client.chat.completions.create(
                model=settings.ai.model,
                messages=messages,
                max_tokens=AI_MAX_TOKENS,
                temperature=AI_TEMPERATURE
            )
            
            if response.choices:
                return response.choices[0].message.content.strip()
            return "AI không trả lời được (không có lựa chọn nào)."
        except Exception as e:
            logger.error(f"AI API error: {e}")
            return f"Lỗi khi gọi AI: {str(e)}"
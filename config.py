# config.py
import os

# ===================== Cài đặt Telegram =====================
TELEGRAM_TOKEN = "7874716410:AAFKDHbXiyeaZZzaJGyA2_Qr6r-5mxf3K-g"
TELEGRAM_CHAT_ID = "-4901296113"

# ===================== Đường dẫn & Model =====================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(DATA_DIR, "Model")
EMBEDDING_FILE = os.path.join(DATA_DIR, "known_embeddings.pkl")
NAMES_FILE = os.path.join(DATA_DIR, "known_names.pkl")

# ===================== Ghi Log =====================
LOG_CSV = os.path.join(BASE_DIR, "events_log.csv")

# ===================== Cài đặt Model Nhận diện =====================
MODEL_NAME = "buffalo_l"
RECOGNITION_THRESHOLD = 0.4

# ===================== Cài đặt YOLO =====================
YOLO_SIZE = "medium"  # Tùy chọn: "medium", "small"
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, YOLO_SIZE.capitalize(), "Fire", f"{YOLO_SIZE}_openvino_model")
YOLO_PERSON_MODEL_PATH = os.path.join(MODEL_DIR, YOLO_SIZE.capitalize(), "Person", f"{YOLO_SIZE}_openvino_model")

# ===================== Cài đặt Ghi hình =====================
RECORD_SECONDS = 10
TMP_DIR = os.path.join(BASE_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)
RECORDER_FPS = 20.0
RECORDER_FOURCC = "mp4v"
FFMPEG_TIMEOUT = 300  # 5 phút
STRANGER_CLIP_DURATION = 8 # Thời lượng clip ngắn gửi ngay khi có người lạ (giây)

# ===================== Cài đặt Phát hiện Cháy =====================
FIRE_WINDOW_SECONDS = 4
FIRE_REQUIRED = 10
FIRE_CONFIDENCE_THRESHOLD = 0.6
FIRE_YELLOW_ALERT_FRAMES = 5
FIRE_RED_ALERT_GROWTH_THRESHOLD = 1.5
FIRE_RED_ALERT_GROWTH_WINDOW = 7
FIRE_RED_ALERT_LOCKDOWN_SECONDS = 300

# ===================== Cài đặt Còi báo động =====================
ALARM_SOUND_FILE = os.path.join(DATA_DIR, "Audio", "alarm.mp3")
ALARM_FADE_IN_DURATION = 2
ALARM_START_VOLUME = 1.0
ALARM_MAX_VOLUME = 5.0

# ===================== Cài đặt Xử lý =====================
FRAMES_REQUIRED = 3
PROCESS_EVERY_N_FRAMES = 3
PROCESS_SIZE = (1280, 720)
DEBOUNCE_SECONDS = 30
TARGET_FPS = 20
YOLO_PERSON_CONFIDENCE = 0.6
FACE_RECOG_COOLDOWN = 1.0
TRACKER_TIMEOUT_SECONDS = 2.0
IOU_THRESHOLD = 0.3
STRANGER_CONFIRM_FRAMES = 25
INSIGHTFACE_DET_SIZE = (640, 640)
INSIGHTFACE_CTX_ID = -1

# ===================== Cài đặt Tải lên Telegram =====================
VIDEO_PREVIEW_LIMIT_MB = 48.0
HTTPX_TIMEOUT = 180
USER_RESPONSE_WINDOW_SECONDS = 30

# ===================== Cài đặt AI (Tùy chọn) =====================
API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
API_KEY = "AIzaSyBvBkXirUSiTAqXNykZjfoHWwdPqZDZYnA"
AI_ENABLED = bool(API_KEY)
AI_MODEL = "gemini-2.5-flash"
AI_MAX_TOKENS = 512
AI_TEMPERATURE = 0.5
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

# ===================== Cài đặt Chống Spam =====================
SPAM_GUARD_MIN_INTERVAL = 10
SPAM_GUARD_MAX_PER_MINUTE = 4

# ===================== Cài đặt Camera IP =====================
IP_CAMERAS = {
    "Phong_Khach": 0,
    "Cua_Truoc": "test_fire.mp4",
}
# config.py
import os

# ===================== Telegram Settings =====================
TELEGRAM_TOKEN = "7874716410:AAFKDHbXiyeaZZzaJGyA2_Qr6r-5mxf3K-g"
TELEGRAM_CHAT_ID = "-4901296113"  # group chat id

# ===================== Paths & Models =====================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "Data", "Image")
MODEL_DIR = os.path.join(BASE_DIR, "Data", "Model")
EMBEDDING_FILE = os.path.join(BASE_DIR, "Data", "known_embeddings.pkl")
NAMES_FILE = os.path.join(BASE_DIR, "Data", "known_names.pkl")

# ===================== Logging =====================
LOG_CSV = os.path.join(BASE_DIR, "events_log.csv")

# ===================== Face Model Settings =====================
MODEL_NAME = "buffalo_l"
RECOGNITION_THRESHOLD = 0.4

# ===================== YOLO Settings =====================
YOLO_SIZE = "medium"  # Options: "medium", "small". This is the default value at startup.

# Automatically build paths based on YOLO_SIZE
# Used to load fire/smoke model
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, YOLO_SIZE.capitalize(), "Fire", f"{YOLO_SIZE}_openvino_model")
# Used to load person detection model (if available)
YOLO_PERSON_MODEL_PATH = os.path.join(MODEL_DIR, YOLO_SIZE.capitalize(), "Person", f"{YOLO_SIZE}_openvino_model")

# ===================== Recording Settings =====================
RECORD_SECONDS = 10
TMP_DIR = os.path.join(BASE_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)
FALSE_POSITIVES_DIR = os.path.join(BASE_DIR, "false_positives")
os.makedirs(FALSE_POSITIVES_DIR, exist_ok=True)
RECORDER_FPS = 20.0
RECORDER_FOURCC = "mp4v"
FFMPEG_TIMEOUT = 300  # 5 phút
STRANGER_CLIP_DURATION = 8 # Thời lượng clip ngắn gửi ngay khi có người lạ (giây)

# ===================== Fire Detection Settings =====================
FIRE_WINDOW_SECONDS = 4
FIRE_REQUIRED = 10 # Giờ đây được dùng như một ngưỡng tối thiểu trong logic mới
FIRE_CONFIDENCE_THRESHOLD = 0.7
FIRE_YELLOW_ALERT_FRAMES = 5        # Số frame liên tục có tín hiệu để kích hoạt cảnh báo Vàng
FIRE_RED_ALERT_GROWTH_THRESHOLD = 1.5 # Ngưỡng tăng trưởng diện tích (50%) để kích hoạt cảnh báo Đỏ
FIRE_RED_ALERT_GROWTH_WINDOW = 1.5  # Khoảng thời gian (giây) để so sánh sự tăng trưởng
FIRE_RED_ALERT_LOCKDOWN_SECONDS = 300 # Thời gian "khóa" cảnh báo Đỏ sau khi kích hoạt (5 phút)

# ===================== Processing Settings =====================
FRAMES_REQUIRED = 3            # number of consecutive frames required to confirm a face
PROCESS_EVERY_N_FRAMES = 3
PROCESS_SIZE = (1280, 720)
DEBOUNCE_SECONDS = 30         # debounce time for the same alert (by type/name)
TARGET_FPS = 20
YOLO_PERSON_CONFIDENCE = 0.6        # Ngưỡng tin cậy cho phát hiện người
FACE_RECOG_COOLDOWN = 1.0           # Thời gian chờ (giây) trước khi nhận diện lại 1 khuôn mặt
TRACKER_TIMEOUT_SECONDS = 2.0       # Thời gian (giây) để xóa tracker nếu không thấy đối tượng
IOU_THRESHOLD = 0.3                 # Ngưỡng IoU để khớp tracker
STRANGER_CONFIRM_FRAMES = 25        # Số frame không nhận diện được trước khi báo người lạ
INSIGHTFACE_DET_SIZE = (640, 640)   # Kích thước ảnh đầu vào cho model mặt
INSIGHTFACE_CTX_ID = -1             # ID của GPU (-1 cho CPU, 0 cho GPU đầu tiên)

# ===================== Telegram Upload Settings =====================
VIDEO_PREVIEW_LIMIT_MB = 48.0
HTTPX_TIMEOUT = 180
USER_RESPONSE_WINDOW_SECONDS = 60   # Thời gian chờ người dùng phản hồi cảnh báo (giây)

# ===================== AI (Optional) Settings =====================
OPENAI_API_KEY = None  # set if you want AI classification
AI_ENABLED = False
if OPENAI_API_KEY:
    AI_ENABLED = True
AI_MODEL = os.getenv("AI_MODEL", "gpt-3.5-turbo")
AI_MAX_TOKENS = 512
AI_TEMPERATURE = 0.6

# ===================== Spam Guard Settings =====================
SPAM_GUARD_MIN_INTERVAL = 10        # Khoảng cách tối thiểu giữa 2 cảnh báo bất kỳ
SPAM_GUARD_MAX_PER_MINUTE = 4       # Số cảnh báo tối đa trong 1 phút

# ===================== IP Camera Settings =====================
IP_CAMERA_URL = 0#   rtsp://admin:XGZBPX@192.168.1.6:554/h264/ch1/sub_stream
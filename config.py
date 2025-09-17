# config.py
import os

# Telegram
TELEGRAM_TOKEN = "7874716410:AAFKDHbXiyeaZZzaJGyA2_Qr6r-5mxf3K-g"
TELEGRAM_CHAT_ID = "-4901296113"  # group chat id

# Paths & models
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "Data", "Image")
MODEL_DIR = os.path.join(BASE_DIR, "Data", "Model")
EMBEDDING_FILE = os.path.join(BASE_DIR, "Data", "known_embeddings.pkl")
NAMES_FILE = os.path.join(BASE_DIR, "Data", "known_names.pkl")

# Face model
MODEL_NAME = "buffalo_l"
RECOGNITION_THRESHOLD = 0.4

# <--- THAY ĐỔI BẮT ĐẦU TỪ ĐÂY --->
# YOLO settings
YOLO_SIZE = "medium"  # Tùy chọn: "medium", "small". Đây là giá trị mặc định khi khởi động.

# Tự động xây dựng đường dẫn dựa trên YOLO_SIZE
# Đường dẫn này sẽ được dùng để tải model lửa/khói
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, YOLO_SIZE.capitalize(), "Fire", f"{YOLO_SIZE}_openvino_model")
# Đường dẫn này sẽ được dùng để tải model người (nếu có)
YOLO_PERSON_MODEL_PATH = os.path.join(MODEL_DIR, YOLO_SIZE.capitalize(), "Person", f"{YOLO_SIZE}_openvino_model")
# <--- KẾT THÚC THAY ĐỔI --->


# Recording
RECORD_SECONDS = 10   # 100s per your last request
TMP_DIR = os.path.join(BASE_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# Fire detection
FIRE_WINDOW_SECONDS = 30
FIRE_REQUIRED = 20
FIRE_CONFIDENCE_THRESHOLD = 0.65  # ngưỡng confidence cho phát hiện lửa/khói

# Processing
FRAMES_REQUIRED = 7            # số frame liên tiếp để xác nhận một face
PROCESS_EVERY_N_FRAMES = 3
PROCESS_SIZE = (1280, 720)
DEBOUNCE_SECONDS = 30         # thời gian debounce cho cùng 1 alert (theo type/name)

# Telegram upload
VIDEO_PREVIEW_LIMIT_MB = 48.0
HTTPX_TIMEOUT = 180

# AI (optional - OpenAI)
OPENAI_API_KEY = None  # set if you want AI classification
AI_ENABLED = False
if OPENAI_API_KEY:
    AI_ENABLED = True

# Logging
LOG_CSV = os.path.join(BASE_DIR, "events_log.csv")

# IP camera
IP_CAMERA_URL = 0# "rtsp://admin:XGZBPX@192.168.1.6:554/h264/ch1/sub_stream"
# rtsp://admin:XGZBPX@192.168.0.104:554/h264/ch1/sub_stream
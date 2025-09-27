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

# ===================== Fire Detection Settings =====================
FIRE_WINDOW_SECONDS = 4
FIRE_REQUIRED = 10
FIRE_CONFIDENCE_THRESHOLD = 0.3

# ===================== Processing Settings =====================
FRAMES_REQUIRED = 7            # number of consecutive frames required to confirm a face
PROCESS_EVERY_N_FRAMES = 3
PROCESS_SIZE = (1280, 720)
DEBOUNCE_SECONDS = 30         # debounce time for the same alert (by type/name)
TARGET_FPS = 20

# ===================== Telegram Upload Settings =====================
VIDEO_PREVIEW_LIMIT_MB = 48.0
HTTPX_TIMEOUT = 180

# ===================== AI (Optional) Settings =====================
OPENAI_API_KEY = None  # set if you want AI classification
AI_ENABLED = False
if OPENAI_API_KEY:
    AI_ENABLED = True

# ===================== IP Camera Settings =====================
IP_CAMERA_URL = 0  # rtsp://admin:XGZBPX@192.168.1.6:554/h264/ch1/sub_stream
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
MODEL_NAME = "buffalo_s"
RECOGNITION_THRESHOLD = 0.4

# YOLO path (your model folder)
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "model_openvino_model")

# Recording
RECORD_SECONDS = 10   # 100s per your last request
TMP_DIR = os.path.join(BASE_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# Fire detection
FIRE_WINDOW_SECONDS = 30
FIRE_REQUIRED_COUNT = 20

# Processing
PROCESS_EVERY_N_FRAMES = 3
PROCESS_SIZE = (416, 416)

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

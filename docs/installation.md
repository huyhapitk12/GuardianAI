# Cài đặt & Thiết lập GuardianAI

Hướng dẫn cài đặt chi tiết cho hệ thống giám sát GuardianAI trên Windows và Linux.

---

## Yêu cầu Hệ thống

### Phần cứng

**Tối thiểu:**
- CPU: Intel Core i5 hoặc tương đương
- RAM: 8GB
- Ổ đĩa: 10GB trống
- Camera: Webcam hoặc IP camera hỗ trợ RTSP

**Khuyến nghị:**
- CPU: Intel Core i7 hoặc AMD Ryzen 7 (OpenVINO optimization)
- RAM: 16GB
- GPU: Không bắt buộc (CPU-optimized)

### Phần mềm

- **OS**: Windows 10/11 hoặc Ubuntu 20.04+
- **Python**: 3.10, 3.11, hoặc 3.13
- **Internet**: Cần thiết cho Telegram API

---

## Cài đặt Nhanh (Quick Start)

### 1. Clone Repository

```bash
git clone https://github.com/your-repo/GuardianAI.git
cd GuardianAI
```

### 2. Tạo Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Cài đặt Dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

**Lưu ý:** Quá trình cài đặt có thể mất 5-10 phút tùy vào tốc độ mạng.

### 4. Cấu hình Cơ bản

Chỉnh sửa `config/config.yaml`:

```yaml
telegram:
  token: YOUR_BOT_TOKEN        # Từ @BotFather
  chat_id: YOUR_CHAT_ID        # Từ @userinfobot

camera:
  sources: '0'                 # 0=webcam, hoặc RTSP URL

ai:
  enabled: false               # Tắt AI assistant ban đầu
```

> [!TIP]
> Sử dụng biến môi trường cho tokens: `token: ${TELEGRAM_TOKEN:""}`

### 5. Chạy Ứng dụng

```bash
python main.py
```

**Kết quả:**
- GUI sẽ mở
- Telegram bot gửi tin "System online"
- Camera feed hiển thị

---

## Chi tiết Cài đặt

### Bước 1: Python Environment

#### Windows

1. **Tải Python 3.13** từ [python.org](https://www.python.org/)
2. **Cài đặt** với options:
   - ✅ Add Python to PATH
   - ✅ Install pip
3. **Xác nhận**:
   ```powershell
   python --version  # Python 3.13.x
   pip --version
   ```

#### Linux (Ubuntu)

```bash
# Update packages
sudo apt update
sudo apt install python3.13 python3.13-venv python3-pip

# Verify
python3 --version
```

### Bước 2: Dependencies

#### Core Dependencies (requirements.txt)

```txt
# Computer Vision
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0

# AI/ML
ultralytics>=8.0.0        # YOLO
insightface>=0.7.3        # Face recognition
onnxruntime>=1.16.0       # ONNX Runtime
openvino>=2024.0.0        # OpenVINO (Intel optimization)

# Telegram
python-telegram-bot>=20.0

# GUI
customtkinter>=5.2.0

# Utilities
PyYAML>=6.0
cryptography>=41.0.0
scipy>=1.11.0
```

#### Optional Dependencies

**Behavior Analysis:**
```bash
pip install mediapipe torch
```

**Better Tracking:**
```bash
pip install supervision
```

### Bước 3: Models

#### Auto-download

Models sẽ tự động được tải khi chạy lần đầu:
- YOLO Fire Detection model
- YOLO Person Detection model
- InsightFace models

#### Manual download

Nếu cần tải thủ công, đặt vào `Data/Model/`:

```
Data/Model/
├── fire_small_openvino/
│   ├── metadata.yaml
│   └── weights/
├── Small/                    # InsightFace detector
│   └── detect.onnx
└── Small/                    # InsightFace recognizer
    └── recog.onnx
```

---

## Cấu hình Ban đầu

### config/config.yaml

#### 1. Telegram (Bắt buộc)

```yaml
telegram:
  token: ${TELEGRAM_TOKEN:""}
  chat_id: ${TELEGRAM_CHAT_ID:""}
  httpx_timeout: 180
```

**Lấy token:**
1. Chat với [@BotFather](https://t.me/BotFather)
2. `/newbot` → nhập tên và username
3. Copy token

**Lấy chat_id:**
1. Chat với [@userinfobot](https://t.me/userinfobot)
2. Copy ID (ví dụ: `123456789`)

**Group chat:**
- Add bot vào group
- Chat_ID sẽ là số âm (ví dụ: `-4901296113`)

#### 2. Camera

```yaml
camera:
  sources: '0'          # Webcam index
  # sources: 'rtsp://user:pass@192.168.1.10:554/stream'  # IP camera
  # sources: 'video.mp4'  # Video file
  
  target_fps: 10
  process_every_n_frames: 5
  process_size: [1280, 720]
```

**Multiple cameras:**
```yaml
camera:
  sources: '0,1'  # Webcam 0 và 1
  # sources: '0,rtsp://user:pass@ip:554/stream'  # Mix
```

#### 3. Models

```yaml
models:
  yolo_format: openvino  # Tốt nhất cho CPU
  yolo_size: medium      # small|medium|large
  
  face:
    detector_name: Small   # Small|Medium|Large
    recognizer_name: Small
```

#### 4. Detection Thresholds

```yaml
detection:
  fire_confidence_threshold: 0.85
  smoke_confidence_threshold: 0.8
  person_confidence_threshold: 0.6
  face_recognition_threshold: 0.45
```

---

## Setup Biến Môi trường

### Windows (PowerShell)

**Tạm thời (session hiện tại):**
```powershell
$env:TELEGRAM_TOKEN="your_token_here"
$env:TELEGRAM_CHAT_ID="your_chat_id"
```

**Vĩnh viễn (System Variables):**
1. Win + R → `sysdm.cpl`
2. Advanced → Environment Variables
3. New → Thêm `TELEGRAM_TOKEN` và `TELEGRAM_CHAT_ID`

### Linux/macOS (Bash)

**Tạm thời:**
```bash
export TELEGRAM_TOKEN="your_token_here"
export TELEGRAM_CHAT_ID="your_chat_id"
```

**Vĩnh viễn (.bashrc hoặc .zshrc):**
```bash
echo 'export TELEGRAM_TOKEN="your_token_here"' >> ~/.bashrc
echo 'export TELEGRAM_CHAT_ID="your_chat_id"' >> ~/.bashrc
source ~/.bashrc
```

---

## Xác nhận Cài đặt

### 1. Test Dependencies

```bash
python -c "import cv2, ultralytics, insightface, telegram; print('All OK')"
```

### 2. Test Camera

```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
# Should print: True
```

### 3. Run Application

```bash
python main.py
```

**Kiểm tra:**
- ✅ GUI mở không lỗi
- ✅ Camera feed hiển thị
- ✅ Telegram nhận tin "System online"

---

## Troubleshooting

### Lỗi: ModuleNotFoundError

**Nguyên nhân:** Thiếu dependency

**Giải pháp:**
```bash
pip install -r requirements.txt --force-reinstall
```

### Lỗi: Camera không mở

**Windows:**
```powershell
# Check camera với Camera app trước
# Nếu được → camera index đúng
```

**Linux:**
```bash
# Check camera devices
ls /dev/video*

# Test với ffplay
ffplay /dev/video0
```

### Lỗi: Telegram connection failed

**Kiểm tra:**
1. Token đúng format? (dài ~45 ký tự)
2. Chat ID đúng?
3. Mạng có block telegram.org?

**Test:**
```bash
curl https://api.telegram.org/bot<YOUR_TOKEN>/getMe
```

### Lỗi: OpenVINO not found (Windows)

**Giải pháp:**
```bash
pip install openvino-dev
```

### Lỗi: InsightFace model không tải

**Kiểm tra:**
- `Data/Model/Small/detect.onnx` có tồn tại?
- `Data/Model/Small/recog.onnx` có tồn tại?

**Download thủ công:**
```python
# Chạy script này một lần
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_s')
app.prepare(ctx_id=0, det_size=(640, 640))
```

---

## Nâng cao

### GPU Support (Optional)

**CUDA (NVIDIA):**
```bash
# Install CUDA toolkit first
pip install onnxruntime-gpu
```

**Cấu hình:**
```yaml
models:
  insightface_ctx_id: 0  # 0=GPU, -1=CPU
```

### Multiple Cameras

```yaml
camera:
  sources: '0,rtsp://user:pass@camera1/stream,rtsp://user:pass@camera2/stream'
```

### Behavior Analysis

```bash
pip install mediapipe torch
```

```yaml
behavior:
  enabled: true
```

### AI Assistant

```yaml
ai:
  enabled: true
  api_base: https://generativelanguage.googleapis.com/v1beta/openai/
  api_key: YOUR_GEMINI_API_KEY
  model: gemini-2.0-flash
```

---

## Kiểm tra Sau Cài đặt

### Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment created & activated
- [ ] Dependencies installed (`requirements.txt`)
- [ ] `config/config.yaml` configured
- [ ] Telegram token & chat_id set
- [ ] Camera working
- [ ] Application runs without errors
- [ ] Telegram bot responds

### Benchmark (Optional)

```bash
python benchmark.py
```

Kết quả mong đợi:
```
Fire Detection: ~20-50 fps (CPU)
Person Detection: ~15-25 fps (CPU)
Face Recognition: ~40-50 fps (CPU)
```

---

## Xem thêm

- [configuration.md](file:///d:/GuardianAI/docs/configuration.md) - Cấu hình chi tiết
- [usage.md](file:///d:/GuardianAI/docs/usage.md) - Hướng dẫn sử dụng
- [config_guide.md](file:///d:/GuardianAI/docs/config_guide.md) - Tuning fire detection
- [troubleshooting.md](file:///d:/GuardianAI/docs/troubleshooting.md) - Khắc phục sự cố

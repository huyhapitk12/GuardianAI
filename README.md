<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-4A90E2?style=for-the-badge" alt="Platform">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/AI-Computer%20Vision-FF6F61?style=for-the-badge" alt="AI">
</p>

<h1 align="center">🛡️ GuardianAI</h1>

<p align="center">
  <strong>Intelligent Security Surveillance System using Computer Vision & AI</strong>
</p>

<p align="center">
  Real-time surveillance solution with Fire/Smoke detection, Stranger/Known person recognition, Fall detection,<br>
  Telegram alerts, GUI management and automatic event recording.
</p>

---

## 🎯 Overview

**GuardianAI** is an intelligent security surveillance system, optimized for 24/7 operation with high performance. The system uses advanced AI models to detect dangers and send instant alerts via Telegram.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔥 **Fire & Smoke Detection** | YOLO (OpenVINO/ONNX) with smart post-processing filters, minimizing False Positives |
| 👤 **Face Recognition** | InsightFace + SORT Tracker, classifying Known/Stranger persons |
| 🚨 **Fall Detection** | RTMPose + ONNX model, real-time pose analysis |
| 📸 **Infrared Camera (IR)** | Auto-detection and appropriate filter switching |
| 📱 **Telegram Alerts** | Send photos/videos, interactive confirmation buttons, periodic heartbeat |
| 🤖 **AI Assistant** | Compatible with OpenAI API (Gemini/LM Studio/Ollama) |
| 🎬 **Auto Recording** | Automatically record clips on events and send with alerts |
| 🖥️ **Management GUI** | Beautiful interface with CustomTkinter |

---

## 🚀 Installation

### System Requirements

- **OS:** Windows 10/11 or Linux
- **Python:** 3.12
- **RAM:** Minimum 4GB (recommended 8GB+)
- **Camera:** Webcam, IP Camera (RTSP), or video file

### Step 1: Clone repository

```bash
git clone https://github.com/your-username/GuardianAI.git
cd GuardianAI
```

### Step 2: Create virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/macOS
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configuration

Edit file `config/config.yaml`:

```yaml
# === Telegram Bot ===
telegram:
  token: ${TELEGRAM_TOKEN:""}
  chat_id: ${TELEGRAM_CHAT_ID:""}

# === Camera ===
camera:
  # Webcam: 0 | File: "video.mp4" | RTSP: "rtsp://user:pass@ip:554/stream"
  sources: 0
  target_fps: 10

# === AI Assistant (optional) ===
ai:
  enabled: false
  api_base: ${AI_API_BASE:"https://api.openai.com/v1"}
  api_key: ${AI_API_KEY:""}
  model: ${AI_MODEL:"gpt-4o-mini"}
```

> 💡 **Tip:** Use environment variables with syntax `${ENV:"default"}` to secure sensitive information.

### Step 5: Run the application

```bash
python main.py
```

---

## 📂 Project Structure

```
GuardianAI/
├── main.py                 # Entry point - Initialize and orchestrate system
├── config/                 
│   ├── config.yaml         # Main configuration file
│   └── settings.py         # Load config, supports ${ENV:"default"}
├── core/                   
│   ├── camera.py           # Camera class: read frames, IR detection, pipeline
│   ├── camera_manager.py   # Multi-camera management, processing threads
│   ├── recorder.py         # Alert video recording
│   └── detection/          
│       ├── fire.py         # FireDetector: YOLO + RGB/IR filters
│       ├── face.py         # FaceDetector: InsightFace
│       ├── person.py       # PersonTracker: YOLO + SORT + Face ID
│       └── fall.py         # FallDetector: RTMPose + ONNX
├── bot/                    
│   └── *.py                # Telegram Bot, AI Assistant
├── gui/                    
│   ├── app.py              # Main GUI Application
│   ├── panels/             # Panels (settings, cameras, faces...)
│   ├── widgets/            # Custom widgets
│   └── styles.py           # Theme and styling
├── utils/                  
│   └── *.py                # StateManager, SpamGuard, AlarmPlayer...
└── Data/                   
    └── Model/              # AI models (YOLO, InsightFace, ONNX...)
```

---

## 🔧 Feature Details

### 🔥 Fire & Smoke Detection

- Uses **YOLO** with OpenVINO/ONNX backend for optimal performance
- **Smart filters** to minimize False Positives:
  - Color analysis (HSV ranges)
  - Flicker detection
  - Reflection and bright light filtering
  - Texture analysis (entropy)
- **Dedicated IR mode**: Auto-switches filters when camera is in infrared mode

### 👤 Face Recognition

- **InsightFace** for high-quality face embeddings
- **SORT Tracker** for continuous multi-person tracking
- **ReID (Re-Identification)**: Remembers people when they leave and return
- **Management GUI**: Add new people, build embeddings

### 🚨 Fall Detection

- **RTMPose** (from rtmlib) for 17-point pose estimation
- **ONNX model** classifies falls based on pose sequences
- Analyzes keypoint velocity and location
- Visual skeleton overlay display

### 📸 Infrared Camera (IR) Support

- Auto-detects IR mode every 10 frames
- Maintains 30-frame history for stable results
- Dedicated IR filters (brightness, variance, hot spot)
- Smoke detection disabled by default in IR mode

---

## 📱 Telegram Bot Usage

### Available Commands

| Command | Description |
|---------|-------------|
| `/start` | Start bot, show instructions |
| `/status` | Check system status |
| `/detect` | Toggle detection mode |
| `/alarm` | Control alarm siren |
| `/get_image` | Capture image from camera |

### Alert Handling

When receiving alerts, you can interact via buttons:
- **✅ Confirm**: This is a real event
- **❌ Dismiss**: False positive
- **🔇 Mute**: Stop alarm siren

---

## ⚙️ Advanced Configuration

See `config/config.yaml` to adjust:

- **Camera**: FPS, resolution, process interval
- **Detection**: Confidence thresholds, filter parameters
- **Tracker**: IOU threshold, max age
- **Recorder**: Duration, extension time
- **Telegram**: Response timeout, spam protection

---

## 📜 License

This project is released under the **MIT** License. See [LICENSE](LICENSE) file for more details.

<p align="center">
  Made with ❤️ by <strong>HuyHAP & Minh Tri</strong>
</p>
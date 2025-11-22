# Cài đặt & Thiết lập

Hướng dẫn cài đặt GuardianAI trên Windows và Linux.

## Yêu cầu hệ thống
- OS: Windows 10/11 hoặc Ubuntu 20.04+
- Python: 3.13
- Quyền truy cập Internet (Telegram)

## Cài đặt nhanh

```bash
pip install -r requirements.txt
```

Nếu dùng OpenVINO/ONNX trên CPU, các wheel sẽ được cài tự động theo `requirements.txt`.

## Cấu hình ban đầu

Chỉnh `config/config.yaml` (hỗ trợ biến môi trường dạng `${ENV:"default"}`):

```yaml
telegram:
  token: ${TELEGRAM_TOKEN:""}
  chat_id: ${TELEGRAM_CHAT_ID:""}

camera:
  sources: 0            # 0: webcam; "file.mp4"; hoặc "rtsp://user:pass@ip:554/stream"
  target_fps: 10

ai:
  enabled: false
  api_base: ${AI_API_BASE:"https://api.openai.com/v1"}
  api_key: ${AI_API_KEY:""}
  model: ${AI_MODEL:"gpt-4o-mini"}
```

- Xem chi tiết tinh chỉnh camera/IR trong `docs/config_guide.md`.

## Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
$env:TELEGRAM_TOKEN="<bot-token>"
$env:TELEGRAM_CHAT_ID="<chat-id>"
python main.py
```

## Linux/macOS (bash)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
export TELEGRAM_TOKEN="<bot-token>"
export TELEGRAM_CHAT_ID="<chat-id>"
python main.py
```

## Tải model
- Thư mục mô hình mặc định: `Data/Model` (đã có sẵn cấu trúc và tệp mẫu)
- `config/settings.py` sẽ tự động định vị model theo `models.yolo_size`, `models.yolo_format`, `paths.model_dir`.

## Kiểm tra nhanh
- Chạy `python main.py` → GUI mở, bot gửi tin nhắn khởi động (nếu cấu hình Telegram hợp lệ).
- Dùng `/get_image` để nhận ảnh camera.

## Nâng cao
- Chỉnh `camera.process_every_n_frames`, `camera.process_size` để cân bằng chất lượng/hiệu năng.
- Bật AI trợ lý: đặt `ai.enabled: true` và cấu hình `api_base`, `api_key`, `model`.

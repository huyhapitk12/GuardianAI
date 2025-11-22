# GuardianAI — Hệ thống Giám sát An ninh Thông minh

GuardianAI là giải pháp giám sát thời gian thực sử dụng Thị giác máy tính và AI để phát hiện Cháy/Khói, Người lạ/Người quen, gửi cảnh báo qua Telegram, hỗ trợ GUI quản lý khuôn mặt và ghi lại sự kiện. Hệ thống tối ưu CPU, chạy 24/7, và hỗ trợ camera hồng ngoại (IR) với bộ lọc chuyên biệt.

## ✨ Tính năng

- **Phát hiện Cháy & Khói**: YOLO (OpenVINO/ONNX/CPU) với các bộ lọc hậu xử lý thông minh.
- **Phát hiện & Nhận diện Người**: Theo dõi, phân loại Người quen/Người lạ (InsightFace + tracker).
- **Tự động nhận biết Camera Hồng ngoại (IR)**: Chuyển bộ lọc phù hợp, mặc định tắt phát hiện khói ở IR.
- **Cảnh báo qua Telegram**: Ảnh/video, nút xác nhận (cháy thật/giả, người quen/lạ), heartbeat định kỳ.
- **Trợ lý AI (tùy chọn)**: Tương thích API kiểu OpenAI (Gemini/LM Studio/Ollama/…); điều khiển bằng ngôn ngữ tự nhiên.
- **Ghi hình sự kiện**: Tự động ghi clip ngắn cho cảnh báo và gửi kèm.
- **GUI quản lý khuôn mặt**: Thêm người, xây dựng embedding, xem camera.
- **Tối ưu hiệu năng**: Giới hạn FPS, bỏ khung hình, kích thước khung xử lý, đa luồng.

## 🚀 Cài đặt nhanh

1) Yêu cầu
- Windows 10/11 hoặc Linux; Python 3.10+

2) Cài đặt thư viện
```bash
pip install -r requirements.txt
```

3) Cấu hình cơ bản trong `config/config.yaml`
- Dùng biến môi trường theo cú pháp `${ENV:"default"}` (được hỗ trợ bởi `config/settings.py`).
```yaml
telegram:
  token: ${TELEGRAM_TOKEN:""}
  chat_id: ${TELEGRAM_CHAT_ID:""}

camera:
  # Webcam mặc định: 0; File: "video.mp4"; RTSP: "rtsp://user:pass@ip:554/stream"
  sources: 0
  target_fps: 10

ai:
  enabled: false
  api_base: ${AI_API_BASE:"https://api.openai.com/v1"}
  api_key: ${AI_API_KEY:""}
  model: ${AI_MODEL:"gpt-4o-mini"}
```

4) Chạy ứng dụng
```bash
python main.py
```

5) Thêm dữ liệu khuôn mặt (GUI)
- Lần chạy đầu sẽ mở GUI. Dùng nút “Thêm Người Mới” và “Xây Dựng Lại Tất Cả”.

## 📚 Tài liệu chi tiết
- `docs/architecture.md` — Kiến trúc & luồng dữ liệu
- `docs/installation.md` — Cài đặt & thiết lập
- `docs/usage.md` — Cách sử dụng (CLI/GUI/Telegram)
- `docs/configuration.md` — Cấu hình và biến môi trường
- `docs/api/core.md` — API chính (Camera, CameraManager, Detectors, Recorder)
- `docs/bot.md` — Bot Telegram & Trợ lý AI
- `docs/gui.md` — Giao diện quản lý
- `docs/testing.md` — Benchmark & kiểm thử
- `docs/security.md` — Bảo mật & riêng tư
- `docs/troubleshooting.md` — Sự cố thường gặp

## 📂 Cấu trúc dự án

| Đường dẫn | Mô tả |
| --- | --- |
| `main.py` | Điểm vào. Khởi tạo `GuardianApp`, Bot, GUI, Recorder, CameraManager. |
| `config/settings.py` | Tải `config/config.yaml`, hỗ trợ `${ENV:"default"}`, cung cấp `settings`. |
| `config/config.yaml` | Toàn bộ tham số cấu hình (camera, ai, telegram, models, paths, recorder…). |
| `core/camera.py` | Lớp `Camera`: đọc khung hình, IR detection, pipeline phát hiện, kết xuất. |
| `core/camera_manager.py` | Quản lý nhiều camera, luồng xử lý, truy cập frame. |
| `core/detection/*.py` | `FireDetector`, `FaceDetector`, `PersonTracker` (theo dõi/nhận diện). |
| `core/recorder.py` | Ghi video cảnh báo. |
| `bot/*.py` | `GuardianBot`, handlers, gửi ảnh/video, tương tác AI. |
| `gui/*.py` | Giao diện quản lý (CustomTkinter). |
| `utils/*.py` | `StateManager`, `SpamGuard`, `alarm_player`, `performance_monitor`. |
| `Data/Model` | Mô hình (YOLO/InsightFace/OpenVINO/ONNX). |

## 🌙 Hỗ trợ Camera Hồng ngoại (IR)
- Tự động nhận biết IR sau mỗi 10 khung hình, duy trì lịch sử 30 khung để ổn định.
- Bộ lọc riêng cho IR (độ sáng/biến thiên/chuyển động), mặc định bỏ qua khói.

## 📊 Benchmark (ví dụ)

Thiết lập (Windows, OpenVINO/CPU, Python 3.13.7):
- Fire (Small): `avg_latency_ms ≈ 20.72` → `fps ≈ 48.26`
- Person (Small): `avg_latency_ms ≈ 54.29` → `fps ≈ 18.42`
- Face (ONNX/CPU): `avg_latency_ms ≈ 22.42` → `fps ≈ 44.60`

Chạy:
```bash
python benchmark.py
```

Ghi chú: Kết quả phụ thuộc cấu hình máy, driver, và tải hệ thống.
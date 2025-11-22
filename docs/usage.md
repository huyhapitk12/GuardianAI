# Sử dụng hệ thống

Hướng dẫn cách vận hành GuardianAI qua CLI, GUI và Telegram.

## Chạy ứng dụng

```bash
python main.py
```

- Khởi tạo các thành phần: CameraManager, Detectors, Recorder, Bot, GUI.
- Nếu cấu hình Telegram hợp lệ, bot gửi tin nhắn khởi động.

## GUI (Quản lý khuôn mặt và xem camera)
- Tự mở khi chạy lần đầu.
- Tính năng chính:
  - Thêm người mới, quản lý ảnh người quen.
  - Xây dựng lại embedding.
  - Xem camera (chọn nguồn), xem trạng thái bật/tắt nhận diện.

## Telegram Bot

Các lệnh chính (xem `bot/handlers.py`):

- `/start` — Bắt đầu, hiển thị hướng dẫn nhanh.
- `/status` — Trạng thái hệ thống: AI, nhận diện, số cảnh báo.
- `/get_image [source]` — Lấy ảnh từ camera. `source` có thể là id nguồn (tùy cấu hình).
- `/detect [index]` — Bật/tắt nhận diện người. Không tham số: hiển thị trạng thái tất cả camera.
- `/alarm` — Bật/tắt còi báo động thủ công.
- `/camera_status` — Báo cáo chi tiết camera (kết nối, FPS, độ phân giải...).
- `/test` — Gửi phản hồi test.
- `/clear` — Xóa lịch sử trò chuyện với AI.

Trong cảnh báo:
- Cảnh báo cháy: nút "✅ Cháy thật", "❌ Báo động giả", "📞 Gọi PCCC (114)".
- Cảnh báo người: nút "✅ Có nhận ra", "❌ Không nhận ra".

Trợ lý AI (nếu `ai.enabled: true`):
- Có thể điều khiển bằng ngôn ngữ tự nhiên (ví dụ: "bật hệ thống lên", "cho xem camera").
- Mã hành động nội tuyến: `[ACTION:TOGGLE_ON|TOGGLE_OFF|GET_IMAGE|ALARM_ON|ALARM_OFF]` (được trích xuất tự động).

## Bật/tắt theo camera

- Dùng `/detect` xem danh sách camera và trạng thái.
- Dùng `/detect <index>` để bật/tắt nhanh camera cụ thể (0-based).

## Ghi hình sự kiện

- Tự động ghi khi có cảnh báo; gửi clip sau khi hoàn tất.
- Thời lượng, FPS, codec thiết lập trong `recorder.*` của `config/config.yaml`.

## Benchmarks

- Chạy: `python benchmark.py`
- Xem README và `docs/testing.md` để tham khảo kết quả mẫu.

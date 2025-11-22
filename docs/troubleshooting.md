# Khắc phục sự cố & Câu hỏi thường gặp

## Sự cố thường gặp

### Không nhận được tin nhắn Telegram
- Kiểm tra `TELEGRAM_TOKEN` và `TELEGRAM_CHAT_ID` (đặt bằng biến môi trường hoặc trong YAML).
- Mạng có chặn `api.telegram.org`?
- Xem log: có lỗi "Connection test failed" khi khởi động bot?

### Không mở được camera
- `camera.sources` đúng định dạng? (0, "file.mp4", hoặc RTSP)
- Driver camera và quyền truy cập OK?
- Dùng `/camera_status` để xem chi tiết kết nối.

### FPS thấp hoặc lag
- Giảm `camera.target_fps`.
- Tăng `camera.process_every_n_frames`.
- Giảm `camera.process_size` (ví dụ 640x360).
- Chọn `models.yolo_format: openvino` cho CPU.

### Cảnh báo cháy sai (false positives)
- Bật `camera.rgb.check_motion`.
- Tăng các ngưỡng trong `CONFIG_GUIDE.md` (độ sáng/biến thiên/pixel sáng...).
- Với IR: siết chặt `red_alert` hoặc tắt `yellow_alert`.

### Bỏ lỡ cháy thật (false negatives)
- Giảm các ngưỡng IR trong `docs/config_guide.md`.
- Giảm `fire_confidence_threshold` trong `detection.*`.

### Nhận diện người kém chính xác
- Thêm nhiều ảnh đa dạng cho mỗi người.
- Chạy lại "Xây Dựng Lại" embeddings.
- Điều chỉnh `face_recognition_threshold`.

### Ghi hình không gửi về Telegram
- Kiểm tra dung lượng file < `telegram.video_preview_limit_mb`.
- Kiểm tra thư mục `tmp/` có file mp4 đã hoàn tất.
- Kiểm tra log Recorder (mở writer, finalize...).

## FAQ

- Hỏi: Có chạy được trên CPU không? 
  - Đáp: Có. Dự án tối ưu cho CPU với OpenVINO/ONNX.

- Hỏi: Có bắt buộc cấu hình AI không?
  - Đáp: Không. AI trợ lý là tùy chọn (`ai.enabled: false`).

- Hỏi: Có hỗ trợ nhiều camera?
  - Đáp: Có. `camera.sources` là danh sách; dùng `CameraManager.add_new_camera` để thêm khi đang chạy.

- Hỏi: Có chế độ IR?
  - Đáp: Có. Tự động nhận biết và áp dụng bộ lọc IR. Xem `INFRARED_DETECTION.md`.

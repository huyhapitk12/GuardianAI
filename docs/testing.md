# Kiểm thử & Benchmark

Hướng dẫn kiểm thử nhanh và đo hiệu năng cho GuardianAI.

## Kiểm thử nhanh

1) Cấu hình `config/config.yaml` với nguồn `camera.sources` phù hợp.
2) Chạy `python main.py`.
3) Xác nhận:
   - GUI hiển thị, xem được camera.
   - Bot Telegram gửi tin nhắn khởi động (nếu đã cấu hình).
   - Lệnh `/get_image` hoạt động.

## Benchmark mô hình

Script có sẵn: `benchmark.py`

Chạy:
```bash
python benchmark.py
```

Đầu ra (ví dụ, rút gọn):
```
[fire]   avg_latency_ms=20.72  fps=48.26
[person] avg_latency_ms=54.29  fps=18.42
[face]   avg_latency_ms=22.42  fps=44.60
```

Lưu ý: Kết quả phụ thuộc phần cứng, hệ điều hành, driver, và tải hệ thống.

## Gợi ý tối ưu

- `camera.target_fps`: giảm để giảm tải CPU.
- `camera.process_every_n_frames`: tăng để bỏ bớt khung.
- `camera.process_size`: giảm kích thước xử lý (ví dụ 640x360).
- Dùng định dạng model `openvino` cho YOLO trên CPU.
- Bật `camera.rgb.check_motion` để giảm false positives ánh sáng tĩnh.

## Kiểm thử IR

- Dùng cảnh ban đêm/IR để xác nhận chỉ báo "IR MODE" và log chuyển đổi chế độ.
- Điều chỉnh ngưỡng trong `docs/config_guide.md` khi cần.

## Quy trình QA đề xuất

- Test ngày/đêm, có/không có chuyển động, nguồn RTSP và webcam.
- Kiểm thử nút xác nhận Telegram và logic `SpamGuard`.
- Xác nhận clip ghi được gửi sau cảnh báo.

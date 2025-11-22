# Cấu hình & Biến môi trường

Tất cả cấu hình tập trung tại `config/config.yaml` và được truy cập qua `config/settings.py` dưới đối tượng `settings`.

## Đọc giá trị từ biến môi trường
- Hỗ trợ cú pháp `${ENV:"default"}` trong YAML.
- Ví dụ:

```yaml
telegram:
  token: ${TELEGRAM_TOKEN:""}
  chat_id: ${TELEGRAM_CHAT_ID:""}
```

- Khi chạy, nếu biến môi trường không tồn tại, giá trị `default` sẽ được dùng.

## Các nhóm cấu hình chính

- `telegram.*`: token, chat_id, timeout, giới hạn video.
- `camera.*`: nguồn (`sources`), tốc độ mục tiêu (`target_fps`), IR, RGB motion check, kích thước xử lý, chu kỳ xử lý.
- `detection.*`: ngưỡng tin cậy cho fire/smoke/person/face.
- `fire_logic.*`: hợp nhất nhiều tiêu chí khu vực/thời gian, cửa sổ tích lũy, lockdown, theo dõi tăng trưởng vùng lửa.
- `models.*`: định dạng/size YOLO, cấu hình InsightFace.
- `paths.*`: thư mục mô hình, ảnh người quen, tệp embedding, tmp, âm thanh.
- `recorder.*`: thời lượng clip, FPS, codec.
- `spam_guard.*`: chống spam cảnh báo (debounce, quotas).
- `reid.*`: tham số theo dõi nhận diện lại.
- `ai.*`: bật/tắt trợ lý, endpoint API (kiểu OpenAI), model, max tokens, nhiệt độ.

## IR & Motion
- Tham khảo chi tiết trong: `docs/config_guide.md` (logic phát hiện IR, ngưỡng, bộ lọc, tips)

## Ví dụ tối thiểu

```yaml
telegram:
  token: ${TELEGRAM_TOKEN:""}
  chat_id: ${TELEGRAM_CHAT_ID:""}

camera:
  sources: 0
  target_fps: 10

ai:
  enabled: false
```

## Gợi ý bảo mật
- Không lưu trực tiếp token/API key trong repo. Sử dụng biến môi trường.
- Nếu cần commit file YAML ví dụ, dùng giá trị rỗng hoặc placeholder.

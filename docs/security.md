# Bảo mật & Riêng tư

Khuyến nghị về bảo mật và xử lý dữ liệu trong GuardianAI.

## Secrets (Token/API Key)
- Không commit token/API key vào repo.
- Sử dụng biến môi trường và cú pháp `${ENV:"default"}` trong `config/config.yaml`.
- Xem `docs/configuration.md` để cấu hình an toàn.

## Dữ liệu nhạy cảm
- Ảnh/video và embeddings khuôn mặt nằm trong `Data/*`.
- Không chia sẻ công khai thư mục `Data/` trừ khi đã ẩn danh hoá.
- Xoá dữ liệu thử nghiệm trước khi mở PR.

## Telegram
- Sử dụng bot token riêng cho môi trường dev và production.
- Cân nhắc giới hạn chat_id để không gửi nhầm kênh.

## Ghi hình & Nhật ký
- Clip ghi cảnh báo lưu ở `tmp/` (mặc định) và có thể gửi qua Telegram.
- Log có thể chứa thông tin kỹ thuật (FPS, chế độ IR…); tránh ghi thông tin cá nhân.

## Quyền truy cập
- Hạn chế quyền filesystem đối với thư mục dữ liệu và model.
- Không chạy dưới quyền admin trừ khi thực sự cần.

## Phụ thuộc & cập nhật
- Cập nhật định kỳ thư viện an ninh (requests/aiohttp/telegram/...)
- Kiểm tra rủi ro giấy phép và CVE nếu dùng bản phân phối lại.

## Kiến trúc an toàn
- `StateManager` quản lý trạng thái đa luồng; tránh race conditions bằng cách sử dụng cấu trúc dữ liệu an toàn.
- Không bắt lỗi rộng (`except Exception`) nếu không có xử lý có ý nghĩa.

## Tuân thủ
- Nếu triển khai trong môi trường thực tế, xem xét yêu cầu pháp lý về giám sát hình ảnh và lưu trữ dữ liệu (GDPR/CCPA hoặc luật địa phương).

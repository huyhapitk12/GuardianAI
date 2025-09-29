# GuardianAI - Hệ thống Giám sát An ninh Thông minh

**GuardianAI** là một giải pháp an ninh toàn diện, mã nguồn mở, sử dụng Trí tuệ Nhân tạo (AI) và Thị giác Máy tính (Computer Vision) để giám sát, phát hiện và cảnh báo các mối đe dọa trong thời gian thực. Hệ thống được thiết kế để hoạt động 24/7, bảo vệ ngôi nhà hoặc cơ sở của bạn khỏi các nguy cơ như cháy nổ và xâm nhập trái phép, với khả năng tương tác và điều khiển thông minh qua Telegram.

## ✨ Tính năng nổi bật

### 🛡️ Chức năng Cốt lõi (Hoạt động độc lập)
- **Phát hiện Cháy & Khói:** Sử dụng model **YOLOv8/YOLO11** để phát hiện lửa và khói với độ chính xác cao.
- **Phát hiện & Nhận diện Người:** Phân biệt giữa "Người quen" và "Người lạ" bằng model **InsightFace**.
- **Cảnh báo Thông minh qua Telegram:** Gửi cảnh báo tức thì kèm hình ảnh/video khi phát hiện sự kiện.
- **Tương tác Trực quan:** Người dùng có thể xác nhận "Cháy thật" / "Báo động giả" qua các nút bấm trên Telegram.
- **Kích hoạt Còi báo động:** Tự động hú còi khi có cảnh báo khẩn cấp hoặc khi người dùng ra lệnh.
- **Ghi hình Sự kiện:** Tự động quay lại video clip của sự kiện và gửi cho người dùng.
- **Giao diện Quản lý (GUI):** Giao diện đồ họa trực quan để dễ dàng quản lý cơ sở dữ liệu khuôn mặt người quen.
- **Điều khiển bằng Lệnh:** Hỗ trợ các lệnh để điều khiển hệ thống ngay cả khi AI không hoạt động: `/alarm_on`, `/get_image`, `/get_image`, `/detect`

### 🧠 Chức năng Nâng cao (Tích hợp AI linh hoạt)
- **Hỗ trợ đa nền tảng AI:** Dễ dàng chuyển đổi giữa **Google Gemini** và bất kỳ **API nào tương thích OpenAI** (bao gồm OpenAI chính thức, các mô hình self-host như LM Studio, Ollama).
- **Phân tích Phản hồi bằng AI:** Hiểu ngôn ngữ tự nhiên của người dùng để xác nhận cảnh báo (ví dụ: "không phải người lạ đâu" -> hệ thống sẽ hiểu và bỏ qua).
- **Điều khiển bằng Ngôn ngữ Tự nhiên:** Ra lệnh cho hệ thống bằng các câu nói thông thường như "bật còi báo động lên" hay "cho xem camera".
- **Trợ lý AI:** Trò chuyện và trả lời các câu hỏi khác của người dùng.

## 🛠️ Hướng dẫn Cài đặt & Vận hành

Thực hiện theo các bước dưới đây để cài đặt và khởi chạy hệ thống GuardianAI trên máy của bạn.

### Bước 1: Yêu cầu Hệ thống
- **Hệ điều hành:** Windows 10/11, Ubuntu 20.04+ (khuyến nghị)
- **Phần mềm:**
  - **Python 3.10 trở lên:** [Tải Python](https://www.python.org/downloads/).
  - **Git:** [Tải Git](https://git-scm.com/downloads/)

### Bước 2: Tải Mã nguồn
Mở Terminal (hoặc Command Prompt/PowerShell trên Windows) và chạy lệnh sau:
```bash
git clone https://github.com/huyhapitk12/GuardianAI.git
cd GuardianAI
```

### Bước 3: Cài đặt các Thư viện
Tất cả các thư viện cần thiết đã được liệt kê trong file `requirements.txt`. Chạy lệnh sau để cài đặt tự động:
```bash
pip install -r requirements.txt
```
*Quá trình này có thể mất vài phút vì nó sẽ tải các model AI và thư viện lớn.*

### Bước 4: Cấu hình Hệ thống
Đây là bước quan trọng nhất. Mở file `config.py` bằng một trình soạn thảo code (như VS Code, Sublime Text) và chỉnh sửa các thông số sau:

1.  **Tích hợp Telegram:**
    -   `TELEGRAM_TOKEN`: Token của bot Telegram của bạn (lấy từ **@BotFather**).
    -   `TELEGRAM_CHAT_ID`: ID của cuộc trò chuyện (cá nhân hoặc nhóm) mà bạn muốn bot gửi cảnh báo.

2.  **Tích hợp AI (Tùy chọn):**
    -   `API_KEY`: Dán API Key của bạn. (Đối với dịch vụ self-host không cần key, có thể điền một chuỗi bất kỳ như `"ollama"`).
    -   `API_BASE`: **Quan trọng!**
        -   Đối với OpenAI chính thức: `"https://api.openai.com/v1"`
         -   Đối với Google Gemini: `"https://generativelanguage.googleapis.com/v1beta/openai/"`
        -   Đối với LM Studio: `"http://localhost:1234/v1"`
        -   Đối với Ollama (qua proxy như LiteLLM): `"http://localhost:8000"`
    -   `AI_MODEL`: Tên model bạn muốn sử dụng (ví dụ: `"gpt-4o-mini"`, `"LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF"`, `"gemini-2.5-flash"`).

3.  **Nguồn Video:**
    -   `IP_CAMERA_URL`: Đây là nguồn video đầu vào.
        -   Để dùng webcam: `IP_CAMERA_URL = 0`
        -   Để dùng camera IP: `IP_CAMERA_URL = "rtsp://user:pass@192.168.1.10:554/stream1"` (thay bằng URL của bạn).
        -   Để thử nghiệm với file video: `IP_CAMERA_URL = "test.mp4"` (thay bằng video của bạn)

### Bước 5: Thêm Dữ liệu Khuôn mặt
- Chạy chương trình lần đầu, một giao diện đồ họa sẽ hiện ra.
- Sử dụng nút **"Thêm Người Mới"** để tạo một thư mục cho người quen và chọn ảnh của họ. Bạn nên thêm nhiều ảnh với các góc mặt khác nhau.
- Sau khi thêm, nhấn nút **"Xây Dựng Lại Tất Cả"**. Hệ thống sẽ quét tất cả các ảnh, mã hóa khuôn mặt và lưu vào file để nhận diện sau này.

### Bước 6: Khởi chạy
Sau khi hoàn tất cấu hình, mở Terminal/Command Prompt tại thư mục dự án và chạy:
```bash
python main.py
```
Hệ thống sẽ bắt đầu giám sát. Mọi hoạt động, cảnh báo sẽ được ghi lại trong console và gửi đến Telegram.

## 📂 Cấu trúc Dự án & Nguồn Module

Dự án được tổ chức một cách logic để dễ dàng bảo trì và mở rộng.

| File / Thư mục | Công dụng và Nguồn gốc |
| :--- | :--- |
| **`main.py`** | **Điểm khởi đầu của chương trình.** Chịu trách nhiệm khởi tạo các luồng (Bot, GUI, Camera), liên kết các thành phần và chạy vòng lặp phát hiện chính. |
| **`config.py`** | **File cấu hình trung tâm.** Chứa tất cả các thông số quan trọng như API keys, đường dẫn, ngưỡng phát hiện, giúp dễ dàng tùy chỉnh hệ thống mà không cần sửa code. |
| **`detection_core.py`** | **Bộ não thị giác của hệ thống.** Tải và vận hành các model AI (YOLO, InsightFace), xử lý từng khung hình, theo dõi đối tượng (SORT tracker), và kích hoạt callback khi có sự kiện. |
| **`telegram_bot.py`** | **Cầu nối với người dùng.** Xử lý tất cả logic của Telegram Bot, bao gồm nhận lệnh, gửi cảnh báo, tương tác với nút bấm và tích hợp với các API AI để xử lý ngôn ngữ tự nhiên. |
| **`gui_manager.py`** | **Giao diện quản lý dữ liệu.** Xây dựng bằng `customtkinter`, cung cấp một giao diện đồ họa thân thiện để người dùng quản lý cơ sở dữ liệu khuôn mặt. |
| **`alarm_player.py`** | **Module phát âm thanh.** Sử dụng `pygame` để phát còi báo động với hiệu ứng tăng dần âm lượng, đảm bảo cảnh báo đủ lớn nhưng không gây sốc. |
| **`video_recorder.py`** | **Module ghi hình.** Chịu trách nhiệm ghi lại các clip video khi có sự kiện, xử lý nén (nếu có `ffmpeg`) và gửi file lên Telegram. |
| **`shared_state.py`** | **Trung tâm chia sẻ trạng thái.** Một giải pháp quan trọng để các luồng (threads) khác nhau có thể truy cập và thay đổi trạng thái của nhau một cách an toàn. |
| **`state_manager.py`** | Quản lý trạng thái logic của ứng dụng, như cảnh báo nào đang chờ phản hồi, tính năng nhận diện đang bật hay tắt. |
| **`spam_guard.py`** | Một bộ lọc thông minh để chống spam cảnh báo, tránh gửi các cảnh báo trùng lặp hoặc quá dồn dập trong một khoảng thời gian ngắn. |
| **`Data/`** | Thư mục chứa dữ liệu: `Model/` (các model AI), `Image/` (ảnh người quen), `Audio/` (file âm thanh báo động). |
| **`Lib/`** | Chứa các thư viện phụ trợ đã được tùy chỉnh hoặc không có sẵn trên pip, như **InsightFace** và **SORT Tracker**. |
| **`requirements.txt`** | Danh sách các thư viện Python cần thiết để cài đặt. |
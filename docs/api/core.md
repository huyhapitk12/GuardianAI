# API tham chiếu (Core)

Tổng hợp các lớp và phương thức chính trong `core/*` để tích hợp/tuỳ biến.

> Lưu ý: Đây là bản tóm tắt thực tiễn từ code hiện tại; tên/tham số có thể thay đổi theo phiên bản.

## Camera (`core/camera.py`)

- Lớp `Camera(source, show_window=False, on_person_alert=None, on_fire_alert=None, person_tracker=None)`
  - Nhiệm vụ: đọc khung hình, phát hiện IR, áp dụng pipeline phát hiện (fire/smoke/person/face), render overlay, gửi callback cảnh báo.
  - Phương thức chính:
    - `start_workers(fire_detector, face_detector)` — khởi tạo detector cho camera.
    - `process_frames(state_manager)` — vòng lặp xử lý khung hình (chạy trong thread).
    - `read() -> (bool, frame)` — trả về frame đã xử lý (render) để hiển thị.
    - `read_raw() -> (bool, frame)` — trả về frame thô (phục vụ recorder, v.v.).
    - `get_connection_status() -> dict` — tình trạng kết nối, sức khoẻ, thời gian frame gần nhất.
    - `get_infrared_status() -> dict` — trạng thái IR: `is_infrared_mode`, `ir_detection_count`, `ir_ratio`.
    - `reset_fire_state()` — đặt lại trạng thái phát hiện cháy cho camera này.
    - `release()` — ngắt vòng lặp và giải phóng tài nguyên.

## CameraManager (`core/camera_manager.py`)

- Lớp `CameraManager(show_windows=False, on_person_alert=None, on_fire_alert=None)`
  - Nhiệm vụ: quản lý nhiều `Camera`, tạo/lưu thread xử lý, thao tác tập trung.
  - Phương thức chính:
    - `start_workers(fire_detector, face_detector, state_manager)` — khởi động workers và thread xử lý cho tất cả camera.
    - `stop_all()` — dừng mọi camera và chờ thread kết thúc.
    - `get_all_frames() -> dict[source, (ok, frame)]` — lấy frame render cho tất cả camera.
    - `get_all_raw_frames() -> dict[source, (ok, frame)]` — lấy frame thô cho tất cả camera.
    - `get_camera(source: str) -> Camera|None` — truy cập camera theo id nguồn.
    - `get_all_connection_statuses() -> dict[source, dict]` — trạng thái kết nối của tất cả camera.
    - `reset_fire_state_for_source(source: str)` — đặt lại trạng thái phát hiện cháy của một camera.
    - `add_new_camera(new_source) -> (bool, str)` — thêm camera khi đang chạy và lưu vào `config.yaml`.

## FireDetector (`core/detection/fire_detector.py`)

- Lớp `FireDetector`
  - `initialize() -> bool` — tải model YOLO (OpenVINO/ONNX/PyTorch) và tối ưu CPU.
  - `detect(image: np.ndarray) -> List[dict]` — phát hiện `fire/smoke`, trả về `[{bbox, class, area, conf}, ...]`.
  - `update_model(size: str|None, format_type: str|None) -> bool` — chuyển kích thước/định dạng model.

## FaceDetector (`core/detection/face_detector.py`)

- Lớp `FaceDetector(model_name: str|None = None)`
  - `initialize() -> bool` — khởi tạo InsightFace (CPU provider).
  - `load_known_faces() -> bool` — tải embeddings + tên từ cache.
  - `detect_faces(image: np.ndarray) -> list` — trả về danh sách khuôn mặt (đối tượng InsightFace).
  - `recognize_face(embedding: np.ndarray) -> (name|None, distance: float)` — đối sánh theo ngưỡng `settings.detection.face_recognition_threshold`.
  - `rebuild_embeddings() -> int` — quét `Data/Image/*` để xây lại embeddings và cache.
  - `update_model(model_name: str) -> bool` — chuyển model InsightFace.

## PersonTracker (`core/detection/person_tracker.py`)

- Lớp `PersonTracker(face_detector=None)`
  - `initialize() -> bool` — tải YOLO person + khởi tạo SORT tracker (nếu khả dụng).
  - `detect_persons(image: np.ndarray) -> List[Tuple[x1,y1,x2,y2]]` — phát hiện người (class=0).
  - `update_tracks(detections, frame, scale_x=1.0, scale_y=1.0) -> Dict[id, data]` — cập nhật theo dõi bằng SORT hoặc IOU.
  - `check_confirmations() -> List[(track_id, alert_type, meta)]` — xác nhận `nguoi_quen`/`nguoi_la` theo ngưỡng `settings.tracker.*`.
  - `clear_all_tracks()` — xoá toàn bộ track.
  - `update_model(size: str|None, format_type: str|None) -> bool` — chuyển kích thước/định dạng YOLO.

## Recorder (`core/recorder.py`)

- Lớp `Recorder`
  - `start(source_id: str, reason="alert", duration=60, wait_for_user=False) -> dict|None` — bắt đầu ghi.
  - `write(frame) -> bool` — ghi một frame (khởi tạo writer ở frame đầu tiên).
  - `extend(extra_seconds: float) -> float` — kéo dài thời gian ghi hiện tại.
  - `resolve_user_wait()` — bỏ trạng thái chờ người dùng để hoàn tất.
  - `check_and_finalize() -> dict|None` — kết thúc ghi khi đủ điều kiện, trả về metadata.
  - `stop_and_discard() -> bool` — dừng và xoá tệp ghi hiện tại.
- Hàm `compress_video(input_path: Path) -> Path|None` — nén bằng ffmpeg nếu có.

## Gợi ý tích hợp

- Luôn kiểm tra `ok` khi gọi `read()/read_raw()`.
- Gọi `CameraManager.stop_all()` trước khi thoát để giải phóng tài nguyên.
- Với Telegram handler, có thể dùng `CameraManager.reset_fire_state_for_source(source)` để tránh cảnh báo liền kề sau khi người dùng xác nhận giả.

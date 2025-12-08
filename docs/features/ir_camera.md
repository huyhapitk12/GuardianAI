# Camera Hồng ngoại (IR Camera)

Tài liệu về tính năng tự động nhận diện và xử lý camera hồng ngoại trong GuardianAI.

---

## Tổng quan

GuardianAI có khả năng **tự động phát hiện** khi camera chuyển sang chế độ hồng ngoại (IR mode) và **tự động điều chỉnh** các bộ lọc phát hiện để phù hợp.

**Tính năng chính:**
- ✅ Tự động phát hiện IR mode
- ✅ Bộ lọc riêng cho RGB và IR
- ✅ Tự động tắt smoke detection trong IR
- ✅ Theo dõi lịch sử để ổn định
- ✅ Hiển thị chỉ báo IR trên GUI

---

## Cách IR Detection Hoạt động

### 1. Đặc điểm IR Frame

Camera hồng ngoại tạo ra khung hình có đặc điểm:
- **Grayscale**: R ≈ G ≈ B (không có màu)
- **Low Saturation**: Saturation rất thấp (gần 0)
- **Low Color Variance**: Độ lệch chuẩn giữa các kênh màu rất nhỏ

### 2. Detection Algorithm

```python
def _detect_ir(self, frame: np.ndarray) -> bool:
    # 1. Tính mean và std của từng kênh
    means = frame.mean(axis=(0, 1))  # [B, G, R]
    stds = frame.std(axis=(0, 1))
    
    # 2. Channel variance
    channel_std = np.std(means)  # Độ lệch giữa các kênh
    
    # 3. Color ratio
    min_val, max_val = means.min(), means.max()
    if max_val > 0:
        color_ratio = min_val / max_val
    else:
        color_ratio = 1.0
    
    # 4. Saturation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean()
    
    # 5. Check thresholds
    is_ir = (
        channel_std < threshold_std and        # 5.0
        color_ratio > threshold_ratio and      # 0.95
        saturation < threshold_saturation      # 20
    )
    
    return is_ir
```

### 3. Lịch sử & Ổn định

```python
# Lưu 30 khung hình gần nhất
ir_history = deque(maxlen=30)

# Check mỗi 10 frames
if frame_idx % 10 == 0:
    is_ir = _detect_ir(frame)
    ir_history.append(is_ir)

# Xác định mode dựa trên tỷ lệ
ir_ratio = sum(ir_history) / len(ir_history)
is_ir_mode = ir_ratio >= detection_threshold  # 0.7 = 70%
```

**Tại sao cần lịch sử?**
- Tránh nhảy mode liên tục do noise
- Ổn định detection khi camera chuyển đổi
- Giảm false switches

---

## Cấu hình IR Detection

### File: `config/config.yaml`

```yaml
camera:
  infrared:
    # Enable/disable IR detection
    enabled: true
    
    # Detection threshold (70% frames must be IR)
    detection_threshold: 0.7
    
    # Auto disable smoke detection in IR
    disable_smoke_detection: true
    
    # Detection parameters
    detection:
      channel_std_threshold: 5.0
      color_ratio_threshold: 0.95
      saturation_threshold: 20
```

### Giải thích Parameters

**`enabled`** (true/false)
- Bật/tắt toàn bộ tính năng IR detection
- Nếu `false`, luôn dùng RGB filters

**`detection_threshold`** (0.0-1.0)
- Tỷ lệ frames phải là IR để xác nhận IR mode
- `0.7` = 70% trong 30 frames gần nhất
- Tăng = ổn định hơn nhưng chậm chuyển đổi
- Giảm = nhạy hơn nhưng dễ nhảy mode

**`disable_smoke_detection`** (true/false)
- Tự động tắt smoke detection trong IR mode
- Khuyến nghị: `true` (khói không rõ trong IR)

**`channel_std_threshold`** (mặc định: 5.0)
- Độ lệch chuẩn giữa R, G, B
- IR có R ≈ G ≈ B nên std nhỏ
- Tăng = dễ detect IR (ít strict)

**`color_ratio_threshold`** (mặc định: 0.95)
- Tỷ lệ min/max giữa channels
- `0.95` = các kênh phải gần nhau 95%
- Tăng = strict hơn (phải rất grayscale)

**`saturation_threshold`** (mặc định: 20)
- Saturation tối đa (0-255)
- IR có saturation rất thấp
- Giảm = strict hơn

---

## RGB vs IR Filters

### RGB Mode Filters

**Fire Detection:**
- ✅ Color-based (HSV hue/saturation)
- ✅ Motion detection (optical flow)
- ✅ Both Fire và Smoke

```yaml
camera:
  fire_filter:
    rgb:
      hue_fire_min: 0
      hue_fire_max: 12
      saturation_min: 80
      brightness_min: 100
      # ... (xem config_guide.md)
```

### IR Mode Filters

**Fire Detection:**
- ✅ Brightness-based (mean/max/std)
- ✅ Hot spot detection
- ✅ Gradient analysis
- ❌ Smoke detection (tự động tắt)

```yaml
camera:
  fire_filter:
    infrared:
      brightness_mean_min: 120
      brightness_max_min: 180
      bright_core_ratio_min: 0.08
      # ... (xem config_guide.md)
```

**Tại sao khác nhau?**
- RGB: Lửa có màu sắc đặc trưng (đỏ-cam-vàng)
- IR: Lửa là vùng **rất sáng** trong ảnh grayscale

---

## GUI Indicators

### Camera Tab

```
📹 Camera 0: 1920x1080 @ 10fps
🔴 Recording | ✅ Connected | 🌙 IR MODE
```

**Chỉ báo:**
- 🌙 **IR MODE**: Camera đang ở chế độ hồng ngoại
- ☀️ **RGB MODE**: Camera ở chế độ thường (có thể không hiển thị)

### Log Messages

```
🌙 IR Mode: ON (ratio=0.83)
☀️ IR Mode: OFF (ratio=0.23)
```

---

## Troubleshooting

### Vấn đề 1: Camera nhảy IR/RGB liên tục

**Nguyên nhân:** Detection threshold quá thấp

**Giải pháp:**

```yaml
camera:
  infrared:
    detection_threshold: 0.8  # Tăng từ 0.7
```

### Vấn đề 2: Không detect IR dù camera đã bật IR

**Nguyên nhân:** Thresholds quá strict

**Giải pháp:**

```yaml
camera:
  infrared:
    detection:
      channel_std_threshold: 8.0   # Tăng từ 5.0
      color_ratio_threshold: 0.90  # Giảm từ 0.95
      saturation_threshold: 30     # Tăng từ 20
```

### Vấn đề 3: Luôn detect IR (false positive)

**Nguyên nhân:** Thresholds quá lỏng hoặc scene grayscale

**Giải pháp:**

```yaml
camera:
  infrared:
    detection:
      channel_std_threshold: 3.0   # Giảm từ 5.0
      color_ratio_threshold: 0.98  # Tăng từ 0.95
      saturation_threshold: 15     # Giảm từ 20
```

### Vấn đề 4: Muốn tắt IR detection

**Giải pháp:**

```yaml
camera:
  infrared:
    enabled: false  # Tắt hoàn toàn
```

---

## Best Practices

### 1. Testing IR Detection

```bash
# Chạy app và quan sát logs
python main.py

# Xem log IR detection
# Tìm messages: "IR Mode: ON/OFF (ratio=...)"
```

### 2. Fine-tuning

1. **Quan sát IR ratio** trong logs
2. **Điều chỉnh detection_threshold** dựa trên ratio
3. **Test với scene thực tế** (ngày/đêm)

### 3. Scene-specific Tuning

**Môi trường tối (luôn IR):**
```yaml
camera:
  infrared:
    detection_threshold: 0.6  # Nhạy hơn
```

**Môi trường sáng (hiếm IR):**
```yaml
camera:
  infrared:
    detection_threshold: 0.8  # Chặt hơn
```

---

## API Reference

### Camera Class

```python
class Camera:
    def _detect_ir(self, frame: np.ndarray) -> bool
        """Detect if frame is infrared"""
    
    def get_infrared_status(self) -> bool
        """Check if camera is in IR mode"""
    
    def _apply_color_filter(self, frame: np.ndarray)
        """Apply filters based on IR/RGB mode"""
```

**Xem thêm:** [docs/api/core.md](file:///d:/GuardianAI/docs/api/core.md)

---

## Xem thêm

- [config_guide.md](file:///d:/GuardianAI/docs/config_guide.md) - Hướng dẫn chi tiết Fire Filters RGB/IR
- [docs/features/fire_detection.md](file:///d:/GuardianAI/docs/features/fire_detection.md) - Fire Detection system
- [architecture.md](file:///d:/GuardianAI/docs/architecture.md) - Kiến trúc hệ thống
- [troubleshooting.md](file:///d:/GuardianAI/docs/troubleshooting.md) - Khắc phục sự cố

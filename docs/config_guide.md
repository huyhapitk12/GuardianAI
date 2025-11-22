# Hướng dẫn Cấu hình GuardianAI

## Tổng quan

Tất cả cấu hình của GuardianAI được quản lý tập trung trong file `config/config.yaml`. Bạn có thể dễ dàng tùy chỉnh các tham số mà không cần sửa code.

---

## 📷 Cấu hình Camera

### 1. Kiểm tra Chuyển động cho RGB (`camera.rgb`)

Cấu hình kiểm tra optical flow cho camera RGB thông thường:

```yaml
camera:
  rgb:
    check_motion: true           # Bật/tắt kiểm tra chuyển động
    motion_threshold: 0.5        # Ngưỡng magnitude chuyển động
    motion_std_min: 0.15         # Độ lệch chuẩn tối thiểu
```

#### Giải thích:

- **`check_motion`** (`true`/`false`)
  - Bật/tắt kiểm tra optical flow cho chế độ RGB
  - Giúp loại bỏ false positives từ ánh sáng tĩnh (đèn, phản xạ)
  - Khuyến nghị: `true` để tăng độ chính xác

- **`motion_threshold`** (mặc định: `0.5`)
  - Ngưỡng magnitude của optical flow vector
  - Giá trị cao hơn = yêu cầu chuyển động mạnh hơn
  - Điều chỉnh nếu có nhiều false positives/negatives

- **`motion_std_min`** (mặc định: `0.15`)
  - Độ lệch chuẩn magnitude tối thiểu
  - Lửa có chuyển động không đều (std cao)
  - Giảm giá trị = dễ pass hơn

---

### 2. Cấu hình Camera Hồng ngoại (Infrared)

#### Vị trí: `camera.infrared`

Phần này điều khiển tính năng phát hiện và xử lý tự động cho camera hồng ngoại.

### Cấu hình Cơ bản

```yaml
camera:
  infrared:
    enabled: true                    # Bật/tắt tính năng phát hiện IR
    detection_threshold: 0.7         # 70% khung hình phải là IR
    disable_smoke_detection: true    # Tắt phát hiện khói trong IR
```

#### Giải thích:

- **`enabled`** (`true`/`false`)
  - Bật hoặc tắt hoàn toàn tính năng phát hiện IR
  - Nếu tắt, hệ thống sẽ luôn dùng bộ lọc RGB thông thường

- **`detection_threshold`** (`0.0` - `1.0`)
  - Tỷ lệ khung hình phải là IR để xác nhận camera đang ở chế độ IR
  - `0.7` = 70% trong 30 khung hình gần nhất phải là IR
  - Giá trị cao hơn = ổn định hơn nhưng chậm hơn khi phát hiện
  - Giá trị thấp hơn = nhạy hơn nhưng dễ nhảy chế độ

- **`disable_smoke_detection`** (`true`/`false`)
  - Tự động bỏ qua smoke detection khi ở chế độ IR
  - Khuyến nghị: `true` vì khói rất khó nhận diện chính xác trong IR

---

### Cấu hình Nâng cao - Phát hiện IR

```yaml
camera:
  infrared:
    detection:
      channel_std_threshold: 5.0     # Độ lệch chuẩn giữa các kênh màu
      color_ratio_threshold: 0.95    # Tỷ lệ chênh lệch màu
      saturation_threshold: 20       # Saturation trung bình
```

#### Giải thích:

Các tham số này xác định một khung hình có phải là IR hay không:

- **`channel_std_threshold`** (mặc định: `5.0`)
  - Độ lệch chuẩn giữa R, G, B channels
  - IR mode có R ≈ G ≈ B nên std rất thấp
  - Giá trị cao hơn = dễ phát hiện IR hơn (ít nghiêm ngặt)
  - Giá trị thấp hơn = chỉ chấp nhận IR rất "thuần"

- **`color_ratio_threshold`** (mặc định: `0.95`)
  - Tỷ lệ min/max giữa các channels (0-1)
  - `0.95` nghĩa là các kênh phải gần bằng nhau (95%)
  - Giá trị cao hơn = nghiêm ngặt hơn (phải rất grayscale)

- **`saturation_threshold`** (mặc định: `20`)
  - Saturation trung bình tối đa (0-255)
  - IR mode có saturation rất thấp (gần như không có màu)
  - Giá trị thấp hơn = nghiêm ngặt hơn

---

### Cấu hình Bộ lọc Cảnh báo Đỏ (Red Alert)

```yaml
camera:
  infrared:
    red_alert:
      brightness_mean_min: 100       # Độ sáng trung bình tối thiểu
      brightness_max_min: 200        # Độ sáng max tối thiểu
      brightness_std_min: 20         # Biến đổi cường độ tối thiểu
      bright_pixel_threshold: 180    # Ngưỡng pixel sáng
      bright_pixel_ratio_min: 0.05   # Tỷ lệ pixel sáng tối thiểu (5%)
      very_bright_threshold: 240     # Ngưỡng pixel rất sáng
      very_bright_ratio_max: 0.8     # Tỷ lệ pixel rất sáng tối đa (80%)
      # Kiểm tra chuyển động
      check_motion: true             # Bật kiểm tra optical flow
      motion_threshold: 0.5          # Ngưỡng magnitude chuyển động
      motion_std_min: 0.15           # Độ lệch chuẩn chuyển động tối thiểu
```

#### Giải thích:

Bộ lọc nghiêm ngặt cho cảnh báo đỏ trong IR mode:

- **`brightness_mean_min`** (mặc định: `100`)
  - Độ sáng trung bình tối thiểu của vùng phát hiện (0-255)
  - Lửa thường rất sáng trong IR
  - Giảm giá trị = chấp nhận lửa tối hơn (nhiều cảnh báo hơn)

- **`brightness_max_min`** (mặc định: `200`)
  - Pixel sáng nhất trong vùng phải > ngưỡng này
  - Hoặc đạt `brightness_mean_min` hoặc `brightness_max_min`
  - Giảm giá trị = dễ pass hơn

- **`brightness_std_min`** (mặc định: `20`)
  - Độ lệch chuẩn độ sáng tối thiểu
  - Lửa có biến đổi cường độ, không đồng đều
  - Giảm giá trị = chấp nhận vùng đồng đều hơn

- **`bright_pixel_threshold`** (mặc định: `180`)
  - Ngưỡng để xác định pixel có "sáng" không
  - Dùng để đếm số pixel sáng

- **`bright_pixel_ratio_min`** (mặc định: `0.05`)
  - Tỷ lệ pixel sáng tối thiểu (5%)
  - Lửa phải có ít nhất 5% vùng rất sáng
  - Giảm giá trị = dễ pass hơn

- **`very_bright_threshold`** (mặc định: `240`)
  - Ngưỡng pixel "quá sáng" (có thể là glare/phản xạ)

- **`very_bright_ratio_max`** (mặc định: `0.8`)
  - Nếu > 80% vùng quá sáng → reject (có thể là glare)
  - Tăng giá trị = ít bị reject hơn

- **`check_motion`** (mặc định: `true`)
  - Bật/tắt kiểm tra chuyển động (optical flow)
  - Lửa thực có chuyển động đặc trưng
  - Tắt nếu muốn chỉ dựa vào độ sáng

- **`motion_threshold`** (mặc định: `0.5`)
  - Ngưỡng magnitude của optical flow vector
  - Giá trị cao hơn = yêu cầu chuyển động mạnh hơn

- **`motion_std_min`** (mặc định: `0.15`)
  - Độ lệch chuẩn magnitude tối thiểu
  - Lửa có chuyển động không đều (std cao)
  - Giảm giá trị = dễ pass hơn

---

### Cấu hình Bộ lọc Cảnh báo Vàng (Yellow Alert)

```yaml
camera:
  infrared:
    yellow_alert:
      brightness_mean_min: 80        # Độ sáng trung bình (lỏng hơn)
      brightness_max_min: 150        # Độ sáng max (lỏng hơn)
      brightness_std_min: 15         # Biến đổi cường độ (lỏng hơn)
      very_bright_threshold: 245     # Ngưỡng pixel rất sáng
      very_bright_ratio_max: 0.9     # Tỷ lệ pixel rất sáng tối đa (90%)
      very_dark_threshold: 30        # Ngưỡng pixel quá tối
      very_dark_ratio_max: 0.9       # Tỷ lệ pixel quá tối tối đa (90%)
      # Kiểm tra chuyển động (lỏng hơn red alert)
      check_motion: true             # Bật kiểm tra optical flow
      motion_threshold: 0.3          # Ngưỡng magnitude (thấp hơn)
      motion_std_min: 0.10           # Độ lệch chuẩn (thấp hơn)
```

#### Giải thích:

Bộ lọc lỏng hơn cho cảnh báo vàng (nghi ngờ):

- **Các tham số brightness** (thấp hơn red alert)
  - Chấp nhận các vùng nghi ngờ với độ sáng/biến đổi thấp hơn
  - Cho phép người dùng xem và xác nhận

- **`very_bright_ratio_max`** (mặc định: `0.9`)
  - Chấp nhận vùng sáng hơn red alert (90% vs 80%)
  - Lỏng hơn với glare

- **`very_dark_threshold`** (mặc định: `30`)
  - Loại bỏ vùng quá tối (không phải lửa)

- **`very_dark_ratio_max`** (mặc định: `0.9`)
  - Nếu > 90% vùng quá tối → reject

- **`check_motion`** (mặc định: `true`)
  - Bật/tắt kiểm tra chuyển động
  - Lỏng hơn red alert để không bỏ sót

- **`motion_threshold`** (mặc định: `0.3`)
  - Ngưỡng magnitude (thấp hơn red alert)
  - Chấp nhận chuyển động yếu hơn

- **`motion_std_min`** (mặc định: `0.10`)
  - Độ lệch chuẩn tối thiểu (thấp hơn red alert)
  - Dễ pass hơn để tránh bỏ sót

---

## 🎯 Hướng dẫn Tùy chỉnh

### Scenario 0: Tắt kiểm tra chuyển động RGB

**Vấn đề**: Motion check gây lag hoặc bỏ lỡ lửa thật

**Giải pháp**:
```yaml
camera:
  rgb:
    check_motion: false    # Tắt motion check cho RGB
```

### Scenario 1: Camera nhảy chế độ IR/RGB liên tục

**Vấn đề**: Camera chuyển đổi giữa IR và RGB quá nhanh

**Giải pháp**:
```yaml
camera:
  infrared:
    detection_threshold: 0.8  # Tăng từ 0.7 lên 0.8 (80%)
```

### Scenario 2: Quá nhiều false positives (cảnh báo sai)

**Vấn đề**: Hệ thống cảnh báo lửa khi không có lửa

**Giải pháp cho Red Alert**:
```yaml
camera:
  infrared:
    red_alert:
      brightness_mean_min: 120      # Tăng từ 100 (yêu cầu sáng hơn)
      brightness_std_min: 25        # Tăng từ 20 (yêu cầu biến đổi nhiều hơn)
      bright_pixel_ratio_min: 0.08  # Tăng từ 0.05 (yêu cầu nhiều pixel sáng hơn)
```

### Scenario 3: Bỏ lỡ lửa thật (false negatives)

**Vấn đề**: Có lửa thật nhưng không cảnh báo

**Giải pháp**:
```yaml
camera:
  infrared:
    red_alert:
      brightness_mean_min: 80       # Giảm từ 100 (chấp nhận tối hơn)
      brightness_max_min: 150       # Giảm từ 200
      brightness_std_min: 15        # Giảm từ 20
      bright_pixel_ratio_min: 0.03  # Giảm từ 0.05 (chỉ cần 3% pixel sáng)
```

### Scenario 4: Camera IR không được phát hiện

**Vấn đề**: Hệ thống không chuyển sang IR mode dù camera đã bật IR

**Giải pháp**:
```yaml
camera:
  infrared:
    detection_threshold: 0.6        # Giảm từ 0.7 (dễ phát hiện hơn)
    detection:
      channel_std_threshold: 8.0    # Tăng từ 5.0 (ít nghiêm ngặt hơn)
      color_ratio_threshold: 0.90   # Giảm từ 0.95
      saturation_threshold: 30      # Tăng từ 20
```

### Scenario 5: Tắt hoàn toàn phát hiện IR

**Vấn đề**: Muốn dùng bộ lọc RGB cho tất cả

**Giải pháp**:
```yaml
camera:
  infrared:
    enabled: false  # Tắt hoàn toàn tính năng IR
```

### Scenario 6: Tắt kiểm tra chuyển động

**Vấn đề**: Kiểm tra motion gây lag hoặc quá nhiều false negatives

**Giải pháp**:
```yaml
camera:
  infrared:
    red_alert:
      check_motion: false    # Tắt motion check cho red alert
    yellow_alert:
      check_motion: false    # Tắt motion check cho yellow alert
```

### Scenario 7: Điều chỉnh độ nhạy chuyển động

**Vấn đề**: Bỏ lỡ lửa do chuyển động yếu

**Giải pháp - Làm lỏng hơn**:
```yaml
camera:
  infrared:
    red_alert:
      motion_threshold: 0.3      # Giảm từ 0.5 (chấp nhận chuyển động yếu)
      motion_std_min: 0.10       # Giảm từ 0.15
```

**Giải pháp - Làm nghiêm ngặt hơn**:
```yaml
camera:
  infrared:
    red_alert:
      motion_threshold: 0.7      # Tăng từ 0.5 (yêu cầu chuyển động mạnh)
      motion_std_min: 0.20       # Tăng từ 0.15
```

---

## 🧪 Testing & Fine-tuning

### Quy trình điều chỉnh:

1. **Chạy hệ thống** và quan sát log
2. **Ghi chú** các giá trị trong log khi có lửa thật/cảnh báo sai
3. **Điều chỉnh** config dựa trên các giá trị quan sát được
4. **Test lại** và lặp lại

### Các giá trị quan trọng trong log:

```
✅ IR PASS T1: Độ sáng OK (mean=145.3, max=234.1)
✅ IR PASS T2: Biến đổi OK (std=35.2)
✅ IR PASS T3: Vùng sáng OK (ratio=18.30%)
```

Dùng các giá trị này để điều chỉnh thresholds phù hợp.

---

## 💡 Tips

1. **Bắt đầu với giá trị mặc định** và chỉ điều chỉnh khi cần
2. **Thay đổi từng tham số một** để hiểu rõ ảnh hưởng
3. **Ghi chú lại** mọi thay đổi và kết quả
4. **Backup config** trước khi thử nghiệm
5. **Test với video thực tế** của môi trường bạn

---

## 📚 Tài liệu liên quan

- `README.md` - Hướng dẫn tổng quan

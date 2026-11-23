# Quy trình chuẩn (SOP) Thu thập & Gắn nhãn Video Ngôn ngữ Ký hiệu Việt Nam (VSL)

> **Phạm vi:** Phần 1 (Dữ liệu) — Bước 2: **Thu thập và nhãn (tự quay, nhiều người)** do **Quý** phụ trách.  
> **Đầu ra cho bước kế tiếp:** Cung cấp video sạch và manifest chuẩn hóa (`data/manifests/self_manifest.csv` → `data/manifests/dataset_manifest.csv`) làm đầu vào trực tiếp cho **Bước 2: Tiền xử lý MediaPipe Holistic** (do **Thiên** phụ trách).

---

## 1. Yêu cầu đa người quay (Multi-signer Requirements)

Mô hình nhận diện ngôn ngữ ký hiệu rất dễ bị học thuộc đặc điểm cơ thể/quần áo/nền của từng cá nhân nếu chia ngẫu nhiên theo clip. Vì vậy, pipeline áp dụng **chia tập theo người (`signer_id` hoàn toàn tách biệt giữa `train` / `val` / `test`)**.

| Hạng mục | Tối thiểu bắt buộc | Khuyến nghị thực tế |
| --- | --- | --- |
| Số người quay riêng biệt (`signer_id`) | **≥ 3 người** (để đủ 3 tập `train`, `val`, `test`) | **6 – 12 người** (đa dạng giới tính, vóc dáng, tốc độ tay) |
| Số lần quay / ký hiệu / người (`--num-takes`) | **3 lần** | **5 lần** (giúp tập train phong phú mà vẫn giữ chất lượng cao) |
| Độ phủ nhãn mỗi người | Mỗi người quay đủ bộ ký hiệu trong `configs/class_list.json` | Ưu tiên quay trọn bộ 30 ký hiệu cốt lõi cho từng `signer_id` |
| Thời lượng mỗi clip (`--duration`) | `2.0 – 3.0 giây` | **2.5 giây** (mặc định của `collect_videos.py`) |

> **Lưu ý chống rò rỉ dữ liệu (Zero Signer Leakage):** Một cá nhân **chỉ được mang duy nhất một `signer_id`** xuyên suốt mọi buổi quay (kể cả quay khác ngày, khác áo, khác phòng). Không được đổi `signer_id` của cùng một người vì sẽ làm rò rỉ sang tập `val`/`test`.

---

## 2. Quy ước đặt tên `signer_id`

- **Định dạng chuẩn:** `signer_<số thứ tự 2 chữ số>` (ví dụ: `signer_01`, `signer_02`, ..., `signer_15`) hoặc `<tên_viết_tắt>_<số>` (ví dụ: `quy_01`, `thien_02`).
- **Quy tắc kỹ thuật:**
  - Chỉ dùng chữ thường không dấu (`a-z`), chữ số (`0-9`), dấu gạch dưới (`_`) hoặc gạch nối (`-`).
  - **Không** dùng dấu tiếng Việt, khoảng trắng hoặc ký tự đặc biệt.
  - **Không** dùng họ tên thật, số điện thoại hay MSSV làm `signer_id` (đảm bảo ẩn danh dữ liệu).
- **Quản lý bảng ánh xạ ẩn danh:** Nhóm trưởng lưu file bảng ánh xạ `signer_id ↔ tên người tham gia + ngày ký đồng thuận` ngoại tuyến (ngoài kho git).

---

## 3. Thiết lập Camera, Góc quay & Khung hình (Tối ưu cho MediaPipe Holistic)

MediaPipe Holistic trích xuất đồng thời **3 nhóm mốc xương**:
1. **Pose** (33 điểm khung người — đặc biệt vai, khuỷu tay, cổ tay, thân trên),
2. **Hands** (21 điểm × 2 bàn tay — ngón tay, lòng bàn tay),
3. **Face Mesh** (468 điểm khuôn mặt — biểu cảm ngữ pháp VSL).

Do đó, khung hình quay phải tuân thủ nghiêm ngặt:

- **Vị trí & góc máy:**
  - Đặt camera **cố định** trên tripod/giá đỡ (hoặc laptop trên mặt phẳng vững chắc), **không cầm tay quay**.
  - Ống kính đặt **ngang tầm ngực – vai**, hướng trực diện (góc nghiêng ≤ 10°).
- **Khoảng cách & căn khung (Framing):**
  - Đứng hoặc ngồi cách camera khoảng **1.2 m – 1.8 m**.
  - Khung hình phải lấy trọn từ **trên đỉnh đầu (dư khoảng 10% phía trên)** xuống đến **ngang thắt lưng / hông**, đồng thời chừa đủ khoảng trống hai bên để khi **giơ tay lên cao hoặc dang rộng hai khuỷu tay**, bàn tay và các đầu ngón tay **không bao giờ bị cắt khỏi khung hình**.
  - Khi chạy `collect_videos.py`, căn đầu vào **vòng oval vàng**, vai ngang **đường kẻ ngang vàng**, và toàn bộ biên độ tay nằm gọn trong **khung hình chữ nhật màu xanh lá**.
- **Không lật gương video lưu trữ:**
  - Trong VSL có nhiều ký hiệu bất đối xứng giữa tay thuận và tay phụ. Công cụ `collect_videos.py` **luôn lưu khung hình gốc từ camera (không lật ngang)** và **không ghi đè khung hướng dẫn (overlay)** vào file `.mp4` đầu ra.

---

## 4. Ánh sáng, Phông nền & Trang phục

- **Ánh sáng:**
  - Nguồn sáng chính đặt **phía trước mặt người quay** (ánh sáng tự nhiên từ cửa sổ phía trước hoặc đèn bàn/ring light dịu).
  - **Tuyệt đối tránh ngược sáng** (không ngồi quay lưng ra cửa sổ sáng hoặc đèn chói phía sau gây tối khuôn mặt và mất khớp ngón tay).
  - Hạn chế nhấp nháy đèn huỳnh quang (flicker).
- **Phông nền (Background):**
  - Chọn tường trơn hoặc nền đơn sắc, ít chi tiết rối, độ tương phản tốt với màu da tay.
  - **Không có người thứ hai xuất hiện hay đi ngang qua phía sau** (MediaPipe Holistic chỉ theo dõi một chủ thể chính; người lạ phía sau có thể làm nhảy khung xương).
- **Trang phục:**
  - Mặc áo trơn, màu tương phản với màu da bàn tay (ví dụ: áo xanh đậm, đen, xám đậm, xanh navy).
  - Tránh áo có hoa văn quá dày, tránh áo trùng màu da, và hạn chế đeo đồng hồ/vòng tay phản quang mạnh hoặc tay áo quá rộng che khuất cổ tay.

---

## 5. Thông số kỹ thuật Video

| Thông số | Giá trị tiêu chuẩn | Ghi chú |
| --- | --- | --- |
| Độ phân giải | **1280 × 720 (720p HD)** | Tối thiểu chấp nhận: `640 × 480` |
| Tốc độ khung hình (FPS) | **30 FPS** | Giữ ổn định, tắt chế độ làm đẹp/lọc nền nặng của webcam gây tụt FPS |
| Định dạng file | **`.mp4`** (`mp4v` / H.264) | Không cần thu âm thanh |
| Quy ước đặt tên file | `<class_slug>/<signer_id>_<class_slug>_<timestamp>.mp4` | Tự động sinh bởi `collect_videos.py` |

Ví dụ đường dẫn chuẩn:
```text
data/raw/self_recorded/giup_do/signer_01_giup_do_20261122_091530_412.mp4
```

---

## 6. Quy trình nhịp quay từng clip (Clip Timing)

Mỗi clip dài **2.5 giây** (có thể tăng lên `3.0s` với các ký hiệu ghép nhiều động tác) cần tuân theo cấu trúc **3 pha rõ ràng**:

```text
[Đếm ngược 3-2-1]  ──►  [0.0s – 0.4s: Nghỉ trung tính]  ──►  [0.4s – 2.1s: Thực hiện ký hiệu VSL]  ──►  [2.1s – 2.5s: Về nghỉ trung tính]
```

1. **Trong lúc đếm ngược `3 → 2 → 1`:** Nhìn thẳng vào camera, hai tay đặt tự nhiên ở tư thế nghỉ (thả nhẹ trước bụng/đùi).
2. **Pha mở đầu (~0.3–0.4s):** Giữ yên tư thế nghỉ trung tính khi thanh tiến độ đỏ bắt đầu chạy.
3. **Pha thực hiện ký hiệu (~1.5–1.8s):** Thực hiện động tác ký hiệu VSL tự nhiên, dứt khoát, đúng tốc độ giao tiếp thường ngày (không quá chậm kiểu robot, không quá nhanh gây nhòe chuyển động) kết hợp biểu cảm khuôn mặt phù hợp.
4. **Pha kết thúc (~0.3–0.4s):** Đưa hai tay trở về tư thế nghỉ trung tính trước khi hết thời gian.

---

## 7. Kiểm duyệt chất lượng tại chỗ & Quyền riêng tư / Đồng thuận

### 7.1. Tiêu chí Chấp nhận (`s` / `Enter`) vs. Quay lại (`r`)
Ngay sau khi quay xong mỗi take, cửa sổ xem lại (`REVIEW`) sẽ phát lặp lại clip vừa ghi:
- **Nhấn `s` hoặc `Enter` (Chấp nhận & Lưu)** khi:
  - Thực hiện đúng ký hiệu theo nhãn đang chọn trong `configs/class_list.json`.
  - Khuôn mặt và cả hai bàn tay luôn nằm trong khung hình, rõ nét, không nhòe nặng.
  - Có đủ đoạn mở đầu và kết thúc ở tư thế nghỉ.
- **Nhấn `r` (Quay lại)** ngay lập tức nếu:
  - Thực hiện nhầm động tác, khựng giữa chừng hoặc hết giờ trước khi xong động tác.
  - Ngón tay/bàn tay bị cắt ở mép khung hình.
  - Camera bị rung, mất nét, tụt sáng hoặc có người khác đi qua hậu cảnh.

### 7.2. Quyền riêng tư & Đồng thuận (Privacy & Consent)
Trước khi ghi hình bất kỳ tình nguyện viên nào, người phụ trách thu thập cần đảm bảo:
- [ ] Giải thích rõ mục đích thu thập: chỉ phục vụ nghiên cứu và huấn luyện mô hình nhận diện VSL trong phạm vi đồ án/dự án.
- [ ] Người tham gia tự nguyện đồng ý ghi hình khuôn mặt và cử chỉ tay (khuyến nghị lưu xác nhận đồng thuận bằng văn bản/biểu mẫu riêng).
- [ ] Dữ liệu lưu trong hệ thống chỉ gắn mã ẩn danh (`signer_id`), không gắn tên thật hay thông tin định danh cá nhân trong tên file và file CSV.
- [ ] Video thô (`data/raw/`) nằm trong `.gitignore` và **không được push lên kho Git công khai**.
- [ ] Người tham gia có quyền yêu cầu xóa toàn bộ clip gắn với `signer_id` của mình trước khi đóng băng bộ dữ liệu.

---

## 8. Hướng dẫn chạy công cụ thu thập & kiểm tra dữ liệu

### 8.1. Chạy công cụ tự quay (`src/data/collect_videos.py`)

**Cách 1 — Chỉ định trực tiếp `signer_id` và nhãn (khuyến nghị khi quay hàng loạt):**
```powershell
python -m src.data.collect_videos `
  --signer-id signer_01 `
  --class-label "giúp đỡ" `
  --num-takes 5 `
  --duration 2.5 `
  --fps 30 `
  --camera-index 0
```

**Cách 2 — Chọn theo `class_id` (từ `configs/class_list.json`):**
```powershell
python -m src.data.collect_videos --signer-id signer_01 --class-id 0 --num-takes 5
```

**Cách 3 — Chế độ tương tác (nhập `signer_id` và chọn ký hiệu từ menu):**
```powershell
python -m src.data.collect_videos --num-takes 5
```

**Bảng phím tắt trên cửa sổ webcam:**

| Giai đoạn | Phím | Chức năng |
| --- | --- | --- |
| Xem trước (Preview) | `SPACE` hoặc `Enter` | Bắt đầu đếm ngược `3-2-1` và ghi hình |
| Xem trước / Đang quay | `q` hoặc `ESC` | Thoát phiên thu thập an toàn |
| Xem lại (Review) | `s` hoặc `Enter` | **Lưu** clip `.mp4` và tự động ghi 1 dòng vào `data/manifests/self_manifest.csv` |
| Xem lại (Review) | `r` | **Hủy & quay lại** take hiện tại |
| Xem lại (Review) | `q` hoặc `ESC` | Thoát (không lưu take chưa chấp nhận) |

### 8.2. Gộp manifest & Kiểm tra chất lượng dữ liệu (`src/data/build_manifest.py`)

Sau mỗi buổi quay (hoặc sau khi tải dữ liệu Kaggle), chạy lệnh sau để gộp `kaggle_manifest.csv` + `self_manifest.csv`, kiểm tra metadata video bằng OpenCV, loại trùng lặp và xuất báo cáo chất lượng:

```powershell
python -m src.data.build_manifest `
  --config configs/data_config.yaml `
  --output data/manifests/dataset_manifest.csv `
  --report data/manifests/quality_report.json
```

Nếu bạn có video tự quay lưu thủ công vào thư mục nhưng chưa có file `self_manifest.csv`, có thể quét trực tiếp thư mục:
```powershell
python -m src.data.build_manifest `
  --scan-self-dir data/raw/self_recorded `
  --output data/manifests/dataset_manifest.csv `
  --report data/manifests/quality_report.json
```

Kiểm tra nhanh `data/manifests/quality_report.json`:
- Đảm bảo `invalid_samples == 0`, `missing_signer_count == 0`, `missing_label_count == 0`.
- Kiểm tra `num_signers >= 3` và `is_ready_for_split == true`.
- Xem mục `class_imbalance_warnings` để bổ sung thêm lượt quay cho các ký hiệu bị thiếu mẫu hoặc thiếu người quay trước khi chuyển sang **Bước 3: Chia tập theo người (`split_dataset.py`)**.

---

## 9. Chế độ chỉ dùng Dataset Kaggle (Không tự quay — Hỗ trợ Đa bộ dữ liệu)

Trong trường hợp nhóm muốn triển khai nhanh **100% trên dữ liệu công khai Kaggle** mà **không cần tự quay bằng webcam**, Bước 2 của pipeline hỗ trợ kết hợp đồng thời **2 bộ dữ liệu VSL tiếng Việt công khai** trong chế độ `--kaggle-only`:

| Bộ dữ liệu | Nguồn (`source`) | Đặc điểm & độ đa dạng người ký hiệu | Vai trò trong phạm vi 30 lớp (`configs/class_list.json` v1.3) |
| --- | --- | --- | --- |
| **`aresusayhi/vsl-vietnamese-sign-languages`** (QIPEDC v5, Apache 2.0) | `kaggle_vsl` | 4.362 clip / 3.315 nhãn; biến thể vùng miền `B` (Bắc → `kaggle_signer_north`), `T` (Trung → `kaggle_signer_central`), `N` (Nam → `kaggle_signer_south`) | 90 clip chuẩn vùng miền (3 biến thể `B`/`T`/`N` cho mỗi lớp) |
| **`nguyenanfms/vsl-vietnamese-sign-language-v2`** (VSL-Cropped v8 / VSL400 + Online, CC BY 4.0) | `kaggle_vsl_v2` | 26.446 clip góc chính diện 224×224 / 472 nhãn, nhiều người ký hiệu (`vsl2_signer_<id>` từ `front_view.json` của VSL400); tự động **loại trừ** `processed_augmented/` để tránh rò rỉ trước Bước 3 | 1.930 clip đa người ký hiệu (55–93 clip/lớp), tổng kết hợp **2.020 video thật** (58–96 clip/lớp, ≥ 8 `signer_id`) |

### 9.1. Cách Bước 2 hoạt động ở chế độ `--kaggle-only`
1. **Bỏ qua bước tự quay webcam (`collect_videos.py`):** Không cần tạo thư mục `data/raw/self_recorded/` hay file `data/manifests/self_manifest.csv`.
2. **Định danh người thực hiện (`signer_id`) nhất quán & chuẩn hóa nhãn Unicode NFC:**
   - Bộ QIPEDC (`kaggle_vsl`): `kaggle_signer_north` (`B`), `kaggle_signer_central` (`T`), `kaggle_signer_south` (`N`).
   - Bộ VSL-v2 (`kaggle_vsl_v2`): `vsl2_signer_<id>` (đọc từ `front_view.json` của VSL400) hoặc `vsl2_signer_train` / `vsl2_signer_test` khi thiếu metadata gốc; nhãn dạng viết hoa chữ cái đầu (ví dụ `"Giúp đỡ"`) được tự động chuẩn hóa về nhãn chuẩn (`"giúp đỡ"`) theo `configs/class_list.json`.
   - Toàn bộ 30 lớp chuẩn trong `configs/class_list.json` đều đã được kiểm chứng có đủ cả 3 biến thể `["B", "T", "N"]` trên QIPEDC (`kaggle_clip_count >= 3`) và 55–93 clip trên VSL-v2 (`kaggle_v2_clip_count`), đảm bảo phủ **100% cả 3 tập `train` / `val` / `test`** khi chia theo người mà không rò rỉ dữ liệu.
3. **Lọc nguồn & Kiểm tra chất lượng nghiêm ngặt (`build_manifest.py --kaggle-only`):**
   - Tự động gộp cả `data/manifests/kaggle_manifest.csv` (`kaggle_vsl`) và `data/manifests/kaggle_v2_manifest.csv` (`kaggle_vsl_v2`) — hoặc quét `--scan-kaggle-dir` / `--scan-kaggle-v2-dir` (tự động bỏ qua `processed_augmented/`).
   - **Loại bỏ hoàn toàn** mọi dòng có `source == "self_recorded"` ngay cả khi trên máy có sẵn `self_manifest.csv`.
   - Kiểm tra metadata video bằng OpenCV, loại trùng lặp theo `sample_id` và `video_path`.
   - Xuất báo cáo `data/manifests/quality_report.json` với các trường chuyên biệt:
     - `"mode": "kaggle_only"`
     - `"kaggle_sources_present"`: các nguồn Kaggle thực tế có mặt (ví dụ `["kaggle_vsl", "kaggle_vsl_v2"]`)
     - `"classes_with_3plus_signers"`: danh sách các nhãn có từ 3 `signer_id` riêng biệt trở lên
     - `"num_classes_with_3plus_signers"`: tổng số nhãn đạt chuẩn ≥ 3 `signer_id`
     - `"is_ready_for_kaggle_only_split"`: `true` khi các nguồn Kaggle có ≥ 3 `signer_id`, có lớp đạt ≥ 3 `signer_id`, không thiếu `signer_id`/`label` và không lẫn dữ liệu `self_recorded`.

### 9.2. Lệnh chạy thực tế (Chế độ chỉ Kaggle — 1 hoặc 2 bộ dữ liệu)

**Bước A — Tải & chuẩn hóa manifest cả 2 bộ Kaggle (hoặc chọn riêng `qipedc` / `vsl_v2`):**
```powershell
# Tải & tạo cả kaggle_manifest.csv lẫn kaggle_v2_manifest.csv:
python -m src.data.download_kaggle --dataset all --config configs/data_config.yaml

# Hoặc chỉ tải bộ QIPEDC (nhẹ hơn):
python -m src.data.download_kaggle --dataset qipedc --config configs/data_config.yaml
```

**Bước B — Gộp manifest Kaggle & kiểm tra chất lượng ở chế độ chỉ Kaggle:**
```powershell
python -m src.data.build_manifest `
  --config configs/data_config.yaml `
  --kaggle-only `
  --output data/manifests/dataset_manifest.csv `
  --report data/manifests/quality_report.json
```

**Bước C — Hoặc chạy trọn gói 4 bước pipeline ở chế độ `--kaggle-only`:**
```powershell
# Chạy thử nghiệm end-to-end với dữ liệu tổng hợp chuẩn đa nguồn QIPEDC + VSL-v2 (không cần tải dataset lớn):
python scripts/run_quy_pipeline.py --demo --kaggle-only

# Chạy trên dữ liệu Kaggle thật đã tải:
python scripts/run_quy_pipeline.py --kaggle-only --config configs/data_config.yaml
```

> **Mở rộng về sau:** Khi nhóm có thêm video tự quay từ `collect_videos.py`, chỉ cần bỏ cờ `--kaggle-only` — `build_manifest.py` sẽ tự động chuyển về chế độ `hybrid` để gộp đồng thời `kaggle_vsl`, `kaggle_vsl_v2` và `self_recorded` vào chung `dataset_manifest.csv`.

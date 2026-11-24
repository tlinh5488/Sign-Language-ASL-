# PHIẾU TÌM HIỂU VÀ PHÂN CÔNG NHÓM — PHẦN 1: DỮ LIỆU (QUY)

- **Đề tài:** Mô hình nhận diện ngôn ngữ ký hiệu (Ngôn ngữ ký hiệu tay tiếng Việt — VSL)
- **Giai đoạn phụ trách:** Phần 1 — Dữ liệu (Chọn phạm vi → Thu thập & gán nhãn → Chia tập theo người → Tăng cường tập train)
- **Người phụ trách:** Quy (Xuân Quy)
- **Bàn giao cho:** Thiên (Phần 2 — Tiền xử lý MediaPipe Holistic)
- **Khung thời gian dự án:** 2–3 tuần

---

## I. Trả lời 5 câu hỏi tìm hiểu (Phần 1: Dữ liệu)

### Câu hỏi 1: Có những dataset công khai nào (ASL Alphabet, WLASL, MS-ASL, dữ liệu ký hiệu tiếng Việt)? So sánh quy mô, số lớp, giấy phép.

| Dataset | Ngôn ngữ | Loại dữ liệu | Quy mô (số mẫu) | Số lớp | Người ký hiệu (`signer_id`) | Giấy phép | Đánh giá & Quyết định |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **ASL Alphabet** (Kaggle) | Tiếng Anh (ASL) | Ảnh tĩnh bàn tay (200×200) | 87.000 ảnh | 29 (A–Z + del/space/nothing) | 1 người lặp lại | GPLv2 / Public | **Loại:** chỉ là ảnh tĩnh chữ cái tiếng Anh, không có chuyển động thời gian cho LSTM/BiLSTM và không phải tiếng Việt. |
| **WLASL** (Word-Level ASL) | Tiếng Anh (ASL) | Video từ đơn động (`.mp4`) | 21.083 video | 2.000 từ (WLASL100/300/1000/2000) | 119 người | C-UDA (Học thuật) | **Tham khảo kiến trúc, không dùng dữ liệu:** chuẩn benchmark video-to-word nhưng là ngôn ngữ ký hiệu Mỹ (ASL). |
| **MS-ASL** (Microsoft ASL) | Tiếng Anh (ASL) | Video YouTube động (URL) | 25.513 video | 1.000 từ | 222 người | C-UDA (Học thuật) | **Loại:** là ASL và nhiều link YouTube đã bị xóa, không ổn định để tái lập trong 2–3 tuần. |
| **QIPEDC VSL** (`aresusayhi/vsl-vietnamese-sign-languages`, Bộ GD&ĐT) | **Tiếng Việt (VSL)** | Video từ điển động 1280×720 (`.mp4`) | 4.362 clip (**90 clip** trong 30 lớp chọn) | 3.315 từ | **5 diễn viên đã xác minh trực quan** (`qipedc_signer_01`..`05`; đã gom nhóm 26 cặp trùng lặp B/T/N) | **Apache 2.0** | **CHỌN (Bộ 1 — Bổ trợ `train`):** từ điển chuẩn Bộ GD&ĐT Việt Nam, giấy phép mở, tăng đa dạng góc/nền/độ phân giải cho tập huấn luyện. |
| **VSL-v2 Cropped / VSL400** (`nguyenanfms/vsl-vietnamese-sign-language-v2`) | **Tiếng Việt (VSL)** | Video động 224×224 góc chính diện (`.mp4`) | 26.446 clip (**1.930 clip** trong 30 lớp chọn) | 472 từ | **26 người ký hiệu studio VSL400 có `signer_id` gốc** (`vsl2_signer_001`..`028` từ `Part_1..Part_7/front_view.json`) | **CC BY 4.0** | **CHỌN (Bộ 2 — Xương sống chính):** quy mô lớn, đa người ký hiệu thật có ground-truth `signer_id`, góc chính diện 224×224 rất sạch cho MediaPipe Holistic. |
| **ViSignLanguage-Video** (`star092304/ViSignLanguage-Video`, PTIT AI Challenge) | Tiếng Việt (VSL) | Video động 224×224 (`.mp4`, 864 MB) | 3.875 clip | 100 từ | Không có `signer_id` tường minh | CC BY 4.0 | **Dự phòng:** nhẹ nhưng thiếu `signer_id` (khó đảm bảo chia tập chống rò rỉ theo người) và chỉ giao 22 lớp với QIPEDC. |

**Lựa chọn cuối cùng (Phương án 1 — Dual Kaggle VSL):** Kết hợp **QIPEDC VSL (`kaggle_vsl`)** + **VSL-v2 Cropped / VSL400 (`kaggle_vsl_v2`)** trên phần giao 30 lớp → **2.020 video tiếng Việt thật (269.92 MiB)**, **31 người ký hiệu đã xác minh**, **0 clip online chưa xác minh**, **0 clip tiền tăng cường từ nguồn ngoài** (loại bỏ `processed_augmented/`).

---

### Câu hỏi 2: Chọn bao nhiêu lớp và những ký hiệu nào để vừa sức trong 2–3 tuần?

- **Số lượng lớp chốt:** **30 lớp ký hiệu VSL giao tiếp hằng ngày** (`class_id: 0..29`, định nghĩa chi tiết tại `configs/class_list.json` v1.3 kèm mô tả động tác tay `action_description` và số tay `handedness: 1_tay / 2_tay`).
- **Lý do chọn 30 lớp vừa sức trong 2–3 tuần:**
  - Đủ phong phú để chứng minh bài toán phân loại chuỗi đa lớp (30 lớp giao tiếp thực tế: giao tiếp, hỏi đáp, gia đình, học tập, sinh hoạt, hành động, thời gian), đồng thời không quá lớn để mô hình LSTM/BiLSTM của Hiếu có thể huấn luyện nhanh, tinh chỉnh siêu tham số và Thiên phân tích confusion matrix 30×30 rõ ràng.
  - Mọi lớp trong 30 lớp đều được chọn từ **48 nhãn giao nhau tốt nhất giữa cả 2 bộ Kaggle**: mỗi lớp đều có **đủ 3 biến thể vùng QIPEDC (`B`, `T`, `N`)** VÀ **55–93 clip đa người trong VSL-v2** → tổng **58–96 clip gốc/lớp** (trung bình **67.33 clip/lớp**).
  - Dung lượng toàn bộ 2.020 video chỉ **269.92 MiB** (tải nhanh trong vài phút, trích MediaPipe Holistic cho Phần 2 nhanh gọn).
- **Danh sách 30 lớp đã chốt:** `giúp đỡ`, `muốn`, `tiếp tục`, `từ chối`, `địa chỉ`, `bạn`, `chị`, `ông bà`, `anh em`, `em trai`, `thầy giáo`, `cô giáo`, `nhà trường`, `quyển vở`, `bảng đen`, `phở`, `sữa`, `bánh mì`, `buổi sáng`, `buổi tối`, `hỏi`, `hiểu`, `ngồi`, `vui mừng`, `chúc mừng`, `thương yêu`, `tha thứ`, `bao giờ?`, `vì sao?`, `đúng không?` (xem bảng đầy đủ trong `configs/class_list.json`).

---

### Câu hỏi 3: Quay bao nhiêu người, bao nhiêu mẫu mỗi lớp, góc quay, ánh sáng, nền? (Và lý do điều chỉnh thực tế của nhóm)

1. **Tiêu chuẩn kỹ thuật nếu tự quay bổ sung (SOP chi tiết tại `docs/collection_guide.md` & tool `src/data/collect_videos.py`):**
   - **Số người ký hiệu:** tối thiểu **≥ 5 người** (bắt buộc ≥ 3 người để chia được 3 tập tách biệt `train`/`val`/`test`).
   - **Số mẫu mỗi lớp:** tối thiểu **3–5 lượt quay/người/lớp** (tương đương ≥ 15–25 video gốc/lớp).
   - **Góc quay & khung hình (chuẩn MediaPipe Holistic):** camera đặt chính diện ngang tầm ngực–vai, khoảng cách 1.2–1.8 m sao cho luôn lấy trọn **đỉnh đầu, khuôn mặt, toàn bộ thân trên đến ngang hông và biên độ mở rộng tối đa của 2 bàn tay** (không cắt cổ tay/ngón tay); **tắt mirror/lật gương** khi lưu để giữ đúng tay trái/tay phải.
   - **Ánh sáng & nền:** ánh sáng đều từ phía trước (tránh ngược sáng cửa sổ gây đen tay), nền đơn sắc ít vật thể chuyển động (hạn chế màu áo trùng màu da), độ phân giải tối thiểu 640×480 @ 30 FPS, mỗi clip 2.0–3.0 giây theo nhịp *tư thế nghỉ (0.3s) → thực hiện ký hiệu → tư thế nghỉ (0.3s)*.
2. **Điều chỉnh thực tế cho Bước 2 trong dự án 2–3 tuần:**
   - Thay vì tự quay thủ công toàn bộ từ đầu (dễ sai động tác VSL chuẩn do không phải người khiếm thính và khó huy động >25 người trong 1 tuần), nhóm sử dụng **2.020 video thật từ 31 người ký hiệu đã xác minh** của 2 bộ công khai **QIPEDC** (5 diễn viên chuẩn Bộ GD&ĐT) + **VSL-v2/VSL400** (26 người ký hiệu studio góc chính diện 224×224).
   - Công cụ `src/data/collect_videos.py` và SOP `docs/collection_guide.md` vẫn hoàn thiện 100% để quay bổ sung nếu Thiên/Hiếu phản hồi cần thêm dữ liệu webcam thực tế sau đánh giá Phần 3.

---

### Câu hỏi 4: Vì sao phải chia tập theo người (Subject-Independent Split)? Cách thực hiện để tránh rò rỉ dữ liệu?

- **Vì sao bắt buộc chia tập theo người:**
  - Nếu chia ngẫu nhiên theo clip (`random split`), cùng một người ký hiệu (cùng vóc dáng, sải tay, tốc độ ký hiệu, màu áo, hậu cảnh) sẽ xuất hiện ở cả `train` và `val`/`test`. Mô hình LSTM/BiLSTM sẽ **học thuộc đặc trưng nhận dạng người/nền (shortcut learning)** thay vì động tác ký hiệu, dẫn đến F1 trên `test` bị ảo (quá cao) nhưng sụt giảm mạnh khi chạy webcam thực tế ở Phần 4 (Linh) với người dùng mới!
- **Cách thực hiện nghiêm ngặt trong pipeline (`src/data/split_dataset.py` & kiểm toán `signer_verification_report.json`):**
  1. **Xác minh danh tính người ký hiệu trước khi chia:**
     - Ánh xạ đủ 7 file `VSL400/Part_1..Part_7/split_*/front_view.json` cho 1.930 video VSL-v2 (26 `vsl2_signer_*` thật, 0 clip online vô danh).
     - Kiểm tra trực quan face-sheet + pixel MAD cho 90 video QIPEDC (`configs/qipedc_signer_map.json`): phát hiện 5 diễn viên thật (`qipedc_signer_01`..`05`) và gom toàn bộ 26 cặp video trùng lặp B/T/N vào cùng `signer_id`.
  2. **Tách biệt tuyệt đối nhóm `signer_id` (`verify_no_signer_leakage`):**
     - `set(train.signer_id) ∩ set(val.signer_id) == ∅`
     - `set(train.signer_id) ∩ set(test.signer_id) == ∅`
     - `set(val.signer_id) ∩ set(test.signer_id) == ∅`
  3. **Cân bằng nguồn miền giữa `val` và `test` (`stratify_by_source=True`):**
     - Đưa toàn bộ 5 diễn viên QIPEDC (90 clip, chỉ phủ 2–20/30 lớp/diễn viên) vào **`train`**, và chia **26 người studio VSL-v2** tách biệt vào `train` (17 người, 1.327 clip), `val` (4 người, 294 clip, 8–16 clip/lớp), `test` (5 người, 309 clip, 9–14 clip/lớp).
     - Nhờ đó `val` và `test` có **100% cùng phân phối miền** (`kaggle_vsl_v2: 1.0`, `max_val_test_source_ratio_diff: 0.0`), phủ đủ **30/30 lớp** và hoàn toàn sạch rò rỉ (`leakage_check_passed: true`).
  4. **Thứ tự bắt buộc:** Loại bỏ `processed_augmented/` từ nguồn ngoài → Chia tập TRƯỚC (`split_dataset.py` từ chối mọi hàng có `is_augmented=True`) → Tăng cường SAU.

---

### Câu hỏi 5: Các kỹ thuật augmentation phù hợp với video và chuỗi landmark; vì sao KHÔNG augment trên tập `val`/`test`?

1. **Các kỹ thuật augmentation an toàn cho ngôn ngữ ký hiệu VSL:**
   - **Trên video (`src/data/augment.py` — Phần 1):**
     - *Biến đổi tốc độ thời gian (`speed`: 0.85x–1.15x)*: mô phỏng người ký hiệu nhanh/chậm tự nhiên.
     - *Cắt biên thời gian nhẹ (`trim`: tối đa 8% đầu/cuối)*: mô phỏng độ trễ bắt đầu/kết thúc cử chỉ.
     - *Biến đổi Affine không gian nhỏ (`affine`: xoay ±8°, phóng to/thu nhỏ 0.92–1.08, dịch ±4%, `BORDER_REPLICATE` cố định cả clip)*: mô phỏng góc đặt camera và khoảng cách ngồi khác nhau mà không cắt mất bàn tay.
     - *Biến thiên ánh sáng & nhiễu nhẹ (`light`: tương phản 0.88–1.12, độ sáng ±20, nhiễu Gaussian σ ≤ 4)*: tăng độ bền cho bước trích MediaPipe Holistic dưới ánh sáng yếu/mạnh.
     - **Tuyệt đối tắt lật ngang mặc định (`allow_hflip: false`):** lật ngang đảo ngược tay trái ↔ tay phải, làm sai tay thuận và hướng chuyển động của các ký hiệu VSL bất đối xứng.
   - **Trên chuỗi landmark (tham khảo bổ trợ cho Phần 2/3 của Thiên & Hiếu):** nhiễu Gaussian nhỏ lên tọa độ `(x, y, z)` sau chuẩn hóa (σ ~ 0.005–0.01), scale nhẹ ±5%, xoay mặt phẳng 2D nhẹ ±8°, time-warping / dropout ngẫu nhiên 1–2 khung hình (mô phỏng mất nhận diện bàn tay tạm thời).
2. **Vì sao KHÔNG BAO GIỜ augment trên `val` và `test`:**
   - Tập `val` (chọn siêu tham số/early stopping) và `test` (đo chất lượng thực tế cuối cùng) phải phản ánh **đúng phân phối video gốc ngoài đời thực**.
   - Nếu augment trên `val`/`test`, độ đo F1 bị làm méo bởi các biến thể nhân tạo; tệ hơn, nếu augment trước khi chia tập thì bản gốc và bản tăng cường của cùng một video có thể rơi sang hai tập khác nhau gây **rò rỉ dữ liệu nghiêm trọng**.
   - Pipeline bảo vệ cứng bằng `ensure_train_only_manifest` trong `src/data/augment.py`: tự động ném `ValueError("Augmentation is strictly allowed on train split only!")` nếu bất kỳ hàng nào có `split != "train"`.

---

## II. Sản phẩm đầu ra đã hoàn thiện (Checklist bàn giao)

- [x] **Bảng so sánh dataset và lựa chọn cuối cùng:** mục I.1 trên đây và mục 2 trong `README.md`.
- [x] **Danh sách 30 lớp ký hiệu VSL đã chốt:** `configs/class_list.json` (v1.3, đầy đủ `class_id`, `label`, `slug`, `category`, `handedness`, `qipedc_variants`, `kaggle_clip_count`, `kaggle_v2_clip_count`, `action_description`).
- [x] **Thư mục dữ liệu thật đã gán nhãn & chia tập theo người:**
  - `data/raw/` (**2.020** video `.mp4` thật, **269.92 MiB**)
  - `data/manifests/scoped_manifest.csv`, `quality_report.json`, `scope_report.json`, `signer_verification_report.json`
  - `data/splits/train.csv` (**1.417** mẫu, 22 người), `val.csv` (**294** mẫu, 4 người), `test.csv` (**309** mẫu, 5 người), `split_report.json` (`leakage_check_passed: true`, `val_test_domain_matched: true`)
- [x] **Script augmentation chạy CHỈ trên tập `train` (nhất quán 2x):** `src/data/augment.py` → `data/splits/train_augmented.csv` (**4.251** mẫu = **1.417** gốc + **2.834** tăng cường 2x, `allow_hflip: false`) và `data/splits/augment_report.json`.
- [x] **Tổng hợp kết quả sản xuất:** `data/pipeline_summary.json` (`mode: "production_kaggle_only"`).

---

## III. Ghi chú của thành viên (Phần 1 — Quy)

1. **Tổng quan dữ liệu bàn giao cho Thiên (Phần 2 — MediaPipe Holistic):**
   - Đã hoàn tất 100% pipeline sản xuất trên dữ liệu video tiếng Việt thật (`data/raw/`, `data/manifests/`, `data/splits/`, `data/augmented/train/`).
   - Toàn bộ 30/30 lớp đều có mặt đầy đủ ở cả 3 tập (`missing_classes: []`): `train_augmented.csv` có **4.251** video (123–198 clip/lớp, TB 141.70), `val.csv` có **294** video gốc (8–16 clip/lớp, TB 9.80), `test.csv` có **309** video gốc (9–14 clip/lớp, TB 10.30).
2. **Điểm kỹ thuật quan trọng Thiên cần lưu ý khi chạy Phần 2:**
   - Đọc trực tiếp 3 file `data/splits/train_augmented.csv`, `data/splits/val.csv`, `data/splits/test.csv` bằng `src.data.schema.load_manifest_df(..., validate=True, require_split=True)`.
   - Trong `train_augmented.csv` có 2 độ phân giải: `kaggle_vsl` (QIPEDC, 1280×720, 270 clip sau 2x) và `kaggle_vsl_v2` (224×224, 3.981 clip sau 2x). Vì MediaPipe Holistic xuất tọa độ chuẩn hóa trong `[0, 1]` và Thiên chuẩn hóa về gốc cổ tay/tỷ lệ bàn tay nên hoàn toàn bất biến với độ phân giải gốc.
   - `val.csv` và `test.csv` hoàn toàn độc lập người ký hiệu với `train` và cùng 100% nguồn `kaggle_vsl_v2` (224×224 góc chính diện) để Thiên và Hiếu so sánh chỉ số F1 validation vs test công bằng, không bị lệch domain.
3. **Kênh phối hợp ngược lại từ Phần 3 (Thiên & Hiếu → Quy):**
   - Sau khi Hiếu huấn luyện xong BiLSTM và Thiên chạy confusion matrix trên 30 lớp, nếu cặp ký hiệu nào bị nhầm do thiếu góc quay thực tế webcam, Quy sẽ dùng ngay `src/data/collect_videos.py` + `scripts/run_quy_pipeline.py --hybrid` để bổ sung mẫu cho đúng các lớp đó.

# Nhận Diện Ngôn Ngữ Ký Hiệu Tay Tiếng Việt (VSL) — Phần 1: Pipeline Dữ Liệu (Quý)

Pipeline chuẩn hóa **Phần 1: Dữ liệu** cho bài toán nhận diện **Ngôn ngữ ký hiệu tay tiếng Việt (Vietnamese Sign Language — VSL)** sử dụng **2 bộ dữ liệu VSL công khai có xác minh người ký hiệu trên Kaggle** (cấu hình mặc định **Phương án 1: Dual Kaggle-only**, không cần tự quay; vẫn giữ sẵn tool tự quay webcam `src/data/collect_videos.py` để bổ sung khi cần):

1. **Bộ dữ liệu 1 — QIPEDC VSL (`kaggle_vsl`)**: [`aresusayhi/vsl-vietnamese-sign-languages`](https://www.kaggle.com/datasets/aresusayhi/vsl-vietnamese-sign-languages) (v5, Apache 2.0, từ điển chuẩn Bộ GD&ĐT — `qipedc.moet.gov.vn`). Trên 30 lớp phạm vi: **90 video `.mp4` thật (1280×720, 47.13 MiB)** từ **5 diễn viên ký hiệu đã xác minh trực quan** (`qipedc_signer_01`..`qipedc_signer_05` trong `configs/qipedc_signer_map.json`).
2. **Bộ dữ liệu 2 — VSL-v2 Cropped / VSL400 (`kaggle_vsl_v2`)**: [`nguyenanfms/vsl-vietnamese-sign-language-v2`](https://www.kaggle.com/datasets/nguyenanfms/vsl-vietnamese-sign-language-v2) (v8, **CC BY 4.0**, 472 từ vựng, 26.446 clip góc chính diện 224×224 chuẩn MediaPipe Holistic). Trên 30 lớp phạm vi: **1.930 video `.mp4` thật (224×224, 222.79 MiB)** từ **26 người ký hiệu studio VSL400 có ground-truth `signer_id`** ánh xạ đầy đủ qua cả 7 phần `Part_1/split_1`..`Part_7/split_7/front_view.json` (0 clip online chưa xác minh).

**Tổng cộng dữ liệu thực tế trên đĩa (`data/raw/`):** **2.020 video `.mp4` hợp lệ** (**269.92 MiB** video thô; **278.23 MiB** cả metadata), **30/30 lớp VSL** (**58–96 clip/lớp**, trung bình **67.33 clip/lớp**), **31 người ký hiệu đã xác minh danh tính**, chia tập tách biệt người ký hiệu và đồng nhất miền dữ liệu `val`/`test` với **0% rò rỉ**, sẵn sàng bàn giao cho **Phần 2: Tiền xử lý MediaPipe Holistic (Thiên)**.

> Tài liệu trả lời đầy đủ 5 câu hỏi nghiên cứu và *Ghi chú của thành viên* theo **PHIẾU TÌM HIỂU VÀ PHÂN CÔNG NHÓM** xem tại: [docs/phieu_tim_hieu_phan_1_quy.md](docs/phieu_tim_hieu_phan_1_quy.md) (bản Word: [docs/Phieu_Tim_Hieu_Phan_1_Quy.docx](docs/Phieu_Tim_Hieu_Phan_1_Quy.docx)).

---

## 1. Tổng quan kết quả thực tế 4 bước của Phần 1 (Dữ liệu)

| Bước | Tên bước | Module chính | Kết quả thực tế trong `data/` |
| :--- | :--- | :--- | :--- |
| **Bước 1** | **Chọn phạm vi** (Scope selection) | `configs/class_list.json`, `src/data/scope.py` | `data/manifests/scoped_manifest.csv`: **2.020** video hợp lệ, **30/30** lớp (`missing_classes: []`), **31** người ký hiệu |
| **Bước 2** | **Thu thập, xác minh người ký hiệu & kiểm định** | `src/data/download_kaggle.py`, `src/data/build_manifest.py`, `configs/qipedc_signer_map.json` | `data/manifests/dataset_manifest.csv`, `quality_report.json`, `signer_verification_report.json` (**2.020/2.020** valid, 0 corrupt, 0 unverified) |
| **Bước 3** | **Chia tập theo người & cân bằng nguồn** (Signer-disjoint, Source-stratified) | `src/data/split_dataset.py` | `train.csv`: **1.417** (70.15%, 22 người, 41–66/lớp) · `val.csv`: **294** (14.55%, 4 người, 8–16/lớp) · `test.csv`: **309** (15.30%, 5 người, 9–14/lớp) · `leakage_check_passed: true` · `val_test_domain_matched: true` |
| **Bước 4** | **Tăng cường dữ liệu — CHỈ trên tập train (2x)** | `src/data/augment.py` | `data/splits/train_augmented.csv`: **4.251** hàng (**1.417** gốc + **2.834** tăng cường 2x, 123–198/lớp, trung bình 141.70/lớp, `allow_hflip: false`) · `val.csv`/`test.csv` giữ nguyên 100% |

---

## 2. Bảng so sánh các bộ dữ liệu công khai & Lựa chọn cuối cùng

| Dataset | Ngôn ngữ | Dạng dữ liệu | Quy mô (mẫu) | Số lớp | Người ký hiệu (`signer_id`) | Giấy phép | Đánh giá mức độ phù hợp với đề tài VSL |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **ASL Alphabet** (Kaggle) | Tiếng Anh (ASL) | Ảnh tĩnh bàn tay (200×200) | 87.000 ảnh | 29 (A–Z + 3 lệnh) | 1 người quay lặp | GPLv2 / Public | **Không phù hợp cho từ động VSL**: chỉ là chữ cái tĩnh tiếng Anh, không có chuỗi chuyển động thời gian cho LSTM/BiLSTM |
| **WLASL** (Word-Level ASL) | Tiếng Anh (ASL) | Video từ đơn động (`.mp4`) | 21.083 clip | 2.000 từ (subsets 100/300/1000/2000) | 119 người (nhiều nguồn web) | C-UDA (Học thuật) | **Tham khảo kiến trúc, không đúng ngôn ngữ**: chất lượng benchmark tốt cho LSTM/BiLSTM nhưng là ký hiệu Mỹ (ASL), không dùng cho tiếng Việt |
| **MS-ASL** (Microsoft ASL) | Tiếng Anh (ASL) | Video YouTube động (URL) | 25.513 clip | 1.000 từ | 222 người | C-UDA (Học thuật) | **Không phù hợp**: là ASL và nhiều link YouTube gốc đã chết theo thời gian, khó tái lập trọn vẹn trong 2–3 tuần |
| **QIPEDC VSL** (`aresusayhi/vsl-vietnamese-sign-languages`) | **Tiếng Việt (VSL)** | Video từ điển động 1280×720 (`.mp4`) | 4.362 clip (90 clip cho 30 lớp giao nhau) | 3.315 từ | **5 diễn viên đã xác minh trực quan** (`qipedc_signer_01..05`; 26 cặp trùng lặp B/T/N đã gom nhóm) | **Apache 2.0** | **Chọn (Bộ 1 — Bổ trợ tập Train)**: chuẩn từ điển Bộ GD&ĐT Việt Nam, độ chính xác ngôn ngữ cao; dùng toàn bộ 90 clip (5 diễn viên) vào `train` để tăng đa dạng góc/nền mà không lệch phân phối `val`/`test` |
| **VSL-v2 Cropped / VSL400** (`nguyenanfms/vsl-vietnamese-sign-language-v2`) | **Tiếng Việt (VSL)** | Video động 224×224 góc chính diện (`.mp4`) | 26.446 clip (**1.930 clip** cho 30 lớp giao nhau) | 472 từ | **26 người ký hiệu studio VSL400 có ground-truth `signer_id`** (`vsl2_signer_001`..`028`, 100% phủ đủ 30/30 lớp) | **CC BY 4.0** | **Chọn (Bộ 2 — Xương sống chính)**: quy mô lớn, đa người ký hiệu thật có `front_view.json` chuẩn xác, 224×224 tối ưu cho MediaPipe Holistic; loại bỏ `processed_augmented/` để chống rò rỉ |
| **ViSignLanguage-Video** (`star092304/ViSignLanguage-Video`, PTIT) | Tiếng Việt (VSL) | Video động 224×224 (`.mp4`, 864 MB) | 3.875 clip | 100 từ | Không có `signer_id` tường minh | CC BY 4.0 | **Dự phòng (Không chọn làm chính)**: tải nhẹ nhưng chỉ giao 22 lớp với QIPEDC, thiếu nhãn `signer_id` nên không đảm bảo chia tập chống rò rỉ theo người |

### Lý do điều chỉnh Bước 2: Dùng 2 bộ dữ liệu VSL công khai có xác minh thay cho tự quay webcam thủ công

Theo kế hoạch sơ bộ trong phiếu phân công, Bước 2 dự kiến tự quay webcam nhiều người. Sau khi khảo sát thực tế trong Giai đoạn tìm hiểu (Ngày 1–3), nhóm thống nhất **điều chỉnh Bước 2 sang kết hợp 2 bộ dataset VSL công khai đã xác minh người ký hiệu (`QIPEDC` + `VSL-v2/VSL400`)** dựa trên 4 lý do kỹ thuật rõ ràng:

1. **Khung thời gian dự án 2–3 tuần**: Để đạt quy mô tương đương **2.020 video** trên **30 lớp** và **31 người ký hiệu**, tự quay thủ công đòi hỏi huy động >25 tình nguyện viên, tổ chức phòng quay, gán nhãn và kiểm duyệt lại mất ít nhất 2–3 tuần — làm trễ toàn bộ Phần 2 (Thiên), Phần 3 (Hiếu) và Phần 4 (Linh).
2. **Độ chuẩn xác ngôn ngữ ký hiệu (Linguistic Validity)**: Các thành viên không phải người khiếm thính hay thông dịch viên VSL chuyên nghiệp; tự học rồi tự quay rất dễ sai hình bàn tay (handshape), vị trí đặt tay hoặc sắc thái biểu cảm. Ngược lại, **QIPEDC** do Bộ GD&ĐT chuẩn hóa và **VSL400** được ghi hình bài bản với người ký hiệu chuẩn.
3. **Quy mô và độ đa dạng vượt trội cho chia tập theo người**: 31 người ký hiệu xác minh (26 người studio VSL-v2 phủ trọn 30/30 lớp + 5 diễn viên QIPEDC) cho phép tập `val` có **294 video (4 người, 8–16 clip/lớp)** và tập `test` có **309 video (5 người, 9–14 clip/lớp)** — đảm bảo ước lượng F1-score và confusion matrix đáng tin cậy về mặt thống kê.
4. **Vẫn sẵn sàng mở rộng tự quay bổ sung**: Bộ công cụ `src/data/collect_videos.py` và hướng dẫn [docs/collection_guide.md](docs/collection_guide.md) vẫn được giữ đầy đủ và kiểm thử sẵn sàng — nếu Thiên/Hiếu phân tích lỗi ở Phần 3 phát hiện lớp nào cần thêm mẫu thực tế webcam, nhóm chỉ cần quay bổ sung và chạy lại với cờ `--hybrid`.

---

## 3. Kiểm toán xác minh người ký hiệu & Chống rò rỉ dữ liệu

Kết quả kiểm toán được lưu đầy đủ tại `data/manifests/signer_verification_report.json`:

1. **Xác minh trực quan 90 video QIPEDC (`configs/qipedc_signer_map.json`)**:
   - Kiểm tra trực quan ghép ảnh khuôn mặt/trang phục + đo sai khác pixel khung giữa (`mid_frame_mad`) chứng minh **hậu tố vùng `B`/`T`/`N` KHÔNG phải 3 người khác nhau**: thực tế chỉ có **5 diễn viên** (`qipedc_signer_01`: 32 clip / 20 lớp; `qipedc_signer_02`: 16 clip / 10 lớp; `qipedc_signer_03`: 33 clip / 20 lớp; `qipedc_signer_04`: 5 clip / 4 lớp; `qipedc_signer_05`: 4 clip / 2 lớp).
   - Phát hiện **26 cặp video gần như giống hệt nhau** giữa các vùng `B`/`T`/`N` (chỉ khác watermark góc, `mid_frame_mad < 1.5`). Toàn bộ 26 cặp này đã được gán **cùng một `qipedc_signer_0X`** (`all_duplicates_grouped_under_same_signer: true`).
2. **Xác minh ground-truth 1.930 video VSL-v2 (`Part_1..Part_7/split_*/front_view.json`)**:
   - Đọc đầy đủ 24.753 bản ghi từ tất cả 7 phần `VSL400`; **100% (1.930/1.930)** video thuộc 30 lớp đều có `signer_id` gốc chính xác (**26 người ký hiệu studio** `vsl2_signer_001`..`028`, mỗi người phủ đủ **30/30 lớp**; 0 clip online không rõ danh tính).
   - Parser loại trừ hoàn toàn thư mục `processed_augmented/` (`exclude_pre_augmented: true`).
3. **Chiến lược chia tập đồng nhất miền `val`/`test` (`src/data/split_dataset.py`)**:
   - Vì 5 diễn viên QIPEDC chỉ phủ một phần lớp (2–20/30 lớp) và có 26 cặp trùng lặp, toàn bộ **5 diễn viên QIPEDC (90 clip)** được đưa vào **`train`**, còn **26 người ký hiệu studio VSL-v2 (1.930 clip)** được chia tách biệt người vào `train` (17 người, 1.327 clip), `val` (4 người, 294 clip), và `test` (5 người, 309 clip).
   - Kết quả: **`val` và `test` có phân phối nguồn giống hệt 100%** (`kaggle_vsl_v2: 1.0`, `max_val_test_source_ratio_diff: 0.0`, `val_test_domain_matched: true`), **0% rò rỉ người ký hiệu** (`train ∩ val = train ∩ test = val ∩ test = ∅`).
4. **Tăng cường dữ liệu nhất quán 2x — CHỈ trên tập `train` (`src/data/augment.py`)**:
   - Tham số nhân bản được thống nhất **100% là `2x`** trên toàn hệ thống (`configs/data_config.yaml`: `num_augments_per_sample: 2` & `kaggle_only_num_augments_per_sample: 2`; `split_report.json`: `recommended_train_aug_copies: 2`; `augment_report.json`: `num_augmentations_per_video: 2`).
   - Từ **1.417** video train gốc sinh thêm **2.834** video tăng cường → tổng **4.251** mẫu train (**123–198 mẫu/lớp**, trung bình **141.70 mẫu/lớp**).
   - Chốt chặn cứng `ensure_train_only_manifest` lập tức ném `ValueError("Augmentation is strictly allowed on train split only!")` nếu bất kỳ hàng nào có `split != "train"`. `allow_hflip: false` mặc định để không đảo tay thuận của ký hiệu VSL bất đối xứng.

---

## 4. Cài đặt môi trường & Chạy Pipeline (Production)

### 4.1. Cài đặt Python 3.10+ và xác thực Kaggle

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Đặt `kaggle.json` tại `C:\Users\<USER>\.kaggle\kaggle.json` (hoặc đặt biến môi trường `$env:KAGGLE_USERNAME`, `$env:KAGGLE_KEY`).

### 4.2. Tải video thật theo phạm vi 30 lớp & Chạy trọn gói 4 bước

```powershell
# Tải 2.020 video thật đúng phạm vi 30 lớp (269.92 MiB) & lập chỉ mục cả 2 bộ Kaggle
python -m src.data.download_kaggle --dataset all --sync-videos --workers 12 --config configs/data_config.yaml

# Chạy trọn gói 4 bước trên dữ liệu thật (mặc định Phương án 1: Dual Kaggle-only)
python scripts/run_quy_pipeline.py --config configs/data_config.yaml
```

Hoặc chạy từng bước riêng lẻ:

```powershell
python -m src.data.build_manifest --config configs/data_config.yaml --kaggle-only
python -m src.data.scope --config configs/data_config.yaml
python -m src.data.split_dataset --config configs/data_config.yaml --kaggle-only
python -m src.data.augment --config configs/data_config.yaml --kaggle-only --workers 8
```

Tùy chọn bổ sung tự quay webcam (chế độ Hybrid):
```powershell
python -m src.data.collect_videos --signer-id signer_01 --class-id 0 --num-takes 5 --duration 2.5 --fps 30
python scripts/run_quy_pipeline.py --hybrid --config configs/data_config.yaml
```

---

## 5. Cấu trúc thư mục dự án

```text
Nhan dien ki hieu/
├── configs/
│   ├── data_config.yaml            # Cấu hình trung tâm (QIPEDC + VSL-v2, scope, split, augment 2x)
│   ├── class_list.json             # 30 lớp VSL giao nhau cả 2 bộ Kaggle (v1.3) + mô tả động tác tay
│   ├── qipedc_signer_map.json      # Bảng ánh xạ 90 clip QIPEDC -> 5 diễn viên xác minh trực quan
│   └── preprocess_config.yaml      # Cấu hình Giai đoạn 2 (Thiên — MediaPipe Holistic)
├── docs/
│   ├── phieu_tim_hieu_phan_1_quy.md  # Phiếu tìm hiểu Phần 1 (5 câu hỏi + Ghi chú của Quy)
│   ├── Phieu_Tim_Hieu_Phan_1_Quy.docx # Bản Word chuẩn nộp phiếu tìm hiểu Phần 1 (Quy)
│   ├── phieu_tim_hieu_phan_2_thien.md # Phiếu tìm hiểu Phần 2 (5 câu hỏi + Ghi chú của Thiên)
│   ├── Phieu_Tim_Hieu_Phan_2_Thien.docx # Bản Word chuẩn nộp phiếu tìm hiểu Phần 2 (Thiên)
│   ├── collection_guide.md           # SOP quay webcam đa người & chế độ Dual Kaggle-only
│   └── preprocess_guide.md           # Tài liệu kỹ thuật Giai đoạn 2 (Thiên — MediaPipe Holistic)
├── scripts/
│   ├── run_quy_pipeline.py           # CLI chạy trọn gói Phần 1 trên dữ liệu sản xuất thật
│   └── run_thien_preprocess.py       # CLI chạy Phần 2 (Thiên) trên dữ liệu sản xuất thật
├── src/
│   ├── data/                         # Phần 1 (Quý): schema, download_kaggle, scope, collect_videos, build_manifest, split_dataset, augment
│   └── preprocess/                   # Phần 2 (Thiên): landmark_schema, extract_landmarks, normalize, sequence, build_features
├── tests/
│   ├── test_quy_pipeline.py          # Bộ kiểm thử tự động pytest Phần 1
│   └── test_thien_preprocess.py      # Bộ kiểm thử tự động pytest Phần 2
├── data/
│   ├── raw/                          # 2.020 video thật (269.92 MiB .mp4)
│   ├── manifests/                    # Manifests & báo cáo (scoped_manifest.csv, quality/scope/signer_verification reports)
│   ├── splits/                       # train.csv (1.417), val.csv (294), test.csv (309), train_augmented.csv (4.251), split/augment reports
│   ├── augmented/train/              # 2.834 video tăng cường tập train
│   ├── landmarks/                    # Landmark MediaPipe Holistic (.npz per-video + *_landmarks_index.csv cho 4.854 video)
│   ├── features/                     # train.npz (4251,48,207), val.npz (294,48,207), test.npz (309,48,207), feature_meta.json, preprocess_summary.json
│   └── pipeline_summary.json         # Tổng hợp kết quả sản xuất Phần 1 (mode: "production_kaggle_only")
├── requirements.txt
└── README.md
```

---

## 6. Đặc tả bàn giao cho Phần 2 — Tiền xử lý MediaPipe Holistic (Thiên)

### 6.1. Bảng số liệu các tập bàn giao chính thức

| Tệp bàn giao | Số mẫu | Tỷ lệ | Số người ký hiệu | Số mẫu/lớp (min–max / TB) | Phân bố nguồn | Ghi chú cho Thiên |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| `data/splits/train.csv` | **1.417** | 70.15% | **22** (5 QIPEDC + 17 VSL-v2) | 41–66 (TB 47.23) | `kaggle_vsl`: 90 · `kaggle_vsl_v2`: 1.327 | Video gốc tập train |
| `data/splits/train_augmented.csv` | **4.251** | — | **22** (giống `train.csv`) | 123–198 (TB 141.70) | `kaggle_vsl`: 270 · `kaggle_vsl_v2`: 3.981 | **Đầu vào huấn luyện chính** (1.417 gốc + 2.834 tăng cường 2x) |
| `data/splits/val.csv` | **294** | 14.55% | **4** (`vsl2_signer_006, 013, 020, 025`) | 8–16 (TB 9.80) | `kaggle_vsl_v2`: 294 (100%) | Video gốc hoàn toàn mới người |
| `data/splits/test.csv` | **309** | 15.30% | **5** (`vsl2_signer_007, 016, 023, 027, 028`) | 9–14 (TB 10.30) | `kaggle_vsl_v2`: 309 (100%) | Video gốc kiểm thử cuối cùng, đồng nhất miền với `val` |

### 6.2. Cấu trúc 16 cột chuẩn (`MANIFEST_COLUMNS` trong `src/data/schema.py`)

Mọi file CSV đều lưu UTF-8 with BOM (`utf-8-sig`) gồm đúng 16 cột:
`sample_id`, `video_path`, `label`, `class_id` (`0..29`), `signer_id`, `source` (`kaggle_vsl` | `kaggle_vsl_v2` | `self_recorded`), `split` (`train` | `val` | `test`), `is_augmented` (`bool`), `parent_sample_id`, `aug_type`, `fps`, `num_frames`, `duration_sec`, `width`, `height`, `is_valid`.

- Đọc manifest bằng `src.data.schema.load_manifest_df(path, validate=True, require_split=True)`.
- Trích xuất landmark độc lập từng split (`train_augmented.csv`, `val.csv`, `test.csv`), không fit scaler trên `val`/`test`.

---

## 7. Giai đoạn 2 — Tiền xử lý MediaPipe Holistic & Đặc trưng (Thiên)

Giai đoạn 2 đã chạy trích xuất **MediaPipe Holistic thật** và chuẩn hóa chuỗi đặc trưng trên toàn bộ **4.854 / 4.854 video sản xuất** (`0` lỗi, `0` video mất tay, `mode: "production"` trong `data/features/preprocess_summary.json`), sẵn sàng bàn giao cho **Hiếu (Phần 3 — LSTM/BiLSTM)** và **Linh (Phần 4 — Triển khai webcam/ONNX)**.

> Tài liệu phiếu tìm hiểu đầy đủ 5 câu hỏi nghiên cứu và *Ghi chú của thành viên (Phần 2 — Thiên)* xem tại: [docs/phieu_tim_hieu_phan_2_thien.md](docs/phieu_tim_hieu_phan_2_thien.md) (bản Word: [docs/Phieu_Tim_Hieu_Phan_2_Thien.docx](docs/Phieu_Tim_Hieu_Phan_2_Thien.docx)) và hướng dẫn kỹ thuật tại [docs/preprocess_guide.md](docs/preprocess_guide.md).

### 7.1. Kết quả thực tế trích xuất landmark & đóng gói đặc trưng (`data/features/`)

| Tập dữ liệu | Tệp đóng gói (`.npz`) | Shape `X` (`float32`) | Shape `mask` (`bool`) | Shape `y` (`int64`) | Số người / Lớp | `zero_hand` | Tỷ lệ nhận diện tay (`hand`) | Tỷ lệ `pose` | Tỷ lệ `face` | Dung lượng nén |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train`** | `data/features/train.npz` | **`(4251, 48, 207)`** | `(4251, 48)` | `(4251,)` (`0..29`) | 22 người · 30/30 lớp | **0** | **93.67%** (`0.9367`) | **100.0%** | **83.34%** | **140.44 MiB** (Git LFS) |
| **`val`** | `data/features/val.npz` | **`(294, 48, 207)`** | `(294, 48)` | `(294,)` (`0..29`) | 4 người · 30/30 lớp | **0** | **94.55%** (`0.9455`) | **100.0%** | **76.40%** | **9.82 MiB** |
| **`test`** | `data/features/test.npz` | **`(309, 48, 207)`** | `(309, 48)` | `(309,)` (`0..29`) | 5 người · 30/30 lớp | **0** | **90.13%** (`0.9013`) | **100.0%** | **62.93%** | **9.94 MiB** |
| **Tổng / Trung bình** | **3 tệp `.npz` nén** | **`(4854, 48, 207)`** | `(4854, 48)` | `(4854,)` | **31 người (0 rò rỉ) · 30/30 lớp** | **0** | **93.50%** (`0.9350`) | **100.0%** | **81.62%** | **160.20 MiB** |

### 7.2. Chạy lại Giai đoạn 2 & Cách nạp đặc trưng cho Phần 3 (Hiếu) / Phần 4 (Linh)

```powershell
# Chạy trọn gói Giai đoạn 2 (MediaPipe Holistic -> chuẩn hóa -> cố định 48 khung hình -> .npy/.npz)
python scripts/run_thien_preprocess.py --config configs/preprocess_config.yaml
```

```python
import numpy as np

# Phần 3 (Hiếu): Nạp trực tiếp tensor đặc trưng (N, L=48, D=207) cho LSTM / BiLSTM
train_bundle = np.load("data/features/train.npz", allow_pickle=False)
X_train, mask_train, y_train = train_bundle["X"], train_bundle["mask"], train_bundle["y"]
# X_train.shape == (4251, 48, 207) float32 | y_train in 0..29 int64 | 0 NaN/Inf

# Phần 4 (Linh): Tái sử dụng đúng hàm chuẩn hóa của Thiên để tránh lệch train/inference
from src.preprocess.normalize import normalize_frame
```

---

## 8. Kiểm thử tự động (Automated Tests)

```powershell
pytest -v
```

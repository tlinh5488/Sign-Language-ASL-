# Nhận Diện Ngôn Ngữ Ký Hiệu Tay Tiếng Việt (VSL) — Phần 1: Pipeline Dữ Liệu (Quý)

Pipeline chuẩn hóa **Phần 1: Dữ liệu** cho bài toán nhận diện **Ngôn ngữ ký hiệu tay tiếng Việt (Vietnamese Sign Language — VSL)** kết hợp **2 bộ dữ liệu công khai chuẩn trên Kaggle** (và tùy chọn bổ sung video tự quay):

1. **Bộ dữ liệu 1 — QIPEDC (`kaggle_vsl`)**: [`aresusayhi/vsl-vietnamese-sign-languages`](https://www.kaggle.com/datasets/aresusayhi/vsl-vietnamese-sign-languages) (v5, Apache 2.0, từ điển chuẩn Bộ GD&ĐT — `qipedc.moet.gov.vn`, 4.362 video, 3 biến thể vùng miền Bắc `B` / Trung `T` / Nam `N`).
2. **Bộ dữ liệu 2 — VSL-v2 Cropped / VSL400 (`kaggle_vsl_v2`)**: [`nguyenanfms/vsl-vietnamese-sign-language-v2`](https://www.kaggle.com/datasets/nguyenanfms/vsl-vietnamese-sign-language-v2) (v8, **CC BY 4.0**, 472 từ vựng, **26.446 video `.mp4` 224×224 góc chính diện chuẩn MediaPipe Holistic**, đa người ký hiệu studio VSL400 có `signer_id` tường minh).

Khi kết hợp cả 2 bộ trên **30 lớp giao tiếp hằng ngày chuẩn (`configs/class_list.json` v1.3)**, nhóm đạt ngay **2.020 video VSL tiếng Việt thật** (**90 QIPEDC + 1.930 VSL-v2**, tức **58–96 clip/lớp** trên **8+ người ký hiệu tách biệt**) mà **KHÔNG cần tự quay video**, đồng thời đảm bảo **KHÔNG rò rỉ dữ liệu theo người (zero signer leakage)** và bàn giao trực tiếp cho **Phần 2: Tiền xử lý MediaPipe Holistic (Thiên)**.

---

## 1. Tổng quan 4 bước của Phần 1 (Dữ liệu)

| Bước | Tên bước | Module chính | Đầu ra tiêu chuẩn |
| :--- | :--- | :--- | :--- |
| **Bước 1** | **Chọn phạm vi** (Scope selection) | `configs/class_list.json`, `src/data/scope.py` | `data/manifests/scoped_manifest.csv`, `data/manifests/scope_report.json` |
| **Bước 2** | **Thu thập & gán nhãn** (2 bộ Kaggle VSL và/hoặc tự quay) & Kiểm định | `src/data/download_kaggle.py`, `src/data/collect_videos.py`, `src/data/build_manifest.py` | `data/manifests/kaggle_manifest.csv`, `data/manifests/kaggle_v2_manifest.csv`, `data/manifests/dataset_manifest.csv`, `data/manifests/quality_report.json` |
| **Bước 3** | **Chia tập dữ liệu — Chia theo người** (Signer-disjoint split) | `src/data/split_dataset.py` | `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv`, `data/splits/split_report.json` |
| **Bước 4** | **Tăng cường dữ liệu — CHỈ trên tập train** | `src/data/augment.py` | `data/augmented/train/<class_slug>/*.mp4`, `data/splits/train_augmented.csv`, `data/splits/augment_report.json` |

---

## 2. Nguồn dữ liệu VSL công khai & So sánh

### 2.1. Hai bộ dữ liệu chính được tích hợp sẵn (`configs/data_config.yaml`)

| Thuộc tính | Dataset 1: QIPEDC (`kaggle_vsl`) | Dataset 2: VSL-v2 Cropped (`kaggle_vsl_v2`) |
| :--- | :--- | :--- |
| **Slug Kaggle** | `aresusayhi/vsl-vietnamese-sign-languages` | `nguyenanfms/vsl-vietnamese-sign-language-v2` |
| **Giấy phép** | Apache 2.0 | CC BY 4.0 |
| **Quy mô toàn bộ** | ~20.77 GB (4.362 clip, 3.315 nhãn) | ~82.37 GB full / ~10 GB `processed` (26.446 clip, 472 nhãn) |
| **Độ phân giải / Góc quay** | Video từ điển QIPEDC `.mp4` | 224×224 cắt gọn góc chính diện (`front_view`), sẵn sàng cho MediaPipe |
| **Định danh người ký hiệu** | Suy luận từ hậu tố vùng: `B` (`kaggle_signer_north`), `T` (`kaggle_signer_central`), `N` (`kaggle_signer_south`) | Ánh xạ `video_id → vsl2_signer_<id>` từ `raw/raw/VSL400/Part_*/split_*/front_view.json` (ví dụ `001`, `002`, `003`, `004`, `006`) |
| **Chống rò rỉ dữ liệu** | Chỉ lấy clip gốc từ `Dataset/Videos/` | Tự động **bỏ qua `processed_augmented/`** (`exclude_pre_augmented: true`) để không rò rỉ trước Bước 3 |
| **Đóng góp trên 30 lớp chuẩn** | **90 clip** (đủ 3 vùng `B`/`T`/`N` cho 30/30 lớp) | **1.930 clip** (55–93 clip/lớp đa người ký hiệu) |

- **Tổng hợp giao nhau giữa 2 bộ**: 151 nhãn giao nhau, trong đó 48 nhãn có đồng thời `>= 3` vùng QIPEDC và 55–93 clip đa người trong VSL-v2. Danh mục **`configs/class_list.json` (v1.3)** chọn 30 lớp giao tiếp hằng ngày tốt nhất (`giúp đỡ`, `muốn`, `tiếp tục`, `từ chối`, `địa chỉ`, `bạn`, `buổi sáng`, ...) → **2.020 video thật trên 8+ người ký hiệu**.

### 2.2. Các bộ dữ liệu VSL công khai dự phòng (Runner-ups để tham khảo)

| Bộ dữ liệu | Nền tảng & Dung lượng | Ưu điểm | Hạn chế so với VSL-v2 |
| :--- | :--- | :--- | :--- |
| **`star092304/ViSignLanguage-Video`** (mirror `namwu/ViSignLanguage-Video`) | HuggingFace (CC BY 4.0, **864 MB**, 3.875 clip, 100 lớp, PTIT AI Challenge) | Tải siêu nhẹ, không cần Kaggle API key, `.mp4` 224×224 | Chỉ giao 22 lớp với QIPEDC (chỉ 2 lớp có đủ 3 vùng B/T/N), không có `signer_id` tường minh, một số nhãn bị lỗi mã hóa NFD |
| **`bowboochua9/vsl400`** | Kaggle (72.2 GB, 400 lớp, 1080p 3 góc `front`/`left`/`right`) | Độ phân giải Full HD 1080p, có `signer_id` tường minh | Đã được nhúng sẵn bên trong `nguyenanfms/vsl-vietnamese-sign-language-v2` (VSL-v2 còn bổ sung thêm 72 từ vựng + bản cắt gọn 224×224 + giấy phép CC BY 4.0) |

---

## 3. Cấu hình mặc định: Phương án 1 (Dual Kaggle-only — Không cần tự quay)

Dự án đã khóa **Phương án 1 (`scope.kaggle_only: true`, `download_kaggle --dataset all`)** làm cấu hình **MẶC ĐỊNH**: chạy trực tiếp `scripts/run_quy_pipeline.py` sẽ vận hành trên **dữ liệu thật** trong `data/manifests/` và `data/splits/` từ 2 bộ Kaggle (`kaggle_vsl` + `kaggle_vsl_v2`) với 0 video tự quay. Nếu muốn bổ sung video tự quay, chỉ cần thêm cờ `--hybrid` (hoặc `--include-self`).

### Lệnh chạy nhanh (Production)

```powershell
# 1. Chạy Production trọn gói 4 bước trên dữ liệu đã lập chỉ mục trong data/manifests/ hoặc data/raw/
python scripts/run_quy_pipeline.py --config configs/data_config.yaml

# 2. Tải & lập chỉ mục CẢ 2 bộ Kaggle rồi chạy trọn gói 4 bước (mặc định không tự quay)
python scripts/run_quy_pipeline.py --download-all-kaggle --config configs/data_config.yaml

# 3. Chạy Production với 2 thư mục Kaggle đã tải sẵn trên máy
python scripts/run_quy_pipeline.py `
  --kaggle-local-dir "D:\datasets\vsl-qipedc" `
  --kaggle-v2-local-dir "D:\datasets\vsl-v2" `
  --config configs/data_config.yaml
```

### Chạy từng bước riêng lẻ (Mặc định Phương án 1)

```powershell
# Bước 2a: Tải & lập chỉ mục mặc định CẢ 2 bộ Kaggle (--dataset mặc định = "all")
python -m src.data.download_kaggle --config configs/data_config.yaml

# Bước 2b: Hợp nhất kaggle_manifest.csv + kaggle_v2_manifest.csv (bỏ qua self_recorded) & kiểm định
python -m src.data.build_manifest --config configs/data_config.yaml --kaggle-only

# Bước 1: Lọc phạm vi 30 lớp chuẩn (mặc định scope.kaggle_only: true -> >= 3 mẫu & >= 3 người/lớp)
python -m src.data.scope --config configs/data_config.yaml

# Bước 3: Chia tập tách biệt tuyệt đối theo signer_id (zero signer leakage)
python -m src.data.split_dataset --config configs/data_config.yaml --kaggle-only

# Bước 4: Tăng cường CHỈ trên tập train (mặc định 4x trong --kaggle-only, allow_hflip=False)
python -m src.data.augment --config configs/data_config.yaml --kaggle-only
```

---

## 4. Cài đặt môi trường & thiết lập Kaggle

### 4.1. Cài đặt thư viện Python (Python 3.10+)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4.2. Cấu hình xác thực Kaggle

Chọn **1 trong 2 cách**:
1. Đặt file `kaggle.json` (tải từ *Kaggle → Settings → Create New Token*) vào `C:\Users\<USER>\.kaggle\kaggle.json`.
2. Hoặc đặt biến môi trường PowerShell:
   ```powershell
   $env:KAGGLE_USERNAME = "ten_tai_khoan_kaggle"
   $env:KAGGLE_KEY = "api_key_cua_ban"
   ```

Kiểm tra nhanh metadata nhãn (tải siêu nhẹ, không tải hàng chục GB video):

```powershell
python -m src.data.download_kaggle --dataset qipedc --only-labels
python -m src.data.download_kaggle --dataset vsl_v2 --only-labels
```

---

## 5. Cấu trúc thư mục dự án

```text
Nhan dien ki hieu/
├── configs/
│   ├── data_config.yaml          # Cấu hình trung tâm (QIPEDC + VSL-v2, scope, split, augment)
│   └── class_list.json           # 30 lớp VSL giao nhau cả 2 bộ Kaggle (v1.3) + mô tả động tác tay
├── docs/
│   └── collection_guide.md       # SOP hướng dẫn tự quay đa người & chế độ chỉ dùng Kaggle
├── scripts/
│   └── run_quy_pipeline.py       # CLI chạy trọn gói 4 bước trên dữ liệu thật (data/manifests/, data/splits/)
├── src/
│   └── data/
│       ├── __init__.py
│       ├── schema.py             # Shared contract: MANIFEST_COLUMNS, KAGGLE_SOURCES, validator, probe
│       ├── download_kaggle.py    # Tải & lập chỉ mục QIPEDC (kaggle_vsl) + VSL-v2 (kaggle_vsl_v2)
│       ├── scope.py              # Bước 1: Lọc phạm vi theo class_list / top_k / min_samples / all
│       ├── collect_videos.py     # Bước 2: Tool quay webcam tương tác đa người (tùy chọn Hybrid)
│       ├── build_manifest.py     # Bước 2: Hợp nhất đa nguồn & xuất quality_report.json
│       ├── split_dataset.py      # Bước 3: Chia train/val/test tách biệt tuyệt đối theo signer_id
│       └── augment.py            # Bước 4: Tăng cường video an toàn cho VSL (CHỈ tập train)
├── tests/
│   └── test_quy_pipeline.py      # Bộ kiểm thử tự động pytest cho cả Hybrid & Dual-Kaggle-only
├── data/                         # Dữ liệu video thật, manifest, splits, augmented
├── requirements.txt
└── README.md
```

---

## 6. Chế độ kết hợp tự quay (Hybrid — Tùy chọn)

Nếu muốn bổ sung thêm người ký hiệu tự quay qua webcam bên cạnh 2 bộ Kaggle, xem quy trình chuẩn tại [docs/collection_guide.md](docs/collection_guide.md):

```powershell
# Quay video webcam cho signer_01 với lớp class_id=0 ("giúp đỡ"), 5 lượt quay
python -m src.data.collect_videos --signer-id signer_01 --class-id 0 --num-takes 5 --duration 2.5 --fps 30

# Chạy trọn gói 4 bước hợp nhất Kaggle + tự quay (bật cờ --hybrid / --include-self)
python scripts/run_quy_pipeline.py --hybrid --config configs/data_config.yaml
```

### Các phép tăng cường an toàn cho VSL (`src/data/augment.py`)
1. **Biến đổi tốc độ thời gian (`speed`)**: nội suy khung hình trong khoảng `0.85x – 1.15x`.
2. **Cắt biên thời gian nhẹ (`trim`)**: cắt tối đa 8% đầu/cuối clip mà vẫn giữ trọn phần cử chỉ chính.
3. **Biến đổi không gian Affine an toàn (`affine`)**: xoay `±8°`, thu phóng `0.92 – 1.08`, dịch `±4%`, dùng `cv2.BORDER_REPLICATE` cố định trên toàn bộ clip để giữ trọn tay và thân trên.
4. **Biến thiên ánh sáng & nhiễu nhẹ (`light`)**: độ tương phản `0.88 – 1.12`, độ sáng `±20`, nhiễu Gaussian nhẹ.
5. **Lật ngang (`allow_hflip`) — MẶC ĐỊNH TẮT (`False`)**:
   > **Cảnh báo**: Lật ngang khung hình sẽ đảo tay trái ↔ tay phải, làm thay đổi tay thuận và hướng chuyển động của các ký hiệu VSL bất đối xứng (có thể đổi nghĩa ký hiệu hoặc làm sai lệch đặc trưng Left/Right Hand của MediaPipe Holistic). Chỉ bật `--allow-hflip` nếu toàn bộ tập nhãn đã được xác minh đối xứng hoàn toàn.

---

## 7. Cam kết chống rò rỉ dữ liệu (Data Leakage Prevention Guarantees)

Pipeline thực thi **4 chốt chặn cứng (hard guards)** ở mức code:
1. **Loại bỏ video đã tăng cường sẵn từ nguồn ngoài (`exclude_pre_augmented: true`)**: parser VSL-v2 tự động bỏ qua `processed_augmented/` để đảm bảo mọi video vào Bước 3 đều là bản gốc.
2. **Tách biệt tuyệt đối người ký hiệu (`verify_no_signer_leakage`)**:
   - `set(train.signer_id) ∩ set(val.signer_id) == ∅`
   - `set(train.signer_id) ∩ set(test.signer_id) == ∅`
   - `set(val.signer_id) ∩ set(test.signer_id) == ∅`
3. **Thứ tự bắt buộc: Chia tập TRƯỚC, Tăng cường SAU**:
   - `split_dataset.py` từ chối đầu vào nếu phát hiện bất kỳ hàng nào có `is_augmented == True`.
4. **Tăng cường CHỈ trên tập `train` (`ensure_train_only_manifest`)**:
   - `augment.py` kiểm tra toàn bộ cột `split` ngay khi đọc CSV; nếu tồn tại **bất kỳ hàng nào** có `split != "train"`, lập tức ném `ValueError("Augmentation is strictly allowed on train split only!")`.
   - `val.csv` và `test.csv` **không bao giờ** bị đọc hay chỉnh sửa bởi bước tăng cường.

---

## 8. Đặc tả bàn giao cho Phần 2 — Tiền xử lý MediaPipe Holistic (Thiên)

### 8.1. Danh sách tệp bàn giao

| Tệp bàn giao | Vai trò cho Phần 2 (MediaPipe Holistic) |
| :--- | :--- |
| `data/splits/train_augmented.csv` | Danh sách video huấn luyện (gồm video gốc `is_augmented=False` + video tăng cường `is_augmented=True`) |
| `data/splits/val.csv` | Danh sách video validation (chỉ video gốc, người ký hiệu tách biệt hoàn toàn với train & test) |
| `data/splits/test.csv` | Danh sách video kiểm thử cuối cùng (chỉ video gốc, không dùng để chỉnh siêu tham số) |
| `data/manifests/scope_report.json` | Danh sách lớp hoạt động (`class_id`, `label`, `slug`, `handedness`, `qipedc_variants`, `action_description`, `kaggle_clip_count`, `kaggle_v2_clip_count`) |
| `data/splits/split_report.json` | Chứng nhận `leakage_check_passed: true`, danh sách `signer_id` và số mẫu từng lớp trên mỗi tập |

### 8.2. Cấu trúc 16 cột chuẩn (`MANIFEST_COLUMNS` trong `src/data/schema.py`)

Mọi file CSV (`dataset_manifest.csv`, `scoped_manifest.csv`, `train.csv`, `val.csv`, `test.csv`, `train_augmented.csv`) đều tuân thủ chính xác cùng 16 cột (UTF-8 with BOM `utf-8-sig`):

| # | Tên cột | Kiểu dữ liệu | Mô tả & Quy ước |
| :-: | :--- | :--- | :--- |
| 1 | `sample_id` | `str` | Mã định danh duy nhất của từng mẫu (ví dụ: `kaggle_d0001b`, `vsl2_vsl2_signer_001_giup_do_v01`, `..._aug1`) |
| 2 | `video_path` | `str` | Đường dẫn tương đối chuẩn POSIX (`/`) tính từ thư mục gốc dự án |
| 3 | `label` | `str` | Nhãn tiếng Việt có dấu chuẩn NFC (ví dụ: `"giúp đỡ"`, `"muốn"`, `"tiếp tục"`) |
| 4 | `class_id` | `int` | Chỉ số lớp số nguyên liên tục `0 .. N-1` dùng trực tiếp làm target cho mô hình |
| 5 | `signer_id` | `str` | Mã người ký hiệu (`kaggle_signer_north/central/south`, `vsl2_signer_001..`, `signer_01..`) |
| 6 | `source` | `str` | Nguồn dữ liệu: `"kaggle_vsl"`, `"kaggle_vsl_v2"`, hoặc `"self_recorded"` |
| 7 | `split` | `str` | Phân vùng tập dữ liệu: `"train"`, `"val"`, hoặc `"test"` (rỗng trước Bước 3) |
| 8 | `is_augmented` | `bool` | `False` với video gốc; `True` với video tăng cường (chỉ xuất hiện trong `train_augmented.csv`) |
| 9 | `parent_sample_id` | `str` | `sample_id` của video gốc nếu `is_augmented=True`; chuỗi rỗng `""` với video gốc |
| 10 | `aug_type` | `str` | Mô tả phép tăng cường đã áp dụng (`speed(...)+affine(...)+light(...)`); rỗng với video gốc |
| 11 | `fps` | `float` | Tốc độ khung hình đo thực tế bằng OpenCV |
| 12 | `num_frames` | `int` | Tổng số khung hình đọc được |
| 13 | `duration_sec` | `float` | Thời lượng clip tính bằng giây (`num_frames / fps`) |
| 14 | `width` | `int` | Chiều rộng khung hình (px) |
| 15 | `height` | `int` | Chiều cao khung hình (px) |
| 16 | `is_valid` | `bool` | `True` nếu video mở được và đọc được khung hình hợp lệ bằng OpenCV |

### 8.3. Lưu ý kỹ thuật khi Thiên đọc dữ liệu vào MediaPipe Holistic

- Đọc manifest bằng `src.data.schema.load_manifest_df(path, validate=True, require_split=True)` để giữ đúng kiểu chuỗi cho `signer_id` / `sample_id`.
- Ghép `Path(project_root) / row["video_path"]` để mở video bằng `cv2.VideoCapture`.
- Trích xuất landmark độc lập theo từng split (`train_augmented.csv` → `features/train/`, `val.csv` → `features/val/`, `test.csv` → `features/test/`), đặt tên file theo `sample_id` (ví dụ `<sample_id>.npz`) để bảo toàn truy vết `parent_sample_id` và `signer_id`.
- Không áp dụng chuẩn hóa/thống kê (mean/std scaler) trên `val` hoặc `test`: chỉ `fit` bộ chuẩn hóa trên tập `train`.

---

## 9. Kiểm thử tự động (Automated Tests)

Chạy toàn bộ bộ test (bao gồm kiểm thử kết hợp 2 bộ Kaggle, chế độ `--kaggle-only`, chế độ Hybrid, chia tập không rò rỉ người ký hiệu và chốt chặn từ chối tăng cường trên `val`/`test`):

```powershell
pytest -v
```

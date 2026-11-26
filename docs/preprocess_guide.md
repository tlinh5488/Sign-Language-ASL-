# Hướng Dẫn Giai Đoạn 2 — Tiền Xử Lý Landmark MediaPipe Holistic (Thiên)

Tài liệu kỹ thuật chuẩn cho **Giai đoạn 2: Tiền xử lý (Thiên)** trong hệ thống **Nhận diện Ngôn ngữ Ký hiệu Tay Tiếng Việt (VSL)**, nối tiếp trực tiếp từ **Giai đoạn 1: Dữ liệu (Quý)** và bàn giao đặc trưng cố định độ dài cho **Giai đoạn 3: Huấn luyện mô hình (Hiếu)** cùng hàm chuẩn hóa dùng chung cho **Giai đoạn 4: Suy luận thời gian thực (Linh)**.

---

## 1. Sơ đồ tổng quan Pipeline Giai đoạn 2

```mermaid
flowchart LR
    subgraph S1["Giai đoạn 1 (Quý)"]
        M1["data/splits/train_augmented.csv"]
        M2["data/splits/val.csv"]
        M3["data/splits/test.csv"]
    end

    subgraph S21["Bước 2.1: Trích landmark (extract_landmarks.py)"]
        E1["MediaPipe Holistic (pose 33×4, face 468×3, left/right hand 21×3)"]
        E2["data/landmarks/<split>/<class_id>/<sample_id>.npz"]
        E3["data/landmarks/<split>_landmarks_index.csv"]
    end

    subgraph S22["Bước 2.2: Chuẩn hóa tọa độ (normalize.py)"]
        N1["Bàn tay: gốc về cổ tay (idx 0), chia khoảng cách cổ tay → MCP giữa (idx 9)"]
        N2["Thân trên: gốc tâm 2 vai (11,12), chia độ rộng vai"]
        N3["Vị trí toàn cục cổ tay trong hệ quy chiếu vai"]
    end

    subgraph S23["Bước 2.3: Chuỗi khung hình (sequence.py)"]
        Q1["Nội suy tuyến tính lỗ mất tay ngắn (<= max_gap=8)"]
        Q2["Cắt frame nghỉ đầu/cuối không có tay (trim_idle)"]
        Q3["Resample tuyến tính / Pad về L = 48 frame (+ vận tốc Δ tùy chọn)"]
    end

    subgraph S24["Bước 2.4: Lưu đặc trưng (build_features.py)"]
        O1["data/features/<split>/<sample_id>.npy (L, D)"]
        O2["data/features/{train,val,test}.npz (X, mask, y, sample_ids, signer_ids)"]
        O3["data/features/feature_meta.json"]
    end

    S1 --> S21 --> S22 --> S23 --> S24
```

| Bước | Module | Nhiệm vụ chính | Đầu ra |
| :--- | :--- | :--- | :--- |
| **Bước 2.1** | `src/preprocess/landmark_schema.py`, `src/preprocess/extract_landmarks.py` | Giải mã video `.mp4`, trích xuất landmark thô bằng MediaPipe Holistic (`mediapipe==0.10.14`, không lật ngang khung hình) | `data/landmarks/<split>/<class_id>/<sample_id>.npz`, `data/landmarks/<split>_landmarks_index.csv` |
| **Bước 2.2** | `src/preprocess/normalize.py` | Chuẩn hóa tọa độ thuần NumPy (bất biến vị trí đứng & kích thước người/tay), dùng chung cho cả offline lẫn real-time | Vector đặc trưng `(T, D)` (`D = 207` mặc định) + `frame_mask (T,)` |
| **Bước 2.3** | `src/preprocess/sequence.py` | Lấp lỗ mất tay ngắn hạn (`<= 8` frame), cắt frame nghỉ đầu/cuối, đưa về độ dài cố định `L = 48` | Tensor `(L, D)` `float32` + `mask (L,)` `bool` |
| **Bước 2.4** | `src/preprocess/build_features.py` | Đóng gói `.npy` từng mẫu, `{train,val,test}.npz`, `feature_meta.json` và tái kiểm định **0 rò rỉ người ký hiệu** | `data/features/{train,val,test}.npz`, `data/features/feature_meta.json` |

---

## 2. Công thức toán học Chuẩn hóa Tọa độ (Bước 2.2 — `src/preprocess/normalize.py`)

Mọi hàm trong `src/preprocess/normalize.py` **chỉ sử dụng NumPy** (không phụ thuộc MediaPipe hay OpenCV) để đảm bảo tốc độ cao và dùng lại nguyên vẹn khi triển khai real-time ở Giai đoạn 4.

### 2.1. Chuẩn hóa hình dạng bàn tay — Gốc về cổ tay (`normalize_hand`)

Với mỗi bàn tay (trái hoặc phải) gồm 21 khớp $\mathbf{p}_0, \mathbf{p}_1, \dots, \mathbf{p}_{20} \in \mathbb{R}^3$ (trong đó $\mathbf{p}_0$ là cổ tay `WRIST = 0`, $\mathbf{p}_9$ là khớp gốc ngón giữa `MIDDLE_MCP = 9`):

1. **Tịnh tiến gốc tọa độ về cổ tay**:
   $$\tilde{\mathbf{p}}_i = \mathbf{p}_i - \mathbf{p}_0, \quad i \in \{0, \dots, 20\}$$
   Khi đó cổ tay luôn nằm tại gốc: $\tilde{\mathbf{p}}_0 = (0, 0, 0)$.
2. **Chuẩn hóa tỉ lệ theo kích thước bàn tay** (khoảng cách Euclid từ cổ tay đến `MIDDLE_MCP`):
   $$s_{\text{hand}} = \|\mathbf{p}_9 - \mathbf{p}_0\|_2$$
   $$\hat{\mathbf{h}}_i = \begin{cases} \dfrac{\mathbf{p}_i - \mathbf{p}_0}{\max(s_{\text{hand}}, \varepsilon)} & \text{nếu phát hiện tay và } s_{\text{hand}} > \varepsilon \\ \mathbf{0} & \text{nếu không phát hiện tay} \end{cases}$$
   (với $\varepsilon = 10^{-6}$).

**Tính chất bất biến**: Với mọi phép biến đổi dịch chuyển $\mathbf{t} \in \mathbb{R}^3$ và phóng đại đồng nhất $\alpha > 0$:
$$\text{normalize\_hand}(\alpha \mathbf{p} + \mathbf{t}) = \text{normalize\_hand}(\mathbf{p}) \quad \text{và} \quad \|\hat{\mathbf{h}}_9\|_2 = 1.$$

### 2.2. Chuẩn hóa thân trên — Hệ quy chiếu hai vai (`normalize_pose`)

Với 33 điểm pose $\mathbf{P}_0, \dots, \mathbf{P}_{32}$, ta giữ 25 điểm thân trên `UPPER_BODY_POSE_IDX = [0..24]` (loại bỏ chân 25..32) và định nghĩa hệ quy chiếu cơ thể từ vai trái (`L_SHOULDER = 11`) và vai phải (`R_SHOULDER = 12`):

$$\mathbf{c}_{\text{body}} = \frac{\mathbf{P}_{11} + \mathbf{P}_{12}}{2}, \qquad s_{\text{body}} = \|\mathbf{P}_{11} - \mathbf{P}_{12}\|_2$$

$$\hat{\mathbf{b}}_j = \frac{\mathbf{P}_j - \mathbf{c}_{\text{body}}}{\max(s_{\text{body}}, \varepsilon)}, \quad j \in \{0, \dots, 24\}$$

- **Cơ chế dự phòng khi thiếu vai (`fill_body_frame`)**: Nếu tại khung hình $t$ hai vai bị khuất hoặc độ tin cậy thấp, $(\mathbf{c}_{\text{body}}(t), s_{\text{body}}(t))$ tự động lấy từ khung hình hợp lệ gần nhất trước đó (hoặc trung vị $s_{\text{body}}$ của video cho các khung hình đầu; nếu cả video không có vai hợp lệ thì dùng `fallback_shoulder_width = 0.3`).

### 2.3. Vị trí toàn cục của bàn tay so với cơ thể (`hand_global_position`)

Vì `normalize_hand` đã đưa cổ tay về $(0,0,0)$ để biểu diễn **hình dạng ngón tay**, ta giữ thêm vector vị trí của hai cổ tay trong hệ quy chiếu vai để mô hình biết **tay đang đặt ở đâu so với ngực/mặt/vai**:

$$\hat{\mathbf{g}}_{\text{left}} = \frac{\mathbf{p}_{0,\text{left}} - \mathbf{c}_{\text{body}}}{\max(s_{\text{body}}, \varepsilon)}, \qquad \hat{\mathbf{g}}_{\text{right}} = \frac{\mathbf{p}_{0,\text{right}} - \mathbf{c}_{\text{body}}}{\max(s_{\text{body}}, \varepsilon)}$$

(Nếu tay không xuất hiện ở khung hình $t$ thì $\hat{\mathbf{g}} = \mathbf{0}$).

### 2.4. Tập con điểm mặt tùy chọn (`normalize_face`)

Khi cấu hình `feature_set: "pose_upper+hands+face_subset"`, 36 điểm mặt quan trọng cho tín hiệu phi thủ công trong VSL (`FACE_SUBSET_IDX`: môi trong/ngoài, lông mày, mắt, mũi, viền hàm) được chuẩn hóa theo gốc đầu mũi (`FACE_NOSE_TIP = 1`) và chia cho độ rộng vai $s_{\text{body}}$:

$$\hat{\mathbf{f}}_k = \frac{\mathbf{F}_k - \mathbf{F}_{\text{nose}}}{\max(s_{\text{body}}, \varepsilon)}, \quad k \in \text{FACE\_SUBSET\_IDX}$$

---

## 3. Bố cục Vector Đặc trưng & Số chiều $D$

Hàm `feature_dim(cfg)` và `feature_names(cfg)` trong `src/preprocess/normalize.py` quy định thứ tự cố định của từng chiều trong vector đặc trưng mỗi khung hình:

### Cấu hình mặc định (`feature_set: "pose_upper+hands"`, `use_z: true`, `add_velocity: false`) $\Rightarrow \mathbf{D = 207}$

| Thứ tự khối | Tên khối (`feature_blocks`) | Dải chỉ số cột | Số điểm × Số trục | Số chiều |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `pose` (thân trên `0..24`) | `0 .. 74` | $25 \times (x, y, z)$ | **75** |
| 2 | `left_hand` (tay trái `0..20`, gốc cổ tay) | `75 .. 137` | $21 \times (x, y, z)$ | **63** |
| 3 | `right_hand` (tay phải `0..20`, gốc cổ tay) | `138 .. 200` | $21 \times (x, y, z)$ | **63** |
| 4 | `left_wrist_global` (cổ tay trái so với tâm vai) | `201 .. 203` | $1 \times (x, y, z)$ | **3** |
| 5 | `right_wrist_global` (cổ tay phải so với tâm vai) | `204 .. 206` | $1 \times (x, y, z)$ | **3** |
| **Tổng** | **`pose_upper+hands` (Mặc định)** | **`0 .. 206`** | | **$\mathbf{D = 207}$** |

### Các biến thể cấu hình khác trong `configs/preprocess_config.yaml`

| Cấu hình | Công thức số chiều | Giá trị $D$ |
| :--- | :--- | :---: |
| `pose_upper+hands`, `use_z: true`, `add_velocity: false` *(mặc định)* | $75 + 63 + 63 + 6$ | **207** |
| `pose_upper+hands`, `use_z: false` (chỉ 2D $x,y$) | $25\times 2 + 21\times 2 + 21\times 2 + 2\times 2$ | **138** |
| `pose_upper+hands+face_subset` (thêm 36 điểm mặt × 3) | $207 + 36 \times 3$ | **315** |
| Bật `add_velocity: true` (nối thêm $\Delta \mathbf{x}_t = \mathbf{x}_t - \mathbf{x}_{t-1}$) | $2 \times D_{\text{base}}$ | **414** (hoặc **630** khi có mặt) |

---

## 4. Xử lý Chuỗi Khung Hình (Bước 2.3 — `src/preprocess/sequence.py`)

Mỗi video có số khung hình gốc $T$ khác nhau và có thể bị mất dấu bàn tay trong vài khung hình do chuyển động nhanh (motion blur):

1. **Nội suy lỗ mất tay ngắn hạn (`interpolate_missing(arr, mask, max_gap=8)`)**:
   - Nếu một bàn tay bị mất liên tiếp $g$ khung hình nằm giữa hai khung hình phát hiện được ($t_{s-1}$ và $t_e$) với $1 \le g \le \text{max\_gap}$ (`max_gap = 8`), tọa độ trong đoạn $[t_s, t_{e-1}]$ được nội suy tuyến tính:
     $$\mathbf{p}(t_s + k - 1) = \left(1 - \frac{k}{g+1}\right)\mathbf{p}(t_{s-1}) + \frac{k}{g+1}\mathbf{p}(t_e), \quad k = 1, \dots, g$$
   - Các khoảng mất tay dài hơn `max_gap` hoặc nằm ở biên đầu/cuối video được giữ nguyên bằng $\mathbf{0}$ (không ngoại suy sai).
2. **Cắt khung hình nghỉ đầu/cuối (`trim_idle_frames`)**:
   - Tự động cắt bỏ các khung hình ở đầu và cuối video khi người ký hiệu chưa đưa tay lên hoặc đã hạ tay xuống (`left_hand_mask == False` và `right_hand_mask == False`).
   - Nếu toàn bộ video không phát hiện tay nào, giữ nguyên chuỗi để không sinh mảng rỗng.
3. **Đưa về độ dài cố định $L = 48$ (`to_fixed_length`)**:
   - **Chế độ `method: "resample"` (mặc định)**: lấy mẫu đều theo thời gian bằng nội suy tuyến tính tại $L = 48$ vị trí $\tau_\ell = \frac{\ell}{L - 1}(T' - 1)$ (`resample_sequence`). Xử lý an toàn cả $T'=0$ (trả về mảng 0, `mask=False`) và $T'=1$ (lặp lại khung hình duy nhất).
   - **Chế độ `method: "pad"`**: giữ nguyên tốc độ khung hình gốc; nếu $T' < L$ thì đệm `pad_value = 0.0` vào cuối chuỗi và đánh dấu `mask[T':L] = False`; nếu $T' > L$ thì cắt chính giữa (`truncate="center"`).

---

## 5. Định dạng File Đầu Ra & Cách Hiếu (Giai đoạn 3) Nạp Dữ Liệu Huấn Luyện

Sau khi chạy Bước 2.4 (`src.preprocess.build_features`), thư mục `data/features/` có cấu trúc:

```text
data/features/
├── train/
│   └── <sample_id>.npy           # 4.251 file (48, 207) float32 cho từng mẫu train
├── val/
│   └── <sample_id>.npy           # 294 file (48, 207) float32 cho từng mẫu val
├── test/
│   └── <sample_id>.npy           # 309 file (48, 207) float32 cho từng mẫu test
├── train.npz                     # Tập train đóng gói (4251, 48, 207) — 140.44 MiB
├── val.npz                       # Tập val đóng gói (294, 48, 207) — 9.82 MiB
├── test.npz                      # Tập test đóng gói (309, 48, 207) — 9.94 MiB
├── feature_meta.json             # Thông số D=207, L=48, feature_names, class_list, signer_lists
└── preprocess_summary.json       # Báo cáo tổng kết sản xuất (mode="production", 4.854 mẫu)
```

### Thống kê Thực tế Bộ Đặc trưng Sản xuất (4.854 video VSL thật — 30 lớp, 31 người ký hiệu)

| Tập dữ liệu | File đóng gói | Số mẫu ($N$) | Shape `X` `(N, L, D)` | Số người ký hiệu | Số lớp | Tỷ lệ phát hiện tay (`hand_detect_ratio`) | Tỷ lệ phát hiện thân trên (`pose`) | Tỷ lệ phát hiện mặt (`face`) | Mẫu mất tay (`zero_hand`) | Dung lượng nén (`.npz`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** (`train_augmented`) | `data/features/train.npz` | **4.251** (1.417 gốc + 2.834 tăng cường 2×) | **`(4251, 48, 207)`** | **22** (5 QIPEDC + 17 VSL-v2) | **30/30** | **93.67%** (`0.9367`) | **100.00%** (`1.0000`) | **83.34%** (`0.8334`) | **0** | **140.44 MiB** |
| **Validation** (`val`) | `data/features/val.npz` | **294** | **`(294, 48, 207)`** | **4** (VSL-v2 độc lập) | **30/30** | **94.55%** (`0.9455`) | **100.00%** (`1.0000`) | **76.40%** (`0.7640`) | **0** | **9.82 MiB** |
| **Test** (`test`) | `data/features/test.npz` | **309** | **`(309, 48, 207)`** | **5** (VSL-v2 độc lập) | **30/30** | **90.13%** (`0.9013`) | **100.00%** (`1.0000`) | **62.93%** (`0.6293`) | **0** | **9.94 MiB** |
| **Tổng / Trung bình** | **3 file `.npz`** | **4.854** | **`(4854, 48, 207)`** | **31 (0 trùng lặp)** | **30/30** | **93.50%** (`0.9350`) | **100.00%** (`1.0000`) | **81.62%** (`0.8162`) | **0** | **160.20 MiB** |

### Bảng các khóa trong `data/features/{train,val,test}.npz`

| Khóa (`key`) | Shape | Dtype | Ý nghĩa |
| :--- | :---: | :---: | :--- |
| `X` | `(N, L, D)` = `(N, 48, 207)` | `float32` | Tensor đặc trưng chuỗi khung hình đã chuẩn hóa (không có `NaN`/`Inf`) |
| `mask` | `(N, L)` = `(N, 48)` | `bool` | `True` tại khung hình hợp lệ, `False` tại khung hình đệm (padding) |
| `y` | `(N,)` | `int64` | Nhãn số nguyên `class_id` $\in \{0, 1, \dots, 29\}$ |
| `sample_ids` | `(N,)` | `<U...` (str) | Định danh mẫu (khớp với manifest Giai đoạn 1) |
| `signer_ids` | `(N,)` | `<U...` (str) | Định danh người ký hiệu (đã kiểm định 0 trùng lặp giữa train/val/test) |
| `labels` | `(N,)` | `<U...` (str) | Nhãn tiếng Việt có dấu (ví dụ `"giúp đỡ"`, `"muốn"`) |
| `is_augmented` | `(N,)` | `bool` | `True` nếu là mẫu tăng cường trong tập `train`, luôn `False` ở `val`/`test` |

### Code mẫu cho Hiếu (Giai đoạn 3 — Huấn luyện LSTM / GRU / Transformer)

```python
import json
import numpy as np

# 1. Đọc trực tiếp file .npz (KHÔNG cần allow_pickle=True)
train_data = np.load("data/features/train.npz")
val_data   = np.load("data/features/val.npz")
test_data  = np.load("data/features/test.npz")

X_train, mask_train, y_train = train_data["X"], train_data["mask"], train_data["y"]
X_val,   mask_val,   y_val   = val_data["X"],   val_data["mask"],   val_data["y"]
X_test,  mask_test,  y_test  = test_data["X"],  test_data["mask"],  test_data["y"]

print("X_train:", X_train.shape, X_train.dtype)  # (N_train, 48, 207) float32
print("y_train:", y_train.shape, y_train.dtype)  # (N_train,) int64 (0..29)

# 2. Đọc metadata để lấy số chiều D, độ dài chuỗi L và danh sách 30 lớp
with open("data/features/feature_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

D = meta["D"]                      # 207
L = meta["L"]                      # 48
num_classes = meta["num_classes"]  # 30
feature_names = meta["feature_names"]
```

---

## 6. Cách Linh (Giai đoạn 4) Tái Sử Dụng `normalize_frame` Trong Thời Gian Thực

Để đảm bảo **nhất quán tuyệt đối giữa huấn luyện và suy luận webcam** (*"Landmark — cùng hàm chuẩn hóa"*), ứng dụng thời gian thực của Linh gọi trực tiếp `results_to_arrays`, `normalize_frame`, và `to_fixed_length` từ `src.preprocess`:

```python
from collections import deque
import numpy as np
import yaml

from src.preprocess.extract_landmarks import results_to_arrays
from src.preprocess.normalize import make_state, normalize_frame
from src.preprocess.sequence import to_fixed_length

with open("configs/preprocess_config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

state = make_state()               # Nhớ tâm vai & độ rộng vai giữa các frame webcam
buffer = deque(maxlen=48)          # Cửa sổ trượt 48 frame gần nhất

# Trong vòng lặp webcam cho mỗi frame đã chạy holistic.process(rgb_frame):
frame_dict = results_to_arrays(results)
feat_t = normalize_frame(frame_dict, cfg, state=state)   # (207,) float32 — khớp 100% với normalize_sequence
buffer.append(feat_t)

if len(buffer) >= 16:
    seq = np.stack(list(buffer), axis=0)                 # (T_cur, 207)
    X_input, mask_input = to_fixed_length(seq, None, cfg)  # (48, 207) float32 sẵn sàng đưa vào model.predict(X_input[None])
```

---

## 7. Lệnh Chạy CLI (Sản Xuất Thực Tế)

```powershell
# 1. Chạy từng bước trên các manifest thực tế từ Giai đoạn 1 (Quý)
python -m src.preprocess.extract_landmarks --config configs/preprocess_config.yaml
python -m src.preprocess.build_features --config configs/preprocess_config.yaml --splits train val test --min-hand-ratio 0.0

# 2. Hoặc chạy trọn gói Bước 2.1 -> 2.4 bằng một lệnh duy nhất
python scripts/run_thien_preprocess.py --config configs/preprocess_config.yaml

# 3. Chạy lại Bước 2.2 -> 2.4 khi đã trích xuất xong raw landmarks (.npz)
python scripts/run_thien_preprocess.py --config configs/preprocess_config.yaml --skip-extract
```

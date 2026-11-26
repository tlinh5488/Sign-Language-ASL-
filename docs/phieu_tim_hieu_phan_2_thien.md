# PHIẾU TÌM HIỂU VÀ PHÂN CÔNG NHÓM — PHẦN 2: TIỀN XỬ LÝ (THIÊN)

- **Đề tài:** Mô hình nhận diện ngôn ngữ ký hiệu (Ngôn ngữ ký hiệu tay tiếng Việt — VSL)
- **Giai đoạn phụ trách:** Phần 2 — Tiền xử lý (Trích landmark MediaPipe Holistic → Chuẩn hóa tọa độ gốc cổ tay/vai → Chuỗi khung hình cố định độ dài → Lưu đặc trưng `.npy`/`.npz`)
- **Người phụ trách:** Thiên
- **Nhận bàn giao từ:** Quy (Phần 1 — Dữ liệu: **4,854** video `.mp4` thật đã chia tập theo **31** người ký hiệu: **4,251** `train_augmented`, **294** `val`, **309** `test`)
- **Bàn giao cho:** Hiếu (Phần 3 — Huấn luyện mô hình LSTM/BiLSTM) & Linh (Phần 4 — Triển khai suy luận thời gian thực trên webcam/video)
- **Khung thời gian dự án:** 2–3 tuần

---

## I. Tóm tắt 4 bước của Phần 2 (Tiền xử lý) trên 4,854 video thực tế

| Bước | Tên bước | Module chính | Đầu vào → Đầu ra thực tế |
| :--- | :--- | :--- | :--- |
| **Bước 2.1** | **Trích landmark** (MediaPipe Holistic) | `src/preprocess/landmark_schema.py`, `src/preprocess/extract_landmarks.py` | **4,854** video `.mp4` thật → **4,854** file raw landmark `data/landmarks/<split>/<class_id>/<sample_id>.npz` + 3 bảng chỉ mục `data/landmarks/{train,val,test}_landmarks_index.csv` (tỷ lệ phát hiện tay trung bình **93.50%**, tư thế thân trên **100.00%**, khuôn mặt **81.62%**) |
| **Bước 2.2** | **Chuẩn hóa tọa độ** (Gốc về cổ tay & tâm hai vai) | `src/preprocess/normalize.py` | Tọa độ MediaPipe thô → Vector đặc trưng bất biến vị trí/tỉ lệ **`D = 207`** chiều/khung hình (`pose` thân trên 75 + `left_hand` 63 + `right_hand` 63 + `wrist_global` 6), thuần NumPy dùng chung cho cả Phần 2 và Phần 4 |
| **Bước 2.3** | **Chuỗi khung hình** (Nội suy lỗ mất tay, cắt nghỉ, cố định độ dài) | `src/preprocess/sequence.py` | Chuỗi độ dài biến thiên $T$ khung hình → Nội suy tuyến tính lỗ mất tay ngắn ($\le 8$ khung hình), cắt khung hình nghỉ đầu/cuối (`trim_idle`), lấy mẫu đều tuyến tính về **`L = 48`** khung hình |
| **Bước 2.4** | **Lưu đặc trưng** (`.npy` từng mẫu & `.npz` nén theo tập) | `src/preprocess/build_features.py`, `scripts/run_thien_preprocess.py` | **4,854** file `data/features/<split>/<sample_id>.npy` `(48, 207)` + 3 gói nén `data/features/train.npz` `(4251, 48, 207)` (**140.44 MiB**), `val.npz` `(294, 48, 207)` (**9.82 MiB**), `test.npz` `(309, 48, 207)` (**9.94 MiB**) + `feature_meta.json` & `preprocess_summary.json` |

---

## II. Trả lời 5 câu hỏi tìm hiểu (Phần 2: Tiền xử lý)

### Câu hỏi 1: MediaPipe Holistic hoạt động thế nào, trả về những nhóm điểm nào, có bao nhiêu điểm mỗi nhóm?

1. **Cơ chế hoạt động của MediaPipe Holistic (`mediapipe==0.10.14`):**
   - MediaPipe Holistic là kiến trúc đa tầng kết hợp đồng thời 3 mô hình con trong một đồ thị suy luận thống nhất:
     1. **BlazePose Detector & Landmark Model:** phát hiện vùng cơ thể và dự đoán 33 điểm khớp toàn thân từ khung hình RGB.
     2. **Hand Crop & MediaPipe Hands (21 điểm × 2 tay):** dùng vị trí cổ tay/bàn tay từ nhánh Pose để cắt vùng ROI (Region of Interest) cho tay trái và tay phải, sau đó tái dự đoán chính xác 21 khớp ngón tay ở độ phân giải cục bộ cao.
     3. **Face Crop & FaceMesh (468 điểm):** cắt vùng đầu từ nhánh Pose để dự đoán lưới 468 điểm khuôn mặt 3D.
   - Ở chế độ luồng video (`static_image_mode=False`), Holistic tận dụng bộ theo dõi khung hình trước (tracking) để tăng tốc độ và độ mượt giữa các khung hình liên tiếp; chỉ khi độ tin cậy theo dõi xuống dưới `min_tracking_confidence` mới kích hoạt lại bộ dò toàn ảnh.
2. **Các nhóm điểm trả về, số lượng điểm và định dạng lưu trữ (`src/preprocess/landmark_schema.py`):**

| Nhóm điểm | Thuộc tính MediaPipe | Số điểm ($N$) | Số tọa độ/điểm | Shape mảng thô | Ý nghĩa trong nhận diện ký hiệu VSL |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Tư thế cơ thể (`pose`)** | `results.pose_landmarks` | **33** (`POSE_N=33`) | **4** (`x, y, z, visibility`) | `(T, 33, 4)` `float32` | Cho biết hệ quy chiếu cơ thể (2 vai, khuỷu tay, cổ tay, đầu, hông). Dự án giữ **25 điểm thân trên** (`UPPER_BODY_POSE_IDX = 0..24`), loại bỏ 8 điểm chân (`25..32`) không tham gia ký hiệu tay. |
| **Bàn tay trái (`left_hand`)** | `results.left_hand_landmarks` | **21** (`HAND_N=21`) | **3** (`x, y, z`) | `(T, 21, 3)` `float32` | Hình dạng chi tiết 21 khớp bàn tay trái giải phẫu của người ký hiệu (`WRIST=0`, 5 ngón × 4 đốt, `MIDDLE_MCP=9`). |
| **Bàn tay phải (`right_hand`)** | `results.right_hand_landmarks` | **21** (`HAND_N=21`) | **3** (`x, y, z`) | `(T, 21, 3)` `float32` | Hình dạng chi tiết 21 khớp bàn tay phải giải phẫu của người ký hiệu. |
| **Khuôn mặt (`face`)** | `results.face_landmarks` | **468** (`FACE_N=468`) | **3** (`x, y, z`) | `(T, 468, 3)` `float32` | Toàn bộ lưới mặt 3D; schema định nghĩa sẵn tập con **36 điểm** (`FACE_SUBSET_IDX`: môi trong/ngoài, lông mày, mắt, mũi, viền hàm) cho tín hiệu phi thủ công. |
| **Mặt nạ phát hiện (`*_mask`)** | Suy ra khi trích xuất | 4 vector | **1** (`bool`) | `(T,)` `bool` × 4 | `pose_mask`, `face_mask`, `left_hand_mask`, `right_hand_mask` (`True` = phát hiện được; khung hình không phát hiện được gán toàn bộ bằng `0.0`). |

---

### Câu hỏi 2: Cách chuẩn hóa tọa độ để không phụ thuộc vị trí và khoảng cách tới camera; tay trái/tay phải xử lý ra sao?

Nếu đưa tọa độ ảnh thô $(x, y, z) \in [0, 1]$ vào mô hình, cùng một ký hiệu nhưng người đứng lệch sang trái/phải (dịch chuyển tịnh tiến) hoặc đứng gần/xa camera (thay đổi tỉ lệ phóng đại) sẽ tạo ra vector số khác hẳn nhau. Module `src/preprocess/normalize.py` giải quyết triệt để bằng **chuẩn hóa phân tách Hình dạng bàn tay (cục bộ) + Vị trí bàn tay trong hệ quy chiếu vai (toàn cục)**:

1. **Chuẩn hóa hình dạng từng bàn tay — Dời gốc về cổ tay (`normalize_hand`):**
   - Với 21 khớp $\mathbf{p}_0, \dots, \mathbf{p}_{20} \in \mathbb{R}^3$ của một bàn tay (trong đó $\mathbf{p}_0$ là cổ tay `WRIST = 0`, $\mathbf{p}_9$ là khớp gốc ngón giữa `MIDDLE_MCP = 9`):
     $$s_{\text{hand}} = \|\mathbf{p}_9 - \mathbf{p}_0\|_2, \qquad \hat{\mathbf{h}}_i = \frac{\mathbf{p}_i - \mathbf{p}_0}{\max(s_{\text{hand}}, 10^{-6})}, \quad i \in \{0, \dots, 20\}$$
   - **Ý nghĩa:** Cổ tay luôn nằm tại $(0, 0, 0)$ và khoảng cách từ cổ tay tới khớp gốc ngón giữa luôn bằng $1.0$. Phép biến đổi này **bất biến tuyệt đối** với mọi dịch chuyển tịnh tiến và phóng đại đồng nhất: $\text{normalize\_hand}(\alpha \mathbf{p} + \mathbf{t}) = \text{normalize\_hand}(\mathbf{p})$ (sai số số học dấu phẩy động thực nghiệm $< 5 \times 10^{-7}$).
2. **Chuẩn hóa thân trên theo tâm hai vai và độ rộng vai (`normalize_pose`):**
   - Từ vai trái (`L_SHOULDER = 11`) và vai phải (`R_SHOULDER = 12`):
     $$\mathbf{c}_{\text{body}} = \frac{\mathbf{P}_{11} + \mathbf{P}_{12}}{2}, \qquad s_{\text{body}} = \|\mathbf{P}_{11} - \mathbf{P}_{12}\|_2, \qquad \hat{\mathbf{b}}_j = \frac{\mathbf{P}_j - \mathbf{c}_{\text{body}}}{\max(s_{\text{body}}, 10^{-6})}, \quad j \in \{0, \dots, 24\}$$
   - Khi một khung hình bị khuất vai tạm thời, hàm `fill_body_frame` tự động lấy $(\mathbf{c}_{\text{body}}, s_{\text{body}})$ từ khung hình hợp lệ gần nhất trước đó (hoặc trung vị $s_{\text{body}}$ của chuỗi).
3. **Giữ vị trí toàn cục của cổ tay so với cơ thể (`hand_global_position`):**
   - Vì `normalize_hand` đã dời gốc về cổ tay nên chỉ mô tả **hình dạng ngón tay** mà mất thông tin **bàn tay đang đặt trước ngực, ngang tai hay trên trán**. Ta bổ sung 2 vector vị trí cổ tay trong hệ quy chiếu vai (6 chiều):
     $$\hat{\mathbf{g}}_{\text{left}} = \frac{\mathbf{p}_{0,\text{left}} - \mathbf{c}_{\text{body}}}{\max(s_{\text{body}}, 10^{-6})}, \qquad \hat{\mathbf{g}}_{\text{right}} = \frac{\mathbf{p}_{0,\text{right}} - \mathbf{c}_{\text{body}}}{\max(s_{\text{body}}, 10^{-6})}$$
4. **Cách xử lý tay trái và tay phải:**
   - **Tách riêng 2 khe cố định (dedicated slots):** Tay trái luôn chiếm các chiều `[75:138]` (hình dạng) và `[201:204]` (vị trí cổ tay toàn cục); tay phải luôn chiếm các chiều `[138:201]` (hình dạng) và `[204:207]` (vị trí cổ tay toàn cục).
   - **Không bao giờ gộp chung hay đảo vị trí hai tay:** Trong VSL có cả ký hiệu 1 tay (`1_tay`) và 2 tay (`2_tay`), việc giữ cố định kênh tay trái và tay phải giúp mô hình phân biệt rõ tay thuận, tay đỡ và hướng chuyển động không gian (đồng bộ với quy tắc `allow_hflip: false` từ Phần 1).
   - **Khi một tay không xuất hiện trong khung hình (`present=False`):** Toàn bộ 63 chiều hình dạng và 3 chiều vị trí toàn cục của tay đó được gán đúng bằng `0.0` (không gây `NaN`/`Inf`).

---

### Câu hỏi 3: Chọn độ dài chuỗi bao nhiêu khung hình; cách padding và nội suy khi bị mất landmark?

1. **Chọn độ dài chuỗi cố định `L = target_len = 48` khung hình (`configs/preprocess_config.yaml`):**
   - Các video ký hiệu từ đơn VSL trong bộ dữ liệu thực tế có độ dài dao động từ ~25 đến ~90 khung hình (trung bình trước khi cắt nghỉ là **62.0** khung hình trên `train`, **64.1** trên `val`, **64.1** trên `test`; sau khi cắt khung hình nghỉ đầu/cuối còn trung bình **60.1** khung hình).
   - Chọn **`L = 48` khung hình** (tương đương ~1.6–1.9 giây ở 25–30 FPS) vừa bảo toàn đầy đủ các pha của một từ ký hiệu (*chuẩn bị → giữ/chuyển động chính → kết thúc*), vừa tránh chuỗi quá dài gây suy giảm đạo hàm (vanishing gradient) và giúp tốc độ suy luận thời gian thực ở Phần 4 đạt độ trễ thấp.
2. **Nội suy khi bị mất landmark bàn tay ngắn hạn (`interpolate_missing` & `interpolate_raw_hands` trong `src/preprocess/sequence.py`):**
   - Khi bàn tay di chuyển nhanh gây nhòe chuyển động (motion blur), MediaPipe có thể mất dấu bàn tay trong vài khung hình liên tiếp.
   - Với mọi khoảng mất tay nằm **ở giữa** hai khung hình phát hiện được ($t_{s-1}$ và $t_e$) có độ dài $g = e - s \le \text{max\_gap}$ (`max_gap = 8` khung hình):
     $$\mathbf{p}(t_s + k - 1) = \left(1 - \frac{k}{g+1}\right)\mathbf{p}(t_{s-1}) + \frac{k}{g+1}\mathbf{p}(t_e), \quad k = 1, \dots, g$$
   - **Quy tắc an toàn:** Các khoảng mất tay dài hơn `max_gap = 8` khung hình (tay thực sự rời khỏi vùng ký hiệu) hoặc nằm ở biên đầu/cuối video **không được nội suy** mà giữ nguyên bằng `0.0` để không bịa đặt quỹ đạo giả.
3. **Cắt khung hình nghỉ (`trim_idle_frames`) & Cố định độ dài (`to_fixed_length`):**
   - **Bước cắt nghỉ (`trim_idle: true`):** Cắt bỏ các khung hình ở đầu và cuối video khi người ký hiệu chưa đưa tay lên hoặc đã hạ cả hai tay xuống (`left_hand_mask == False` và `right_hand_mask == False`), chỉ giữ lại đoạn hoạt động chính từ khung hình có tay đầu tiên đến khung hình có tay cuối cùng.
   - **Bước đưa về đúng `L = 48` khung hình:**
     - *Phương pháp chính (`method: "resample"` — mặc định):* Lấy mẫu đều theo thời gian bằng nội suy tuyến tính (`resample_sequence`) tại 48 mốc $\tau_\ell = \frac{\ell}{47}(T' - 1)$ ($\ell = 0, \dots, 47$). Cách này giúp chuẩn hóa cả tốc độ ký hiệu nhanh/chậm về cùng 48 bước thời gian với `mask` toàn `True`.
     - *Phương pháp dự phòng (`method: "pad"` / `pad_or_truncate`):* Nếu $T' < 48$ thì đệm `pad_value = 0.0` vào cuối chuỗi và đánh dấu `mask[T':48] = False`; nếu $T' > 48$ thì cắt chính giữa (`truncate="center"`).

---

### Câu hỏi 4: Định dạng lưu `.npy` / `.npz` và quy ước shape (số mẫu, số khung hình, số đặc trưng)?

1. **Quy ước chiều đặc trưng mỗi khung hình (`D = 207` chiều mặc định — `feature_set: "pose_upper+hands"`):**

| Thứ tự khối | Tên khối (`feature_blocks`) | Chỉ số cột | Công thức số chiều | Số chiều |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `pose` (25 điểm thân trên `0..24` trong hệ quy chiếu vai) | `0 .. 74` | $25 \times (x, y, z)$ | **75** |
| 2 | `left_hand` (21 điểm tay trái, gốc tại cổ tay trái) | `75 .. 137` | $21 \times (x, y, z)$ | **63** |
| 3 | `right_hand` (21 điểm tay phải, gốc tại cổ tay phải) | `138 .. 200` | $21 \times (x, y, z)$ | **63** |
| 4 | `left_wrist_global` (vị trí cổ tay trái so với tâm vai) | `201 .. 203` | $1 \times (x, y, z)$ | **3** |
| 5 | `right_wrist_global` (vị trí cổ tay phải so với tâm vai) | `204 .. 206` | $1 \times (x, y, z)$ | **3** |
| **Tổng** | **Vector đặc trưng 1 khung hình** | **`0 .. 206`** | $75 + 63 + 63 + 3 + 3$ | **`D = 207`** |

2. **Định dạng lưu trữ 2 cấp độ trong `data/features/` (`src/preprocess/build_features.py`):**
   - **Cấp độ từng mẫu (`.npy`):** `data/features/<split>/<sample_id>.npy` có shape **`(L, D) = (48, 207)`**, kiểu `float32` (tổng cộng **4,854** file `.npy`).
   - **Cấp độ đóng gói theo tập (`.npz` nén bằng `np.savez_compressed`):** `data/features/train.npz`, `data/features/val.npz`, `data/features/test.npz`, nạp trực tiếp bằng `np.load("data/features/train.npz")` với `allow_pickle=False` mặc định:

| Tập dữ liệu | File đóng gói | Shape `X` `(N, L, D)` | Shape `mask` `(N, L)` | Shape `y` `(N,)` | Số người ký hiệu | Số lớp phủ | Dung lượng nén |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** (từ `train_augmented`) | `data/features/train.npz` | **`(4251, 48, 207)`** `float32` | `(4251, 48)` `bool` | `(4251,)` `int64` (`0..29`) | **22** người | **30/30** | **140.44 MiB** |
| **Validation** | `data/features/val.npz` | **`(294, 48, 207)`** `float32` | `(294, 48)` `bool` | `(294,)` `int64` (`0..29`) | **4** người | **30/30** | **9.82 MiB** |
| **Test** | `data/features/test.npz` | **`(309, 48, 207)`** `float32` | `(309, 48)` `bool` | `(309,)` `int64` (`0..29`) | **5** người | **30/30** | **9.94 MiB** |
| **Tổng cộng** | **3 file `.npz` + `feature_meta.json`** | **`(4854, 48, 207)`** | **`(4854, 48)`** | **`(4854,)`** | **31 người (0 trùng)** | **30/30** | **160.20 MiB** |

   - Mỗi file `.npz` chứa đúng 7 khóa chuẩn: `X` `(N, 48, 207) float32`, `mask` `(N, 48) bool`, `y` `(N,) int64`, `sample_ids` `(N,) <U`, `signer_ids` `(N,) <U`, `labels` `(N,) <U`, `is_augmented` `(N,) bool`.

---

### Câu hỏi 5: Làm sao để hàm chuẩn hóa dùng lại được y nguyên ở Phần 4 (tránh lệch giữa lúc huấn luyện và lúc chạy thật)?

Hiện tượng lệch huấn luyện – suy luận (*training-serving skew*) xảy ra khi ứng dụng webcam ở Phần 4 viết lại công thức chuẩn hóa khác với Phần 2 (khác chỉ số khớp, khác thứ tự ghép mảng, khác cách xử lý khi mất vai/tay). Nhóm thiết kế kiến trúc **Zero-Skew Shared Normalization** như sau:

1. **Tách module thuần NumPy (`src/preprocess/normalize.py` & `src/preprocess/sequence.py`):**
   - Tuyệt đối không import `mediapipe` hay `cv2` bên trong `normalize.py` và `sequence.py` (được kiểm tra tự động bằng unit test trong `tests/test_thien_preprocess.py`). Nhờ đó ứng dụng Phần 4 (dù chạy ONNX Runtime hay TFLite) import trực tiếp `from src.preprocess.normalize import normalize_frame, make_state` mà không kéo theo phụ thuộc nặng.
2. **Dùng chung một nhân tính toán `_normalize_arrays` cho cả chuỗi (`normalize_sequence`) và từng khung hình (`normalize_frame`):**
   - `normalize_sequence(raw, cfg)` (dùng khi tạo dữ liệu huấn luyện cho Hiếu) và `normalize_frame(frame_dict, cfg, state=state)` (dùng trên từng khung hình webcam cho Linh) đều gọi cùng một hàm lõi `_normalize_arrays`.
   - Biến trạng thái `state = make_state()` trong `normalize_frame` lưu lại $(\mathbf{c}_{\text{body}}, s_{\text{body}})$ của khung hình có vai gần nhất, mô phỏng chính xác cơ chế `fill_body_frame` của chế độ chuỗi.
   - **Kiểm chứng toán học:** `max(|normalize_sequence(raw)[t] - normalize_frame(raw[t], state)|) == 0.0` (khớp tuyệt đối tới từng bit `float32`).
3. **Đọc chung một file cấu hình duy nhất `configs/preprocess_config.yaml` và kiểm tra bằng `feature_meta.json`:**
   - Cả Phần 2, Phần 3 và Phần 4 đều đọc `configs/preprocess_config.yaml` và kiểm tra `feature_dim(cfg) == 207`, `target_len == 48`.

---

## III. Sản phẩm đầu ra đã hoàn thiện (Checklist bàn giao Phần 2)

- [x] **Script trích landmark từ toàn bộ dữ liệu thực tế (4,854 video):** `src/preprocess/landmark_schema.py`, `src/preprocess/extract_landmarks.py` → `data/landmarks/{train,val,test}_landmarks_index.csv` và **4,854** file raw `.npz`.
- [x] **Module hàm chuẩn hóa & xử lý chuỗi dùng chung (import trực tiếp ở Phần 4):** `src/preprocess/normalize.py` (`normalize_hand`, `normalize_pose`, `normalize_frame`, `normalize_sequence`, `feature_dim`, `feature_names`) và `src/preprocess/sequence.py` (`interpolate_missing`, `trim_idle_frames`, `resample_sequence`, `pad_or_truncate`, `to_fixed_length`).
- [x] **Bộ đặc trưng `.npy`/`.npz` thực tế cho `train`, `val`, `test`:** `data/features/train.npz` `(4251, 48, 207)`, `data/features/val.npz` `(294, 48, 207)`, `data/features/test.npz` `(309, 48, 207)`, cùng `data/features/feature_meta.json` và `data/features/preprocess_summary.json` (`mode: "production"`).
- [x] **Tài liệu kỹ thuật mô tả công thức, shape và ý nghĩa từng chiều:** `docs/preprocess_guide.md` và phiếu tìm hiểu `docs/phieu_tim_hieu_phan_2_thien.md` + `docs/Phieu_Tim_Hieu_Phan_2_Thien.docx`.

---

## IV. Ghi chú của thành viên (Phần 2 — Thiên)

1. **Thống kê thực tế sau khi trích xuất MediaPipe Holistic & chuẩn hóa trên 4,854 video (`data/features/preprocess_summary.json`):**
   - **Tập `train` (`train_augmented.csv`):** **4,251 / 4,251** mẫu thành công (`X.shape = (4251, 48, 207)`, dung lượng nén **140.44 MiB**), **22** người ký hiệu, tỷ lệ khung hình phát hiện bàn tay trung bình **93.67%**, pose **100.00%**, face **83.34%**.
   - **Tập `val` (`val.csv`):** **294 / 294** mẫu thành công (`X.shape = (294, 48, 207)`, dung lượng nén **9.82 MiB**), **4** người ký hiệu độc lập, tỷ lệ phát hiện bàn tay trung bình **94.55%**, pose **100.00%**, face **76.40%**.
   - **Tập `test` (`test.csv`):** **309 / 309** mẫu thành công (`X.shape = (309, 48, 207)`, dung lượng nén **9.94 MiB**), **5** người ký hiệu độc lập, tỷ lệ phát hiện bàn tay trung bình **90.13%**, pose **100.00%**, face **62.93%**.
   - **Bảo toàn mẫu 1-đối-1 & chống rò rỉ:** `skip_zero_hands=False` giữ nguyên 100% số mẫu của Phần 1 (`4,251 + 294 + 309 = 4,854`, số mẫu không phát hiện được tay nào `zero_hand_samples_count = 0`: train=0, val=0, test=0); kiểm định `signer_overlap_check_passed = true` (0 người ký hiệu trùng giữa `train`, `val`, `test`), `0` giá trị `NaN`/`Inf`.
2. **Hướng dẫn bàn giao nhanh cho Hiếu (Phần 3 — Huấn luyện LSTM / BiLSTM):**
   - Nạp trực tiếp bằng `data = np.load("data/features/train.npz")` (`X = data["X"]` shape `(4251, 48, 207)` `float32`, `y = data["y"]` shape `(4251,)` `int64` từ `0..29`, `mask = data["mask"]`).
   - Toàn bộ 3 file `train.npz`, `val.npz`, `test.npz` chỉ nặng tổng cộng **160.20 MiB**, đã được đẩy trực tiếp lên GitHub nên Hiếu có thể tải về hoặc mở trên Google Colab/Kaggle huấn luyện ngay lập tức mà không cần chạy lại MediaPipe.
3. **Hướng dẫn bàn giao nhanh cho Linh (Phần 4 — Triển khai thời gian thực):**
   - Trong vòng lặp webcam OpenCV, sau khi gọi `results = holistic.process(rgb_frame)`, gọi `frame_dict = results_to_arrays(results)` rồi `vec = normalize_frame(frame_dict, cfg, state=state)` để nhận vector `(207,) float32`, đẩy vào cửa sổ trượt `deque(maxlen=48)` và gọi `X_in, _ = to_fixed_length(np.stack(buffer), None, cfg)` để có tensor `(48, 207)` đưa vào ONNX Runtime / TFLite.

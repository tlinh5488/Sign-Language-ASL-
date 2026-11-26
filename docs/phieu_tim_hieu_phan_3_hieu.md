# PHIẾU TÌM HIỂU VÀ PHÂN CÔNG NHÓM — PHẦN 3: MÔ HÌNH (HIẾU)

- **Đề tài:** Mô hình nhận diện ngôn ngữ ký hiệu (Ngôn ngữ ký hiệu tay tiếng Việt — VSL), 30 lớp
- **Giai đoạn phụ trách:** Phần 3 — Huấn luyện (Baseline LSTM → BiLSTM nhiều tầng → Tinh chỉnh siêu tham số → Xuất ONNX)
- **Người phụ trách:** Hiếu (phần đánh giá chuyên sâu F1/confusion matrix/phân tích lỗi: Thiên)
- **Nhận bàn giao từ:** Thiên (Phần 2) — `data/features/{train,val,test}.npz`, shape `(N, 48, 207)`: train **4.251** (22 người), val **294** (4 người), test **309** (5 người), 0 người trùng giữa các tập
- **Bàn giao cho:** Linh (Phần 4) — `artifacts/final/model.onnx` + `model_card.json`; Thiên (đánh giá) — `predictions_{val,test}.npz`
- **Môi trường chạy thực tế:** CPU 8 luồng, không GPU, PyTorch 2.14 (CPU), ONNX Runtime 1.23

---

## I. Tóm tắt kết quả

| Bước | Nội dung | Kết quả |
| :--- | :--- | :--- |
| **3.1 Baseline** | LSTM 1 chiều, 1 tầng, 128 ẩn (176K tham số) | val F1 **0.897 ± 0.014**, test F1 **0.848 ± 0.005** (3 seed) |
| **3.2 Mô hình cải tiến** | BiLSTM nhiều tầng (1/2/3 tầng) + biến thể Attention | Với cùng số tầng, BiLSTM luôn tốt hơn LSTM; 3 tầng có val cao nhất |
| **3.3 Tinh chỉnh** | lr, batch size, dropout, hidden, augmentation (one-factor-at-a-time) + gộp + 3 seed | Cấu hình cuối: **BiLSTM 3 tầng, hidden 128, dropout 0.5, lr 1e-3, batch 64** |
| **3.4 Xuất mô hình** | ONNX opset 17, kiểm tra khớp PyTorch trên toàn bộ 309 mẫu test | Sai khác tối đa **3.3e-6**, argmax trùng **100%** (batch 1 và batch 64) |

**Mô hình bàn giao** (`artifacts/final/`, run `B_dropout_0.5`, seed 42, 1.089.822 tham số):

| Tập | Accuracy | Top-3 | Macro-P | Macro-R | **Macro-F1** | Weighted-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| val (294) | 0.935 | 0.959 | 0.941 | 0.939 | **0.938** | 0.935 |
| test (309) | 0.851 | 0.935 | 0.861 | 0.853 | **0.851** | 0.849 |

Cùng cấu hình qua 3 seed (42/7/2024): val F1 **0.929 ± 0.010**, test F1 **0.854 ± 0.002**, test acc **0.853 ± 0.002**.

### So sánh 3 ứng viên chính (mỗi ứng viên 3 seed, TB ± độ lệch chuẩn)

| Cấu hình | Tham số | Giây/epoch (CPU) | val macro-F1 | test macro-F1 |
| :--- | ---: | :---: | :---: | :---: |
| Baseline LSTM 1 tầng | 176K | 3.2 | 0.897 ± 0.014 | 0.848 ± 0.005 |
| BiLSTM 2 tầng, dropout 0.3 | 695K | 9.3 | 0.917 ± 0.001 | 0.865 ± 0.021 |
| **BiLSTM 3 tầng, dropout 0.5 (chọn)** | 1.09M | 12.9 | **0.929 ± 0.010** | 0.854 ± 0.002 |

> **Lưu ý trung thực về việc chọn mô hình.** Mô hình được chọn **chỉ theo val macro-F1**; tập test không được dùng để chọn. BiLSTM 2 tầng từng đạt test 0.894 ở seed 42, nhưng 2 seed khác chỉ đạt 0.858 và 0.843 — con số 0.894 là may mắn của một seed. Trên test, ba ứng viên chênh nhau ≤ 1.7 điểm, nhỏ hơn độ dao động giữa các seed của BiLSTM 2 tầng. Kết luận chắc chắn: **BiLSTM tốt hơn baseline LSTM rõ rệt trên val (+2 đến +3 điểm) và nhỉnh hơn trên test**; còn 2 tầng hay 3 tầng thì chưa phân biệt được với lượng dữ liệu val/test hiện tại.

---

## II. Trả lời câu hỏi tìm hiểu (Phần 3 — Hiếu)

### Câu hỏi 1: LSTM và BiLSTM khác nhau thế nào; vì sao BiLSTM có thể tốt hơn cho chuỗi ký hiệu?

1. **LSTM (Long Short-Term Memory)** đọc chuỗi theo **một chiều** từ frame 1 → 48. Tại mỗi bước *t*, ô nhớ $c_t$ được cập nhật qua 3 cổng (quên $f_t$, vào $i_t$, ra $o_t$), giúp giữ thông tin xa hơn RNN thường mà không bị tiêu biến gradient. Trạng thái ẩn $h_t$ chỉ chứa thông tin **quá khứ** (frame 1..t). Baseline của dự án (`VSLLSTMBaseline`) lấy $h$ tại frame hợp lệ cuối cùng làm vector đại diện rồi đưa qua lớp Linear 30 lớp.
2. **BiLSTM (Bidirectional LSTM)** chạy **hai LSTM song song**: một xuôi (1 → 48) và một ngược (48 → 1), rồi nối $[\overrightarrow{h_t}; \overleftarrow{h_t}]$ (256 chiều với hidden 128). Mỗi frame vì vậy "nhìn thấy" cả **trước và sau** nó.
3. **Vì sao hợp với ký hiệu:**
   - Một ký hiệu là cả một **cử động hoàn chỉnh** đã quay xong (không phải dự đoán tương lai), nên dùng được thông tin hai chiều.
   - Nhiều cặp ký hiệu **giống nhau ở đầu, khác nhau ở cuối** (ví dụ *ông nội / ông ngoại*, *buổi sáng / buổi trưa / buổi tối*). LSTM một chiều phải "nhớ" phần đầu tới tận frame cuối; BiLSTM có chiều ngược nên frame đầu cũng biết kết thúc ra sao.
   - Mô hình BiLSTM của dự án (`VSLStackedBiLSTM`) lấy **trung bình theo thời gian (masked mean)** của mọi frame thay vì chỉ frame cuối, nên ít phụ thuộc vào vài frame cuối bị nhiễu/mất tay.
4. **Thực nghiệm (cùng siêu tham số, seed 42):** LSTM 1 tầng val 0.908 / test 0.855 so với BiLSTM 1 tầng val 0.911 / test 0.866; LSTM 2 tầng val 0.894 / test 0.822 so với BiLSTM 2 tầng val 0.918 / test 0.894. BiLSTM cũng **hội tụ nhanh hơn rất nhiều**: đạt val F1 ≈ 0.80 ngay epoch 1, còn LSTM cần khoảng 6 epoch (hình `docs/figures/hieu_stage_a_curves.png`).
5. **Cái giá:** BiLSTM có số tham số khoảng gấp đôi và chạy chậm hơn khoảng 2–3 lần, đồng thời **không dùng được cho dự đoán từng frame trực tuyến**. Điều này không ảnh hưởng tới Phần 4 vì Linh gom cửa sổ trượt 48 frame rồi mới suy luận.

### Câu hỏi 2: Số tầng, số đơn vị ẩn, dropout; chống overfitting bằng early stopping, regularization

**Kết quả tinh chỉnh** (gốc: BiLSTM 3 tầng, hidden 128, dropout 0.3, lr 1e-3, batch 64; mỗi lần đổi **một** yếu tố):

| Yếu tố | Giá trị thử | val macro-F1 | test macro-F1 | Nhận xét |
| :--- | :--- | :--- | :--- | :--- |
| Số tầng (Stage A) | 1 / 2 / **3** | 0.911 / 0.918 / **0.925** | 0.866 / 0.894 / 0.860 | Sâu hơn giúp val nhưng không ổn định trên test |
| Learning rate | 5e-4 / **1e-3** / 3e-3 | 0.919 / **0.925** / 0.916 | 0.867 / 0.860 / 0.853 | 1e-3 (mặc định của AdamW) là tốt nhất |
| Batch size | 32 / **64** / 128 | 0.9248 / 0.9247 / 0.9242 | 0.869 / 0.860 / 0.867 | Gần như không ảnh hưởng (32 chỉ hơn 0.0001, bằng nhiễu); batch 32 chậm gấp khoảng 5 lần trên CPU. Cấu hình cuối dùng 64 |
| Dropout | 0.1 / 0.3 / **0.5** | 0.915 / 0.925 / **0.938** | 0.866 / 0.860 / 0.851 | Dropout cao giảm khoảng cách train–val từ 7.5 xuống 6.5 điểm |
| Hidden | **128** / 256 | 0.925 / 0.928 | 0.860 / 0.822 | 256 có 4.3M tham số, chậm gấp khoảng 7 lần, **test giảm mạnh** → overfit, bỏ |
| Augmentation landmark | **bật** / tắt | 0.925 / 0.922 | 0.860 / 0.878 | Khác biệt nằm trong nhiễu seed |
| Gộp tất cả "người thắng" | 3 tầng, h256, d0.5, b32 | 0.916 | 0.844 | **Gộp không cộng dồn** → tệ hơn từng yếu tố riêng lẻ |

**Các biện pháp chống overfitting đang dùng** (`src/train/train.py`):

1. **Early stopping** theo val macro-F1, `patience = 7`: dừng khi 7 epoch liên tiếp không cải thiện và nạp lại checkpoint tốt nhất. Ở 21 lần chạy, best epoch nằm trong khoảng 12–27 (tối đa 30 epoch).
2. **Dropout**: giữa các tầng LSTM, trước lớp phân loại, và dropout/2 sau lớp chiếu đầu vào.
3. **Weight decay** (AdamW, 1e-4) và **label smoothing** 0.05, giúp mô hình bớt quá tự tin; điều này có ích cho ngưỡng tin cậy ở Phần 4.
4. **Gradient clipping** (norm 1.0), tránh bùng nổ gradient khi dùng LSTM.
5. **Cosine learning-rate schedule**: lr giảm dần về 0 theo số epoch.
6. **Augmentation trên landmark** lúc train (nhiễu Gaussian 0.012, co giãn ±5%, che ngẫu nhiên 1–4 frame), cộng thêm augmentation video 2x của Quy.

**Dấu hiệu overfit còn lại:** train accuracy đạt **100%** trong khi val 92–94% và test 85%. Nguyên nhân chính **không nằm ở mô hình** mà ở **khác biệt giữa người ký hiệu** (xem mục IV). Vì vậy tăng regularization thêm cũng không kéo được test lên.

### Câu hỏi 3: Huấn luyện trên Colab/Kaggle, lưu checkpoint, ghi lại kết quả từng lần thử

1. **Chạy được ở mọi nơi**: script tự chọn `cuda` nếu có GPU, nếu không thì chạy CPU. Toàn bộ 17 + 4 lần chạy của phiếu này được huấn luyện **trên CPU** (BiLSTM khoảng 13 giây/epoch, mỗi lần chạy 4–7 phút). Trên Colab/Kaggle chỉ cần:
   ```bash
   !git clone <repo> && cd <repo> && git lfs pull        # lấy data/features/*.npz
   !pip install -r requirements.txt onnx onnxruntime
   !python scripts/run_hieu_experiments.py --config configs/train_experiments.yaml
   ```
   Trên Colab nên chép `artifacts/` sang Google Drive sau mỗi lần chạy, vì phiên Colab có thể bị ngắt.
2. **Checkpoint**: `best_model.pt` chỉ được lưu khi val macro-F1 cải thiện. File chứa `model_state_dict` cùng **toàn bộ cấu hình** (arch, số tầng, hidden, dropout, lr, seed, `class_names`, `input_dim`, `seq_len`). Nhờ vậy `build_model_from_checkpoint()` dựng lại đúng mô hình mà không cần nhớ tham số.
3. **Ghi lại kết quả từng lần thử**: mỗi lần chạy có một thư mục `artifacts/experiments/<tên>/` gồm `train_history.json` (loss/acc/F1 từng epoch), `eval_report.json` (chỉ số val + test, confusion matrix, ONNX parity) và `predictions_{val,test}.npz`. Script tổng hợp ra `results.csv` / `results.md` (bản sao: [docs/hieu_experiment_results.md](hieu_experiment_results.md)).
4. **Chạy lại không mất công**: lần nào đã có `eval_report.json` sẽ tự bỏ qua, nên khi bị ngắt giữa chừng chỉ cần chạy lại cùng lệnh.
5. **Tái lập kết quả**: cố định seed cho PyTorch, NumPy và DataLoader. Mọi kết luận quan trọng đều được kiểm tra lại với **3 seed**.

### Câu hỏi 4 (chung với Thiên): Xuất mô hình sang ONNX/TFLite và kiểm tra kết quả sau khi xuất

1. **Chọn ONNX** thay vì TFLite, vì mô hình viết bằng PyTorch nên xuất ONNX là trực tiếp (`torch.onnx.export`, opset 17), còn TFLite phải đi vòng PyTorch → ONNX → TensorFlow → TFLite rất dễ lỗi với LSTM. ONNX Runtime chạy tốt trên CPU Windows/Linux/macOS, đủ cho demo webcam của Linh.
2. **Giao kèo đầu vào/đầu ra** (giữ nguyên như bản demo của Linh, không cần sửa code Phần 4):

   | Tên | Kiểu | Shape | Ý nghĩa |
   | :--- | :--- | :--- | :--- |
   | `features` | float32 | `(batch, 48, 207)` | Đầu ra của `normalize_frame` + `process_realtime_window` (Phần 2) |
   | `mask` | bool | `(batch, 48)` | `True` = frame hợp lệ (với `resample` luôn toàn `True`) |
   | `logits` | float32 | `(batch, 30)` | **Chưa softmax**; nhãn = `class_names[argmax]` trong `model_card.json` |

3. **Kiểm tra sau khi xuất** (`verify_onnx_parity`): chạy cả PyTorch lẫn ONNX Runtime trên **toàn bộ 309 mẫu test**, ở **batch 1** (giống realtime) và **batch 64**. Kết quả: sai khác logits tối đa **3.3 × 10⁻⁶**, argmax trùng **100%**. Việc kiểm tra batch khác 1 là cần thiết vì PyTorch cảnh báo LSTM xuất ONNX có thể sai khi đổi batch; thực tế đã kiểm chứng là không sai.
4. Mọi lần chạy (21/21) đều tự xuất ONNX và đều qua kiểm tra này.

---

## III. Sản phẩm đầu ra (checklist bàn giao Phần 3)

| Sản phẩm | Vị trí | Trạng thái |
| :--- | :--- | :---: |
| Mã mô hình: `VSLLSTMBaseline`, `VSLStackedBiLSTM`, `VSLBiLSTMAttention`, `build_model_from_checkpoint` | [src/train/model.py](../src/train/model.py) | ✅ |
| Script huấn luyện, đánh giá, xuất ONNX, kiểm tra ONNX | [src/train/train.py](../src/train/train.py) | ✅ |
| Kế hoạch thí nghiệm và script chạy toàn bộ | [configs/train_experiments.yaml](../configs/train_experiments.yaml), [scripts/run_hieu_experiments.py](../scripts/run_hieu_experiments.py) | ✅ |
| Mô hình baseline LSTM và kết quả đo | `artifacts/experiments/A1_lstm_l1`, `D_lstm_l1_seed*` | ✅ |
| Mô hình BiLSTM cải tiến và bảng so sánh các lần tinh chỉnh | [docs/hieu_experiment_results.md](hieu_experiment_results.md) | ✅ |
| Learning curves, confusion matrix | `docs/figures/hieu_stage_a_curves.png`, `hieu_final_curves.png`, `hieu_final_confusion_test.png` | ✅ |
| **Mô hình xuất cho Linh** | `artifacts/final/model.onnx` + `model_card.json` (+ `best_model.pt`) | ✅ |
| Dự đoán val/test cho Thiên phân tích lỗi | `artifacts/final/predictions_{val,test}.npz` (logits, y_true, y_pred, sample_ids, signer_ids) | ✅ |
| Kiểm thử tự động | [tests/test_hieu_training.py](../tests/test_hieu_training.py) (10 test) | ✅ |

**Lệnh dùng nhanh:**

```powershell
# Chạy lại toàn bộ thí nghiệm (bỏ qua lần đã có kết quả)
python scripts/run_hieu_experiments.py --config configs/train_experiments.yaml

# Huấn luyện một cấu hình riêng lẻ
python -m src.train.train --arch bilstm --num-layers 3 --dropout 0.5 --output-dir artifacts/experiments/my_run

# Linh chạy demo webcam với mô hình mới (đường dẫn mặc định đã trỏ sang artifacts/final)
python scripts/run_webcam_demo.py --model artifacts/final/model.onnx --meta data/features/feature_meta.json
```

---

## IV. Ghi chú của thành viên (Phần 3 — Hiếu) — phản hồi cho Thiên và Quy

1. **Khoảng cách val → test (khoảng 8 điểm) chủ yếu do 2 người ký hiệu, không phải do mô hình.** Accuracy của mô hình bàn giao theo từng người:

   | Người ký hiệu | Tập | Số mẫu | Tỷ lệ bắt được mặt | Tỷ lệ bắt được tay | **Accuracy** |
   | :--- | :---: | :---: | :---: | :---: | :---: |
   | vsl2_signer_006 / 013 / 020 | val | 214 | 96–99% | 96–98% | 97.6–100% |
   | vsl2_signer_025 | val | 80 | **20%** | 88% | **78.8%** |
   | vsl2_signer_007 / 016 / 023 | test | 194 | 97–98% | 84–97% | 93.9–97.0% |
   | vsl2_signer_027 | test | 60 | **1.8%** | 96% | **78.3%** |
   | vsl2_signer_028 | test | 55 | **7.4%** | 79% | **54.5%** |

   Cả 3 người mô hình nhận kém đều là người mà MediaPipe **gần như không bắt được khuôn mặt**, trong khi 6 người còn lại đều ≥ 96%. Đặc trưng 207 chiều không dùng mặt, nên lý do nhiều khả năng là **video của những người này được quay hoặc cắt khung khác** (đầu bị cắt khỏi khung, camera gần/lệch). Khi đó tâm và độ rộng vai dùng để chuẩn hóa bị sai, kéo theo toàn bộ đặc trưng bị lệch.
   - **Đề nghị Thiên:** xem trực tiếp vài video của `signer_027/028/025` để xác nhận; thử chuẩn hóa ít phụ thuộc vai hơn, hoặc augmentation dịch/cắt khung mạnh hơn ở bước tiền xử lý.
   - **Đề nghị Quy:** kiểm tra trong 22 người của tập train có ai cùng kiểu khung hình này không. Nếu không có, cân nhắc đổi 1 người kiểu này vào train (giữ nguyên quy tắc chia theo người), hoặc quay bổ sung theo kiểu khung đó.
2. **Các cặp ký hiệu hay nhầm trên test** (mô hình bàn giao): *ông ngoại → ông nội* (4), *thơm → muốn* (3), *giúp đỡ → mới*, *từ chối → thông minh*, *không quen → trứng*, *ông nội → hứa* (2 mỗi cặp). Các lớp có F1 thấp nhất: *thơm* 0.67, *ông ngoại* 0.70, *không quen* 0.71, *yếu* 0.71, *thông minh* 0.74, *ông nội* 0.75. Cặp *ông nội / ông ngoại* khác nhau chủ yếu ở động tác cuối, nên đây là cặp ưu tiên để Thiên phân tích và Quy bổ sung dữ liệu.
3. **Tập val nhỏ (294 mẫu, 4 người) nên chọn mô hình khá nhiễu**: 1 mẫu tương đương khoảng 0.34% F1. Vì vậy mọi kết luận trong phiếu đều dựa trên 3 seed. Nếu có thời gian, nên chuyển sang **cross-validation theo người** (ví dụ 5 fold trên 26 người VSL-v2) để so sánh kiến trúc tin cậy hơn.
4. **Cho Linh — ngưỡng tin cậy:** mô hình xuất ra logits chưa softmax; `VSLRealtimePredictor` đã tự softmax và làm mượt EMA. Đo trên 309 mẫu test (chưa qua EMA), mẫu đúng có xác suất cao nhất trung vị **0.96**, mẫu sai trung vị **0.85**:

   | Ngưỡng | % mẫu được hiển thị | Accuracy trên phần hiển thị | % mẫu đúng giữ lại | % mẫu sai lọt qua |
   | :---: | :---: | :---: | :---: | :---: |
   | 0.65 (hiện tại) | 94.2% | 89.3% | 98.9% | 67.4% |
   | 0.80 | 91.9% | 91.2% | 98.5% | 54.3% |
   | **0.90** | 87.4% | **94.1%** | 96.6% | 34.8% |
   | 0.93 | 84.5% | 95.4% | 94.7% | 26.1% |

   Ngưỡng 0.65 cho lọt 2/3 số dự đoán sai, nên **đề xuất nâng `conf_threshold` lên khoảng 0.9** (chỉ mất khoảng 3% dự đoán đúng). Linh nên chỉnh lại trên webcam thật. Nếu dùng file `.pt` thay ONNX, predictor đã được cập nhật để tự dựng đúng kiến trúc từ checkpoint.
5. **Hướng cải tiến tiếp theo** (chưa làm): (a) thử đặc trưng 414 chiều có vận tốc (`data/features_vel414/`, Thiên đã chuẩn bị); (b) ensemble 3 seed của cấu hình cuối; (c) sửa lỗi dữ liệu ở mục 1, vì đây có lẽ là cách tăng test F1 hiệu quả nhất.

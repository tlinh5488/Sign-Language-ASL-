# Kết quả thí nghiệm Giai đoạn 3 (Hiếu)

Sinh bởi `scripts/run_hieu_experiments.py` (Stage A–C) + 4 lần chạy seed bổ sung (Stage D). Hình: `docs/figures/hieu_*.png`.

Chọn mô hình theo **val macro-F1**; cột test chỉ để báo cáo.

| stage | run | arch | layers | hidden | dropout | lr | batch | augment | seed | params | best_epoch | sec_per_epoch | train_acc | val_acc | val_macro_f1 | test_acc | test_macro_f1 | train_val_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | A1_lstm_l1 | lstm | 1 | 128 | 0.3 | 0.001 | 64 | True | 42 | 176,414 | 22 | 3.18 | 0.9979 | 0.9048 | 0.9083 | 0.8544 | 0.8546 | 0.0931 |
| A | A2_lstm_l2 | lstm | 2 | 128 | 0.3 | 0.001 | 64 | True | 42 | 308,510 | 15 | 4.38 | 0.9753 | 0.8912 | 0.8937 | 0.822 | 0.8221 | 0.0841 |
| A | A3_bilstm_l1 | bilstm | 1 | 128 | 0.3 | 0.001 | 64 | True | 42 | 299,294 | 14 | 4.48 | 0.9974 | 0.9048 | 0.9105 | 0.8641 | 0.8657 | 0.0926 |
| A | A4_bilstm_l2 | bilstm | 2 | 128 | 0.3 | 0.001 | 64 | True | 42 | 694,558 | 22 | 9.3 | 1.0 | 0.915 | 0.9181 | 0.8932 | 0.8938 | 0.085 |
| A | A5_bilstm_l3 | bilstm | 3 | 128 | 0.3 | 0.001 | 64 | True | 42 | 1,089,822 | 14 | 14.55 | 1.0 | 0.9252 | 0.9247 | 0.8576 | 0.8601 | 0.0748 |
| A | A6_bilstm_attn_l2 | bilstm_attn | 2 | 128 | 0.3 | 0.001 | 64 | True | 42 | 776,863 | 12 | 11.36 | 0.9986 | 0.9252 | 0.9274 | 0.8738 | 0.8759 | 0.0734 |
| B | B_lr_0.0005 | bilstm | 3 | 128 | 0.3 | 0.0005 | 64 | True | 42 | 1,089,822 | 16 | 14.44 | 1.0 | 0.9184 | 0.9192 | 0.8673 | 0.8668 | 0.0816 |
| B | B_lr_0.003 | bilstm | 3 | 128 | 0.3 | 0.003 | 64 | True | 42 | 1,089,822 | 18 | 15.15 | 1.0 | 0.915 | 0.9157 | 0.8544 | 0.8532 | 0.085 |
| B | B_batch_size_32 | bilstm | 3 | 128 | 0.3 | 0.001 | 32 | True | 42 | 1,089,822 | 13 | 69.34 | 0.9967 | 0.9218 | 0.9248 | 0.8706 | 0.8691 | 0.0749 |
| B | B_batch_size_128 | bilstm | 3 | 128 | 0.3 | 0.001 | 128 | True | 42 | 1,089,822 | 16 | 27.94 | 0.9991 | 0.9218 | 0.9242 | 0.8673 | 0.8674 | 0.0773 |
| B | B_dropout_0.1 | bilstm | 3 | 128 | 0.1 | 0.001 | 64 | True | 42 | 1,089,822 | 13 | 13.41 | 0.996 | 0.915 | 0.9148 | 0.8641 | 0.8658 | 0.081 |
| B | B_dropout_0.5 | bilstm | 3 | 128 | 0.5 | 0.001 | 64 | True | 42 | 1,089,822 | 23 | 12.94 | 1.0 | 0.9354 | 0.9376 | 0.8511 | 0.851 | 0.0646 |
| B | B_hidden_dim_256 | bilstm | 3 | 256 | 0.3 | 0.001 | 64 | True | 42 | 4,276,766 | 14 | 94.22 | 0.9929 | 0.9252 | 0.9283 | 0.8252 | 0.8218 | 0.0677 |
| B | B_augment_false | bilstm | 3 | 128 | 0.3 | 0.001 | 64 | False | 42 | 1,089,822 | 16 | 12.45 | 0.9986 | 0.9218 | 0.9224 | 0.877 | 0.878 | 0.0768 |
| C | C_combined_seed42 | bilstm | 3 | 256 | 0.5 | 0.001 | 32 | True | 42 | 4,276,766 | 25 | 41.98 | 1.0 | 0.915 | 0.9157 | 0.8447 | 0.8444 | 0.085 |
| C | C_final_seed7 | bilstm | 3 | 128 | 0.5 | 0.001 | 64 | True | 7 | 1,089,822 | 24 | 13.17 | 1.0 | 0.932 | 0.9348 | 0.8544 | 0.8552 | 0.068 |
| C | C_final_seed2024 | bilstm | 3 | 128 | 0.5 | 0.001 | 64 | True | 2024 | 1,089,822 | 16 | 13.44 | 0.9991 | 0.9116 | 0.9154 | 0.8544 | 0.8563 | 0.0875 |
| D | D_lstm_l1_seed7 | lstm | 1 | 128 | 0.3 | 0.001 | 64 | True | 7 | 176,414 | 23 | 2.81 | 0.9969 | 0.8741 | 0.8767 | 0.8414 | 0.8415 | 0.1228 |
| D | D_lstm_l1_seed2024 | lstm | 1 | 128 | 0.3 | 0.001 | 64 | True | 2024 | 176,414 | 27 | 2.57 | 0.9998 | 0.9014 | 0.9049 | 0.8479 | 0.847 | 0.0984 |
| D | D_bilstm_l2_seed7 | bilstm | 2 | 128 | 0.3 | 0.001 | 64 | True | 7 | 694,558 | 20 | 9.81 | 1.0 | 0.915 | 0.9153 | 0.8576 | 0.8576 | 0.085 |
| D | D_bilstm_l2_seed2024 | bilstm | 2 | 128 | 0.3 | 0.001 | 64 | True | 2024 | 694,558 | 14 | 9.58 | 1.0 | 0.915 | 0.9166 | 0.8414 | 0.8433 | 0.085 |

## Stage D — So sánh 3 ứng viên qua 3 seed (42 / 7 / 2024)

| Cấu hình | val macro-F1 từng seed | val TB ± std | test macro-F1 từng seed | test TB ± std |
|---|---|---|---|---|
| Baseline LSTM 1 tầng | [0.9083, 0.8767, 0.9049] | 0.8966 ± 0.0142 | [0.8546, 0.8415, 0.847] | 0.8477 ± 0.0054 |
| BiLSTM 2 tầng, dropout 0.3 | [0.9181, 0.9153, 0.9166] | 0.9167 ± 0.0011 | [0.8938, 0.8576, 0.8433] | 0.8649 ± 0.0213 |
| BiLSTM 3 tầng, dropout 0.5 (bàn giao) | [0.9376, 0.9348, 0.9154] | 0.9293 ± 0.0099 | [0.851, 0.8552, 0.8563] | 0.8542 ± 0.0023 |

Stage D chỉ dùng để kiểm tra lại quyết định: chọn theo **trung bình val** vẫn là BiLSTM 3 tầng dropout 0.5 → giữ nguyên mô hình bàn giao.

## Quyết định từng bước

- Stage A: baseline LSTM tốt nhất = A1_lstm_l1 (val F1 0.9083); BiLSTM tốt nhất = A5_bilstm_l3 (val F1 0.9247) -> làm gốc cho Stage B.
- Stage B [lr]: thử [0.001, 0.0005, 0.003] -> val F1 [0.9247, 0.9192, 0.9157] -> chọn 0.001.
- Stage B [batch_size]: thử [64, 32, 128] -> val F1 [0.9247, 0.9248, 0.9242] -> chọn 32.
- Stage B [dropout]: thử [0.3, 0.1, 0.5] -> val F1 [0.9247, 0.9148, 0.9376] -> chọn 0.5.
- Stage B [hidden_dim]: thử [128, 256] -> val F1 [0.9247, 0.9283] -> chọn 256.
- Stage B [augment]: thử [True, False] -> val F1 [0.9247, 0.9224] -> chọn True.
- Stage C: gộp {'lr': 0.001, 'batch_size': 32, 'dropout': 0.5, 'hidden_dim': 256, 'augment': True} -> val F1 0.9157.
- Cấu hình cuối (val F1 cao nhất) = B_dropout_0.5: {'arch': 'bilstm', 'num_layers': 3, 'hidden_dim': 128, 'dropout': 0.5, 'lr': 0.001, 'batch_size': 64, 'augment': True}.
- Mô hình bàn giao = B_dropout_0.5 (seed 42, val F1 cao nhất trong các seed).

## Độ ổn định theo seed (cấu hình cuối)

```json
{
  "seeds": [
    42,
    7,
    2024
  ],
  "val_macro_f1": {
    "mean": 0.9293,
    "std": 0.0099
  },
  "test_macro_f1": {
    "mean": 0.8542,
    "std": 0.0023
  },
  "test_accuracy": {
    "mean": 0.8533,
    "std": 0.0016
  }
}
```

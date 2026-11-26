"""Kiến trúc mô hình Giai đoạn 3 (Hiếu) cho nhận diện VSL 30 lớp từ chuỗi landmark.

Ba kiến trúc, cùng một giao kèo đầu vào/đầu ra để Phần 4 (Linh) dùng chung:
    forward(x: (B, L, D) float32, mask: (B, L) bool) -> logits (B, num_classes)

- ``lstm``        : Baseline LSTM một chiều, lấy hidden state tại frame hợp lệ cuối cùng.
- ``bilstm``      : BiLSTM nhiều tầng (stacked), gộp masked-mean theo thời gian.
- ``bilstm_attn`` : Bản thử nghiệm Linear + Conv1D + BiLSTM + Temporal Attention.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Trung bình ``(B, L, H)`` theo thời gian, chỉ tính các frame ``mask=True``."""
    if mask is None:
        return x.mean(dim=1)
    w = mask.to(x.dtype).unsqueeze(-1)  # (B, L, 1)
    return (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)


def last_valid_step(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Lấy output tại frame hợp lệ cuối cùng của mỗi chuỗi (dùng cho LSTM một chiều)."""
    if mask is None:
        return x[:, -1, :]
    L = x.shape[1]
    # Chỉ số frame hợp lệ cuối cùng; chuỗi rỗng -> frame 0.
    pos = torch.arange(L, device=x.device).unsqueeze(0).expand_as(mask)
    idx = torch.where(mask.bool(), pos, torch.zeros_like(pos)).amax(dim=1)  # (B,)
    idx = idx.view(-1, 1, 1).expand(-1, 1, x.shape[2])
    return x.gather(1, idx).squeeze(1)


class VSLLSTMBaseline(nn.Module):
    """Baseline: LSTM một chiều nhiều tầng -> hidden tại frame cuối -> Linear."""

    def __init__(
        self,
        input_dim: int = 207,
        num_classes: int = 30,
        *,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(last_valid_step(out, mask))


class VSLStackedBiLSTM(nn.Module):
    """Mô hình cải tiến: Linear chiếu đầu vào -> BiLSTM nhiều tầng -> masked mean -> Linear."""

    def __init__(
        self,
        input_dim: int = 207,
        num_classes: int = 30,
        *,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        proj_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        proj_dim = int(proj_dim or hidden_dim)
        self.in_proj = nn.Sequential(
            nn.Linear(self.input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.lstm(self.in_proj(x))
        return self.head(masked_mean_pool(out, mask))


class TemporalAttentionPool(nn.Module):
    """Gộp chuỗi ``(B, L, H)`` thành vector ``(B, H)`` theo trọng số attention có mask."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.proj(x).squeeze(-1)  # (B, L)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e4)
        attn = torch.softmax(scores, dim=1)  # (B, L)
        return torch.bmm(attn.unsqueeze(1), x).squeeze(1)


class VSLBiLSTMAttention(nn.Module):
    """Biến thể thử nghiệm: Linear + Conv1D residual + BiLSTM + Temporal Attention."""

    def __init__(
        self,
        input_dim: int = 207,
        num_classes: int = 30,
        *,
        proj_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

        self.in_proj = nn.Sequential(
            nn.Linear(self.input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.conv1d = nn.Sequential(
            nn.Conv1d(proj_dim, proj_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.pool = TemporalAttentionPool(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, L, D), mask: (B, L) bool
        h = self.in_proj(x)
        h = self.conv1d(h.transpose(1, 2)).transpose(1, 2) + h
        out, _ = self.lstm(h)
        pooled = self.pool(out, mask)
        return self.head(pooled)


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "lstm": VSLLSTMBaseline,
    "bilstm": VSLStackedBiLSTM,
    "bilstm_attn": VSLBiLSTMAttention,
}


def build_model(
    arch: str,
    input_dim: int,
    num_classes: int,
    *,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
) -> nn.Module:
    """Dựng mô hình theo tên kiến trúc trong ``MODEL_REGISTRY``."""
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Kiến trúc không hợp lệ: {arch!r}. Chọn một trong {sorted(MODEL_REGISTRY)}")
    kwargs: dict[str, Any] = {"hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout}
    if arch == "bilstm_attn":
        kwargs["proj_dim"] = hidden_dim
    return MODEL_REGISTRY[arch](input_dim=input_dim, num_classes=num_classes, **kwargs)


def build_model_from_checkpoint(ckpt: Mapping[str, Any]) -> nn.Module:
    """Dựng lại mô hình từ checkpoint ``best_model.pt`` (tương thích checkpoint cũ không có ``arch``)."""
    model = build_model(
        str(ckpt.get("arch", "bilstm_attn")),
        input_dim=int(ckpt["input_dim"]),
        num_classes=int(ckpt["num_classes"]),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        num_layers=int(ckpt.get("num_layers", 2)),
        dropout=float(ckpt.get("dropout", 0.3)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

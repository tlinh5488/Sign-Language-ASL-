"""Kiến trúc nhẹ 1D-CNN + BiLSTM + Temporal Attention cho nhận diện VSL 30 lớp."""

from __future__ import annotations

import torch
import torch.nn as nn


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
    """Mô hình nhẹ ~450k tham số, train nhanh trên cả CPU và GPU phổ thông."""

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


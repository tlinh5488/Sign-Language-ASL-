"""PyTorch Dataset và Landmark-space Augmentation cho Giai đoạn 3 (Hiếu)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def load_npz_split(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Nạp file ``{train,val,test}.npz`` từ Bước 2.4 với ``allow_pickle=False``."""
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file đặc trưng: {p}")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


class VSLFeatureDataset(Dataset):
    """Dataset chuỗi đặc trưng ``(N, L, D)`` kèm augmentation nhẹ trên landmark lúc train."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        augment: bool = False,
        noise_std: float = 0.012,
        scale_jitter: float = 0.05,
        time_mask_prob: float = 0.25,
        max_time_mask: int = 4,
        seed: int = 42,
    ) -> None:
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.mask = (
            np.ones(self.X.shape[:2], dtype=bool)
            if mask is None
            else np.asarray(mask, dtype=bool)
        )
        self.augment = bool(augment)
        self.noise_std = float(noise_std)
        self.scale_jitter = float(scale_jitter)
        self.time_mask_prob = float(time_mask_prob)
        self.max_time_mask = int(max_time_mask)
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_npz(
        cls,
        npz_path: str | Path,
        *,
        augment: bool = False,
        **kwargs: Any,
    ) -> "VSLFeatureDataset":
        data = load_npz_split(npz_path)
        return cls(
            data["X"],
            data["y"],
            data.get("mask"),
            augment=augment,
            **kwargs,
        )

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def _augment_sample(self, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        out = x.copy()
        valid = m[:, None]
        # 1. Scale jitter nhẹ mô phỏng sai số tỉ lệ vai/tay khi camera ở góc khác
        if self.scale_jitter > 0.0:
            s = float(self.rng.uniform(1.0 - self.scale_jitter, 1.0 + self.scale_jitter))
            out = out * s
        # 2. Nhiễu Gaussian nhẹ trên các frame hợp lệ (chỉ tác động lên giá trị khác 0)
        if self.noise_std > 0.0:
            nonzero = (np.abs(out) > 1e-7) & valid
            noise = self.rng.normal(0.0, self.noise_std, size=out.shape).astype(np.float32)
            out = np.where(nonzero, out + noise, out)
        # 3. Che đoạn thời gian ngắn ngẫu nhiên (SpecAugment-style) tăng bền mất tay
        L = out.shape[0]
        if self.time_mask_prob > 0.0 and self.max_time_mask > 0 and L > self.max_time_mask + 2:
            if float(self.rng.random()) < self.time_mask_prob:
                w = int(self.rng.integers(1, self.max_time_mask + 1))
                t0 = int(self.rng.integers(0, L - w))
                out[t0 : t0 + w] = 0.0
        return out

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        m = self.mask[idx]
        if self.augment:
            x = self._augment_sample(x, m)
        return (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(m, dtype=bool)),
            torch.tensor(int(self.y[idx]), dtype=torch.long),
        )


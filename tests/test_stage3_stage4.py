"""Tests cho các phần mở rộng Phần 2 (geometric + smooth), Phần 3 (BiLSTM+Attention/ONNX)
và Phần 4 (VSLRealtimePredictor với Activity Gate & EMA)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.predictor import VSLRealtimePredictor
from src.preprocess.landmark_schema import HAND_N, POSE_N
from src.preprocess.normalize import (
    feature_dim,
    feature_names,
    make_state,
    normalize_frame,
    normalize_sequence,
)
from src.preprocess.sequence import smooth_sequence
from src.train.dataset import VSLFeatureDataset
from src.train.model import VSLBiLSTMAttention
from src.train.train import export_onnx


def _synthetic_raw(T: int = 20, seed: int = 11) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    pose = rng.normal(0.5, 0.04, size=(T, POSE_N, 4)).astype(np.float32)
    pose[:, :, 3] = 0.98
    pose[:, 11, :3] = [0.4, 0.5, 0.0]
    pose[:, 12, :3] = [0.6, 0.5, 0.0]
    lh = rng.normal(0.35, 0.03, size=(T, HAND_N, 3)).astype(np.float32)
    rh = rng.normal(0.65, 0.03, size=(T, HAND_N, 3)).astype(np.float32)
    return {
        "pose": pose,
        "left_hand": lh,
        "right_hand": rh,
        "pose_mask": np.ones(T, dtype=bool),
        "left_hand_mask": np.ones(T, dtype=bool),
        "right_hand_mask": np.ones(T, dtype=bool),
    }


def test_geometric_features_and_frame_sequence_consistency() -> None:
    cfg = {"normalization": {"include_geometric_features": True}}
    assert feature_dim(cfg) == 207 + 24
    assert len(feature_names(cfg)) == 231

    raw = _synthetic_raw(T=12)
    seq_feats, _ = normalize_sequence(raw, cfg)
    assert seq_feats.shape == (12, 231)
    assert np.all(np.isfinite(seq_feats))

    state = make_state()
    frame_feats = np.stack(
        [
            normalize_frame({k: v[t] for k, v in raw.items()}, cfg, state=state)
            for t in range(12)
        ],
        axis=0,
    )
    assert np.allclose(seq_feats, frame_feats, atol=1e-5)


def test_smooth_sequence_reduces_jitter_and_preserves_masked_zeros() -> None:
    T, D = 20, 10
    t = np.linspace(0, 2 * np.pi, T, dtype=np.float32)[:, None]
    clean = np.repeat(np.sin(t), D, axis=1)
    rng = np.random.default_rng(0)
    noisy = clean + rng.normal(0.0, 0.15, size=clean.shape).astype(np.float32)
    mask = np.ones(T, dtype=bool)
    mask[8:10] = False
    noisy[~mask] = 0.0

    sm = smooth_sequence(noisy, window=5, mask=mask)
    assert np.allclose(sm[~mask], 0.0)
    err_before = float(np.mean((noisy[mask] - clean[mask]) ** 2))
    err_after = float(np.mean((sm[mask] - clean[mask]) ** 2))
    assert err_after < err_before


def test_dataset_augmentation_and_model_onnx_parity(tmp_path: Path) -> None:
    N, L, D, C = 16, 48, 207, 30
    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 0.5, size=(N, L, D)).astype(np.float32)
    y = np.arange(N, dtype=np.int64) % C
    mask = np.ones((N, L), dtype=bool)

    ds = VSLFeatureDataset(X, y, mask, augment=True, seed=5)
    x_aug, m_aug, y_aug = ds[0]
    assert x_aug.shape == (L, D) and m_aug.shape == (L,) and int(y_aug) == int(y[0])
    assert torch.isfinite(x_aug).all()

    model = VSLBiLSTMAttention(input_dim=D, num_classes=C, proj_dim=64, hidden_dim=64)
    model.eval()
    bx = torch.from_numpy(X[:2])
    bm = torch.from_numpy(mask[:2])
    with torch.no_grad():
        torch_logits = model(bx, bm).numpy()
    assert torch_logits.shape == (2, C)

    onnx_path = tmp_path / "model.onnx"
    export_onnx(model, onnx_path, seq_len=L, input_dim=D)
    assert onnx_path.exists()

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_logits = sess.run(None, {"features": X[:2], "mask": mask[:2]})[0]
    assert np.allclose(torch_logits, ort_logits, atol=1e-4)


def test_realtime_predictor_activity_gate_and_sentence(tmp_path: Path) -> None:
    D, L, C = 207, 48, 30
    model = VSLBiLSTMAttention(input_dim=D, num_classes=C, proj_dim=64, hidden_dim=64)
    onnx_path = tmp_path / "model.onnx"
    export_onnx(model, onnx_path, seq_len=L, input_dim=D)

    meta = {
        "D": D,
        "L": L,
        "num_classes": C,
        "class_list": [{"class_id": i, "label": f"lop_{i}"} for i in range(C)],
        "config_snapshot": {
            "normalization": {"feature_set": "pose_upper+hands", "use_z": True, "add_velocity": False},
            "sequence": {"target_len": L, "method": "resample", "trim_idle": True, "max_gap": 8},
        },
    }
    meta_path = tmp_path / "feature_meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    predictor = VSLRealtimePredictor(
        onnx_path,
        meta_path,
        min_active_frames=4,
        conf_threshold=0.0,  # luôn đủ ngưỡng để kiểm tra cơ chế commit + cooldown
        cooldown_frames=10,
    )

    # 1. Frame không có tay -> active=False
    idle_raw = _synthetic_raw(T=1)
    idle_frame = {
        "pose": idle_raw["pose"][0],
        "left_hand": np.zeros((HAND_N, 3), dtype=np.float32),
        "right_hand": np.zeros((HAND_N, 3), dtype=np.float32),
        "pose_mask": True,
        "left_hand_mask": False,
        "right_hand_mask": False,
    }
    res_idle = predictor.step(idle_frame)
    assert res_idle.active is False
    assert res_idle.sentence == []

    # 2. Đẩy 6 frame có tay -> kích hoạt suy luận & commit đúng 1 từ do cooldown
    active_raw = _synthetic_raw(T=6)
    last_res = None
    for t in range(6):
        last_res = predictor.step({k: v[t] for k, v in active_raw.items()})
    assert last_res is not None and last_res.active is True
    assert last_res.label is not None and len(last_res.sentence) == 1

    predictor.clear_sentence()
    assert predictor.sentence == []


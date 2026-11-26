"""Tests Giai đoạn 3 (Hiếu): kiến trúc LSTM/BiLSTM, mask, ONNX parity, checkpoint, train end-to-end."""

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

from src.train.model import MODEL_REGISTRY, build_model, build_model_from_checkpoint, last_valid_step, masked_mean_pool
from src.train.train import compute_metrics, export_onnx, train_and_evaluate, verify_onnx_parity

L, D, C = 48, 207, 30


@pytest.mark.parametrize("arch", sorted(MODEL_REGISTRY))
def test_model_contract_shapes(arch: str) -> None:
    model = build_model(arch, D, C, hidden_dim=32, num_layers=2).eval()
    x = torch.randn(3, L, D)
    m = torch.ones(3, L, dtype=torch.bool)
    with torch.no_grad():
        out = model(x, m)
    assert out.shape == (3, C) and torch.isfinite(out).all()


def test_pooling_ignores_masked_frames() -> None:
    x = torch.arange(2 * 5 * 1, dtype=torch.float32).view(2, 5, 1)
    m = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.allclose(masked_mean_pool(x, m).squeeze(-1), torch.tensor([1.0, 7.0]))
    assert torch.allclose(last_valid_step(x, m).squeeze(-1), torch.tensor([2.0, 9.0]))


def test_lstm_ignores_trailing_padding() -> None:
    """LSTM một chiều lấy frame hợp lệ cuối -> frame padding phía sau không ảnh hưởng đầu ra.

    (BiLSTM chiều ngược vẫn "thấy" padding; không sao vì Phần 2 resample nên mask luôn toàn True
    cả lúc train lẫn realtime — xem ``to_fixed_length(method="resample")``.)
    """
    torch.manual_seed(0)
    model = build_model("lstm", D, C, hidden_dim=32, num_layers=1).eval()
    x = torch.randn(1, L, D)
    m = torch.ones(1, L, dtype=torch.bool)
    m[:, 40:] = False
    x2 = x.clone()
    x2[:, 40:] = 99.0
    with torch.no_grad():
        assert torch.allclose(model(x, m), model(x2, m), atol=1e-6)


@pytest.mark.parametrize("arch", sorted(MODEL_REGISTRY))
def test_onnx_parity_and_checkpoint_roundtrip(arch: str, tmp_path: Path) -> None:
    model = build_model(arch, D, C, hidden_dim=32, num_layers=2).eval()
    X = np.random.default_rng(0).normal(size=(10, L, D)).astype(np.float32)
    M = np.ones((10, L), dtype=bool)
    onnx_path = tmp_path / "m.onnx"
    export_onnx(model, onnx_path, seq_len=L, input_dim=D)
    parity = verify_onnx_parity(model, onnx_path, X, M, batch_size=4)
    assert parity["passed"], parity

    ckpt = {"model_state_dict": model.state_dict(), "arch": arch, "input_dim": D, "num_classes": C,
            "hidden_dim": 32, "num_layers": 2, "dropout": 0.3}
    rebuilt = build_model_from_checkpoint(ckpt)
    with torch.no_grad():
        assert torch.allclose(model(torch.from_numpy(X), torch.from_numpy(M)),
                              rebuilt(torch.from_numpy(X), torch.from_numpy(M)), atol=1e-6)


def test_compute_metrics_perfect_and_confusions() -> None:
    y = np.array([0, 1, 2, 2])
    logits = np.eye(3)[[0, 1, 2, 1]] * 5.0
    m = compute_metrics(y, logits, ["a", "b", "c"])
    assert m["accuracy"] == 0.75
    assert m["top_confusions"][0] == {"true_id": 2, "pred_id": 1, "true_label": "c", "pred_label": "b", "count": 1}
    assert np.array(m["confusion_matrix"]).sum() == 4


def test_train_end_to_end_tiny(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    fdir = tmp_path / "features"
    fdir.mkdir()
    n_cls = 3
    for split, n in (("train", 30), ("val", 9), ("test", 9)):
        y = np.arange(n) % n_cls
        X = rng.normal(size=(n, 8, 12)).astype(np.float32) + y[:, None, None].astype(np.float32)
        np.savez(fdir / f"{split}.npz", X=X, y=y.astype(np.int64), mask=np.ones((n, 8), bool),
                 sample_ids=np.array([f"{split}_{i}" for i in range(n)]), signer_ids=np.array(["s"] * n))
    (fdir / "feature_meta.json").write_text(json.dumps(
        {"num_classes": n_cls, "class_list": [{"label": f"c{i}"} for i in range(n_cls)]}), encoding="utf-8")

    out = tmp_path / "run"
    rep = train_and_evaluate(fdir, out, arch="bilstm", epochs=3, hidden_dim=16, num_layers=1,
                             batch_size=8, patience=5, verbose=False)
    for f in ("best_model.pt", "model.onnx", "eval_report.json", "train_history.json",
              "predictions_val.npz", "predictions_test.npz"):
        assert (out / f).exists(), f
    assert rep["onnx_parity"]["passed"]
    assert rep["class_names"] == ["c0", "c1", "c2"]
    with np.load(out / "predictions_test.npz") as z:
        assert z["logits"].shape == (9, n_cls) and list(z["sample_ids"][:2]) == ["test_0", "test_1"]

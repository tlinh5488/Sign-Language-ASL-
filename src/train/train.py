"""Huấn luyện, đánh giá và export ONNX cho Giai đoạn 3 (Hiếu).

Một lần chạy = một thí nghiệm, ghi đầy đủ vào ``<output-dir>/``:
    best_model.pt         checkpoint tốt nhất theo val macro-F1 (kèm cấu hình để dựng lại)
    train_history.json    loss/acc/F1 từng epoch
    eval_report.json      chỉ số val + test, ONNX parity
    predictions_val.npz   logits/y/sample_ids/signer_ids tập val  -> Thiên phân tích lỗi
    predictions_test.npz  logits/y/sample_ids/signer_ids tập test -> Thiên phân tích lỗi
    model.onnx            mô hình xuất cho Linh (features, mask) -> logits

Ví dụ:
    python -m src.train.train --arch lstm   --output-dir artifacts/experiments/lstm_baseline
    python -m src.train.train --arch bilstm --num-layers 2 --output-dir artifacts/experiments/bilstm_l2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.dataset import VSLFeatureDataset, load_npz_split
from src.train.model import MODEL_REGISTRY, build_model


def compute_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    class_names: list[str] | None = None,
    top_k_confusions: int = 10,
) -> dict[str, Any]:
    """Accuracy, top-3, precision/recall/F1 (macro + weighted), F1 từng lớp, cặp nhầm nhiều nhất."""
    num_classes = int(logits.shape[1])
    labels = list(range(num_classes))
    names = class_names if class_names and len(class_names) == num_classes else [str(i) for i in labels]
    preds = np.argmax(logits, axis=1)
    top3 = np.argsort(logits, axis=1)[:, -3:]

    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, preds, labels=labels, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, preds, labels=labels, average="weighted", zero_division=0)
    _, _, f_c, sup_c = precision_recall_fscore_support(y_true, preds, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=labels)

    confusions = [
        {"true_id": i, "pred_id": j, "true_label": names[i], "pred_label": names[j], "count": int(cm[i, j])}
        for i in labels
        for j in labels
        if i != j and cm[i, j] > 0
    ]
    confusions.sort(key=lambda r: r["count"], reverse=True)

    return {
        "n": int(len(y_true)),
        "accuracy": round(float(np.mean(preds == y_true)), 4),
        "top3_accuracy": round(float(np.mean([yt in row for yt, row in zip(y_true, top3)])), 4),
        "macro_precision": round(float(p_m), 4),
        "macro_recall": round(float(r_m), 4),
        "macro_f1": round(float(f_m), 4),
        "weighted_f1": round(float(f_w), 4),
        "weighted_precision": round(float(p_w), 4),
        "weighted_recall": round(float(r_w), 4),
        "per_class_f1": {names[c]: round(float(f_c[c]), 4) for c in labels},
        "per_class_support": {names[c]: int(sup_c[c]) for c in labels},
        "top_confusions": confusions[:top_k_confusions],
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_y = [], []
    for x, m, y in loader:
        all_logits.append(model(x.to(device), m.to(device)).cpu().numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_y, axis=0)


def _ce_loss(logits: np.ndarray, y: np.ndarray) -> float:
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def export_onnx(model: nn.Module, onnx_path: Path, seq_len: int, input_dim: int) -> None:
    """Export giao kèo chung cho Phần 4: ``features (B,L,D) float32``, ``mask (B,L) bool`` -> ``logits``."""
    model_cpu = model.to("cpu").eval()
    dummy_x = torch.randn(1, seq_len, input_dim, dtype=torch.float32)
    dummy_m = torch.ones(1, seq_len, dtype=torch.bool)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model_cpu,
        (dummy_x, dummy_m),
        str(onnx_path),
        input_names=["features", "mask"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "mask": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def verify_onnx_parity(
    model: nn.Module,
    onnx_path: Path,
    X: np.ndarray,
    mask: np.ndarray,
    *,
    batch_size: int = 64,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """So sánh logits ONNX Runtime với PyTorch trên toàn bộ ``X`` ở batch=1 (như realtime) và batch lớn."""
    import onnxruntime as ort

    model_cpu = model.to("cpu").eval()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    X = X.astype(np.float32)
    mask = mask.astype(bool)
    with torch.no_grad():
        ref = np.concatenate(
            [model_cpu(torch.from_numpy(X[i : i + 256]), torch.from_numpy(mask[i : i + 256])).numpy() for i in range(0, len(X), 256)]
        )

    out: dict[str, Any] = {"n_samples": int(len(X)), "atol": atol}
    for name, bs in (("batch1", 1), (f"batch{batch_size}", batch_size)):
        try:
            got = np.concatenate(
                [sess.run(None, {"features": X[i : i + bs], "mask": mask[i : i + bs]})[0] for i in range(0, len(X), bs)]
            )
            diff = float(np.abs(got - ref).max())
            agree = float(np.mean(got.argmax(1) == ref.argmax(1)))
            out[name] = {"max_abs_diff": diff, "argmax_agreement": agree, "passed": bool(diff <= atol and agree == 1.0)}
        except Exception as exc:  # pragma: no cover - ghi lại lỗi thay vì dừng cả thí nghiệm
            out[name] = {"error": f"{type(exc).__name__}: {exc}", "passed": False}
    out["passed"] = all(v.get("passed", False) for k, v in out.items() if isinstance(v, dict))
    return out


def train_and_evaluate(
    features_dir: str | Path = "data/features",
    output_dir: str | Path = "artifacts/experiments/my_run",
    *,
    arch: str = "bilstm",
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    augment: bool = True,
    patience: int = 8,
    seed: int = 42,
    export_onnx_model: bool = True,
    num_threads: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if num_threads:
        torch.set_num_threads(int(num_threads))

    fdir = Path(features_dir)
    odir = Path(output_dir)
    odir.mkdir(parents=True, exist_ok=True)

    meta_path = fdir / "feature_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    class_names = [c["label"] for c in meta.get("class_list", [])] or None

    splits = {s: load_npz_split(fdir / f"{s}.npz") for s in ("train", "val", "test")}
    train_ds = VSLFeatureDataset(splits["train"]["X"], splits["train"]["y"], splits["train"].get("mask"), augment=augment, seed=seed)
    val_ds = VSLFeatureDataset(splits["val"]["X"], splits["val"]["y"], splits["val"].get("mask"))
    test_ds = VSLFeatureDataset(splits["test"]["X"], splits["test"]["y"], splits["test"].get("mask"))

    seq_len = int(train_ds.X.shape[1])
    input_dim = int(train_ds.X.shape[2])
    num_classes = int(meta.get("num_classes", int(train_ds.y.max()) + 1))

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(arch, input_dim, num_classes, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    hparams = {
        "arch": arch, "hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout,
        "lr": lr, "batch_size": batch_size, "weight_decay": weight_decay,
        "label_smoothing": label_smoothing, "augment": augment, "epochs": epochs,
        "patience": patience, "seed": seed,
    }
    ckpt_path = odir / "best_model.pt"
    best = {"f1": -1.0, "loss": float("inf"), "epoch": 0}
    no_improve = 0
    history: list[dict[str, Any]] = []

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0
        for x, m, y in train_loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, m)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += float(loss.item()) * int(y.size(0))
            tr_correct += int((logits.argmax(1) == y).sum().item())
            tr_n += int(y.size(0))
        scheduler.step()

        val_logits, val_y = predict_logits(model, val_loader, device)
        val_m = compute_metrics(val_y, val_logits, class_names)
        val_loss = _ce_loss(val_logits, val_y)
        rec = {
            "epoch": epoch,
            "lr": round(float(optimizer.param_groups[0]["lr"]), 7),
            "train_loss": round(tr_loss / max(1, tr_n), 4),
            "train_acc": round(tr_correct / max(1, tr_n), 4),
            "val_loss": round(val_loss, 4),
            "val_acc": val_m["accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "elapsed_sec": round(time.time() - t_start, 1),
        }
        history.append(rec)

        # Early stopping theo val macro-F1; hòa F1 thì ưu tiên val loss thấp hơn.
        improved = val_m["macro_f1"] > best["f1"] or (val_m["macro_f1"] == best["f1"] and val_loss < best["loss"])
        if improved:
            best = {"f1": val_m["macro_f1"], "loss": val_loss, "epoch": epoch}
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    **hparams,
                    "input_dim": input_dim,
                    "seq_len": seq_len,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "features_dir": str(fdir),
                    "best_epoch": epoch,
                },
                ckpt_path,
            )
        else:
            no_improve += 1
        if verbose:
            flag = "*" if improved else " "
            print(
                f"[{arch}] ep{epoch:02d}{flag} tr_loss={rec['train_loss']:.4f} tr_acc={rec['train_acc']:.4f} "
                f"val_loss={rec['val_loss']:.4f} val_acc={rec['val_acc']:.4f} val_f1={rec['val_macro_f1']:.4f} "
                f"({rec['elapsed_sec']}s)",
                flush=True,
            )
        if no_improve >= patience:
            break

    elapsed_sec = round(time.time() - t_start, 1)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    results: dict[str, Any] = {}
    for split, loader in (("val", val_loader), ("test", test_loader)):
        logits, y = predict_logits(model, loader, device)
        results[split] = {"loss": round(_ce_loss(logits, y), 4), **compute_metrics(y, logits, class_names)}
        np.savez_compressed(
            odir / f"predictions_{split}.npz",
            logits=logits.astype(np.float32),
            y_true=y.astype(np.int64),
            y_pred=logits.argmax(1).astype(np.int64),
            sample_ids=splits[split].get("sample_ids", np.arange(len(y)).astype(str)),
            signer_ids=splits[split].get("signer_ids", np.array([""] * len(y))),
        )

    onnx_path = odir / "model.onnx"
    parity = None
    if export_onnx_model:
        export_onnx(model, onnx_path, seq_len=seq_len, input_dim=input_dim)
        parity = verify_onnx_parity(model, onnx_path, splits["test"]["X"], splits["test"].get("mask", np.ones(splits["test"]["X"].shape[:2], bool)))

    last = history[-1]
    best_rec = history[best["epoch"] - 1]
    report = {
        "run_name": odir.name,
        "device": str(device),
        "features_dir": str(fdir),
        "input_dim": input_dim,
        "seq_len": seq_len,
        "num_classes": num_classes,
        "class_names": class_names,
        "num_params": num_params,
        "hparams": hparams,
        "epochs_run": len(history),
        "best_epoch": best["epoch"],
        "stopped_early": len(history) < epochs,
        "elapsed_sec": elapsed_sec,
        "sec_per_epoch": round(elapsed_sec / max(1, len(history)), 2),
        "train_at_best": {"loss": best_rec["train_loss"], "accuracy": best_rec["train_acc"]},
        "train_val_gap_at_best": round(best_rec["train_acc"] - best_rec["val_acc"], 4),
        "last_epoch": last,
        "val": results["val"],
        "test": results["test"],
        "checkpoint": str(ckpt_path),
        "onnx_path": str(onnx_path) if export_onnx_model else None,
        "onnx_parity": parity,
    }
    (odir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (odir / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train VSL LSTM/BiLSTM & export ONNX (Stage 3 — Hiếu)")
    p.add_argument("--features-dir", default="data/features")
    p.add_argument("--output-dir", default="artifacts/experiments/my_run")
    p.add_argument("--arch", choices=sorted(MODEL_REGISTRY), default="bilstm")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--no-augment", action="store_true", help="Tắt augmentation landmark lúc train")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-threads", type=int, default=os.cpu_count())
    p.add_argument("--no-onnx", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_arg_parser().parse_args(argv)
    r = train_and_evaluate(
        features_dir=a.features_dir, output_dir=a.output_dir, arch=a.arch, epochs=a.epochs,
        batch_size=a.batch_size, lr=a.lr, hidden_dim=a.hidden_dim, num_layers=a.num_layers,
        dropout=a.dropout, weight_decay=a.weight_decay, label_smoothing=a.label_smoothing,
        augment=not a.no_augment, patience=a.patience, seed=a.seed,
        export_onnx_model=not a.no_onnx, num_threads=a.num_threads,
    )
    print(
        f"[train] {r['run_name']} params={r['num_params']:,} time={r['elapsed_sec']}s best_ep={r['best_epoch']} | "
        f"val_acc={r['val']['accuracy']:.4f} val_f1={r['val']['macro_f1']:.4f} | "
        f"test_acc={r['test']['accuracy']:.4f} test_f1={r['test']['macro_f1']:.4f} "
        f"| onnx_parity={None if r['onnx_parity'] is None else r['onnx_parity']['passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

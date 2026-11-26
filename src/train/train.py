"""Script huấn luyện, đánh giá và export ONNX cho Giai đoạn 3 (Hiếu).

Ví dụ chạy trên bản 207 chiều (hoặc ``data/features_vel414``):
    python -m src.train.train --features-dir data/features --output-dir artifacts/bilstm_207 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.dataset import VSLFeatureDataset
from src.train.model import VSLBiLSTMAttention


def _compute_metrics(y_true: np.ndarray, logits: np.ndarray, class_names: list[str] | None = None) -> dict[str, Any]:
    preds = np.argmax(logits, axis=1)
    acc = float(np.mean(preds == y_true))
    top3_idx = np.argsort(logits, axis=1)[:, -3:]
    top3_acc = float(np.mean([yt in row for yt, row in zip(y_true, top3_idx)]))

    num_classes = int(logits.shape[1])
    f1s: list[float] = []
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for yt, yp in zip(y_true, preds):
        cm[int(yt), int(yp)] += 1
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    macro_f1 = float(np.mean(f1s))

    # Top các cặp hay nhầm nhất
    confusions: list[dict[str, Any]] = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "true_id": i,
                    "pred_id": j,
                    "true_label": class_names[i] if class_names and i < len(class_names) else str(i),
                    "pred_label": class_names[j] if class_names and j < len(class_names) else str(j),
                    "count": int(cm[i, j]),
                })
    confusions.sort(key=lambda r: r["count"], reverse=True)

    return {
        "accuracy": round(acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "top_confusions": confusions[:10],
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str] | None = None,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_logits: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for x, m, y in loader:
        x, m, y = x.to(device), m.to(device), y.to(device)
        logits = model(x, m)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * int(y.size(0))
        total_n += int(y.size(0))
        all_logits.append(logits.cpu().numpy())
        all_y.append(y.cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    y_np = np.concatenate(all_y, axis=0)
    metrics = _compute_metrics(y_np, logits_np, class_names=class_names)
    return total_loss / max(1, total_n), metrics


def export_onnx(
    model: nn.Module,
    onnx_path: Path,
    seq_len: int,
    input_dim: int,
) -> None:
    model_cpu = model.to("cpu").eval()
    dummy_x = torch.randn(1, seq_len, input_dim, dtype=torch.float32)
    dummy_m = torch.ones(1, seq_len, dtype=torch.bool)
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


def train_and_evaluate(
    features_dir: str | Path = "data/features",
    output_dir: str | Path = "artifacts/bilstm_207",
    *,
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    patience: int = 10,
    seed: int = 42,
    export_onnx_model: bool = True,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    fdir = Path(features_dir)
    odir = Path(output_dir)
    odir.mkdir(parents=True, exist_ok=True)

    meta_path = fdir / "feature_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    class_names = [c["label"] for c in meta.get("class_list", [])] if meta else None

    train_ds = VSLFeatureDataset.from_npz(fdir / "train.npz", augment=True, seed=seed)
    val_ds = VSLFeatureDataset.from_npz(fdir / "val.npz", augment=False, seed=seed)
    test_ds = VSLFeatureDataset.from_npz(fdir / "test.npz", augment=False, seed=seed)

    seq_len = int(train_ds.X.shape[1])
    input_dim = int(train_ds.X.shape[2])
    num_classes = int(meta.get("num_classes", 30)) if meta else 30

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VSLBiLSTMAttention(
        input_dim=input_dim,
        num_classes=num_classes,
        proj_dim=hidden_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_val_f1 = -1.0
    best_epoch = 0
    no_improve = 0
    history: list[dict[str, Any]] = []
    ckpt_path = odir / "best_model.pt"

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for x, m, y in train_loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, m)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += float(loss.item()) * int(y.size(0))
            tr_n += int(y.size(0))
        scheduler.step()

        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, class_names=class_names)
        rec = {
            "epoch": epoch,
            "train_loss": round(tr_loss / max(1, tr_n), 4),
            "val_loss": round(val_loss, 4),
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(rec)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "seq_len": seq_len,
                    "num_classes": num_classes,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "features_dir": str(fdir),
                },
                ckpt_path,
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    elapsed_sec = round(time.time() - t_start, 2)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    val_loss, val_best = evaluate(model, val_loader, criterion, device, class_names=class_names)
    test_loss, test_best = evaluate(model, test_loader, criterion, device, class_names=class_names)

    onnx_path = odir / "model.onnx"
    if export_onnx_model:
        export_onnx(model, onnx_path, seq_len=seq_len, input_dim=input_dim)

    report = {
        "device": str(device),
        "features_dir": str(fdir),
        "input_dim": input_dim,
        "seq_len": seq_len,
        "num_classes": num_classes,
        "num_params": num_params,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed_sec,
        "val": {"loss": round(val_loss, 4), **val_best},
        "test": {"loss": round(test_loss, 4), **test_best},
        "checkpoint": str(ckpt_path),
        "onnx_path": str(onnx_path) if export_onnx_model else None,
    }
    (odir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (odir / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train VSL BiLSTM+Attention & export ONNX (Stage 3)")
    p.add_argument("--features-dir", default="data/features")
    p.add_argument("--output-dir", default="artifacts/bilstm_207")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-onnx", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = train_and_evaluate(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        export_onnx_model=not args.no_onnx,
    )
    print(
        f"[train] device={report['device']} params={report['num_params']:,} "
        f"time={report['elapsed_sec']}s best_ep={report['best_epoch']} | "
        f"val_acc={report['val']['accuracy']:.4f} val_f1={report['val']['macro_f1']:.4f} | "
        f"test_acc={report['test']['accuracy']:.4f} test_f1={report['test']['macro_f1']:.4f} "
        f"test_top3={report['test']['top3_accuracy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

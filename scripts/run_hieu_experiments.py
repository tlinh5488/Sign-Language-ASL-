"""Chạy toàn bộ thí nghiệm Giai đoạn 3 (Hiếu): baseline LSTM -> BiLSTM nhiều tầng -> tinh chỉnh -> xuất ONNX.

    python scripts/run_hieu_experiments.py --config configs/train_experiments.yaml

- Có thể chạy lại: lần chạy nào đã có ``eval_report.json`` sẽ được bỏ qua (dùng ``--force`` để chạy lại hết).
- Chọn mô hình CHỈ theo val macro-F1 (hòa -> val loss thấp hơn). Test chỉ để báo cáo.
- Kết quả:
    artifacts/experiments/<run>/...          mỗi lần chạy một thư mục
    artifacts/experiments/results.csv|.md    bảng so sánh mọi lần chạy
    artifacts/final/                         mô hình bàn giao cho Linh (ONNX + model_card.json)
    docs/figures/hieu_*.png                  learning curves + confusion matrix
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.train import train_and_evaluate  # noqa: E402

HP_KEYS = ("arch", "num_layers", "hidden_dim", "dropout", "lr", "batch_size", "weight_decay",
           "label_smoothing", "augment", "epochs", "patience", "seed")


def _score(r: dict[str, Any]) -> tuple[float, float]:
    return (r["val"]["macro_f1"], -r["val"]["loss"])


class Runner:
    def __init__(self, cfg: dict[str, Any], force: bool = False) -> None:
        self.cfg = cfg
        self.root = REPO_ROOT / cfg["output_root"]
        self.features_dir = REPO_ROOT / cfg["features_dir"]
        self.force = force
        self.rows: list[dict[str, Any]] = []

    def run(self, stage: str, name: str, hp: dict[str, Any]) -> dict[str, Any]:
        odir = self.root / name
        rep_path = odir / "eval_report.json"
        if rep_path.exists() and not self.force:
            report = json.loads(rep_path.read_text(encoding="utf-8"))
            print(f"[skip] {name} (đã có kết quả) val_f1={report['val']['macro_f1']:.4f}", flush=True)
        else:
            print(f"\n===== [{stage}] {name}: {hp} =====", flush=True)
            report = train_and_evaluate(features_dir=self.features_dir, output_dir=odir,
                                        **{k: hp[k] for k in HP_KEYS})
            print(f"[done] {name} val_f1={report['val']['macro_f1']:.4f} test_f1={report['test']['macro_f1']:.4f} "
                  f"time={report['elapsed_sec']}s", flush=True)
        report["_stage"] = stage
        report["_hp"] = {k: hp[k] for k in HP_KEYS}
        if name not in {r["run_name"] for r in self.rows}:
            self.rows.append(report)
        return report


def _row(r: dict[str, Any]) -> dict[str, Any]:
    hp = r["_hp"]
    return {
        "stage": r["_stage"], "run": r["run_name"], "arch": hp["arch"], "layers": hp["num_layers"],
        "hidden": hp["hidden_dim"], "dropout": hp["dropout"], "lr": hp["lr"], "batch": hp["batch_size"],
        "augment": hp["augment"], "seed": hp["seed"], "params": r["num_params"],
        "epochs_run": r["epochs_run"], "best_epoch": r["best_epoch"], "sec_per_epoch": r["sec_per_epoch"],
        "train_acc": r["train_at_best"]["accuracy"], "val_loss": r["val"]["loss"],
        "val_acc": r["val"]["accuracy"], "val_macro_f1": r["val"]["macro_f1"],
        "test_acc": r["test"]["accuracy"], "test_macro_f1": r["test"]["macro_f1"],
        "test_weighted_f1": r["test"]["weighted_f1"], "test_top3": r["test"]["top3_accuracy"],
        "train_val_gap": r["train_val_gap_at_best"],
        "onnx_ok": (r.get("onnx_parity") or {}).get("passed"),
    }


def write_tables(rows: list[dict[str, Any]], root: Path, seed_summary: dict[str, Any], decisions: list[str]) -> None:
    table = [_row(r) for r in rows]
    with open(root / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    cols = ["stage", "run", "arch", "layers", "hidden", "dropout", "lr", "batch", "augment", "seed", "params",
            "best_epoch", "sec_per_epoch", "train_acc", "val_acc", "val_macro_f1", "test_acc", "test_macro_f1", "train_val_gap"]
    lines = ["# Kết quả thí nghiệm Giai đoạn 3 (Hiếu)", "",
             "Chọn mô hình theo **val macro-F1**; cột test chỉ để báo cáo.", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for t in table:
        lines.append("| " + " | ".join(f"{t[c]:,}" if c == "params" else str(t[c]) for c in cols) + " |")
    lines += ["", "## Quyết định từng bước", ""] + [f"- {d}" for d in decisions]
    lines += ["", "## Độ ổn định theo seed (cấu hình cuối)", "", "```json", json.dumps(seed_summary, indent=2, ensure_ascii=False), "```"]
    (root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(rows: list[dict[str, Any]], final: dict[str, Any], root: Path, fig_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = []

    # 1) Learning curves Stage A (val macro-F1 & loss theo epoch)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for r in [r for r in rows if r["_stage"] == "A"]:
        h = json.loads((root / r["run_name"] / "train_history.json").read_text(encoding="utf-8"))
        ep = [x["epoch"] for x in h]
        axes[0].plot(ep, [x["val_macro_f1"] for x in h], label=r["run_name"])
        axes[1].plot(ep, [x["val_loss"] for x in h], label=r["run_name"])
    axes[0].set(title="Stage A — val macro-F1", xlabel="epoch", ylabel="macro-F1")
    axes[1].set(title="Stage A — val loss", xlabel="epoch", ylabel="loss")
    for a in axes:
        a.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    p = fig_dir / "hieu_stage_a_curves.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(str(p))

    # 2) Train vs val của mô hình cuối (kiểm tra overfitting)
    h = json.loads((root / final["run_name"] / "train_history.json").read_text(encoding="utf-8"))
    ep = [x["epoch"] for x in h]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(ep, [x["train_loss"] for x in h], label="train")
    axes[0].plot(ep, [x["val_loss"] for x in h], label="val")
    axes[1].plot(ep, [x["train_acc"] for x in h], label="train acc")
    axes[1].plot(ep, [x["val_acc"] for x in h], label="val acc")
    axes[1].plot(ep, [x["val_macro_f1"] for x in h], label="val macro-F1", linestyle="--")
    for a in axes:
        a.axvline(final["best_epoch"], color="gray", linestyle=":", label="best epoch")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    axes[0].set(title=f"{final['run_name']} — loss", xlabel="epoch")
    axes[1].set(title=f"{final['run_name']} — accuracy / F1", xlabel="epoch")
    fig.tight_layout()
    p = fig_dir / "hieu_final_curves.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(str(p))

    # 3) Confusion matrix (chuẩn hóa theo hàng) trên test của mô hình cuối
    cm = np.asarray(final["test"]["confusion_matrix"], dtype=float)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    names = final.get("class_names") or [str(i) for i in range(len(cm))]
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=75, fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set(xlabel="Dự đoán", ylabel="Thực tế",
           title=f"Confusion matrix test — {final['run_name']} (macro-F1={final['test']['macro_f1']:.3f})")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    p = fig_dir / "hieu_final_confusion_test.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(str(p))
    return out


def export_final(final: dict[str, Any], root: Path, final_dir: Path, features_dir: Path, seed_summary: dict[str, Any]) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    src = root / final["run_name"]
    for f in ("model.onnx", "best_model.pt", "eval_report.json", "train_history.json",
              "predictions_val.npz", "predictions_test.npz"):
        shutil.copy2(src / f, final_dir / f)
    meta = json.loads((features_dir / "feature_meta.json").read_text(encoding="utf-8"))
    card = {
        "model_name": "VSL-30 sign classifier",
        "source_run": final["run_name"],
        "hparams": final["_hp"],
        "num_params": final["num_params"],
        "onnx": {
            "file": "model.onnx",
            "opset": 17,
            "inputs": {
                "features": {"dtype": "float32", "shape": ["batch", final["seq_len"], final["input_dim"]]},
                "mask": {"dtype": "bool", "shape": ["batch", final["seq_len"]], "meaning": "True = frame hợp lệ"},
            },
            "outputs": {"logits": {"dtype": "float32", "shape": ["batch", final["num_classes"]],
                                   "note": "Chưa softmax; class_id = argmax"}},
            "parity_vs_pytorch_on_test": final.get("onnx_parity"),
        },
        "preprocessing": {
            "must_use": "src.preprocess.normalize.normalize_frame + src.preprocess.sequence.process_realtime_window",
            "feature_meta": str((features_dir / "feature_meta.json").relative_to(REPO_ROOT)).replace("\\", "/"),
            "feature_dim": meta.get("D"),
            "seq_len": meta.get("L"),
        },
        "class_names": final.get("class_names"),
        "metrics": {
            "val": {k: final["val"][k] for k in ("accuracy", "top3_accuracy", "macro_f1", "weighted_f1")},
            "test": {k: final["test"][k] for k in ("accuracy", "top3_accuracy", "macro_f1", "weighted_f1")},
            "seed_stability": seed_summary,
        },
    }
    (final_dir / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/train_experiments.yaml")
    ap.add_argument("--force", action="store_true", help="Chạy lại cả những run đã có kết quả")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console Windows cp1252

    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    R = Runner(cfg, force=args.force)
    R.root.mkdir(parents=True, exist_ok=True)
    base = {"arch": "bilstm", **cfg["defaults"]}
    decisions: list[str] = []

    # ---------- Stage A: kiến trúc ----------
    stage_a = [R.run("A", e["name"], {**base, **{k: v for k, v in e.items() if k != "name"}}) for e in cfg["stage_a"]]
    lstm_best = max([r for r in stage_a if r["_hp"]["arch"] == "lstm"], key=_score)
    bilstm_runs = [r for r in stage_a if r["_hp"]["arch"] == "bilstm"]
    base_run = max(bilstm_runs, key=_score)
    decisions.append(f"Stage A: baseline LSTM tốt nhất = {lstm_best['run_name']} (val F1 {lstm_best['val']['macro_f1']}); "
                     f"BiLSTM tốt nhất = {base_run['run_name']} (val F1 {base_run['val']['macro_f1']}) -> làm gốc cho Stage B.")
    base_hp = dict(base_run["_hp"])

    # ---------- Stage B: tinh chỉnh từng yếu tố ----------
    winners: dict[str, Any] = {}
    candidates = [base_run]
    for factor, values in cfg["stage_b"].items():
        trials = [base_run]
        for v in values:
            name = f"B_{factor}_{str(v).lower()}"
            trials.append(R.run("B", name, {**base_hp, factor: v}))
        best_t = max(trials, key=_score)
        winners[factor] = best_t["_hp"][factor]
        candidates += trials[1:]
        decisions.append(f"Stage B [{factor}]: thử {[base_hp[factor]] + list(values)} -> val F1 "
                         f"{[t['val']['macro_f1'] for t in trials]} -> chọn {winners[factor]}.")

    # ---------- Stage C: gộp + nhiều seed ----------
    combined_hp = {**base_hp, **winners}
    if combined_hp != base_hp:
        combined = R.run("C", "C_combined_seed42", combined_hp)
        candidates.append(combined)
        decisions.append(f"Stage C: gộp {winners} -> val F1 {combined['val']['macro_f1']}.")
    final_cfg_run = max(candidates, key=_score)
    final_hp = dict(final_cfg_run["_hp"])
    decisions.append(f"Cấu hình cuối (val F1 cao nhất) = {final_cfg_run['run_name']}: "
                     f"{ {k: final_hp[k] for k in ('arch', 'num_layers', 'hidden_dim', 'dropout', 'lr', 'batch_size', 'augment')} }.")

    seed_runs = []
    for s in cfg["stage_c"]["seeds"]:
        if s == final_hp["seed"]:
            seed_runs.append(final_cfg_run)
        else:
            seed_runs.append(R.run("C", f"C_final_seed{s}", {**final_hp, "seed": s}))

    def ms(vals: list[float]) -> dict[str, float]:
        return {"mean": round(statistics.mean(vals), 4), "std": round(statistics.pstdev(vals), 4)}

    seed_summary = {
        "seeds": [r["_hp"]["seed"] for r in seed_runs],
        "val_macro_f1": ms([r["val"]["macro_f1"] for r in seed_runs]),
        "test_macro_f1": ms([r["test"]["macro_f1"] for r in seed_runs]),
        "test_accuracy": ms([r["test"]["accuracy"] for r in seed_runs]),
    }
    final = max(seed_runs, key=_score)
    decisions.append(f"Mô hình bàn giao = {final['run_name']} (seed {final['_hp']['seed']}, val F1 cao nhất trong các seed).")

    write_tables(R.rows, R.root, seed_summary, decisions)
    export_final(final, R.root, REPO_ROOT / cfg["final_dir"], R.features_dir, seed_summary)
    figs = make_figures(R.rows, final, R.root, REPO_ROOT / "docs" / "figures")

    summary = {
        "baseline_lstm": {"run": lstm_best["run_name"], "val_f1": lstm_best["val"]["macro_f1"], "test_f1": lstm_best["test"]["macro_f1"]},
        "final": {"run": final["run_name"], "val_f1": final["val"]["macro_f1"], "test_f1": final["test"]["macro_f1"],
                  "test_acc": final["test"]["accuracy"], "onnx_parity": (final.get("onnx_parity") or {}).get("passed")},
        "seed_summary": seed_summary,
        "decisions": decisions,
        "figures": figs,
    }
    (R.root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

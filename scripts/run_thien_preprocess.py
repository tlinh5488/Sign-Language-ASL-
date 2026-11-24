"""End-to-end production CLI runner cho Giai đoạn 2 — Tiền xử lý (Thiên).

Kết nối trọn vẹn 4 bước tiền xử lý trên dữ liệu video thực tế từ Giai đoạn 1 (Quý):
    - Bước 2.1: Trích xuất landmark thô bằng MediaPipe Holistic (``src.preprocess.extract_landmarks``)
    - Bước 2.2: Chuẩn hóa tọa độ về gốc cổ tay / tâm vai (``src.preprocess.normalize``)
    - Bước 2.3: Nội suy tay bị mất, cắt frame nghỉ, cố định độ dài L (``src.preprocess.sequence``)
    - Bước 2.4: Lưu đặc trưng ``.npy`` từng mẫu + ``{train,val,test}.npz`` + ``feature_meta.json``
      (``src.preprocess.build_features``)

CLI::

    # Chạy trọn gói Bước 2.1 -> 2.4 trên các manifest thực tế từ Giai đoạn 1 (Quý)
    python scripts/run_thien_preprocess.py --config configs/preprocess_config.yaml

    # Hoặc chỉ xây dựng đặc trưng khi đã trích xuất xong raw landmarks (.npz)
    python scripts/run_thien_preprocess.py --config configs/preprocess_config.yaml --skip-extract
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.preprocess.build_features import (
    _load_yaml,
    _resolve,
    assert_no_signer_overlap,
    build_all_features,
)
from src.preprocess.extract_landmarks import (
    DEFAULT_HOLISTIC,
    extract_manifest,
    resolve_manifests,
)
from src.preprocess.normalize import (
    feature_dim,
    feature_names,
)

LOGGER = logging.getLogger("run_thien_preprocess")


def verify_packed_features(
    cfg: Mapping[str, Any],
    features_dir: Path,
    *,
    splits: Iterable[str] = ("train", "val", "test"),
) -> dict[str, Any]:
    """Kiểm định định dạng, tính hữu hạn (không NaN/Inf) và 0 rò rỉ signer của các file .npz."""
    D = feature_dim(cfg)
    L = int((cfg.get("sequence", {}) or {}).get("target_len", 48))
    packed_summary: dict[str, Any] = {}
    split_signers: dict[str, list[str]] = {}
    for split in splits:
        p = features_dir / f"{split}.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=False) as z:
            X = z["X"]
            mask = z["mask"]
            y = z["y"]
            sids = z["sample_ids"]
            signers = z["signer_ids"]
            assert X.ndim == 3 and X.shape[1:] == (L, D), f"Bad X shape: {X.shape}"
            assert X.dtype == np.float32
            assert mask.shape == X.shape[:2] and mask.dtype == np.bool_
            assert y.shape == (X.shape[0],) and y.dtype == np.int64
            assert np.all((y >= 0) & (y < 30)), f"Class IDs out of 0..29: {y}"
            assert np.all(np.isfinite(X)), "X contains NaN or Inf"
            split_signers[split] = sorted(set(signers.tolist()))
            packed_summary[split] = {
                "X_shape": list(X.shape),
                "mask_true_ratio": round(float(np.mean(mask)), 4) if mask.size else 0.0,
                "classes": sorted(set(int(v) for v in y.tolist())),
                "signers": split_signers[split],
                "num_samples": int(len(sids)),
            }

    assert_no_signer_overlap(split_signers)
    return {
        "D": D,
        "L": L,
        "num_feature_names": len(feature_names(cfg)),
        "packed_splits": packed_summary,
        "zero_signer_leakage": True,
    }


def run_real_preprocess(
    cfg: Mapping[str, Any] | str | Path | None = "configs/preprocess_config.yaml",
    *,
    splits: Iterable[str] = ("train", "val", "test"),
    manifests: list[str | Path] | None = None,
    landmarks_dir: str | Path | None = None,
    features_dir: str | Path | None = None,
    root_dir: str | Path | None = None,
    overwrite: bool = False,
    num_workers: int | None = None,
    limit: int | None = None,
    min_hand_ratio: float = 0.0,
    skip_extract: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Chạy pipeline sản xuất thực tế: Bước 2.1 (extract_landmarks) -> Bước 2.2-2.4 (build_features)."""
    if isinstance(cfg, (str, Path)):
        cfg_dict = _load_yaml(cfg)
    else:
        cfg_dict = dict(cfg or {})

    paths = cfg_dict.get("paths", {}) or {}
    ext = cfg_dict.get("extraction", {}) or {}
    root = Path(root_dir or paths.get("root_dir", ".")).resolve()
    ldir = _resolve(landmarks_dir or paths.get("landmarks_dir", "data/landmarks"), root)
    fdir = _resolve(features_dir or paths.get("features_dir", "data/features"), root)

    split_list = list(splits)
    if not skip_extract:
        hol = dict(DEFAULT_HOLISTIC)
        hol.update({k: v for k, v in (cfg_dict.get("holistic", {}) or {}).items() if k in DEFAULT_HOLISTIC})
        hol["max_frames"] = (cfg_dict.get("holistic", {}) or {}).get("max_frames")
        manifest_paths = (
            [_resolve(m, root) for m in manifests]
            if manifests
            else resolve_manifests(cfg_dict, split_list, root)
        )
        workers = num_workers if num_workers is not None else int(ext.get("num_workers", 1) or 1)
        ovw = overwrite or bool(ext.get("overwrite", False))
        lim = limit if limit is not None else ext.get("limit")
        for m in manifest_paths:
            if not m.exists():
                raise FileNotFoundError(f"Không tìm thấy manifest Giai đoạn 1: {m}")
            extract_manifest(
                m,
                output_dir=ldir,
                root_dir=root,
                overwrite=ovw,
                num_workers=workers,
                limit=lim,
                holistic_kwargs=hol,
                show_progress=show_progress,
            )

    meta = build_all_features(
        cfg_dict,
        splits=split_list,
        landmarks_dir=ldir,
        features_dir=fdir,
        root_dir=root,
        min_hand_ratio=min_hand_ratio,
        save_per_sample=True,
    )
    verification = verify_packed_features(cfg_dict, fdir, splits=split_list)
    summary = {
        "mode": "production",
        "landmarks_dir": str(ldir),
        "features_dir": str(fdir),
        "feature_meta_path": str(fdir / "feature_meta.json"),
        "D": meta["D"],
        "L": meta["L"],
        "per_split_counts": meta["per_split_counts"],
        "signer_lists": meta["signer_lists"],
        "verification": verification,
    }
    summary_path = fdir / "preprocess_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


run_production_preprocess = run_real_preprocess


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End-to-end production Stage 2 runner — Tiền xử lý MediaPipe Holistic (Thiên)"
    )
    p.add_argument("--config", default="configs/preprocess_config.yaml")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--manifest", nargs="+", default=None)
    p.add_argument("--landmarks-dir", default=None)
    p.add_argument("--features-dir", default=None)
    p.add_argument("--root-dir", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-hand-ratio", type=float, default=0.0)
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    summary = run_real_preprocess(
        args.config,
        splits=args.splits,
        manifests=args.manifest,
        landmarks_dir=args.landmarks_dir,
        features_dir=args.features_dir,
        root_dir=args.root_dir,
        overwrite=args.overwrite,
        num_workers=args.workers,
        limit=args.limit,
        min_hand_ratio=args.min_hand_ratio,
        skip_extract=args.skip_extract,
        show_progress=not args.no_progress,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

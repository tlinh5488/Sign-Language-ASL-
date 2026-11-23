"""Bước 2.4 — Xây dựng & lưu đặc trưng cố định độ dài (.npy từng mẫu + .npz theo split).

Đọc các file chỉ mục landmark từ Bước 2.1 (``data/landmarks/<split>_landmarks_index.csv``
hoặc ``train_augmented_landmarks_index.csv`` cho tập train), thực hiện chuỗi xử lý:

1. Nội suy tay bị mất ngắn hạn (:func:`src.preprocess.sequence.interpolate_raw_hands`)
2. Chuẩn hóa tọa độ về gốc cổ tay / tâm vai (:func:`src.preprocess.normalize.normalize_sequence`)
3. Cắt bỏ các frame nghỉ đầu/cuối không có tay (:func:`src.preprocess.sequence.trim_idle_frames`)
4. Đưa về độ dài cố định ``L = target_len`` + tùy chọn vận tốc Δ (:func:`src.preprocess.sequence.to_fixed_length`)

Đầu ra:
* Từng mẫu: ``<features_dir>/<split>/<sample_id>.npy`` — ``(L, D) float32``
* Đóng gói theo split: ``<features_dir>/{train,val,test}.npz`` với các khóa:
  ``X (N, L, D) float32``, ``mask (N, L) bool``, ``y (N,) int64``,
  ``sample_ids (N,) str``, ``signer_ids (N,) str``, ``labels (N,) str``,
  ``is_augmented (N,) bool`` (đọc trực tiếp bằng ``np.load(..., allow_pickle=False)``).
* Metadata: ``<features_dir>/feature_meta.json`` (D, L, feature_names, config snapshot,
  class_list, per_split_counts, signer_lists, samples_skipped_zero_hands...).
* Kiểm định bắt buộc: **không trùng ``signer_id`` giữa các tập train / val / test**.

CLI::

    python -m src.preprocess.build_features --config configs/preprocess_config.yaml \\
        [--splits train val test] [--min-hand-ratio 0.0]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.preprocess.landmark_schema import load_raw_landmarks
from src.preprocess.normalize import (
    feature_blocks,
    feature_dim,
    feature_names,
    get_norm_cfg,
    normalize_sequence,
)
from src.preprocess.sequence import (
    get_seq_cfg,
    interpolate_raw_hands,
    to_fixed_length,
    trim_idle_frames,
)

LOGGER = logging.getLogger("build_features")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve(path: str | Path, root: Path) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else root / p


def _bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes", "y", "t"}
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    return bool(v)


def _canonical_split(split: str) -> str:
    s = str(split).strip()
    return "train" if s in {"train", "train_augmented"} else s


def resolve_index_csv(
    landmarks_dir: str | Path,
    split: str,
    index_suffix: str = "_landmarks_index.csv",
) -> Path:
    """Tìm file CSV chỉ mục landmark cho ``split``.

    Với ``train`` (hoặc ``train_augmented``), ưu tiên ``train_augmented_landmarks_index.csv``
    nếu tồn tại, sau đó đến ``train_landmarks_index.csv`` (do ``extract_landmarks.py`` nhóm
    theo cột ``split == "train"`` từ ``train_augmented.csv``).
    """
    ldir = Path(landmarks_dir)
    s = str(split).strip()
    if s in {"train", "train_augmented"}:
        candidates = [
            ldir / f"train_augmented{index_suffix}",
            ldir / f"train{index_suffix}",
        ]
        if s == "train" and candidates[1].exists() and not candidates[0].exists():
            return candidates[1]
        for c in candidates:
            if c.exists():
                return c
        return candidates[1]
    return ldir / f"{s}{index_suffix}"


def _resolve_landmark_npz(
    landmark_path: str | Path,
    landmarks_dir: Path,
    root_dir: Path,
) -> Path:
    p = Path(str(landmark_path))
    if p.is_absolute() and p.exists():
        return p
    cand1 = landmarks_dir / p
    if cand1.exists():
        return cand1
    cand2 = root_dir / p
    if cand2.exists():
        return cand2
    return cand1


def _load_class_list(cfg: Mapping[str, Any], root_dir: Path) -> list[dict[str, Any]]:
    data_cfg_path = _resolve(cfg.get("data_config", "configs/data_config.yaml"), root_dir)
    data_cfg = _load_yaml(data_cfg_path)
    cl_rel = (
        cfg.get("class_list")
        or (cfg.get("paths", {}) or {}).get("class_list")
        or (data_cfg.get("paths", {}) or {}).get("class_list")
        or "configs/class_list.json"
    )
    cl_path = _resolve(cl_rel, root_dir)
    if not cl_path.exists():
        return []
    with open(cl_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return list(payload.get("classes", []))
    if isinstance(payload, list):
        return payload
    return []


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------
def process_sample(
    raw: Mapping[str, Any],
    cfg: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Chạy trọn quy trình Bước 2.2–2.3 cho một mẫu raw landmarks.

    Thứ tự:
    1. Nội suy tay bị mất ngắn hạn (``interpolate_missing_hands``, ``max_gap``)
    2. Chuẩn hóa tọa độ (``normalize_sequence``)
    3. Cắt frame nghỉ đầu/cuối không có tay (``trim_idle_frames``)
    4. Đưa về độ dài cố định ``L`` (+ vận tốc Δ nếu ``add_velocity``)

    Returns:
        ``(X (L, D) float32, mask (L,) bool, info dict)``.
    """
    scfg = get_seq_cfg(cfg)
    ncfg = get_norm_cfg(cfg)

    lh_m0 = np.asarray(raw.get("left_hand_mask", []), dtype=bool)
    rh_m0 = np.asarray(raw.get("right_hand_mask", []), dtype=bool)
    if lh_m0.size and rh_m0.size:
        hand_m0 = lh_m0 | rh_m0
        raw_hand_ratio = float(np.mean(hand_m0))
    else:
        raw_hand_ratio = 0.0

    work = dict(raw)
    if bool(scfg.get("interpolate_missing_hands", True)):
        work = interpolate_raw_hands(work, max_gap=int(scfg.get("max_gap", 8)))

    # Chuẩn hóa tọa độ cơ sở (chưa gắn velocity để tránh tính đạo hàm trước khi cắt/resample)
    cfg_no_vel = dict(cfg or {})
    norm_no_vel = dict(ncfg)
    norm_no_vel["add_velocity"] = False
    cfg_no_vel["normalization"] = norm_no_vel
    if "sequence" in cfg_no_vel and isinstance(cfg_no_vel["sequence"], Mapping):
        seq_copy = dict(cfg_no_vel["sequence"])
        seq_copy.pop("add_velocity", None)
        cfg_no_vel["sequence"] = seq_copy

    feats_base, frame_mask, masks = normalize_sequence(work, cfg_no_vel, return_masks=True)
    num_frames_raw = int(feats_base.shape[0])

    if bool(scfg.get("trim_idle", True)):
        feats_base, frame_mask = trim_idle_frames(
            feats_base,
            masks["hand_mask"],
            frame_mask,
            margin=int(scfg.get("trim_margin", 0)),
        )
    num_frames_trimmed = int(feats_base.shape[0])

    X, out_mask = to_fixed_length(feats_base, frame_mask, cfg)
    info = {
        "raw_hand_ratio": raw_hand_ratio,
        "num_frames_raw": num_frames_raw,
        "num_frames_trimmed": num_frames_trimmed,
    }
    return X, out_mask, info


# ---------------------------------------------------------------------------
# Signer leakage check
# ---------------------------------------------------------------------------
def assert_no_signer_overlap(split_signers: Mapping[str, Iterable[str]]) -> None:
    """Ném ``ValueError`` nếu bất kỳ hai split nào có chung ``signer_id`` (khác rỗng)."""
    cleaned: dict[str, set[str]] = {
        s: {str(x).strip() for x in vals if str(x).strip()}
        for s, vals in split_signers.items()
    }
    names = list(cleaned.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s1, s2 = names[i], names[j]
            overlap = cleaned[s1] & cleaned[s2]
            if overlap:
                raise ValueError(
                    f"Phát hiện rò rỉ người ký hiệu (signer overlap) giữa {s1} và {s2}: "
                    f"{sorted(overlap)}"
                )


# ---------------------------------------------------------------------------
# Split & full dataset builders
# ---------------------------------------------------------------------------
def build_split_features(
    index_csv: str | Path,
    cfg: Mapping[str, Any] | None = None,
    *,
    split: str | None = None,
    landmarks_dir: str | Path | None = None,
    features_dir: str | Path | None = None,
    root_dir: str | Path = ".",
    min_hand_ratio: float = 0.0,
    skip_zero_hands: bool = False,
    save_per_sample: bool = True,
) -> dict[str, Any]:
    """Xây dựng đặc trưng cho một split từ file ``<split>_landmarks_index.csv``.

    Ghi:
    * ``<features_dir>/<split>/<sample_id>.npy`` (nếu ``save_per_sample=True``)
    * ``<features_dir>/<split>.npz``

    Returns:
        Dict kết quả của split (arrays + thống kê).
    """
    cfg = dict(cfg or {})
    paths = cfg.get("paths", {}) or {}
    root = Path(root_dir or paths.get("root_dir", ".")).resolve()
    ldir = _resolve(landmarks_dir or paths.get("landmarks_dir", "data/landmarks"), root)
    fdir = _resolve(features_dir or paths.get("features_dir", "data/features"), root)

    idx_path = _resolve(index_csv, root)
    if not idx_path.exists():
        raise FileNotFoundError(f"Không tìm thấy landmark index CSV: {idx_path}")

    df = pd.read_csv(idx_path, encoding="utf-8")
    if split is None:
        if "split" in df.columns and len(df):
            split = _canonical_split(str(df["split"].iloc[0]))
        else:
            stem = idx_path.name.replace("_landmarks_index.csv", "")
            split = _canonical_split(stem)
    else:
        split = _canonical_split(split)

    D = feature_dim(cfg)
    L = int(get_seq_cfg(cfg).get("target_len", 48))

    X_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    y_list: list[int] = []
    sample_ids: list[str] = []
    signer_ids: list[str] = []
    labels: list[str] = []
    is_aug_list: list[bool] = []
    skipped_zero_hands: list[str] = []
    zero_hand_samples: list[str] = []
    skipped_errors: list[str] = []

    sample_out_dir = fdir / split
    if save_per_sample:
        sample_out_dir.mkdir(parents=True, exist_ok=True)

    for rec in df.to_dict("records"):
        sid = str(rec.get("sample_id", "")).strip()
        status = str(rec.get("status", "ok")).strip().lower()
        if status == "error":
            skipped_errors.append(sid)
            continue

        rel_lm = rec.get("landmark_path", "")
        npz_path = _resolve_landmark_npz(rel_lm, ldir, root)
        if not npz_path.exists():
            skipped_errors.append(sid)
            LOGGER.warning("Bỏ qua %s: không tìm thấy %s", sid, npz_path)
            continue

        raw = load_raw_landmarks(npz_path)
        X_i, mask_i, info = process_sample(raw, cfg)
        hr = float(rec.get("hand_detect_ratio", info["raw_hand_ratio"]) or info["raw_hand_ratio"])
        if info["raw_hand_ratio"] == 0.0 and hr == 0.0:
            zero_hand_samples.append(sid)
            if skip_zero_hands or min_hand_ratio > 0.0:
                skipped_zero_hands.append(sid)
                continue
        if min_hand_ratio > 0.0 and max(hr, info["raw_hand_ratio"]) < float(min_hand_ratio):
            skipped_zero_hands.append(sid)
            continue

        if save_per_sample:
            safe_id = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in sid)
            np.save(sample_out_dir / f"{safe_id}.npy", X_i)

        X_list.append(X_i)
        mask_list.append(mask_i)
        y_list.append(int(rec.get("class_id", raw.get("class_id", -1))))
        sample_ids.append(sid)
        signer_ids.append(str(rec.get("signer_id", raw.get("signer_id", ""))).strip())
        labels.append(str(rec.get("label", raw.get("label", ""))).strip())
        is_aug_list.append(_bool(rec.get("is_augmented", False)))

    N = len(X_list)
    if N > 0:
        X_arr = np.stack(X_list, axis=0).astype(np.float32)
        mask_arr = np.stack(mask_list, axis=0).astype(bool)
        y_arr = np.asarray(y_list, dtype=np.int64)
        sid_arr = np.asarray(sample_ids, dtype="U")
        signer_arr = np.asarray(signer_ids, dtype="U")
        label_arr = np.asarray(labels, dtype="U")
        aug_arr = np.asarray(is_aug_list, dtype=bool)
    else:
        X_arr = np.zeros((0, L, D), dtype=np.float32)
        mask_arr = np.zeros((0, L), dtype=bool)
        y_arr = np.zeros((0,), dtype=np.int64)
        sid_arr = np.asarray([], dtype="U1")
        signer_arr = np.asarray([], dtype="U1")
        label_arr = np.asarray([], dtype="U1")
        aug_arr = np.zeros((0,), dtype=bool)

    fdir.mkdir(parents=True, exist_ok=True)
    packed_path = fdir / f"{split}.npz"
    tmp_path = fdir / f"{split}.tmp.npz"
    np.savez_compressed(
        tmp_path,
        X=X_arr,
        mask=mask_arr,
        y=y_arr,
        sample_ids=sid_arr,
        signer_ids=signer_arr,
        labels=label_arr,
        is_augmented=aug_arr,
    )
    tmp_path.replace(packed_path)

    unique_signers = sorted({s for s in signer_ids if s})
    unique_classes = sorted({int(c) for c in y_list})
    return {
        "split": split,
        "packed_path": str(packed_path),
        "index_csv": str(idx_path),
        "count": N,
        "shape": list(X_arr.shape),
        "signers": unique_signers,
        "class_ids": unique_classes,
        "skipped_zero_hands": skipped_zero_hands,
        "zero_hand_samples": zero_hand_samples,
        "skipped_errors": skipped_errors,
    }


def build_all_features(
    cfg: Mapping[str, Any] | str | Path | None = None,
    *,
    splits: Iterable[str] = ("train", "val", "test"),
    landmarks_dir: str | Path | None = None,
    features_dir: str | Path | None = None,
    root_dir: str | Path | None = None,
    min_hand_ratio: float = 0.0,
    skip_zero_hands: bool = False,
    save_per_sample: bool = True,
) -> dict[str, Any]:
    """Xây dựng đặc trưng cho tất cả các split, kiểm tra rò rỉ signer và ghi ``feature_meta.json``."""
    if isinstance(cfg, (str, Path)):
        cfg_dict = _load_yaml(cfg)
    else:
        cfg_dict = dict(cfg or {})

    paths = cfg_dict.get("paths", {}) or {}
    root = Path(root_dir or paths.get("root_dir", ".")).resolve()
    ldir = _resolve(landmarks_dir or paths.get("landmarks_dir", "data/landmarks"), root)
    fdir = _resolve(features_dir or paths.get("features_dir", "data/features"), root)
    index_suffix = str((cfg_dict.get("output", {}) or {}).get("index_suffix", "_landmarks_index.csv"))

    D = feature_dim(cfg_dict)
    L = int(get_seq_cfg(cfg_dict).get("target_len", 48))
    names = feature_names(cfg_dict)
    blocks = dict(feature_blocks(cfg_dict))

    split_reports: dict[str, dict[str, Any]] = {}
    for req_split in splits:
        canon = _canonical_split(req_split)
        idx_csv = resolve_index_csv(ldir, req_split, index_suffix=index_suffix)
        if not idx_csv.exists():
            LOGGER.warning("Bỏ qua split %s: chưa có %s", req_split, idx_csv)
            continue
        rep = build_split_features(
            idx_csv,
            cfg_dict,
            split=canon,
            landmarks_dir=ldir,
            features_dir=fdir,
            root_dir=root,
            min_hand_ratio=min_hand_ratio,
            skip_zero_hands=skip_zero_hands,
            save_per_sample=save_per_sample,
        )
        split_reports[canon] = rep

    # Kiểm định zero signer overlap giữa các split đã đóng gói
    split_signers = {s: r["signers"] for s, r in split_reports.items()}
    assert_no_signer_overlap(split_signers)

    classes = _load_class_list(cfg_dict, root)
    all_skipped_zero = [
        sid for r in split_reports.values() for sid in r.get("skipped_zero_hands", [])
    ]
    all_zero_hand = [
        sid for r in split_reports.values() for sid in r.get("zero_hand_samples", [])
    ]

    cfg_snapshot = {
        "normalization": get_norm_cfg(cfg_dict),
        "sequence": get_seq_cfg(cfg_dict),
        "paths": {
            "landmarks_dir": str(ldir),
            "features_dir": str(fdir),
        },
    }
    meta = {
        "schema_version": "2.0",
        "D": D,
        "feature_dim": D,
        "L": L,
        "target_len": L,
        "feature_names": names,
        "feature_blocks": blocks,
        "config": cfg_snapshot,
        "config_snapshot": cfg_snapshot,
        "num_classes": len(classes) if classes else 30,
        "class_list": classes,
        "classes": classes,
        "per_split_counts": {s: r["count"] for s, r in split_reports.items()},
        "signer_lists": split_signers,
        "signer_overlap_check_passed": True,
        "min_hand_ratio": float(min_hand_ratio),
        "samples_skipped_zero_hands": all_skipped_zero,
        "skipped_zero_hand_samples": all_skipped_zero,
        "zero_hand_detection_samples": all_zero_hand,
        "splits": split_reports,
    }
    fdir.mkdir(parents=True, exist_ok=True)
    meta_path = fdir / "feature_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build normalized fixed-length VSL features (.npy + .npz) (Stage 2.2-2.4)"
    )
    p.add_argument("--config", default="configs/preprocess_config.yaml")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--landmarks-dir", default=None)
    p.add_argument("--features-dir", default=None)
    p.add_argument("--root-dir", default=None)
    p.add_argument("--min-hand-ratio", type=float, default=0.0)
    p.add_argument("--skip-zero-hands", action="store_true")
    p.add_argument("--no-per-sample", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    cfg = _load_yaml(args.config)
    meta = build_all_features(
        cfg,
        splits=args.splits,
        landmarks_dir=args.landmarks_dir,
        features_dir=args.features_dir,
        root_dir=args.root_dir,
        min_hand_ratio=args.min_hand_ratio,
        skip_zero_hands=args.skip_zero_hands,
        save_per_sample=not args.no_per_sample,
    )
    print(
        f"[build_features] D={meta['D']}, L={meta['L']}, "
        f"counts={meta['per_split_counts']}, "
        f"signers={meta['signer_lists']}, "
        f"skipped_zero_hands={len(meta['samples_skipped_zero_hands'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

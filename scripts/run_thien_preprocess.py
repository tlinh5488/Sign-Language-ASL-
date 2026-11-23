"""End-to-end runner cho Giai đoạn 2 — Tiền xử lý (Thiên).

Kết nối trọn vẹn 4 bước của Giai đoạn 2:
    - Bước 2.1: Trích xuất landmark thô bằng MediaPipe Holistic (``src.preprocess.extract_landmarks``)
    - Bước 2.2: Chuẩn hóa tọa độ về gốc cổ tay / tâm vai (``src.preprocess.normalize``)
    - Bước 2.3: Nội suy tay bị mất, cắt frame nghỉ, cố định độ dài L (``src.preprocess.sequence``)
    - Bước 2.4: Lưu đặc trưng ``.npy`` từng mẫu + ``{train,val,test}.npz`` + ``feature_meta.json``
      (``src.preprocess.build_features``)

Chế độ ``--demo``:
    Không cần MediaPipe hay video thật — tự sinh dữ liệu landmark thô ``.npz`` mô phỏng
    chuyển động ký hiệu tay thực tế (có độ dài chuỗi biến thiên T=28..68, có frame nghỉ ở
    đầu/cuối, có lỗ mất tay ngắn hạn để kiểm chứng nội suy, nhiều người ký hiệu tách biệt
    tuyệt đối giữa train/val/test, nhãn từ ``configs/class_list.json``), sau đó chạy toàn bộ
    chuẩn hóa -> chuỗi -> đóng gói ``.npz`` và tự kiểm định kết quả.

CLI::

    # Chạy demo trọn gói (không cần MediaPipe / video thật)
    python scripts/run_thien_preprocess.py --demo

    # Chạy thật trên các manifest từ Giai đoạn 1 (Quý)
    python scripts/run_thien_preprocess.py --config configs/preprocess_config.yaml
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
import pandas as pd

from src.preprocess.build_features import (
    _load_yaml,
    _resolve,
    assert_no_signer_overlap,
    build_all_features,
)
from src.preprocess.extract_landmarks import (
    DEFAULT_HOLISTIC,
    INDEX_COLUMNS,
    extract_manifest,
    landmark_rel_path,
    resolve_manifests,
)
from src.preprocess.landmark_schema import (
    FACE_N,
    HAND_N,
    L_ELBOW,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    MIDDLE_MCP,
    NOSE,
    POSE_N,
    R_ELBOW,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    WRIST,
    detection_ratios,
    save_raw_landmarks,
)
from src.preprocess.normalize import (
    feature_dim,
    feature_names,
    make_state,
    normalize_frame,
    normalize_hand,
    normalize_sequence,
)

LOGGER = logging.getLogger("run_thien_preprocess")


# ---------------------------------------------------------------------------
# Synthetic realistic landmark generator for --demo
# ---------------------------------------------------------------------------
def _hand_skeleton(wrist_xyz: np.ndarray, hand_size: float, phase: float, side: int) -> np.ndarray:
    """Sinh 21 điểm bàn tay có cấu trúc giải phẫu quanh cổ tay (WRIST=0, MIDDLE_MCP=9)."""
    pts = np.zeros((HAND_N, 3), dtype=np.float32)
    pts[WRIST] = wrist_xyz
    # 5 ngón: (cmc/mcp, pip, dip, tip)
    finger_angles = [-0.65, -0.28, 0.0, 0.28, 0.58]
    curl = 0.15 * np.sin(phase * 2.0 * np.pi)
    for f_idx, ang in enumerate(finger_angles):
        base_idx = 1 + f_idx * 4
        theta = ang * side
        dx = np.sin(theta)
        dy = -np.cos(theta)
        for j in range(4):
            r = hand_size * (1.0 + 0.38 * j) * (1.0 - curl * 0.15 * j)
            pts[base_idx + j, 0] = wrist_xyz[0] + r * dx
            pts[base_idx + j, 1] = wrist_xyz[1] + r * dy
            pts[base_idx + j, 2] = wrist_xyz[2] - 0.01 * j
    # Bảo đảm khoảng cách WRIST (0) -> MIDDLE_MCP (9) đúng bằng hand_size
    dir9 = np.array([0.0, -1.0, -0.05], dtype=np.float32)
    dir9 /= np.linalg.norm(dir9)
    pts[MIDDLE_MCP] = wrist_xyz + hand_size * dir9
    return pts


def generate_synthetic_raw_landmarks(
    *,
    num_frames: int = 40,
    class_id: int = 0,
    signer_idx: int = 0,
    sample_id: str = "demo_sample",
    label: str = "giúp đỡ",
    signer_id: str = "kaggle_signer_north",
    split: str = "train",
    fps: float = 30.0,
    width: int = 224,
    height: int = 224,
    idle_frames: int = 2,
    dropout_gap: int = 3,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Sinh một mẫu raw landmarks hợp lệ theo ``landmark_schema`` có chuyển động tay & lỗ mất tay."""
    rng = np.random.default_rng(seed + class_id * 101 + signer_idx * 37)
    T = int(num_frames)

    pose = np.zeros((T, POSE_N, 4), dtype=np.float32)
    face = np.zeros((T, FACE_N, 3), dtype=np.float32)
    left_hand = np.zeros((T, HAND_N, 3), dtype=np.float32)
    right_hand = np.zeros((T, HAND_N, 3), dtype=np.float32)

    pose_mask = np.ones(T, dtype=bool)
    face_mask = np.ones(T, dtype=bool)
    left_hand_mask = np.ones(T, dtype=bool)
    right_hand_mask = np.ones(T, dtype=bool)

    # Mỗi người ký hiệu có vị trí đứng & độ rộng vai hơi khác nhau
    cx = 0.50 + 0.03 * ((signer_idx % 3) - 1)
    cy = 0.44 + 0.02 * ((signer_idx % 2) - 0.5)
    shoulder_w = 0.22 + 0.02 * (signer_idx % 4)
    hand_size = 0.075 + 0.005 * (signer_idx % 3)

    for t in range(T):
        ph = t / max(1, T - 1)
        sway = 0.004 * np.sin(ph * 2.0 * np.pi)
        sc = np.array([cx + sway, cy, 0.0], dtype=np.float32)

        # Điền các điểm thân trên (0..24)
        for j in range(25):
            pose[t, j, :3] = sc + np.array(
                [0.02 * ((j % 5) - 2), -0.18 + 0.015 * j, 0.0], dtype=np.float32
            )
            pose[t, j, 3] = 0.95
        pose[t, NOSE] = [sc[0], sc[1] - 0.18, -0.05, 0.99]
        pose[t, L_SHOULDER] = [sc[0] + 0.5 * shoulder_w, sc[1], 0.0, 0.99]
        pose[t, R_SHOULDER] = [sc[0] - 0.5 * shoulder_w, sc[1], 0.0, 0.99]
        pose[t, L_HIP] = [sc[0] + 0.38 * shoulder_w, sc[1] + 0.36, 0.0, 0.92]
        pose[t, R_HIP] = [sc[0] - 0.38 * shoulder_w, sc[1] + 0.36, 0.0, 0.92]

        # Quỹ đạo hai cổ tay đặc trưng theo class_id
        omega = 2.0 * np.pi * (1.0 + 0.1 * (class_id % 5))
        rw = np.array(
            [
                sc[0] - 0.35 * shoulder_w + 0.12 * np.cos(omega * ph + class_id * 0.3),
                sc[1] + 0.05 - 0.14 * np.sin(np.pi * ph),
                -0.06 - 0.02 * np.sin(omega * ph),
            ],
            dtype=np.float32,
        )
        lw = np.array(
            [
                sc[0] + 0.35 * shoulder_w - 0.10 * np.sin(omega * ph + class_id * 0.2),
                sc[1] + 0.08 - 0.12 * np.sin(np.pi * ph),
                -0.05 + 0.02 * np.cos(omega * ph),
            ],
            dtype=np.float32,
        )
        pose[t, R_ELBOW] = [0.5 * (pose[t, R_SHOULDER, 0] + rw[0]), sc[1] + 0.12, -0.02, 0.95]
        pose[t, L_ELBOW] = [0.5 * (pose[t, L_SHOULDER, 0] + lw[0]), sc[1] + 0.12, -0.02, 0.95]
        pose[t, R_WRIST] = [rw[0], rw[1], rw[2], 0.95]
        pose[t, L_WRIST] = [lw[0], lw[1], lw[2], 0.95]

        right_hand[t] = _hand_skeleton(rw, hand_size, ph, side=-1)
        left_hand[t] = _hand_skeleton(lw, hand_size, ph, side=1)

        # Face mesh quanh đầu mũi
        nose_xyz = pose[t, NOSE, :3]
        grid = np.linspace(-0.06, 0.06, FACE_N, dtype=np.float32)
        face[t, :, 0] = nose_xyz[0] + grid
        face[t, :, 1] = nose_xyz[1] + 0.5 * np.sin(grid * 20.0) * 0.04
        face[t, :, 2] = nose_xyz[2] + rng.normal(0.0, 0.001, size=FACE_N).astype(np.float32)
        face[t, 1] = nose_xyz  # FACE_NOSE_TIP = 1

    # Frame nghỉ ở đầu và cuối (không phát hiện tay -> phải bằng 0 theo contract)
    idle = max(0, min(int(idle_frames), T // 5))
    if idle > 0:
        for sl in (slice(0, idle), slice(T - idle, T)):
            left_hand[sl] = 0.0
            right_hand[sl] = 0.0
            left_hand_mask[sl] = False
            right_hand_mask[sl] = False

    # Lỗ mất tay ngắn ở giữa chuỗi (<= max_gap) để kiểm chứng interpolate_missing
    gap = max(0, min(int(dropout_gap), T // 4))
    if gap > 0 and T >= 16:
        g_start = T // 2 - gap // 2
        g_end = g_start + gap
        right_hand[g_start:g_end] = 0.0
        right_hand_mask[g_start:g_end] = False

    arrays = {
        "pose": pose,
        "face": face,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "pose_mask": pose_mask,
        "face_mask": face_mask,
        "left_hand_mask": left_hand_mask,
        "right_hand_mask": right_hand_mask,
    }
    meta = {
        "fps": float(fps),
        "num_frames": T,
        "width": int(width),
        "height": int(height),
        "sample_id": str(sample_id),
        "label": str(label),
        "class_id": int(class_id),
        "signer_id": str(signer_id),
        "split": str(split),
    }
    return arrays, meta


def _load_demo_classes(root: Path, max_classes: int = 5) -> list[dict[str, Any]]:
    cl_path = root / "configs" / "class_list.json"
    if cl_path.exists():
        with open(cl_path, "r", encoding="utf-8") as f:
            classes = json.load(f).get("classes", [])
        if classes:
            return classes[:max_classes]
    return [
        {"class_id": 0, "label": "giúp đỡ"},
        {"class_id": 1, "label": "muốn"},
        {"class_id": 2, "label": "tiếp tục"},
        {"class_id": 3, "label": "từ chối"},
        {"class_id": 4, "label": "địa chỉ"},
    ][:max_classes]


def generate_demo_landmarks(
    landmarks_dir: str | Path,
    *,
    root_dir: str | Path = ".",
    num_classes: int = 5,
    seed: int = 42,
) -> dict[str, Path]:
    """Sinh bộ raw landmarks ``.npz`` + ``<split>_landmarks_index.csv`` cho 3 tập train/val/test.

    Người ký hiệu tách biệt tuyệt đối giữa 3 tập (zero signer leakage):
    * ``train``: ``kaggle_signer_north``, ``kaggle_signer_central``, ``vsl2_signer_001``, ``vsl2_signer_002``
      (gồm cả mẫu gốc và mẫu ``is_augmented=True`` như trong ``train_augmented.csv``).
    * ``val``:   ``vsl2_signer_003``
    * ``test``:  ``kaggle_signer_south``
    """
    root = Path(root_dir).resolve()
    ldir = _resolve(landmarks_dir, root)
    ldir.mkdir(parents=True, exist_ok=True)

    classes = _load_demo_classes(root, max_classes=num_classes)
    split_signers = {
        "train": [
            ("kaggle_signer_north", 0),
            ("kaggle_signer_central", 1),
            ("vsl2_signer_001", 2),
            ("vsl2_signer_002", 3),
        ],
        "val": [("vsl2_signer_003", 4)],
        "test": [("kaggle_signer_south", 5)],
    }
    frame_lengths = [28, 36, 44, 52, 64]

    index_paths: dict[str, Path] = {}
    for split, signers in split_signers.items():
        rows: list[dict[str, Any]] = []
        for s_name, s_idx in signers:
            for c_pos, c_info in enumerate(classes):
                cid = int(c_info["class_id"])
                label = str(c_info["label"])
                variants = [(False, "orig")]
                if split == "train":
                    variants.append((True, "aug01"))
                for is_aug, suffix in variants:
                    sid = f"{split}_{s_name}_c{cid:02d}_{suffix}"
                    T = frame_lengths[(c_pos + s_idx + (1 if is_aug else 0)) % len(frame_lengths)]
                    arrays, meta = generate_synthetic_raw_landmarks(
                        num_frames=T,
                        class_id=cid,
                        signer_idx=s_idx,
                        sample_id=sid,
                        label=label,
                        signer_id=s_name,
                        split=split,
                        seed=seed + (1000 if is_aug else 0),
                    )
                    rel = landmark_rel_path(split, cid, sid)
                    save_raw_landmarks(ldir / rel, arrays, meta)
                    ratios = detection_ratios(arrays)
                    rows.append(
                        {
                            "sample_id": sid,
                            "landmark_path": rel,
                            "video_path": f"data/demo/raw/{sid}.mp4",
                            "label": label,
                            "class_id": cid,
                            "signer_id": s_name,
                            "split": split,
                            "is_augmented": is_aug,
                            "num_frames": T,
                            "fps": 30.0,
                            "hand_detect_ratio": round(ratios["hand"], 4),
                            "left_hand_detect_ratio": round(ratios["left_hand"], 4),
                            "right_hand_detect_ratio": round(ratios["right_hand"], 4),
                            "pose_detect_ratio": round(ratios["pose"], 4),
                            "face_detect_ratio": round(ratios["face"], 4),
                            "status": "ok",
                            "error": "",
                        }
                    )
        df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
        idx_path = ldir / f"{split}_landmarks_index.csv"
        df.to_csv(idx_path, index=False, encoding="utf-8")
        if split == "train":
            # Ghi thêm bản train_augmented_landmarks_index.csv để tương thích cả 2 quy ước tên
            df.to_csv(ldir / "train_augmented_landmarks_index.csv", index=False, encoding="utf-8")
        index_paths[split] = idx_path

    return index_paths


def verify_preprocess_invariants(cfg: Mapping[str, Any], features_dir: Path) -> dict[str, Any]:
    """Kiểm tra nhanh các bất biến toán học & định dạng file đầu ra."""
    rng = np.random.default_rng(7)
    hand = rng.uniform(0.1, 0.9, size=(10, 21, 3)).astype(np.float32)
    h1 = normalize_hand(hand)
    h2 = normalize_hand(hand * 2.75 + np.array([0.4, -0.25, 0.15], dtype=np.float32))
    max_invariance_err = float(np.max(np.abs(h1 - h2)))
    assert max_invariance_err < 1e-5, f"Hand invariance error too large: {max_invariance_err}"

    # Tay vắng mặt -> toàn 0
    absent = normalize_hand(hand, present=np.zeros(10, dtype=bool))
    assert float(np.max(np.abs(absent))) == 0.0

    # normalize_frame == normalize_sequence trên từng frame
    arrays, meta = generate_synthetic_raw_landmarks(num_frames=16, idle_frames=0, dropout_gap=0)
    raw = {**arrays, **meta}
    seq_feats, _ = normalize_sequence(raw, cfg)
    state = make_state()
    frame_feats = np.stack(
        [
            normalize_frame(
                {k: (v[t] if isinstance(v, np.ndarray) else v) for k, v in raw.items()},
                cfg,
                state=state,
            )
            for t in range(16)
        ],
        axis=0,
    )
    frame_seq_diff = float(np.max(np.abs(seq_feats - frame_feats)))
    assert frame_seq_diff < 1e-5, f"normalize_frame != normalize_sequence: {frame_seq_diff}"

    # Kiểm tra các file .npz đã đóng gói
    D = feature_dim(cfg)
    L = int((cfg.get("sequence", {}) or {}).get("target_len", 48))
    packed_summary: dict[str, Any] = {}
    split_signers: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        p = features_dir / f"{split}.npz"
        assert p.exists(), f"Missing packed file: {p}"
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
        "hand_translation_scale_invariance_max_err": max_invariance_err,
        "frame_vs_sequence_max_err": frame_seq_diff,
        "D": D,
        "L": L,
        "num_feature_names": len(feature_names(cfg)),
        "packed_splits": packed_summary,
        "zero_signer_leakage": True,
    }


def run_demo_preprocess(
    cfg: Mapping[str, Any] | str | Path | None = "configs/preprocess_config.yaml",
    *,
    landmarks_dir: str | Path | None = None,
    features_dir: str | Path | None = None,
    root_dir: str | Path | None = None,
    num_classes: int = 5,
    min_hand_ratio: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Chạy trọn gói chế độ ``--demo`` (sinh landmark giả lập -> chuẩn hóa -> chuỗi -> lưu .npz)."""
    if isinstance(cfg, (str, Path)):
        cfg_dict = _load_yaml(cfg)
    else:
        cfg_dict = dict(cfg or {})

    paths = cfg_dict.get("paths", {}) or {}
    root = Path(root_dir or paths.get("root_dir", ".")).resolve()
    ldir = _resolve(landmarks_dir or paths.get("landmarks_dir", "data/landmarks"), root)
    fdir = _resolve(features_dir or paths.get("features_dir", "data/features"), root)

    index_paths = generate_demo_landmarks(
        ldir, root_dir=root, num_classes=num_classes, seed=seed
    )
    meta = build_all_features(
        cfg_dict,
        splits=("train", "val", "test"),
        landmarks_dir=ldir,
        features_dir=fdir,
        root_dir=root,
        min_hand_ratio=min_hand_ratio,
        save_per_sample=True,
    )
    checks = verify_preprocess_invariants(cfg_dict, fdir)
    summary = {
        "mode": "demo",
        "landmarks_dir": str(ldir),
        "features_dir": str(fdir),
        "index_csvs": {k: str(v) for k, v in index_paths.items()},
        "feature_meta_path": str(fdir / "feature_meta.json"),
        "D": meta["D"],
        "L": meta["L"],
        "per_split_counts": meta["per_split_counts"],
        "signer_lists": meta["signer_lists"],
        "verification": checks,
    }
    summary_path = fdir / "preprocess_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


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
    """Chạy pipeline thật: Bước 2.1 (extract_landmarks) -> Bước 2.2-2.4 (build_features)."""
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
    return {
        "mode": "real",
        "landmarks_dir": str(ldir),
        "features_dir": str(fdir),
        "feature_meta_path": str(fdir / "feature_meta.json"),
        "D": meta["D"],
        "L": meta["L"],
        "per_split_counts": meta["per_split_counts"],
        "signer_lists": meta["signer_lists"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="End-to-end Stage 2 runner — Tiền xử lý (Thiên)")
    p.add_argument("--config", default="configs/preprocess_config.yaml")
    p.add_argument("--demo", action="store_true", help="Chạy demo giả lập không cần MediaPipe/video thật")
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
    p.add_argument("--num-demo-classes", type=int, default=5)
    p.add_argument("--no-progress", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.demo:
        summary = run_demo_preprocess(
            args.config,
            landmarks_dir=args.landmarks_dir,
            features_dir=args.features_dir,
            root_dir=args.root_dir,
            num_classes=args.num_demo_classes,
            min_hand_ratio=args.min_hand_ratio,
        )
    else:
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

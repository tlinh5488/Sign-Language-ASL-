"""Comprehensive automated tests for Giai đoạn 2 — Tiền xử lý (Thiên).

Covers Steps 2.1–2.4 using fast synthetic landmark arrays & fake MediaPipe result
objects inside `tmp_path` (no real webcam or MediaPipe detections required):
  - Step 2.1: `landmark_schema.py`, `extract_landmarks.py` (`results_to_arrays`,
    raw `.npz` save/load/validate)
  - Step 2.2: `normalize.py` (wrist-origin + translation/scale invariance,
    missing hands -> zeros, shoulder-based pose normalization, `normalize_frame`
    vs `normalize_sequence` consistency for Linh's real-time deployment)
  - Step 2.3: `sequence.py` (`interpolate_missing` <= max_gap only, idle trimming,
    resample/pad exact `target_len` & mask, velocity features, no NaN/Inf)
  - Step 2.4: `build_features.py` & `scripts/run_thien_preprocess.py` (production
    runner, packed `{train,val,test}.npz` shapes, `y` in 0..29, `D == feature_dim(cfg)`,
    zero signer overlap, consistent `feature_meta.json`, `--demo` strictly removed,
    and no fake demo files in `data/landmarks/` or `data/features/`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_thien_preprocess as thien_runner
from scripts.run_thien_preprocess import (
    build_arg_parser as build_thien_arg_parser,
    run_real_preprocess,
)
from src.preprocess import normalize as normalize_mod
from src.preprocess import sequence as sequence_mod
from src.preprocess.build_features import (
    assert_no_signer_overlap,
    build_all_features,
    process_sample,
)
from src.preprocess.extract_landmarks import INDEX_COLUMNS, landmark_rel_path, results_to_arrays
from src.preprocess.landmark_schema import (
    FACE_DIM,
    FACE_N,
    FACE_NOSE_TIP,
    FACE_SUBSET_IDX,
    HAND_DIM,
    HAND_N,
    L_ELBOW,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    MIDDLE_MCP,
    NOSE,
    POSE_DIM,
    POSE_N,
    R_ELBOW,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    UPPER_BODY_POSE_IDX,
    WRIST,
    detection_ratios,
    empty_frame_landmarks,
    empty_video_landmarks,
    load_raw_landmarks,
    save_raw_landmarks,
    validate_raw_landmarks,
)
from src.preprocess.normalize import (
    feature_dim,
    feature_names,
    hand_global_position,
    make_state,
    normalize_frame,
    normalize_hand,
    normalize_pose,
    normalize_sequence,
)
from src.preprocess.sequence import (
    add_velocity,
    interpolate_missing,
    pad_or_truncate,
    process_realtime_window,
    resample_sequence,
    to_fixed_length,
    trim_idle_frames,
)


def _load_preprocess_cfg() -> dict[str, Any]:
    cfg_path = REPO_ROOT / "configs" / "preprocess_config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fake_landmark_list(coords: np.ndarray, *, with_visibility: bool = False) -> SimpleNamespace:
    lms = []
    for row in coords:
        if with_visibility:
            lms.append(
                SimpleNamespace(
                    x=float(row[0]),
                    y=float(row[1]),
                    z=float(row[2]),
                    visibility=float(row[3]),
                )
            )
        else:
            lms.append(SimpleNamespace(x=float(row[0]), y=float(row[1]), z=float(row[2])))
    return SimpleNamespace(landmark=lms)


def _make_hand_skeleton(wrist_xyz: np.ndarray, hand_size: float, phase: float, side: int) -> np.ndarray:
    pts = np.zeros((HAND_N, 3), dtype=np.float32)
    pts[WRIST] = wrist_xyz
    finger_angles = [-0.65, -0.28, 0.0, 0.28, 0.58]
    for f_idx, ang in enumerate(finger_angles):
        base_idx = 1 + f_idx * 4
        dx = float(np.sin(ang * side))
        dy = float(-np.cos(ang * side))
        for j in range(4):
            r = hand_size * (1.0 + 0.35 * j)
            pts[base_idx + j, 0] = wrist_xyz[0] + r * dx
            pts[base_idx + j, 1] = wrist_xyz[1] + r * dy
            pts[base_idx + j, 2] = wrist_xyz[2] - 0.01 * j + 0.005 * np.sin(phase * 2.0 * np.pi)
    dir9 = np.array([0.0, -1.0, -0.05], dtype=np.float32)
    dir9 /= np.linalg.norm(dir9)
    pts[MIDDLE_MCP] = wrist_xyz + hand_size * dir9
    return pts


def _make_synthetic_raw_landmarks(
    *,
    num_frames: int = 32,
    class_id: int = 0,
    signer_idx: int = 0,
    sample_id: str = "test_sample",
    label: str = "giúp đỡ",
    signer_id: str = "vsl2_signer_001",
    split: str = "train",
    idle_frames: int = 2,
    dropout_gap: int = 3,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rng = np.random.default_rng(seed + class_id * 101 + signer_idx * 37)
    T = int(num_frames)
    pose = np.zeros((T, POSE_N, POSE_DIM), dtype=np.float32)
    face = np.zeros((T, FACE_N, FACE_DIM), dtype=np.float32)
    left_hand = np.zeros((T, HAND_N, HAND_DIM), dtype=np.float32)
    right_hand = np.zeros((T, HAND_N, HAND_DIM), dtype=np.float32)

    pose_mask = np.ones(T, dtype=bool)
    face_mask = np.ones(T, dtype=bool)
    left_hand_mask = np.ones(T, dtype=bool)
    right_hand_mask = np.ones(T, dtype=bool)

    cx = 0.50 + 0.02 * ((signer_idx % 3) - 1)
    cy = 0.44
    shoulder_w = 0.22 + 0.02 * (signer_idx % 4)
    hand_size = 0.08

    for t in range(T):
        ph = t / max(1, T - 1)
        sc = np.array([cx, cy, 0.0], dtype=np.float32)
        for j in range(25):
            pose[t, j, :3] = sc + np.array([0.02 * ((j % 5) - 2), -0.18 + 0.015 * j, 0.0], dtype=np.float32)
            pose[t, j, 3] = 0.95
        pose[t, NOSE] = [sc[0], sc[1] - 0.18, -0.05, 0.99]
        pose[t, L_SHOULDER] = [sc[0] + 0.5 * shoulder_w, sc[1], 0.0, 0.99]
        pose[t, R_SHOULDER] = [sc[0] - 0.5 * shoulder_w, sc[1], 0.0, 0.99]
        pose[t, L_HIP] = [sc[0] + 0.35 * shoulder_w, sc[1] + 0.35, 0.0, 0.92]
        pose[t, R_HIP] = [sc[0] - 0.35 * shoulder_w, sc[1] + 0.35, 0.0, 0.92]

        rw = np.array([sc[0] - 0.08 + 0.08 * np.cos(2.0 * np.pi * ph), sc[1] - 0.05 * np.sin(np.pi * ph), -0.06], dtype=np.float32)
        lw = np.array([sc[0] + 0.08 - 0.06 * np.sin(2.0 * np.pi * ph), sc[1] - 0.04 * np.sin(np.pi * ph), -0.05], dtype=np.float32)
        pose[t, R_WRIST] = [rw[0], rw[1], rw[2], 0.95]
        pose[t, L_WRIST] = [lw[0], lw[1], lw[2], 0.95]
        right_hand[t] = _make_hand_skeleton(rw, hand_size, ph, side=-1)
        left_hand[t] = _make_hand_skeleton(lw, hand_size, ph, side=1)

        nose_xyz = pose[t, NOSE, :3]
        grid = np.linspace(-0.05, 0.05, FACE_N, dtype=np.float32)
        face[t, :, 0] = nose_xyz[0] + grid
        face[t, :, 1] = nose_xyz[1] + 0.02 * np.sin(grid * 15.0)
        face[t, :, 2] = nose_xyz[2] + rng.normal(0.0, 0.001, size=FACE_N).astype(np.float32)
        face[t, FACE_NOSE_TIP] = nose_xyz

    idle = max(0, min(int(idle_frames), T // 5))
    if idle > 0:
        for sl in (slice(0, idle), slice(T - idle, T)):
            left_hand[sl] = 0.0
            right_hand[sl] = 0.0
            left_hand_mask[sl] = False
            right_hand_mask[sl] = False

    gap = max(0, min(int(dropout_gap), T // 4))
    if gap > 0 and T >= 14:
        g0 = T // 2 - gap // 2
        right_hand[g0 : g0 + gap] = 0.0
        right_hand_mask[g0 : g0 + gap] = False

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
        "fps": 30.0,
        "num_frames": T,
        "width": 224,
        "height": 224,
        "sample_id": sample_id,
        "label": label,
        "class_id": int(class_id),
        "signer_id": signer_id,
        "split": split,
    }
    return arrays, meta


def _write_synthetic_landmark_tree(ldir: Path, *, num_classes: int = 3) -> None:
    """Write synthetic per-video `.npz` files & index CSVs for train/val/test inside `tmp_path`."""
    split_signers = {
        "train": [("vsl2_signer_001", 0), ("vsl2_signer_002", 1)],
        "val": [("vsl2_signer_003", 2)],
        "test": [("vsl2_signer_004", 3)],
    }
    labels = ["giúp đỡ", "muốn", "tiếp tục", "từ chối", "hứa"][:num_classes]
    for split, signers in split_signers.items():
        rows: list[dict[str, Any]] = []
        for s_name, s_idx in signers:
            for cid, lbl in enumerate(labels):
                sid = f"{split}_{s_name}_c{cid:02d}"
                arrays, meta = _make_synthetic_raw_landmarks(
                    num_frames=28 + 4 * cid,
                    class_id=cid,
                    signer_idx=s_idx,
                    sample_id=sid,
                    label=lbl,
                    signer_id=s_name,
                    split=split,
                )
                rel = landmark_rel_path(split, cid, sid)
                save_raw_landmarks(ldir / rel, arrays, meta)
                ratios = detection_ratios(arrays)
                rows.append(
                    {
                        "sample_id": sid,
                        "landmark_path": rel,
                        "video_path": f"tmp/{sid}.mp4",
                        "label": lbl,
                        "class_id": cid,
                        "signer_id": s_name,
                        "split": split,
                        "is_augmented": False,
                        "num_frames": meta["num_frames"],
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
        df.to_csv(ldir / f"{split}_landmarks_index.csv", index=False, encoding="utf-8")
        if split == "train":
            df.to_csv(ldir / "train_augmented_landmarks_index.csv", index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Schema, Config & Raw `.npz` I/O Tests (Step 2.1)
# ---------------------------------------------------------------------------


def test_landmark_schema_constants_and_preprocess_config() -> None:
    assert POSE_N == 33 and POSE_DIM == 4
    assert FACE_N == 468 and FACE_DIM == 3
    assert HAND_N == 21 and HAND_DIM == 3

    assert WRIST == 0 and MIDDLE_MCP == 9
    assert (NOSE, L_SHOULDER, R_SHOULDER) == (0, 11, 12)
    assert (L_ELBOW, R_ELBOW, L_WRIST, R_WRIST) == (13, 14, 15, 16)
    assert (L_HIP, R_HIP) == (23, 24)
    assert UPPER_BODY_POSE_IDX == list(range(0, 25))
    assert 20 <= len(FACE_SUBSET_IDX) <= 45
    assert all(0 <= i < FACE_N for i in FACE_SUBSET_IDX)
    assert FACE_NOSE_TIP in FACE_SUBSET_IDX

    # Pure numpy requirement for real-time deployment (no mediapipe / cv2 module imports)
    assert "mediapipe" not in normalize_mod.__dict__
    assert "cv2" not in normalize_mod.__dict__
    assert "mediapipe" not in sequence_mod.__dict__
    assert "cv2" not in sequence_mod.__dict__

    cfg = _load_preprocess_cfg()
    assert cfg["paths"]["landmarks_dir"] == "data/landmarks"
    assert cfg["paths"]["features_dir"] == "data/features"
    assert cfg["normalization"]["feature_set"] == "pose_upper+hands"
    assert cfg["normalization"]["hand_origin"] == "wrist"
    assert cfg["normalization"]["hand_scale"] == "wrist_to_middle_mcp"
    assert cfg["normalization"]["pose_origin"] == "shoulder_center"
    assert cfg["normalization"]["pose_scale"] == "shoulder_width"
    assert cfg["sequence"]["target_len"] == 48
    assert cfg["sequence"]["max_gap"] == 8

    # Default feature dimension and names consistency
    assert feature_dim(cfg) == 207
    assert len(feature_names(cfg)) == 207

    cfg_face = {**cfg, "normalization": {**cfg["normalization"], "feature_set": "pose_upper+hands+face_subset"}}
    assert feature_dim(cfg_face) == 207 + len(FACE_SUBSET_IDX) * 3
    assert len(feature_names(cfg_face)) == feature_dim(cfg_face)

    cfg_2d = {**cfg, "normalization": {**cfg["normalization"], "use_z": False}}
    assert feature_dim(cfg_2d) == 138
    assert len(feature_names(cfg_2d)) == 138

    cfg_vel = {**cfg, "normalization": {**cfg["normalization"], "add_velocity": True}}
    assert feature_dim(cfg_vel) == 414
    assert len(feature_names(cfg_vel)) == 414


def test_results_to_arrays_with_fake_holistic_objects() -> None:
    empty = results_to_arrays(None)
    assert empty["pose"].shape == (POSE_N, POSE_DIM) and empty["pose"].dtype == np.float32
    assert empty["face"].shape == (FACE_N, FACE_DIM) and empty["face"].dtype == np.float32
    assert empty["left_hand"].shape == (HAND_N, HAND_DIM)
    assert empty["right_hand"].shape == (HAND_N, HAND_DIM)
    assert not (
        empty["pose_mask"] or empty["face_mask"] or empty["left_hand_mask"] or empty["right_hand_mask"]
    )
    assert float(np.max(np.abs(empty["pose"]))) == 0.0

    rng = np.random.default_rng(123)
    pose_arr = rng.uniform(0.1, 0.9, size=(POSE_N, POSE_DIM)).astype(np.float32)
    face_refined = rng.uniform(0.1, 0.9, size=(478, FACE_DIM)).astype(np.float32)
    lh_arr = rng.uniform(0.1, 0.9, size=(HAND_N, HAND_DIM)).astype(np.float32)

    fake_res = SimpleNamespace(
        pose_landmarks=_fake_landmark_list(pose_arr, with_visibility=True),
        face_landmarks=_fake_landmark_list(face_refined, with_visibility=False),
        left_hand_landmarks=_fake_landmark_list(lh_arr, with_visibility=False),
        right_hand_landmarks=None,
    )
    out = results_to_arrays(fake_res)
    assert out["pose_mask"] is True
    assert out["face_mask"] is True
    assert out["left_hand_mask"] is True
    assert out["right_hand_mask"] is False
    np.testing.assert_allclose(out["pose"], pose_arr, atol=1e-6)
    np.testing.assert_allclose(out["face"], face_refined[:FACE_N], atol=1e-6)
    np.testing.assert_allclose(out["left_hand"], lh_arr, atol=1e-6)
    assert float(np.max(np.abs(out["right_hand"]))) == 0.0

    bad_lh = lh_arr.copy()
    bad_lh[3, 1] = np.nan
    fake_bad = SimpleNamespace(
        pose_landmarks=None,
        face_landmarks=None,
        left_hand_landmarks=_fake_landmark_list(bad_lh),
        right_hand_landmarks=None,
    )
    out_bad = results_to_arrays(fake_bad)
    assert out_bad["left_hand_mask"] is False
    assert float(np.max(np.abs(out_bad["left_hand"]))) == 0.0


def test_raw_landmarks_npz_save_load_and_validation(tmp_path: Path) -> None:
    arrays, meta = _make_synthetic_raw_landmarks(
        num_frames=18,
        class_id=4,
        sample_id="vsl_test_001",
        label="hứa",
        signer_id="vsl2_signer_001",
        split="train",
    )
    out_npz = tmp_path / "train" / "4" / "vsl_test_001.npz"
    save_raw_landmarks(out_npz, arrays, meta)
    assert out_npz.exists()

    loaded = load_raw_landmarks(out_npz)
    validate_raw_landmarks(loaded)

    assert loaded["pose"].shape == (18, POSE_N, POSE_DIM) and loaded["pose"].dtype == np.float32
    assert loaded["face"].shape == (18, FACE_N, FACE_DIM) and loaded["face"].dtype == np.float32
    assert loaded["left_hand"].shape == (18, HAND_N, HAND_DIM)
    assert loaded["right_hand"].shape == (18, HAND_N, HAND_DIM)
    assert loaded["pose_mask"].dtype == np.bool_ and loaded["pose_mask"].shape == (18,)
    assert loaded["fps"] == pytest.approx(30.0)
    assert loaded["num_frames"] == 18
    assert loaded["width"] == 224 and loaded["height"] == 224
    assert loaded["sample_id"] == "vsl_test_001"
    assert loaded["label"] == "hứa"
    assert loaded["class_id"] == 4
    assert loaded["signer_id"] == "vsl2_signer_001"
    assert loaded["split"] == "train"

    corrupt = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in loaded.items()}
    corrupt["left_hand_mask"][0] = False
    corrupt["left_hand"][0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="zero-filled"):
        validate_raw_landmarks(corrupt)

    corrupt_nan = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in loaded.items()}
    corrupt_nan["pose"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate_raw_landmarks(corrupt_nan)

    corrupt_shape = dict(loaded)
    corrupt_shape["right_hand"] = np.zeros((18, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        validate_raw_landmarks(corrupt_shape)

    ef = empty_frame_landmarks()
    assert ef["pose"].shape == (POSE_N, POSE_DIM) and ef["pose_mask"] is False
    ev = empty_video_landmarks(5)
    validate_raw_landmarks(ev)


# ---------------------------------------------------------------------------
# 2. Normalization & Deployment Consistency Tests (Step 2.2)
# ---------------------------------------------------------------------------


def test_hand_normalization_wrist_origin_invariance_and_missing() -> None:
    rng = np.random.default_rng(42)
    hand = rng.uniform(0.15, 0.85, size=(12, HAND_N, 3)).astype(np.float32)
    hand[:, MIDDLE_MCP, :] = hand[:, WRIST, :] + np.array([0.04, -0.09, -0.02], dtype=np.float32)

    norm = normalize_hand(hand)
    assert norm.dtype == np.float32 and norm.shape == (12, HAND_N, 3)
    assert np.all(np.isfinite(norm))

    np.testing.assert_allclose(norm[:, WRIST, :], 0.0, atol=1e-6)
    mcp_norms = np.linalg.norm(norm[:, MIDDLE_MCP, :], axis=-1)
    np.testing.assert_allclose(mcp_norms, 1.0, atol=1e-5)

    shift = np.array([1.75, -0.82, 0.33], dtype=np.float32)
    transformed = hand * 3.4 + shift
    norm_transformed = normalize_hand(transformed)
    np.testing.assert_allclose(norm_transformed, norm, atol=1e-5)

    present = np.ones(12, dtype=bool)
    present[[2, 5, 9]] = False
    norm_masked = normalize_hand(hand, present=present)
    np.testing.assert_array_equal(norm_masked[~present], 0.0)
    assert np.any(norm_masked[present] != 0.0)

    degen = hand.copy()
    degen[:, MIDDLE_MCP, :] = degen[:, WRIST, :]
    norm_degen = normalize_hand(degen)
    assert np.all(np.isfinite(norm_degen))
    np.testing.assert_array_equal(norm_degen, 0.0)


def test_pose_normalization_and_hand_global_position() -> None:
    T = 6
    pose = np.zeros((T, POSE_N, POSE_DIM), dtype=np.float32)
    center_true = np.array([0.48, 0.42, 0.05], dtype=np.float32)
    shoulder_w = 0.24
    for t in range(T):
        for j in range(25):
            pose[t, j, :3] = center_true + np.array([0.01 * (j - 12), 0.02 * (j - 10), 0.0], dtype=np.float32)
            pose[t, j, 3] = 0.95
        pose[t, L_SHOULDER] = [center_true[0] + shoulder_w / 2, center_true[1], center_true[2], 0.99]
        pose[t, R_SHOULDER] = [center_true[0] - shoulder_w / 2, center_true[1], center_true[2], 0.99]

    pose[3, L_SHOULDER] = 0.0
    pose[3, R_SHOULDER] = 0.0

    norm_p = normalize_pose(pose)
    assert norm_p.shape == (T, len(UPPER_BODY_POSE_IDX), 3)
    assert np.all(np.isfinite(norm_p))

    for t in (0, 1, 2, 4, 5):
        mid = 0.5 * (norm_p[t, L_SHOULDER] + norm_p[t, R_SHOULDER])
        dist = float(np.linalg.norm(norm_p[t, L_SHOULDER] - norm_p[t, R_SHOULDER]))
        np.testing.assert_allclose(mid, 0.0, atol=1e-6)
        assert dist == pytest.approx(1.0, abs=1e-5)

    np.testing.assert_allclose(norm_p[3, NOSE], norm_p[2, NOSE], atol=1e-5)

    hand = np.zeros((T, HAND_N, 3), dtype=np.float32)
    hand[:, WRIST, :] = center_true + np.array([shoulder_w * 0.5, -shoulder_w * 0.25, 0.0], dtype=np.float32)
    present = np.array([True, True, False, True, True, True], dtype=bool)
    g = hand_global_position(
        hand,
        present,
        center=np.repeat(center_true[None, :], T, axis=0),
        scale=np.full(T, shoulder_w, dtype=np.float32),
    )
    assert g.shape == (T, 3)
    np.testing.assert_allclose(g[0], [0.5, -0.25, 0.0], atol=1e-5)
    np.testing.assert_array_equal(g[2], 0.0)


def test_normalize_frame_matches_normalize_sequence_for_realtime_deployment() -> None:
    """Critical deployment invariant for Linh (Giai đoạn 4):
    `normalize_frame` on each frame MUST match the corresponding row of `normalize_sequence`."""
    base_cfg = _load_preprocess_cfg()
    arrays, meta = _make_synthetic_raw_landmarks(num_frames=14, idle_frames=2, dropout_gap=2, seed=99)
    arrays["pose"][6, L_SHOULDER] = 0.0
    arrays["pose"][6, R_SHOULDER] = 0.0
    raw = {**arrays, **meta}

    configs_to_check = [
        base_cfg,
        {**base_cfg, "normalization": {**base_cfg["normalization"], "feature_set": "pose_upper+hands+face_subset"}},
        {**base_cfg, "normalization": {**base_cfg["normalization"], "add_velocity": True}},
        {**base_cfg, "normalization": {**base_cfg["normalization"], "use_z": False}},
    ]

    for cfg in configs_to_check:
        seq_feats, frame_mask = normalize_sequence(raw, cfg)
        assert seq_feats.shape == (14, feature_dim(cfg))
        assert seq_feats.dtype == np.float32
        assert frame_mask.shape == (14,) and frame_mask.dtype == np.bool_
        assert np.all(np.isfinite(seq_feats))

        state = make_state()
        for t in range(14):
            frame_dict = {
                k: (v[t] if isinstance(v, np.ndarray) else v)
                for k, v in raw.items()
            }
            vec = normalize_frame(frame_dict, cfg, state=state)
            assert vec.shape == (feature_dim(cfg),)
            assert vec.dtype == np.float32
            assert np.all(np.isfinite(vec))
            np.testing.assert_allclose(vec, seq_feats[t], atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Sequence Interpolation, Trimming, Resample & Pad Tests (Step 2.3)
# ---------------------------------------------------------------------------


def test_interpolate_missing_respects_max_gap_and_boundaries() -> None:
    T = 16
    arr = np.zeros((T, 2, 3), dtype=np.float32)
    mask = np.zeros(T, dtype=bool)
    for t in (2, 6, 12, 13):
        arr[t] = float(t)
        mask[t] = True

    filled, new_mask = interpolate_missing(arr, mask, max_gap=4, return_mask=True)
    assert np.all(np.isfinite(filled))

    assert not new_mask[0] and not new_mask[1]
    assert not new_mask[14] and not new_mask[15]
    np.testing.assert_array_equal(filled[[0, 1, 14, 15]], 0.0)

    assert np.all(new_mask[2:7])
    np.testing.assert_allclose(filled[3], 3.0, atol=1e-5)
    np.testing.assert_allclose(filled[4], 4.0, atol=1e-5)
    np.testing.assert_allclose(filled[5], 5.0, atol=1e-5)

    assert not np.any(new_mask[7:12])
    np.testing.assert_array_equal(filled[7:12], 0.0)


def test_trim_resample_pad_and_to_fixed_length() -> None:
    cfg = _load_preprocess_cfg()
    D = feature_dim(cfg)

    feats = np.arange(10 * 4, dtype=np.float32).reshape(10, 4)
    hand_mask = np.array([False, False, True, True, False, True, False, False, False, False])
    trimmed, trimmed_mask = trim_idle_frames(feats, hand_mask)
    assert trimmed.shape == (4, 4)
    np.testing.assert_array_equal(trimmed[0], feats[2])
    np.testing.assert_array_equal(trimmed[-1], feats[5])
    assert trimmed_mask.shape == (4,)

    kept_feats, _ = trim_idle_frames(feats, np.zeros(10, dtype=bool))
    assert kept_feats.shape == (10, 4)

    assert resample_sequence(np.zeros((0, 5), dtype=np.float32), 48).shape == (48, 5)
    r1 = resample_sequence(np.full((1, 5), 3.5, dtype=np.float32), 48)
    assert r1.shape == (48, 5) and np.allclose(r1, 3.5)

    ramp = np.linspace(0.0, 10.0, 20, dtype=np.float32)[:, None]
    r_up = resample_sequence(ramp, 48)
    assert r_up.shape == (48, 1)
    assert r_up[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert r_up[-1, 0] == pytest.approx(10.0, abs=1e-5)
    assert resample_sequence(np.ones((75, 5), dtype=np.float32), 48).shape == (48, 5)

    short = np.ones((12, 4), dtype=np.float32)
    padded, pmask = pad_or_truncate(short, 48, pad_value=0.0)
    assert padded.shape == (48, 4) and pmask.shape == (48,) and pmask.dtype == np.bool_
    assert np.all(pmask[:12]) and not np.any(pmask[12:])
    np.testing.assert_array_equal(padded[12:], 0.0)

    long_x = np.arange(60 * 4, dtype=np.float32).reshape(60, 4)
    trunc, tmask = pad_or_truncate(long_x, 48, truncate="center")
    assert trunc.shape == (48, 4) and np.all(tmask)
    np.testing.assert_array_equal(trunc[0], long_x[6])

    seq_in = np.ones((25, D), dtype=np.float32)
    X_res, m_res = to_fixed_length(seq_in, np.ones(25, dtype=bool), cfg)
    assert X_res.shape == (48, D) and m_res.shape == (48,) and np.all(m_res) and np.all(np.isfinite(X_res))

    cfg_pad = {**cfg, "sequence": {**cfg["sequence"], "method": "pad"}}
    X_pad, m_pad = to_fixed_length(seq_in, np.ones(25, dtype=bool), cfg_pad)
    assert X_pad.shape == (48, D) and int(m_pad.sum()) == 25 and not np.any(m_pad[25:])

    X_vel = add_velocity(X_pad, m_pad)
    assert X_vel.shape == (48, 2 * D)
    np.testing.assert_array_equal(X_vel[25:, D:], 0.0)
    assert np.all(np.isfinite(X_vel))


# ---------------------------------------------------------------------------
# 4. Feature Builder, Signer Leakage Guard & Production Runner (Step 2.4)
# ---------------------------------------------------------------------------


def test_build_features_packing_zero_hand_skip_and_signer_leakage_guard(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="signer overlap"):
        assert_no_signer_overlap({"train": ["s1", "s2"], "val": ["s2"], "test": ["s3"]})
    with pytest.raises(ValueError, match="signer overlap"):
        assert_no_signer_overlap({"train": ["s1"], "val": ["s2"], "test": ["s1"]})
    assert_no_signer_overlap({"train": ["s1", "s2"], "val": ["s3"], "test": ["s4"]})

    cfg = _load_preprocess_cfg()
    ldir = tmp_path / "landmarks"
    fdir = tmp_path / "features"
    _write_synthetic_landmark_tree(ldir, num_classes=3)

    zero_arrays = empty_video_landmarks(20)
    zero_meta = {
        "fps": 30.0,
        "num_frames": 20,
        "width": 224,
        "height": 224,
        "sample_id": "train_zero_hands_sample",
        "label": "giúp đỡ",
        "class_id": 0,
        "signer_id": "vsl2_signer_001",
        "split": "train",
    }
    zero_rel = landmark_rel_path("train", 0, "train_zero_hands_sample")
    save_raw_landmarks(ldir / zero_rel, zero_arrays, zero_meta)

    train_idx_csv = ldir / "train_augmented_landmarks_index.csv"
    df_train = pd.read_csv(train_idx_csv)
    extra_row = pd.DataFrame(
        [
            {
                "sample_id": "train_zero_hands_sample",
                "landmark_path": zero_rel,
                "video_path": "fake.mp4",
                "label": "giúp đỡ",
                "class_id": 0,
                "signer_id": "vsl2_signer_001",
                "split": "train",
                "is_augmented": False,
                "num_frames": 20,
                "fps": 30.0,
                "hand_detect_ratio": 0.0,
                "left_hand_detect_ratio": 0.0,
                "right_hand_detect_ratio": 0.0,
                "pose_detect_ratio": 0.0,
                "face_detect_ratio": 0.0,
                "status": "ok",
                "error": "",
            }
        ],
        columns=INDEX_COLUMNS,
    )
    pd.concat([df_train, extra_row], ignore_index=True).to_csv(train_idx_csv, index=False, encoding="utf-8")

    meta = build_all_features(
        cfg,
        splits=("train", "val", "test"),
        landmarks_dir=ldir,
        features_dir=fdir,
        root_dir=REPO_ROOT,
        min_hand_ratio=0.0,
        skip_zero_hands=True,
        save_per_sample=True,
    )

    D = feature_dim(cfg)
    L = int(cfg["sequence"]["target_len"])
    assert meta["D"] == D == 207
    assert meta["L"] == L == 48
    assert len(meta["feature_names"]) == D
    assert meta["signer_overlap_check_passed"] is True
    assert "train_zero_hands_sample" in meta["samples_skipped_zero_hands"]

    meta_json = json.loads((fdir / "feature_meta.json").read_text(encoding="utf-8"))
    assert meta_json["D"] == D
    assert meta_json["L"] == L
    assert meta_json["per_split_counts"] == meta["per_split_counts"]
    assert len(meta_json["class_list"]) == 30

    split_signers: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        packed_path = fdir / f"{split}.npz"
        assert packed_path.exists()
        with np.load(packed_path, allow_pickle=False) as z:
            X = z["X"]
            mask = z["mask"]
            y = z["y"]
            sample_ids = z["sample_ids"]
            signer_ids = z["signer_ids"]
            labels = z["labels"]
            is_aug = z["is_augmented"]

            N = meta["per_split_counts"][split]
            assert X.shape == (N, L, D) and X.dtype == np.float32
            assert mask.shape == (N, L) and mask.dtype == np.bool_
            assert y.shape == (N,) and y.dtype == np.int64
            assert len(sample_ids) == len(signer_ids) == len(labels) == len(is_aug) == N
            assert np.all((y >= 0) & (y < 30))
            assert np.all(np.isfinite(X))
            assert "train_zero_hands_sample" not in set(sample_ids.tolist())

            first_npy = fdir / split / f"{sample_ids[0]}.npy"
            assert first_npy.exists()
            np.testing.assert_array_equal(np.load(first_npy), X[0])

            split_signers[split] = set(signer_ids.tolist())

    assert split_signers["train"] & split_signers["val"] == set()
    assert split_signers["train"] & split_signers["test"] == set()
    assert split_signers["val"] & split_signers["test"] == set()

    raw_arrs, raw_m = _make_synthetic_raw_landmarks(num_frames=32, idle_frames=3, dropout_gap=4)
    X_s, mask_s, info_s = process_sample({**raw_arrs, **raw_m}, cfg)
    assert X_s.shape == (L, D) and mask_s.shape == (L,) and np.all(np.isfinite(X_s))
    assert info_s["num_frames_trimmed"] < info_s["num_frames_raw"]


def test_thien_no_demo_and_production_runner(tmp_path: Path) -> None:
    """Verify `--demo` is strictly rejected, zero fake demo files exist in `data/landmarks`
    or `data/features`, `docs/preprocess_guide.md` has zero `--demo` references, and
    `run_real_preprocess(..., skip_extract=True)` works end-to-end in `tmp_path`."""
    assert not hasattr(thien_runner, "run_demo_preprocess")
    assert not hasattr(thien_runner, "generate_demo_landmarks")

    parser = build_thien_arg_parser()
    option_strings = {opt for action in parser._actions for opt in action.option_strings}
    assert "--demo" not in option_strings
    assert "--demo-dir" not in option_strings
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--demo"])
    assert exc_info.value.code == 2

    guide_text = (REPO_ROOT / "docs" / "preprocess_guide.md").read_text(encoding="utf-8")
    assert "demo" not in guide_text.lower()

    # Verify production runner end-to-end (skip_extract=True on isolated tmp_path landmarks)
    ldir = tmp_path / "prod_landmarks"
    fdir = tmp_path / "prod_features"
    _write_synthetic_landmark_tree(ldir, num_classes=4)

    summary = run_real_preprocess(
        REPO_ROOT / "configs" / "preprocess_config.yaml",
        splits=("train", "val", "test"),
        landmarks_dir=ldir,
        features_dir=fdir,
        root_dir=REPO_ROOT,
        skip_extract=True,
    )
    assert summary["mode"] == "production"
    assert summary["D"] == 207
    assert summary["L"] == 48
    assert summary["per_split_counts"] == {"train": 8, "val": 4, "test": 4}
    assert summary["verification"]["zero_signer_leakage"] is True
    for fname in ("train.npz", "val.npz", "test.npz", "feature_meta.json", "preprocess_summary.json"):
        assert (fdir / fname).exists()


def test_real_production_stage2_artifacts_and_worksheet() -> None:
    """Strict QA gate on REAL 30-class production Stage 2 (Thiên) outputs in
    `data/landmarks/`, `data/features/`, `docs/`, and `README.md`."""
    landmarks_dir = REPO_ROOT / "data" / "landmarks"
    features_dir = REPO_ROOT / "data" / "features"

    expected_counts = {"train": 4251, "val": 294, "test": 309}
    expected_signer_counts = {"train": 22, "val": 4, "test": 5}

    # 1. Verify landmark index CSVs across all 4,854 real videos (0 errors)
    for idx_name, exp_n in (
        ("train_landmarks_index.csv", 4251),
        ("train_augmented_landmarks_index.csv", 4251),
        ("val_landmarks_index.csv", 294),
        ("test_landmarks_index.csv", 309),
    ):
        idx_path = landmarks_dir / idx_name
        assert idx_path.exists(), f"Missing landmark index CSV: {idx_path}"
        idx_df = pd.read_csv(idx_path)
        assert len(idx_df) == exp_n
        assert set(idx_df["status"].astype(str).str.lower()) <= {"ok", "skipped"}
        assert float(idx_df["hand_detect_ratio"].mean()) > 0.90
        assert int((idx_df["hand_detect_ratio"] <= 0.0).sum()) == 0

    # 2. Verify packed compressed feature bundles load cleanly with allow_pickle=False
    split_signers: dict[str, set[str]] = {}
    for split, exp_n in expected_counts.items():
        npz_path = features_dir / f"{split}.npz"
        assert npz_path.exists(), f"Missing packed feature bundle: {npz_path}"
        with np.load(npz_path, allow_pickle=False) as z:
            X = z["X"]
            mask = z["mask"]
            y = z["y"]
            sample_ids = z["sample_ids"]
            signer_ids = z["signer_ids"]
            labels = z["labels"]
            is_aug = z["is_augmented"]

            assert X.shape == (exp_n, 48, 207) and X.dtype == np.float32
            assert mask.shape == (exp_n, 48) and mask.dtype == np.bool_
            assert y.shape == (exp_n,) and y.dtype == np.int64
            assert len(sample_ids) == len(signer_ids) == len(labels) == len(is_aug) == exp_n
            assert set(int(c) for c in np.unique(y)) == set(range(30))
            assert np.all(np.isfinite(X)), f"{split}.npz contains NaN or Inf"

            signers = set(signer_ids.tolist())
            assert len(signers) == expected_signer_counts[split]
            split_signers[split] = signers

    # Zero signer overlap across train / val / test
    assert split_signers["train"] & split_signers["val"] == set()
    assert split_signers["train"] & split_signers["test"] == set()
    assert split_signers["val"] & split_signers["test"] == set()
    assert len(split_signers["train"] | split_signers["val"] | split_signers["test"]) == 31

    # 3. Verify preprocess_summary.json & feature_meta.json
    summary_raw = (features_dir / "preprocess_summary.json").read_text(encoding="utf-8")
    assert "demo" not in summary_raw.lower()
    summary = json.loads(summary_raw)
    assert summary["mode"] == "production"
    assert summary["D"] == 207 and summary["L"] == 48
    assert summary["total_samples"] == 4854
    assert summary["per_split_counts"] == expected_counts
    assert summary["zero_hand_samples_count"] == 0
    assert summary["overall_detection_stats"]["mean_hand_detect_ratio"] > 0.90
    assert summary["overall_detection_stats"]["mean_pose_detect_ratio"] == pytest.approx(1.0)
    assert summary["zero_signer_leakage"] is True

    meta_raw = (features_dir / "feature_meta.json").read_text(encoding="utf-8")
    assert "demo" not in meta_raw.lower()
    meta = json.loads(meta_raw)
    assert meta["D"] == 207 and meta["L"] == 48
    assert len(meta["feature_names"]) == 207
    assert meta["per_split_counts"] == expected_counts
    assert meta["signer_overlap_check_passed"] is True
    assert len(meta["class_list"]) == 30

    # 4. Verify Thiên's worksheet (.md & .docx) and README Stage 2 section (zero demo)
    phieu_md = REPO_ROOT / "docs" / "phieu_tim_hieu_phan_2_thien.md"
    phieu_docx = REPO_ROOT / "docs" / "Phieu_Tim_Hieu_Phan_2_Thien.docx"
    assert phieu_md.exists() and phieu_md.stat().st_size > 2000
    assert phieu_docx.exists() and phieu_docx.stat().st_size > 5000
    phieu_text = phieu_md.read_text(encoding="utf-8")
    assert "demo" not in phieu_text.lower()
    for token in ("207", "48", "4.251", "294", "309"):
        assert token in phieu_text or token.replace(".", ",") in phieu_text

    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "demo" not in readme_text.lower()
    assert "phieu_tim_hieu_phan_2_thien" in readme_text
    assert "4251, 48, 207" in readme_text or "(4251, 48, 207)" in readme_text


def test_process_realtime_window_matches_pipeline() -> None:
    """Kiểm tra helper cửa sổ trượt cho Phần 4 (nội suy tay mất ngắn, cắt nghỉ,
    resample về L=48 và hỗ trợ add_velocity=True)."""
    cfg = _load_preprocess_cfg()
    raw, _ = _make_synthetic_raw_landmarks(num_frames=30, idle_frames=3, dropout_gap=2, seed=7)
    state = make_state()
    window = []
    for t in range(30):
        frame = {
            "pose": raw["pose"][t],
            "left_hand": raw["left_hand"][t],
            "right_hand": raw["right_hand"][t],
            "face": raw["face"][t],
            "pose_mask": raw["pose_mask"][t],
            "left_hand_mask": raw["left_hand_mask"][t],
            "right_hand_mask": raw["right_hand_mask"][t],
        }
        window.append(normalize_frame(frame, cfg, state=state))
    buf = np.stack(window, axis=0)
    assert buf.shape == (30, 207)

    X_rt, mask_rt = process_realtime_window(buf, cfg)
    assert X_rt.shape == (48, 207) and X_rt.dtype == np.float32
    assert mask_rt.shape == (48,) and mask_rt.dtype == np.bool_
    assert bool(mask_rt.all())
    assert np.all(np.isfinite(X_rt))

    # Kiểm tra các frame giữa đã được nội suy không còn toàn 0 ở tay
    offset = 0
    lh_s, lh_e = 0, 0
    for bname, bdim in normalize_mod.feature_blocks(cfg):
        if bname == "left_hand":
            lh_s, lh_e = offset, offset + int(bdim)
            break
        offset += int(bdim)
    assert np.all(np.any(np.abs(X_rt[:, lh_s:lh_e]) > 0.0, axis=1))

    # Kiểm tra với add_velocity=True -> 414 chiều
    cfg_vel = json.loads(json.dumps(cfg))
    cfg_vel["normalization"]["add_velocity"] = True
    X_vel, mask_vel = process_realtime_window(buf, cfg_vel)
    assert X_vel.shape == (48, 414) and X_vel.dtype == np.float32
    assert np.all(np.isfinite(X_vel))
    assert bool(mask_vel.all())

    # Cửa sổ rỗng vẫn trả shape chuẩn toàn 0, mask False
    X_empty, mask_empty = process_realtime_window(np.zeros((0, 207), dtype=np.float32), cfg)
    assert X_empty.shape == (48, 207)
    assert not bool(mask_empty.any())


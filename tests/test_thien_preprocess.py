"""Comprehensive automated tests for Giai đoạn 2 — Tiền xử lý (Thiên).

Covers Steps 2.1–2.4 using fast synthetic landmark arrays & fake MediaPipe result
objects in `tmp_path` (no real webcam or MediaPipe detections required):
  - Step 2.1: `landmark_schema.py`, `extract_landmarks.py` (`results_to_arrays`,
    raw `.npz` save/load/validate)
  - Step 2.2: `normalize.py` (wrist-origin + translation/scale invariance,
    missing hands -> zeros, shoulder-based pose normalization, `normalize_frame`
    vs `normalize_sequence` consistency for Linh's real-time deployment)
  - Step 2.3: `sequence.py` (`interpolate_missing` <= max_gap only, idle trimming,
    resample/pad exact `target_len` & mask, velocity features, no NaN/Inf)
  - Step 2.4: `build_features.py` & `scripts/run_thien_preprocess.py --demo`
    (packed `{train,val,test}.npz` shapes, `y` in 0..29, `D == feature_dim(cfg)`,
    zero signer overlap, consistent `feature_meta.json`)
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

from scripts.run_thien_preprocess import (
    generate_demo_landmarks,
    generate_synthetic_raw_landmarks,
    run_demo_preprocess,
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
    # None results -> all zeros + all False masks
    empty = results_to_arrays(None)
    assert empty["pose"].shape == (POSE_N, POSE_DIM) and empty["pose"].dtype == np.float32
    assert empty["face"].shape == (FACE_N, FACE_DIM) and empty["face"].dtype == np.float32
    assert empty["left_hand"].shape == (HAND_N, HAND_DIM)
    assert empty["right_hand"].shape == (HAND_N, HAND_DIM)
    assert not (
        empty["pose_mask"] or empty["face_mask"] or empty["left_hand_mask"] or empty["right_hand_mask"]
    )
    assert float(np.max(np.abs(empty["pose"]))) == 0.0

    # Fake results with pose, refined 478-pt face (truncated to 468), left hand present, right hand None
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

    # NaN in fake landmarks degrades safely to zero + mask False
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
    arrays, meta = generate_synthetic_raw_landmarks(
        num_frames=18,
        class_id=4,
        sample_id="vsl_test_001",
        label="hứa",
        signer_id="kaggle_signer_north",
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
    assert loaded["signer_id"] == "kaggle_signer_north"
    assert loaded["split"] == "train"

    # Negative validation: undetected frame must be zero-filled
    corrupt = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in loaded.items()}
    corrupt["left_hand_mask"][0] = False
    corrupt["left_hand"][0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="zero-filled"):
        validate_raw_landmarks(corrupt)

    # Negative validation: NaN/Inf rejected
    corrupt_nan = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in loaded.items()}
    corrupt_nan["pose"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate_raw_landmarks(corrupt_nan)

    # Negative validation: wrong shape rejected
    corrupt_shape = dict(loaded)
    corrupt_shape["right_hand"] = np.zeros((18, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        validate_raw_landmarks(corrupt_shape)

    # Empty frame contract helper check
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
    # Ensure non-degenerate wrist -> middle_mcp distance
    hand[:, MIDDLE_MCP, :] = hand[:, WRIST, :] + np.array([0.04, -0.09, -0.02], dtype=np.float32)

    norm = normalize_hand(hand)
    assert norm.dtype == np.float32 and norm.shape == (12, HAND_N, 3)
    assert np.all(np.isfinite(norm))

    # Wrist is exact origin (0, 0, 0), and ||middle_mcp|| == 1.0
    np.testing.assert_allclose(norm[:, WRIST, :], 0.0, atol=1e-6)
    mcp_norms = np.linalg.norm(norm[:, MIDDLE_MCP, :], axis=-1)
    np.testing.assert_allclose(mcp_norms, 1.0, atol=1e-5)

    # Invariant to arbitrary translation + positive uniform scale
    shift = np.array([1.75, -0.82, 0.33], dtype=np.float32)
    transformed = hand * 3.4 + shift
    norm_transformed = normalize_hand(transformed)
    np.testing.assert_allclose(norm_transformed, norm, atol=1e-5)

    # Missing hands -> exact zeros (via present=False mask AND degenerate zero scale)
    present = np.ones(12, dtype=bool)
    present[[2, 5, 9]] = False
    norm_masked = normalize_hand(hand, present=present)
    np.testing.assert_array_equal(norm_masked[~present], 0.0)
    assert np.any(norm_masked[present] != 0.0)

    # Degenerate zero wrist->middle_mcp distance -> zeros (no NaN/Inf)
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

    # Frame 3 loses shoulders (low visibility / zeros) -> should fall back to frame 2's shoulder frame
    pose[3, L_SHOULDER] = 0.0
    pose[3, R_SHOULDER] = 0.0

    norm_p = normalize_pose(pose)
    assert norm_p.shape == (T, len(UPPER_BODY_POSE_IDX), 3)
    assert np.all(np.isfinite(norm_p))

    # Shoulder midpoint is origin (0, 0, 0) and shoulder width is 1.0 on valid frames
    for t in (0, 1, 2, 4, 5):
        mid = 0.5 * (norm_p[t, L_SHOULDER] + norm_p[t, R_SHOULDER])
        dist = float(np.linalg.norm(norm_p[t, L_SHOULDER] - norm_p[t, R_SHOULDER]))
        np.testing.assert_allclose(mid, 0.0, atol=1e-6)
        assert dist == pytest.approx(1.0, abs=1e-5)

    # Frame 3 non-shoulder upper-body joints still normalized cleanly using previous shoulder center/scale
    np.testing.assert_allclose(norm_p[3, NOSE], norm_p[2, NOSE], atol=1e-5)

    # Hand global position relative to shoulder center / shoulder width
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
    arrays, meta = generate_synthetic_raw_landmarks(num_frames=14, idle_frames=2, dropout_gap=2, seed=99)
    # Also simulate a mid-sequence shoulder dropout at frame 6 to verify stateful parity
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
    # T = 16:
    #  [0..1]   leading missing (2 frames) -> MUST remain 0 / False
    #  [2]      value = 2.0
    #  [3..5]   internal gap of 3 (<= max_gap=4) -> MUST be linearly interpolated (3.0, 4.0, 5.0)
    #  [6]      value = 6.0
    #  [7..11]  internal gap of 5 (> max_gap=4) -> MUST remain 0 / False
    #  [12..13] values = 12.0, 13.0
    #  [14..15] trailing missing (2 frames) -> MUST remain 0 / False
    T = 16
    arr = np.zeros((T, 2, 3), dtype=np.float32)
    mask = np.zeros(T, dtype=bool)
    for t in (2, 6, 12, 13):
        arr[t] = float(t)
        mask[t] = True

    filled, new_mask = interpolate_missing(arr, mask, max_gap=4, return_mask=True)
    assert np.all(np.isfinite(filled))

    # Leading & trailing gaps untouched
    assert not new_mask[0] and not new_mask[1]
    assert not new_mask[14] and not new_mask[15]
    np.testing.assert_array_equal(filled[[0, 1, 14, 15]], 0.0)

    # Short gap (len 3 <= 4) linearly interpolated
    assert np.all(new_mask[2:7])
    np.testing.assert_allclose(filled[3], 3.0, atol=1e-5)
    np.testing.assert_allclose(filled[4], 4.0, atol=1e-5)
    np.testing.assert_allclose(filled[5], 5.0, atol=1e-5)

    # Long gap (len 5 > 4) left zero & False
    assert not np.any(new_mask[7:12])
    np.testing.assert_array_equal(filled[7:12], 0.0)


def test_trim_resample_pad_and_to_fixed_length() -> None:
    cfg = _load_preprocess_cfg()
    D = feature_dim(cfg)

    # trim_idle_frames drops leading/trailing idle frames
    feats = np.arange(10 * 4, dtype=np.float32).reshape(10, 4)
    hand_mask = np.array([False, False, True, True, False, True, False, False, False, False])
    trimmed, trimmed_mask = trim_idle_frames(feats, hand_mask)
    assert trimmed.shape == (4, 4)
    np.testing.assert_array_equal(trimmed[0], feats[2])
    np.testing.assert_array_equal(trimmed[-1], feats[5])
    assert trimmed_mask.shape == (4,)

    # All-false hand_mask -> preserves non-empty sequence
    kept_feats, _ = trim_idle_frames(feats, np.zeros(10, dtype=bool))
    assert kept_feats.shape == (10, 4)

    # resample_sequence: exact target_len across T=0, T=1, T<L, T>L
    assert resample_sequence(np.zeros((0, 5), dtype=np.float32), 48).shape == (48, 5)
    r1 = resample_sequence(np.full((1, 5), 3.5, dtype=np.float32), 48)
    assert r1.shape == (48, 5) and np.allclose(r1, 3.5)

    ramp = np.linspace(0.0, 10.0, 20, dtype=np.float32)[:, None]
    r_up = resample_sequence(ramp, 48)
    assert r_up.shape == (48, 1)
    assert r_up[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert r_up[-1, 0] == pytest.approx(10.0, abs=1e-5)
    assert resample_sequence(np.ones((75, 5), dtype=np.float32), 48).shape == (48, 5)

    # pad_or_truncate: exact target_len + accurate mask
    short = np.ones((12, 4), dtype=np.float32)
    padded, pmask = pad_or_truncate(short, 48, pad_value=0.0)
    assert padded.shape == (48, 4) and pmask.shape == (48,) and pmask.dtype == np.bool_
    assert np.all(pmask[:12]) and not np.any(pmask[12:])
    np.testing.assert_array_equal(padded[12:], 0.0)

    long_x = np.arange(60 * 4, dtype=np.float32).reshape(60, 4)
    trunc, tmask = pad_or_truncate(long_x, 48, truncate="center")
    assert trunc.shape == (48, 4) and np.all(tmask)
    np.testing.assert_array_equal(trunc[0], long_x[6])

    # to_fixed_length with resample, pad, empty input, and velocity
    seq_in = np.ones((25, D), dtype=np.float32)
    X_res, m_res = to_fixed_length(seq_in, np.ones(25, dtype=bool), cfg)
    assert X_res.shape == (48, D) and m_res.shape == (48,) and np.all(m_res) and np.all(np.isfinite(X_res))

    cfg_pad = {**cfg, "sequence": {**cfg["sequence"], "method": "pad"}}
    X_pad, m_pad = to_fixed_length(seq_in, np.ones(25, dtype=bool), cfg_pad)
    assert X_pad.shape == (48, D) and int(m_pad.sum()) == 25 and not np.any(m_pad[25:])

    # Velocity doubles D and zeroes velocity on padded frames
    X_vel = add_velocity(X_pad, m_pad)
    assert X_vel.shape == (48, 2 * D)
    np.testing.assert_array_equal(X_vel[25:, D:], 0.0)
    assert np.all(np.isfinite(X_vel))


# ---------------------------------------------------------------------------
# 4. Feature Builder, Signer Leakage Guard & End-to-End Demo (Step 2.4)
# ---------------------------------------------------------------------------


def test_build_features_packing_zero_hand_skip_and_signer_leakage_guard(tmp_path: Path) -> None:
    # Negative test: assert_no_signer_overlap catches any overlap across splits
    with pytest.raises(ValueError, match="signer overlap"):
        assert_no_signer_overlap(
            {"train": ["s1", "s2"], "val": ["s2"], "test": ["s3"]}
        )
    with pytest.raises(ValueError, match="signer overlap"):
        assert_no_signer_overlap(
            {"train": ["s1"], "val": ["s2"], "test": ["s1"]}
        )
    assert_no_signer_overlap({"train": ["s1", "s2"], "val": ["s3"], "test": ["s4"]})

    # Generate synthetic landmarks in tmp_path, plus 1 zero-hand sample in train
    cfg = _load_preprocess_cfg()
    ldir = tmp_path / "landmarks"
    fdir = tmp_path / "features"
    generate_demo_landmarks(ldir, root_dir=REPO_ROOT, num_classes=3, seed=42)

    # Append a zero-hand video to train_augmented_landmarks_index.csv to verify zero-hand tracking/skipping
    zero_arrays = empty_video_landmarks(20)
    zero_meta = {
        "fps": 30.0,
        "num_frames": 20,
        "width": 224,
        "height": 224,
        "sample_id": "train_zero_hands_sample",
        "label": "giúp đỡ",
        "class_id": 0,
        "signer_id": "kaggle_signer_north",
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
                "signer_id": "kaggle_signer_north",
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

    # Build with skip_zero_hands=True
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

            # Verify per-sample .npy matches packed X[0]
            first_npy = fdir / split / f"{sample_ids[0]}.npy"
            assert first_npy.exists()
            np.testing.assert_array_equal(np.load(first_npy), X[0])

            split_signers[split] = set(signer_ids.tolist())

    assert split_signers["train"] & split_signers["val"] == set()
    assert split_signers["train"] & split_signers["test"] == set()
    assert split_signers["val"] & split_signers["test"] == set()

    # process_sample sanity check
    raw_arrs, raw_m = generate_synthetic_raw_landmarks(num_frames=32, idle_frames=3, dropout_gap=4)
    X_s, mask_s, info_s = process_sample({**raw_arrs, **raw_m}, cfg)
    assert X_s.shape == (L, D) and mask_s.shape == (L,) and np.all(np.isfinite(X_s))
    assert info_s["num_frames_trimmed"] < info_s["num_frames_raw"]


def test_end_to_end_thien_demo_preprocess(tmp_path: Path) -> None:
    summary = run_demo_preprocess(
        cfg=REPO_ROOT / "configs" / "preprocess_config.yaml",
        landmarks_dir=tmp_path / "demo_landmarks",
        features_dir=tmp_path / "demo_features",
        root_dir=REPO_ROOT,
        num_classes=5,
        seed=42,
    )

    assert summary["mode"] == "demo"
    assert summary["D"] == 207
    assert summary["L"] == 48
    assert summary["per_split_counts"] == {"train": 40, "val": 5, "test": 5}
    assert summary["verification"]["zero_signer_leakage"] is True
    assert summary["verification"]["hand_translation_scale_invariance_max_err"] < 1e-5
    assert summary["verification"]["frame_vs_sequence_max_err"] < 1e-5

    for fname in ("train.npz", "val.npz", "test.npz", "feature_meta.json", "preprocess_summary.json"):
        assert (tmp_path / "demo_features" / fname).exists()

"""Giai đoạn 2 — Tiền xử lý dữ liệu (Thiên).

Submodules:
- :mod:`src.preprocess.landmark_schema` — shared landmark schema & raw ``.npz`` I/O
- :mod:`src.preprocess.extract_landmarks` — MediaPipe Holistic extraction (Step 2.1)
- :mod:`src.preprocess.normalize` — body/hand landmark normalization (Step 2.2)
- :mod:`src.preprocess.sequence` — gap interpolation, idle trim, fixed length (Step 2.3)
- :mod:`src.preprocess.build_features` — offline feature builder (Step 2.4)
"""

from __future__ import annotations

from typing import Any

from src.preprocess.landmark_schema import (
    ARRAY_KEYS,
    FACE_DIM,
    FACE_N,
    FACE_NOSE_TIP,
    FACE_SUBSET_GROUPS,
    FACE_SUBSET_IDX,
    HAND_DIM,
    HAND_N,
    INDEX_MCP,
    INDEX_TIP,
    L_ELBOW,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    MASK_KEYS,
    META_KEYS,
    MIDDLE_MCP,
    MIDDLE_TIP,
    NOSE,
    PARTS,
    PINKY_MCP,
    PINKY_TIP,
    POSE_DIM,
    POSE_N,
    R_ELBOW,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    RING_MCP,
    RING_TIP,
    THUMB_TIP,
    UPPER_BODY_POSE_IDX,
    WRIST,
    detection_ratios,
    empty_frame_landmarks,
    empty_video_landmarks,
    load_raw_landmarks,
    save_raw_landmarks,
    stack_frames,
    validate_raw_landmarks,
)

_EXTRACT_EXPORTS = {
    "DEFAULT_HOLISTIC",
    "INDEX_COLUMNS",
    "create_holistic",
    "extract_manifest",
    "extract_video_landmarks",
    "landmark_rel_path",
    "results_to_arrays",
}


def __getattr__(name: str) -> Any:
    if name in _EXTRACT_EXPORTS:
        from src.preprocess import extract_landmarks as _ext

        return getattr(_ext, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "POSE_N", "FACE_N", "HAND_N", "POSE_DIM", "FACE_DIM", "HAND_DIM",
    "WRIST", "THUMB_TIP", "INDEX_MCP", "INDEX_TIP", "MIDDLE_MCP", "MIDDLE_TIP",
    "RING_MCP", "RING_TIP", "PINKY_MCP", "PINKY_TIP",
    "NOSE", "L_SHOULDER", "R_SHOULDER", "L_ELBOW", "R_ELBOW", "L_WRIST", "R_WRIST",
    "L_HIP", "R_HIP", "UPPER_BODY_POSE_IDX", "FACE_SUBSET_GROUPS", "FACE_SUBSET_IDX",
    "FACE_NOSE_TIP",
    "PARTS", "ARRAY_KEYS", "MASK_KEYS", "META_KEYS",
    "empty_frame_landmarks", "empty_video_landmarks", "stack_frames",
    "save_raw_landmarks", "load_raw_landmarks", "validate_raw_landmarks",
    "detection_ratios",
    "DEFAULT_HOLISTIC", "INDEX_COLUMNS", "results_to_arrays", "create_holistic",
    "extract_video_landmarks", "extract_manifest", "landmark_rel_path",
]

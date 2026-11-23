"""Shared landmark schema for Giai đoạn 2 — Tiền xử lý (Thiên).

This module is the contract between the landmark extractor
(``src.preprocess.extract_landmarks``) and the normalizer / feature builder.
It has NO dependency on MediaPipe so it can be imported anywhere (tests, CI).

Raw per-video landmark file (``.npz``)
--------------------------------------
Arrays (T = number of decoded frames):

=================  ===============  =======================================
key                shape            dtype / meaning
=================  ===============  =======================================
pose               (T, 33, 4)       float32  x, y, z, visibility
face               (T, 468, 3)      float32  x, y, z
left_hand          (T, 21, 3)       float32  x, y, z
right_hand         (T, 21, 3)       float32  x, y, z
pose_mask          (T,)             bool     True = pose detected
face_mask          (T,)             bool     True = face detected
left_hand_mask     (T,)             bool     True = left hand detected
right_hand_mask    (T,)             bool     True = right hand detected
=================  ===============  =======================================

Undetected frames are filled with zeros (and their mask is False).

Scalar metadata (stored as 0-d arrays, returned as Python scalars by
:func:`load_raw_landmarks`): ``fps`` (float), ``num_frames`` (int),
``width`` (int), ``height`` (int), ``sample_id`` (str), ``label`` (str),
``class_id`` (int), ``signer_id`` (str), ``split`` (str).

Coordinates are MediaPipe *normalized image coordinates*: x, y in [0, 1]
relative to image width/height (may slightly exceed the range when a
landmark is out of frame), z is relative depth (pose: hip-center origin;
hands: wrist origin; face: face-center origin), same scale as x.
NO normalization is applied here — that is the normalizer's job.

Handedness
----------
MediaPipe Holistic's ``left_hand_landmarks`` is the signer's *anatomical*
left hand when the input video is NOT mirrored (a normal, front-facing
camera recording). In the image it therefore usually appears on the
viewer's RIGHT side. Holistic derives handedness from the pose model, so
this holds for all Stage-1 sources (QIPEDC, VSL-v2, self-recorded videos
saved without mirroring). Frames are NEVER flipped in this pipeline
(``augmentation.allow_hflip: false`` in Stage 1) because VSL signs depend on
the dominant hand and spatial direction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# Sizes
# ---------------------------------------------------------------------------
POSE_N = 33
FACE_N = 468
HAND_N = 21

POSE_DIM = 4  # x, y, z, visibility
FACE_DIM = 3  # x, y, z
HAND_DIM = 3  # x, y, z

# ---------------------------------------------------------------------------
# Hand landmark indices (MediaPipe Hands, 21 points)
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_MCP = 13
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20

# ---------------------------------------------------------------------------
# Pose landmark indices (MediaPipe BlazePose, 33 points)
# ---------------------------------------------------------------------------
NOSE = 0
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_HIP = 23
R_HIP = 24

# 0..24: head (0-10), shoulders, elbows, wrists, hand points (17-22), hips.
# Legs/feet (25-32) are irrelevant for sign language and often out of frame.
UPPER_BODY_POSE_IDX: list[int] = list(range(0, 25))

# ---------------------------------------------------------------------------
# Face subset (indices into the 468-point FaceMesh) for VSL non-manual signals
# (mouth shape, eyebrow raise/frown, eye opening, head orientation).
# ---------------------------------------------------------------------------
FACE_SUBSET_GROUPS: dict[str, list[int]] = {
    # Outer lips: corners 61/291, upper 0/37/267, lower 17/84/314
    "lips_outer": [61, 291, 0, 17, 37, 267, 84, 314],
    # Inner lips: corners 78/308, upper 13, lower 14
    "lips_inner": [78, 308, 13, 14],
    # Eyebrows (inner -> outer)
    "left_eyebrow": [336, 296, 334, 293, 300],
    "right_eyebrow": [107, 66, 105, 63, 70],
    # Eyes: outer corner, inner corner, upper lid, lower lid
    "left_eye": [263, 362, 386, 374],
    "right_eye": [33, 133, 159, 145],
    # Nose: tip, bridge
    "nose": [1, 168],
    # Face contour: forehead, chin, left/right cheek (head pose cues)
    "contour": [10, 152, 454, 234],
}
FACE_SUBSET_IDX: list[int] = [i for grp in FACE_SUBSET_GROUPS.values() for i in grp]
FACE_NOSE_TIP = 1

# ---------------------------------------------------------------------------
# NPZ keys
# ---------------------------------------------------------------------------
PARTS: dict[str, tuple[int, int]] = {
    "pose": (POSE_N, POSE_DIM),
    "face": (FACE_N, FACE_DIM),
    "left_hand": (HAND_N, HAND_DIM),
    "right_hand": (HAND_N, HAND_DIM),
}
ARRAY_KEYS: tuple[str, ...] = tuple(PARTS.keys())
MASK_KEYS: tuple[str, ...] = tuple(f"{k}_mask" for k in ARRAY_KEYS)
META_KEYS: tuple[str, ...] = (
    "fps",
    "num_frames",
    "width",
    "height",
    "sample_id",
    "label",
    "class_id",
    "signer_id",
    "split",
)
_META_TYPES: dict[str, type] = {
    "fps": float,
    "num_frames": int,
    "width": int,
    "height": int,
    "sample_id": str,
    "label": str,
    "class_id": int,
    "signer_id": str,
    "split": str,
}


def empty_frame_landmarks() -> dict[str, Any]:
    """Return zero-filled landmarks for ONE frame with all masks False.

    Keys: ``pose (33,4)``, ``face (468,3)``, ``left_hand (21,3)``,
    ``right_hand (21,3)`` as float32 and the four ``*_mask`` as Python bools.
    """
    out: dict[str, Any] = {}
    for key, (n, d) in PARTS.items():
        out[key] = np.zeros((n, d), dtype=np.float32)
        out[f"{key}_mask"] = False
    return out


def empty_video_landmarks(num_frames: int) -> dict[str, np.ndarray]:
    """Zero-filled arrays + all-False masks for a video of ``num_frames``."""
    t = int(num_frames)
    out: dict[str, np.ndarray] = {}
    for key, (n, d) in PARTS.items():
        out[key] = np.zeros((t, n, d), dtype=np.float32)
        out[f"{key}_mask"] = np.zeros((t,), dtype=bool)
    return out


def stack_frames(frames: list[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Stack per-frame dicts (as from :func:`empty_frame_landmarks`) into
    ``(T, ...)`` arrays and ``(T,)`` masks. Works for ``T == 0``."""
    if not frames:
        return empty_video_landmarks(0)
    out: dict[str, np.ndarray] = {}
    for key in ARRAY_KEYS:
        out[key] = np.stack([np.asarray(f[key], dtype=np.float32) for f in frames]).astype(
            np.float32, copy=False
        )
        out[f"{key}_mask"] = np.asarray([bool(f[f"{key}_mask"]) for f in frames], dtype=bool)
    return out


def _coerce_meta(key: str, value: Any) -> Any:
    typ = _META_TYPES[key]
    if value is None:
        return "" if typ is str else (float("nan") if typ is float else -1)
    if isinstance(value, np.ndarray):
        value = value.item()
    if typ is str:
        return str(value)
    try:
        if typ is int:
            return int(value)
        return float(value)
    except (TypeError, ValueError):
        return -1 if typ is int else float("nan")


def validate_raw_landmarks(d: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if ``d`` does not follow the raw landmark contract.

    Checks presence, dtype and shape of all arrays/masks, consistent ``T``
    across parts, ``num_frames == T`` when present, finite values, and that
    undetected frames are all-zero.
    """
    missing = [k for k in (*ARRAY_KEYS, *MASK_KEYS) if k not in d]
    if missing:
        raise ValueError(f"missing landmark keys: {missing}")

    t: int | None = None
    for key, (n, dim) in PARTS.items():
        arr = np.asarray(d[key])
        mask = np.asarray(d[f"{key}_mask"])
        if arr.dtype != np.float32:
            raise ValueError(f"{key}: dtype {arr.dtype}, expected float32")
        if arr.ndim != 3 or arr.shape[1:] != (n, dim):
            raise ValueError(f"{key}: shape {arr.shape}, expected (T, {n}, {dim})")
        if mask.dtype != np.bool_:
            raise ValueError(f"{key}_mask: dtype {mask.dtype}, expected bool")
        if mask.shape != (arr.shape[0],):
            raise ValueError(f"{key}_mask: shape {mask.shape}, expected ({arr.shape[0]},)")
        if t is None:
            t = arr.shape[0]
        elif arr.shape[0] != t:
            raise ValueError(f"{key}: T={arr.shape[0]} differs from T={t} of other parts")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{key}: contains NaN/Inf")
        if arr.shape[0] and np.any(arr[~mask] != 0):
            raise ValueError(f"{key}: undetected frames must be zero-filled")

    if "num_frames" in d:
        nf = _coerce_meta("num_frames", d["num_frames"])
        if nf != t:
            raise ValueError(f"num_frames={nf} but arrays have T={t}")


def save_raw_landmarks(
    path: str | Path, arrays: Mapping[str, Any], meta: Mapping[str, Any] | None = None
) -> Path:
    """Validate and write a raw landmark ``.npz`` (compressed).

    ``arrays`` must contain the 4 part arrays and 4 masks. ``meta`` may contain
    any of :data:`META_KEYS`; missing keys get defaults (``num_frames`` is
    always set from the arrays). Written atomically via a temp file.
    """
    path = Path(path)
    meta = dict(meta or {})
    payload: dict[str, np.ndarray] = {}
    for key in ARRAY_KEYS:
        payload[key] = np.asarray(arrays[key], dtype=np.float32)
        payload[f"{key}_mask"] = np.asarray(arrays[f"{key}_mask"], dtype=bool)
    meta["num_frames"] = int(payload["pose"].shape[0])
    for key in META_KEYS:
        payload[key] = np.asarray(_coerce_meta(key, meta.get(key)))
    validate_raw_landmarks(payload)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)
    return path


def load_raw_landmarks(path: str | Path) -> dict[str, Any]:
    """Load a raw landmark ``.npz``: arrays as numpy, metadata as Python scalars."""
    out: dict[str, Any] = {}
    with np.load(Path(path), allow_pickle=False) as z:
        for key in z.files:
            val = z[key]
            if key in _META_TYPES:
                out[key] = _coerce_meta(key, val)
            else:
                out[key] = val
    validate_raw_landmarks(out)
    return out


def detection_ratios(d: Mapping[str, Any]) -> dict[str, float]:
    """Fraction of frames where each part was detected (+ ``hand`` = either hand)."""
    t = int(np.asarray(d["pose_mask"]).shape[0])
    if t == 0:
        return {"pose": 0.0, "face": 0.0, "left_hand": 0.0, "right_hand": 0.0, "hand": 0.0}
    r = {k: float(np.mean(d[f"{k}_mask"])) for k in ARRAY_KEYS}
    r["hand"] = float(np.mean(np.logical_or(d["left_hand_mask"], d["right_hand_mask"])))
    return r


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
]

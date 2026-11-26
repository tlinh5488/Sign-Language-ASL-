"""Giai đoạn 2 — Tiền xử lý (Thiên), Step 2.1: landmark extraction with MediaPipe Holistic.

Reads Stage-1 manifests (``data/splits/train_augmented.csv``, ``val.csv``,
``test.csv``), runs MediaPipe Holistic frame by frame on every video and writes
one raw landmark file per video::

    <output_dir>/<split>/<class_id>/<sample_id>.npz

plus an index CSV per split ``<output_dir>/<split>_landmarks_index.csv`` (and
``train_augmented_landmarks_index.csv`` when extracting ``train_augmented.csv``).
The ``.npz`` layout is defined in :mod:`src.preprocess.landmark_schema`.
Coordinates are left as MediaPipe normalized image coords (no normalization),
frames are never flipped.

MediaPipe is imported lazily (inside functions) so this module and
:func:`results_to_arrays` can be imported/tested without it.

CLI::

    python -m src.preprocess.extract_landmarks --config configs/preprocess_config.yaml \
        [--manifest CSV ...] [--splits train val test] [--output-dir data/landmarks] \
        [--overwrite] [--limit N] [--workers N]
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.preprocess.landmark_schema import (
    ARRAY_KEYS,
    PARTS,
    detection_ratios,
    empty_frame_landmarks,
    load_raw_landmarks,
    save_raw_landmarks,
    stack_frames,
)

LOGGER = logging.getLogger("extract_landmarks")

INDEX_COLUMNS = [
    "sample_id",
    "landmark_path",
    "video_path",
    "label",
    "class_id",
    "signer_id",
    "split",
    "is_augmented",
    "num_frames",
    "fps",
    "hand_detect_ratio",
    "left_hand_detect_ratio",
    "right_hand_detect_ratio",
    "pose_detect_ratio",
    "face_detect_ratio",
    "status",
    "error",
]

DEFAULT_HOLISTIC: dict[str, Any] = {
    "static_image_mode": False,
    "model_complexity": 1,
    "smooth_landmarks": True,
    "refine_face_landmarks": False,
    "min_detection_confidence": 0.4,
    "min_tracking_confidence": 0.4,
    "min_side": 0,
    "max_side": 640,
}

# Process-local persistent Holistic instance for multi-worker extraction
_WORKER_HOLISTIC: Any = None
_WORKER_HOLISTIC_CFG: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Pure conversion (testable without MediaPipe)
# ---------------------------------------------------------------------------
def _landmark_list_to_array(landmark_list: Any, n: int, dim: int) -> tuple[np.ndarray, bool]:
    """Convert a MediaPipe ``NormalizedLandmarkList`` (or any object with a
    ``.landmark`` sequence of objects with x/y/z[/visibility]) to ``(n, dim)``."""
    out = np.zeros((n, dim), dtype=np.float32)
    if landmark_list is None:
        return out, False
    lms = getattr(landmark_list, "landmark", landmark_list)
    lms = list(lms) if lms is not None else []
    if not lms:
        return out, False
    for i, lm in enumerate(lms[:n]):
        out[i, 0] = float(getattr(lm, "x", 0.0))
        out[i, 1] = float(getattr(lm, "y", 0.0))
        out[i, 2] = float(getattr(lm, "z", 0.0))
        if dim >= 4:
            out[i, 3] = float(getattr(lm, "visibility", 0.0))
    if not np.all(np.isfinite(out)):
        return np.zeros((n, dim), dtype=np.float32), False
    return out, True


def results_to_arrays(results: Any) -> dict[str, Any]:
    """Convert one MediaPipe Holistic ``results`` object to per-frame arrays.

    Reads ``pose_landmarks``, ``face_landmarks``, ``left_hand_landmarks``,
    ``right_hand_landmarks`` (each may be ``None``). Returns the same layout as
    :func:`landmark_schema.empty_frame_landmarks`: ``pose (33,4)``,
    ``face (468,3)``, ``left_hand (21,3)``, ``right_hand (21,3)`` float32 and
    ``*_mask`` bools. Missing parts are zeros with mask False. Extra points
    (e.g. 478 refined face points) are truncated to the contract size.
    """
    frame = empty_frame_landmarks()
    if results is None:
        return frame
    attr = {
        "pose": "pose_landmarks",
        "face": "face_landmarks",
        "left_hand": "left_hand_landmarks",
        "right_hand": "right_hand_landmarks",
    }
    for key, (n, dim) in PARTS.items():
        arr, ok = _landmark_list_to_array(getattr(results, attr[key], None), n, dim)
        frame[key] = arr
        frame[f"{key}_mask"] = ok
    return frame


# ---------------------------------------------------------------------------
# Video extraction
# ---------------------------------------------------------------------------
def create_holistic(
    *,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.4,
    min_tracking_confidence: float = 0.4,
    static_image_mode: bool = False,
    smooth_landmarks: bool = True,
    refine_face_landmarks: bool = False,
    **_ignored: Any,
) -> Any:
    """Create a MediaPipe Holistic instance (legacy ``mp.solutions`` API)."""
    if "tensorflow.tools.docs" not in sys.modules:
        import types

        docs_mod = types.ModuleType("tensorflow.tools.docs")
        docs_mod.doc_controls = types.SimpleNamespace(do_not_generate_docs=lambda x: x)  # type: ignore[attr-defined]
        sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))
        sys.modules.setdefault("tensorflow.tools", types.ModuleType("tensorflow.tools"))
        sys.modules["tensorflow.tools.docs"] = docs_mod
    try:
        import mediapipe as mp  # lazy import
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "mediapipe is required for landmark extraction: pip install mediapipe==0.10.14"
        ) from exc
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "holistic"):
        raise ImportError(
            f"mediapipe {getattr(mp, '__version__', '?')} has no legacy mp.solutions.holistic; "
            "install mediapipe==0.10.14"
        )
    return mp.solutions.holistic.Holistic(
        static_image_mode=static_image_mode,
        model_complexity=int(model_complexity),
        smooth_landmarks=smooth_landmarks,
        refine_face_landmarks=refine_face_landmarks,
        min_detection_confidence=float(min_detection_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
    )


def _resize_for_holistic(bgr: np.ndarray, min_side: int | None, max_side: int | None) -> np.ndarray:
    """Resize frame preserving aspect ratio so ``min(H, W) >= min_side`` and
    ``max(H, W) <= max_side``. Since MediaPipe outputs normalized [0, 1] image
    coordinates, aspect-preserving resize keeps x, y in the exact same
    normalized coordinate system while speeding up 720p videos."""
    if not min_side and not max_side:
        return bgr
    import cv2

    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return bgr
    scale = 1.0
    if min_side and min(h, w) < int(min_side):
        scale = float(min_side) / float(min(h, w))
    elif max_side and max(h, w) > int(max_side):
        scale = float(max_side) / float(max(h, w))
    if abs(scale - 1.0) < 1e-3:
        return bgr
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(bgr, (new_w, new_h), interpolation=interp)


def extract_video_landmarks(
    video_path: str | Path,
    holistic: Any = None,
    *,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.4,
    min_tracking_confidence: float = 0.4,
    max_frames: int | None = None,
    min_side: int | None = 0,
    max_side: int | None = 640,
    reset_holistic: bool = True,
    **holistic_kwargs: Any,
) -> dict[str, Any]:
    """Run Holistic on every frame of ``video_path``.

    If ``holistic`` is None a fresh instance is created (and closed) for this
    video. When reusing a ``holistic`` instance across videos, ``holistic.reset()``
    is called before processing if ``reset_holistic=True``.

    Returns a dict with ``pose, face, left_hand, right_hand`` ``(T, N, D)``
    float32 arrays, ``*_mask`` ``(T,)`` bool, and ``fps`` (float),
    ``num_frames`` (int), ``width``, ``height`` (int, original video dimensions).
    Raises ``IOError`` if the video cannot be opened.
    """
    import cv2  # lazy: keep schema-level imports light

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise IOError(f"cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    own = holistic is None
    if own:
        holistic = create_holistic(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            **holistic_kwargs,
        )
    elif reset_holistic and hasattr(holistic, "reset"):
        holistic.reset()

    frames: list[dict[str, Any]] = []
    try:
        while True:
            if max_frames is not None and len(frames) >= int(max_frames):
                break
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            if not width or not height:
                height, width = bgr.shape[:2]
            proc_bgr = _resize_for_holistic(bgr, min_side=min_side, max_side=max_side)
            rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            frames.append(results_to_arrays(results))
    finally:
        cap.release()
        if own:
            holistic.close()

    out: dict[str, Any] = dict(stack_frames(frames))
    out["fps"] = fps if np.isfinite(fps) else 0.0
    out["num_frames"] = len(frames)
    out["width"] = int(width)
    out["height"] = int(height)
    return out


# ---------------------------------------------------------------------------
# Exact deterministic Stage-1 augmentation of parent MediaPipe landmarks
# ---------------------------------------------------------------------------
def _aug_rng(seed: int, parent_sample_id: str, aug_index: int) -> np.random.Generator:
    key = f"{seed}:{parent_sample_id}:{aug_index}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    seed_int = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(seed_int)


def derive_augmented_landmarks(
    parent_raw: Mapping[str, Any],
    *,
    sample_id: str,
    parent_sample_id: str,
    expected_num_frames: int | None = None,
    seed: int = 42,
    speed_range: tuple[float, float] = (0.85, 1.15),
    trim_max_fraction: float = 0.08,
    rotation_deg_range: tuple[float, float] = (-8.0, 8.0),
    scale_range: tuple[float, float] = (0.92, 1.08),
    translation_frac_range: tuple[float, float] = (-0.04, 0.04),
) -> dict[str, Any]:
    """Derive an augmented train clip's raw landmarks from its parent's real
    MediaPipe Holistic landmarks using the exact Stage-1 deterministic temporal
    speed resampling, start/end trim, and 2D spatial affine transform."""
    import cv2

    if "_aug" not in sample_id:
        raise ValueError(f"cannot parse aug_index from sample_id={sample_id!r}")
    aug_idx = int(sample_id.rsplit("_aug", 1)[1])
    rng = _aug_rng(seed, parent_sample_id, aug_idx)

    nf = int(parent_raw["num_frames"])
    width = int(parent_raw.get("width", 224) or 224)
    height = int(parent_raw.get("height", 224) or 224)

    # 1. Temporal speed perturbation
    low_s, high_s = speed_range
    speed = float(rng.uniform(low_s, high_s))
    if abs(speed - 1.0) < 0.03:
        speed = high_s if speed >= 1.0 else low_s
    if nf <= 2 or abs(speed - 1.0) < 1e-4:
        indices = list(range(nf))
    else:
        target_len = max(4, int(round(nf / float(speed))))
        indices = [int(round(i)) for i in np.linspace(0, nf - 1, num=target_len)]

    # 2. Temporal start/end trim
    if trim_max_fraction > 0 and len(indices) > 8:
        start_trim = float(rng.uniform(0.0, trim_max_fraction))
        end_trim = float(rng.uniform(0.0, trim_max_fraction))
        if start_trim > 0.01 or end_trim > 0.01:
            s_i = int(round(len(indices) * max(0.0, start_trim)))
            e_i = len(indices) - int(round(len(indices) * max(0.0, end_trim)))
            if e_i - s_i >= 4:
                indices = indices[s_i:e_i]

    if expected_num_frames is not None and expected_num_frames > 0 and len(indices) != int(expected_num_frames):
        raise ValueError(f"derived T={len(indices)} != expected_num_frames={expected_num_frames}")

    # 3. Spatial affine transform (rotation, scale, translation in pixel space -> normalized)
    angle = float(rng.uniform(*rotation_deg_range))
    scale = float(rng.uniform(*scale_range))
    tx = float(rng.uniform(*translation_frac_range))
    ty = float(rng.uniform(*translation_frac_range))

    center = (width / 2.0, height / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
    mat[0, 2] += float(tx) * width
    mat[1, 2] += float(ty) * height

    idx_arr = np.asarray(indices, dtype=np.int64)
    out: dict[str, Any] = {}
    w_f = float(width)
    h_f = float(height)

    for key in ARRAY_KEYS:
        arr = np.asarray(parent_raw[key], dtype=np.float32)[idx_arr].copy()
        mask = np.asarray(parent_raw[f"{key}_mask"], dtype=bool)[idx_arr].copy()
        if mask.any():
            pts = arr[mask]  # (M, N, D)
            px = pts[..., 0] * w_f
            py = pts[..., 1] * h_f
            new_px = mat[0, 0] * px + mat[0, 1] * py + mat[0, 2]
            new_py = mat[1, 0] * px + mat[1, 1] * py + mat[1, 2]
            pts[..., 0] = new_px / w_f
            pts[..., 1] = new_py / h_f
            pts[..., 2] = pts[..., 2] * float(scale)
            arr[mask] = pts
        arr[~mask] = 0.0
        out[key] = arr.astype(np.float32, copy=False)
        out[f"{key}_mask"] = mask

    out["fps"] = float(parent_raw.get("fps", 30.0) or 30.0)
    out["num_frames"] = len(indices)
    out["width"] = width
    out["height"] = height
    return out


# ---------------------------------------------------------------------------
# Worker pool initializer & Manifest extraction
# ---------------------------------------------------------------------------
def _close_worker_holistic() -> None:
    global _WORKER_HOLISTIC
    if _WORKER_HOLISTIC is not None:
        try:
            _WORKER_HOLISTIC.close()
        except Exception:
            pass
        _WORKER_HOLISTIC = None


def _init_worker(holistic_kwargs: dict[str, Any]) -> None:
    """Initialize one persistent MediaPipe Holistic per worker process and pin
    OpenMP/OpenCV thread counts to 1 to avoid multi-worker thread contention."""
    global _WORKER_HOLISTIC, _WORKER_HOLISTIC_CFG
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass
    _WORKER_HOLISTIC_CFG = dict(holistic_kwargs)
    _WORKER_HOLISTIC = create_holistic(**holistic_kwargs)
    atexit.register(_close_worker_holistic)


def _bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes", "y", "t"}
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    return bool(v)


def _str(v: Any) -> str:
    try:
        if v is None or pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


def _int(v: Any, default: int = -1) -> int:
    try:
        if v is None or pd.isna(v):
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def landmark_rel_path(split: str, class_id: Any, sample_id: str) -> str:
    """Relative path ``<split>/<class_id>/<sample_id>.npz`` (posix)."""
    safe_id = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(sample_id))
    return f"{split}/{_int(class_id)}/{safe_id}.npz"


def _index_row(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": job["sample_id"],
        "landmark_path": job["landmark_path"],
        "video_path": job["video_path"],
        "label": job["label"],
        "class_id": job["class_id"],
        "signer_id": job["signer_id"],
        "split": job["split"],
        "is_augmented": job["is_augmented"],
        "num_frames": 0,
        "fps": 0.0,
        "hand_detect_ratio": 0.0,
        "left_hand_detect_ratio": 0.0,
        "right_hand_detect_ratio": 0.0,
        "pose_detect_ratio": 0.0,
        "face_detect_ratio": 0.0,
        "status": "",
        "error": "",
    }


def _fill_stats(row: dict[str, Any], d: Mapping[str, Any]) -> None:
    r = detection_ratios(d)
    row["num_frames"] = int(np.asarray(d["pose_mask"]).shape[0])
    row["fps"] = float(d.get("fps", 0.0) or 0.0)
    row["hand_detect_ratio"] = round(r["hand"], 4)
    row["left_hand_detect_ratio"] = round(r["left_hand"], 4)
    row["right_hand_detect_ratio"] = round(r["right_hand"], 4)
    row["pose_detect_ratio"] = round(r["pose"], 4)
    row["face_detect_ratio"] = round(r["face"], 4)


def _process_job(job: dict[str, Any]) -> dict[str, Any]:
    """Extract one video (never raises; errors are recorded in the row)."""
    global _WORKER_HOLISTIC
    row = _index_row(job)
    out_path = Path(job["abs_out"])
    try:
        if out_path.exists() and not job["overwrite"]:
            try:
                _fill_stats(row, load_raw_landmarks(out_path))
                row["status"] = "skipped"
                return row
            except Exception as exc:  # corrupted file -> re-extract
                LOGGER.warning("re-extracting corrupted %s (%s)", out_path, exc)

        parent_out = job.get("parent_abs_out")
        if job.get("derive_augmented") and parent_out and Path(parent_out).exists():
            try:
                parent_raw = load_raw_landmarks(parent_out)
                d = derive_augmented_landmarks(
                    parent_raw,
                    sample_id=job["sample_id"],
                    parent_sample_id=job["parent_sample_id"],
                    expected_num_frames=job.get("expected_num_frames"),
                )
                meta = {
                    "fps": d["fps"],
                    "num_frames": d["num_frames"],
                    "width": d["width"],
                    "height": d["height"],
                    "sample_id": job["sample_id"],
                    "label": job["label"],
                    "class_id": job["class_id"],
                    "signer_id": job["signer_id"],
                    "split": job["split"],
                }
                save_raw_landmarks(out_path, d, meta)
                _fill_stats(row, d)
                row["status"] = "ok"
                return row
            except Exception as exc:
                LOGGER.warning(
                    "fallback to video extraction for %s (%s)", job["sample_id"], exc
                )

        video = Path(job["abs_video"])
        if not video.exists():
            raise FileNotFoundError(f"video not found: {video}")
        d = extract_video_landmarks(video, holistic=_WORKER_HOLISTIC, **job["holistic"])
        if d["num_frames"] == 0:
            raise ValueError("no frames decoded")
        meta = {
            "fps": d["fps"],
            "num_frames": d["num_frames"],
            "width": d["width"],
            "height": d["height"],
            "sample_id": job["sample_id"],
            "label": job["label"],
            "class_id": job["class_id"],
            "signer_id": job["signer_id"],
            "split": job["split"],
        }
        save_raw_landmarks(out_path, d, meta)
        _fill_stats(row, d)
        row["status"] = "ok"
    except Exception as exc:  # never crash the whole run on one video
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def _resolve(path: str | Path, root: Path) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else root / p


def _run_job_batch(
    batch: list[tuple[int, dict[str, Any]]],
    results: dict[int, dict[str, Any]],
    *,
    workers: int,
    hk: dict[str, Any],
    progress: Any,
    use_holistic_pool: bool = True,
) -> None:
    global _WORKER_HOLISTIC
    if not batch:
        return
    if workers == 1 or len(batch) <= 1 or not use_holistic_pool:
        need_holistic = use_holistic_pool and any(
            (j["overwrite"] or not Path(j["abs_out"]).exists())
            and not (j.get("derive_augmented") and j.get("parent_abs_out") and Path(j["parent_abs_out"]).exists())
            for _, j in batch
        )
        own_holistic = False
        if need_holistic and _WORKER_HOLISTIC is None:
            _WORKER_HOLISTIC = create_holistic(**hk)
            own_holistic = True
        try:
            for idx, job in batch:
                results[idx] = _process_job(job)
                if progress:
                    progress.update(1)
        finally:
            if own_holistic:
                _close_worker_holistic()
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(hk,),
        ) as ex:
            futs = {ex.submit(_process_job, job): (idx, job) for idx, job in batch}
            for fut in as_completed(futs):
                idx, job = futs[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:  # worker crash
                    row = _index_row(job)
                    row["status"] = "error"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    results[idx] = row
                if progress:
                    progress.update(1)


def extract_manifest(
    manifest_csv: str | Path,
    output_dir: str | Path = "data/landmarks",
    root_dir: str | Path = ".",
    overwrite: bool = False,
    num_workers: int = 1,
    limit: int | None = None,
    *,
    holistic_kwargs: Mapping[str, Any] | None = None,
    derive_augmented: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Extract landmarks for every row of a Stage-1 manifest CSV.

    Writes ``<output_dir>/<split>/<class_id>/<sample_id>.npz`` and, per split
    present in the manifest, ``<output_dir>/<split>_landmarks_index.csv``
    (``landmark_path`` is relative to ``output_dir``, posix). When the input
    manifest is ``train_augmented.csv``, also writes
    ``<output_dir>/train_augmented_landmarks_index.csv``.
    """
    root = Path(root_dir).resolve()
    manifest_csv = _resolve(manifest_csv, root)
    out_root = _resolve(output_dir, root)
    df = pd.read_csv(manifest_csv, encoding="utf-8")
    required = {"sample_id", "video_path", "label", "class_id", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{manifest_csv}: missing columns {sorted(missing)}")
    if limit is not None:
        df = df.head(int(limit))

    hk = dict(DEFAULT_HOLISTIC)
    hk.update({k: v for k, v in (holistic_kwargs or {}).items() if v is not None or k in ("max_frames", "min_side", "max_side")})

    out_path_by_sid: dict[str, str] = {}
    for rec in df.to_dict("records"):
        split = _str(rec.get("split")) or "unknown"
        sid = _str(rec.get("sample_id"))
        rel = landmark_rel_path(split, rec.get("class_id"), sid)
        out_path_by_sid[sid] = str(out_root / rel)

    jobs: list[dict[str, Any]] = []
    for rec in df.to_dict("records"):
        split = _str(rec.get("split")) or "unknown"
        sid = _str(rec.get("sample_id"))
        pid = _str(rec.get("parent_sample_id"))
        is_aug = _bool(rec.get("is_augmented", False))
        rel = landmark_rel_path(split, rec.get("class_id"), sid)
        parent_abs_out = out_path_by_sid.get(pid, "") if (is_aug and pid) else ""
        if not parent_abs_out and is_aug and pid:
            cand = out_root / landmark_rel_path(split, rec.get("class_id"), pid)
            if cand.exists():
                parent_abs_out = str(cand)
        jobs.append(
            {
                "sample_id": sid,
                "parent_sample_id": pid,
                "landmark_path": rel,
                "video_path": _str(rec.get("video_path")),
                "label": _str(rec.get("label")),
                "class_id": _int(rec.get("class_id")),
                "signer_id": _str(rec.get("signer_id")),
                "split": split,
                "is_augmented": is_aug,
                "expected_num_frames": _int(rec.get("num_frames"), default=-1),
                "abs_video": str(_resolve(_str(rec.get("video_path")), root)),
                "abs_out": str(out_root / rel),
                "parent_abs_out": parent_abs_out,
                "derive_augmented": bool(derive_augmented and is_aug and pid),
                "overwrite": bool(overwrite),
                "holistic": hk,
            }
        )

    t0 = time.time()
    progress = None
    if show_progress:
        try:
            from tqdm import tqdm

            progress = tqdm(total=len(jobs), desc=f"landmarks:{manifest_csv.name}", unit="video")
        except ImportError:
            progress = None

    workers = max(1, int(num_workers or 1))
    results: dict[int, dict[str, Any]] = {}

    primary_batch = [(i, j) for i, j in enumerate(jobs) if not j["derive_augmented"]]
    derived_batch = [(i, j) for i, j in enumerate(jobs) if j["derive_augmented"]]

    _run_job_batch(
        primary_batch,
        results,
        workers=workers,
        hk=hk,
        progress=progress,
        use_holistic_pool=True,
    )
    _run_job_batch(
        derived_batch,
        results,
        workers=1,
        hk=hk,
        progress=progress,
        use_holistic_pool=False,
    )

    if progress:
        progress.close()

    rows = [results[i] for i in range(len(jobs))]
    index = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    out_root.mkdir(parents=True, exist_ok=True)
    for split, part in index.groupby("split", sort=False):
        idx_path = out_root / f"{split}_landmarks_index.csv"
        part.to_csv(idx_path, index=False, encoding="utf-8")
        LOGGER.info("wrote %s (%d rows)", idx_path, len(part))
        if split == "train" and manifest_csv.stem == "train_augmented":
            aug_idx_path = out_root / "train_augmented_landmarks_index.csv"
            part.to_csv(aug_idx_path, index=False, encoding="utf-8")
            LOGGER.info("wrote %s (%d rows)", aug_idx_path, len(part))

    counts = index["status"].value_counts().to_dict()
    LOGGER.info(
        "%s: %d videos in %.1fs -> %s", manifest_csv.name, len(index), time.time() - t0, counts
    )
    return index


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_manifests(cfg: Mapping[str, Any], splits: Iterable[str], root: Path) -> list[Path]:
    """Map split names to Stage-1 manifest paths via ``configs/data_config.yaml``."""
    data_cfg_path = _resolve(cfg.get("data_config", "configs/data_config.yaml"), root)
    data_paths = _load_yaml(data_cfg_path).get("paths", {}) if data_cfg_path.exists() else {}
    mapping = cfg.get("manifests") or {
        "train": "train_augmented_manifest",
        "val": "val_manifest",
        "test": "test_manifest",
    }
    defaults = {
        "train_augmented_manifest": "data/splits/train_augmented.csv",
        "val_manifest": "data/splits/val.csv",
        "test_manifest": "data/splits/test.csv",
        "train_manifest": "data/splits/train.csv",
    }
    out = []
    for s in splits:
        key = mapping.get(s, f"{s}_manifest")
        out.append(_resolve(data_paths.get(key, defaults.get(key, f"data/splits/{s}.csv")), root))
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract MediaPipe Holistic landmarks (Stage 2.1)")
    p.add_argument("--config", default="configs/preprocess_config.yaml")
    p.add_argument("--manifest", nargs="+", default=None, help="explicit manifest CSV(s)")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--root-dir", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="first N rows per manifest")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=None)
    p.add_argument("--no-derive-augmented", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    cfg = _load_yaml(args.config) if Path(args.config).exists() else {}
    paths = cfg.get("paths", {}) or {}
    ext = cfg.get("extraction", {}) or {}
    root = Path(args.root_dir or paths.get("root_dir", ".")).resolve()
    output_dir = args.output_dir or paths.get("landmarks_dir", "data/landmarks")

    hol = dict(DEFAULT_HOLISTIC)
    hol.update({k: v for k, v in (cfg.get("holistic", {}) or {}).items() if k in DEFAULT_HOLISTIC})
    max_frames = args.max_frames if args.max_frames is not None else (cfg.get("holistic", {}) or {}).get("max_frames")
    hol["max_frames"] = max_frames
    if args.model_complexity is not None:
        hol["model_complexity"] = args.model_complexity

    manifests = (
        [_resolve(m, root) for m in args.manifest]
        if args.manifest
        else resolve_manifests(cfg, args.splits, root)
    )
    workers = args.workers if args.workers is not None else int(ext.get("num_workers", 10) or 10)
    overwrite = args.overwrite or bool(ext.get("overwrite", False))
    limit = args.limit if args.limit is not None else ext.get("limit")
    derive_aug = not args.no_derive_augmented and bool(ext.get("derive_augmented", True))

    n_err = 0
    for m in manifests:
        if not m.exists():
            LOGGER.error("manifest not found: %s", m)
            n_err += 1
            continue
        index = extract_manifest(
            m,
            output_dir=output_dir,
            root_dir=root,
            overwrite=overwrite,
            num_workers=workers,
            limit=limit,
            holistic_kwargs=hol,
            derive_augmented=derive_aug,
            show_progress=not args.no_progress,
        )
        errs = index[index["status"] == "error"]
        for _, r in errs.head(5).iterrows():
            LOGGER.warning("error %s: %s", r["sample_id"], r["error"])
        print(
            f"[extract_landmarks] {m.name}: {len(index)} rows, "
            f"status={index['status'].value_counts().to_dict()}, "
            f"mean hand_detect_ratio={index['hand_detect_ratio'].mean() if len(index) else 0:.4f}, "
            f"mean pose_detect_ratio={index['pose_detect_ratio'].mean() if len(index) else 0:.4f}, "
            f"mean face_detect_ratio={index['face_detect_ratio'].mean() if len(index) else 0:.4f}",
            flush=True,
        )
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())

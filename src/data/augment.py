"""Step 4: Train-only sign-language-safe video augmentation (Tăng cường - chỉ trên tập train).

HARD SAFETY INVARIANT:
    Augmentation is strictly restricted to rows with ``split == "train"``.
    Passing ANY manifest containing ``val``, ``test``, or non-``train`` rows raises
    ``ValueError("Augmentation is strictly allowed on train split only!")``.
    ``val.csv`` and ``test.csv`` are NEVER read or modified by this module.

Sign-Language-Safe Transformations (deterministic via seed):
    1. Temporal speed perturbation (0.85x - 1.15x via frame resampling)
    2. Temporal shift / slight start-end trim
    3. Safe spatial affine transform (rotation +-8 deg, scale 0.92-1.08, translation +-4%,
       using BORDER_REPLICATE so hands and upper body remain cleanly framed)
    4. Photometric lighting jitter (contrast/brightness shift + mild Gaussian noise)
    5. Optional horizontal flip (``allow_hflip=False`` by default) — WARNING: horizontal
       flipping mirrors left and right hands, which swaps the dominant hand and movement
       direction in asymmetric Vietnamese Sign Language (VSL) signs. Keep disabled unless
       every target class is verified mirror-symmetric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.data.schema import (
    KAGGLE_SOURCES,
    MANIFEST_COLUMNS,
    is_kaggle_source,
    load_class_list,
    load_config,
    load_manifest_df,
    normalize_rel_path,
    probe_video_metadata,
    save_manifest_df,
    slugify_label,
    validate_manifest_df,
)

logger = logging.getLogger(__name__)

TRAIN_ONLY_ERROR_MSG = "Augmentation is strictly allowed on train split only!"
KAGGLE_ONLY_DEFAULT_NUM_AUG = 4


@dataclass
class AugmentConfig:
    """Configuration parameters for sign-language-safe train-only video augmentation."""

    num_augmentations_per_video: int = 2
    random_seed: int = 42
    speed_range: tuple[float, float] = (0.85, 1.15)
    trim_max_fraction: float = 0.08
    rotation_deg_range: tuple[float, float] = (-8.0, 8.0)
    scale_range: tuple[float, float] = (0.92, 1.08)
    translation_frac_range: tuple[float, float] = (-0.04, 0.04)
    contrast_range: tuple[float, float] = (0.85, 1.15)
    brightness_range: tuple[float, float] = (-20.0, 20.0)
    noise_std_range: tuple[float, float] = (0.0, 5.0)
    allow_hflip: bool = False
    hflip_prob: float = 0.5


def ensure_train_only_manifest(df: pd.DataFrame) -> None:
    """Hard guard ensuring every single row in ``df`` belongs exclusively to ``train``.

    Raises:
        ValueError: With message ``"Augmentation is strictly allowed on train split only!"``
            whenever the ``split`` column is missing or ANY row has ``split != "train"``.
    """
    if "split" not in df.columns:
        raise ValueError(TRAIN_ONLY_ERROR_MSG)
    if df.empty:
        return

    splits = df["split"].fillna("").astype(str).str.strip().str.lower()
    if not (splits == "train").all():
        bad_splits = sorted(splits.loc[splits != "train"].unique().tolist())
        logger.error("Rejected augmentation input containing non-train splits: %s", bad_splits)
        raise ValueError(TRAIN_ONLY_ERROR_MSG)


def _deterministic_rng(seed: int, sample_id: str, aug_index: int) -> np.random.Generator:
    """Build a deterministic per-sample, per-copy NumPy Generator independent of PYTHONHASHSEED."""
    key = f"{seed}:{sample_id}:{aug_index}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    seed_int = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(seed_int)


def read_video_frames(video_path: str | Path) -> tuple[list[np.ndarray], float]:
    """Read all BGR frames and FPS from a video file using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video for augmentation: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1.0 or fps > 240.0:
        fps = 30.0

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"Video contains zero readable frames: {video_path}")

    return frames, fps


def write_video_frames(
    frames: list[np.ndarray],
    output_path: str | Path,
    fps: float = 30.0,
) -> Path:
    """Write BGR frames to an ``.mp4`` file safely using OpenCV."""
    if not frames:
        raise ValueError("Cannot write empty frame list to video.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    safe_fps = float(fps) if 1.0 <= float(fps) <= 240.0 else 30.0

    fourcc_candidates = ["mp4v", "avc1", "XVID"]
    writer: cv2.VideoWriter | None = None
    for codec in fourcc_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        cand_writer = cv2.VideoWriter(str(out_path), fourcc, safe_fps, (width, height))
        if cand_writer.isOpened():
            writer = cand_writer
            break
        cand_writer.release()

    if writer is None:
        raise IOError(f"Could not initialize OpenCV VideoWriter for {out_path}")

    try:
        for frame in frames:
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    return out_path


def augment_temporal_speed(
    frames: list[np.ndarray],
    speed_factor: float,
) -> list[np.ndarray]:
    """Perturb signing speed by resampling frame indices (e.g. 0.85x slower to 1.15x faster)."""
    num_frames = len(frames)
    if num_frames <= 2 or abs(speed_factor - 1.0) < 1e-4:
        return list(frames)

    target_len = max(4, int(round(num_frames / float(speed_factor))))
    indices = np.linspace(0, num_frames - 1, num=target_len)
    sampled = [frames[int(round(idx))] for idx in indices]
    return sampled


def augment_temporal_trim(
    frames: list[np.ndarray],
    start_trim_frac: float = 0.0,
    end_trim_frac: float = 0.0,
) -> list[np.ndarray]:
    """Apply a slight start/end temporal trim while preserving the core sign motion."""
    num_frames = len(frames)
    if num_frames <= 6:
        return list(frames)

    start_idx = int(round(num_frames * max(0.0, start_trim_frac)))
    end_cut = int(round(num_frames * max(0.0, end_trim_frac)))
    end_idx = num_frames - end_cut

    # Guarantee at least 4 frames remain
    if end_idx - start_idx < 4:
        return list(frames)
    return frames[start_idx:end_idx]


def augment_spatial_affine(
    frames: list[np.ndarray],
    angle_deg: float,
    scale: float,
    tx_frac: float,
    ty_frac: float,
) -> list[np.ndarray]:
    """Apply one consistent small affine transform across every frame in a clip.

    Uses ``BORDER_REPLICATE`` so small rotations/scales/shifts keep the signer's
    hands, face, and upper body cleanly visible without harsh black borders.
    """
    if not frames:
        return []

    height, width = frames[0].shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), float(scale))
    matrix[0, 2] += float(tx_frac) * width
    matrix[1, 2] += float(ty_frac) * height

    return [
        cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        for frame in frames
    ]


def augment_lighting_jitter(
    frames: list[np.ndarray],
    contrast: float,
    brightness: float,
    noise_std: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Apply clip-consistent brightness/contrast jitter plus mild Gaussian sensor noise."""
    lut = np.clip(
        np.arange(256, dtype=np.float32) * float(contrast) + float(brightness),
        0.0,
        255.0,
    ).astype(np.uint8)
    std = float(noise_std)

    out: list[np.ndarray] = []
    for frame in frames:
        adjusted = cv2.LUT(frame, lut)
        if std > 0.0:
            h, w = frame.shape[:2]
            channels = frame.shape[2] if frame.ndim == 3 else 1
            if h * w > 320 * 240:
                nh, nw = min(h, 180), min(w, 320)
                noise_small = (
                    rng.standard_normal((nh, nw, channels), dtype=np.float32) * np.float32(std)
                    if frame.ndim == 3
                    else rng.standard_normal((nh, nw), dtype=np.float32) * np.float32(std)
                )
                noise_full = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                noise_full = (
                    rng.standard_normal(frame.shape, dtype=np.float32) * np.float32(std)
                )
            noisy = adjusted.astype(np.int16) + np.rint(noise_full).astype(np.int16)
            adjusted = np.clip(noisy, 0, 255).astype(np.uint8)
        out.append(adjusted)
    return out


def augment_horizontal_flip(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Horizontally mirror every frame in the clip.

    WARNING:
        Horizontal flipping swaps left and right hands! In asymmetric Vietnamese Sign
        Language (VSL) signs, mirroring swaps the dominant hand and reverses directional
        trajectories, which can alter sign semantics or confuse MediaPipe Holistic
        left/right hand features. Only enable ``allow_hflip=True`` when all target
        classes are confirmed symmetric or explicitly handled downstream.
    """
    return [cv2.flip(frame, 1) for frame in frames]


def apply_video_augmentations(
    frames: list[np.ndarray],
    rng: np.random.Generator,
    aug_cfg: AugmentConfig | None = None,
    allow_hflip: bool | None = None,
) -> tuple[list[np.ndarray], str]:
    """Apply a deterministic sign-language-safe combination of augmentations to frames.

    Returns:
        Tuple of ``(augmented_frames, aug_type_description)``.
    """
    cfg = aug_cfg or AugmentConfig()
    use_hflip = cfg.allow_hflip if allow_hflip is None else bool(allow_hflip)

    current = [f.copy() for f in frames]
    applied_ops: list[str] = []

    # 1. Temporal speed perturbation (avoid near-1.0 no-op)
    low_s, high_s = cfg.speed_range
    speed = float(rng.uniform(low_s, high_s))
    if abs(speed - 1.0) < 0.03:
        speed = high_s if speed >= 1.0 else low_s
    current = augment_temporal_speed(current, speed)
    applied_ops.append(f"speed({speed:.2f}x)")

    # 2. Temporal shift / slight start-end trim
    if cfg.trim_max_fraction > 0 and len(current) > 8:
        start_trim = float(rng.uniform(0.0, cfg.trim_max_fraction))
        end_trim = float(rng.uniform(0.0, cfg.trim_max_fraction))
        if start_trim > 0.01 or end_trim > 0.01:
            current = augment_temporal_trim(current, start_trim, end_trim)
            applied_ops.append(f"trim({start_trim:.2f},{end_trim:.2f})")

    # 3. Safe spatial affine transform (consistent across all frames)
    angle = float(rng.uniform(*cfg.rotation_deg_range))
    scale = float(rng.uniform(*cfg.scale_range))
    tx = float(rng.uniform(*cfg.translation_frac_range))
    ty = float(rng.uniform(*cfg.translation_frac_range))
    current = augment_spatial_affine(current, angle, scale, tx, ty)
    applied_ops.append(f"affine(r={angle:+.1f},s={scale:.2f},tx={tx:+.2f},ty={ty:+.2f})")

    # 4. Photometric lighting jitter + mild Gaussian noise
    contrast = float(rng.uniform(*cfg.contrast_range))
    brightness = float(rng.uniform(*cfg.brightness_range))
    noise_std = float(rng.uniform(*cfg.noise_std_range))
    current = augment_lighting_jitter(current, contrast, brightness, noise_std, rng)
    applied_ops.append(f"light(c={contrast:.2f},b={brightness:+.0f},n={noise_std:.1f})")

    # 5. Optional horizontal flip (STRICTLY OFF by default for VSL safety)
    if use_hflip and float(rng.random()) < cfg.hflip_prob:
        current = augment_horizontal_flip(current)
        applied_ops.append("hflip")

    return current, "+".join(applied_ops)


def _extract_augment_config(config: dict[str, Any] | None) -> tuple[AugmentConfig, dict[str, str]]:
    """Extract augmentation parameters and default paths from ``configs/data_config.yaml``."""
    cfg = config or {}
    aug_section = cfg.get("augmentation", cfg.get("augment", {})) or {}
    paths_section = cfg.get("paths", {}) or {}

    def _pair(keys: tuple[str, ...], default: tuple[float, float]) -> tuple[float, float]:
        for key in keys:
            val = aug_section.get(key)
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return (float(val[0]), float(val[1]))
        return default

    rot_val = aug_section.get("rotation_deg_range", aug_section.get("max_rotation_deg"))
    if isinstance(rot_val, (list, tuple)) and len(rot_val) == 2:
        rotation_range = (float(rot_val[0]), float(rot_val[1]))
    elif isinstance(rot_val, (int, float)):
        rotation_range = (-abs(float(rot_val)), abs(float(rot_val)))
    else:
        rotation_range = (-8.0, 8.0)

    trans_val = aug_section.get(
        "translate_ratio_range",
        aug_section.get("translation_frac_range", aug_section.get("max_translation_frac")),
    )
    if isinstance(trans_val, (list, tuple)) and len(trans_val) == 2:
        translation_range = (float(trans_val[0]), float(trans_val[1]))
    elif isinstance(trans_val, (int, float)):
        translation_range = (-abs(float(trans_val)), abs(float(trans_val)))
    else:
        translation_range = (-0.04, 0.04)

    noise_val = aug_section.get("noise_std_range", aug_section.get("gaussian_noise_std"))
    if isinstance(noise_val, (list, tuple)) and len(noise_val) == 2:
        noise_std_range = (float(noise_val[0]), float(noise_val[1]))
    elif isinstance(noise_val, (int, float)):
        noise_std_range = (0.0, abs(float(noise_val)))
    else:
        noise_std_range = (0.0, 4.0)

    aug_cfg = AugmentConfig(
        num_augmentations_per_video=int(
            aug_section.get(
                "num_augments_per_sample",
                aug_section.get(
                    "num_augmentations_per_video",
                    aug_section.get("copies_per_video", aug_section.get("num_copies", 2)),
                ),
            )
        ),
        random_seed=int(
            aug_section.get("random_seed", aug_section.get("seed", cfg.get("random_seed", 42)))
        ),
        speed_range=_pair(("speed_range",), (0.85, 1.15)),
        trim_max_fraction=float(
            aug_section.get(
                "temporal_trim_max_ratio",
                aug_section.get("trim_max_fraction", 0.08),
            )
        ),
        rotation_deg_range=rotation_range,
        scale_range=_pair(("scale_range",), (0.92, 1.08)),
        translation_frac_range=translation_range,
        contrast_range=_pair(("contrast_range",), (0.88, 1.12)),
        brightness_range=_pair(("brightness_delta_range", "brightness_range"), (-20.0, 20.0)),
        noise_std_range=noise_std_range,
        allow_hflip=bool(aug_section.get("allow_hflip", False)),
        hflip_prob=float(aug_section.get("hflip_prob", 0.5)),
    )

    splits_dir = str(paths_section.get("splits_dir") or "data/splits")
    default_paths = {
        "train_manifest": str(
            paths_section.get("train_manifest") or Path(splits_dir) / "train.csv"
        ),
        "output_manifest": str(
            paths_section.get("train_augmented_manifest")
            or Path(splits_dir) / "train_augmented.csv"
        ),
        "output_video_dir": str(
            paths_section.get("augmented_train_dir")
            or paths_section.get("augmented_dir")
            or "data/augmented/train"
        ),
        "class_list": str(
            paths_section.get("class_list_path")
            or paths_section.get("class_list")
            or "configs/class_list.json"
        ),
    }
    return aug_cfg, default_paths


def _build_slug_lookup(class_list_path: str | Path | None) -> dict[str, str]:
    """Load label -> slug mapping from ``configs/class_list.json`` if available."""
    if not class_list_path or not Path(class_list_path).exists():
        return {}
    try:
        entries = load_class_list(class_list_path)
    except Exception:
        return {}
    lookup: dict[str, str] = {}
    for item in entries:
        label = str(item.get("label", "")).strip()
        slug = str(item.get("slug", "")).strip()
        if label and slug:
            lookup[label] = slug
    return lookup


def _safe_rel_path(path: Path, project_root: Path) -> str:
    """Normalize path relative to ``project_root`` using schema helper when possible."""
    try:
        return normalize_rel_path(path, project_root)
    except Exception:
        return path.resolve().as_posix()


def augment_train_manifest(
    train_manifest_path: str | Path,
    output_manifest_path: str | Path = "data/splits/train_augmented.csv",
    output_video_dir: str | Path = "data/augmented/train",
    num_augmentations_per_video: int | None = None,
    random_seed: int | None = None,
    allow_hflip: bool | None = None,
    config_path: str | Path | None = None,
    class_list_path: str | Path | None = None,
    project_root: str | Path | None = None,
    aug_config: AugmentConfig | None = None,
    kaggle_only: bool = False,
) -> dict[str, Any]:
    """Generate train-only augmented videos and write ``train_augmented.csv``.

    Args:
        train_manifest_path: Path to ``train.csv`` (MUST contain only ``split == "train"``).
        output_manifest_path: Destination CSV for original + augmented train rows.
        output_video_dir: Root directory where augmented ``.mp4`` clips are saved under
            ``<output_video_dir>/<class_slug>/<sample_id>_aug<k>.mp4``.
        num_augmentations_per_video: Number of augmented variants per original train clip.
            When ``kaggle_only=True`` and this is omitted, defaults to 4 (or config's
            ``augmentation.kaggle_only_num_augments``) to compensate for 1-2 raw clips/class.
        random_seed: Deterministic seed for reproducibility.
        allow_hflip: Whether horizontal flip is allowed (defaults to ``False``).
        config_path: Optional YAML config path.
        class_list_path: Optional ``class_list.json`` path for canonical class slugs.
        project_root: Base directory for resolving relative ``video_path`` entries.
        aug_config: Optional explicit ``AugmentConfig`` instance.
        kaggle_only: If True, verify all rows originate from supported Kaggle sources
            (``kaggle_vsl``, ``kaggle_vsl_v2``) and use higher default augmentation
            multiplier (4x) when ``num_augmentations_per_video`` is not explicitly set.

    Returns:
        Summary dictionary of the augmentation run (also written to ``augment_report.json``).
    """
    cfg_dict = load_config(config_path) if config_path and Path(config_path).exists() else {}
    extracted_cfg, default_paths = _extract_augment_config(cfg_dict)
    eff_cfg = aug_config if aug_config is not None else extracted_cfg

    if num_augmentations_per_video is not None:
        eff_cfg.num_augmentations_per_video = int(num_augmentations_per_video)
    elif kaggle_only and aug_config is None:
        aug_section = (cfg_dict or {}).get("augmentation", {}) or {}
        eff_cfg.num_augmentations_per_video = int(
            aug_section.get(
                "kaggle_only_num_augments_per_sample",
                aug_section.get("kaggle_only_num_augments", KAGGLE_ONLY_DEFAULT_NUM_AUG),
            )
        )
    if random_seed is not None:
        eff_cfg.random_seed = int(random_seed)
    if allow_hflip is not None:
        eff_cfg.allow_hflip = bool(allow_hflip)

    if eff_cfg.num_augmentations_per_video < 0:
        raise ValueError("num_augmentations_per_video must be >= 0.")

    train_csv = Path(train_manifest_path)
    if not train_csv.exists():
        raise FileNotFoundError(f"Train manifest not found: {train_csv}")

    df = load_manifest_df(train_csv)

    # HARD GUARD: Refuse ANY non-train rows immediately before touching disk
    ensure_train_only_manifest(df)

    missing_cols = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Train manifest missing required columns: {missing_cols}")

    if kaggle_only and len(df) > 0:
        non_kaggle = df.loc[~df["source"].map(is_kaggle_source)]
        if len(non_kaggle) > 0:
            bad_sources = sorted(non_kaggle["source"].astype(str).unique().tolist())
            raise ValueError(
                f"kaggle_only mode requires source in {KAGGLE_SOURCES} for all train rows, "
                f"found {len(non_kaggle)} row(s) with source={bad_sources}."
            )

    root_dir = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    out_vid_root = Path(output_video_dir)
    out_vid_root.mkdir(parents=True, exist_ok=True)
    out_csv = Path(output_manifest_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    slug_lookup = _build_slug_lookup(class_list_path or default_paths["class_list"])

    # Normalize original train rows
    orig_df = df[MANIFEST_COLUMNS].copy()
    if len(orig_df) > 0:
        orig_df["split"] = "train"
        orig_df["is_augmented"] = False
        orig_df["parent_sample_id"] = orig_df["parent_sample_id"].fillna("").astype(str)
        orig_df["aug_type"] = orig_df["aug_type"].fillna("").astype(str)

    augmented_rows: list[dict[str, Any]] = []
    skipped_invalid = 0
    failed_videos: list[str] = []

    for _, row in orig_df.iterrows():
        is_valid_val = row["is_valid"]
        if isinstance(is_valid_val, str):
            is_valid_bool = is_valid_val.strip().lower() in {"true", "1", "yes"}
        else:
            is_valid_bool = bool(is_valid_val)
        if not is_valid_bool:
            skipped_invalid += 1
            continue

        sample_id = str(row["sample_id"]).strip()
        label = str(row["label"]).strip()
        class_slug = slug_lookup.get(label) or slugify_label(label)

        raw_vpath = Path(str(row["video_path"]))
        src_video_path = raw_vpath if raw_vpath.is_absolute() else (root_dir / raw_vpath)
        if not src_video_path.exists():
            # Fallback relative to train_manifest directory if needed
            alt_path = train_csv.parent / raw_vpath
            if alt_path.exists():
                src_video_path = alt_path
            else:
                failed_videos.append(str(raw_vpath))
                logger.warning("Skipping missing source train video: %s", src_video_path)
                continue

        frames, fps = read_video_frames(src_video_path)

        for k in range(1, eff_cfg.num_augmentations_per_video + 1):
            rng = _deterministic_rng(eff_cfg.random_seed, sample_id, k)
            aug_frames, aug_desc = apply_video_augmentations(
                frames=frames,
                rng=rng,
                aug_cfg=eff_cfg,
                allow_hflip=eff_cfg.allow_hflip,
            )

            aug_sample_id = f"{sample_id}_aug{k}"
            out_video_path = out_vid_root / class_slug / f"{aug_sample_id}.mp4"
            write_video_frames(aug_frames, out_video_path, fps=fps)

            meta = probe_video_metadata(out_video_path)
            if not meta.get("is_valid", False):
                failed_videos.append(str(out_video_path))
                continue

            augmented_rows.append(
                {
                    "sample_id": aug_sample_id,
                    "video_path": _safe_rel_path(out_video_path, root_dir),
                    "label": label,
                    "class_id": int(row["class_id"]),
                    "signer_id": str(row["signer_id"]).strip(),
                    "source": str(row["source"]).strip(),
                    "split": "train",
                    "is_augmented": True,
                    "parent_sample_id": sample_id,
                    "aug_type": aug_desc,
                    "fps": float(meta["fps"]),
                    "num_frames": int(meta["num_frames"]),
                    "duration_sec": float(meta["duration_sec"]),
                    "width": int(meta["width"]),
                    "height": int(meta["height"]),
                    "is_valid": bool(meta["is_valid"]),
                }
            )

    aug_df = (
        pd.DataFrame(augmented_rows, columns=MANIFEST_COLUMNS)
        if augmented_rows
        else pd.DataFrame(columns=MANIFEST_COLUMNS)
    )
    combined_df = pd.concat([orig_df, aug_df], ignore_index=True)[MANIFEST_COLUMNS]

    # Re-verify hard guard & schema on final output
    ensure_train_only_manifest(combined_df)
    if len(combined_df) > 0:
        validate_manifest_df(combined_df, require_split=True)

    save_manifest_df(
        combined_df,
        out_csv,
        validate=len(combined_df) > 0,
        require_split=True,
    )

    report: dict[str, Any] = {
        "mode": "kaggle_only" if kaggle_only else "standard",
        "input_train_manifest": str(train_csv),
        "output_manifest": str(out_csv),
        "output_video_dir": str(out_vid_root),
        "random_seed": eff_cfg.random_seed,
        "allow_hflip": eff_cfg.allow_hflip,
        "num_augmentations_per_video": eff_cfg.num_augmentations_per_video,
        "original_train_samples": int(len(orig_df)),
        "generated_augmented_samples": int(len(aug_df)),
        "total_train_augmented_samples": int(len(combined_df)),
        "skipped_invalid_samples": skipped_invalid,
        "failed_videos": failed_videos,
        "train_only_guard_passed": True,
    }

    report_path = out_csv.parent / "augment_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report["report_path"] = str(report_path)

    logger.info(
        "Train-only augmentation completed: orig=%d, augmented=%d, total=%d (allow_hflip=%s)",
        len(orig_df),
        len(aug_df),
        len(combined_df),
        eff_cfg.allow_hflip,
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4: Train-only sign-language-safe video augmentation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data_config.yaml (default: configs/data_config.yaml).",
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default=None,
        help="Path to train.csv (STRICTLY split=='train' only).",
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        default=None,
        help="Output CSV path (default: data/splits/train_augmented.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for augmented .mp4 videos (default: data/augmented/train).",
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=None,
        help="Number of augmented copies per original train video.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic augmentation.",
    )
    parser.add_argument(
        "--allow-hflip",
        action="store_true",
        default=False,
        help=(
            "Enable horizontal flip (DEFAULT: False). WARNING: flipping swaps left/right "
            "dominant hand in asymmetric VSL signs."
        ),
    )
    parser.add_argument(
        "--kaggle-only",
        action="store_true",
        default=False,
        help=(
            "Kaggle-only mode: verify all train rows come from Kaggle sources "
            "(kaggle_vsl / kaggle_vsl_v2) and default to 4 augmented copies per clip "
            "unless --num-aug is explicitly provided."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv)

    cfg_path = Path(args.config) if args.config else None
    cfg_dict = load_config(cfg_path) if cfg_path and cfg_path.exists() else {}
    _, default_paths = _extract_augment_config(cfg_dict)

    train_manifest = args.train_manifest or default_paths["train_manifest"]
    output_manifest = args.output_manifest or default_paths["output_manifest"]
    output_dir = args.output_dir or default_paths["output_video_dir"]

    report = augment_train_manifest(
        train_manifest_path=train_manifest,
        output_manifest_path=output_manifest,
        output_video_dir=output_dir,
        num_augmentations_per_video=args.num_aug,
        random_seed=args.seed,
        allow_hflip=True if args.allow_hflip else None,
        config_path=cfg_path,
        class_list_path=default_paths["class_list"],
        kaggle_only=args.kaggle_only,
    )
    print(
        f"[augment] Saved {report['generated_augmented_samples']} augmented videos -> "
        f"{report['output_manifest']} (total train rows: {report['total_train_augmented_samples']}, "
        f"allow_hflip={report['allow_hflip']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

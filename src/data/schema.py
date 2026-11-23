"""Shared data schema, configuration loaders, and video metadata utilities.

This module defines the canonical contract for Quy's VSL Data Pipeline (Part 1: Data)
and the handoff schema to Step 2 (MediaPipe Holistic preprocessing).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import yaml

MANIFEST_COLUMNS: list[str] = [
    "sample_id",
    "video_path",
    "label",
    "class_id",
    "signer_id",
    "source",
    "split",
    "is_augmented",
    "parent_sample_id",
    "aug_type",
    "fps",
    "num_frames",
    "duration_sec",
    "width",
    "height",
    "is_valid",
]

VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")
KAGGLE_SOURCES: tuple[str, ...] = ("kaggle_vsl", "kaggle_vsl_v2")
VALID_SOURCES: tuple[str, ...] = ("kaggle_vsl", "kaggle_vsl_v2", "self_recorded")

_VIETNAMESE_CHAR_MAP = str.maketrans(
    {
        "đ": "d",
        "Đ": "D",
    }
)


def is_kaggle_source(source: Any) -> bool:
    """Return True if ``source`` identifies one of the supported Kaggle VSL datasets."""
    return str(source).strip() in KAGGLE_SOURCES


def normalize_vietnamese_label(label: str) -> str:
    """Normalize a Vietnamese label using Unicode NFC and collapsed whitespace."""
    return " ".join(unicodedata.normalize("NFC", str(label)).strip().split())


def slugify_label(label: str) -> str:
    """Convert a Vietnamese label into a deterministic ASCII snake_case slug."""
    text = normalize_vietnamese_label(label).translate(_VIETNAMESE_CHAR_MAP)
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    ascii_text = ascii_text.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    return slug or "unknown"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and return a YAML configuration file as a dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}, got {type(data).__name__}")
    return data


def load_class_list(path: str | Path) -> list[dict[str, Any]]:
    """Load the curated class scope definition from JSON.

    Supports either a top-level list of class objects or a JSON object containing
    a ``"classes"`` list. Each item is validated to have ``class_id``, ``label``,
    ``slug``, and ``category``.
    """
    class_path = Path(path)
    if not class_path.exists():
        raise FileNotFoundError(f"Class list file not found: {class_path}")
    with class_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        classes = raw.get("classes")
        if not isinstance(classes, list):
            raise ValueError(f"Expected 'classes' list in {class_path}")
    elif isinstance(raw, list):
        classes = raw
    else:
        raise ValueError(f"Unsupported class_list.json structure in {class_path}")

    required_keys = ("class_id", "label", "slug", "category")
    seen_ids: set[int] = set()
    validated: list[dict[str, Any]] = []
    for idx, item in enumerate(classes):
        if not isinstance(item, dict):
            raise ValueError(f"Class entry at index {idx} must be a dict")
        missing = [k for k in required_keys if k not in item]
        if missing:
            raise ValueError(f"Class entry at index {idx} missing keys: {missing}")
        cid = int(item["class_id"])
        if cid in seen_ids:
            raise ValueError(f"Duplicate class_id detected in {class_path}: {cid}")
        seen_ids.add(cid)
        entry = dict(item)
        entry["class_id"] = cid
        entry["label"] = str(item["label"]).strip()
        entry["slug"] = str(item["slug"]).strip() or slugify_label(entry["label"])
        entry["category"] = str(item["category"]).strip()
        validated.append(entry)

    return validated


def _is_missing_text(val: Any) -> bool:
    if val is None:
        return True
    try:
        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    return str(val).strip() == ""


def _coerce_bool_value(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)) and not pd.isna(val):
        if val in (0, 1):
            return bool(int(val))
    if isinstance(val, str):
        low = val.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
    raise ValueError(f"Invalid boolean value: {val!r}")


def validate_manifest_df(df: pd.DataFrame, *, require_split: bool = False) -> None:
    """Validate that a manifest DataFrame conforms to the shared schema contract.

    Args:
        df: Manifest DataFrame to validate.
        require_split: If True, every row must have a non-empty split in VALID_SPLITS.

    Raises:
        ValueError: If columns are missing or field invariants are violated.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas.DataFrame, got {type(df).__name__}")

    missing_cols = [col for col in MANIFEST_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Manifest DataFrame missing required columns: {missing_cols}")

    if df.empty:
        return

    # Check required non-empty text fields
    for col in ("sample_id", "video_path", "label", "signer_id", "source"):
        bad_mask = df[col].map(_is_missing_text)
        if bad_mask.any():
            bad_indices = df.index[bad_mask].tolist()[:5]
            raise ValueError(f"Column '{col}' contains null or empty values at rows: {bad_indices}")

    # Check unique sample_id
    sample_ids = df["sample_id"].astype(str).str.strip()
    dup_mask = sample_ids.duplicated(keep=False)
    if dup_mask.any():
        dups = sample_ids[dup_mask].unique().tolist()[:5]
        raise ValueError(f"Duplicate sample_id values found: {dups}")

    # Check source values
    sources = df["source"].astype(str).str.strip()
    invalid_sources = sorted(set(sources) - set(VALID_SOURCES))
    if invalid_sources:
        raise ValueError(
            f"Invalid source values: {invalid_sources}. Allowed: {VALID_SOURCES}"
        )

    # Check split values (allow empty/unassigned before Step 3 unless require_split=True)
    for idx, raw_split in df["split"].items():
        if _is_missing_text(raw_split):
            if require_split:
                raise ValueError(f"Row {idx} has missing/empty 'split' (require_split=True)")
            continue
        split_val = str(raw_split).strip()
        if split_val not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split value {split_val!r} at row {idx}. Allowed: {VALID_SPLITS}"
            )

    # Check class_id can be parsed as integer
    try:
        pd.to_numeric(df["class_id"], errors="raise")
    except Exception as exc:
        raise ValueError(f"Column 'class_id' contains non-numeric values: {exc}") from exc

    # Check boolean-like flags
    for bool_col in ("is_augmented", "is_valid"):
        for idx, val in df[bool_col].items():
            try:
                _coerce_bool_value(val)
            except ValueError as exc:
                raise ValueError(
                    f"Column '{bool_col}' has invalid boolean value at row {idx}: {val!r}"
                ) from exc

    # Augmented rows must reference a parent_sample_id and belong to train split when split is set
    for idx, row in df.iterrows():
        is_aug = _coerce_bool_value(row["is_augmented"])
        if is_aug:
            if _is_missing_text(row["parent_sample_id"]):
                raise ValueError(
                    f"Augmented row {idx} (sample_id={row['sample_id']!r}) must have 'parent_sample_id'"
                )
            if not _is_missing_text(row["split"]) and str(row["split"]).strip() != "train":
                raise ValueError(
                    f"Augmented row {idx} (sample_id={row['sample_id']!r}) cannot have split={row['split']!r}; only 'train' is allowed"
                )


def normalize_rel_path(path: str | Path, root: str | Path) -> str:
    """Normalize a file path to a POSIX relative path with respect to ``root``.

    Always uses forward slashes (``/``) for cross-platform reproducibility.
    If ``path`` is not located under ``root`` (e.g., different Windows drive),
    falls back cleanly to its resolved POSIX string.
    """
    p = Path(path)
    r = Path(root).resolve()

    if not p.is_absolute():
        candidate = (r / p).resolve()
    else:
        candidate = p.resolve()

    try:
        return candidate.relative_to(r).as_posix()
    except ValueError:
        return candidate.as_posix()


def probe_video_metadata(video_path: str | Path) -> dict[str, Any]:
    """Safely inspect video metadata using ``cv2.VideoCapture``.

    Returns:
        Dictionary with keys ``fps``, ``num_frames``, ``duration_sec``,
        ``width``, ``height``, and ``is_valid``. Never raises on corrupt or
        missing videos.
    """
    invalid_result: dict[str, Any] = {
        "fps": 0.0,
        "num_frames": 0,
        "duration_sec": 0.0,
        "width": 0,
        "height": 0,
        "is_valid": False,
    }

    vp = Path(video_path)
    if not vp.exists() or not vp.is_file() or vp.stat().st_size == 0:
        return invalid_result

    cap: cv2.VideoCapture | None = None
    try:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            return invalid_result

        ret, first_frame = cap.read()
        if not ret or first_frame is None or first_frame.size == 0:
            return invalid_result

        height, width = int(first_frame.shape[0]), int(first_frame.shape[1])
        prop_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
        prop_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
        if prop_width > 0 and prop_height > 0:
            width, height = prop_width, prop_height

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0 or fps != fps:  # NaN or non-positive
            fps = 0.0

        num_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        if num_frames <= 0:
            # Fallback: count frames sequentially (we already read frame 1)
            count = 1
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                count += 1
            num_frames = count

        duration_sec = round(num_frames / fps, 4) if fps > 0.0 and num_frames > 0 else 0.0
        is_valid = bool(num_frames > 0 and width > 0 and height > 0)

        return {
            "fps": round(fps, 4),
            "num_frames": int(num_frames),
            "duration_sec": float(duration_sec),
            "width": int(width),
            "height": int(height),
            "is_valid": is_valid,
        }
    except Exception:
        return invalid_result
    finally:
        if cap is not None:
            cap.release()


def make_empty_manifest() -> pd.DataFrame:
    """Create an empty manifest DataFrame with the canonical column order."""
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def save_manifest_df(
    df: pd.DataFrame,
    path: str | Path,
    *,
    validate: bool = True,
    require_split: bool = False,
) -> Path:
    """Save a manifest DataFrame to CSV in canonical column order."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if validate:
        validate_manifest_df(df, require_split=require_split)
    ordered = df.loc[:, MANIFEST_COLUMNS] if not df.empty else make_empty_manifest()
    ordered.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def load_manifest_df(
    path: str | Path,
    *,
    validate: bool = False,
    require_split: bool = False,
) -> pd.DataFrame:
    """Load a manifest CSV file with clean string defaults for optional text columns."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {csv_path}")
    df = pd.read_csv(
        csv_path,
        dtype={
            "sample_id": str,
            "video_path": str,
            "label": str,
            "signer_id": str,
            "source": str,
            "split": str,
            "parent_sample_id": str,
            "aug_type": str,
        },
        keep_default_na=True,
    )
    for text_col in ("split", "parent_sample_id", "aug_type"):
        if text_col in df.columns:
            df[text_col] = df[text_col].fillna("")
    if validate:
        validate_manifest_df(df, require_split=require_split)
    return df

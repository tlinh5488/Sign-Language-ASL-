"""Unified manifest builder and data quality auditor for Quy's VSL pipeline.

Merges Kaggle VSL manifests (`kaggle_manifest.csv`, `kaggle_v2_manifest.csv`) and
self-recorded manifest (`self_manifest.csv`) — or scans raw video directories — into
a single unified manifest (`data/manifests/dataset_manifest.csv`), probes/validates
video metadata, deduplicates samples, checks signer/label integrity, and produces
`data/manifests/quality_report.json`.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data.collect_videos import slugify
from src.data.schema import (
    KAGGLE_SOURCES,
    MANIFEST_COLUMNS,
    is_kaggle_source,
    load_class_list,
    load_config,
    normalize_rel_path,
    normalize_vietnamese_label,
    probe_video_metadata,
    validate_manifest_df,
)

DEFAULT_CONFIG_PATH = Path("configs/data_config.yaml")
DEFAULT_KAGGLE_MANIFEST = Path("data/manifests/kaggle_manifest.csv")
DEFAULT_KAGGLE_V2_MANIFEST = Path("data/manifests/kaggle_v2_manifest.csv")
DEFAULT_SELF_MANIFEST = Path("data/manifests/self_manifest.csv")
DEFAULT_OUTPUT_MANIFEST = Path("data/manifests/dataset_manifest.csv")
DEFAULT_QUALITY_REPORT = Path("data/manifests/quality_report.json")
DEFAULT_CLASS_LIST_PATH = Path("configs/class_list.json")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _is_missing_text(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    s = str(val).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def _resolve_video_path(video_path: str | Path, root_dir: str | Path | None = None) -> Path:
    vp = Path(str(video_path))
    if vp.is_absolute():
        return vp
    if root_dir is not None:
        candidate = Path(root_dir) / vp
        if candidate.exists():
            return candidate
        return candidate
    return vp


def _normalize_video_key(video_path: str | Path, root_dir: str | Path | None = None) -> str:
    vp = Path(str(video_path))
    if root_dir is not None:
        try:
            return normalize_rel_path(vp, root_dir)
        except Exception:
            pass
    return vp.as_posix().lstrip("./")


def infer_signer_from_filename(
    video_path: str | Path,
    class_slug: str = "",
    source: str = "self_recorded",
) -> str:
    """Infer a deterministic signer_id from a video filename or parent directory."""
    stem = Path(video_path).stem

    # 1. Exact pattern from collect_videos.py: <signer_id>_<class_slug>_<YYYYMMDD_HHMMSS...>
    if class_slug:
        marker = f"_{class_slug}_"
        idx = stem.lower().find(marker.lower())
        if idx > 0:
            candidate = stem[:idx].strip("_-")
            if candidate:
                return candidate.lower()

    # 2. Explicit signer/person/user prefix: signer_01_..., vsl2_signer_001_..., p02_..., user-3_...
    m = re.match(
        r"^(vsl2_signer[_-]?[a-z0-9]+|signer[_-]?\d+|person[_-]?\d+|user[_-]?\d+|s\d+|p\d+)[_-]",
        stem,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).lower().replace("-", "_")

    # 3. Kaggle QIPEDC regional suffix (e.g. D0001B / D0001N / D0001T)
    if source == "kaggle_vsl":
        reg_match = re.search(r"([BNT])$", stem)
        if reg_match:
            region_map = {
                "B": "kaggle_signer_north",
                "N": "kaggle_signer_south",
                "T": "kaggle_signer_central",
            }
            return region_map[reg_match.group(1)]
        prefix = stem[0].lower() if stem else "x"
        return f"kaggle_signer_common_{prefix}"

    # 3b. Kaggle VSL-v2 fallback signer inference from directory layout (train/test)
    if source == "kaggle_vsl_v2":
        parts_lower = [p.lower() for p in Path(video_path).parts]
        for split_hint in ("train", "test", "val"):
            if split_hint in parts_lower:
                return f"vsl2_signer_{split_hint}"
        return "vsl2_signer_default"

    # 4. Fallback: first underscore token if descriptive
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0]:
        return parts[0].lower()

    return f"{source}_signer_unknown"


def _build_class_lookup(
    class_list: Sequence[dict[str, Any]] | None,
) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    by_key: dict[str, dict[str, Any]] = {}
    by_id: dict[int, dict[str, Any]] = {}
    if not class_list:
        return by_key, by_id
    for entry in class_list:
        cid = int(entry["class_id"])
        label = normalize_vietnamese_label(str(entry["label"]))
        slug = str(entry.get("slug") or slugify(label)).strip().lower()
        normalized = {"class_id": cid, "label": label, "slug": slug}
        by_id[cid] = normalized
        by_key[label.lower()] = normalized
        by_key[slug] = normalized
        try:
            by_key[slugify(label)] = normalized
        except ValueError:
            pass
    return by_key, by_id


def scan_raw_directory(
    raw_dir: str | Path,
    source: str = "self_recorded",
    root_dir: str | Path | None = None,
    class_list: Sequence[dict[str, Any]] | None = None,
    probe_videos: bool = True,
) -> pd.DataFrame:
    """Scan a raw video directory (`<raw_dir>/<class_folder>/*.mp4`) into a manifest DataFrame."""
    base = Path(raw_dir)
    if not base.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    by_key, _ = _build_class_lookup(class_list)
    rows: list[dict[str, Any]] = []

    video_files = sorted(
        p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    for vp in video_files:
        rel_parts = vp.relative_to(base).parts
        # Exclude pre-augmented folders (e.g., VSL-v2 processed_augmented/) to avoid leakage before Step 3
        if any("processed_augmented" in part.lower() for part in rel_parts[:-1]):
            continue

        # Handle both `<class_folder>/<file>.mp4` and `frame_splited/{train,test}/<class_folder>/<file>.mp4`
        folder_name = vp.parent.name if len(rel_parts) >= 2 else "unknown"
        folder_name = normalize_vietnamese_label(folder_name)
        folder_slug = slugify(folder_name) if folder_name != "unknown" else "unknown"

        class_info = by_key.get(folder_name.lower()) or by_key.get(folder_slug)
        if class_info is not None:
            label = class_info["label"]
            class_id = int(class_info["class_id"])
            class_slug = class_info["slug"]
        else:
            label = folder_name
            class_id = -1
            class_slug = folder_slug

        signer_id = infer_signer_from_filename(vp, class_slug=class_slug, source=source)
        if source == "self_recorded":
            sample_prefix = "self"
        elif source == "kaggle_vsl_v2":
            sample_prefix = "kaggle_v2"
        else:
            sample_prefix = "kaggle"
        sample_id = f"{sample_prefix}_{slugify(class_slug)}_{slugify(vp.stem)}"

        if probe_videos:
            meta = probe_video_metadata(vp)
        else:
            meta = {
                "fps": 0.0,
                "num_frames": 0,
                "duration_sec": 0.0,
                "width": 0,
                "height": 0,
                "is_valid": vp.exists() and vp.stat().st_size > 0,
            }

        rel_video_path = (
            normalize_rel_path(vp, root_dir) if root_dir is not None else vp.as_posix()
        )

        rows.append(
            {
                "sample_id": sample_id,
                "video_path": rel_video_path,
                "label": label,
                "class_id": class_id,
                "signer_id": signer_id,
                "source": source,
                "split": "",
                "is_augmented": False,
                "parent_sample_id": "",
                "aug_type": "",
                "fps": float(meta.get("fps", 0.0) or 0.0),
                "num_frames": int(meta.get("num_frames", 0) or 0),
                "duration_sec": float(meta.get("duration_sec", 0.0) or 0.0),
                "width": int(meta.get("width", 0) or 0),
                "height": int(meta.get("height", 0) or 0),
                "is_valid": bool(meta.get("is_valid", False)),
            }
        )

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def _coerce_manifest_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults: dict[str, Any] = {
        "sample_id": "",
        "video_path": "",
        "label": "",
        "class_id": -1,
        "signer_id": "",
        "source": "self_recorded",
        "split": "",
        "is_augmented": False,
        "parent_sample_id": "",
        "aug_type": "",
        "fps": 0.0,
        "num_frames": 0,
        "duration_sec": 0.0,
        "width": 0,
        "height": 0,
        "is_valid": False,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    out = out[MANIFEST_COLUMNS].copy()
    for text_col in ("sample_id", "video_path", "label", "signer_id", "source", "split", "parent_sample_id", "aug_type"):
        out[text_col] = out[text_col].apply(lambda v: "" if _is_missing_text(v) else str(v).strip())
    out["label"] = out["label"].apply(lambda v: normalize_vietnamese_label(v) if v else "")

    out["is_augmented"] = out["is_augmented"].apply(
        lambda v: str(v).strip().lower() in {"true", "1", "yes"} if not isinstance(v, bool) else v
    )
    out["is_valid"] = out["is_valid"].apply(
        lambda v: str(v).strip().lower() in {"true", "1", "yes"} if not isinstance(v, bool) else v
    )

    for int_col in ("class_id", "num_frames", "width", "height"):
        out[int_col] = pd.to_numeric(out[int_col], errors="coerce").fillna(-1 if int_col == "class_id" else 0).astype(int)
    for float_col in ("fps", "duration_sec"):
        out[float_col] = pd.to_numeric(out[float_col], errors="coerce").fillna(0.0).astype(float)

    return out


def merge_manifests(
    manifest_paths: Sequence[str | Path] | None = None,
    raw_dirs: Sequence[tuple[str | Path, str]] | None = None,
    output_path: str | Path | None = None,
    root_dir: str | Path | None = None,
    class_list_path: str | Path | None = None,
    probe_videos: bool = True,
    reprobe_all: bool = False,
    drop_invalid: bool = False,
    strict: bool = False,
    kaggle_only: bool = False,
) -> pd.DataFrame:
    """Merge manifest CSVs and/or scanned raw video directories into a unified manifest.

    Steps performed:
      1. Loads all existing CSV manifests in `manifest_paths` and scans any `raw_dirs`.
         When `kaggle_only=True`, strictly keeps only Kaggle VSL sources
         (`kaggle_vsl`, `kaggle_vsl_v2`) and ignores any `self_recorded` inputs.
      2. Aligns columns to `MANIFEST_COLUMNS` and normalizes `video_path` relative to `root_dir`.
      3. Maps & canonicalizes `class_id`/`label` using `class_list.json` (or deterministic sorted IDs).
      4. Deduplicates by `sample_id` and normalized `video_path` (keeps first occurrence).
      5. Probes video metadata (`fps`, `num_frames`, `duration_sec`, `width`, `height`, `is_valid`).
      6. Verifies every row has non-empty `signer_id` and `label` (marks `is_valid=False` or raises if `strict=True`).
      7. Saves unified CSV to `output_path` if provided.
    """
    mode = "kaggle_only" if kaggle_only else "hybrid"
    class_list: list[dict[str, Any]] | None = None
    if class_list_path is not None and Path(class_list_path).exists():
        class_list = load_class_list(class_list_path)
    by_key, by_id = _build_class_lookup(class_list)

    frames: list[pd.DataFrame] = []

    for mp in manifest_paths or []:
        p = Path(mp)
        if p.exists() and p.stat().st_size > 0:
            df_part = pd.read_csv(
                p,
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
            )
            if not df_part.empty:
                coerced = _coerce_manifest_columns(df_part)
                if kaggle_only:
                    coerced = coerced[coerced["source"].map(is_kaggle_source)].reset_index(drop=True)
                if not coerced.empty:
                    frames.append(coerced)

    for raw_dir, source in raw_dirs or []:
        if kaggle_only and not is_kaggle_source(source):
            continue
        scanned = scan_raw_directory(
            raw_dir=raw_dir,
            source=source,
            root_dir=root_dir,
            class_list=class_list,
            probe_videos=False,  # probed uniformly below after dedup
        )
        if not scanned.empty:
            frames.append(_coerce_manifest_columns(scanned))

    if not frames:
        empty_df = pd.DataFrame(columns=MANIFEST_COLUMNS)
        empty_df.attrs["duplicates_removed"] = 0
        empty_df.attrs["mode"] = mode
        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            empty_df.to_csv(out_p, index=False, encoding="utf-8-sig")
        return empty_df

    combined = pd.concat(frames, ignore_index=True)
    if kaggle_only:
        combined = combined[combined["source"].map(is_kaggle_source)].reset_index(drop=True)
        if combined.empty:
            empty_df = pd.DataFrame(columns=MANIFEST_COLUMNS)
            empty_df.attrs["duplicates_removed"] = 0
            empty_df.attrs["mode"] = mode
            if output_path is not None:
                out_p = Path(output_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                empty_df.to_csv(out_p, index=False, encoding="utf-8-sig")
            return empty_df

    # Normalize video_path relative to root_dir if provided
    if root_dir is not None:
        combined["video_path"] = combined["video_path"].apply(
            lambda vp: _normalize_video_key(vp, root_dir) if vp else ""
        )
    else:
        combined["video_path"] = combined["video_path"].apply(
            lambda vp: Path(vp).as_posix() if vp else ""
        )

    # Fill missing sample_id deterministically if blank
    for idx, row in combined.iterrows():
        if not row["sample_id"]:
            vp_stem = Path(row["video_path"]).stem if row["video_path"] else f"row_{idx}"
            combined.at[idx, "sample_id"] = f"{row['source']}_{slugify(vp_stem)}"

    # Deduplicate by sample_id and normalized video_path
    before_dedup = len(combined)
    combined["_norm_vp"] = combined["video_path"].str.lower()
    combined = combined.drop_duplicates(subset=["sample_id"], keep="first")
    # Only dedup on non-empty video_path so multiple missing-path rows stay visible in audit
    non_empty_mask = combined["_norm_vp"] != ""
    dedup_non_empty = combined[non_empty_mask].drop_duplicates(subset=["_norm_vp"], keep="first")
    combined = pd.concat([dedup_non_empty, combined[~non_empty_mask]], ignore_index=True)
    combined = combined.drop(columns=["_norm_vp"]).reset_index(drop=True)
    duplicates_removed = before_dedup - len(combined)

    # Align & canonicalize label <-> class_id against class_list when provided
    for idx, row in combined.iterrows():
        label = normalize_vietnamese_label(str(row["label"])) if str(row["label"]).strip() else ""
        cid = int(row["class_id"])
        if label and by_key:
            info = by_key.get(label.lower())
            if info is None:
                try:
                    info = by_key.get(slugify(label))
                except ValueError:
                    info = None
            if info is not None:
                combined.at[idx, "class_id"] = int(info["class_id"])
                combined.at[idx, "label"] = info["label"]
        elif not label and cid in by_id:
            combined.at[idx, "label"] = by_id[cid]["label"]

    # Assign deterministic class_id for any remaining labeled rows with class_id < 0
    unresolved_labels = sorted(
        {
            str(lbl).strip()
            for lbl, cid in zip(combined["label"], combined["class_id"])
            if str(lbl).strip() and int(cid) < 0
        }
    )
    if unresolved_labels:
        existing_max = int(combined["class_id"].max()) if (combined["class_id"] >= 0).any() else -1
        fallback_map = {lbl: existing_max + 1 + i for i, lbl in enumerate(unresolved_labels)}
        for idx, row in combined.iterrows():
            if str(row["label"]).strip() and int(row["class_id"]) < 0:
                combined.at[idx, "class_id"] = fallback_map[str(row["label"]).strip()]

    # Probe video metadata and validate integrity
    for idx, row in combined.iterrows():
        needs_probe = reprobe_all or (
            probe_videos
            and (
                int(row["num_frames"]) <= 0
                or float(row["fps"]) <= 0.0
                or int(row["width"]) <= 0
                or int(row["height"]) <= 0
                or not bool(row["is_valid"])
            )
        )
        if needs_probe:
            resolved_vp = _resolve_video_path(row["video_path"], root_dir=root_dir)
            meta = probe_video_metadata(resolved_vp)
            combined.at[idx, "fps"] = float(meta.get("fps", 0.0) or 0.0)
            combined.at[idx, "num_frames"] = int(meta.get("num_frames", 0) or 0)
            combined.at[idx, "duration_sec"] = float(meta.get("duration_sec", 0.0) or 0.0)
            combined.at[idx, "width"] = int(meta.get("width", 0) or 0)
            combined.at[idx, "height"] = int(meta.get("height", 0) or 0)
            combined.at[idx, "is_valid"] = bool(meta.get("is_valid", False))

        # Every valid row MUST have non-empty signer_id, non-empty label, and class_id >= 0
        missing_signer = not str(combined.at[idx, "signer_id"]).strip()
        missing_label = not str(combined.at[idx, "label"]).strip() or int(combined.at[idx, "class_id"]) < 0
        if missing_signer or missing_label:
            if strict:
                raise ValueError(
                    f"Row {combined.at[idx, 'sample_id']!r} has missing metadata: "
                    f"signer_id={combined.at[idx, 'signer_id']!r}, "
                    f"label={combined.at[idx, 'label']!r}, class_id={combined.at[idx, 'class_id']}"
                )
            combined.at[idx, "is_valid"] = False

    if drop_invalid:
        combined = combined[combined["is_valid"]].reset_index(drop=True)

    combined = combined[MANIFEST_COLUMNS].copy()
    combined.attrs["duplicates_removed"] = int(duplicates_removed)
    combined.attrs["mode"] = mode

    if strict or (not combined.empty and combined["is_valid"].all()):
        validate_manifest_df(combined)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_p, index=False, encoding="utf-8-sig")

    return combined


def audit_manifest(
    manifest: pd.DataFrame | str | Path,
    report_path: str | Path | None = None,
    root_dir: str | Path | None = None,
    min_samples_per_class: int | None = None,
    min_signers_per_class: int = 3,
    imbalance_ratio_threshold: float = 3.0,
    reprobe: bool = False,
    kaggle_only: bool | None = None,
) -> dict[str, Any]:
    """Audit a manifest DataFrame or CSV and generate a structured quality report.

    Reports counts per class, per signer, per source, unique signers per class,
    classes with >= 3 distinct signers (`classes_with_3plus_signers`),
    `is_ready_for_kaggle_only_split`, invalid video details, duration & FPS
    statistics, and class-imbalance warnings.
    """
    duplicates_removed = 0
    inferred_mode: str | None = None
    if isinstance(manifest, (str, Path)):
        df = pd.read_csv(
            manifest,
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
        )
    else:
        duplicates_removed = int(manifest.attrs.get("duplicates_removed", 0))
        inferred_mode = manifest.attrs.get("mode")
        df = manifest.copy()

    is_kaggle_only = bool(kaggle_only) if kaggle_only is not None else (inferred_mode == "kaggle_only")
    mode = "kaggle_only" if is_kaggle_only else "hybrid"
    effective_min_samples = (
        int(min_samples_per_class)
        if min_samples_per_class is not None
        else (3 if is_kaggle_only else 5)
    )

    df = _coerce_manifest_columns(df)

    invalid_videos: list[dict[str, Any]] = []
    missing_signer_rows: list[str] = []
    missing_label_rows: list[str] = []

    for idx, row in df.iterrows():
        reasons: list[str] = []
        if reprobe:
            resolved = _resolve_video_path(row["video_path"], root_dir=root_dir)
            meta = probe_video_metadata(resolved)
            df.at[idx, "fps"] = float(meta.get("fps", 0.0) or 0.0)
            df.at[idx, "num_frames"] = int(meta.get("num_frames", 0) or 0)
            df.at[idx, "duration_sec"] = float(meta.get("duration_sec", 0.0) or 0.0)
            df.at[idx, "width"] = int(meta.get("width", 0) or 0)
            df.at[idx, "height"] = int(meta.get("height", 0) or 0)
            df.at[idx, "is_valid"] = bool(meta.get("is_valid", False))

        if not str(df.at[idx, "signer_id"]).strip():
            reasons.append("missing_signer_id")
            missing_signer_rows.append(str(df.at[idx, "sample_id"]))
        if not str(df.at[idx, "label"]).strip() or int(df.at[idx, "class_id"]) < 0:
            reasons.append("missing_or_invalid_label")
            missing_label_rows.append(str(df.at[idx, "sample_id"]))
        if not bool(df.at[idx, "is_valid"]):
            if int(df.at[idx, "num_frames"]) <= 0:
                reasons.append("unreadable_or_empty_video")
            else:
                reasons.append("invalid_video_metadata")

        if reasons:
            df.at[idx, "is_valid"] = False
            invalid_videos.append(
                {
                    "sample_id": str(df.at[idx, "sample_id"]),
                    "video_path": str(df.at[idx, "video_path"]),
                    "label": str(df.at[idx, "label"]),
                    "signer_id": str(df.at[idx, "signer_id"]),
                    "reasons": reasons,
                }
            )

    valid_df = df[df["is_valid"]].copy()

    counts_per_class = {
        str(k): int(v)
        for k, v in valid_df["label"].value_counts().sort_index().items()
        if str(k).strip()
    }
    counts_per_signer = {
        str(k): int(v)
        for k, v in valid_df["signer_id"].value_counts().sort_index().items()
        if str(k).strip()
    }
    counts_per_source = {
        str(k): int(v)
        for k, v in valid_df["source"].value_counts().sort_index().items()
        if str(k).strip()
    }
    signers_per_class = {
        str(lbl): int(grp["signer_id"].nunique())
        for lbl, grp in valid_df.groupby("label")
        if str(lbl).strip()
    }

    # Duration & FPS statistics over valid videos
    durations = valid_df["duration_sec"].astype(float).to_numpy()
    durations = durations[durations > 0]
    if durations.size > 0:
        duration_stats: dict[str, float | None] = {
            "min": round(float(np.min(durations)), 4),
            "max": round(float(np.max(durations)), 4),
            "mean": round(float(np.mean(durations)), 4),
            "median": round(float(np.median(durations)), 4),
            "std": round(float(np.std(durations)), 4),
            "p10": round(float(np.percentile(durations, 10)), 4),
            "p90": round(float(np.percentile(durations, 90)), 4),
        }
    else:
        duration_stats = {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "p10": None,
            "p90": None,
        }

    fps_vals = valid_df["fps"].astype(float).to_numpy()
    fps_vals = fps_vals[fps_vals > 0]
    fps_stats: dict[str, float | None] = (
        {
            "min": round(float(np.min(fps_vals)), 2),
            "max": round(float(np.max(fps_vals)), 2),
            "mean": round(float(np.mean(fps_vals)), 2),
        }
        if fps_vals.size > 0
        else {"min": None, "max": None, "mean": None}
    )

    resolutions = (
        valid_df.apply(lambda r: f"{int(r['width'])}x{int(r['height'])}", axis=1)
        .value_counts()
        .to_dict()
        if not valid_df.empty
        else {}
    )
    resolution_counts = {str(k): int(v) for k, v in resolutions.items()}

    # Class imbalance & split-readiness warnings
    class_imbalance_warnings: list[str] = []
    for lbl, cnt in counts_per_class.items():
        if cnt < effective_min_samples:
            class_imbalance_warnings.append(
                f"Class '{lbl}' has only {cnt} valid sample(s) (< min_samples_per_class={effective_min_samples})."
            )
    for lbl, n_signers in signers_per_class.items():
        if n_signers < min_signers_per_class:
            class_imbalance_warnings.append(
                f"Class '{lbl}' has only {n_signers} distinct signer(s) (< {min_signers_per_class} needed for full 3-way signer-disjoint coverage)."
            )

    if len(counts_per_class) >= 2:
        min_cls = min(counts_per_class, key=counts_per_class.get)  # type: ignore[arg-type]
        max_cls = max(counts_per_class, key=counts_per_class.get)  # type: ignore[arg-type]
        min_cnt = counts_per_class[min_cls]
        max_cnt = counts_per_class[max_cls]
        ratio = float(max_cnt) / max(float(min_cnt), 1.0)
        if ratio > imbalance_ratio_threshold:
            class_imbalance_warnings.append(
                f"Class imbalance ratio is {ratio:.2f}x (max='{max_cls}':{max_cnt} vs min='{min_cls}':{min_cnt}, "
                f"threshold={imbalance_ratio_threshold:.1f}x)."
            )

    unique_signers = len(counts_per_signer)
    classes_with_3plus_signers = sorted(
        lbl for lbl, n_s in signers_per_class.items() if n_s >= 3
    )
    kaggle_valid_df = valid_df[valid_df["source"].map(is_kaggle_source)]
    kaggle_unique_signers = (
        int(kaggle_valid_df["signer_id"].nunique()) if not kaggle_valid_df.empty else 0
    )
    kaggle_signers_per_class = {
        str(lbl): int(grp["signer_id"].nunique())
        for lbl, grp in kaggle_valid_df.groupby("label")
        if str(lbl).strip()
    }
    kaggle_classes_with_3plus = sorted(
        lbl for lbl, n_s in kaggle_signers_per_class.items() if n_s >= 3
    )
    kaggle_sources_present = sorted(
        s for s in counts_per_source if is_kaggle_source(s)
    )
    all_valid_sources_kaggle = all(is_kaggle_source(s) for s in valid_df["source"].unique())

    is_ready_for_split = (
        len(valid_df) > 0
        and unique_signers >= 3
        and len(missing_signer_rows) == 0
        and len(missing_label_rows) == 0
    )
    is_ready_for_kaggle_only_split = (
        len(kaggle_valid_df) > 0
        and kaggle_unique_signers >= 3
        and len(kaggle_classes_with_3plus) > 0
        and len(missing_signer_rows) == 0
        and len(missing_label_rows) == 0
        and (not is_kaggle_only or all_valid_sources_kaggle)
    )

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "kaggle_sources_supported": list(KAGGLE_SOURCES),
        "kaggle_sources_present": kaggle_sources_present,
        "total_samples": int(len(df)),
        "valid_samples": int(len(valid_df)),
        "invalid_samples": int(len(invalid_videos)),
        "duplicates_removed": int(duplicates_removed),
        "num_classes": int(len(counts_per_class)),
        "num_signers": int(unique_signers),
        "counts_per_class": counts_per_class,
        "counts_per_signer": counts_per_signer,
        "counts_per_source": counts_per_source,
        "signers_per_class": signers_per_class,
        "classes_with_3plus_signers": classes_with_3plus_signers,
        "num_classes_with_3plus_signers": int(len(classes_with_3plus_signers)),
        "invalid_videos": invalid_videos,
        "missing_signer_count": int(len(missing_signer_rows)),
        "missing_label_count": int(len(missing_label_rows)),
        "duration_stats": duration_stats,
        "fps_stats": fps_stats,
        "resolution_counts": resolution_counts,
        "class_imbalance_warnings": class_imbalance_warnings,
        "is_ready_for_split": bool(is_ready_for_split),
        "is_ready_for_kaggle_only_split": bool(is_ready_for_kaggle_only_split),
    }

    if report_path is not None:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge Kaggle & self-recorded manifests and run VSL dataset quality audit."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to central YAML config (default: configs/data_config.yaml).",
    )
    parser.add_argument(
        "--kaggle-manifest",
        type=Path,
        default=None,
        help="Path to primary Kaggle QIPEDC manifest CSV (default: data/manifests/kaggle_manifest.csv).",
    )
    parser.add_argument(
        "--kaggle-v2-manifest",
        type=Path,
        default=None,
        help="Path to secondary Kaggle VSL-v2 manifest CSV (default: data/manifests/kaggle_v2_manifest.csv).",
    )
    parser.add_argument(
        "--self-manifest",
        type=Path,
        default=None,
        help="Path to self-recorded manifest CSV (default: data/manifests/self_manifest.csv).",
    )
    parser.add_argument(
        "--manifests",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit list of manifest CSV paths to merge (overrides default manifest list).",
    )
    parser.add_argument(
        "--scan-self-dir",
        type=Path,
        default=None,
        help="Optional raw self-recorded directory to scan if self_manifest.csv is absent.",
    )
    parser.add_argument(
        "--scan-kaggle-dir",
        type=Path,
        default=None,
        help="Optional raw Kaggle QIPEDC directory to scan if kaggle_manifest.csv is absent.",
    )
    parser.add_argument(
        "--scan-kaggle-v2-dir",
        type=Path,
        default=None,
        help="Optional raw Kaggle VSL-v2 directory to scan if kaggle_v2_manifest.csv is absent.",
    )
    parser.add_argument(
        "--class-list",
        type=Path,
        default=None,
        help="Path to class_list.json for label <-> class_id alignment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Unified output manifest CSV (default: data/manifests/dataset_manifest.csv).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output quality audit JSON report (default: data/manifests/quality_report.json).",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Project root for resolving/normalizing video_path (default: current working directory).",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip probing videos with OpenCV if metadata is already populated.",
    )
    parser.add_argument(
        "--reprobe-all",
        action="store_true",
        help="Force OpenCV metadata re-probing on every video row.",
    )
    parser.add_argument(
        "--drop-invalid",
        action="store_true",
        help="Drop invalid video rows from the unified manifest CSV.",
    )
    parser.add_argument(
        "--kaggle-only",
        action="store_true",
        help=(
            "Build dataset_manifest.csv strictly from Kaggle VSL sources "
            "(kaggle_vsl + kaggle_vsl_v2; ignores self_manifest.csv, --scan-self-dir, "
            "and any self_recorded rows)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg: dict[str, Any] = {}
    if args.config and Path(args.config).exists():
        cfg = load_config(args.config)
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg, dict) else {}

    if args.manifests is not None:
        manifest_paths = list(args.manifests)
    else:
        kaggle_manifest = args.kaggle_manifest or Path(
            paths_cfg.get("kaggle_manifest", DEFAULT_KAGGLE_MANIFEST)
        )
        kaggle_v2_manifest = args.kaggle_v2_manifest or Path(
            paths_cfg.get("kaggle_v2_manifest", DEFAULT_KAGGLE_V2_MANIFEST)
        )
        if args.kaggle_only:
            manifest_paths = [kaggle_manifest, kaggle_v2_manifest]
        else:
            self_manifest = args.self_manifest or Path(
                paths_cfg.get("self_manifest", DEFAULT_SELF_MANIFEST)
            )
            manifest_paths = [kaggle_manifest, kaggle_v2_manifest, self_manifest]

    raw_dirs: list[tuple[Path, str]] = []
    if args.scan_kaggle_dir is not None:
        raw_dirs.append((args.scan_kaggle_dir, "kaggle_vsl"))
    if args.scan_kaggle_v2_dir is not None:
        raw_dirs.append((args.scan_kaggle_v2_dir, "kaggle_vsl_v2"))
    if args.scan_self_dir is not None and not args.kaggle_only:
        raw_dirs.append((args.scan_self_dir, "self_recorded"))

    class_list_path = args.class_list or Path(
        paths_cfg.get("class_list_path", paths_cfg.get("class_list", DEFAULT_CLASS_LIST_PATH))
    )
    output_path = args.output or Path(
        paths_cfg.get("dataset_manifest", DEFAULT_OUTPUT_MANIFEST)
    )
    report_path = args.report or Path(
        paths_cfg.get("quality_report", DEFAULT_QUALITY_REPORT)
    )
    rec_cfg = cfg.get("recording", {}) if isinstance(cfg, dict) else {}
    imbalance_ratio = float(rec_cfg.get("imbalance_ratio_warning", 3.0))
    root_dir = args.root_dir if args.root_dir is not None else Path.cwd()

    merged_df = merge_manifests(
        manifest_paths=manifest_paths,
        raw_dirs=raw_dirs,
        output_path=output_path,
        root_dir=root_dir,
        class_list_path=class_list_path if class_list_path.exists() else None,
        probe_videos=not args.no_probe,
        reprobe_all=args.reprobe_all,
        drop_invalid=args.drop_invalid,
        kaggle_only=args.kaggle_only,
    )

    report = audit_manifest(
        merged_df,
        report_path=report_path,
        root_dir=root_dir,
        imbalance_ratio_threshold=imbalance_ratio,
        reprobe=False,
        kaggle_only=args.kaggle_only,
    )

    print(
        f"[DONE] Unified manifest saved to {output_path} "
        f"(mode={report['mode']}, total={report['total_samples']}, valid={report['valid_samples']}, "
        f"invalid={report['invalid_samples']}, dedup={report['duplicates_removed']})."
    )
    print(f"[DONE] Quality report saved to {report_path}.")
    if report["class_imbalance_warnings"]:
        print("[WARN] Quality warnings:")
        for w in report["class_imbalance_warnings"]:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

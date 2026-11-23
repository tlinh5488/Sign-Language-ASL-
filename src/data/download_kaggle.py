"""Download and index public Kaggle VSL datasets into the standardized manifest schema.

Supported public Kaggle Vietnamese Sign Language (VSL) datasets:
1. Primary (`qipedc`, source=``"kaggle_vsl"``):
   ``aresusayhi/vsl-vietnamese-sign-languages`` (QIPEDC v5, ~20.77 GB, Apache 2.0,
   4,362 videos, 3,315 labels):
   - ``Dataset/Labels/label.csv`` with columns ``ID``, ``VIDEO``, ``LABEL``
   - ``Dataset/Videos/<VIDEO>`` (e.g. ``D0001B.mp4``, ``D0001N.mp4``, ``D0001T.mp4``, ``W0002.mp4``)
2. Secondary (`vsl_v2`, source=``"kaggle_vsl_v2"``):
   ``nguyenanfms/vsl-vietnamese-sign-language-v2`` (VSL - Vietnamese Sign Language
   Cropped v8, CC BY 4.0, 472 Vietnamese glosses, 26,446 canonical cropped front-view
   224x224 clips compiled from multi-signer VSL400 + online sources):
   - ``processed/processed/frame_splited/{train,test}/{gloss}/{videoid}.mp4``
   - ``processed/processed/frame_splited/split_manifest.tsv``
   - ``raw/raw/VSL400/Part_*/split_*/front_view.json`` (contains explicit ``signer_id`` per ``video_id``)

Both download via ``kagglehub`` with automatic fallback to the ``kaggle`` CLI, and
support offline parsing of already-downloaded/synthetic directories via ``--local-dir``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.data.schema import (
    MANIFEST_COLUMNS,
    load_class_list,
    load_config,
    make_empty_manifest,
    normalize_rel_path,
    normalize_vietnamese_label,
    probe_video_metadata,
    save_manifest_df,
    slugify_label,
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

DEFAULT_REGION_SUFFIX_MAP: dict[str, str] = {
    "B": "kaggle_signer_north",
    "N": "kaggle_signer_south",
    "T": "kaggle_signer_central",
}

_QIPEDC_STEM_RE = re.compile(r"^([A-Za-z]+)(\d+)([A-Za-z]+)?$")
_SIGNER_TOKEN_RE = re.compile(r"(signer[_-]?[a-z0-9]+|person[_-]?[a-z0-9]+|p\d{2,})", re.IGNORECASE)
_SPLIT_DIR_NAMES = {"train", "val", "valid", "validation", "test"}


def download_kaggle_dataset(
    dataset_slug: str = "aresusayhi/vsl-vietnamese-sign-languages",
    target_dir: str | Path | None = None,
    *,
    copy_to_target: bool = False,
    only_labels: bool = False,
    labels_rel_path: str = "Dataset/Labels/label.csv",
) -> Path:
    """Download a Kaggle VSL dataset via ``kagglehub`` with fallback to ``kaggle`` CLI.

    Args:
        dataset_slug: Kaggle dataset identifier (``owner/dataset-name``).
        target_dir: Optional local directory for fallback CLI download or copying.
        copy_to_target: If True and ``target_dir`` is set, copies downloaded files into ``target_dir``.
        only_labels: If True, downloads only ``labels_rel_path`` (tiny metadata check).
        labels_rel_path: Relative path of the label/manifest metadata file inside the dataset.

    Returns:
        Path to the local dataset directory (or downloaded metadata file when ``only_labels=True``).
    """
    kagglehub_err: Exception | None = None
    try:
        import kagglehub  # type: ignore

        if only_labels:
            downloaded = Path(
                kagglehub.dataset_download(dataset_slug, path=labels_rel_path)
            )
            if copy_to_target and target_dir is not None:
                dest = Path(target_dir) / labels_rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(downloaded, dest)
                return dest
            return downloaded

        downloaded_dir = Path(kagglehub.dataset_download(dataset_slug))
        if copy_to_target and target_dir is not None:
            dest_dir = Path(target_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(downloaded_dir, dest_dir, dirs_exist_ok=True)
            return dest_dir
        return downloaded_dir
    except Exception as exc:
        kagglehub_err = exc

    # Fallback to `kaggle` CLI
    cli_target = Path(target_dir) if target_dir is not None else Path("data/raw/kaggle_vsl")
    cli_target.mkdir(parents=True, exist_ok=True)

    kaggle_exe = shutil.which("kaggle")
    if kaggle_exe is None:
        raise RuntimeError(
            "Failed to download dataset via kagglehub "
            f"({kagglehub_err}) and 'kaggle' CLI was not found on PATH. "
            "Configure Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY) "
            "or install compatible kagglehub/kaggle packages."
        ) from kagglehub_err

    if only_labels:
        cmd = [
            kaggle_exe,
            "datasets",
            "download",
            "-d",
            dataset_slug,
            "-f",
            labels_rel_path,
            "-p",
            str(cli_target),
            "--unzip",
        ]
    else:
        cmd = [
            kaggle_exe,
            "datasets",
            "download",
            "-d",
            dataset_slug,
            "-p",
            str(cli_target),
            "--unzip",
        ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Both kagglehub ({kagglehub_err}) and kaggle CLI failed "
            f"(exit {result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
        ) from kagglehub_err

    if only_labels:
        candidate = cli_target / Path(labels_rel_path).name
        if candidate.exists():
            return candidate
    return cli_target


def infer_kaggle_signer_id(
    video_name: str,
    rel_path: str | Path | None = None,
    row_meta: dict[str, Any] | None = None,
    *,
    region_suffix_map: dict[str, str] | None = None,
    default_signer_prefix: str = "kaggle_signer_common",
) -> str:
    """Deterministically infer a ``signer_id`` for a QIPEDC Kaggle VSL video.

    Resolution priority:
    1. Explicit ``signer_id`` / ``signer`` / ``person_id`` column in metadata.
    2. Explicit ``region`` / ``dialect`` column in metadata.
    3. Signer token in relative folder path or filename (e.g., ``signer_01``, ``p03``).
    4. QIPEDC regional suffix in filename stem (e.g., ``D0001B.mp4`` -> ``kaggle_signer_north``,
       ``D0001N.mp4`` -> ``kaggle_signer_south``, ``D0001T.mp4`` -> ``kaggle_signer_central``).
    5. Deterministic fallback using QIPEDC prefix group (e.g., ``kaggle_signer_common_d`` or
       ``kaggle_signer_common_w``) or parent folder name.
    """
    suffix_map = dict(DEFAULT_REGION_SUFFIX_MAP)
    if region_suffix_map:
        suffix_map.update({str(k).upper(): str(v) for k, v in region_suffix_map.items()})

    meta = row_meta or {}
    for key in ("signer_id", "SIGNER_ID", "signer", "SIGNER", "person_id", "PERSON_ID"):
        val = meta.get(key)
        if val is not None and not pd.isna(val) and str(val).strip():
            raw = slugify_label(str(val).strip())
            return raw if raw.startswith("kaggle_") else f"kaggle_{raw}"

    for key in ("region", "REGION", "dialect", "DIALECT"):
        val = meta.get(key)
        if val is not None and not pd.isna(val) and str(val).strip():
            region_str = str(val).strip()
            upper = region_str.upper()
            if upper in suffix_map:
                return suffix_map[upper]
            return f"kaggle_signer_{slugify_label(region_str)}"

    if rel_path is not None:
        rel_posix = Path(rel_path).as_posix()
        token_match = _SIGNER_TOKEN_RE.search(rel_posix)
        if token_match:
            return f"kaggle_{slugify_label(token_match.group(1))}"

    stem = Path(video_name).stem.strip()
    token_match = _SIGNER_TOKEN_RE.search(stem)
    if token_match:
        return f"kaggle_{slugify_label(token_match.group(1))}"

    qipedc_match = _QIPEDC_STEM_RE.match(stem)
    if qipedc_match:
        prefix, _digits, suffix = qipedc_match.groups()
        if suffix:
            upper_suffix = suffix.upper()
            if upper_suffix in suffix_map:
                return suffix_map[upper_suffix]
            return f"kaggle_signer_{slugify_label(suffix)}"
        return f"{default_signer_prefix}_{prefix.lower()}"

    if rel_path is not None:
        parts = [p for p in Path(rel_path).parts[:-1] if p.lower() not in ("dataset", "videos", "video", ".")]
        if parts:
            return f"kaggle_signer_{slugify_label(parts[0])}"

    return default_signer_prefix


def load_vsl400_signer_map(
    dataset_root: str | Path,
    *,
    default_prefix: str = "vsl2_signer",
) -> dict[str, str]:
    """Build a ``video_id -> signer_id`` lookup from VSL400 ``front_view.json`` files if present."""
    root = Path(dataset_root)
    signer_map: dict[str, str] = {}
    if not root.exists():
        return signer_map

    for json_path in sorted(root.rglob("front_view.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            vid = entry.get("video_id") or entry.get("videoid")
            sid = entry.get("signer_id") or entry.get("signer")
            if vid is not None and sid is not None and str(vid).strip() and str(sid).strip():
                clean_vid = Path(str(vid).strip()).stem
                clean_sid = slugify_label(str(sid).strip())
                signer_id = clean_sid if clean_sid.startswith(default_prefix) else f"{default_prefix}_{clean_sid}"
                signer_map[clean_vid] = signer_id
    return signer_map


def infer_vsl_v2_signer_id(
    video_name: str,
    rel_path: str | Path | None = None,
    *,
    signer_map: dict[str, str] | None = None,
    split_hint: str | None = None,
    default_prefix: str = "vsl2_signer",
) -> str:
    """Deterministically infer ``signer_id`` for a ``nguyenanfms/vsl-vietnamese-sign-language-v2`` clip.

    Resolution priority:
    1. Exact ``video_id`` match in VSL400 ``front_view.json`` signer map (e.g., ``000000`` -> ``vsl2_signer_001``).
    2. Explicit signer token in relative path or filename (e.g., ``signer_02`` -> ``vsl2_signer_02``).
    3. Upstream VSL-v2 unseen-signer split bucket (``train`` -> ``vsl2_signer_train``,
       ``test`` -> ``vsl2_signer_test``) to preserve upstream signer-disjoint boundaries.
    4. Deterministic fallback ``<default_prefix>_unknown``.
    """
    stem = Path(video_name).stem.strip()
    if signer_map:
        if stem in signer_map:
            return signer_map[stem]
        base_stem = stem.split("_")[0]
        if base_stem in signer_map:
            return signer_map[base_stem]

    if rel_path is not None:
        token_match = _SIGNER_TOKEN_RE.search(Path(rel_path).as_posix())
        if token_match:
            tok = slugify_label(token_match.group(1))
            return tok if tok.startswith(default_prefix) else f"{default_prefix}_{tok}"

    token_match = _SIGNER_TOKEN_RE.search(stem)
    if token_match:
        tok = slugify_label(token_match.group(1))
        return tok if tok.startswith(default_prefix) else f"{default_prefix}_{tok}"

    if split_hint and str(split_hint).strip().lower() in _SPLIT_DIR_NAMES:
        norm_split = str(split_hint).strip().lower()
        if norm_split in ("valid", "validation"):
            norm_split = "val"
        return f"{default_prefix}_{norm_split}"

    return f"{default_prefix}_unknown"


def find_kaggle_label_csv(dataset_root: Path, preferred_rel_path: str = "Dataset/Labels/label.csv") -> Path | None:
    """Locate the label/metadata CSV inside a Kaggle VSL directory."""
    preferred = dataset_root / preferred_rel_path
    if preferred.exists() and preferred.is_file():
        return preferred

    candidates = sorted(dataset_root.rglob("*.csv"))
    for name_hint in ("label.csv", "labels.csv", "metadata.csv"):
        for cand in candidates:
            if cand.name.lower() == name_hint:
                return cand
    return candidates[0] if candidates else None


def _pick_column(columns: list[str], preferred: str, fallbacks: tuple[str, ...]) -> str | None:
    col_map = {c.strip().lower(): c for c in columns}
    if preferred.lower() in col_map:
        return col_map[preferred.lower()]
    for fb in fallbacks:
        if fb.lower() in col_map:
            return col_map[fb.lower()]
    return None


def _build_class_lookups(
    class_list: list[dict[str, Any]] | None,
) -> tuple[dict[str, tuple[int, str]], dict[str, tuple[int, str]]]:
    label_lookup: dict[str, tuple[int, str]] = {}
    slug_lookup: dict[str, tuple[int, str]] = {}
    if class_list:
        for entry in class_list:
            cid = int(entry["class_id"])
            canon = normalize_vietnamese_label(str(entry["label"]))
            low = canon.lower()
            slg = str(entry.get("slug") or slugify_label(canon)).strip().lower()
            label_lookup[low] = (cid, canon)
            slug_lookup[slg] = (cid, canon)
    return label_lookup, slug_lookup


def _resolve_canonical_class(
    raw_label: str,
    label_lookup: dict[str, tuple[int, str]],
    slug_lookup: dict[str, tuple[int, str]],
) -> tuple[int, str]:
    clean = normalize_vietnamese_label(raw_label)
    low = clean.lower()
    if low in label_lookup:
        return label_lookup[low]
    slg = slugify_label(clean)
    if slg in slug_lookup:
        return slug_lookup[slg]
    return -1, clean


def _build_video_index(dataset_root: Path, preferred_videos_rel_dir: str = "Dataset/Videos") -> dict[str, Path]:
    """Index video files by relative POSIX path and basename (case-insensitive) for fast lookup."""
    index: dict[str, Path] = {}
    videos_dir = dataset_root / preferred_videos_rel_dir
    search_roots = [videos_dir, dataset_root] if videos_dir.exists() else [dataset_root]

    seen: set[Path] = set()
    for root in search_roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and path not in seen:
                seen.add(path)
                rel_posix = path.relative_to(dataset_root).as_posix()
                index.setdefault(rel_posix, path)
                index.setdefault(rel_posix.lower(), path)
                index.setdefault(path.name, path)
                index.setdefault(path.name.lower(), path)
                index.setdefault(path.stem.lower(), path)
    return index


def _resolve_video_path(
    video_ref: str,
    dataset_root: Path,
    videos_rel_dir: str,
    video_index: dict[str, Path],
) -> Path:
    ref = str(video_ref).strip().replace("\\", "/")
    direct_candidates = [
        dataset_root / ref,
        dataset_root / videos_rel_dir / ref,
        dataset_root / videos_rel_dir / Path(ref).name,
    ]
    for cand in direct_candidates:
        if cand.exists() and cand.is_file():
            return cand

    for key in (ref, ref.lower(), Path(ref).name, Path(ref).name.lower(), Path(ref).stem.lower()):
        if key in video_index:
            return video_index[key]

    return dataset_root / videos_rel_dir / Path(ref).name


def parse_kaggle_dataset_to_manifest(
    dataset_root: str | Path,
    *,
    project_root: str | Path = ".",
    class_list: list[dict[str, Any]] | None = None,
    labels_rel_path: str = "Dataset/Labels/label.csv",
    videos_rel_dir: str = "Dataset/Videos",
    id_column: str = "ID",
    video_column: str = "VIDEO",
    label_column: str = "LABEL",
    region_suffix_map: dict[str, str] | None = None,
    default_signer_prefix: str = "kaggle_signer_common",
    probe_videos: bool = True,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Parse a downloaded QIPEDC Kaggle VSL directory into a standardized manifest DataFrame.

    Supports:
    1. Canonical QIPEDC layout (``Dataset/Labels/label.csv`` + ``Dataset/Videos/*.mp4``)
    2. Generic CSV metadata layouts with auto-detected column names
    3. Folder-per-class video directory fallback if no CSV file is present
    """
    root_path = Path(dataset_root).resolve()
    proj_root = Path(project_root).resolve()

    label_lookup, slug_lookup = _build_class_lookups(class_list)

    label_csv = find_kaggle_label_csv(root_path, preferred_rel_path=labels_rel_path)
    video_index = _build_video_index(root_path, preferred_videos_rel_dir=videos_rel_dir)
    rows: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()

    def _make_unique_sample_id(base_id: str) -> str:
        candidate = base_id
        counter = 1
        while candidate in seen_sample_ids:
            counter += 1
            candidate = f"{base_id}_{counter}"
        seen_sample_ids.add(candidate)
        return candidate

    if label_csv is not None:
        raw_df = pd.read_csv(label_csv, encoding="utf-8-sig")
        cols = list(raw_df.columns)
        vid_col = _pick_column(
            cols,
            video_column,
            ("video", "video_name", "filename", "file_name", "file_path", "filepath", "video_path"),
        )
        lbl_col = _pick_column(
            cols,
            label_column,
            ("label", "gloss", "word", "class", "class_name", "meaning"),
        )
        id_col = _pick_column(cols, id_column, ("id", "sample_id", "index"))

        if vid_col is None or lbl_col is None:
            raise ValueError(
                f"Could not detect video/label columns in {label_csv}. Found columns: {cols}"
            )

        iterator = raw_df.iterrows()
        if show_progress:
            iterator = tqdm(iterator, total=len(raw_df), desc="Indexing Kaggle VSL (QIPEDC)")

        for idx, rec in iterator:
            video_ref = str(rec[vid_col]).strip()
            raw_label = normalize_vietnamese_label(str(rec[lbl_col]))
            if not video_ref or not raw_label or raw_label.lower() == "nan":
                continue

            cid, canonical_label = _resolve_canonical_class(raw_label, label_lookup, slug_lookup)
            resolved_video = _resolve_video_path(video_ref, root_path, videos_rel_dir, video_index)
            rel_inside_dataset = (
                resolved_video.relative_to(root_path)
                if resolved_video.is_relative_to(root_path)
                else Path(video_ref)
            )
            signer_id = infer_kaggle_signer_id(
                Path(video_ref).name,
                rel_path=rel_inside_dataset,
                row_meta=rec.to_dict(),
                region_suffix_map=region_suffix_map,
                default_signer_prefix=default_signer_prefix,
            )

            raw_id = str(rec[id_col]).strip() if id_col and not pd.isna(rec[id_col]) else str(idx + 1)
            stem_slug = slugify_label(Path(video_ref).stem)
            sample_id = _make_unique_sample_id(f"kaggle_{raw_id}_{stem_slug}")

            if probe_videos:
                meta = probe_video_metadata(resolved_video)
            else:
                exists = resolved_video.exists() and resolved_video.is_file() and resolved_video.stat().st_size > 0
                meta = {
                    "fps": 0.0,
                    "num_frames": 0,
                    "duration_sec": 0.0,
                    "width": 0,
                    "height": 0,
                    "is_valid": bool(exists),
                }

            rows.append(
                {
                    "sample_id": sample_id,
                    "video_path": normalize_rel_path(resolved_video, proj_root),
                    "label": canonical_label,
                    "class_id": cid,
                    "signer_id": signer_id,
                    "source": "kaggle_vsl",
                    "split": "",
                    "is_augmented": False,
                    "parent_sample_id": "",
                    "aug_type": "",
                    "fps": meta["fps"],
                    "num_frames": meta["num_frames"],
                    "duration_sec": meta["duration_sec"],
                    "width": meta["width"],
                    "height": meta["height"],
                    "is_valid": meta["is_valid"],
                }
            )
    else:
        # Folder-per-class fallback: <dataset_root>/<class_label>/*.mp4
        video_files = sorted(
            p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        iterator = tqdm(video_files, desc="Scanning Kaggle VSL videos") if show_progress else video_files
        for idx, vp in enumerate(iterator, start=1):
            cid, canonical_label = _resolve_canonical_class(vp.parent.name, label_lookup, slug_lookup)
            rel_inside = vp.relative_to(root_path)
            signer_id = infer_kaggle_signer_id(
                vp.name,
                rel_path=rel_inside,
                region_suffix_map=region_suffix_map,
                default_signer_prefix=default_signer_prefix,
            )
            sample_id = _make_unique_sample_id(f"kaggle_{idx}_{slugify_label(vp.stem)}")
            meta = probe_video_metadata(vp) if probe_videos else {
                "fps": 0.0,
                "num_frames": 0,
                "duration_sec": 0.0,
                "width": 0,
                "height": 0,
                "is_valid": True,
            }
            rows.append(
                {
                    "sample_id": sample_id,
                    "video_path": normalize_rel_path(vp, proj_root),
                    "label": canonical_label,
                    "class_id": cid,
                    "signer_id": signer_id,
                    "source": "kaggle_vsl",
                    "split": "",
                    "is_augmented": False,
                    "parent_sample_id": "",
                    "aug_type": "",
                    "fps": meta["fps"],
                    "num_frames": meta["num_frames"],
                    "duration_sec": meta["duration_sec"],
                    "width": meta["width"],
                    "height": meta["height"],
                    "is_valid": meta["is_valid"],
                }
            )

    if not rows:
        return make_empty_manifest()
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def parse_kaggle_v2_dataset_to_manifest(
    dataset_root: str | Path,
    *,
    project_root: str | Path = ".",
    class_list: list[dict[str, Any]] | None = None,
    frame_splited_rel_dir: str = "processed/processed/frame_splited",
    raw_vsl400_rel_dir: str = "raw/raw/VSL400",
    source: str = "kaggle_vsl_v2",
    default_signer_prefix: str = "vsl2_signer",
    exclude_pre_augmented: bool = True,
    probe_videos: bool = True,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Parse ``nguyenanfms/vsl-vietnamese-sign-language-v2`` (or raw VSL400) into a manifest DataFrame.

    Auto-detects layout in priority order:
    1. Canonical cropped front-view folders:
       ``processed/processed/frame_splited/{train,val,test}/{gloss}/{videoid}.mp4``
       (also supports ``frame_splited/{train,test}/{gloss}/*.mp4``). Skips
       ``processed_augmented/`` when ``exclude_pre_augmented=True`` so pre-augmented
       videos never leak into Step 3 splitting.
    2. Raw VSL400 layout (``Part_*/split_*/front_view.json`` + ``front_view/{video_id}.mp4``)
       with explicit ``signer_id`` per clip.
    3. Generic ``{train,test}/{gloss}/*.mp4`` or ``{gloss}/*.mp4`` folder layout.

    Normalizes Vietnamese gloss labels via Unicode NFC and maps them against ``class_list``
    so labels like ``'Giúp đỡ'`` align cleanly with canonical ``'giúp đỡ'``.
    Leaves ``split=""`` unassigned so Step 3 (``split_dataset.py``) performs unified
    signer-disjoint splitting across all combined datasets.
    """
    root_path = Path(dataset_root).resolve()
    proj_root = Path(project_root).resolve()
    label_lookup, slug_lookup = _build_class_lookups(class_list)

    # Always scan for VSL400 front_view.json signer mappings if present anywhere under root
    signer_map = load_vsl400_signer_map(root_path, default_prefix=default_signer_prefix)

    rows: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()

    def _make_unique_sample_id(base_id: str) -> str:
        candidate = base_id
        counter = 1
        while candidate in seen_sample_ids:
            counter += 1
            candidate = f"{base_id}_{counter}"
        seen_sample_ids.add(candidate)
        return candidate

    def _probe(vp: Path) -> dict[str, Any]:
        if probe_videos:
            return probe_video_metadata(vp)
        exists = vp.exists() and vp.is_file() and vp.stat().st_size > 0
        return {
            "fps": 0.0,
            "num_frames": 0,
            "duration_sec": 0.0,
            "width": 0,
            "height": 0,
            "is_valid": bool(exists),
        }

    # Mode 1: Preferred canonical `frame_splited` directory
    frame_candidates = [
        root_path / frame_splited_rel_dir,
        root_path / "frame_splited",
    ]
    frame_dir = next((d for d in frame_candidates if d.exists() and d.is_dir()), None)

    if frame_dir is not None:
        video_files = sorted(
            p for p in frame_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        iterator = (
            tqdm(video_files, desc="Indexing Kaggle VSL v2 (frame_splited)")
            if show_progress
            else video_files
        )
        for vp in iterator:
            rel_from_frame = vp.relative_to(frame_dir)
            parts = rel_from_frame.parts
            if len(parts) < 2:
                continue
            # Expected: <split>/<gloss>/<videoid>.mp4 or <gloss>/<videoid>.mp4
            if parts[0].lower() in _SPLIT_DIR_NAMES and len(parts) >= 3:
                split_hint = parts[0].lower()
                raw_gloss = parts[1]
            else:
                split_hint = None
                raw_gloss = parts[-2]

            cid, canonical_label = _resolve_canonical_class(raw_gloss, label_lookup, slug_lookup)
            signer_id = infer_vsl_v2_signer_id(
                vp.name,
                rel_path=rel_from_frame,
                signer_map=signer_map,
                split_hint=split_hint,
                default_prefix=default_signer_prefix,
            )
            stem_slug = slugify_label(vp.stem)
            gloss_slug = slugify_label(canonical_label)
            sample_id = _make_unique_sample_id(f"vsl2_{gloss_slug}_{stem_slug}")
            meta = _probe(vp)
            rows.append(
                {
                    "sample_id": sample_id,
                    "video_path": normalize_rel_path(vp, proj_root),
                    "label": canonical_label,
                    "class_id": cid,
                    "signer_id": signer_id,
                    "source": source,
                    "split": "",
                    "is_augmented": False,
                    "parent_sample_id": "",
                    "aug_type": "",
                    "fps": meta["fps"],
                    "num_frames": meta["num_frames"],
                    "duration_sec": meta["duration_sec"],
                    "width": meta["width"],
                    "height": meta["height"],
                    "is_valid": meta["is_valid"],
                }
            )
        return pd.DataFrame(rows, columns=MANIFEST_COLUMNS) if rows else make_empty_manifest()

    # Mode 2: Raw VSL400 (`front_view.json` + `front_view/{video_id}.mp4`)
    vsl400_root = root_path / raw_vsl400_rel_dir
    search_vsl400 = vsl400_root if vsl400_root.exists() else root_path
    front_jsons = sorted(search_vsl400.rglob("front_view.json"))
    if front_jsons:
        entries_with_dir: list[tuple[Path, dict[str, Any]]] = []
        for jp in front_jsons:
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        entries_with_dir.append((jp.parent, item))

        iterator = (
            tqdm(entries_with_dir, desc="Indexing VSL400 front_view.json")
            if show_progress
            else entries_with_dir
        )
        for base_dir, item in iterator:
            vid_id = str(item.get("video_id") or item.get("videoid") or "").strip()
            raw_gloss = str(item.get("gloss") or item.get("label") or "").strip()
            if not vid_id or not raw_gloss:
                continue
            fname = vid_id if Path(vid_id).suffix else f"{vid_id}.mp4"
            vp = base_dir / "front_view" / fname
            if not vp.exists():
                vp = base_dir / fname

            cid, canonical_label = _resolve_canonical_class(raw_gloss, label_lookup, slug_lookup)
            raw_sid = item.get("signer_id") or item.get("signer")
            if raw_sid is not None and str(raw_sid).strip():
                clean_sid = slugify_label(str(raw_sid).strip())
                signer_id = (
                    clean_sid if clean_sid.startswith(default_signer_prefix) else f"{default_signer_prefix}_{clean_sid}"
                )
            else:
                signer_id = infer_vsl_v2_signer_id(
                    fname,
                    rel_path=vp.relative_to(root_path) if vp.is_relative_to(root_path) else Path(fname),
                    signer_map=signer_map,
                    default_prefix=default_signer_prefix,
                )

            sample_id = _make_unique_sample_id(
                f"vsl2_{slugify_label(canonical_label)}_{slugify_label(Path(fname).stem)}"
            )
            meta = _probe(vp)
            rows.append(
                {
                    "sample_id": sample_id,
                    "video_path": normalize_rel_path(vp, proj_root),
                    "label": canonical_label,
                    "class_id": cid,
                    "signer_id": signer_id,
                    "source": source,
                    "split": "",
                    "is_augmented": False,
                    "parent_sample_id": "",
                    "aug_type": "",
                    "fps": meta["fps"],
                    "num_frames": meta["num_frames"],
                    "duration_sec": meta["duration_sec"],
                    "width": meta["width"],
                    "height": meta["height"],
                    "is_valid": meta["is_valid"],
                }
            )
        return pd.DataFrame(rows, columns=MANIFEST_COLUMNS) if rows else make_empty_manifest()

    # Mode 3: Generic folder-per-class fallback (skipping processed_augmented if requested)
    video_files = []
    for p in sorted(root_path.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS):
            continue
        rel_parts_low = [part.lower() for part in p.relative_to(root_path).parts]
        if exclude_pre_augmented and any("augmented" in part for part in rel_parts_low):
            continue
        video_files.append(p)

    iterator = tqdm(video_files, desc="Scanning Kaggle VSL v2 videos") if show_progress else video_files
    for vp in iterator:
        rel_inside = vp.relative_to(root_path)
        parts = rel_inside.parts
        split_hint = next((p.lower() for p in parts[:-1] if p.lower() in _SPLIT_DIR_NAMES), None)
        raw_gloss = vp.parent.name
        cid, canonical_label = _resolve_canonical_class(raw_gloss, label_lookup, slug_lookup)
        signer_id = infer_vsl_v2_signer_id(
            vp.name,
            rel_path=rel_inside,
            signer_map=signer_map,
            split_hint=split_hint,
            default_prefix=default_signer_prefix,
        )
        sample_id = _make_unique_sample_id(f"vsl2_{slugify_label(canonical_label)}_{slugify_label(vp.stem)}")
        meta = _probe(vp)
        rows.append(
            {
                "sample_id": sample_id,
                "video_path": normalize_rel_path(vp, proj_root),
                "label": canonical_label,
                "class_id": cid,
                "signer_id": signer_id,
                "source": source,
                "split": "",
                "is_augmented": False,
                "parent_sample_id": "",
                "aug_type": "",
                "fps": meta["fps"],
                "num_frames": meta["num_frames"],
                "duration_sec": meta["duration_sec"],
                "width": meta["width"],
                "height": meta["height"],
                "is_valid": meta["is_valid"],
            }
        )

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS) if rows else make_empty_manifest()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download & parse public Kaggle VSL datasets "
            "(qipedc: aresusayhi/vsl-vietnamese-sign-languages; "
            "vsl_v2: nguyenanfms/vsl-vietnamese-sign-language-v2) into manifest CSVs."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data_config.yaml",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("qipedc", "vsl_v2", "all"),
        default="all",
        help="Which Kaggle VSL dataset to download/index (default: all = qipedc + vsl_v2).",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default=None,
        help="Override Kaggle dataset slug for the selected single dataset.",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Parse an already-downloaded dataset directory without hitting Kaggle API.",
    )
    parser.add_argument(
        "--local-v2-dir",
        type=str,
        default=None,
        help="Local directory for vsl_v2 when running with --dataset all.",
    )
    parser.add_argument(
        "--copy-to-raw",
        action="store_true",
        help="Copy downloaded Kaggle files into raw directory from config.",
    )
    parser.add_argument(
        "--only-labels",
        action="store_true",
        help="Download only tiny metadata CSV/TSV (label.csv or split_manifest.tsv) for inspection.",
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        default=None,
        help="Override output CSV path (single-dataset mode).",
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="Skip OpenCV video probing (faster metadata-only indexing).",
    )
    return parser


def _run_qipedc(args: argparse.Namespace, cfg: dict[str, Any], project_root: Path, class_list: list[dict[str, Any]] | None) -> int:
    paths_cfg = cfg.get("paths", {})
    kaggle_cfg = cfg.get("kaggle", {})

    dataset_slug = args.dataset_slug or kaggle_cfg.get(
        "dataset_slug", "aresusayhi/vsl-vietnamese-sign-languages"
    )
    labels_rel_path = kaggle_cfg.get("labels_rel_path", "Dataset/Labels/label.csv")
    videos_rel_dir = kaggle_cfg.get("videos_rel_dir", "Dataset/Videos")
    raw_kaggle_dir = project_root / paths_cfg.get("raw_kaggle_dir", "data/raw/kaggle_vsl")
    output_manifest = (
        Path(args.output_manifest)
        if args.output_manifest and args.dataset == "qipedc"
        else project_root / paths_cfg.get("kaggle_manifest", "data/manifests/kaggle_manifest.csv")
    )

    if args.only_labels:
        csv_path = download_kaggle_dataset(
            dataset_slug=dataset_slug,
            target_dir=raw_kaggle_dir,
            copy_to_target=args.copy_to_raw,
            only_labels=True,
            labels_rel_path=labels_rel_path,
        )
        df_labels = pd.read_csv(csv_path)
        print(f"[qipedc] Downloaded label CSV: {csv_path}")
        print(f"[qipedc] Rows: {len(df_labels)}, Columns: {list(df_labels.columns)}")
        if "LABEL" in df_labels.columns:
            print(f"[qipedc] Unique labels: {df_labels['LABEL'].nunique()}")
        return 0

    if args.local_dir:
        dataset_dir = Path(args.local_dir).resolve()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Local Kaggle directory does not exist: {dataset_dir}")
    else:
        dataset_dir = download_kaggle_dataset(
            dataset_slug=dataset_slug,
            target_dir=raw_kaggle_dir,
            copy_to_target=args.copy_to_raw,
            only_labels=False,
            labels_rel_path=labels_rel_path,
        )

    manifest_df = parse_kaggle_dataset_to_manifest(
        dataset_dir,
        project_root=project_root,
        class_list=class_list,
        labels_rel_path=labels_rel_path,
        videos_rel_dir=videos_rel_dir,
        id_column=kaggle_cfg.get("id_column", "ID"),
        video_column=kaggle_cfg.get("video_column", "VIDEO"),
        label_column=kaggle_cfg.get("label_column", "LABEL"),
        region_suffix_map=kaggle_cfg.get("region_suffix_map"),
        default_signer_prefix=kaggle_cfg.get("default_signer_prefix", "kaggle_signer_common"),
        probe_videos=not args.skip_probe,
        show_progress=True,
    )

    saved = save_manifest_df(manifest_df, output_manifest, validate=True)
    valid_count = int(manifest_df["is_valid"].astype(bool).sum()) if not manifest_df.empty else 0
    print(
        f"[qipedc] Saved manifest to {saved} "
        f"({len(manifest_df)} rows, {valid_count} valid videos, "
        f"{manifest_df['label'].nunique() if not manifest_df.empty else 0} unique labels, "
        f"{manifest_df['signer_id'].nunique() if not manifest_df.empty else 0} unique signers)."
    )
    return 0


def _run_vsl_v2(args: argparse.Namespace, cfg: dict[str, Any], project_root: Path, class_list: list[dict[str, Any]] | None) -> int:
    paths_cfg = cfg.get("paths", {})
    v2_cfg = cfg.get("kaggle_v2", {})

    dataset_slug = (
        args.dataset_slug
        if args.dataset_slug and args.dataset == "vsl_v2"
        else v2_cfg.get("dataset_slug", "nguyenanfms/vsl-vietnamese-sign-language-v2")
    )
    split_manifest_rel = v2_cfg.get(
        "split_manifest_rel_path", "processed/processed/frame_splited/split_manifest.tsv"
    )
    frame_splited_rel_dir = v2_cfg.get(
        "frame_splited_rel_dir", "processed/processed/frame_splited"
    )
    raw_vsl400_rel_dir = v2_cfg.get("raw_vsl400_rel_dir", "raw/raw/VSL400")
    raw_v2_dir = project_root / paths_cfg.get("raw_kaggle_v2_dir", "data/raw/kaggle_vsl_v2")
    output_manifest = (
        Path(args.output_manifest)
        if args.output_manifest and args.dataset == "vsl_v2"
        else project_root / paths_cfg.get("kaggle_v2_manifest", "data/manifests/kaggle_v2_manifest.csv")
    )

    if args.only_labels:
        tsv_path = download_kaggle_dataset(
            dataset_slug=dataset_slug,
            target_dir=raw_v2_dir,
            copy_to_target=args.copy_to_raw,
            only_labels=True,
            labels_rel_path=split_manifest_rel,
        )
        df_meta = pd.read_csv(tsv_path, sep="\t")
        print(f"[vsl_v2] Downloaded split manifest TSV: {tsv_path}")
        print(
            f"[vsl_v2] Glosses: {len(df_meta)}, Total canonical clips: "
            f"{int(df_meta['total'].sum()) if 'total' in df_meta.columns else 'N/A'}"
        )
        return 0

    local_override = args.local_v2_dir or (args.local_dir if args.dataset == "vsl_v2" else None)
    if local_override:
        dataset_dir = Path(local_override).resolve()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Local VSL v2 directory does not exist: {dataset_dir}")
    else:
        dataset_dir = download_kaggle_dataset(
            dataset_slug=dataset_slug,
            target_dir=raw_v2_dir,
            copy_to_target=args.copy_to_raw,
            only_labels=False,
            labels_rel_path=split_manifest_rel,
        )

    manifest_df = parse_kaggle_v2_dataset_to_manifest(
        dataset_dir,
        project_root=project_root,
        class_list=class_list,
        frame_splited_rel_dir=frame_splited_rel_dir,
        raw_vsl400_rel_dir=raw_vsl400_rel_dir,
        source=v2_cfg.get("source_name", "kaggle_vsl_v2"),
        default_signer_prefix=v2_cfg.get("default_signer_prefix", "vsl2_signer"),
        exclude_pre_augmented=bool(v2_cfg.get("exclude_pre_augmented", True)),
        probe_videos=not args.skip_probe,
        show_progress=True,
    )

    saved = save_manifest_df(manifest_df, output_manifest, validate=True)
    valid_count = int(manifest_df["is_valid"].astype(bool).sum()) if not manifest_df.empty else 0
    print(
        f"[vsl_v2] Saved manifest to {saved} "
        f"({len(manifest_df)} rows, {valid_count} valid videos, "
        f"{manifest_df['label'].nunique() if not manifest_df.empty else 0} unique labels, "
        f"{manifest_df['signer_id'].nunique() if not manifest_df.empty else 0} unique signers)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    project_root = Path(cfg.get("project", {}).get("root_dir", ".")).resolve()
    paths_cfg = cfg.get("paths", {})
    class_list_path = project_root / paths_cfg.get("class_list_path", "configs/class_list.json")
    class_list = load_class_list(class_list_path) if class_list_path.exists() else None

    if args.dataset == "qipedc":
        return _run_qipedc(args, cfg, project_root, class_list)
    if args.dataset == "vsl_v2":
        return _run_vsl_v2(args, cfg, project_root, class_list)
    # "all"
    _run_qipedc(args, cfg, project_root, class_list)
    return _run_vsl_v2(args, cfg, project_root, class_list)


if __name__ == "__main__":
    sys.exit(main())

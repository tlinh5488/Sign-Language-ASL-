"""End-to-end production CLI runner for Quy's VSL Data Pipeline (Phần 1: Dữ liệu).

Orchestrates all 4 stages strictly on real project data in ``data/manifests/``
and ``data/splits/``:
    - (Optional) Dual-dataset Kaggle ingestion (`src.data.download_kaggle`):
        1. QIPEDC (`aresusayhi/vsl-vietnamese-sign-languages`, source=`kaggle_vsl`)
        2. VSL-v2 Cropped / VSL400 (`nguyenanfms/vsl-vietnamese-sign-language-v2`, source=`kaggle_vsl_v2`)
    - Stage 2: Manifest merge & data quality audit (`src.data.build_manifest`)
    - Stage 1: Scope selection (`src.data.scope`)
    - Stage 3: Strict signer-disjoint train / val / test split (`src.data.split_dataset`)
    - Stage 4: Train-only sign-language-safe video augmentation (`src.data.augment`)

Defaults to Phương án 1 (``scope.kaggle_only: true`` in ``configs/data_config.yaml``):
100% public Kaggle datasets (``kaggle_vsl`` + ``kaggle_vsl_v2``) with zero self-recorded
videos. Pass ``--hybrid`` / ``--include-self`` to opt into hybrid mode with self-recorded
videos.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is importable when invoked as `python scripts/run_quy_pipeline.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.augment import augment_train_manifest
from src.data.build_manifest import audit_manifest, merge_manifests
from src.data.download_kaggle import (
    download_kaggle_dataset,
    parse_kaggle_dataset_to_manifest,
    parse_kaggle_v2_dataset_to_manifest,
)
from src.data.schema import (
    load_class_list,
    load_config,
    save_manifest_df,
)
from src.data.scope import run_scope
from src.data.split_dataset import split_dataset

logger = logging.getLogger(__name__)


def run_production_pipeline(
    *,
    config_path: Path,
    download_kaggle: bool = False,
    download_kaggle_v2: bool = False,
    kaggle_local_dir: Path | None = None,
    kaggle_v2_local_dir: Path | None = None,
    scope_mode: str | None = None,
    skip_augment: bool = False,
    num_aug: int | None = None,
    seed: int | None = None,
    allow_hflip: bool | None = None,
    allow_2way: bool = False,
    fallback_warning: bool = False,
    kaggle_only: bool | None = None,
) -> dict[str, Any]:
    """Run Quy's 4-stage data pipeline on real project data configured in data_config.yaml.

    When ``kaggle_only`` is ``None``, defaults to ``scope.kaggle_only`` from
    ``configs/data_config.yaml`` (Phương án 1: Dual Kaggle-only by default).
    """
    cfg = load_config(config_path)
    project_root = Path(cfg.get("project", {}).get("root_dir", ".")).resolve()
    paths_cfg = cfg.get("paths", {})
    kaggle_cfg = cfg.get("kaggle", {})
    kaggle_v2_cfg = cfg.get("kaggle_v2", {})
    scope_cfg = cfg.get("scope", {})

    kaggle_only = (
        bool(scope_cfg.get("kaggle_only", True))
        if kaggle_only is None
        else bool(kaggle_only)
    )

    raw_kaggle_dir = project_root / paths_cfg.get("raw_kaggle_dir", "data/raw/kaggle_vsl")
    raw_kaggle_v2_dir = project_root / paths_cfg.get("raw_kaggle_v2_dir", "data/raw/kaggle_vsl_v2")
    raw_self_dir = project_root / paths_cfg.get("raw_self_dir", "data/raw/self_recorded")
    kaggle_manifest = project_root / paths_cfg.get("kaggle_manifest", "data/manifests/kaggle_manifest.csv")
    kaggle_v2_manifest = project_root / paths_cfg.get(
        "kaggle_v2_manifest", "data/manifests/kaggle_v2_manifest.csv"
    )
    self_manifest = project_root / paths_cfg.get("self_manifest", "data/manifests/self_manifest.csv")
    dataset_manifest = project_root / paths_cfg.get("dataset_manifest", "data/manifests/dataset_manifest.csv")
    quality_report_path = project_root / paths_cfg.get("quality_report", "data/manifests/quality_report.json")
    scoped_manifest = project_root / paths_cfg.get("scoped_manifest", "data/manifests/scoped_manifest.csv")
    scope_report_path = project_root / paths_cfg.get("scope_report", "data/manifests/scope_report.json")
    splits_dir = project_root / paths_cfg.get("splits_dir", "data/splits")
    aug_dir = project_root / paths_cfg.get("augmented_train_dir", "data/augmented/train")
    class_list_path = project_root / paths_cfg.get("class_list_path", "configs/class_list.json")

    class_list = load_class_list(class_list_path) if class_list_path.exists() else None

    # Optional Kaggle Dataset 1 (QIPEDC) download / indexing
    if download_kaggle or kaggle_local_dir is not None:
        print("[Kaggle 1/2] Indexing QIPEDC VSL dataset (aresusayhi/vsl-vietnamese-sign-languages)...")
        if kaggle_local_dir is not None:
            ds_dir = kaggle_local_dir.resolve()
        else:
            ds_dir = download_kaggle_dataset(
                dataset_slug=kaggle_cfg.get("dataset_slug", "aresusayhi/vsl-vietnamese-sign-languages"),
                target_dir=raw_kaggle_dir,
                copy_to_target=False,
                only_labels=False,
                labels_rel_path=kaggle_cfg.get("labels_rel_path", "Dataset/Labels/label.csv"),
            )
        kaggle_df = parse_kaggle_dataset_to_manifest(
            ds_dir,
            project_root=project_root,
            class_list=class_list,
            labels_rel_path=kaggle_cfg.get("labels_rel_path", "Dataset/Labels/label.csv"),
            videos_rel_dir=kaggle_cfg.get("videos_rel_dir", "Dataset/Videos"),
            id_column=kaggle_cfg.get("id_column", "ID"),
            video_column=kaggle_cfg.get("video_column", "VIDEO"),
            label_column=kaggle_cfg.get("label_column", "LABEL"),
            region_suffix_map=kaggle_cfg.get("region_suffix_map"),
            default_signer_prefix=kaggle_cfg.get("default_signer_prefix", "kaggle_signer_common"),
            probe_videos=True,
            show_progress=True,
        )
        save_manifest_df(kaggle_df, kaggle_manifest, validate=True)
        print(f"  -> Saved QIPEDC manifest ({len(kaggle_df)} rows): {kaggle_manifest}")

    # Optional Kaggle Dataset 2 (VSL-v2 Cropped / VSL400) download / indexing
    if download_kaggle_v2 or kaggle_v2_local_dir is not None:
        print(
            "[Kaggle 2/2] Indexing VSL-v2 Cropped dataset "
            "(nguyenanfms/vsl-vietnamese-sign-language-v2)..."
        )
        if kaggle_v2_local_dir is not None:
            ds2_dir = kaggle_v2_local_dir.resolve()
        else:
            ds2_dir = download_kaggle_dataset(
                dataset_slug=kaggle_v2_cfg.get(
                    "dataset_slug", "nguyenanfms/vsl-vietnamese-sign-language-v2"
                ),
                target_dir=raw_kaggle_v2_dir,
                copy_to_target=False,
                only_labels=False,
                labels_rel_path=kaggle_v2_cfg.get(
                    "split_manifest_rel_path", "processed/processed/frame_splited/split_manifest.tsv"
                ),
            )
        kaggle_v2_df = parse_kaggle_v2_dataset_to_manifest(
            ds2_dir,
            project_root=project_root,
            class_list=class_list,
            frame_splited_rel_dir=kaggle_v2_cfg.get(
                "frame_splited_rel_dir", "processed/processed/frame_splited"
            ),
            raw_vsl400_rel_dir=kaggle_v2_cfg.get("raw_vsl400_rel_dir", "raw/raw/VSL400"),
            source=kaggle_v2_cfg.get("source_name", "kaggle_vsl_v2"),
            default_signer_prefix=kaggle_v2_cfg.get("default_signer_prefix", "vsl2_signer"),
            exclude_pre_augmented=bool(kaggle_v2_cfg.get("exclude_pre_augmented", True)),
            probe_videos=True,
            show_progress=True,
        )
        save_manifest_df(kaggle_v2_df, kaggle_v2_manifest, validate=True)
        print(f"  -> Saved VSL-v2 manifest ({len(kaggle_v2_df)} rows): {kaggle_v2_manifest}")

    # Stage 2: Merge manifests & audit
    print(
        f"[Stage 2/4] Merging manifests & auditing video quality "
        f"({'kaggle_only' if kaggle_only else 'hybrid'})..."
    )
    if kaggle_only:
        manifest_inputs = [p for p in (kaggle_manifest, kaggle_v2_manifest) if p.exists()]
        raw_dirs: list[tuple[Path, str]] = []
        if not kaggle_manifest.exists() and raw_kaggle_dir.exists():
            raw_dirs.append((raw_kaggle_dir, "kaggle_vsl"))
        if not kaggle_v2_manifest.exists() and raw_kaggle_v2_dir.exists():
            raw_dirs.append((raw_kaggle_v2_dir, "kaggle_vsl_v2"))
    else:
        manifest_inputs = [
            p for p in (kaggle_manifest, kaggle_v2_manifest, self_manifest) if p.exists()
        ]
        raw_dirs = []
        if not self_manifest.exists() and raw_self_dir.exists():
            raw_dirs.append((raw_self_dir, "self_recorded"))
        if not kaggle_manifest.exists() and raw_kaggle_dir.exists():
            raw_dirs.append((raw_kaggle_dir, "kaggle_vsl"))
        if not kaggle_v2_manifest.exists() and raw_kaggle_v2_dir.exists():
            raw_dirs.append((raw_kaggle_v2_dir, "kaggle_vsl_v2"))

    if not manifest_inputs and not raw_dirs:
        hint = (
            "Run with --download-all-kaggle or --kaggle-local-dir <DIR> / --kaggle-v2-local-dir <DIR>."
            if kaggle_only
            else "Run with --download-all-kaggle or record videos with `python -m src.data.collect_videos`."
        )
        raise FileNotFoundError(f"No input manifests or raw video directories found! {hint}")

    merged_df = merge_manifests(
        manifest_paths=manifest_inputs,
        raw_dirs=raw_dirs,
        output_path=dataset_manifest,
        root_dir=project_root,
        class_list_path=class_list_path if class_list_path.exists() else None,
        probe_videos=True,
        kaggle_only=kaggle_only,
    )
    quality_report = audit_manifest(
        merged_df,
        report_path=quality_report_path,
        root_dir=project_root,
        reprobe=False,
        kaggle_only=kaggle_only,
    )

    # Stage 1: Scope selection
    print("[Stage 1/4] Filtering dataset by configured scope...")
    eff_mode = scope_mode or scope_cfg.get("mode", "class_list")
    cfg_min_samples = int(scope_cfg.get("min_samples_per_class", 1))
    cfg_min_signers = int(scope_cfg.get("min_signers_per_class", 1))
    min_samples = max(3, cfg_min_samples) if kaggle_only else cfg_min_samples
    min_signers = max(3, cfg_min_signers) if kaggle_only else cfg_min_signers

    _, scope_report = run_scope(
        input_manifest=dataset_manifest,
        output_manifest=scoped_manifest,
        report_path=scope_report_path,
        class_list_path=class_list_path if eff_mode == "class_list" else None,
        mode=eff_mode,
        top_k=scope_cfg.get("top_k", 30),
        min_samples=min_samples,
        min_signers=min_signers,
        case_insensitive=bool(scope_cfg.get("case_insensitive_match", True)),
        normalize_whitespace=bool(scope_cfg.get("normalize_whitespace", True)),
        only_valid=bool(scope_cfg.get("only_valid_videos", True)),
        kaggle_only=kaggle_only,
    )

    # Stage 3: Signer-disjoint split
    print("[Stage 3/4] Running strict signer-disjoint train/val/test split...")
    split_report = split_dataset(
        manifest_path=scoped_manifest,
        output_dir=splits_dir,
        random_seed=seed,
        config_path=config_path,
        allow_2way=allow_2way,
        fallback_warning=fallback_warning,
        kaggle_only=kaggle_only,
    )

    # Stage 4: Train-only augmentation
    augment_report: dict[str, Any] | None = None
    if skip_augment or not bool(cfg.get("augmentation", {}).get("enabled", True)):
        print("[Stage 4/4] Train-only augmentation skipped.")
    else:
        print("[Stage 4/4] Running train-only video augmentation...")
        augment_report = augment_train_manifest(
            train_manifest_path=splits_dir / "train.csv",
            output_manifest_path=splits_dir / "train_augmented.csv",
            output_video_dir=aug_dir,
            num_augmentations_per_video=num_aug,
            random_seed=seed,
            allow_hflip=allow_hflip,
            config_path=config_path,
            class_list_path=class_list_path,
            project_root=project_root,
            kaggle_only=kaggle_only,
        )

    summary: dict[str, Any] = {
        "mode": "production_kaggle_only" if kaggle_only else "production",
        "kaggle_only": bool(kaggle_only),
        "quality_report": quality_report,
        "scope_report": scope_report,
        "split_report": split_report,
        "augment_report": augment_report,
    }
    summary_path = splits_dir.parent / "pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end production runner for Quy's 4-stage VSL Data Pipeline (Phần 1: Dữ liệu)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "data_config.yaml",
        help="Path to central YAML configuration (default: configs/data_config.yaml).",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--kaggle-only",
        dest="kaggle_only",
        action="store_true",
        default=None,
        help=(
            "Force Phương án 1 (DEFAULT via configs/data_config.yaml): use ONLY Kaggle VSL "
            "datasets (kaggle_vsl + kaggle_vsl_v2) with ZERO self-recorded videos."
        ),
    )
    source_group.add_argument(
        "--hybrid",
        "--include-self",
        dest="kaggle_only",
        action="store_false",
        help=(
            "Opt into hybrid mode including self-recorded videos (`self_manifest.csv` / "
            "`data/raw/self_recorded`) alongside Kaggle datasets."
        ),
    )
    parser.add_argument(
        "--download-kaggle",
        action="store_true",
        help="Download & index primary Kaggle QIPEDC dataset (aresusayhi/vsl-vietnamese-sign-languages).",
    )
    parser.add_argument(
        "--download-kaggle-v2",
        action="store_true",
        help="Download & index secondary Kaggle VSL-v2 dataset (nguyenanfms/vsl-vietnamese-sign-language-v2).",
    )
    parser.add_argument(
        "--download-all-kaggle",
        action="store_true",
        help="Download & index BOTH Kaggle VSL datasets (QIPEDC + VSL-v2 Cropped).",
    )
    parser.add_argument(
        "--kaggle-local-dir",
        type=Path,
        default=None,
        help="Optional already-downloaded QIPEDC Kaggle directory to index without hitting Kaggle API.",
    )
    parser.add_argument(
        "--kaggle-v2-local-dir",
        type=Path,
        default=None,
        help="Optional already-downloaded VSL-v2 Kaggle directory to index without hitting Kaggle API.",
    )
    parser.add_argument(
        "--scope-mode",
        type=str,
        choices=("class_list", "top_k", "min_samples", "all"),
        default=None,
        help="Override Step 1 scope filtering mode.",
    )
    parser.add_argument(
        "--skip-augment",
        action="store_true",
        help="Skip Step 4 train-only augmentation.",
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=None,
        help="Override number of augmented copies per original train video (default: 2, or 4 in --kaggle-only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for deterministic splitting & augmentation.",
    )
    parser.add_argument(
        "--allow-hflip",
        action="store_true",
        default=False,
        help="Enable horizontal flip during train augmentation (DEFAULT: False for VSL handedness safety).",
    )
    parser.add_argument(
        "--allow-2way",
        action="store_true",
        help="Allow 2-way (train/val) split if only 2 distinct signers exist.",
    )
    parser.add_argument(
        "--fallback-warning",
        action="store_true",
        help="Allow fallback split with warnings if fewer than 3 distinct signers exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_arg_parser().parse_args(argv)

    run_production_pipeline(
        config_path=args.config,
        download_kaggle=bool(args.download_kaggle or args.download_all_kaggle),
        download_kaggle_v2=bool(args.download_kaggle_v2 or args.download_all_kaggle),
        kaggle_local_dir=args.kaggle_local_dir,
        kaggle_v2_local_dir=args.kaggle_v2_local_dir,
        scope_mode=args.scope_mode,
        skip_augment=args.skip_augment,
        num_aug=args.num_aug,
        seed=args.seed,
        allow_hflip=True if args.allow_hflip else None,
        allow_2way=args.allow_2way,
        fallback_warning=args.fallback_warning,
        kaggle_only=args.kaggle_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

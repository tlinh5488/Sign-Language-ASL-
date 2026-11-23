"""Step 1 (Chọn phạm vi): Scope selection and class_id assignment for Quy's VSL pipeline.

Filters an input manifest CSV by either:
- Curated ``configs/class_list.json`` (mode = ``"class_list"``), OR
- Dynamic selection from dataset labels:
  - ``"top_k"``: keep top-K most frequent labels meeting minimum sample/signer criteria
  - ``"min_samples"``: keep all labels with >= min_samples and >= min_signers
  - ``"all"``: keep all valid labels and assign deterministic class_ids

Outputs:
- ``data/manifests/scoped_manifest.csv`` (canonical schema with updated ``class_id``)
- ``data/manifests/scope_report.json`` (coverage, missing classes, per-class stats, warnings)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schema import (
    MANIFEST_COLUMNS,
    is_kaggle_source,
    load_class_list,
    load_config,
    load_manifest_df,
    make_empty_manifest,
    normalize_vietnamese_label,
    save_manifest_df,
    slugify_label,
    validate_manifest_df,
)

VALID_SCOPE_MODES = ("class_list", "top_k", "min_samples", "all")


def _normalize_label(label: str, *, case_insensitive: bool = True, normalize_whitespace: bool = True) -> str:
    text = normalize_vietnamese_label(label) if normalize_whitespace else str(label).strip()
    if case_insensitive:
        text = text.lower()
    return text


def select_scope(
    manifest_df: pd.DataFrame,
    *,
    class_list: list[dict[str, Any]] | None = None,
    mode: str = "class_list",
    top_k: int | None = 30,
    min_samples: int = 1,
    min_signers: int = 1,
    kaggle_only: bool = False,
    case_insensitive: bool = True,
    normalize_whitespace: bool = True,
    only_valid: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Filter a manifest DataFrame according to scope criteria and assign canonical ``class_id``.

    When ``kaggle_only=True``, restricts rows strictly to ``source == "kaggle_vsl"``
    and enforces ``min_samples >= 3`` and ``min_signers >= 3`` so every kept class
    can cover all 3 splits (``train``, ``val``, ``test``) under signer-disjoint splitting.

    Returns:
        Tuple of ``(scoped_df, scope_report, active_class_list)``.
    """
    if mode not in VALID_SCOPE_MODES:
        raise ValueError(f"Unsupported scope mode {mode!r}. Allowed: {VALID_SCOPE_MODES}")

    validate_manifest_df(manifest_df)
    input_total = int(len(manifest_df))

    effective_min_samples = max(int(min_samples), 3) if kaggle_only else int(min_samples)
    effective_min_signers = max(int(min_signers), 3) if kaggle_only else int(min_signers)

    if manifest_df.empty:
        empty_report = {
            "mode": mode,
            "kaggle_only": bool(kaggle_only),
            "effective_min_samples": effective_min_samples,
            "effective_min_signers": effective_min_signers,
            "input_samples": 0,
            "output_samples": 0,
            "dropped_non_kaggle_samples": 0,
            "dropped_invalid_samples": 0,
            "dropped_out_of_scope_samples": 0,
            "num_classes_configured": len(class_list) if class_list else 0,
            "num_classes_present": 0,
            "missing_classes": [c["label"] for c in class_list] if class_list else [],
            "num_unique_signers": 0,
            "classes": [dict(c) for c in class_list] if class_list else [],
            "per_class_stats": [],
            "warnings": ["Input manifest is empty."],
        }
        return make_empty_manifest(), empty_report, list(class_list or [])

    df = manifest_df.copy()
    dropped_non_kaggle = 0
    if kaggle_only:
        kaggle_mask = df["source"].map(is_kaggle_source)
        dropped_non_kaggle = int((~kaggle_mask).sum())
        df = df.loc[kaggle_mask].copy()

    dropped_invalid = 0
    if only_valid:
        valid_mask = df["is_valid"].astype(bool)
        dropped_invalid = int((~valid_mask).sum())
        df = df.loc[valid_mask].copy()

    df["_norm_label"] = df["label"].map(
        lambda x: _normalize_label(
            x,
            case_insensitive=case_insensitive,
            normalize_whitespace=normalize_whitespace,
        )
    )
    df["_slug"] = df["label"].map(slugify_label)

    active_classes: list[dict[str, Any]] = []
    missing_classes: list[str] = []

    if mode == "class_list":
        if not class_list:
            raise ValueError("mode='class_list' requires a non-empty class_list")

        norm_to_entry: dict[str, dict[str, Any]] = {}
        slug_to_entry: dict[str, dict[str, Any]] = {}
        for entry in class_list:
            norm_key = _normalize_label(
                entry["label"],
                case_insensitive=case_insensitive,
                normalize_whitespace=normalize_whitespace,
            )
            norm_to_entry[norm_key] = entry
            slug_key = str(entry.get("slug") or slugify_label(entry["label"])).strip().lower()
            slug_to_entry[slug_key] = entry

        matched_entries: list[dict[str, Any] | None] = []
        for norm_lbl, slg in zip(df["_norm_label"], df["_slug"]):
            entry = norm_to_entry.get(norm_lbl) or slug_to_entry.get(slg)
            matched_entries.append(entry)

        df["_matched_entry"] = matched_entries
        df = df.loc[df["_matched_entry"].notna()].copy()

        if not df.empty:
            df["class_id"] = df["_matched_entry"].map(lambda e: int(e["class_id"]))
            df["label"] = df["_matched_entry"].map(lambda e: str(e["label"]))

            # Enforce optional min_samples / min_signers thresholds
            if effective_min_samples > 1 or effective_min_signers > 1:
                counts = df.groupby("class_id").agg(
                    n_samples=("sample_id", "count"),
                    n_signers=("signer_id", "nunique"),
                )
                keep_ids = set(
                    counts.index[
                        (counts["n_samples"] >= effective_min_samples)
                        & (counts["n_signers"] >= effective_min_signers)
                    ].tolist()
                )
                df = df.loc[df["class_id"].isin(keep_ids)].copy()

        present_ids = set(df["class_id"].astype(int).tolist()) if not df.empty else set()
        active_classes = [dict(c) for c in class_list]
        missing_classes = [
            str(c["label"]) for c in class_list if int(c["class_id"]) not in present_ids
        ]
    else:
        # Dynamic modes: top_k, min_samples, all
        if df.empty:
            active_classes = []
        else:
            grouped = (
                df.groupby("_norm_label", as_index=False)
                .agg(
                    canonical_label=("label", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
                    n_samples=("sample_id", "count"),
                    n_signers=("signer_id", "nunique"),
                )
            )
            eligible = grouped.loc[
                (grouped["n_samples"] >= effective_min_samples)
                & (grouped["n_signers"] >= effective_min_signers)
            ].copy()

            if mode == "top_k":
                eligible = eligible.sort_values(
                    by=["n_samples", "n_signers", "canonical_label"],
                    ascending=[False, False, True],
                )
                if top_k is not None and top_k > 0:
                    eligible = eligible.head(int(top_k))
            else:
                eligible = eligible.sort_values(by=["canonical_label"], ascending=[True])

            eligible = eligible.reset_index(drop=True)
            norm_to_cid: dict[str, int] = {}
            norm_to_canonical: dict[str, str] = {}
            for cid, row in eligible.iterrows():
                norm_lbl = str(row["_norm_label"])
                canon = " ".join(str(row["canonical_label"]).strip().split())
                norm_to_cid[norm_lbl] = int(cid)
                norm_to_canonical[norm_lbl] = canon
                active_classes.append(
                    {
                        "class_id": int(cid),
                        "label": canon,
                        "slug": slugify_label(canon),
                        "category": "dynamic_from_data",
                        "handedness": "unknown",
                        "qipedc_variants": [],
                        "action_description": f"Ký hiệu tay tiếng Việt cho từ '{canon}' (tham chiếu video chuẩn QIPEDC).",
                    }
                )

            df = df.loc[df["_norm_label"].isin(norm_to_cid)].copy()
            if not df.empty:
                df["class_id"] = df["_norm_label"].map(norm_to_cid).astype(int)
                df["label"] = df["_norm_label"].map(norm_to_canonical)

    # Clean temporary columns and preserve canonical column order
    scoped_df = (
        df.loc[:, MANIFEST_COLUMNS].reset_index(drop=True)
        if not df.empty
        else make_empty_manifest()
    )
    if not scoped_df.empty:
        scoped_df["class_id"] = scoped_df["class_id"].astype(int)

    dropped_out_of_scope = input_total - dropped_non_kaggle - dropped_invalid - int(len(scoped_df))

    # Build per-class summary & warnings
    per_class_stats: list[dict[str, Any]] = []
    warnings: list[str] = []
    class_meta_by_id: dict[int, dict[str, Any]] = {
        int(c["class_id"]): c for c in active_classes
    }

    if not scoped_df.empty:
        for cid, grp in scoped_df.groupby("class_id", sort=True):
            lbl = str(grp["label"].iloc[0])
            n_samp = int(len(grp))
            n_sig = int(grp["signer_id"].nunique())
            src_counts = {str(k): int(v) for k, v in grp["source"].value_counts().items()}
            cmeta = class_meta_by_id.get(int(cid), {})
            stat_entry: dict[str, Any] = {
                "class_id": int(cid),
                "label": lbl,
                "slug": str(cmeta.get("slug") or slugify_label(lbl)),
                "category": str(cmeta.get("category", "")),
                "handedness": str(cmeta.get("handedness", "")),
                "qipedc_variants": list(cmeta.get("qipedc_variants", [])),
                "action_description": str(cmeta.get("action_description", "")),
                "num_samples": n_samp,
                "num_signers": n_sig,
                "sources": src_counts,
            }
            if "kaggle_clip_count" in cmeta:
                stat_entry["kaggle_clip_count"] = int(cmeta["kaggle_clip_count"])
            if "kaggle_v2_clip_count" in cmeta:
                stat_entry["kaggle_v2_clip_count"] = int(cmeta["kaggle_v2_clip_count"])
            per_class_stats.append(stat_entry)
            if n_sig < 3:
                warnings.append(
                    f"Class '{lbl}' (class_id={int(cid)}) has only {n_sig} unique signer(s); "
                    "recommend >= 3 signers for signer-disjoint train/val/test split."
                )

    if missing_classes:
        warnings.append(
            f"{len(missing_classes)} configured class(es) had 0 matching valid samples in the manifest."
        )

    scope_report: dict[str, Any] = {
        "mode": mode,
        "kaggle_only": bool(kaggle_only),
        "effective_min_samples": effective_min_samples,
        "effective_min_signers": effective_min_signers,
        "input_samples": input_total,
        "output_samples": int(len(scoped_df)),
        "dropped_non_kaggle_samples": dropped_non_kaggle,
        "dropped_invalid_samples": dropped_invalid,
        "dropped_out_of_scope_samples": int(dropped_out_of_scope),
        "num_classes_configured": len(active_classes),
        "num_classes_present": int(scoped_df["class_id"].nunique()) if not scoped_df.empty else 0,
        "missing_classes": missing_classes,
        "num_unique_signers": int(scoped_df["signer_id"].nunique()) if not scoped_df.empty else 0,
        "classes": active_classes,
        "per_class_stats": per_class_stats,
        "warnings": warnings,
    }

    return scoped_df, scope_report, active_classes


def run_scope(
    input_manifest: str | Path,
    output_manifest: str | Path,
    report_path: str | Path,
    *,
    class_list_path: str | Path | None = None,
    mode: str = "class_list",
    top_k: int | None = 30,
    min_samples: int = 1,
    min_signers: int = 1,
    kaggle_only: bool = False,
    case_insensitive: bool = True,
    normalize_whitespace: bool = True,
    only_valid: bool = True,
    export_class_list_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run Step 1 scope filtering end-to-end and write output CSV & JSON report."""
    manifest_df = load_manifest_df(input_manifest, validate=True)
    class_list: list[dict[str, Any]] | None = None
    if mode == "class_list":
        if class_list_path is None:
            raise ValueError("class_list_path must be provided when mode='class_list'")
        class_list = load_class_list(class_list_path)

    scoped_df, scope_report, active_classes = select_scope(
        manifest_df,
        class_list=class_list,
        mode=mode,
        top_k=top_k,
        min_samples=min_samples,
        min_signers=min_signers,
        kaggle_only=kaggle_only,
        case_insensitive=case_insensitive,
        normalize_whitespace=normalize_whitespace,
        only_valid=only_valid,
    )

    save_manifest_df(scoped_df, output_manifest, validate=True)

    rep_path = Path(report_path)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with rep_path.open("w", encoding="utf-8") as f:
        json.dump(scope_report, f, ensure_ascii=False, indent=2)

    if export_class_list_path is not None:
        export_path = Path(export_class_list_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "generated_by": "src.data.scope",
            "mode": mode,
            "classes": active_classes,
        }
        with export_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return scoped_df, scope_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 1 (Chọn phạm vi): Filter manifest by class_list.json or dynamic top-K/min-samples criteria."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--input-manifest",
        type=str,
        default=None,
        help="Input manifest CSV path (defaults to dataset_manifest or kaggle_manifest from config).",
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        default=None,
        help="Output scoped manifest CSV path (default: paths.scoped_manifest).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Output JSON report path (default: paths.scope_report).",
    )
    parser.add_argument(
        "--class-list",
        type=str,
        default=None,
        help="Path to class_list.json (default: scope.class_list_path).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=VALID_SCOPE_MODES,
        default=None,
        help="Scope selection mode: class_list, top_k, min_samples, or all.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of top classes to select when --mode top_k.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Minimum number of valid samples per class.",
    )
    parser.add_argument(
        "--min-signers",
        type=int,
        default=None,
        help="Minimum number of unique signers per class.",
    )
    parser.add_argument(
        "--kaggle-only",
        action="store_true",
        help="Restrict strictly to Kaggle VSL sources (kaggle_vsl, kaggle_vsl_v2) and enforce min_samples>=3, min_signers>=3.",
    )
    parser.add_argument(
        "--include-self",
        "--hybrid",
        dest="include_self",
        action="store_true",
        help="Include self_recorded samples even when config scope.kaggle_only is true.",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Do not filter out rows with is_valid=False.",
    )
    parser.add_argument(
        "--export-class-list",
        type=str,
        default=None,
        help="Optional path to export the resolved/generated class list JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    project_root = Path(cfg.get("project", {}).get("root_dir", ".")).resolve()
    paths_cfg = cfg.get("paths", {})
    scope_cfg = cfg.get("scope", {})

    if args.input_manifest:
        input_manifest = Path(args.input_manifest)
    else:
        dataset_manifest = project_root / paths_cfg.get(
            "dataset_manifest", "data/manifests/dataset_manifest.csv"
        )
        kaggle_manifest = project_root / paths_cfg.get(
            "kaggle_manifest", "data/manifests/kaggle_manifest.csv"
        )
        input_manifest = dataset_manifest if dataset_manifest.exists() else kaggle_manifest

    output_manifest = (
        Path(args.output_manifest)
        if args.output_manifest
        else project_root / paths_cfg.get("scoped_manifest", "data/manifests/scoped_manifest.csv")
    )
    report_path = (
        Path(args.report)
        if args.report
        else project_root / paths_cfg.get("scope_report", "data/manifests/scope_report.json")
    )
    class_list_path = (
        Path(args.class_list)
        if args.class_list
        else project_root / scope_cfg.get("class_list_path", paths_cfg.get("class_list_path", "configs/class_list.json"))
    )

    mode = args.mode or scope_cfg.get("mode", "class_list")
    top_k = args.top_k if args.top_k is not None else scope_cfg.get("top_k", 30)
    if args.include_self:
        kaggle_only = False
    else:
        kaggle_only = bool(args.kaggle_only or scope_cfg.get("kaggle_only", False))
    default_min_samples = (
        int(scope_cfg.get("kaggle_only_min_samples_per_class", 3))
        if kaggle_only
        else int(scope_cfg.get("min_samples_per_class", 1))
    )
    default_min_signers = (
        int(scope_cfg.get("kaggle_only_min_signers_per_class", 3))
        if kaggle_only
        else int(scope_cfg.get("min_signers_per_class", 1))
    )
    min_samples = args.min_samples if args.min_samples is not None else default_min_samples
    min_signers = args.min_signers if args.min_signers is not None else default_min_signers
    only_valid = not args.include_invalid and bool(scope_cfg.get("only_valid_videos", True))

    scoped_df, report = run_scope(
        input_manifest=input_manifest,
        output_manifest=output_manifest,
        report_path=report_path,
        class_list_path=class_list_path,
        mode=mode,
        top_k=top_k,
        min_samples=min_samples,
        min_signers=min_signers,
        kaggle_only=kaggle_only,
        case_insensitive=bool(scope_cfg.get("case_insensitive_match", True)),
        normalize_whitespace=bool(scope_cfg.get("normalize_whitespace", True)),
        only_valid=only_valid,
        export_class_list_path=args.export_class_list,
    )

    print(
        f"Scope filtering complete ({mode}): "
        f"{report['output_samples']}/{report['input_samples']} samples kept across "
        f"{report['num_classes_present']} classes and {report['num_unique_signers']} signers."
    )
    print(f"Saved manifest: {output_manifest}")
    print(f"Saved report:   {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

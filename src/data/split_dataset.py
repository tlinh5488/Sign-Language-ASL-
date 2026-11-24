"""Step 3: Signer-disjoint dataset splitting (Chia tập dữ liệu - chia theo người).

Splits a VSL video manifest into train / val / test strictly grouped by ``signer_id``
so that NO signer appears in more than one split (zero signer leakage), while
deterministically balancing sample counts, class coverage, and source domain parity
across splits.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schema import (
    KAGGLE_SOURCES,
    MANIFEST_COLUMNS,
    is_kaggle_source,
    load_config,
    load_manifest_df,
    save_manifest_df,
    validate_manifest_df,
)

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_RANDOM_SEED = 42


def _parse_bool_series(series: pd.Series) -> pd.Series:
    """Coerce a boolean-like pandas Series safely to bool dtype."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def verify_no_signer_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> bool:
    """Assert that train, val, and test have zero overlapping ``signer_id`` values.

    Raises:
        ValueError: If any signer appears in more than one split.

    Returns:
        ``True`` when all pairwise intersections are empty.
    """
    train_signers = (
        set(train_df["signer_id"].dropna().astype(str).str.strip())
        if len(train_df) > 0
        else set()
    )
    val_signers = (
        set(val_df["signer_id"].dropna().astype(str).str.strip())
        if len(val_df) > 0
        else set()
    )
    test_signers = (
        set(test_df["signer_id"].dropna().astype(str).str.strip())
        if len(test_df) > 0
        else set()
    )

    overlap_train_val = sorted(train_signers & val_signers)
    overlap_train_test = sorted(train_signers & test_signers)
    overlap_val_test = sorted(val_signers & test_signers)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            "Signer leakage detected across splits! "
            f"train∩val={overlap_train_val}, "
            f"train∩test={overlap_train_test}, "
            f"val∩test={overlap_val_test}"
        )

    assert len(train_signers & val_signers) == 0
    assert len(train_signers & test_signers) == 0
    assert len(val_signers & test_signers) == 0
    return True


def _extract_split_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Extract split & augmentation defaults flexibly from ``configs/data_config.yaml``."""
    cfg = config or {}
    split_cfg = cfg.get("split", cfg.get("splitting", {})) or {}
    aug_cfg = cfg.get("augmentation", cfg.get("augment", {})) or {}
    paths_cfg = cfg.get("paths", {}) or {}

    train_ratio = float(
        split_cfg.get(
            "train_ratio",
            split_cfg.get("ratios", {}).get("train", DEFAULT_TRAIN_RATIO)
            if isinstance(split_cfg.get("ratios"), dict)
            else DEFAULT_TRAIN_RATIO,
        )
    )
    val_ratio = float(
        split_cfg.get(
            "val_ratio",
            split_cfg.get("ratios", {}).get("val", DEFAULT_VAL_RATIO)
            if isinstance(split_cfg.get("ratios"), dict)
            else DEFAULT_VAL_RATIO,
        )
    )
    test_ratio = float(
        split_cfg.get(
            "test_ratio",
            split_cfg.get("ratios", {}).get("test", DEFAULT_TEST_RATIO)
            if isinstance(split_cfg.get("ratios"), dict)
            else DEFAULT_TEST_RATIO,
        )
    )
    random_seed = int(
        split_cfg.get("random_seed", split_cfg.get("seed", cfg.get("random_seed", DEFAULT_RANDOM_SEED)))
    )
    stratify_by_source = bool(split_cfg.get("stratify_by_source", True))

    num_aug_default = int(
        aug_cfg.get(
            "num_augments_per_sample",
            aug_cfg.get("num_augmentations_per_video", 2),
        )
    )
    kaggle_only_num_aug = int(
        aug_cfg.get(
            "kaggle_only_num_augments_per_sample",
            aug_cfg.get("kaggle_only_num_augments", num_aug_default),
        )
    )

    default_manifest = (
        paths_cfg.get("scoped_manifest")
        or paths_cfg.get("dataset_manifest")
        or "data/manifests/scoped_manifest.csv"
    )
    default_splits_dir = paths_cfg.get("splits_dir") or "data/splits"

    return {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_seed": random_seed,
        "stratify_by_source": stratify_by_source,
        "num_aug_default": num_aug_default,
        "kaggle_only_num_aug": kaggle_only_num_aug,
        "manifest": str(default_manifest),
        "splits_dir": str(default_splits_dir),
    }


def _compute_assignment_loss(
    assignment: dict[str, str],
    active_splits: list[str],
    target_ratios: dict[str, float],
    signer_totals: dict[str, int],
    signer_classes: dict[str, Counter[str]],
    total_samples: int,
    class_totals: dict[str, int],
    signer_sources: dict[str, Counter[str]] | None = None,
) -> float:
    """Evaluate how well a signer-to-split assignment balances counts, classes & sources."""
    split_signer_counts: dict[str, int] = {s: 0 for s in active_splits}
    split_sample_counts: dict[str, int] = {s: 0 for s in active_splits}
    split_class_counts: dict[str, Counter[str]] = {s: Counter() for s in active_splits}
    split_source_counts: dict[str, Counter[str]] = {s: Counter() for s in active_splits}

    for signer, split_name in assignment.items():
        split_signer_counts[split_name] += 1
        split_sample_counts[split_name] += signer_totals[signer]
        split_class_counts[split_name].update(signer_classes[signer])
        if signer_sources is not None:
            split_source_counts[split_name].update(signer_sources[signer])

    # Hard penalty if any active split has 0 signers
    empty_penalty = sum(1000.0 for s in active_splits if split_signer_counts[s] == 0)

    # Sample count proportion MSE
    sample_loss = 0.0
    if total_samples > 0:
        for s in active_splits:
            actual = split_sample_counts[s] / total_samples
            diff = actual - target_ratios[s]
            sample_loss += diff * diff

    # Class coverage & per-class stratification loss
    all_classes = list(class_totals.keys())
    num_classes = max(1, len(all_classes))
    class_ratio_loss = 0.0
    missing_class_penalty = 0.0

    for cls in all_classes:
        cls_total = class_totals[cls]
        for s in active_splits:
            c_count = split_class_counts[s].get(cls, 0)
            if c_count == 0:
                missing_class_penalty += 2.5 if s == "train" else 1.5
            actual_cls_ratio = c_count / cls_total
            diff = actual_cls_ratio - target_ratios[s]
            class_ratio_loss += diff * diff

    class_ratio_loss /= num_classes
    missing_class_penalty /= num_classes

    # Val-vs-test source domain parity loss (prevents val=100% source A vs test=100% source B)
    source_parity_loss = 0.0
    if (
        signer_sources is not None
        and "val" in active_splits
        and "test" in active_splits
        and split_sample_counts["val"] > 0
        and split_sample_counts["test"] > 0
    ):
        all_sources = set(split_source_counts["val"].keys()) | set(split_source_counts["test"].keys())
        for src in all_sources:
            r_val = split_source_counts["val"].get(src, 0) / split_sample_counts["val"]
            r_test = split_source_counts["test"].get(src, 0) / split_sample_counts["test"]
            diff = r_val - r_test
            source_parity_loss += diff * diff

    return (
        empty_penalty
        + 10.0 * sample_loss
        + 6.0 * class_ratio_loss
        + 3.0 * missing_class_penalty
        + 8.0 * source_parity_loss
    )


def _split_single_pool_signers(
    df: pd.DataFrame,
    active_splits: list[str],
    target_ratios: dict[str, float],
    random_seed: int,
    penalize_source_parity: bool = True,
) -> dict[str, str]:
    """Deterministically split signers from a single DataFrame pool across ``active_splits``."""
    signer_series = df["signer_id"].astype(str).str.strip()
    unique_signers = sorted(signer_series.unique().tolist())
    num_signers = len(unique_signers)

    if len(active_splits) == 1:
        return {s: active_splits[0] for s in unique_signers}

    signer_totals: dict[str, int] = {}
    signer_classes: dict[str, Counter[str]] = {}
    signer_sources: dict[str, Counter[str]] = {}
    class_totals: Counter[str] = Counter()

    has_source = "source" in df.columns
    for signer, group in df.groupby(signer_series):
        label_counts = Counter(group["label"].astype(str).tolist())
        signer_totals[str(signer)] = int(len(group))
        signer_classes[str(signer)] = label_counts
        class_totals.update(label_counts)
        if has_source:
            signer_sources[str(signer)] = Counter(group["source"].astype(str).str.strip().tolist())

    total_samples = int(len(df))
    src_map = signer_sources if (penalize_source_parity and has_source) else None

    # Exact search when signer count is small (<= 8 -> at most 3^8 = 6561 states, instant)
    if num_signers <= 8 and len(active_splits) <= 3:
        best_assignment: dict[str, str] | None = None
        best_loss = float("inf")
        best_tie_key: tuple[str, ...] = ()

        rng = random.Random(random_seed)
        ordered_signers = list(unique_signers)
        rng.shuffle(ordered_signers)

        n_splits = len(active_splits)
        total_states = n_splits ** num_signers
        for state in range(total_states):
            tmp = state
            cand: dict[str, str] = {}
            counts = [0] * n_splits
            for signer in ordered_signers:
                idx = tmp % n_splits
                tmp //= n_splits
                cand[signer] = active_splits[idx]
                counts[idx] += 1
            if any(c == 0 for c in counts):
                continue
            loss = _compute_assignment_loss(
                cand,
                active_splits,
                target_ratios,
                signer_totals,
                signer_classes,
                total_samples,
                dict(class_totals),
                signer_sources=src_map,
            )
            tie_key = tuple(cand[s] for s in ordered_signers)
            if loss < best_loss - 1e-12 or (
                abs(loss - best_loss) <= 1e-12 and (not best_tie_key or tie_key < best_tie_key)
            ):
                best_loss = loss
                best_assignment = cand
                best_tie_key = tie_key

        assert best_assignment is not None
        return best_assignment

    # Multi-restart greedy + local search for larger signer counts
    rng = random.Random(random_seed)
    num_restarts = 24
    best_assignment = {}
    best_loss = float("inf")

    for restart in range(num_restarts):
        ordered = list(unique_signers)
        if restart == 0:
            ordered.sort(key=lambda g: (-len(signer_classes[g]), -signer_totals[g], g))
        else:
            rng.shuffle(ordered)

        assignment: dict[str, str] = {}
        split_counts: dict[str, int] = {s: 0 for s in active_splits}

        for split_name, signer in zip(active_splits, ordered):
            assignment[signer] = split_name
            split_counts[split_name] += 1

        for signer in ordered[len(active_splits) :]:
            best_s = active_splits[0]
            best_cand_loss = float("inf")
            for s in active_splits:
                assignment[signer] = s
                cand_loss = _compute_assignment_loss(
                    assignment,
                    active_splits,
                    target_ratios,
                    signer_totals,
                    signer_classes,
                    total_samples,
                    dict(class_totals),
                    signer_sources=src_map,
                )
                if cand_loss < best_cand_loss - 1e-12:
                    best_cand_loss = cand_loss
                    best_s = s
            assignment[signer] = best_s
            split_counts[best_s] += 1

        current_loss = _compute_assignment_loss(
            assignment,
            active_splits,
            target_ratios,
            signer_totals,
            signer_classes,
            total_samples,
            dict(class_totals),
            signer_sources=src_map,
        )
        for _ in range(30):
            improved = False
            for signer in ordered:
                orig_s = assignment[signer]
                if split_counts[orig_s] <= 1:
                    continue
                for target_s in active_splits:
                    if target_s == orig_s:
                        continue
                    assignment[signer] = target_s
                    cand_loss = _compute_assignment_loss(
                        assignment,
                        active_splits,
                        target_ratios,
                        signer_totals,
                        signer_classes,
                        total_samples,
                        dict(class_totals),
                        signer_sources=src_map,
                    )
                    if cand_loss < current_loss - 1e-12:
                        split_counts[orig_s] -= 1
                        split_counts[target_s] += 1
                        current_loss = cand_loss
                        improved = True
                        break
                    assignment[signer] = orig_s
            if not improved and len(ordered) <= 100:
                for i in range(len(ordered)):
                    s_i = ordered[i]
                    for j in range(i + 1, len(ordered)):
                        s_j = ordered[j]
                        if assignment[s_i] == assignment[s_j]:
                            continue
                        assignment[s_i], assignment[s_j] = assignment[s_j], assignment[s_i]
                        cand_loss = _compute_assignment_loss(
                            assignment,
                            active_splits,
                            target_ratios,
                            signer_totals,
                            signer_classes,
                            total_samples,
                            dict(class_totals),
                            signer_sources=src_map,
                        )
                        if cand_loss < current_loss - 1e-12:
                            current_loss = cand_loss
                            improved = True
                        else:
                            assignment[s_i], assignment[s_j] = assignment[s_j], assignment[s_i]
            if not improved:
                break

        if current_loss < best_loss - 1e-12:
            best_loss = current_loss
            best_assignment = dict(assignment)

    return best_assignment


def _is_source_balanced_for_split(
    df_src: pd.DataFrame,
    trial_assignment: dict[str, str],
    active_splits: list[str],
) -> bool:
    """Check whether splitting a single source yields balanced val & test coverage."""
    if "val" not in active_splits or "test" not in active_splits:
        return True
    mapped = df_src["signer_id"].astype(str).str.strip().map(trial_assignment)
    val_df = df_src.loc[mapped == "val"]
    test_df = df_src.loc[mapped == "test"]
    if len(val_df) == 0 or len(test_df) == 0:
        return False

    # Sample count symmetry between val & test within this source
    min_n = min(len(val_df), len(test_df))
    max_n = max(len(val_df), len(test_df))
    if max_n / max(1, min_n) > 1.6:
        return False

    # Class coverage symmetry: both val & test must cover >= 80% of this source's classes
    src_classes = df_src["label"].astype(str).nunique()
    if src_classes > 0:
        val_cov = val_df["label"].astype(str).nunique() / src_classes
        test_cov = test_df["label"].astype(str).nunique() / src_classes
        if val_cov < 0.80 or test_cov < 0.80:
            return False
    return True


def stratified_group_split_signers(
    df: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
    allow_2way: bool = False,
    fallback_warning: bool = False,
    stratify_by_source: bool = True,
) -> tuple[dict[str, str], list[str]]:
    """Assign each unique ``signer_id`` deterministically to ``train``, ``val``, or ``test``.

    When ``stratify_by_source=True`` and multiple sources exist (e.g. ``kaggle_vsl`` +
    ``kaggle_vsl_v2``):
    - Each source with ``>= len(active_splits)`` verified signers that yields symmetric
      ``val``/``test`` coverage is split independently within that source so ``val`` and
      ``test`` share matching source proportions.
    - Any smaller or asymmetric source (e.g. fewer than 3 verified actors or unequal
      per-actor class coverage) is assigned wholly to ``train`` whenever another eligible
      multi-signer source covers the dataset classes — guaranteeing ``val`` and ``test``
      have identical domain distributions (never 100% source A in val vs 100% source B
      in test).
    - If no single source alone has enough signers, falls back to global multi-source
      stratified splitting with val-vs-test source-parity optimization.

    Returns:
        Tuple of ``(signer_to_split, warnings)``.
    """
    if df.empty:
        raise ValueError("Cannot split an empty manifest.")

    ratios = {"train": float(train_ratio), "val": float(val_ratio), "test": float(test_ratio)}
    if any(v < 0 for v in ratios.values()):
        raise ValueError(f"Split ratios must be non-negative, got {ratios}.")
    if ratios["train"] <= 0:
        raise ValueError(f"train_ratio must be positive, got {ratios['train']}.")
    ratio_sum = sum(ratios.values())
    if abs(ratio_sum - 1.0) > 1e-5:
        raise ValueError(
            f"Split ratios must sum to 1.0 (within 1e-5), got {ratios} (sum={ratio_sum:.6f})."
        )

    active_splits = [s for s in ("train", "val", "test") if ratios[s] > 0]
    warnings: list[str] = []

    signer_series = df["signer_id"].astype(str).str.strip()
    unique_signers = sorted(signer_series.unique().tolist())
    num_signers = len(unique_signers)

    # Edge-case handling when fewer signers exist than requested active splits
    if num_signers < len(active_splits):
        if len(active_splits) == 3 and num_signers < 3:
            if num_signers == 2 and (allow_2way or fallback_warning):
                active_splits = ["train", "val"]
                norm_total = ratios["train"] + ratios["val"]
                ratios = {
                    "train": ratios["train"] / norm_total,
                    "val": ratios["val"] / norm_total,
                    "test": 0.0,
                }
                msg = (
                    f"Only 2 distinct signers ({unique_signers}) available; "
                    "falling back to 2-way signer-disjoint split (train/val, empty test)."
                )
                warnings.append(msg)
                logger.warning(msg)
            elif num_signers == 1 and fallback_warning:
                active_splits = ["train"]
                ratios = {"train": 1.0, "val": 0.0, "test": 0.0}
                msg = (
                    f"Only 1 distinct signer ({unique_signers}) available; "
                    "falling back to train-only split (empty val and test) to avoid signer leakage."
                )
                warnings.append(msg)
                logger.warning(msg)
            else:
                raise ValueError(
                    f"At least 3 distinct signers are required for a 3-way signer-disjoint split, "
                    f"but found {num_signers}: {unique_signers}. "
                    "Collect recordings from more signers, or explicitly pass "
                    "--allow-2way (for 2 signers) or --fallback-warning."
                )
        elif num_signers == 1 and len(active_splits) == 2:
            if fallback_warning:
                active_splits = ["train"]
                ratios = {"train": 1.0, "val": 0.0, "test": 0.0}
                msg = (
                    f"Only 1 distinct signer ({unique_signers}) available for 2-way split; "
                    "falling back to train-only split to preserve zero signer leakage."
                )
                warnings.append(msg)
                logger.warning(msg)
            else:
                raise ValueError(
                    f"At least 2 distinct signers are required for a 2-way split, "
                    f"but found {num_signers}: {unique_signers}. "
                    "Pass --fallback-warning to assign the single signer to train only."
                )

    if len(active_splits) == 1:
        return {s: active_splits[0] for s in unique_signers}, warnings

    target_ratios = {s: ratios[s] for s in active_splits}

    # Source-stratified multi-signer splitting when multiple sources are present
    has_source = "source" in df.columns
    if stratify_by_source and has_source:
        src_series = df["source"].astype(str).str.strip()
        unique_sources = sorted(src_series.unique().tolist())
        # Ensure signers don't cross multiple sources before per-source splitting
        signer_src_counts = df.groupby(signer_series)["source"].nunique()
        disjoint_source_signers = bool((signer_src_counts == 1).all())

        if len(unique_sources) >= 2 and disjoint_source_signers:
            eligible_sources: list[str] = []
            train_only_sources: list[str] = []
            trial_assignments: dict[str, dict[str, str]] = {}

            for idx, src in enumerate(unique_sources):
                df_src = df.loc[src_series == src]
                n_src_signers = int(df_src["signer_id"].astype(str).str.strip().nunique())
                if n_src_signers >= len(active_splits):
                    trial = _split_single_pool_signers(
                        df_src,
                        active_splits=active_splits,
                        target_ratios=target_ratios,
                        random_seed=random_seed + idx * 101,
                        penalize_source_parity=False,
                    )
                    if _is_source_balanced_for_split(df_src, trial, active_splits):
                        eligible_sources.append(src)
                        trial_assignments[src] = trial
                    else:
                        train_only_sources.append(src)
                else:
                    train_only_sources.append(src)

            # Use source-stratified assignment if eligible source(s) cover all dataset classes
            if eligible_sources:
                all_classes = set(df["label"].astype(str).unique())
                eligible_classes = set(
                    df.loc[src_series.isin(eligible_sources), "label"].astype(str).unique()
                )
                if eligible_classes >= all_classes:
                    combined: dict[str, str] = {}
                    train_only_samples = 0
                    for src in train_only_sources:
                        df_src = df.loc[src_series == src]
                        src_signers = sorted(df_src["signer_id"].astype(str).str.strip().unique())
                        for s_id in src_signers:
                            combined[s_id] = "train"
                        train_only_samples += len(df_src)
                        logger.info(
                            "Source '%s' (%d signers, %d samples) assigned to train to preserve "
                            "identical val/test domain distribution across eligible source(s) %s.",
                            src,
                            len(src_signers),
                            len(df_src),
                            eligible_sources,
                        )

                    # Adjust target ratios on eligible sources so overall val/test share stays accurate
                    eligible_total = int(len(df) - train_only_samples)
                    if train_only_samples > 0 and eligible_total > 0 and "train" in active_splits:
                        adj_ratios: dict[str, float] = {}
                        non_train_sum = 0.0
                        for sp in active_splits:
                            if sp == "train":
                                continue
                            # Scale up val/test target within eligible pool so global fraction matches
                            scaled = min(0.30, target_ratios[sp] * (len(df) / eligible_total))
                            adj_ratios[sp] = scaled
                            non_train_sum += scaled
                        if non_train_sum >= 0.60:
                            scale_down = 0.50 / non_train_sum
                            for sp in list(adj_ratios.keys()):
                                adj_ratios[sp] *= scale_down
                            non_train_sum = 0.50
                        adj_ratios["train"] = max(0.40, 1.0 - non_train_sum)
                        norm_sum = sum(adj_ratios.values())
                        eff_ratios = {k: v / norm_sum for k, v in adj_ratios.items()}
                    else:
                        eff_ratios = target_ratios

                    for idx, src in enumerate(eligible_sources):
                        df_src = df.loc[src_series == src]
                        if eff_ratios == target_ratios and src in trial_assignments:
                            sub_assign = trial_assignments[src]
                        else:
                            sub_assign = _split_single_pool_signers(
                                df_src,
                                active_splits=active_splits,
                                target_ratios=eff_ratios,
                                random_seed=random_seed + idx * 101,
                                penalize_source_parity=False,
                            )
                        combined.update(sub_assign)

                    return combined, warnings

    # Global fallback (single source or small multi-source dataset where no single source has >=3 signers)
    assignment = _split_single_pool_signers(
        df,
        active_splits=active_splits,
        target_ratios=target_ratios,
        random_seed=random_seed,
        penalize_source_parity=stratify_by_source,
    )
    return assignment, warnings


def split_dataset(
    manifest_path: str | Path,
    output_dir: str | Path = "data/splits",
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    random_seed: int | None = None,
    config_path: str | Path | None = None,
    allow_2way: bool = False,
    fallback_warning: bool = False,
    include_invalid: bool = False,
    kaggle_only: bool = False,
    stratify_by_source: bool | None = None,
) -> dict[str, Any]:
    """Run Step 3: signer-disjoint train / val / test split and save CSVs + report.

    Args:
        manifest_path: Input manifest CSV (e.g. ``data/manifests/scoped_manifest.csv``).
        output_dir: Directory to write ``train.csv``, ``val.csv``, ``test.csv``,
            and ``split_report.json``.
        train_ratio: Optional override for train split ratio.
        val_ratio: Optional override for validation split ratio.
        test_ratio: Optional override for test split ratio.
        random_seed: Optional override for deterministic seed.
        config_path: Optional YAML config path to load default split ratios & seed.
        allow_2way: Allow falling back to a 2-way (train/val) split when only 2 signers exist.
        fallback_warning: Allow falling back with warnings when fewer than 3 signers exist.
        include_invalid: If False (default), exclude rows where ``is_valid`` is False.
        kaggle_only: If True, enforce that all input rows come strictly from supported
            Kaggle sources (``kaggle_vsl``, ``kaggle_vsl_v2``) and record Kaggle-only
            split metadata in ``split_report.json``.
        stratify_by_source: If True (default), preserve val/test source domain parity by
            splitting within eligible multi-signer sources (and placing small/asymmetric
            sources into train so val and test share identical domain distributions).

    Returns:
        Dictionary containing split report metrics and output paths.
    """
    cfg_dict = load_config(config_path) if config_path and Path(config_path).exists() else {}
    defaults = _extract_split_config(cfg_dict)

    tr = defaults["train_ratio"] if train_ratio is None else float(train_ratio)
    vr = defaults["val_ratio"] if val_ratio is None else float(val_ratio)
    te = defaults["test_ratio"] if test_ratio is None else float(test_ratio)
    seed = defaults["random_seed"] if random_seed is None else int(random_seed)
    use_source_strat = (
        defaults["stratify_by_source"]
        if stratify_by_source is None
        else bool(stratify_by_source)
    )

    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

    df = load_manifest_df(manifest_file)
    missing_cols = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input manifest is missing required columns: {missing_cols}")

    # Enforce pipeline ordering: do not split pre-augmented data
    if len(df) > 0 and "is_augmented" in df.columns:
        aug_mask = _parse_bool_series(df["is_augmented"])
        if aug_mask.any():
            raise ValueError(
                f"Input manifest contains {int(aug_mask.sum())} pre-augmented rows. "
                "Splitting MUST run before train-only augmentation to prevent data leakage."
            )

    # Filter invalid videos by default
    excluded_invalid = 0
    if not include_invalid and len(df) > 0 and "is_valid" in df.columns:
        valid_mask = _parse_bool_series(df["is_valid"])
        excluded_invalid = int((~valid_mask).sum())
        df = df.loc[valid_mask].copy()

    if df.empty:
        raise ValueError(
            "No valid samples available to split after filtering invalid videos."
        )

    # Validate non-empty signer_id on every row
    signer_clean = df["signer_id"].fillna("").astype(str).str.strip()
    empty_signers = int((signer_clean == "").sum())
    if empty_signers > 0:
        raise ValueError(
            f"Found {empty_signers} rows with missing/empty signer_id. "
            "Every sample must have a valid signer_id for signer-disjoint splitting."
        )
    df["signer_id"] = signer_clean

    if kaggle_only:
        non_kaggle = df.loc[~df["source"].map(is_kaggle_source)]
        if len(non_kaggle) > 0:
            bad_sources = sorted(non_kaggle["source"].astype(str).unique().tolist())
            raise ValueError(
                f"kaggle_only mode requires source in {KAGGLE_SOURCES} for all rows, "
                f"but found {len(non_kaggle)} row(s) with source={bad_sources}."
            )

    signer_to_split, warnings = stratified_group_split_signers(
        df=df,
        train_ratio=tr,
        val_ratio=vr,
        test_ratio=te,
        random_seed=seed,
        allow_2way=allow_2way,
        fallback_warning=fallback_warning,
        stratify_by_source=use_source_strat,
    )

    df = df.copy()
    df["split"] = df["signer_id"].map(signer_to_split)
    df["is_augmented"] = False
    if "parent_sample_id" in df.columns:
        df["parent_sample_id"] = df["parent_sample_id"].fillna("")
    if "aug_type" in df.columns:
        df["aug_type"] = df["aug_type"].fillna("")

    split_dfs: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        sub = df.loc[df["split"] == split_name, MANIFEST_COLUMNS].reset_index(drop=True)
        if len(sub) > 0:
            validate_manifest_df(sub, require_split=True)
        split_dfs[split_name] = sub

    # Strict zero-signer-leakage enforcement
    leakage_passed = verify_no_signer_leakage(
        split_dfs["train"], split_dfs["val"], split_dfs["test"]
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_files: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        out_csv = out_dir / f"{split_name}.csv"
        save_manifest_df(
            split_dfs[split_name],
            out_csv,
            validate=len(split_dfs[split_name]) > 0,
            require_split=True,
        )
        output_files[split_name] = str(out_csv)

    all_labels = sorted(df["label"].astype(str).unique().tolist())
    total_samples = int(len(df))

    splits_summary: dict[str, Any] = {}
    actual_ratios: dict[str, float] = {}
    for split_name in ("train", "val", "test"):
        sub = split_dfs[split_name]
        signers = sorted(sub["signer_id"].astype(str).unique().tolist()) if len(sub) > 0 else []
        class_counts = (
            {str(k): int(v) for k, v in sub["label"].astype(str).value_counts().sort_index().items()}
            if len(sub) > 0
            else {}
        )
        source_counts = (
            {str(k): int(v) for k, v in sub["source"].astype(str).value_counts().sort_index().items()}
            if len(sub) > 0
            else {}
        )
        source_ratios = (
            {k: round(v / len(sub), 4) for k, v in source_counts.items()}
            if len(sub) > 0
            else {}
        )
        missing_classes = [lbl for lbl in all_labels if lbl not in class_counts]
        if missing_classes and split_name in ("val", "test") and len(sub) > 0:
            warnings.append(
                f"Split '{split_name}' is missing {len(missing_classes)} class(es): {missing_classes}"
            )

        counts_list = list(class_counts.values())
        ratio_val = round(len(sub) / total_samples, 4) if total_samples > 0 else 0.0
        actual_ratios[split_name] = ratio_val
        splits_summary[split_name] = {
            "num_samples": int(len(sub)),
            "ratio": ratio_val,
            "num_signers": len(signers),
            "signers": signers,
            "num_classes": len(class_counts),
            "min_samples_per_class": int(min(counts_list)) if counts_list else 0,
            "max_samples_per_class": int(max(counts_list)) if counts_list else 0,
            "avg_samples_per_class": round(sum(counts_list) / len(counts_list), 2) if counts_list else 0.0,
            "class_counts": class_counts,
            "source_counts": source_counts,
            "source_ratios": source_ratios,
            "missing_classes": missing_classes,
            "csv_path": output_files[split_name],
        }

    train_s = set(splits_summary["train"]["signers"])
    val_s = set(splits_summary["val"]["signers"])
    test_s = set(splits_summary["test"]["signers"])

    val_src_ratios = splits_summary["val"]["source_ratios"]
    test_src_ratios = splits_summary["test"]["source_ratios"]
    eval_sources = sorted(set(val_src_ratios.keys()) | set(test_src_ratios.keys()))
    max_src_diff = (
        round(
            max(abs(val_src_ratios.get(s, 0.0) - test_src_ratios.get(s, 0.0)) for s in eval_sources),
            4,
        )
        if eval_sources
        else 0.0
    )

    # Keep recommended_train_aug_copies 100% consistent with configs/data_config.yaml
    recommended_aug_copies = (
        defaults["kaggle_only_num_aug"] if kaggle_only else defaults["num_aug_default"]
    )

    report: dict[str, Any] = {
        "mode": "kaggle_only" if kaggle_only else "standard",
        "stratify_by_source": use_source_strat,
        "input_manifest": str(manifest_file),
        "random_seed": seed,
        "requested_ratios": {"train": tr, "val": vr, "test": te},
        "actual_ratios": actual_ratios,
        "total_samples": total_samples,
        "excluded_invalid_samples": excluded_invalid,
        "total_signers": int(df["signer_id"].nunique()),
        "total_classes": len(all_labels),
        "all_classes": all_labels,
        "splits": splits_summary,
        "source_parity_check": {
            "val_source_ratios": val_src_ratios,
            "test_source_ratios": test_src_ratios,
            "max_val_test_source_ratio_diff": max_src_diff,
            "val_test_domain_matched": bool(max_src_diff <= 0.15),
        },
        "recommended_train_aug_copies": int(recommended_aug_copies),
        "leakage_check_passed": bool(leakage_passed),
        "signer_overlaps": {
            "train_val": sorted(train_s & val_s),
            "train_test": sorted(train_s & test_s),
            "val_test": sorted(val_s & test_s),
        },
        "warnings": warnings,
    }

    report_path = out_dir / "split_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report["report_path"] = str(report_path)

    logger.info(
        "Signer-disjoint split completed: train=%d (%d signers, %s), val=%d (%d signers, %s), "
        "test=%d (%d signers, %s) | leakage_check_passed=%s | val_test_domain_matched=%s",
        splits_summary["train"]["num_samples"],
        splits_summary["train"]["num_signers"],
        splits_summary["train"]["source_counts"],
        splits_summary["val"]["num_samples"],
        splits_summary["val"]["num_signers"],
        splits_summary["val"]["source_counts"],
        splits_summary["test"]["num_samples"],
        splits_summary["test"]["num_signers"],
        splits_summary["test"]["source_counts"],
        leakage_passed,
        report["source_parity_check"]["val_test_domain_matched"],
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 3: Signer-disjoint train/val/test split grouped by signer_id."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data_config.yaml (default: configs/data_config.yaml).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Input manifest CSV (defaults to scoped_manifest from config).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for train.csv, val.csv, test.csv, and split_report.json.",
    )
    parser.add_argument("--train-ratio", type=float, default=None, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=None, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic split.")
    parser.add_argument(
        "--allow-2way",
        action="store_true",
        help="Allow 2-way (train/val) split when only 2 distinct signers exist.",
    )
    parser.add_argument(
        "--fallback-warning",
        action="store_true",
        help="Allow fallback with warnings when fewer than 3 distinct signers exist.",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Include rows with is_valid=False instead of filtering them out.",
    )
    parser.add_argument(
        "--kaggle-only",
        action="store_true",
        help="Enforce Kaggle-only rows (kaggle_vsl / kaggle_vsl_v2) and log split recommendations.",
    )
    parser.add_argument(
        "--no-stratify-by-source",
        action="store_true",
        help="Disable source-stratified val/test domain parity (pool all sources globally).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv)

    cfg_path = Path(args.config) if args.config else None
    cfg_dict = load_config(cfg_path) if cfg_path and cfg_path.exists() else {}
    defaults = _extract_split_config(cfg_dict)

    manifest_path = args.manifest or defaults["manifest"]
    output_dir = args.output_dir or defaults["splits_dir"]

    report = split_dataset(
        manifest_path=manifest_path,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        config_path=cfg_path,
        allow_2way=args.allow_2way,
        fallback_warning=args.fallback_warning,
        include_invalid=args.include_invalid,
        kaggle_only=args.kaggle_only,
        stratify_by_source=False if args.no_stratify_by_source else None,
    )
    print(
        f"[split_dataset] Saved splits to {output_dir} | "
        f"train={report['splits']['train']['num_samples']} "
        f"({report['splits']['train']['num_signers']} signers, {report['splits']['train']['source_counts']}), "
        f"val={report['splits']['val']['num_samples']} "
        f"({report['splits']['val']['num_signers']} signers, {report['splits']['val']['source_counts']}), "
        f"test={report['splits']['test']['num_samples']} "
        f"({report['splits']['test']['num_signers']} signers, {report['splits']['test']['source_counts']}) | "
        f"leakage_check_passed={report['leakage_check_passed']} | "
        f"val_test_domain_matched={report['source_parity_check']['val_test_domain_matched']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

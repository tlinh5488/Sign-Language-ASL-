"""Comprehensive automated tests for Quy's VSL Data Pipeline (Phần 1: Dữ liệu).

Covers all 4 stages using fast synthetic OpenCV `.mp4` fixtures in `tmp_path`:
  - Shared schema & config contract (`configs/`, `src/data/schema.py`)
  - Step 1: Kaggle VSL ingestion & scope selection (`download_kaggle.py`, `scope.py`)
  - Step 2: Self-recording headless helpers & manifest merge/audit (`collect_videos.py`, `build_manifest.py`)
  - Step 3: Strict signer-disjoint splitting + leakage & edge-case guards (`split_dataset.py`)
  - Step 4: Train-only augmentation + hard val/test refusal & immutability (`augment.py`)
  - End-to-end demo runner (`scripts/run_quy_pipeline.py`)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_quy_pipeline import (
    build_arg_parser as build_pipeline_arg_parser,
    run_production_pipeline,
)
from src.data.augment import (
    TRAIN_ONLY_ERROR_MSG,
    AugmentConfig,
    apply_video_augmentations,
    augment_train_manifest,
    ensure_train_only_manifest,
)
from src.data.build_manifest import audit_manifest, merge_manifests, scan_raw_directory
from src.data.collect_videos import (
    append_manifest_row,
    build_sample_id,
    build_video_filename,
    build_video_path,
    create_manifest_row,
    draw_framing_guide,
    normalize_signer_id,
    resolve_class_entry,
    write_frames_to_video,
)
from src.data.download_kaggle import (
    build_arg_parser as build_download_kaggle_arg_parser,
    infer_kaggle_signer_id,
    parse_kaggle_dataset_to_manifest,
    parse_kaggle_v2_dataset_to_manifest,
)
from src.data.schema import (
    KAGGLE_SOURCES,
    MANIFEST_COLUMNS,
    VALID_SOURCES,
    VALID_SPLITS,
    is_kaggle_source,
    load_class_list,
    load_config,
    load_manifest_df,
    normalize_rel_path,
    normalize_vietnamese_label,
    probe_video_metadata,
    save_manifest_df,
    slugify_label,
    validate_manifest_df,
)
from src.data.scope import run_scope, select_scope
from src.data.split_dataset import (
    split_dataset,
    stratified_group_split_signers,
    verify_no_signer_leakage,
)


# ---------------------------------------------------------------------------
# Synthetic video & manifest fixture helpers
# ---------------------------------------------------------------------------


def _make_synthetic_video(
    path: Path,
    *,
    num_frames: int = 12,
    fps: float = 15.0,
    width: int = 96,
    height: int = 72,
    seed: int = 0,
) -> Path:
    """Create a small valid OpenCV `.mp4` clip for fast headless unit tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    for t in range(num_frames):
        frame = np.full(
            (height, width, 3),
            ((40 + seed * 17) % 200, (70 + t * 8) % 220, (90 + seed * 23) % 210),
            dtype=np.uint8,
        )
        cx = int(width * (0.2 + 0.6 * (t / max(1, num_frames - 1))))
        cy = height // 2
        cv2.circle(frame, (cx, cy), 8, (240, 230, 200), -1)
        frames.append(frame)
    return write_frames_to_video(frames, path, fps=fps)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_manifest_row(
    *,
    sample_id: str,
    video_path: str,
    label: str,
    class_id: int,
    signer_id: str,
    source: str = "self_recorded",
    split: str = "",
    is_augmented: bool = False,
    parent_sample_id: str = "",
    aug_type: str = "",
    fps: float = 15.0,
    num_frames: int = 12,
    duration_sec: float = 0.8,
    width: int = 96,
    height: int = 72,
    is_valid: bool = True,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "video_path": video_path,
        "label": label,
        "class_id": class_id,
        "signer_id": signer_id,
        "source": source,
        "split": split,
        "is_augmented": is_augmented,
        "parent_sample_id": parent_sample_id,
        "aug_type": aug_type,
        "fps": fps,
        "num_frames": num_frames,
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
        "is_valid": is_valid,
    }


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> dict[str, Any]:
    """Build a synthetic 6-signer x 4-class x 2-take dataset with real `.mp4` files."""
    classes = [
        {
            "class_id": 0,
            "label": "chào",
            "slug": "chao",
            "category": "greetings",
            "handedness": "1_tay",
            "qipedc_variants": ["common"],
            "action_description": "Vẫy bàn tay thuận ngang vai.",
        },
        {
            "class_id": 1,
            "label": "xin lỗi",
            "slug": "xin_loi",
            "category": "greetings",
            "handedness": "1_tay",
            "qipedc_variants": ["common"],
            "action_description": "Đặt tay thuận lên ngực xoay tròn nhẹ.",
        },
        {
            "class_id": 2,
            "label": "giúp đỡ",
            "slug": "giup_do",
            "category": "communication",
            "handedness": "2_tay",
            "qipedc_variants": ["B", "T", "N"],
            "action_description": "Hai tay nâng nhẹ trước ngực.",
        },
        {
            "class_id": 3,
            "label": "sách",
            "slug": "sach",
            "category": "objects_food",
            "handedness": "2_tay",
            "qipedc_variants": ["common"],
            "action_description": "Mở hai lòng bàn tay trước ngực như mở sách.",
        },
    ]
    class_list_path = tmp_path / "class_list.json"
    class_list_path.write_text(
        json.dumps({"schema_version": "1.1", "classes": classes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    signers = [
        ("kaggle_signer_north", "kaggle_vsl"),
        ("kaggle_signer_south", "kaggle_vsl"),
        ("kaggle_signer_central", "kaggle_vsl"),
        ("signer_01", "self_recorded"),
        ("signer_02", "self_recorded"),
        ("signer_03", "self_recorded"),
    ]

    rows: list[dict[str, Any]] = []
    video_counter = 0
    for signer_id, source in signers:
        for cls in classes:
            for take in (1, 2):
                video_counter += 1
                rel_dir = "raw/kaggle" if source == "kaggle_vsl" else "raw/self"
                vid_path = (
                    tmp_path
                    / rel_dir
                    / cls["slug"]
                    / f"{signer_id}_{cls['slug']}_take{take:02d}.mp4"
                )
                _make_synthetic_video(vid_path, seed=video_counter)
                meta = probe_video_metadata(vid_path)
                rows.append(
                    _make_manifest_row(
                        sample_id=f"{source}_{signer_id}_{cls['slug']}_t{take:02d}",
                        video_path=normalize_rel_path(vid_path, tmp_path),
                        label=cls["label"],
                        class_id=cls["class_id"],
                        signer_id=signer_id,
                        source=source,
                        fps=meta["fps"],
                        num_frames=meta["num_frames"],
                        duration_sec=meta["duration_sec"],
                        width=meta["width"],
                        height=meta["height"],
                        is_valid=meta["is_valid"],
                    )
                )

    manifest_df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest_path = tmp_path / "manifests" / "scoped_manifest.csv"
    save_manifest_df(manifest_df, manifest_path, validate=True)

    return {
        "root": tmp_path,
        "classes": classes,
        "class_list_path": class_list_path,
        "manifest_df": manifest_df,
        "manifest_path": manifest_path,
    }


# ---------------------------------------------------------------------------
# 1. Shared Schema & Configuration Contract Tests
# ---------------------------------------------------------------------------


def test_schema_contract_and_project_configs() -> None:
    expected_columns = [
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
    assert MANIFEST_COLUMNS == expected_columns
    assert VALID_SPLITS == ("train", "val", "test")
    assert KAGGLE_SOURCES == ("kaggle_vsl", "kaggle_vsl_v2")
    assert VALID_SOURCES == ("kaggle_vsl", "kaggle_vsl_v2", "self_recorded")
    assert is_kaggle_source("kaggle_vsl") is True
    assert is_kaggle_source("kaggle_vsl_v2") is True
    assert is_kaggle_source("self_recorded") is False
    assert normalize_vietnamese_label("  Giu\u0301p   \u0111\u01a1\u0303 ") == "Giúp đỡ"

    cfg = load_config(REPO_ROOT / "configs" / "data_config.yaml")
    assert cfg["kaggle"]["dataset_slug"] == "aresusayhi/vsl-vietnamese-sign-languages"
    assert cfg["kaggle_v2"]["dataset_slug"] == "nguyenanfms/vsl-vietnamese-sign-language-v2"
    assert cfg["kaggle_v2"]["source_name"] == "kaggle_vsl_v2"
    assert cfg["kaggle_v2"]["exclude_pre_augmented"] is True
    assert cfg["scope"]["kaggle_only"] is True
    assert cfg["scope"]["kaggle_only_min_samples_per_class"] == 3
    assert cfg["scope"]["kaggle_only_min_signers_per_class"] == 3
    assert build_download_kaggle_arg_parser().parse_args([]).dataset == "all"
    assert cfg["split"]["train_ratio"] == pytest.approx(0.70)
    assert cfg["split"]["val_ratio"] == pytest.approx(0.15)
    assert cfg["split"]["test_ratio"] == pytest.approx(0.15)
    assert cfg["split"]["random_seed"] == 42
    assert cfg["augmentation"]["allow_hflip"] is False
    assert cfg["augmentation"]["train_only"] is True
    assert cfg["augmentation"]["kaggle_only_num_augments_per_sample"] == 4

    classes = load_class_list(REPO_ROOT / "configs" / "class_list.json")
    assert len(classes) == 30
    assert [c["class_id"] for c in classes] == list(range(30))
    assert len({c["label"] for c in classes}) == 30
    assert len({c["slug"] for c in classes}) == 30
    for c in classes:
        assert c["handedness"] in {"1_tay", "2_tay"}
        assert isinstance(c["qipedc_variants"], list) and set(c["qipedc_variants"]) >= {"B", "T", "N"}
        assert int(c["kaggle_clip_count"]) >= 3
        assert int(c["kaggle_v2_clip_count"]) >= 55
        assert isinstance(c["action_description"], str) and len(c["action_description"].strip()) >= 15


def test_validate_manifest_df_and_video_probe(tmp_path: Path) -> None:
    vid = _make_synthetic_video(tmp_path / "clips" / "sample.mp4", num_frames=10, fps=20.0)
    meta = probe_video_metadata(vid)
    assert meta["is_valid"] is True
    assert meta["num_frames"] >= 10
    assert meta["width"] == 96 and meta["height"] == 72
    assert meta["fps"] > 0 and meta["duration_sec"] > 0

    # Missing & corrupt videos must safely return is_valid=False
    assert probe_video_metadata(tmp_path / "does_not_exist.mp4")["is_valid"] is False
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a real mp4 file")
    assert probe_video_metadata(corrupt)["is_valid"] is False

    # POSIX relative path normalization
    rel = normalize_rel_path(vid, tmp_path)
    assert rel == "clips/sample.mp4"
    assert "\\" not in rel

    # Valid manifest passes
    row = _make_manifest_row(
        sample_id="s1",
        video_path=rel,
        label="chào",
        class_id=0,
        signer_id="signer_01",
    )
    df = pd.DataFrame([row], columns=MANIFEST_COLUMNS)
    validate_manifest_df(df)

    # require_split rejects empty split
    with pytest.raises(ValueError, match="split"):
        validate_manifest_df(df, require_split=True)

    # Invalid source rejected
    bad_source = df.copy()
    bad_source["source"] = "youtube_random"
    with pytest.raises(ValueError, match="source"):
        validate_manifest_df(bad_source)

    # Duplicate sample_id rejected
    dup_df = pd.concat([df, df], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        validate_manifest_df(dup_df)

    # Augmented row without parent_sample_id rejected
    bad_aug_parent = df.copy()
    bad_aug_parent["is_augmented"] = True
    bad_aug_parent["split"] = "train"
    with pytest.raises(ValueError, match="parent_sample_id"):
        validate_manifest_df(bad_aug_parent)

    # Augmented row in val/test rejected
    bad_aug_split = df.copy()
    bad_aug_split["is_augmented"] = True
    bad_aug_split["parent_sample_id"] = "s0"
    bad_aug_split["split"] = "val"
    with pytest.raises(ValueError, match="only 'train' is allowed"):
        validate_manifest_df(bad_aug_split)


# ---------------------------------------------------------------------------
# 2. Step 1: Kaggle Ingestion & Scope Selection Tests
# ---------------------------------------------------------------------------


def test_kaggle_signer_inference_and_manifest_parsing(tmp_path: Path) -> None:
    assert infer_kaggle_signer_id("D0001B.mp4") == "kaggle_signer_north"
    assert infer_kaggle_signer_id("D0001N.mp4") == "kaggle_signer_south"
    assert infer_kaggle_signer_id("D0001T.mp4") == "kaggle_signer_central"
    assert infer_kaggle_signer_id("W0002.mp4") == "kaggle_signer_common_w"
    assert infer_kaggle_signer_id("D0010.mp4") == "kaggle_signer_common_d"
    assert infer_kaggle_signer_id("clip.mp4", row_meta={"signer_id": "p07"}) == "kaggle_p07"

    # Build synthetic QIPEDC layout: Dataset/Labels/label.csv + Dataset/Videos/*.mp4
    kaggle_dir = tmp_path / "raw_kaggle"
    labels_csv = kaggle_dir / "Dataset" / "Labels" / "label.csv"
    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    videos_dir = kaggle_dir / "Dataset" / "Videos"
    _make_synthetic_video(videos_dir / "D0001B.mp4", seed=1)
    _make_synthetic_video(videos_dir / "D0001N.mp4", seed=2)
    _make_synthetic_video(videos_dir / "W0002.mp4", seed=3)

    pd.DataFrame(
        [
            {"ID": 1, "VIDEO": "D0001B.mp4", "LABEL": "giúp đỡ"},
            {"ID": 2, "VIDEO": "D0001N.mp4", "LABEL": "giúp đỡ"},
            {"ID": 3, "VIDEO": "W0002.mp4", "LABEL": "chào"},
            {"ID": 4, "VIDEO": "MISSING999T.mp4", "LABEL": "từ lạ"},
        ]
    ).to_csv(labels_csv, index=False, encoding="utf-8")

    class_list = load_class_list(REPO_ROOT / "configs" / "class_list.json")
    df = parse_kaggle_dataset_to_manifest(
        kaggle_dir,
        project_root=tmp_path,
        class_list=class_list,
        probe_videos=True,
    )
    validate_manifest_df(df)
    assert len(df) == 4
    assert set(df["source"]) == {"kaggle_vsl"}
    assert df.loc[df["video_path"].str.endswith("D0001B.mp4"), "signer_id"].iloc[0] == "kaggle_signer_north"
    assert df.loc[df["video_path"].str.endswith("D0001N.mp4"), "signer_id"].iloc[0] == "kaggle_signer_south"
    assert df.loc[df["video_path"].str.endswith("W0002.mp4"), "signer_id"].iloc[0] == "kaggle_signer_common_w"
    assert int(df["is_valid"].sum()) == 3
    assert df.loc[df["label"] == "từ lạ", "is_valid"].iloc[0] == False  # noqa: E712


def test_step1_scope_filtering_and_enriched_report(synthetic_dataset: dict[str, Any]) -> None:
    tmp_path: Path = synthetic_dataset["root"]
    df = synthetic_dataset["manifest_df"].copy()

    # Add 1 out-of-scope valid row and 1 in-scope invalid row
    oos_vid = _make_synthetic_video(tmp_path / "raw/self/extra/extra_1.mp4", seed=99)
    extra_rows = pd.DataFrame(
        [
            _make_manifest_row(
                sample_id="oos_sample_1",
                video_path=normalize_rel_path(oos_vid, tmp_path),
                label="từ ngoài phạm vi",
                class_id=-1,
                signer_id="signer_01",
                is_valid=True,
            ),
            _make_manifest_row(
                sample_id="invalid_in_scope_1",
                video_path="raw/self/chao/broken.mp4",
                label="chào",
                class_id=0,
                signer_id="signer_01",
                is_valid=False,
            ),
        ],
        columns=MANIFEST_COLUMNS,
    )
    full_df = pd.concat([df, extra_rows], ignore_index=True)
    input_csv = tmp_path / "manifests" / "unscoped.csv"
    save_manifest_df(full_df, input_csv, validate=True)

    scoped_csv = tmp_path / "manifests" / "scoped_out.csv"
    report_json = tmp_path / "manifests" / "scope_report.json"

    scoped_df, report = run_scope(
        input_manifest=input_csv,
        output_manifest=scoped_csv,
        report_path=report_json,
        class_list_path=synthetic_dataset["class_list_path"],
        mode="class_list",
        only_valid=True,
    )
    assert len(scoped_df) == len(df)
    assert report["dropped_invalid_samples"] == 1
    assert report["dropped_out_of_scope_samples"] == 1
    assert report["num_classes_present"] == 4
    assert "classes" in report and len(report["classes"]) == 4
    assert all(c.get("action_description") for c in report["classes"])
    assert all(s.get("action_description") and s.get("handedness") for s in report["per_class_stats"])

    saved_report = json.loads(report_json.read_text(encoding="utf-8"))
    assert saved_report["output_samples"] == len(df)
    assert len(saved_report["per_class_stats"]) == 4

    # Dynamic top_k mode + export class list
    exported_json = tmp_path / "manifests" / "exported_classes.json"
    top2_df, top2_report = run_scope(
        input_manifest=input_csv,
        output_manifest=tmp_path / "manifests" / "top2.csv",
        report_path=tmp_path / "manifests" / "top2_report.json",
        mode="top_k",
        top_k=2,
        export_class_list_path=exported_json,
    )
    assert top2_report["num_classes_present"] == 2
    assert set(top2_df["class_id"]) == {0, 1}
    exported = json.loads(exported_json.read_text(encoding="utf-8"))
    assert len(exported["classes"]) == 2


# ---------------------------------------------------------------------------
# 3. Step 2: Self-Recording Headless Helpers & Manifest Merge/Audit Tests
# ---------------------------------------------------------------------------


def test_step2_collect_videos_headless_helpers(tmp_path: Path) -> None:
    classes = load_class_list(REPO_ROOT / "configs" / "class_list.json")

    assert normalize_signer_id("  Signer 01 ") == "signer_01"
    with pytest.raises(ValueError):
        normalize_signer_id("   ")

    # Resolve by class_id, Vietnamese label, or ASCII slug
    by_id = resolve_class_entry(classes, class_id=2)
    by_lbl = resolve_class_entry(classes, class_label="tiếp tục")
    by_slug = resolve_class_entry(classes, class_label="tiep_tuc")
    assert by_id["class_id"] == by_lbl["class_id"] == by_slug["class_id"] == 2

    with pytest.raises(ValueError, match="Conflicting"):
        resolve_class_entry(classes, class_id=0, class_label="tiếp tục")

    # Deterministic naming `<class_slug>/<signer_id>_<class_slug>_<timestamp>.mp4`
    ts = "20261122_093000_123"
    fname = build_video_filename("signer_01", by_id["slug"], timestamp=ts)
    assert fname == "signer_01_tiep_tuc_20261122_093000_123.mp4"

    vid_path = build_video_path(tmp_path / "raw/self_recorded", "signer_01", by_id["slug"], timestamp=ts)
    assert vid_path.parent.name == "tiep_tuc"
    assert vid_path.name == fname

    # Framing guide must not mutate clean raw frames
    raw_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    raw_copy = raw_frame.copy()
    annotated = draw_framing_guide(raw_frame, status_text="Signer: signer_01", countdown=3, progress=0.5, is_recording=True)
    assert np.array_equal(raw_frame, raw_copy)
    assert not np.array_equal(annotated, raw_frame)

    # Write clip and append multiple signers to self_manifest.csv
    self_manifest = tmp_path / "manifests" / "self_manifest.csv"
    for signer in ("signer_01", "signer_02"):
        vp = build_video_path(tmp_path / "raw/self_recorded", signer, by_id["slug"], timestamp=ts)
        _make_synthetic_video(vp, num_frames=10, fps=30.0)
        row = create_manifest_row(
            video_path=vp,
            class_entry=by_id,
            signer_id=signer,
            root_dir=tmp_path,
            sample_id=build_sample_id(signer, by_id["slug"], timestamp=ts),
        )
        combined = append_manifest_row(self_manifest, row, validate=True)

    assert len(combined) == 2
    assert set(combined["signer_id"]) == {"signer_01", "signer_02"}
    assert set(combined["source"]) == {"self_recorded"}
    assert all(combined["is_valid"])


def test_step2_merge_manifests_dedup_and_quality_audit(tmp_path: Path) -> None:
    classes = load_class_list(REPO_ROOT / "configs" / "class_list.json")

    v1 = _make_synthetic_video(tmp_path / "raw/kaggle/giup_do/D0001B.mp4", seed=1)
    v2 = _make_synthetic_video(tmp_path / "raw/self/giup_do/signer_01_giup_do_20261122.mp4", seed=2)
    v3 = _make_synthetic_video(tmp_path / "raw/self/tiep_tuc/signer_02_tiep_tuc_20261122.mp4", seed=3)
    corrupt_v = tmp_path / "raw/self/giup_do/signer_03_giup_do_corrupt.mp4"
    corrupt_v.parent.mkdir(parents=True, exist_ok=True)
    corrupt_v.write_bytes(b"corrupt")

    kaggle_csv = tmp_path / "manifests" / "kaggle_manifest.csv"
    self_csv = tmp_path / "manifests" / "self_manifest.csv"

    save_manifest_df(
        pd.DataFrame(
            [
                _make_manifest_row(
                    sample_id="k_1",
                    video_path=normalize_rel_path(v1, tmp_path),
                    label="giúp đỡ",
                    class_id=0,
                    signer_id="kaggle_signer_north",
                    source="kaggle_vsl",
                )
            ],
            columns=MANIFEST_COLUMNS,
        ),
        kaggle_csv,
        validate=True,
    )

    # Include 1 duplicate sample_id, 1 duplicate video_path, 1 corrupt video, 1 missing signer_id
    self_df = pd.DataFrame(
        [
            _make_manifest_row(
                sample_id="s_1",
                video_path=normalize_rel_path(v2, tmp_path),
                label="giúp đỡ",
                class_id=-1,
                signer_id="signer_01",
                source="self_recorded",
            ),
            _make_manifest_row(
                sample_id="s_1",  # duplicate sample_id
                video_path=normalize_rel_path(v2, tmp_path),
                label="giúp đỡ",
                class_id=-1,
                signer_id="signer_01",
                source="self_recorded",
            ),
            _make_manifest_row(
                sample_id="s_dup_path",
                video_path=normalize_rel_path(v1, tmp_path),  # duplicate video_path from kaggle
                label="giúp đỡ",
                class_id=0,
                signer_id="signer_01",
                source="self_recorded",
            ),
            _make_manifest_row(
                sample_id="s_2",
                video_path=normalize_rel_path(v3, tmp_path),
                label="tiếp tục",
                class_id=-1,
                signer_id="signer_02",
                source="self_recorded",
            ),
            _make_manifest_row(
                sample_id="s_corrupt",
                video_path=normalize_rel_path(corrupt_v, tmp_path),
                label="giúp đỡ",
                class_id=0,
                signer_id="signer_03",
                source="self_recorded",
                is_valid=False,
            ),
            _make_manifest_row(
                sample_id="s_no_signer",
                video_path=normalize_rel_path(
                    _make_synthetic_video(tmp_path / "raw/self/giup_do/nosig.mp4", seed=4),
                    tmp_path,
                ),
                label="giúp đỡ",
                class_id=0,
                signer_id="",  # missing signer_id
                source="self_recorded",
            ),
        ],
        columns=MANIFEST_COLUMNS,
    )
    self_csv.parent.mkdir(parents=True, exist_ok=True)
    self_df.to_csv(self_csv, index=False, encoding="utf-8-sig")

    unified_csv = tmp_path / "manifests" / "dataset_manifest.csv"
    report_path = tmp_path / "manifests" / "quality_report.json"

    merged = merge_manifests(
        manifest_paths=[kaggle_csv, self_csv],
        output_path=unified_csv,
        root_dir=tmp_path,
        class_list_path=REPO_ROOT / "configs" / "class_list.json",
        probe_videos=True,
    )
    assert merged.attrs["duplicates_removed"] == 2
    assert len(merged) == 5  # k_1, s_1, s_2, s_corrupt, s_no_signer
    assert int(merged.loc[merged["sample_id"] == "s_1", "class_id"].iloc[0]) == 0
    assert int(merged.loc[merged["sample_id"] == "s_2", "class_id"].iloc[0]) == 2
    assert bool(merged.loc[merged["sample_id"] == "s_corrupt", "is_valid"].iloc[0]) is False
    assert bool(merged.loc[merged["sample_id"] == "s_no_signer", "is_valid"].iloc[0]) is False

    report = audit_manifest(merged, report_path=report_path, root_dir=tmp_path)
    assert report["total_samples"] == 5
    assert report["valid_samples"] == 3
    assert report["invalid_samples"] == 2
    assert report["missing_signer_count"] == 1
    assert report["duplicates_removed"] == 2
    assert report["counts_per_source"] == {"kaggle_vsl": 1, "self_recorded": 2}
    assert report["duration_stats"]["mean"] is not None
    assert report["is_ready_for_split"] is False  # missing signer row present
    assert report_path.exists()

    # strict=True must raise when missing signer_id is encountered
    with pytest.raises(ValueError, match="missing metadata"):
        merge_manifests(manifest_paths=[self_csv], root_dir=tmp_path, strict=True)

    # Raw directory scanning helper works standalone
    scanned = scan_raw_directory(tmp_path / "raw/self", source="self_recorded", root_dir=tmp_path, class_list=classes)
    assert not scanned.empty


# ---------------------------------------------------------------------------
# 4. Step 3: Signer-Disjoint Split & Leakage Guard Tests
# ---------------------------------------------------------------------------


def test_step3_signer_disjoint_split_zero_leakage_and_determinism(
    synthetic_dataset: dict[str, Any],
) -> None:
    tmp_path: Path = synthetic_dataset["root"]
    splits_dir_1 = tmp_path / "splits_run1"
    splits_dir_2 = tmp_path / "splits_run2"

    report1 = split_dataset(
        manifest_path=synthetic_dataset["manifest_path"],
        output_dir=splits_dir_1,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )
    report2 = split_dataset(
        manifest_path=synthetic_dataset["manifest_path"],
        output_dir=splits_dir_2,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )

    assert report1["leakage_check_passed"] is True
    assert report1["signer_overlaps"] == {"train_val": [], "train_test": [], "val_test": []}

    train_df = load_manifest_df(splits_dir_1 / "train.csv", validate=True, require_split=True)
    val_df = load_manifest_df(splits_dir_1 / "val.csv", validate=True, require_split=True)
    test_df = load_manifest_df(splits_dir_1 / "test.csv", validate=True, require_split=True)

    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0
    assert len(train_df) + len(val_df) + len(test_df) == len(synthetic_dataset["manifest_df"])

    train_signers = set(train_df["signer_id"])
    val_signers = set(val_df["signer_id"])
    test_signers = set(test_df["signer_id"])
    assert train_signers & val_signers == set()
    assert train_signers & test_signers == set()
    assert val_signers & test_signers == set()
    assert verify_no_signer_leakage(train_df, val_df, test_df) is True

    # Determinism check
    assert report1["splits"]["train"]["signers"] == report2["splits"]["train"]["signers"]
    assert report1["splits"]["val"]["signers"] == report2["splits"]["val"]["signers"]
    assert report1["splits"]["test"]["signers"] == report2["splits"]["test"]["signers"]
    assert _file_sha256(splits_dir_1 / "train.csv") == _file_sha256(splits_dir_2 / "train.csv")
    assert _file_sha256(splits_dir_1 / "val.csv") == _file_sha256(splits_dir_2 / "val.csv")
    assert _file_sha256(splits_dir_1 / "test.csv") == _file_sha256(splits_dir_2 / "test.csv")


def test_step3_large_signer_pool_and_negative_edge_cases(tmp_path: Path) -> None:
    # Negative test: verify_no_signer_leakage MUST catch overlaps
    df_a = pd.DataFrame([_make_manifest_row(sample_id="a", video_path="a.mp4", label="chào", class_id=0, signer_id="signer_01", split="train")])
    df_b = pd.DataFrame([_make_manifest_row(sample_id="b", video_path="b.mp4", label="chào", class_id=0, signer_id="signer_01", split="val")])
    df_c = pd.DataFrame([_make_manifest_row(sample_id="c", video_path="c.mp4", label="chào", class_id=0, signer_id="signer_03", split="test")])

    with pytest.raises(ValueError, match="Signer leakage detected"):
        verify_no_signer_leakage(df_a, df_b, df_c)
    with pytest.raises(ValueError, match="Signer leakage detected"):
        verify_no_signer_leakage(df_a, df_c, df_b)
    with pytest.raises(ValueError, match="Signer leakage detected"):
        verify_no_signer_leakage(df_c, df_a, df_b)

    # Edge case: < 3 signers raises ValueError unless allow_2way / fallback_warning
    two_signer_df = pd.DataFrame(
        [
            _make_manifest_row(sample_id="s1", video_path="1.mp4", label="chào", class_id=0, signer_id="signer_01"),
            _make_manifest_row(sample_id="s2", video_path="2.mp4", label="chào", class_id=0, signer_id="signer_02"),
        ],
        columns=MANIFEST_COLUMNS,
    )
    two_signer_csv = tmp_path / "two_signers.csv"
    save_manifest_df(two_signer_df, two_signer_csv, validate=True)

    with pytest.raises(ValueError, match="At least 3 distinct signers"):
        split_dataset(two_signer_csv, output_dir=tmp_path / "splits_fail")

    rep_2way = split_dataset(two_signer_csv, output_dir=tmp_path / "splits_2way", allow_2way=True)
    assert rep_2way["leakage_check_passed"] is True
    assert rep_2way["splits"]["test"]["num_samples"] == 0
    assert len(rep_2way["warnings"]) >= 1

    # Edge case: 1 signer with fallback_warning
    one_signer_df = two_signer_df.iloc[[0]].copy()
    one_signer_csv = tmp_path / "one_signer.csv"
    save_manifest_df(one_signer_df, one_signer_csv, validate=True)
    rep_1way = split_dataset(one_signer_csv, output_dir=tmp_path / "splits_1way", fallback_warning=True)
    assert rep_1way["splits"]["train"]["num_samples"] == 1
    assert rep_1way["splits"]["val"]["num_samples"] == 0
    assert rep_1way["splits"]["test"]["num_samples"] == 0

    # Edge case: pre-augmented rows must be rejected before split
    pre_aug = two_signer_df.copy()
    pre_aug.loc[0, "is_augmented"] = True
    pre_aug.loc[0, "parent_sample_id"] = "orig_0"
    pre_aug_csv = tmp_path / "pre_aug.csv"
    pre_aug.to_csv(pre_aug_csv, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="pre-augmented"):
        split_dataset(pre_aug_csv, output_dir=tmp_path / "splits_preaug", allow_2way=True)

    # Larger signer pool (>8 signers -> exercises heuristic + hill-climbing branch)
    large_rows = [
        _make_manifest_row(
            sample_id=f"large_s{s:02d}_c{c}",
            video_path=f"v_{s}_{c}.mp4",
            label=f"class_{c}",
            class_id=c,
            signer_id=f"signer_{s:02d}",
        )
        for s in range(12)
        for c in range(3)
    ]
    large_df = pd.DataFrame(large_rows, columns=MANIFEST_COLUMNS)
    assignment, _ = stratified_group_split_signers(large_df, random_seed=42)
    assert len(assignment) == 12
    assert set(assignment.values()) == {"train", "val", "test"}


# ---------------------------------------------------------------------------
# 5. Step 4: Train-Only Augmentation & Hard Guard Tests
# ---------------------------------------------------------------------------


def test_step4_train_only_guard_rejects_val_test_and_mixed(
    synthetic_dataset: dict[str, Any],
) -> None:
    tmp_path: Path = synthetic_dataset["root"]
    splits_dir = tmp_path / "splits"
    split_dataset(
        manifest_path=synthetic_dataset["manifest_path"],
        output_dir=splits_dir,
        random_seed=42,
    )

    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"
    train_csv = splits_dir / "train.csv"

    # Hard refusal on val.csv
    with pytest.raises(ValueError, match=TRAIN_ONLY_ERROR_MSG):
        augment_train_manifest(
            train_manifest_path=val_csv,
            output_manifest_path=splits_dir / "should_not_exist_val.csv",
            output_video_dir=tmp_path / "aug_bad_val",
            project_root=tmp_path,
        )
    assert not (splits_dir / "should_not_exist_val.csv").exists()

    # Hard refusal on test.csv
    with pytest.raises(ValueError, match=TRAIN_ONLY_ERROR_MSG):
        augment_train_manifest(
            train_manifest_path=test_csv,
            output_manifest_path=splits_dir / "should_not_exist_test.csv",
            output_video_dir=tmp_path / "aug_bad_test",
            project_root=tmp_path,
        )
    assert not (splits_dir / "should_not_exist_test.csv").exists()

    # Hard refusal on mixed train + val CSV
    mixed_df = pd.concat([load_manifest_df(train_csv), load_manifest_df(val_csv)], ignore_index=True)
    mixed_csv = splits_dir / "mixed.csv"
    save_manifest_df(mixed_df, mixed_csv, validate=True, require_split=True)
    with pytest.raises(ValueError, match=TRAIN_ONLY_ERROR_MSG):
        augment_train_manifest(
            train_manifest_path=mixed_csv,
            output_manifest_path=splits_dir / "should_not_exist_mixed.csv",
            output_video_dir=tmp_path / "aug_bad_mixed",
            project_root=tmp_path,
        )

    # Hard refusal on unassigned (empty split) manifest
    with pytest.raises(ValueError, match=TRAIN_ONLY_ERROR_MSG):
        ensure_train_only_manifest(synthetic_dataset["manifest_df"])


def test_step4_train_augmentation_outputs_hflip_default_and_val_test_untouched(
    synthetic_dataset: dict[str, Any],
) -> None:
    tmp_path: Path = synthetic_dataset["root"]
    splits_dir = tmp_path / "splits"
    split_dataset(
        manifest_path=synthetic_dataset["manifest_path"],
        output_dir=splits_dir,
        random_seed=42,
    )

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"
    train_aug_csv = splits_dir / "train_augmented.csv"
    aug_vid_dir = tmp_path / "augmented" / "train"

    val_hash_before = _file_sha256(val_csv)
    test_hash_before = _file_sha256(test_csv)

    # Default allow_hflip MUST be False
    assert AugmentConfig().allow_hflip is False

    report = augment_train_manifest(
        train_manifest_path=train_csv,
        output_manifest_path=train_aug_csv,
        output_video_dir=aug_vid_dir,
        num_augmentations_per_video=2,
        random_seed=42,
        class_list_path=synthetic_dataset["class_list_path"],
        project_root=tmp_path,
    )

    assert report["allow_hflip"] is False
    assert report["train_only_guard_passed"] is True
    assert report["failed_videos"] == []

    orig_train_df = load_manifest_df(train_csv, validate=True, require_split=True)
    aug_train_df = load_manifest_df(train_aug_csv, validate=True, require_split=True)

    assert len(aug_train_df) == len(orig_train_df) * 3  # 1 original + 2 augmented per clip
    assert set(aug_train_df["split"]) == {"train"}

    only_orig = aug_train_df[~aug_train_df["is_augmented"].astype(bool)]
    only_aug = aug_train_df[aug_train_df["is_augmented"].astype(bool)]
    assert len(only_orig) == len(orig_train_df)
    assert len(only_aug) == len(orig_train_df) * 2

    orig_lookup = orig_train_df.set_index("sample_id")
    for _, row in only_aug.iterrows():
        parent_id = str(row["parent_sample_id"])
        assert parent_id in orig_lookup.index
        assert row["signer_id"] == orig_lookup.loc[parent_id, "signer_id"]
        assert int(row["class_id"]) == int(orig_lookup.loc[parent_id, "class_id"])
        assert row["aug_type"] != ""
        assert "hflip" not in str(row["aug_type"])

        # Every augmented video file must exist and be readable by OpenCV
        abs_vid = tmp_path / str(row["video_path"])
        assert abs_vid.exists()
        meta = probe_video_metadata(abs_vid)
        assert meta["is_valid"] is True
        assert meta["num_frames"] > 0

    # val.csv and test.csv MUST remain byte-for-byte untouched
    assert _file_sha256(val_csv) == val_hash_before
    assert _file_sha256(test_csv) == test_hash_before

    # Zero signer leakage still holds after augmentation
    verify_no_signer_leakage(
        aug_train_df,
        load_manifest_df(val_csv),
        load_manifest_df(test_csv),
    )

    # Explicit opt-in allow_hflip=True works when requested
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(10)]
    _, desc_hflip = apply_video_augmentations(
        frames,
        rng=np.random.default_rng(7),
        aug_cfg=AugmentConfig(allow_hflip=True, hflip_prob=1.0),
    )
    assert "hflip" in desc_hflip


# ---------------------------------------------------------------------------
# 6. End-to-End Production Pipeline Integration Tests (`run_production_pipeline`)
# ---------------------------------------------------------------------------


def test_end_to_end_production_pipeline_isolated(tmp_path: Path) -> None:
    """Verify `run_production_pipeline` end-to-end in an isolated `tmp_path` project root
    for both default Phương án 1 (`kaggle_only=None -> True`) and optional `--hybrid` (`False`),
    and confirm `--demo` has been completely removed from `scripts/run_quy_pipeline.py`."""
    # Confirm --demo flag no longer exists on CLI
    parser = build_pipeline_arg_parser()
    option_strings = {opt for action in parser._actions for opt in action.option_strings}
    assert "--demo" not in option_strings
    assert "--demo-dir" not in option_strings

    # Prepare an isolated project root with 3 classes x (3 QIPEDC + 3 VSL-v2 + 1 self signer)
    full_classes = load_class_list(REPO_ROOT / "configs" / "class_list.json")
    sub_classes = [dict(c, class_id=i) for i, c in enumerate(full_classes[:3])]
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cls_path = cfg_dir / "class_list.json"
    cls_path.write_text(
        json.dumps({"schema_version": "1.3", "classes": sub_classes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    base_cfg = load_config(REPO_ROOT / "configs" / "data_config.yaml")
    base_cfg["project"]["root_dir"] = str(tmp_path)
    cfg_path = cfg_dir / "data_config.yaml"
    cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    manifests_dir = tmp_path / "data" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    kaggle_rows: list[dict[str, Any]] = []
    kaggle_v2_rows: list[dict[str, Any]] = []
    self_rows: list[dict[str, Any]] = []
    seed_i = 0

    for cls in sub_classes:
        cid, label, slug = int(cls["class_id"]), str(cls["label"]), str(cls["slug"])
        for suffix, signer_id in (("B", "kaggle_signer_north"), ("T", "kaggle_signer_central"), ("N", "kaggle_signer_south")):
            seed_i += 1
            vp = tmp_path / "data/raw/kaggle_vsl/Dataset/Videos" / f"D{cid + 1:04d}{suffix}.mp4"
            _make_synthetic_video(vp, seed=seed_i)
            meta = probe_video_metadata(vp)
            kaggle_rows.append(
                _make_manifest_row(
                    sample_id=f"kaggle_d{cid + 1:04d}{suffix.lower()}",
                    video_path=normalize_rel_path(vp, tmp_path),
                    label=label,
                    class_id=cid,
                    signer_id=signer_id,
                    source="kaggle_vsl",
                    **{k: meta[k] for k in ("fps", "num_frames", "duration_sec", "width", "height", "is_valid")},
                )
            )
        for s_num in ("001", "002", "003"):
            seed_i += 1
            signer_id = f"vsl2_signer_{s_num}"
            vp = tmp_path / "data/raw/kaggle_vsl_v2/processed/processed/frame_splited/train" / label / f"v_{cid}_{s_num}.mp4"
            _make_synthetic_video(vp, seed=seed_i)
            meta = probe_video_metadata(vp)
            kaggle_v2_rows.append(
                _make_manifest_row(
                    sample_id=f"vsl2_{slug}_{s_num}",
                    video_path=normalize_rel_path(vp, tmp_path),
                    label=label,
                    class_id=cid,
                    signer_id=signer_id,
                    source="kaggle_vsl_v2",
                    **{k: meta[k] for k in ("fps", "num_frames", "duration_sec", "width", "height", "is_valid")},
                )
            )
        seed_i += 1
        vp_self = tmp_path / "data/raw/self_recorded" / slug / f"signer_01_{slug}.mp4"
        _make_synthetic_video(vp_self, seed=seed_i)
        meta = probe_video_metadata(vp_self)
        self_rows.append(
            _make_manifest_row(
                sample_id=f"self_signer_01_{slug}",
                video_path=normalize_rel_path(vp_self, tmp_path),
                label=label,
                class_id=cid,
                signer_id="signer_01",
                source="self_recorded",
                **{k: meta[k] for k in ("fps", "num_frames", "duration_sec", "width", "height", "is_valid")},
            )
        )

    save_manifest_df(pd.DataFrame(kaggle_rows, columns=MANIFEST_COLUMNS), manifests_dir / "kaggle_manifest.csv", validate=True)
    save_manifest_df(pd.DataFrame(kaggle_v2_rows, columns=MANIFEST_COLUMNS), manifests_dir / "kaggle_v2_manifest.csv", validate=True)
    save_manifest_df(pd.DataFrame(self_rows, columns=MANIFEST_COLUMNS), manifests_dir / "self_manifest.csv", validate=True)

    # 1) Default run (kaggle_only=None -> reads scope.kaggle_only: true -> production_kaggle_only)
    summary = run_production_pipeline(
        config_path=cfg_path,
        num_aug=1,
        seed=42,
        allow_hflip=False,
    )
    assert summary["mode"] == "production_kaggle_only"
    assert summary["kaggle_only"] is True
    assert summary["quality_report"]["mode"] == "kaggle_only"
    assert summary["quality_report"]["total_samples"] == 18  # 3 classes x 6 Kaggle signers; self ignored
    assert summary["scope_report"]["output_samples"] == 18
    assert summary["split_report"]["leakage_check_passed"] is True
    assert summary["split_report"]["total_signers"] == 6
    assert summary["augment_report"]["allow_hflip"] is False
    assert summary["augment_report"]["failed_videos"] == []
    assert (tmp_path / "data" / "pipeline_summary.json").exists()

    # 2) Hybrid run (kaggle_only=False -> includes self_recorded signer_01 too -> 21 samples, 7 signers)
    summary_hybrid = run_production_pipeline(
        config_path=cfg_path,
        skip_augment=True,
        seed=42,
        kaggle_only=False,
    )
    assert summary_hybrid["mode"] == "production"
    assert summary_hybrid["kaggle_only"] is False
    assert summary_hybrid["scope_report"]["output_samples"] == 21
    assert summary_hybrid["split_report"]["total_signers"] == 7
    assert summary_hybrid["split_report"]["leakage_check_passed"] is True


# ---------------------------------------------------------------------------
# 7. Dedicated 100% Kaggle-Only Workflow Tests (`--kaggle-only`)
# ---------------------------------------------------------------------------


def test_kaggle_only_stage_guards_and_3signer_split(tmp_path: Path) -> None:
    """Verify every stage's `--kaggle-only` behavior, guards, and 3-signer QIPEDC split."""
    # 1. Verify all 30 classes in configs/class_list.json have kaggle_clip_count >= 3
    full_classes = load_class_list(REPO_ROOT / "configs" / "class_list.json")
    assert len(full_classes) == 30
    assert all(int(c["kaggle_clip_count"]) >= 3 for c in full_classes)

    selected_classes = [dict(c, class_id=i) for i, c in enumerate(full_classes[:4])]
    cls_path = tmp_path / "kaggle_class_list.json"
    cls_path.write_text(
        json.dumps({"schema_version": "1.2", "classes": selected_classes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    regional = [
        ("B", "kaggle_signer_north"),
        ("T", "kaggle_signer_central"),
        ("N", "kaggle_signer_south"),
    ]
    kaggle_rows: list[dict[str, Any]] = []
    seed_i = 0
    for cls in selected_classes:
        for suffix, signer_id in regional:
            seed_i += 1
            vid = _make_synthetic_video(
                tmp_path / "raw/kaggle/Dataset/Videos" / f"D{cls['class_id'] + 1:04d}{suffix}.mp4",
                seed=seed_i,
            )
            meta = probe_video_metadata(vid)
            kaggle_rows.append(
                _make_manifest_row(
                    sample_id=f"kaggle_d{cls['class_id'] + 1:04d}{suffix.lower()}",
                    video_path=normalize_rel_path(vid, tmp_path),
                    label=cls["label"],
                    class_id=cls["class_id"],
                    signer_id=signer_id,
                    source="kaggle_vsl",
                    fps=meta["fps"],
                    num_frames=meta["num_frames"],
                    duration_sec=meta["duration_sec"],
                    width=meta["width"],
                    height=meta["height"],
                    is_valid=meta["is_valid"],
                )
            )

    # Add a 1-clip Kaggle class (should be dropped by kaggle_only scope min_signers>=3)
    single_vid = _make_synthetic_video(tmp_path / "raw/kaggle/Dataset/Videos/W9001.mp4", seed=90)
    single_meta = probe_video_metadata(single_vid)
    kaggle_rows.append(
        _make_manifest_row(
            sample_id="kaggle_w9001",
            video_path=normalize_rel_path(single_vid, tmp_path),
            label="nhãn đơn 1 clip",
            class_id=-1,
            signer_id="kaggle_signer_common_w",
            source="kaggle_vsl",
            fps=single_meta["fps"],
            num_frames=single_meta["num_frames"],
            duration_sec=single_meta["duration_sec"],
            width=single_meta["width"],
            height=single_meta["height"],
            is_valid=single_meta["is_valid"],
        )
    )

    # Create an extraneous self_recorded manifest that MUST be ignored when kaggle_only=True
    self_vid = _make_synthetic_video(tmp_path / "raw/self/chuc_mung/signer_01.mp4", seed=91)
    self_meta = probe_video_metadata(self_vid)
    self_rows = [
        _make_manifest_row(
            sample_id="self_should_be_ignored",
            video_path=normalize_rel_path(self_vid, tmp_path),
            label=selected_classes[0]["label"],
            class_id=0,
            signer_id="signer_01",
            source="self_recorded",
            fps=self_meta["fps"],
            num_frames=self_meta["num_frames"],
            duration_sec=self_meta["duration_sec"],
            width=self_meta["width"],
            height=self_meta["height"],
            is_valid=self_meta["is_valid"],
        )
    ]

    kaggle_csv = tmp_path / "manifests/kaggle_manifest.csv"
    self_csv = tmp_path / "manifests/self_manifest.csv"
    save_manifest_df(pd.DataFrame(kaggle_rows, columns=MANIFEST_COLUMNS), kaggle_csv, validate=True)
    save_manifest_df(pd.DataFrame(self_rows, columns=MANIFEST_COLUMNS), self_csv, validate=True)

    # Step 2: merge_manifests(..., kaggle_only=True) must yield ZERO self_recorded rows
    dataset_csv = tmp_path / "manifests/dataset_manifest.csv"
    quality_json = tmp_path / "manifests/quality_report.json"
    merged_df = merge_manifests(
        manifest_paths=[kaggle_csv, self_csv],
        output_path=dataset_csv,
        root_dir=tmp_path,
        class_list_path=cls_path,
        kaggle_only=True,
    )
    assert set(merged_df["source"]) == {"kaggle_vsl"}
    assert "self_recorded" not in set(merged_df["source"])
    assert len(merged_df) == 13  # 12 (4 classes x 3 regional signers) + 1 single-clip class

    quality_rep = audit_manifest(merged_df, report_path=quality_json, root_dir=tmp_path, kaggle_only=True)
    assert quality_rep["mode"] == "kaggle_only"
    assert quality_rep["num_classes_with_3plus_signers"] == 4
    assert quality_rep["is_ready_for_kaggle_only_split"] is True

    # Step 1: scope(..., kaggle_only=True) drops <3-signer class and keeps only 4x3=12 clips
    scoped_csv = tmp_path / "manifests/scoped_manifest.csv"
    scope_json = tmp_path / "manifests/scope_report.json"
    scoped_df, scope_rep = run_scope(
        input_manifest=dataset_csv,
        output_manifest=scoped_csv,
        report_path=scope_json,
        class_list_path=cls_path,
        mode="class_list",
        kaggle_only=True,
    )
    assert scope_rep["kaggle_only"] is True
    assert scope_rep["effective_min_samples"] >= 3
    assert scope_rep["effective_min_signers"] >= 3
    assert len(scoped_df) == 12
    assert set(scoped_df["source"]) == {"kaggle_vsl"}

    # Step 3: 3-signer QIPEDC regional split -> 100% class coverage in train/val/test, 0 leakage
    splits_dir = tmp_path / "splits"
    split_rep = split_dataset(
        manifest_path=scoped_csv,
        output_dir=splits_dir,
        config_path=REPO_ROOT / "configs/data_config.yaml",
        kaggle_only=True,
    )
    assert split_rep["mode"] == "kaggle_only"
    assert split_rep["leakage_check_passed"] is True
    assert split_rep["recommended_train_aug_copies"] == 4
    for sp in ("train", "val", "test"):
        assert split_rep["splits"][sp]["num_signers"] == 1
        assert split_rep["splits"][sp]["num_samples"] == 4
        assert split_rep["splits"][sp]["num_classes"] == 4
        assert split_rep["splits"][sp]["missing_classes"] == []

    # Guard: split_dataset(..., kaggle_only=True) rejects self_recorded input
    with pytest.raises(ValueError, match="kaggle_only"):
        split_dataset(self_csv, output_dir=tmp_path / "splits_bad", allow_2way=True, fallback_warning=True, kaggle_only=True)

    # Step 4: Train-only augmentation defaults to 4x in kaggle_only mode; val/test untouched & refused
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"
    val_hash = _file_sha256(val_csv)
    test_hash = _file_sha256(test_csv)

    with pytest.raises(ValueError, match=TRAIN_ONLY_ERROR_MSG):
        augment_train_manifest(val_csv, output_manifest_path=splits_dir / "bad_val.csv", output_video_dir=tmp_path / "bad_v", project_root=tmp_path, kaggle_only=True)

    aug_rep = augment_train_manifest(
        train_manifest_path=train_csv,
        output_manifest_path=splits_dir / "train_augmented.csv",
        output_video_dir=tmp_path / "augmented/train",
        config_path=REPO_ROOT / "configs/data_config.yaml",
        class_list_path=cls_path,
        project_root=tmp_path,
        kaggle_only=True,
    )
    assert aug_rep["mode"] == "kaggle_only"
    assert aug_rep["num_augmentations_per_video"] == 4
    assert aug_rep["allow_hflip"] is False
    assert aug_rep["original_train_samples"] == 4
    assert aug_rep["generated_augmented_samples"] == 16
    assert aug_rep["total_train_augmented_samples"] == 20
    assert aug_rep["failed_videos"] == []
    assert _file_sha256(val_csv) == val_hash
    assert _file_sha256(test_csv) == test_hash

    verify_no_signer_leakage(
        load_manifest_df(splits_dir / "train_augmented.csv"),
        load_manifest_df(val_csv),
        load_manifest_df(test_csv),
    )


def test_real_production_artifacts_and_no_demo_dir() -> None:
    """Verify real production outputs in `data/manifests/` and `data/splits/` and confirm
    `data/demo` has been completely deleted."""
    assert not (REPO_ROOT / "data" / "demo").exists(), "data/demo must be completely deleted!"

    manifests_dir = REPO_ROOT / "data" / "manifests"
    splits_dir = REPO_ROOT / "data" / "splits"

    scoped_csv = manifests_dir / "scoped_manifest.csv"
    assert scoped_csv.exists()
    scoped_df = load_manifest_df(scoped_csv, validate=True)
    assert len(scoped_df) == 210
    assert set(scoped_df["source"]) == {"kaggle_vsl", "kaggle_vsl_v2"}
    assert "self_recorded" not in set(scoped_df["source"])
    assert scoped_df["class_id"].nunique() == 30
    assert scoped_df["signer_id"].nunique() == 7
    assert scoped_df["is_valid"].astype(bool).all()

    train_df = load_manifest_df(splits_dir / "train.csv", validate=True, require_split=True)
    val_df = load_manifest_df(splits_dir / "val.csv", validate=True, require_split=True)
    test_df = load_manifest_df(splits_dir / "test.csv", validate=True, require_split=True)
    train_aug_df = load_manifest_df(splits_dir / "train_augmented.csv", validate=True, require_split=True)

    assert len(train_df) + len(val_df) + len(test_df) == len(scoped_df) == 210
    assert train_df["class_id"].nunique() == 30
    assert val_df["class_id"].nunique() == 30
    assert test_df["class_id"].nunique() == 30
    assert set(train_df["split"]) == {"train"}
    assert set(val_df["split"]) == {"val"}
    assert set(test_df["split"]) == {"test"}
    assert set(train_aug_df["split"]) == {"train"}
    assert len(train_aug_df) > len(train_df)

    # Zero signer leakage across train (including augmented), val, and test
    assert verify_no_signer_leakage(train_df, val_df, test_df) is True
    assert verify_no_signer_leakage(train_aug_df, val_df, test_df) is True

    split_report = json.loads((splits_dir / "split_report.json").read_text(encoding="utf-8"))
    assert split_report["mode"] == "kaggle_only"
    assert split_report["leakage_check_passed"] is True
    assert split_report["signer_overlaps"] == {"train_val": [], "train_test": [], "val_test": []}
    for sp in ("train", "val", "test"):
        assert split_report["splits"][sp]["num_classes"] == 30
        assert split_report["splits"][sp]["missing_classes"] == []

    aug_report = json.loads((splits_dir / "augment_report.json").read_text(encoding="utf-8"))
    assert aug_report["mode"] == "kaggle_only"
    assert aug_report["allow_hflip"] is False
    assert aug_report["train_only_guard_passed"] is True
    assert aug_report["failed_videos"] == []
    assert aug_report["total_train_augmented_samples"] == len(train_aug_df)

    summary_path = REPO_ROOT / "data" / "pipeline_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["mode"] == "production_kaggle_only"
    assert summary["kaggle_only"] is True


def test_kaggle_v2_parser_and_dual_dataset_merge(tmp_path: Path) -> None:
    """Verify VSL-v2 parser (VSL400 signers, Title Case NFC normalization, excluding processed_augmented)
    and dual-dataset merge/split/augment across `kaggle_vsl` + `kaggle_vsl_v2`."""
    full_classes = load_class_list(REPO_ROOT / "configs" / "class_list.json")
    v2_root = tmp_path / "raw_kaggle_v2"

    # VSL400 explicit signer metadata JSON
    front_json = v2_root / "raw" / "raw" / "VSL400" / "Part_1" / "split_1" / "front_view.json"
    front_json.parent.mkdir(parents=True, exist_ok=True)
    front_json.write_text(
        json.dumps(
            [
                {"video_id": "vsl_0001", "gloss": "Giúp đỡ", "signer_id": "001"},
                {"video_id": "vsl_0002", "gloss": "Giúp đỡ", "signer_id": "002"},
                {"video_id": "vsl_0003", "gloss": "Muốn", "signer_id": "003"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Canonical cropped front-view clips under processed/processed/frame_splited/{train,test}/{Title Case Gloss}/
    frame_dir = v2_root / "processed" / "processed" / "frame_splited"
    _make_synthetic_video(frame_dir / "train" / "Giúp đỡ" / "vsl_0001.mp4", seed=11)
    _make_synthetic_video(frame_dir / "test" / "Giúp đỡ" / "vsl_0002.mp4", seed=12)
    _make_synthetic_video(frame_dir / "train" / "Muốn" / "vsl_0003.mp4", seed=13)

    # Pre-augmented directory that MUST be excluded
    _make_synthetic_video(
        v2_root / "processed_augmented" / "train" / "Giúp đỡ" / "vsl_0001_aug.mp4",
        seed=99,
    )

    v2_df = parse_kaggle_v2_dataset_to_manifest(
        v2_root,
        project_root=tmp_path,
        class_list=full_classes,
        exclude_pre_augmented=True,
        probe_videos=True,
    )
    validate_manifest_df(v2_df)
    assert len(v2_df) == 3  # processed_augmented excluded
    assert set(v2_df["source"]) == {"kaggle_vsl_v2"}
    assert set(v2_df["signer_id"]) == {"vsl2_signer_001", "vsl2_signer_002", "vsl2_signer_003"}
    assert not any("augmented" in vp for vp in v2_df["video_path"])
    # Title Case normalized to canonical lowercase Vietnamese labels with matching class_id
    giup_rows = v2_df[v2_df["label"] == "giúp đỡ"]
    assert len(giup_rows) == 2
    assert set(giup_rows["class_id"]) == {0}
    muon_rows = v2_df[v2_df["label"] == "muốn"]
    assert len(muon_rows) == 1
    assert set(muon_rows["class_id"]) == {1}

    # Combine VSL-v2 with QIPEDC Kaggle v1 + extraneous self_recorded under kaggle_only=True
    v1_root = tmp_path / "raw_kaggle_v1"
    v1_vid = _make_synthetic_video(v1_root / "Dataset" / "Videos" / "D0001B.mp4", seed=14)
    v1_meta = probe_video_metadata(v1_vid)
    v1_df = pd.DataFrame(
        [
            _make_manifest_row(
                sample_id="kaggle_d0001b",
                video_path=normalize_rel_path(v1_vid, tmp_path),
                label="giúp đỡ",
                class_id=0,
                signer_id="kaggle_signer_north",
                source="kaggle_vsl",
                fps=v1_meta["fps"],
                num_frames=v1_meta["num_frames"],
                duration_sec=v1_meta["duration_sec"],
                width=v1_meta["width"],
                height=v1_meta["height"],
                is_valid=v1_meta["is_valid"],
            )
        ],
        columns=MANIFEST_COLUMNS,
    )
    self_vid = _make_synthetic_video(tmp_path / "raw_self" / "giup_do" / "s01.mp4", seed=15)
    self_meta = probe_video_metadata(self_vid)
    self_df = pd.DataFrame(
        [
            _make_manifest_row(
                sample_id="self_s01",
                video_path=normalize_rel_path(self_vid, tmp_path),
                label="giúp đỡ",
                class_id=0,
                signer_id="signer_01",
                source="self_recorded",
                fps=self_meta["fps"],
                num_frames=self_meta["num_frames"],
                duration_sec=self_meta["duration_sec"],
                width=self_meta["width"],
                height=self_meta["height"],
                is_valid=self_meta["is_valid"],
            )
        ],
        columns=MANIFEST_COLUMNS,
    )

    v1_csv = tmp_path / "manifests" / "kaggle_manifest.csv"
    v2_csv = tmp_path / "manifests" / "kaggle_v2_manifest.csv"
    self_csv = tmp_path / "manifests" / "self_manifest.csv"
    save_manifest_df(v1_df, v1_csv, validate=True)
    save_manifest_df(v2_df, v2_csv, validate=True)
    save_manifest_df(self_df, self_csv, validate=True)

    merged_csv = tmp_path / "manifests" / "dual_dataset_manifest.csv"
    merged_df = merge_manifests(
        manifest_paths=[v1_csv, v2_csv, self_csv],
        output_path=merged_csv,
        root_dir=tmp_path,
        class_list_path=REPO_ROOT / "configs" / "class_list.json",
        kaggle_only=True,
    )
    assert len(merged_df) == 4
    assert set(merged_df["source"]) == {"kaggle_vsl", "kaggle_vsl_v2"}
    assert "self_recorded" not in set(merged_df["source"])

    qrep = audit_manifest(merged_df, root_dir=tmp_path, kaggle_only=True)
    assert qrep["mode"] == "kaggle_only"
    assert set(qrep["kaggle_sources_present"]) == {"kaggle_vsl", "kaggle_vsl_v2"}
    assert "giúp đỡ" in qrep["classes_with_3plus_signers"]
    assert qrep["is_ready_for_kaggle_only_split"] is True


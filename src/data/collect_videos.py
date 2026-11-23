"""Interactive webcam video collector & labeler for multi-signer VSL data collection.

Step 2 of Quy's data pipeline ("Thu thập và nhãn - tự quay, nhiều người").
Separates pure headless helpers (slugification, deterministic naming, class lookup,
manifest row creation/appending, video writing, framing guide drawing) from live
webcam capture I/O so all core logic can be unit-tested without a physical camera.
"""

from __future__ import annotations

import argparse
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd

from src.data.schema import (
    MANIFEST_COLUMNS,
    load_class_list,
    load_config,
    normalize_rel_path,
    probe_video_metadata,
    validate_manifest_df,
)

DEFAULT_CONFIG_PATH = Path("configs/data_config.yaml")
DEFAULT_CLASS_LIST_PATH = Path("configs/class_list.json")
DEFAULT_OUTPUT_DIR = Path("data/raw/self_recorded")
DEFAULT_MANIFEST_PATH = Path("data/manifests/self_manifest.csv")


# ---------------------------------------------------------------------------
# Pure headless helper functions (unit-testable without a webcam)
# ---------------------------------------------------------------------------


def strip_vietnamese_diacritics(text: str) -> str:
    """Convert Vietnamese text to ASCII while preserving readability."""
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def slugify(text: str) -> str:
    """Create a deterministic filesystem-safe ASCII snake_case slug."""
    ascii_text = strip_vietnamese_diacritics(str(text)).lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    if not slug:
        raise ValueError(f"Cannot generate non-empty slug from {text!r}")
    return slug


def normalize_signer_id(signer_id: str) -> str:
    """Validate and normalize a signer identifier.

    Signer IDs must be non-empty and safe for manifests and filenames
    (alphanumeric, underscores, or hyphens).
    """
    if signer_id is None:
        raise ValueError("signer_id must not be None or empty.")
    cleaned = strip_vietnamese_diacritics(str(signer_id)).strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_-]+", "", cleaned)
    if not cleaned:
        raise ValueError(f"Invalid signer_id {signer_id!r}: must contain alphanumeric characters.")
    return cleaned


def resolve_class_entry(
    classes: Sequence[dict[str, Any]],
    class_id: int | None = None,
    class_label: str | None = None,
) -> dict[str, Any]:
    """Look up a class entry from class_list by class_id, label, or slug.

    Raises:
        ValueError: If the class list is empty, neither identifier is provided,
            the class cannot be found, or class_id and class_label refer to
            different classes.
    """
    if not classes:
        raise ValueError("Class list is empty.")
    if class_id is None and (class_label is None or not str(class_label).strip()):
        raise ValueError("Either class_id or class_label must be provided.")

    by_id: dict[str, Any] | None = None
    if class_id is not None:
        for entry in classes:
            if int(entry["class_id"]) == int(class_id):
                by_id = dict(entry)
                break
        if by_id is None:
            raise ValueError(f"class_id={class_id} not found in class_list.")

    by_label: dict[str, Any] | None = None
    if class_label is not None and str(class_label).strip():
        query_raw = str(class_label).strip().lower()
        query_slug = slugify(query_raw)
        for entry in classes:
            entry_label = str(entry.get("label", "")).strip().lower()
            entry_slug = str(entry.get("slug") or slugify(entry_label)).strip().lower()
            if query_raw == entry_label or query_raw == entry_slug or query_slug == entry_slug:
                by_label = dict(entry)
                break
        if by_label is None:
            raise ValueError(f"class_label={class_label!r} not found in class_list.")

    if by_id is not None and by_label is not None:
        if int(by_id["class_id"]) != int(by_label["class_id"]):
            raise ValueError(
                f"Conflicting class_id={class_id} ({by_id['label']!r}) and "
                f"class_label={class_label!r} (class_id={by_label['class_id']})."
            )
        resolved = by_id
    else:
        resolved = by_id if by_id is not None else by_label  # type: ignore[assignment]

    if not resolved.get("slug"):
        resolved["slug"] = slugify(str(resolved["label"]))
    resolved["class_id"] = int(resolved["class_id"])
    return resolved


def make_timestamp(dt: datetime | None = None) -> str:
    """Return a deterministic millisecond-precision timestamp string."""
    now = dt or datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"


def build_video_filename(
    signer_id: str,
    class_slug: str,
    timestamp: str | None = None,
    ext: str = ".mp4",
) -> str:
    """Build deterministic filename `<signer_id>_<class_slug>_<timestamp>.mp4`."""
    norm_signer = normalize_signer_id(signer_id)
    norm_slug = slugify(class_slug)
    ts = timestamp or make_timestamp()
    ext_clean = ext if ext.startswith(".") else f".{ext}"
    return f"{norm_signer}_{norm_slug}_{ts}{ext_clean}"


def build_video_path(
    output_dir: str | Path,
    signer_id: str,
    class_slug: str,
    timestamp: str | None = None,
    ext: str = ".mp4",
) -> Path:
    """Build deterministic path `<output_dir>/<class_slug>/<signer_id>_<class_slug>_<timestamp>.mp4`."""
    norm_slug = slugify(class_slug)
    filename = build_video_filename(signer_id, norm_slug, timestamp=timestamp, ext=ext)
    return Path(output_dir) / norm_slug / filename


def build_sample_id(
    signer_id: str,
    class_slug: str,
    timestamp: str | None = None,
    prefix: str = "self",
) -> str:
    """Build deterministic unique sample_id for self-recorded clips."""
    norm_signer = normalize_signer_id(signer_id)
    norm_slug = slugify(class_slug)
    ts = timestamp or make_timestamp()
    return f"{prefix}_{norm_signer}_{norm_slug}_{ts}"


def write_frames_to_video(
    frames: Sequence[np.ndarray],
    output_path: str | Path,
    fps: float = 30.0,
    fourcc: str = "mp4v",
) -> Path:
    """Write a sequence of BGR NumPy frames to an .mp4 video file."""
    if not frames:
        raise ValueError("Cannot write video: frames sequence is empty.")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    codec = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(out), codec, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out} with fourcc={fourcc!r}.")

    try:
        for frame in frames:
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (int(width), int(height)))
            writer.write(frame)
    finally:
        writer.release()

    return out


def create_manifest_row(
    video_path: str | Path,
    class_entry: dict[str, Any],
    signer_id: str,
    root_dir: str | Path | None = None,
    sample_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    source: str = "self_recorded",
) -> dict[str, Any]:
    """Create a schema-compliant manifest row dict matching MANIFEST_COLUMNS."""
    norm_signer = normalize_signer_id(signer_id)
    video_p = Path(video_path)
    meta = dict(metadata) if metadata is not None else probe_video_metadata(video_p)

    if root_dir is not None:
        rel_video_path = normalize_rel_path(video_p, root_dir)
    else:
        rel_video_path = video_p.as_posix()

    class_slug = str(class_entry.get("slug") or slugify(str(class_entry["label"])))
    resolved_sample_id = sample_id or f"self_{video_p.stem}"
    if not resolved_sample_id:
        resolved_sample_id = build_sample_id(norm_signer, class_slug)

    row: dict[str, Any] = {
        "sample_id": resolved_sample_id,
        "video_path": rel_video_path,
        "label": str(class_entry["label"]),
        "class_id": int(class_entry["class_id"]),
        "signer_id": norm_signer,
        "source": str(source),
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
    return {col: row[col] for col in MANIFEST_COLUMNS}


def append_manifest_row(
    manifest_path: str | Path,
    row: dict[str, Any],
    validate: bool = True,
) -> pd.DataFrame:
    """Append a single manifest row to a CSV file (creating it if needed)."""
    manifest_p = Path(manifest_path)
    manifest_p.parent.mkdir(parents=True, exist_ok=True)

    missing_cols = [c for c in MANIFEST_COLUMNS if c not in row]
    if missing_cols:
        raise ValueError(f"Manifest row is missing required columns: {missing_cols}")

    new_df = pd.DataFrame([{col: row[col] for col in MANIFEST_COLUMNS}], columns=MANIFEST_COLUMNS)

    if manifest_p.exists() and manifest_p.stat().st_size > 0:
        existing_df = pd.read_csv(manifest_p, dtype={"split": str, "parent_sample_id": str, "aug_type": str})
        for col in MANIFEST_COLUMNS:
            if col not in existing_df.columns:
                existing_df[col] = ""
        existing_df = existing_df[MANIFEST_COLUMNS]
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    for text_col in ("split", "parent_sample_id", "aug_type"):
        combined[text_col] = combined[text_col].fillna("").astype(str)

    if validate:
        validate_manifest_df(combined)

    combined.to_csv(manifest_p, index=False, encoding="utf-8-sig")
    return combined


def draw_framing_guide(
    frame: np.ndarray,
    status_text: str = "",
    prompt_text: str = "",
    countdown: int | None = None,
    progress: float | None = None,
    is_recording: bool = False,
) -> np.ndarray:
    """Draw upper-body & hand framing guides for MediaPipe Holistic onto a frame copy.

    Never mutates the input frame so clean raw frames can be saved to disk.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Outer signing zone box (10%-90% width, 6%-94% height) for head + torso + both hands
    box_x1, box_y1 = int(w * 0.10), int(h * 0.06)
    box_x2, box_y2 = int(w * 0.90), int(h * 0.94)
    box_color = (0, 0, 255) if is_recording else (0, 220, 140)
    cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), box_color, 2)

    # Head oval guide (centered top-middle)
    head_center = (int(w * 0.50), int(h * 0.24))
    head_axes = (max(1, int(w * 0.085)), max(1, int(h * 0.125)))
    cv2.ellipse(canvas, head_center, head_axes, 0, 0, 360, (255, 210, 0), 2)

    # Shoulder / upper-torso guide line
    shoulder_y = int(h * 0.42)
    cv2.line(canvas, (int(w * 0.28), shoulder_y), (int(w * 0.72), shoulder_y), (255, 210, 0), 1)

    # Top HUD bar
    banner_h = max(36, int(h * 0.08))
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

    if status_text:
        safe_status = strip_vietnamese_diacritics(status_text)
        cv2.putText(
            canvas,
            safe_status,
            (14, int(banner_h * 0.68)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if is_recording:
        cv2.circle(canvas, (w - 30, banner_h // 2), 10, (0, 0, 255), -1)
        cv2.putText(
            canvas,
            "REC",
            (w - 92, int(banner_h * 0.68)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Large countdown digit in center
    if countdown is not None and countdown > 0:
        label = str(countdown)
        font_scale = max(2.0, h / 180.0)
        thickness = max(4, int(font_scale * 2))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = (w - tw) // 2
        ty = (h + th) // 2
        cv2.putText(canvas, label, (tx + 3, ty + 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3)
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

    # Recording progress bar near bottom
    if progress is not None:
        clamped = max(0.0, min(1.0, float(progress)))
        bar_x1, bar_x2 = int(w * 0.12), int(w * 0.88)
        bar_y1, bar_y2 = h - 54, h - 34
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (60, 60, 60), -1)
        fill_x = bar_x1 + int((bar_x2 - bar_x1) * clamped)
        cv2.rectangle(canvas, (bar_x1, bar_y1), (fill_x, bar_y2), (0, 0, 255), -1)
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (255, 255, 255), 1)

    # Bottom prompt bar
    if prompt_text:
        safe_prompt = strip_vietnamese_diacritics(prompt_text)
        cv2.rectangle(canvas, (0, h - 28), (w, h), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            safe_prompt,
            (14, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    return canvas


# ---------------------------------------------------------------------------
# Interactive terminal & OpenCV webcam collection flow
# ---------------------------------------------------------------------------


def prompt_signer_id() -> str:
    """Prompt interactively for a non-empty signer_id."""
    while True:
        raw = input("Nhap signer_id (vi du: signer_01): ").strip()
        try:
            return normalize_signer_id(raw)
        except ValueError as exc:
            print(f"[!] {exc}")


def prompt_class_selection(classes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Display an interactive menu from class_list.json and return chosen class entry."""
    print("\n=== Danh sach ky hieu VSL (class_list.json) ===")
    for entry in classes:
        cid = entry["class_id"]
        label = entry["label"]
        slug = entry.get("slug") or slugify(str(label))
        cat = entry.get("category", "")
        cat_str = f" [{cat}]" if cat else ""
        print(f"  {cid:>3d}: {label} ({slug}){cat_str}")

    while True:
        choice = input("\nChon class_id hoac nhap nhan ky hieu: ").strip()
        if not choice:
            continue
        try:
            if choice.isdigit():
                return resolve_class_entry(classes, class_id=int(choice))
            return resolve_class_entry(classes, class_label=choice)
        except ValueError as exc:
            print(f"[!] {exc}")


def run_collection_session(
    signer_id: str,
    class_entry: dict[str, Any],
    num_takes: int = 5,
    duration: float = 2.5,
    fps: int = 30,
    camera_index: int = 0,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    root_dir: str | Path | None = None,
    countdown_sec: float = 3.0,
    width: int = 1280,
    height: int = 720,
) -> list[dict[str, Any]]:
    """Run an interactive OpenCV webcam capture session for one signer and class.

    Returns a list of manifest rows that were accepted and appended.
    """
    norm_signer = normalize_signer_id(signer_id)
    class_slug = str(class_entry.get("slug") or slugify(str(class_entry["label"])))
    class_label = str(class_entry["label"])
    class_id = int(class_entry["class_id"])
    project_root = Path(root_dir) if root_dir is not None else Path.cwd()

    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        raise RuntimeError(
            f"Khong the mo webcam (camera_index={camera_index}). "
            "Kiem tra ket noi camera hoac thu --camera-index khac."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))

    window_name = "VSL Collector - Quy Pipeline (Step 2)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    accepted_rows: list[dict[str, Any]] = []
    take_idx = 1

    print(
        f"\n[INFO] Bat dau thu thap: signer={norm_signer}, class={class_id}:{class_label} "
        f"({class_slug}), takes={num_takes}, duration={duration:.1f}s @ {fps}fps"
    )
    if class_entry.get("handedness"):
        print(f"[INFO] Tay thuc hien: {class_entry['handedness']}")
    if class_entry.get("action_description"):
        print(f"[INFO] Mo ta dong tac VSL: {class_entry['action_description']}")
    print("[INFO] Cua so camera: SPACE/Enter = bat dau quay, s/Enter = luu, r = quay lai, q = thoat.")

    try:
        while take_idx <= num_takes:
            status_header = (
                f"Signer: {norm_signer} | Class [{class_id}]: {class_slug} | "
                f"Take {take_idx}/{num_takes}"
            )

            # 1) Live preview — wait for SPACE/Enter to trigger countdown or q to quit
            start_requested = False
            quit_requested = False
            while not start_requested and not quit_requested:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Khong doc duoc khung hinh tu webcam.")
                display = draw_framing_guide(
                    frame,
                    status_text=status_header,
                    prompt_text="[SPACE/Enter] Bat dau dem nguoc | [q] Thoat",
                )
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord(" "), 13, 10):
                    start_requested = True
                elif key in (ord("q"), ord("Q"), 27):
                    quit_requested = True

            if quit_requested:
                break

            # 2) Countdown 3-2-1
            countdown_end = time.perf_counter() + float(countdown_sec)
            while True:
                remaining = countdown_end - time.perf_counter()
                if remaining <= 0:
                    break
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Mat ket noi webcam trong khi dem nguoc.")
                display = draw_framing_guide(
                    frame,
                    status_text=status_header,
                    countdown=int(np.ceil(remaining)),
                    prompt_text="Giu tu the nghi (Neutral pose) trong khung xanh...",
                )
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    quit_requested = True
                    break

            if quit_requested:
                break

            # 3) Record clean raw frames for `duration` seconds
            recorded_frames: list[np.ndarray] = []
            rec_start = time.perf_counter()
            while True:
                elapsed = time.perf_counter() - rec_start
                if elapsed >= float(duration):
                    break
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Mat ket noi webcam trong khi ghi hinh.")
                recorded_frames.append(frame.copy())
                display = draw_framing_guide(
                    frame,
                    status_text=status_header,
                    progress=elapsed / max(float(duration), 1e-6),
                    prompt_text=f"Dang quay... {elapsed:.1f}s / {duration:.1f}s (khong co overlay trong video luu)",
                    is_recording=True,
                )
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    quit_requested = True
                    break

            actual_elapsed = max(time.perf_counter() - rec_start, 1e-3)
            if quit_requested:
                break

            if not recorded_frames:
                print("[WARN] Khong ghi duoc khung hinh nao, vui long quay lai.")
                continue

            # 4) Post-record review loop: Accept (s/Enter), Redo (r), Quit (q)
            decision: str | None = None
            frame_cursor = 0
            playback_delay_ms = max(15, int(1000 / max(fps, 1)))
            while decision is None:
                review_frame = recorded_frames[frame_cursor % len(recorded_frames)]
                frame_cursor += 1
                display = draw_framing_guide(
                    review_frame,
                    status_text=f"REVIEW | {status_header} ({len(recorded_frames)} frames)",
                    prompt_text="[s / Enter] Luu (Accept) | [r] Quay lai (Redo) | [q] Thoat (Quit)",
                )
                cv2.imshow(window_name, display)
                key = cv2.waitKey(playback_delay_ms) & 0xFF
                if key in (ord("s"), ord("S"), 13, 10):
                    decision = "accept"
                elif key in (ord("r"), ord("R")):
                    decision = "redo"
                elif key in (ord("q"), ord("Q"), 27):
                    decision = "quit"

            if decision == "quit":
                break
            if decision == "redo":
                print(f"[INFO] Bo qua take {take_idx}, chuan bi quay lai...")
                continue

            # Accept & save clip + manifest row
            ts = make_timestamp()
            video_out = build_video_path(output_dir, norm_signer, class_slug, timestamp=ts)
            sample_id = build_sample_id(norm_signer, class_slug, timestamp=ts)
            effective_fps = len(recorded_frames) / actual_elapsed
            save_fps = float(fps) if abs(effective_fps - fps) <= 5.0 else float(round(effective_fps, 2))
            save_fps = max(5.0, save_fps)

            write_frames_to_video(recorded_frames, video_out, fps=save_fps)
            row = create_manifest_row(
                video_path=video_out,
                class_entry=class_entry,
                signer_id=norm_signer,
                root_dir=project_root,
                sample_id=sample_id,
            )
            append_manifest_row(manifest_path, row)
            accepted_rows.append(row)
            print(
                f"[SAVED] Take {take_idx}/{num_takes}: {row['video_path']} "
                f"(frames={row['num_frames']}, duration={row['duration_sec']:.2f}s, valid={row['is_valid']})"
            )
            take_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[DONE] Da luu {len(accepted_rows)}/{num_takes} clip vao {manifest_path}.")
    return accepted_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive OpenCV webcam recorder & labeler for multi-signer VSL collection."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to central YAML config (default: configs/data_config.yaml).",
    )
    parser.add_argument(
        "--class-list",
        type=Path,
        default=None,
        help="Path to class_list.json (defaults to config or configs/class_list.json).",
    )
    parser.add_argument(
        "--signer-id",
        type=str,
        default=None,
        help="Signer identifier (e.g., signer_01). Prompted interactively if omitted.",
    )
    parser.add_argument(
        "--class-label",
        type=str,
        default=None,
        help="Target VSL class label or slug (from class_list.json).",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Target VSL class_id (from class_list.json).",
    )
    parser.add_argument(
        "--num-takes",
        type=int,
        default=None,
        help="Number of takes to record for this class (default: 5).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Clip duration in seconds (default: 2.5).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Target recording FPS (default: 30).",
    )
    parser.add_argument(
        "--countdown",
        type=float,
        default=None,
        help="Countdown duration before recording in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="OpenCV camera index (default: from config or 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store recorded videos (default: data/raw/self_recorded).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Output CSV manifest path (default: data/manifests/self_manifest.csv).",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Project root for relative video_path normalization (default: current working directory).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg: dict[str, Any] = {}
    if args.config and Path(args.config).exists():
        cfg = load_config(args.config)

    paths_cfg = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    rec_cfg = (
        cfg.get("collection")
        or cfg.get("recording")
        or cfg.get("video_recording")
        or {}
    )
    if not isinstance(rec_cfg, dict):
        rec_cfg = {}

    class_list_path = args.class_list or Path(
        paths_cfg.get("class_list_path", paths_cfg.get("class_list", DEFAULT_CLASS_LIST_PATH))
    )
    output_dir = args.output_dir or Path(
        paths_cfg.get("raw_self_dir", paths_cfg.get("self_recorded_dir", DEFAULT_OUTPUT_DIR))
    )
    manifest_path = args.manifest or Path(
        paths_cfg.get("self_manifest", DEFAULT_MANIFEST_PATH)
    )

    num_takes = (
        args.num_takes
        if args.num_takes is not None
        else int(
            rec_cfg.get(
                "takes_per_signer_per_class",
                rec_cfg.get("num_takes", rec_cfg.get("default_num_takes", 3)),
            )
        )
    )
    duration = args.duration if args.duration is not None else float(rec_cfg.get("duration_sec", rec_cfg.get("duration", 2.5)))
    fps = args.fps if args.fps is not None else int(rec_cfg.get("fps", 30))
    countdown = args.countdown if args.countdown is not None else float(rec_cfg.get("countdown_sec", 3.0))
    width = int(rec_cfg.get("width", 640))
    height = int(rec_cfg.get("height", 480))
    camera_index = args.camera_index if args.camera_index is not None else int(rec_cfg.get("camera_index", 0))

    classes = load_class_list(class_list_path)
    signer_id = normalize_signer_id(args.signer_id) if args.signer_id else prompt_signer_id()

    if args.class_id is not None or args.class_label is not None:
        class_entry = resolve_class_entry(classes, class_id=args.class_id, class_label=args.class_label)
    else:
        class_entry = prompt_class_selection(classes)

    run_collection_session(
        signer_id=signer_id,
        class_entry=class_entry,
        num_takes=num_takes,
        duration=duration,
        fps=fps,
        camera_index=camera_index,
        output_dir=output_dir,
        manifest_path=manifest_path,
        root_dir=args.root_dir,
        countdown_sec=countdown,
        width=width,
        height=height,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

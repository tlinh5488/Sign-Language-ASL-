"""Trực quan hóa mẫu gốc vs mẫu augmented từ ``data/features/train.npz``.

Sinh ảnh PNG kiểm tra khung xương (pose thân trên + tay trái + tay phải + quỹ đạo
cổ tay) tại ``docs/figures/sample_augmentation_check.png`` để xác nhận không bị
méo tỉ lệ hay đảo trục sau Bước 2.2-2.4.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Kết nối khung xương thân trên (chỉ số 0..24 theo MediaPipe Pose)
POSE_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (0, 11), (0, 12),
]

# Kết nối 21 điểm bàn tay (chỉ số 0..20 theo MediaPipe Hands)
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _unpack_frame(vec207: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pose = vec207[0:75].reshape(25, 3)
    lh = vec207[75:138].reshape(21, 3)
    rh = vec207[138:201].reshape(21, 3)
    lw_g = vec207[201:204]
    rw_g = vec207[204:207]
    return pose, lh, rh, lw_g, rw_g


def _plot_skeleton_frame(ax, vec207: np.ndarray, title: str) -> None:
    pose, lh, rh, lw_g, rw_g = _unpack_frame(vec207)

    # Thân trên (trục y đảo ngược để đúng hướng nhìn camera)
    for i, j in POSE_EDGES:
        ax.plot([pose[i, 0], pose[j, 0]], [-pose[i, 1], -pose[j, 1]], color="#475569", lw=1.8)
    ax.scatter(pose[:, 0], -pose[:, 1], s=14, color="#334155", zorder=3)

    # Đặt bàn tay đã chuẩn hóa thu nhỏ tại vị trí cổ tay toàn cục để dễ quan sát
    hand_scale = 0.18
    for hand, wrist_g, color, label in (
        (lh, lw_g, "#2563eb", "Tay trái"),
        (rh, rw_g, "#dc2626", "Tay phải"),
    ):
        if np.any(np.abs(hand) > 0):
            pts = hand * hand_scale + wrist_g
            for i, j in HAND_EDGES:
                ax.plot([pts[i, 0], pts[j, 0]], [-pts[i, 1], -pts[j, 1]], color=color, lw=1.2)
            ax.scatter(pts[:, 0], -pts[:, 1], s=8, color=color, zorder=4, label=label)

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def main() -> int:
    npz_path = REPO_ROOT / "data" / "features" / "train.npz"
    out_path = REPO_ROOT / "docs" / "figures" / "sample_augmentation_check.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path, allow_pickle=False) as z:
        X = z["X"]
        sample_ids = z["sample_ids"]
        labels = z["labels"]
        is_aug = z["is_augmented"]

    # Tìm cặp mẫu gốc và mẫu augmented đầu tiên cùng prefix
    orig_map = {
        str(sid): idx
        for idx, (sid, aug) in enumerate(zip(sample_ids, is_aug))
        if not bool(aug)
    }
    pair = None
    for idx, (sid, aug) in enumerate(zip(sample_ids, is_aug)):
        if bool(aug):
            base_id = str(sid).split("_aug")[0]
            if base_id in orig_map:
                pair = (orig_map[base_id], idx, base_id, str(sid), str(labels[idx]))
                break

    if pair is None:
        raise RuntimeError("Không tìm thấy cặp mẫu gốc - augmented trong train.npz")

    orig_idx, aug_idx, orig_id, aug_id, label = pair
    frame_indices = [0, 12, 24, 36, 47]

    fig, axes = plt.subplots(2, len(frame_indices) + 1, figsize=(16, 6.2), dpi=150)
    fig.suptitle(
        f"Kiểm tra trực quan (48, 207) — Lớp '{label}' | Gốc: {orig_id} vs Augmented: {aug_id}",
        fontsize=12,
        fontweight="bold",
    )

    for row, (sample_idx, row_title) in enumerate(
        [(orig_idx, "Mẫu gốc"), (aug_idx, "Mẫu augmented")]
    ):
        seq = X[sample_idx]
        for col, f_idx in enumerate(frame_indices):
            _plot_skeleton_frame(axes[row, col], seq[f_idx], f"{row_title} — Frame {f_idx}")

        # Cột cuối: quỹ đạo cổ tay toàn cục (x, y) theo 48 frame
        ax_traj = axes[row, -1]
        lw_traj = seq[:, 201:203]
        rw_traj = seq[:, 204:206]
        lw_ok = np.any(np.abs(lw_traj) > 0, axis=1)
        rw_ok = np.any(np.abs(rw_traj) > 0, axis=1)
        if lw_ok.any():
            ax_traj.plot(lw_traj[lw_ok, 0], -lw_traj[lw_ok, 1], color="#2563eb", lw=1.6, label="Cổ tay trái")
        if rw_ok.any():
            ax_traj.plot(rw_traj[rw_ok, 0], -rw_traj[rw_ok, 1], color="#dc2626", lw=1.6, label="Cổ tay phải")
        ax_traj.set_xlim(-1.4, 1.4)
        ax_traj.set_ylim(-1.4, 1.2)
        ax_traj.set_aspect("equal")
        ax_traj.grid(True, alpha=0.25)
        ax_traj.set_title(f"{row_title} — Quỹ đạo cổ tay", fontsize=9)
        ax_traj.legend(loc="lower right", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu ảnh kiểm tra trực quan: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Phân tích tỷ lệ phát hiện bàn tay theo từng lớp trên tập test.

Đọc ``data/landmarks/test_landmarks_index.csv``, thống kê theo 30 lớp ký hiệu
và xuất báo cáo Markdown + biểu đồ PNG để nhóm nắm trước các lớp dễ mất tay.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    csv_path = REPO_ROOT / "data" / "landmarks" / "test_landmarks_index.csv"
    df = pd.read_csv(csv_path, encoding="utf-8")

    grouped = (
        df.groupby(["class_id", "label"], as_index=False)
        .agg(
            num_samples=("sample_id", "count"),
            hand_mean=("hand_detect_ratio", "mean"),
            hand_min=("hand_detect_ratio", "min"),
            left_mean=("left_hand_detect_ratio", "mean"),
            right_mean=("right_hand_detect_ratio", "mean"),
            low_hand_samples=("hand_detect_ratio", lambda s: int((s < 0.70).sum())),
        )
        .sort_values("hand_mean", ascending=True)
        .reset_index(drop=True)
    )

    # Xuất biểu đồ cột ngang theo 30 lớp
    fig_path = REPO_ROOT / "docs" / "figures" / "test_hand_detection_by_class.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=150)
    y_pos = range(len(grouped))
    colors = ["#dc2626" if v < 0.85 else "#2563eb" for v in grouped["hand_mean"]]
    ax.barh(list(y_pos), grouped["hand_mean"] * 100.0, color=colors, alpha=0.88)
    ax.axvline(float(df["hand_detect_ratio"].mean()) * 100.0, color="#0f172a", ls="--", lw=1.2, label="Trung bình test (90.13%)")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f"{r['label']} (id={r['class_id']})" for _, r in grouped.iterrows()], fontsize=8.5)
    ax.set_xlabel("Tỷ lệ frame phát hiện ít nhất 1 tay (%)")
    ax.set_title("Tỷ lệ phát hiện bàn tay theo 30 lớp trên tập test (309 video)")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # Xuất báo cáo Markdown
    md_path = REPO_ROOT / "docs" / "test_hand_detection_analysis.md"
    low_df = df[df["hand_detect_ratio"] < 0.70].sort_values("hand_detect_ratio")

    lines = [
        "# Phân tích tỷ lệ phát hiện bàn tay trên tập `test` (309 mẫu, 30 lớp)",
        "",
        f"- **Trung bình chung (`hand_detect_ratio`):** {df['hand_detect_ratio'].mean():.2%}",
        f"- **Tay trái trung bình:** {df['left_hand_detect_ratio'].mean():.2%} | **Tay phải trung bình:** {df['right_hand_detect_ratio'].mean():.2%}",
        f"- **Số mẫu có `hand_detect_ratio < 70%`:** {len(low_df)} / {len(df)} mẫu",
        "",
        "## Top 10 lớp có tỷ lệ phát hiện tay thấp nhất ở tập `test`",
        "",
        "| Class ID | Nhãn | Số mẫu | TB có tay | Thấp nhất | Tay trái TB | Tay phải TB | Số mẫu < 70% |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in grouped.head(10).iterrows():
        lines.append(
            f"| {int(r['class_id'])} | {r['label']} | {int(r['num_samples'])} | "
            f"{r['hand_mean']:.2%} | {r['hand_min']:.2%} | {r['left_mean']:.2%} | "
            f"{r['right_mean']:.2%} | {int(r['low_hand_samples'])} |"
        )

    lines.extend([
        "",
        "## Các mẫu cụ thể bị mất tay nhiều nhất (`hand_detect_ratio < 50%`)",
        "",
        "| Sample ID | Nhãn | Signer | Tỷ lệ có tay | Tay trái | Tay phải |",
        "| :--- | :--- | :--- | :---: | :---: | :---: |",
    ])
    very_low = df[df["hand_detect_ratio"] < 0.50].sort_values("hand_detect_ratio")
    for _, r in very_low.iterrows():
        lines.append(
            f"| `{r['sample_id']}` | {r['label']} | `{r['signer_id']}` | "
            f"{r['hand_detect_ratio']:.2%} | {r['left_hand_detect_ratio']:.2%} | {r['right_hand_detect_ratio']:.2%} |"
        )

    lines.extend([
        "",
        "![Biểu đồ phát hiện tay theo lớp](figures/test_hand_detection_by_class.png)",
        "",
    ])

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Đã tạo báo cáo: {md_path} và biểu đồ: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


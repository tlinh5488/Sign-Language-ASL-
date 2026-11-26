"""Demo nhận diện VSL thời gian thực qua Webcam cho Giai đoạn 4 (Linh).

Chạy sau khi đã huấn luyện & export ONNX ở Giai đoạn 3:
    python scripts/run_webcam_demo.py --model artifacts/bilstm_207/model.onnx --meta data/features/feature_meta.json

Phím tắt trên cửa sổ webcam:
    - c: Xóa câu đang ghép
    - s: Đọc câu thành tiếng (nếu có pyttsx3)
    - q / ESC: Thoát
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.predictor import VSLRealtimePredictor
from src.preprocess.extract_landmarks import results_to_arrays


def _load_vn_font(size: int = 24):
    """Tìm font hỗ trợ Unicode tiếng Việt (Windows/Linux), fallback về font mặc định."""
    from PIL import ImageFont

    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_utf8_text(
    frame_bgr,
    text: str,
    pos: tuple[int, int],
    color_bgr: tuple[int, int, int] = (255, 255, 255),
    font_size: int = 26,
):
    """Vẽ chữ tiếng Việt có dấu lên frame OpenCV bằng Pillow (fallback cv2 nếu thiếu PIL)."""
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw

        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        font = _load_vn_font(font_size)
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        draw.text(pos, text, font=font, fill=color_rgb)
        return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        import cv2

        cv2.putText(frame_bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
        return frame_bgr


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Real-time VSL Webcam Demo (Stage 4)")
    p.add_argument("--model", default="artifacts/bilstm_207/model.onnx")
    p.add_argument("--meta", default="data/features/feature_meta.json")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.65)
    args = p.parse_args(argv)

    import cv2
    import mediapipe as mp

    predictor = VSLRealtimePredictor(
        model_path=args.model,
        feature_meta_path=args.meta,
        conf_threshold=args.conf,
    )
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được webcam index={args.camera}")

    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            frame_dict = results_to_arrays(results)
            res = predictor.step(frame_dict)

            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            status = f"{res.label} ({res.confidence:.0%})" if res.active else "Đang chờ tay..."
            frame = draw_utf8_text(
                frame,
                status,
                (20, 18),
                color_bgr=(0, 220, 0) if res.active else (0, 180, 255),
                font_size=28,
            )
            sent_text = " ".join(res.sentence[-8:])
            frame = draw_utf8_text(
                frame,
                f"Câu: {sent_text}",
                (20, 58),
                color_bgr=(255, 255, 255),
                font_size=22,
            )

            cv2.imshow("VSL Real-time Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                predictor.clear_sentence()
            elif key == ord("s"):
                predictor.speak_sentence_async()

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


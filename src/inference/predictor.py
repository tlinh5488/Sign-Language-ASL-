"""Engine suy luận thời gian thực cho Giai đoạn 4 (Linh).

Tích hợp trọn vẹn:
1. Chuẩn hóa từng frame dùng chung ``normalize_frame`` + ``make_state()``
2. Xử lý cửa sổ trượt bằng ``process_realtime_window`` (nội suy + cắt nghỉ + resample + smooth)
3. Suy luận ONNX Runtime (hoặc PyTorch fallback)
4. Activity Gate + EMA smoothing + Cooldown + Sentence Builder + Optional TTS
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Any, Mapping

import numpy as np

from src.preprocess.normalize import feature_dim, make_state, normalize_frame
from src.preprocess.sequence import process_realtime_window


@dataclass
class PredictionResult:
    active: bool
    label: str | None
    class_id: int | None
    confidence: float
    committed_word: str | None
    sentence: list[str]
    probs: np.ndarray | None = None


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


class VSLRealtimePredictor:
    """Bộ suy luận real-time Zero-Skew dùng chung cấu hình với Phần 2 và ONNX từ Phần 3."""

    def __init__(
        self,
        model_path: str | Path,
        feature_meta_path: str | Path,
        *,
        window_size: int = 48,
        min_active_frames: int = 10,
        conf_threshold: float = 0.65,
        ema_alpha: float = 0.45,
        cooldown_frames: int = 18,
        smooth_window: int = 3,
    ) -> None:
        self.model_path = Path(model_path)
        meta = json.loads(Path(feature_meta_path).read_text(encoding="utf-8"))
        self.cfg: dict[str, Any] = dict(meta.get("config_snapshot") or meta.get("config") or {})
        if smooth_window >= 3:
            seq_cfg = dict(self.cfg.get("sequence", {}) or {})
            seq_cfg.setdefault("smooth_window", smooth_window)
            self.cfg["sequence"] = seq_cfg

        self.expected_dim = int(meta.get("D", feature_dim(self.cfg)))
        self.target_len = int(meta.get("L", 48))
        self.class_names: list[str] = [
            str(c["label"]) for c in meta.get("class_list", [])
        ] or [str(i) for i in range(int(meta.get("num_classes", 30)))]

        self.window_size = int(window_size)
        self.min_active_frames = int(min_active_frames)
        self.conf_threshold = float(conf_threshold)
        self.ema_alpha = float(ema_alpha)
        self.cooldown_frames = int(cooldown_frames)

        self._norm_state = make_state()
        self._feat_buf: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._hand_buf: deque[bool] = deque(maxlen=self.window_size)
        self._ema_probs: np.ndarray | None = None
        self._cooldown = 0
        self._last_committed: str | None = None
        self.sentence: list[str] = []

        self._ort_session = None
        self._torch_model = None
        self._load_model()

    def _load_model(self) -> None:
        if self.model_path.suffix.lower() == ".onnx":
            import onnxruntime as ort

            self._ort_session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
        else:
            import torch
            from src.train.model import build_model_from_checkpoint

            # Checkpoint có chứa class_names (list[str]) nên không dùng weights_only=True.
            ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
            self._torch_model = build_model_from_checkpoint(ckpt)

    def _predict_logits(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        x_in = X[None, ...].astype(np.float32)
        m_in = mask[None, ...].astype(bool)
        if self._ort_session is not None:
            logits = self._ort_session.run(None, {"features": x_in, "mask": m_in})[0]
            return logits[0]
        import torch

        with torch.no_grad():
            logits_t = self._torch_model(torch.from_numpy(x_in), torch.from_numpy(m_in))
            return logits_t.cpu().numpy()[0]

    def reset_buffer(self) -> None:
        self._feat_buf.clear()
        self._hand_buf.clear()
        self._ema_probs = None
        self._cooldown = 0

    def clear_sentence(self) -> None:
        self.sentence.clear()
        self._last_committed = None

    def step(self, frame_dict: Mapping[str, Any]) -> PredictionResult:
        """Nhận một frame raw dict (từ ``results_to_arrays``) và trả dự đoán hiện tại."""
        vec = normalize_frame(frame_dict, self.cfg, state=self._norm_state)
        lh_m = bool(frame_dict.get("left_hand_mask", False))
        rh_m = bool(frame_dict.get("right_hand_mask", False))
        has_hand = lh_m or rh_m

        self._feat_buf.append(vec)
        self._hand_buf.append(has_hand)
        if self._cooldown > 0:
            self._cooldown -= 1

        active_count = int(sum(self._hand_buf))
        if active_count < self.min_active_frames:
            self._ema_probs = None
            if active_count == 0:
                self._last_committed = None
            return PredictionResult(
                active=False,
                label=None,
                class_id=None,
                confidence=0.0,
                committed_word=None,
                sentence=list(self.sentence),
            )

        buf_arr = np.stack(list(self._feat_buf), axis=0)
        hand_arr = np.asarray(list(self._hand_buf), dtype=bool)
        X_in, mask_in = process_realtime_window(buf_arr, self.cfg, hand_mask=hand_arr)
        logits = self._predict_logits(X_in, mask_in)
        probs = _softmax(logits)

        if self._ema_probs is None:
            self._ema_probs = probs
        else:
            self._ema_probs = self.ema_alpha * probs + (1.0 - self.ema_alpha) * self._ema_probs

        pred_id = int(np.argmax(self._ema_probs))
        conf = float(self._ema_probs[pred_id])
        label = self.class_names[pred_id] if pred_id < len(self.class_names) else str(pred_id)

        committed: str | None = None
        if conf >= self.conf_threshold and self._cooldown == 0 and label != self._last_committed:
            self.sentence.append(label)
            self._last_committed = label
            self._cooldown = self.cooldown_frames
            committed = label

        return PredictionResult(
            active=True,
            label=label,
            class_id=pred_id,
            confidence=conf,
            committed_word=committed,
            sentence=list(self.sentence),
            probs=self._ema_probs.copy(),
        )

    def speak_sentence_async(self) -> None:
        """Đọc câu hiện tại thành tiếng trên luồng riêng (không chặn webcam)."""
        if not self.sentence:
            return
        text = " ".join(self.sentence)

        def _run() -> None:
            try:
                import pyttsx3  # type: ignore

                eng = pyttsx3.init()
                eng.say(text)
                eng.runAndWait()
            except Exception:
                pass

        threading.Thread(target=_run, daemon=True).start()


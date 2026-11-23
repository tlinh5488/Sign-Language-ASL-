"""Bước 2.3 — Chuỗi khung hình: nội suy tay bị mất, cắt frame nghỉ, cố định độ dài.

Pure numpy, dùng chung cho offline (build_features) và real-time (Linh).

API chính
---------
* ``interpolate_missing(arr, mask, max_gap=8) -> np.ndarray``
  (hoặc ``return_mask=True -> (arr_filled, mask_filled)``)
* ``trim_idle_frames(features, hand_mask, frame_mask=None, *, margin=0) -> (features_trim, mask_trim)``
* ``resample_sequence(x, target_len) -> np.ndarray (target_len, ...)``
* ``pad_or_truncate(x, target_len, pad_value=0.0, *, mask=None) -> (x_fixed, mask_fixed)``
* ``to_fixed_length(features, mask=None, cfg=None) -> (X (L, D), mask (L,))``
* ``add_velocity(x, mask=None) -> np.ndarray (L, 2*D)``
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

DEFAULT_SEQUENCE_CFG: dict[str, Any] = {
    "target_len": 48,
    "method": "resample",  # "resample" (nội suy tuyến tính đều theo thời gian) | "pad"
    "fallback": "pad",
    "interpolate_missing_hands": True,
    "max_gap": 8,
    "trim_idle": True,
    "trim_margin": 0,
    "pad_value": 0.0,
    "truncate": "center",  # khi method=pad và T > L: "center" | "start" | "resample"
}


def get_seq_cfg(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Section ``sequence`` trộn với mặc định (nhận config đầy đủ hoặc chỉ section)."""
    out = dict(DEFAULT_SEQUENCE_CFG)
    if not cfg:
        return out
    section = cfg.get("sequence", cfg) if isinstance(cfg, Mapping) else {}
    if isinstance(section, Mapping):
        for k, v in section.items():
            if v is not None:
                out[k] = v
    return out


# ---------------------------------------------------------------------------
# Missing-hand interpolation
# ---------------------------------------------------------------------------
def _false_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Các đoạn liên tiếp ``False`` dạng ``[start, end)``."""
    runs: list[tuple[int, int]] = []
    T = len(mask)
    t = 0
    while t < T:
        if not mask[t]:
            s = t
            while t < T and not mask[t]:
                t += 1
            runs.append((s, t))
        else:
            t += 1
    return runs


def interpolate_missing(
    arr: np.ndarray,
    mask: np.ndarray,
    max_gap: int = 8,
    *,
    return_mask: bool = False,
):
    """Nội suy tuyến tính các lỗ ngắn (mất tay tạm thời) dọc trục thời gian.

    Chỉ lấp các lỗ **nằm giữa** hai frame hợp lệ và dài ``<= max_gap`` frame; lỗ dài
    hơn ``max_gap`` hoặc lỗ ở đầu/cuối chuỗi giữ nguyên bằng 0.

    Args:
        arr: ``(T, ...)`` ví dụ ``(T, 21, 3)`` hoặc ``(T, D)``.
        mask: ``(T,)`` bool — True = có phát hiện.
        max_gap: số frame liên tiếp tối đa của một lỗ được phép nội suy (<=0 -> tắt).
        return_mask: True -> trả ``(arr_filled, mask_filled)``, False (mặc định) -> trả ``arr_filled``.
    Returns:
        ``arr_filled`` (hoặc ``(arr_filled, mask_filled)`` nếu ``return_mask=True``).
    """
    arr_out = np.array(arr, dtype=np.float32, copy=True)
    mask_out = np.array(mask, dtype=bool, copy=True)
    T = arr_out.shape[0]
    # Đảm bảo các frame mask=False ban đầu là 0 trước khi nội suy
    if T > 0:
        arr_out[~mask_out] = 0.0
    if T == 0 or max_gap is None or int(max_gap) <= 0 or mask_out.all() or not mask_out.any():
        return (arr_out, mask_out) if return_mask else arr_out

    limit = int(max_gap)
    for s, e in _false_runs(mask_out):
        if s == 0 or e == T or (e - s) > limit:
            continue
        a, b = arr_out[s - 1], arr_out[e]
        n = e - s + 1
        for k, t in enumerate(range(s, e), start=1):
            w = np.float32(k / n)
            arr_out[t] = (np.float32(1.0) - w) * a + w * b
        mask_out[s:e] = True
    return (arr_out, mask_out) if return_mask else arr_out


def interpolate_raw_hands(raw: Mapping[str, Any], max_gap: int = 8) -> dict[str, Any]:
    """Áp dụng :func:`interpolate_missing` cho ``left_hand`` và ``right_hand`` của dict raw."""
    out = {k: raw[k] for k in raw.keys()}
    for side in ("left_hand", "right_hand"):
        if side in out and out[side] is not None:
            arr = np.asarray(out[side], dtype=np.float32)
            m = out.get(f"{side}_mask")
            m_arr = np.any(np.abs(arr) > 0, axis=(1, 2)) if m is None else np.asarray(m, dtype=bool)
            out[side], out[f"{side}_mask"] = interpolate_missing(
                arr, m_arr, max_gap=max_gap, return_mask=True
            )
    return out


# ---------------------------------------------------------------------------
# Trimming idle frames
# ---------------------------------------------------------------------------
def idle_bounds(hand_mask: np.ndarray, margin: int = 0) -> tuple[int, int]:
    """Chỉ số ``[start, end)`` từ frame có tay đầu tiên đến frame có tay cuối cùng (+margin).

    Nếu không có frame nào có tay, trả về ``(0, T)`` để không làm rỗng chuỗi.
    """
    hand_mask = np.asarray(hand_mask, dtype=bool)
    T = len(hand_mask)
    if T == 0 or not hand_mask.any():
        return 0, T
    idx = np.flatnonzero(hand_mask)
    m = max(0, int(margin))
    return max(0, int(idx[0]) - m), min(T, int(idx[-1]) + 1 + m)


def trim_idle_frames(
    features: np.ndarray,
    hand_mask: np.ndarray,
    frame_mask: np.ndarray | None = None,
    *,
    margin: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bỏ các frame đầu và cuối không có tay nào (tư thế nghỉ trước/sau khi ký hiệu).

    Nếu không có frame nào có tay trong cả video, giữ nguyên chuỗi để tránh tạo chuỗi rỗng.

    Args:
        features: ``(T, ...)`` mảng đặc trưng hoặc tọa độ.
        hand_mask: ``(T,)`` bool — True ở frame có ít nhất 1 bàn tay.
        frame_mask: ``(T,)`` bool tùy chọn; nếu không truyền thì cắt chính ``hand_mask``.
        margin: số frame đệm giữ lại ở hai đầu đoạn có tay.
    Returns:
        ``(features_trimmed, mask_trimmed)``.
    """
    features = np.asarray(features)
    hand_mask = np.asarray(hand_mask, dtype=bool)
    active_mask = hand_mask if frame_mask is None else np.asarray(frame_mask, dtype=bool)
    s, e = idle_bounds(hand_mask, margin=margin)
    return features[s:e], active_mask[s:e]


# ---------------------------------------------------------------------------
# Resample / Pad / Fixed length
# ---------------------------------------------------------------------------
def resample_sequence(x: np.ndarray, target_len: int) -> np.ndarray:
    """Nội suy tuyến tính đều theo thời gian về đúng ``target_len`` frame.

    * ``T == 0`` -> mảng toàn 0 shape ``(target_len, ...)``.
    * ``T == 1`` -> lặp lại frame duy nhất ``target_len`` lần.
    * ``T >= 2`` -> nội suy tuyến tính tại ``linspace(0, T-1, target_len)``.
    """
    x = np.asarray(x, dtype=np.float32)
    T = x.shape[0]
    L = int(target_len)
    if L <= 0:
        return np.zeros((0,) + x.shape[1:], dtype=np.float32)
    if T == 0:
        return np.zeros((L,) + x.shape[1:], dtype=np.float32)
    if T == 1:
        return np.repeat(x, L, axis=0).astype(np.float32)
    if T == L:
        return x.copy()
    pos = np.linspace(0.0, T - 1, L)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (pos - lo).astype(np.float32).reshape((L,) + (1,) * (x.ndim - 1))
    return ((1.0 - w) * x[lo] + w * x[hi]).astype(np.float32)


def pad_or_truncate(
    x: np.ndarray,
    target_len: int,
    pad_value: float = 0.0,
    *,
    truncate: str = "center",
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad cuối chuỗi bằng ``pad_value`` hoặc cắt về ``target_len`` frame.

    Returns:
        ``(x_fixed (L, ...), mask_fixed (L,) bool)`` — True ở frame hợp lệ, False ở vùng pad.
    """
    x = np.asarray(x, dtype=np.float32)
    T = x.shape[0]
    L = int(target_len)
    src_mask = np.ones(T, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if L <= 0:
        return np.zeros((0,) + x.shape[1:], dtype=np.float32), np.zeros(0, dtype=bool)
    if T >= L:
        if truncate == "resample":
            return resample_sequence(x, L), np.full(L, bool(src_mask.any()), dtype=bool)
        start = (T - L) // 2 if truncate == "center" else 0
        return x[start : start + L].copy(), src_mask[start : start + L].copy()
    out = np.full((L,) + x.shape[1:], pad_value, dtype=np.float32)
    out[:T] = x
    out_mask = np.zeros(L, dtype=bool)
    out_mask[:T] = src_mask
    return out, out_mask


def add_velocity(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Nối thêm đặc trưng vận tốc ``Δx_t = x_t - x_{t-1}`` (``Δx_0 = 0``, vùng pad = 0).

    Args:
        x: ``(L, D)`` float32.
        mask: ``(L,)`` bool tùy chọn.
    Returns:
        ``(L, 2*D)`` float32.
    """
    x = np.asarray(x, dtype=np.float32)
    d = np.zeros_like(x)
    if x.shape[0] > 1:
        d[1:] = x[1:] - x[:-1]
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        valid = np.zeros_like(m)
        if len(m) > 1:
            valid[1:] = m[1:] & m[:-1]
        d[~valid] = 0.0
    return np.concatenate([x, d], axis=-1).astype(np.float32)


def to_fixed_length(
    features: np.ndarray,
    mask: np.ndarray | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Đưa chuỗi đặc trưng ``(T, D)`` về độ dài cố định ``(L, D_out)`` theo ``cfg``.

    * ``method="resample"`` (mặc định): nội suy tuyến tính đều về ``L = target_len`` frame.
      Nếu ``T == 0`` (hoặc ``mask`` toàn False) thì fallback sang mảng ``pad_value`` với
      ``out_mask`` toàn False.
    * ``method="pad"``: pad cuối bằng ``pad_value`` hoặc cắt về ``L`` frame.

    Nếu ``normalization.add_velocity`` bật (và ``features`` chưa kèm velocity), hàm tự
    nối thêm vận tốc ``Δ`` sau khi cố định độ dài.
    """
    from .normalize import base_feature_dim, get_norm_cfg

    scfg = get_seq_cfg(cfg)
    ncfg = get_norm_cfg(cfg)
    L = int(scfg.get("target_len", 48))
    pad_val = float(scfg.get("pad_value", 0.0))
    method = str(scfg.get("method", "resample")).lower()

    features = np.asarray(features, dtype=np.float32)
    T = features.shape[0]
    want_vel = bool(ncfg.get("add_velocity", False))
    d_base = base_feature_dim(cfg)
    if want_vel and features.ndim == 2 and features.shape[1] == 2 * d_base:
        # Lấy phần tọa độ gốc để resample rồi tính lại vận tốc nhất quán trên lưới thời gian mới
        features = features[:, :d_base]

    has_valid = T > 0 and (mask is None or bool(np.any(mask)))
    if not has_valid:
        d_in = features.shape[1] if features.ndim == 2 else d_base
        X = np.full((L, d_in), pad_val, dtype=np.float32)
        out_mask = np.zeros(L, dtype=bool)
    elif method == "pad":
        X, out_mask = pad_or_truncate(
            features,
            L,
            pad_val,
            truncate=str(scfg.get("truncate", "center")),
            mask=mask,
        )
    else:
        X = resample_sequence(features, L)
        out_mask = np.ones(L, dtype=bool)

    if want_vel:
        X = add_velocity(X, out_mask)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X, out_mask


__all__ = [
    "DEFAULT_SEQUENCE_CFG",
    "get_seq_cfg",
    "interpolate_missing",
    "interpolate_raw_hands",
    "idle_bounds",
    "trim_idle_frames",
    "resample_sequence",
    "pad_or_truncate",
    "add_velocity",
    "to_fixed_length",
]

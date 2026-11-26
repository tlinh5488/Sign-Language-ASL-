"""Bước 2.2 — Chuẩn hóa tọa độ landmark (gốc về cổ tay / tâm vai).

Module này **chỉ dùng numpy** (không import MediaPipe / OpenCV) để cả pipeline offline
(Thiên, Giai đoạn 2) lẫn suy luận thời gian thực (Linh, Giai đoạn 4) gọi **cùng một hàm
chuẩn hóa**:

* Offline (cả video):  ``features, frame_mask = normalize_sequence(raw, cfg)``
* Real-time (từng frame): ``vec = normalize_frame(frame, cfg, state=state)``

Hai đường đi cho ra **cùng một vector** cho cùng một frame.

Toán học (chi tiết trong ``docs/preprocess_guide.md``)
------------------------------------------------------
* Bàn tay (hình dạng, bất biến tịnh tiến + tỉ lệ)::

    s_h   = || p_9 - p_0 ||            (cổ tay idx 0 -> khớp MCP ngón giữa idx 9)
    h_i   = (p_i - p_0) / max(s_h, eps)   ; tay không phát hiện -> 0

* Thân trên (pose)::

    c     = (P_11 + P_12) / 2           (tâm hai vai)
    s_b   = || P_11 - P_12 ||           (độ rộng vai)
    b_j   = (P_j - c) / s_b , j in UPPER_BODY_POSE_IDX (0..24)

  Frame thiếu vai -> dùng (c, s_b) của frame hợp lệ gần nhất trước đó; các frame đầu
  chưa có frame hợp lệ -> dùng frame hợp lệ đầu tiên / trung vị s_b của chuỗi.

* Vị trí toàn cục của tay (tay ĐANG Ở ĐÂU so với cơ thể)::

    g = (p_0 - c) / s_b

* (Tùy chọn) mặt::  f_k = (F_k - F_origin) / s_b , k in FACE_SUBSET_IDX
  (``F_origin`` mặc định là đầu mũi idx 1, hoặc tâm vai nếu ``face.origin == "shoulder_center"``).

Bố cục vector đặc trưng (mặc định ``pose_upper+hands``, use_z=True) — D = 207::

    [ pose 25x3 = 75 | left_hand 21x3 = 63 | right_hand 21x3 = 63 |
      left_wrist_global 3 | right_wrist_global 3 ]

Nếu bật ``feature_set: "pose_upper+hands+face_subset"`` (36 điểm mặt) -> D = 207 + 108 = 315.
Nếu bật ``add_velocity: true`` -> nối thêm vận tốc Δ -> D nhân đôi (414 hoặc 630).
``feature_names(cfg)`` trả về tên từng chiều theo đúng thứ tự trên.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# Hằng số schema — lấy từ landmark_schema (extractor); fallback để module này
# vẫn chạy độc lập.
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from src.preprocess import landmark_schema as _schema
except Exception:  # pragma: no cover
    try:
        from . import landmark_schema as _schema  # type: ignore
    except Exception:
        _schema = None


def _const(name: str, default: Any) -> Any:
    return getattr(_schema, name, default) if _schema is not None else default


POSE_N: int = int(_const("POSE_N", 33))
FACE_N: int = int(_const("FACE_N", 468))
HAND_N: int = int(_const("HAND_N", 21))
WRIST: int = int(_const("WRIST", 0))
MIDDLE_MCP: int = int(_const("MIDDLE_MCP", 9))
NOSE: int = int(_const("NOSE", 0))
L_SHOULDER: int = int(_const("L_SHOULDER", 11))
R_SHOULDER: int = int(_const("R_SHOULDER", 12))
UPPER_BODY_POSE_IDX: list[int] = list(_const("UPPER_BODY_POSE_IDX", list(range(0, 25))))
_FALLBACK_FACE_SUBSET: list[int] = [
    61, 291, 0, 17, 37, 267, 84, 314,
    78, 308, 13, 14,
    336, 296, 334, 293, 300,
    107, 66, 105, 63, 70,
    263, 362, 386, 374,
    33, 133, 159, 145,
    1, 168,
    10, 152, 454, 234,
]
FACE_SUBSET_IDX: list[int] = list(_const("FACE_SUBSET_IDX", _FALLBACK_FACE_SUBSET))
FACE_NOSE_TIP: int = int(_const("FACE_NOSE_TIP", 1))

EPS: float = 1e-6

HAND_LANDMARK_NAMES: list[str] = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
POSE_LANDMARK_NAMES: list[str] = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]

DEFAULT_NORMALIZATION_CFG: dict[str, Any] = {
    "feature_set": "pose_upper+hands",  # hoặc "pose_upper+hands+face_subset"
    "hand_origin": "wrist",
    "hand_scale": "wrist_to_middle_mcp",
    "pose_origin": "shoulder_center",
    "pose_scale": "shoulder_width",
    "face_origin": "nose",  # "nose" (FaceMesh idx 1) hoặc "shoulder_center"
    "face_scale": "shoulder_width",
    "include_hand_global_position": True,
    "include_hand_presence": False,
    "include_pose_visibility": False,
    "include_geometric_features": False,
    "use_z": True,
    "add_velocity": False,
    "correct_aspect_ratio": False,
    "min_shoulder_visibility": 0.1,
    "fallback_shoulder_width": 0.3,
    "clip_value": None,
    "eps": EPS,
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def get_norm_cfg(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Trả về section ``normalization`` đã trộn với giá trị mặc định.

    Chấp nhận cả config đầy đủ (có key ``normalization``) lẫn chỉ section đó,
    đồng thời hiểu các khóa con ``hand.{origin,scale}``, ``pose.{origin,scale}``,
    ``face.{origin,scale}`` trong ``configs/preprocess_config.yaml``.
    """
    out = dict(DEFAULT_NORMALIZATION_CFG)
    if not cfg:
        return out
    section = cfg.get("normalization", cfg) if isinstance(cfg, Mapping) else {}
    if isinstance(section, Mapping):
        for k, v in section.items():
            if isinstance(v, Mapping):
                if k in ("hand", "pose", "face"):
                    if "origin" in v and v["origin"] is not None:
                        out[f"{k}_origin"] = v["origin"]
                    if "scale" in v and v["scale"] is not None:
                        out[f"{k}_scale"] = v["scale"]
            elif v is not None:
                out[k] = v
    seq = cfg.get("sequence") if isinstance(cfg, Mapping) else None
    if isinstance(seq, Mapping) and "add_velocity" in seq and seq["add_velocity"] is not None:
        out["add_velocity"] = bool(seq["add_velocity"])
    return out


def _use_face(ncfg: Mapping[str, Any]) -> bool:
    return "face" in str(ncfg.get("feature_set", "pose_upper+hands")).lower()


def _use_pose(ncfg: Mapping[str, Any]) -> bool:
    return "pose" in str(ncfg.get("feature_set", "pose_upper+hands")).lower()


def _ndim(ncfg: Mapping[str, Any]) -> int:
    return 3 if bool(ncfg.get("use_z", True)) else 2


# ---------------------------------------------------------------------------
# Low-level geometric helpers (vectorised over arbitrary leading dims)
# ---------------------------------------------------------------------------
def _aspect(width: Any, height: Any) -> float:
    try:
        w, h = float(width), float(height)
        if w > 0 and h > 0:
            return w / h
    except (TypeError, ValueError):
        pass
    return 1.0


def to_isotropic(coords: np.ndarray, aspect: float) -> np.ndarray:
    """Nhân x và z với W/H để khoảng cách đẳng hướng (đơn vị = chiều cao ảnh)."""
    out = np.array(coords, dtype=np.float32, copy=True)
    if aspect != 1.0:
        out[..., 0] *= aspect
        if out.shape[-1] >= 3:
            out[..., 2] *= aspect
    return out


def _present_from(arr: np.ndarray, present: Any, point_axes: int = 2) -> np.ndarray:
    lead = arr.shape[:-point_axes]
    nonzero = np.any(np.abs(arr) > 0, axis=tuple(range(-point_axes, 0)))
    if present is None:
        return nonzero.reshape(lead)
    p = np.asarray(present, dtype=bool)
    return np.broadcast_to(p, lead) & nonzero


def normalize_hand(
    hand: np.ndarray,
    present: Any = None,
    *,
    use_z: bool = True,
    eps: float = EPS,
) -> np.ndarray:
    """Chuẩn hóa bàn tay: gốc = cổ tay (idx 0), tỉ lệ = ||cổ tay -> MCP ngón giữa (idx 9)||.

    Args:
        hand: ``(..., 21, 3)`` tọa độ bàn tay.
        present: bool ``(...)``; None -> suy ra từ việc khác 0.
        use_z: False -> chỉ trả x,y (``(..., 21, 2)``), tỉ lệ tính 2D.
        eps: ngưỡng tránh chia cho 0.
    Returns:
        ``(..., 21, 3|2)`` float32; tay vắng mặt hoặc tỉ lệ <= eps -> toàn 0.
    """
    hand = np.asarray(hand, dtype=np.float32)
    nd = min(int(hand.shape[-1]), 3 if use_z else 2)
    xyz = hand[..., :nd]
    pres = _present_from(hand[..., :nd], present)
    wrist = xyz[..., WRIST : WRIST + 1, :]
    centered = xyz - wrist
    scale = np.linalg.norm(centered[..., MIDDLE_MCP, :], axis=-1)  # (...)
    valid = pres & (scale > eps)
    safe = np.where(valid, scale, 1.0)[..., None, None]
    out = np.where(valid[..., None, None], centered / safe, 0.0)
    return out.astype(np.float32)


def shoulder_frame(
    pose: np.ndarray,
    present: Any = None,
    *,
    min_visibility: float = 0.1,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tâm vai, độ rộng vai và cờ hợp lệ cho từng frame.

    Args:
        pose: ``(..., 33, 3|4)``.
    Returns:
        ``center (..., 3)``, ``scale (...)``, ``valid (...) bool``.
    """
    pose = np.asarray(pose, dtype=np.float32)
    ls = pose[..., L_SHOULDER, :3]
    rs = pose[..., R_SHOULDER, :3]
    center = 0.5 * (ls + rs)
    scale = np.linalg.norm(ls - rs, axis=-1)
    shoulders_present = (np.linalg.norm(ls, axis=-1) > eps) & (np.linalg.norm(rs, axis=-1) > eps)
    valid = _present_from(pose[..., :3], present) & shoulders_present & (scale > eps)
    if pose.shape[-1] >= 4 and min_visibility and min_visibility > 0:
        vis = np.minimum(pose[..., L_SHOULDER, 3], pose[..., R_SHOULDER, 3])
        has_vis = np.any(pose[..., :, 3] > 0, axis=-1)
        valid = valid & (~has_vis | (vis >= min_visibility))
    return center.astype(np.float32), scale.astype(np.float32), valid


def fill_body_frame(
    center: np.ndarray,
    scale: np.ndarray,
    valid: np.ndarray,
    *,
    fallback_scale: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Điền (center, scale) cho frame thiếu vai theo chuỗi ``(T, ...)``.

    Frame thiếu -> giá trị của frame hợp lệ gần nhất **trước đó**; các frame đầu
    (chưa có frame hợp lệ) -> center của frame hợp lệ đầu tiên, scale = trung vị
    các scale hợp lệ. Không có frame hợp lệ nào -> center (0.5, 0.5, 0) và
    ``fallback_scale``.
    """
    center = np.array(center, dtype=np.float32, copy=True)
    scale = np.array(scale, dtype=np.float32, copy=True)
    valid = np.asarray(valid, dtype=bool)
    T = scale.shape[0]
    if T == 0:
        return center, scale
    if not valid.any():
        center[:] = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        scale[:] = np.float32(fallback_scale)
        return center, scale
    idx = np.where(valid, np.arange(T), -1)
    idx = np.maximum.accumulate(idx)
    first = int(np.argmax(valid))
    med = float(np.median(scale[valid]))
    lead = idx < 0
    idx[lead] = first
    center = center[idx]
    out_scale = scale[idx]
    out_scale[lead] = med
    return center, out_scale.astype(np.float32)


def normalize_pose(
    pose: np.ndarray,
    present: Any = None,
    *,
    center: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    use_z: bool = True,
    include_visibility: bool = False,
    keep_visibility: bool | None = None,
    subset: list[int] | None = None,
    min_visibility: float = 0.1,
    fallback_scale: float = 0.3,
    eps: float = EPS,
) -> np.ndarray:
    """Chuẩn hóa pose thân trên: gốc = tâm vai, tỉ lệ = độ rộng vai.

    Nếu ``center``/``scale`` không truyền vào: với input chuỗi ``(T,33,4)`` tự điền
    frame thiếu vai bằng :func:`fill_body_frame`; với một frame ``(33,4)`` thiếu vai
    dùng ``fallback_scale``.

    Returns:
        ``(..., len(subset), 3|2 [+1 visibility])`` float32; pose vắng -> 0.
    """
    if keep_visibility is not None:
        include_visibility = bool(keep_visibility)
    pose = np.asarray(pose, dtype=np.float32)
    subset = list(UPPER_BODY_POSE_IDX if subset is None else subset)
    nd = 3 if use_z else 2
    pres = _present_from(pose[..., :3], present)
    if center is None or scale is None:
        c, s, v = shoulder_frame(pose, pres, min_visibility=min_visibility, eps=eps)
        if pose.ndim == 3:
            c, s = fill_body_frame(c, s, v, fallback_scale=fallback_scale)
        else:
            c = np.where(v[..., None], c, np.array([0.5, 0.5, 0.0], np.float32))
            s = np.where(v, s, np.float32(fallback_scale))
        center = c if center is None else center
        scale = s if scale is None else scale
    center = np.asarray(center, dtype=np.float32)
    scale = np.maximum(np.asarray(scale, dtype=np.float32), eps)
    pts = pose[..., subset, :3]
    pt_present = np.any(np.abs(pts) > 0, axis=-1, keepdims=True)
    out = np.where(pt_present, (pts - center[..., None, :]) / scale[..., None, None], 0.0)
    out = out[..., :nd]
    if include_visibility and pose.shape[-1] >= 4:
        out = np.concatenate([out, pose[..., subset, 3:4]], axis=-1)
    out = np.where(pres[..., None, None], out, 0.0)
    return out.astype(np.float32)


def hand_global_position(
    hand: np.ndarray,
    present: Any,
    center: np.ndarray,
    scale: np.ndarray,
    *,
    use_z: bool = True,
    eps: float = EPS,
) -> np.ndarray:
    """Vị trí cổ tay so với tâm vai, chia độ rộng vai: ``(p_0 - c) / s_b`` -> ``(..., 3|2)``."""
    hand = np.asarray(hand, dtype=np.float32)
    nd = 3 if use_z else 2
    pres = _present_from(hand[..., :3], present)
    scale = np.maximum(np.asarray(scale, dtype=np.float32), eps)
    g = (hand[..., WRIST, :3] - np.asarray(center, np.float32)) / scale[..., None]
    g = np.where(pres[..., None], g[..., :nd], 0.0)
    return g.astype(np.float32)


def normalize_face(
    face: np.ndarray,
    present: Any,
    scale: np.ndarray,
    *,
    center: np.ndarray | None = None,
    origin: str = "nose",
    subset: list[int] | None = None,
    use_z: bool = True,
    eps: float = EPS,
) -> np.ndarray:
    """Chuẩn hóa tập con điểm mặt (36 điểm): gốc = đầu mũi (hoặc tâm vai), tỉ lệ = độ rộng vai."""
    face = np.asarray(face, dtype=np.float32)
    subset = list(FACE_SUBSET_IDX if subset is None else subset)
    nd = 3 if use_z else 2
    pres = _present_from(face[..., :3], present)
    scale = np.maximum(np.asarray(scale, dtype=np.float32), eps)
    if str(origin).lower() == "shoulder_center" and center is not None:
        ref = np.asarray(center, dtype=np.float32)[..., None, :3]
    else:
        ref = face[..., FACE_NOSE_TIP : FACE_NOSE_TIP + 1, :3]
    out = (face[..., subset, :3] - ref) / scale[..., None, None]
    out = np.where(pres[..., None, None], out[..., :nd], 0.0)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature layout
# ---------------------------------------------------------------------------
def feature_blocks(cfg: Mapping[str, Any] | None = None) -> list[tuple[str, int]]:
    """Danh sách ``(tên_khối, số_chiều)`` theo đúng thứ tự trong vector (chưa gồm velocity)."""
    ncfg = get_norm_cfg(cfg)
    nd = _ndim(ncfg)
    blocks: list[tuple[str, int]] = []
    if _use_pose(ncfg):
        per = nd + (1 if ncfg.get("include_pose_visibility") else 0)
        blocks.append(("pose", len(UPPER_BODY_POSE_IDX) * per))
    blocks.append(("left_hand", HAND_N * nd))
    blocks.append(("right_hand", HAND_N * nd))
    if ncfg.get("include_hand_global_position", True):
        blocks.append(("left_wrist_global", nd))
        blocks.append(("right_wrist_global", nd))
    if ncfg.get("include_hand_presence", False):
        blocks.append(("left_hand_present", 1))
        blocks.append(("right_hand_present", 1))
    if ncfg.get("include_geometric_features", False):
        blocks.append(("geometric", 24))
    if _use_face(ncfg):
        blocks.append(("face", len(FACE_SUBSET_IDX) * nd))
    return blocks


def base_feature_dim(cfg: Mapping[str, Any] | None = None) -> int:
    """Số chiều mỗi frame KHÔNG gồm velocity."""
    return int(sum(n for _, n in feature_blocks(cfg)))


def feature_dim(cfg: Mapping[str, Any] | None = None) -> int:
    """Số chiều D của vector đặc trưng cuối cùng (gồm velocity nếu ``add_velocity``)."""
    d = base_feature_dim(cfg)
    return 2 * d if get_norm_cfg(cfg).get("add_velocity") else d


def feature_names(cfg: Mapping[str, Any] | None = None) -> list[str]:
    """Tên từng chiều của vector đặc trưng, đúng thứ tự (độ dài = ``feature_dim(cfg)``)."""
    ncfg = get_norm_cfg(cfg)
    axes = ["x", "y", "z"][: _ndim(ncfg)]
    names: list[str] = []
    for block, _ in feature_blocks(cfg):
        if block == "pose":
            ax = axes + (["vis"] if ncfg.get("include_pose_visibility") else [])
            for j in UPPER_BODY_POSE_IDX:
                pname = POSE_LANDMARK_NAMES[j] if j < len(POSE_LANDMARK_NAMES) else f"p{j}"
                names += [f"pose_{j:02d}_{pname}_{a}" for a in ax]
        elif block in ("left_hand", "right_hand"):
            side = block.split("_")[0]
            for i in range(HAND_N):
                hname = HAND_LANDMARK_NAMES[i] if i < len(HAND_LANDMARK_NAMES) else f"h{i}"
                names += [f"{side}_hand_{i:02d}_{hname}_{a}" for a in axes]
        elif block.endswith("_wrist_global"):
            names += [f"{block}_{a}" for a in axes]
        elif block.endswith("_present"):
            names.append(block)
        elif block == "geometric":
            names += [
                "geom_lh_thumb_open", "geom_lh_index_open", "geom_lh_middle_open",
                "geom_lh_ring_open", "geom_lh_pinky_open",
                "geom_rh_thumb_open", "geom_rh_index_open", "geom_rh_middle_open",
                "geom_rh_ring_open", "geom_rh_pinky_open",
                "geom_lh_pinch_index", "geom_lh_pinch_middle",
                "geom_rh_pinch_index", "geom_rh_pinch_middle",
                "geom_lh_mean_open", "geom_rh_mean_open",
                "geom_inter_wrist_dist", "geom_lh_to_nose_dist", "geom_rh_to_nose_dist",
                "geom_lh_to_shoulder_dist", "geom_rh_to_shoulder_dist",
                "geom_lh_rel_y", "geom_rh_rel_y", "geom_wrist_dz",
            ]
        elif block == "face":
            for k in FACE_SUBSET_IDX:
                names += [f"face_{k:03d}_{a}" for a in axes]
    if ncfg.get("add_velocity"):
        names = names + [f"d_{n}" for n in names]
    return names


# ---------------------------------------------------------------------------
# Core: normalise arrays with leading time (or no time) dimension
# ---------------------------------------------------------------------------
def _get(d: Mapping[str, Any], key: str, shape: tuple[int, ...], lead: tuple[int, ...]) -> np.ndarray:
    v = d.get(key)
    if v is None:
        return np.zeros(lead + shape, dtype=np.float32)
    return np.asarray(v, dtype=np.float32)


def _get_mask(d: Mapping[str, Any], key: str, arr: np.ndarray, point_axes: int = 2) -> np.ndarray:
    m = d.get(key)
    return _present_from(arr[..., :3], None if m is None else np.asarray(m, bool), point_axes)


def geometric_features(
    pose: np.ndarray,
    pose_mask: np.ndarray,
    lh: np.ndarray,
    lh_mask: np.ndarray,
    rh: np.ndarray,
    rh_mask: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    *,
    use_z: bool = True,
    eps: float = EPS,
) -> np.ndarray:
    """Tính 24 đặc trưng hình học bất biến tịnh tiến/tỉ lệ (thuần NumPy).

    Gồm: độ mở 5 đầu ngón (10), kẹp ngón cái–trỏ/giữa (4), độ mở TB mỗi tay (2),
    khoảng cách giữa 2 cổ tay (1), cổ tay→mũi (2), cổ tay→tâm vai (2),
    cao độ tương đối y của 2 cổ tay (2) và chênh lệch độ sâu z (1).
    """
    nd = 3 if use_z else 2
    sb = np.maximum(np.asarray(scale, dtype=np.float32), eps)[..., None]
    c = np.asarray(center, dtype=np.float32)[..., :nd]

    def _hand_geom(h: np.ndarray, hm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hn = normalize_hand(h, hm, use_z=use_z, eps=eps)
        tips = hn[..., [4, 8, 12, 16, 20], :]
        open_d = np.linalg.norm(tips, axis=-1)  # (..., 5)
        pinch_idx = np.linalg.norm(hn[..., 4, :] - hn[..., 8, :], axis=-1, keepdims=True)
        pinch_mid = np.linalg.norm(hn[..., 4, :] - hn[..., 12, :], axis=-1, keepdims=True)
        mean_o = np.mean(open_d, axis=-1, keepdims=True)
        pinch = np.concatenate([pinch_idx, pinch_mid], axis=-1)
        pres = np.asarray(hm, dtype=bool)[..., None]
        return np.where(pres, open_d, 0.0), np.where(pres, pinch, 0.0), np.where(pres, mean_o, 0.0)

    lh_open, lh_pinch, lh_mean = _hand_geom(lh, lh_mask)
    rh_open, rh_pinch, rh_mean = _hand_geom(rh, rh_mask)

    lw = hand_global_position(lh, lh_mask, center, scale, use_z=use_z, eps=eps)
    rw = hand_global_position(rh, rh_mask, center, scale, use_z=use_z, eps=eps)
    l_pres = np.asarray(lh_mask, dtype=bool)
    r_pres = np.asarray(rh_mask, dtype=bool)
    both_pres = (l_pres & r_pres)[..., None]

    inter_wrist = np.where(both_pres, np.linalg.norm(lw - rw, axis=-1, keepdims=True), 0.0)

    nose_norm = (np.asarray(pose[..., NOSE, :nd], dtype=np.float32) - c) / sb
    p_pres = np.asarray(pose_mask, dtype=bool)
    lh_nose = np.where((l_pres & p_pres)[..., None], np.linalg.norm(lw - nose_norm, axis=-1, keepdims=True), 0.0)
    rh_nose = np.where((r_pres & p_pres)[..., None], np.linalg.norm(rw - nose_norm, axis=-1, keepdims=True), 0.0)

    lh_sh = np.where(l_pres[..., None], np.linalg.norm(lw, axis=-1, keepdims=True), 0.0)
    rh_sh = np.where(r_pres[..., None], np.linalg.norm(rw, axis=-1, keepdims=True), 0.0)

    lh_y = np.where(l_pres[..., None], lw[..., 1:2], 0.0)
    rh_y = np.where(r_pres[..., None], rw[..., 1:2], 0.0)
    dz = np.where(both_pres & use_z, (lw[..., 2:3] - rw[..., 2:3]) if use_z else np.zeros_like(lh_y), 0.0)

    return np.concatenate(
        [
            lh_open, rh_open,
            lh_pinch, rh_pinch,
            lh_mean, rh_mean,
            inter_wrist, lh_nose, rh_nose,
            lh_sh, rh_sh,
            lh_y, rh_y, dz,
        ],
        axis=-1,
    ).astype(np.float32)


def _normalize_arrays(
    raw: Mapping[str, Any],
    cfg: Mapping[str, Any] | None,
    *,
    center: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    sequence: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ncfg = get_norm_cfg(cfg)
    use_z = bool(ncfg.get("use_z", True))
    eps = float(ncfg.get("eps", EPS))
    aspect = _aspect(raw.get("width"), raw.get("height")) if ncfg.get("correct_aspect_ratio", False) else 1.0

    ref = None
    for k in ("pose", "left_hand", "right_hand", "face"):
        if raw.get(k) is not None:
            ref = np.asarray(raw[k])
            break
    if ref is None:
        raise ValueError("raw landmarks thiếu cả pose, left_hand, right_hand, face")
    lead = ref.shape[:1] if sequence else ()

    pose = to_isotropic(_get(raw, "pose", (POSE_N, 4), lead), aspect)
    lh = to_isotropic(_get(raw, "left_hand", (HAND_N, 3), lead), aspect)
    rh = to_isotropic(_get(raw, "right_hand", (HAND_N, 3), lead), aspect)
    pose_m = _get_mask(raw, "pose_mask", pose)
    lh_m = _get_mask(raw, "left_hand_mask", lh)
    rh_m = _get_mask(raw, "right_hand_mask", rh)

    if center is None or scale is None:
        c, s, v = shoulder_frame(
            pose, pose_m, min_visibility=float(ncfg.get("min_shoulder_visibility", 0.1)), eps=eps
        )
        fb = float(ncfg.get("fallback_shoulder_width", 0.3))
        if sequence:
            c, s = fill_body_frame(c, s, v, fallback_scale=fb)
        else:
            c = np.where(v[..., None], c, np.array([0.5, 0.5, 0.0], np.float32))
            s = np.where(v, s, np.float32(fb))
        center = c if center is None else np.asarray(center, np.float32)
        scale = s if scale is None else np.asarray(scale, np.float32)

    parts: list[np.ndarray] = []
    flat = lead + (-1,)
    for block, _ in feature_blocks(cfg):
        if block == "pose":
            p = normalize_pose(
                pose,
                pose_m,
                center=center,
                scale=scale,
                use_z=use_z,
                include_visibility=bool(ncfg.get("include_pose_visibility", False)),
                eps=eps,
            )
            parts.append(p.reshape(flat))
        elif block == "left_hand":
            parts.append(normalize_hand(lh, lh_m, use_z=use_z, eps=eps).reshape(flat))
        elif block == "right_hand":
            parts.append(normalize_hand(rh, rh_m, use_z=use_z, eps=eps).reshape(flat))
        elif block == "left_wrist_global":
            parts.append(hand_global_position(lh, lh_m, center, scale, use_z=use_z, eps=eps))
        elif block == "right_wrist_global":
            parts.append(hand_global_position(rh, rh_m, center, scale, use_z=use_z, eps=eps))
        elif block == "left_hand_present":
            parts.append(lh_m.astype(np.float32)[..., None])
        elif block == "right_hand_present":
            parts.append(rh_m.astype(np.float32)[..., None])
        elif block == "geometric":
            parts.append(
                geometric_features(
                    pose, pose_m, lh, lh_m, rh, rh_m, center, scale, use_z=use_z, eps=eps
                )
            )
        elif block == "face":
            face = to_isotropic(_get(raw, "face", (FACE_N, 3), lead), aspect)
            face_m = _get_mask(raw, "face_mask", face)
            parts.append(
                normalize_face(
                    face,
                    face_m,
                    scale,
                    center=center,
                    origin=str(ncfg.get("face_origin", "nose")),
                    use_z=use_z,
                    eps=eps,
                ).reshape(flat)
            )
    feats = np.concatenate(parts, axis=-1).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    clip = ncfg.get("clip_value")
    if clip:
        np.clip(feats, -float(clip), float(clip), out=feats)
    masks = {
        "pose_mask": pose_m,
        "left_hand_mask": lh_m,
        "right_hand_mask": rh_m,
        "hand_mask": lh_m | rh_m,
        "center": np.asarray(center, np.float32),
        "scale": np.asarray(scale, np.float32),
    }
    return feats, masks


def normalize_sequence(
    raw: Mapping[str, Any],
    cfg: Mapping[str, Any] | None = None,
    *,
    return_masks: bool = False,
):
    """Chuẩn hóa cả chuỗi landmark của một video.

    Args:
        raw: dict theo contract của extractor (``pose (T,33,4)``, ``left_hand (T,21,3)``,
            ``right_hand (T,21,3)``, ``face (T,468,3)``, các ``*_mask (T,)``).
        cfg: config đầy đủ hoặc section ``normalization``.
        return_masks: True -> trả thêm dict mask chi tiết (pose/left/right/hand_mask,
            center, scale).
    Returns:
        ``features (T, D) float32`` (``D == feature_dim(cfg)``),
        ``frame_mask (T,) bool`` = frame có phát hiện pose hoặc ít nhất một tay.
    """
    feats, masks = _normalize_arrays(raw, cfg, sequence=True)
    frame_mask = masks["pose_mask"] | masks["hand_mask"]
    if get_norm_cfg(cfg).get("add_velocity"):
        d = np.zeros_like(feats)
        if feats.shape[0] > 1:
            d[1:] = feats[1:] - feats[:-1]
            valid_d = frame_mask[1:] & frame_mask[:-1]
            d[1:][~valid_d] = 0.0
        feats = np.concatenate([feats, d], axis=-1)
    if return_masks:
        return feats, frame_mask, masks
    return feats, frame_mask


def normalize_frame(
    frame: Mapping[str, Any],
    cfg: Mapping[str, Any] | None = None,
    *,
    state: dict | None = None,
) -> np.ndarray:
    """Chuẩn hóa MỘT frame (dùng cho real-time — Giai đoạn 4, Linh).

    Args:
        frame: dict ``pose (33,4)``, ``left_hand (21,3)``, ``right_hand (21,3)``,
            ``face (468,3)`` (tùy chọn), mask bool tùy chọn (``left_hand_mask``...;
            thiếu -> suy ra từ việc khác 0).
        state: dict có thể thay đổi để nhớ ``center``/``scale`` của frame có vai gần
            nhất (truyền cùng một dict cho mọi frame của một luồng webcam).
    Returns:
        ``(D,) float32`` — giống hệt hàng tương ứng của ``normalize_sequence``
        khi frame có vai hợp lệ.
    """
    ncfg = get_norm_cfg(cfg)
    eps = float(ncfg.get("eps", EPS))
    aspect = _aspect(frame.get("width"), frame.get("height")) if ncfg.get("correct_aspect_ratio", False) else 1.0
    pose = to_isotropic(_get(frame, "pose", (POSE_N, 4), ()), aspect)
    pose_m = _get_mask(frame, "pose_mask", pose)
    c, s, v = shoulder_frame(
        pose, pose_m, min_visibility=float(ncfg.get("min_shoulder_visibility", 0.1)), eps=eps
    )
    if bool(v):
        if state is not None:
            state["center"], state["scale"] = c, s
    elif state is not None and "center" in state:
        c, s = state["center"], state["scale"]
    else:
        c = np.array([0.5, 0.5, 0.0], np.float32)
        s = np.float32(ncfg.get("fallback_shoulder_width", 0.3))
    feats, masks = _normalize_arrays(
        frame, cfg, center=c, scale=np.asarray(s, np.float32), sequence=False
    )
    if ncfg.get("add_velocity"):
        fmask = bool(masks["pose_mask"] or masks["hand_mask"])
        if state is not None and "prev_base" in state and state.get("prev_valid") and fmask:
            vel = (feats - state["prev_base"]).astype(np.float32)
        else:
            vel = np.zeros_like(feats)
        if state is not None:
            state["prev_base"] = feats.copy()
            state["prev_valid"] = fmask
        feats = np.concatenate([feats, vel], axis=-1)
    return feats


def make_state() -> dict:
    """Tạo state rỗng cho :func:`normalize_frame` (một luồng video / webcam)."""
    return {}


__all__ = [
    "DEFAULT_NORMALIZATION_CFG",
    "get_norm_cfg",
    "normalize_hand",
    "normalize_pose",
    "normalize_face",
    "hand_global_position",
    "shoulder_frame",
    "fill_body_frame",
    "normalize_frame",
    "normalize_sequence",
    "make_state",
    "feature_blocks",
    "base_feature_dim",
    "feature_dim",
    "feature_names",
]

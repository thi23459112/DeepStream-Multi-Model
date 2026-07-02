"""
C-BIoU 匹配模組（終極物理防禦版）
1. 絕對像素保底 (大車不黑洞、小車不掉線)
2. 底部中心點距離 (擁擠拆解)
3. 拓撲軟懲罰 (低速停等防互換)
4. 🚀 運動學錐體約束 (Kinematic Cone) - 嚴防對向車道幽靈繼承
"""

from __future__ import annotations
import numpy as np
from boxmot.utils.iou import AssociationFunction

# ─────────────────────────────────────────────────────────────────────
# 1. BIoU 距離（核心回歸：使用絕對像素限制）
# ─────────────────────────────────────────────────────────────────────
def _expand_boxes(objects, b: float, min_buf_px: float, max_buf_px: float) -> np.ndarray:
    boxes = []
    for t in objects:
        box = (t.copy() if isinstance(t, np.ndarray) else t.xyxy.copy()).astype(np.float64)
        w, h = box[2] - box[0], box[3] - box[1]
        dx = float(np.clip(b * w, min_buf_px, max_buf_px))
        dy = float(np.clip(b * h, min_buf_px, max_buf_px))
        box[0] -= dx; box[1] -= dy; box[2] += dx; box[3] += dy
        boxes.append(box)
    return np.asarray(boxes, dtype=np.float32)

def biou_distance(atracks, btracks, b: float = 0.0, min_buf_px: float = 40.0, max_buf_px: float = 80.0) -> np.ndarray:
    if len(atracks) == 0 or len(btracks) == 0:
        return np.empty((len(atracks), len(btracks)), dtype=np.float32)
    a_exp = _expand_boxes(atracks, b, min_buf_px, max_buf_px)
    b_exp = _expand_boxes(btracks, b, min_buf_px, max_buf_px)
    ious = AssociationFunction.iou_batch(a_exp, b_exp)
    return (1.0 - ious).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────
# 2. 底部中心點距離 & 拓撲懲罰
# ─────────────────────────────────────────────────────────────────────
def _get_bottom_center(obj) -> np.ndarray:
    box = obj if isinstance(obj, np.ndarray) else obj.xyxy
    return np.array([(box[0] + box[2]) / 2.0, box[3]], dtype=np.float64)

def bottom_center_distance(atracks, btracks, img_diag: float) -> np.ndarray:
    M, N = len(atracks), len(btracks)
    if M == 0 or N == 0: return np.empty((M, N), dtype=np.float32)
    a_bc = np.stack([_get_bottom_center(t) for t in atracks])
    b_bc = np.stack([_get_bottom_center(d) for d in btracks])
    dist = np.sqrt(((a_bc[:, None, :] - b_bc[None, :, :]) ** 2).sum(axis=2))
    if img_diag > 0: dist /= img_diag
    return dist.astype(np.float32)

def topology_penalty(atracks, btracks, track_speeds: np.ndarray, speed_thresh: float) -> np.ndarray:
    M, N = len(atracks), len(btracks)
    if M == 0 or N == 0: return np.empty((M, N), dtype=np.float32)
    trk_bc_x = np.array([_get_bottom_center(t)[0] for t in atracks], dtype=np.float64)
    det_bc_x = np.array([_get_bottom_center(d)[0] for d in btracks], dtype=np.float64)
    low_speed_mask = track_speeds < speed_thresh
    penalty = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        if not low_speed_mask[i]: continue
        ti_x = trk_bc_x[i]
        for j in range(N):
            dj_x = det_bc_x[j]
            x_lo, x_hi = min(ti_x, dj_x), max(ti_x, dj_x)
            crossings = sum(1 for k in range(M) if k != i and low_speed_mask[k] and x_lo < trk_bc_x[k] < x_hi)
            penalty[i, j] = float(crossings)
    return penalty

# ─────────────────────────────────────────────────────────────────────
# 3. 🚀 運動學錐體約束 (Kinematic Cone Penalty)
# ─────────────────────────────────────────────────────────────────────
def kinematic_cone_penalty(
    atracks, btracks, 
    track_speeds: np.ndarray, 
    track_velocities: np.ndarray,
    speed_thresh: float
) -> np.ndarray:
    """
    強迫高速車輛的下一個偵測框必須出現在「行駛方向的前方 60 度角內」。
    防堵北上車輛匹配到南下車輛 (大於 90 度位移)。
    """
    M, N = len(atracks), len(btracks)
    penalty = np.zeros((M, N), dtype=np.float32)
    if M == 0 or N == 0 or track_velocities is None:
        return penalty

    det_c = np.array([[(d.xyxy[0]+d.xyxy[2])/2, (d.xyxy[1]+d.xyxy[3])/2] for d in btracks])
    
    trk_last_c = np.zeros((M, 2), dtype=np.float64)
    for i, t in enumerate(atracks):
        # 提取消失前「最後一次真實偵測」的位置
        last = t._hist[-1] if hasattr(t, '_hist') and len(t._hist) > 0 else t.xyxy
        trk_last_c[i, 0] = (last[0] + last[2]) / 2.0
        trk_last_c[i, 1] = (last[1] + last[3]) / 2.0
        
    trk_v_c = np.zeros((M, 2), dtype=np.float64)
    trk_v_c[:, 0] = (track_velocities[:, 0] + track_velocities[:, 2]) / 2.0
    trk_v_c[:, 1] = (track_velocities[:, 1] + track_velocities[:, 3]) / 2.0

    for i in range(M):
        if track_speeds[i] < speed_thresh:
            continue # 低速轉彎不設限
            
        v_hist = trk_v_c[i]
        norm_v = np.linalg.norm(v_hist)
        if norm_v == 0: continue
        
        for j in range(N):
            v_diff = det_c[j] - trk_last_c[i]
            norm_diff = np.linalg.norm(v_diff)
            
            if norm_diff < speed_thresh:
                continue # 原地抖動不懲罰
                
            cos_sim = np.dot(v_hist, v_diff) / (norm_v * norm_diff)
            
            # cos(60度) = 0.5。如果位移方向與歷史速度方向夾角 > 60度 (如對向來車是 180度)
            # 代表這絕對是物理上不可能的匹配，直接斬斷！
            if cos_sim < 0.5:
                penalty[i, j] = 1000.0

    return penalty

# ─────────────────────────────────────────────────────────────────────
# 4. 組合代價矩陣
# ─────────────────────────────────────────────────────────────────────
def combined_cost(
    atracks, btracks, b: float, img_hw: tuple[int, int],
    min_buf_px: float = 40.0, max_buf_px: float = 80.0,
    track_speeds: np.ndarray | None = None,
    track_velocities: np.ndarray | None = None, # 🚀 重新加入以供錐體約束使用
    lambda_bcd: float = 0.15, lambda_topo: float = 0.1, topo_speed_thresh: float = 5.0,
) -> np.ndarray:
    M, N = len(atracks), len(btracks)
    if M == 0 or N == 0: return np.empty((M, N), dtype=np.float32)

    cost_biou = biou_distance(atracks, btracks, b=b, min_buf_px=min_buf_px, max_buf_px=max_buf_px)

    h, w = img_hw
    img_diag = float(np.sqrt(h * h + w * w)) if (h > 0 and w > 0) else 2202.0
    cost_bcd = bottom_center_distance(atracks, btracks, img_diag=img_diag)

    if track_speeds is not None:
        cost_topo = topology_penalty(atracks, btracks, track_speeds=track_speeds, speed_thresh=topo_speed_thresh)
    else:
        cost_topo = np.zeros((M, N), dtype=np.float32)

    # 🚀 啟動運動學錐體約束
    if track_speeds is not None and track_velocities is not None:
        cost_cone = kinematic_cone_penalty(atracks, btracks, track_speeds, track_velocities, speed_thresh=topo_speed_thresh)
    else:
        cost_cone = np.zeros((M, N), dtype=np.float32)

    cost = cost_biou + lambda_bcd * cost_bcd + lambda_topo * cost_topo + cost_cone
    return cost.astype(np.float32)
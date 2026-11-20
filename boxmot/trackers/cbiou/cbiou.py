"""
Cascaded-Buffered IoU (C-BIoU) Tracker
包含速度感知智慧回收，並傳遞向量供「運動學錐體」斬斷對向幽靈繼承。
"""

from __future__ import annotations
import numpy as np
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.cbiou.cbiou_matching import combined_cost
from boxmot.utils.matching import linear_assignment

class TrackState:
    Tracked = 0
    Lost    = 1
    Removed = 2

class CBIoUTrack:
    _count: int = 0

    def __init__(self, det: np.ndarray, n_motion: int = 5) -> None:
        det = np.asarray(det, dtype=np.float64).ravel()
        self.xyxy    = det[:4].copy()
        self.conf    = float(det[4])
        self.cls     = int(det[5])
        self.det_ind = int(det[6])
        self.track_id    : int  = 0
        self.is_activated: bool = False
        self.state       : int  = TrackState.Lost
        self.n_motion = n_motion
        self._hist   : list[np.ndarray] = []
        self.age        : int = 0
        self.hits       : int = 0
        self.frame_id   : int = 0
        self.start_frame: int = 0

    @property
    def id(self) -> int: return self.track_id
    @property
    def end_frame(self) -> int: return self.frame_id

    @classmethod
    def next_id(cls) -> int:
        cls._count += 1
        return cls._count

    def get_velocity(self) -> np.ndarray:
        if len(self._hist) >= 2:
            n   = min(self.n_motion, len(self._hist) - 1)
            pts = self._hist[-(n + 1):]
            vels = np.stack([pts[i + 1] - pts[i] for i in range(len(pts) - 1)])
            return vels.mean(axis=0)
        return np.zeros(4, dtype=np.float64)

    def get_center_speed(self) -> float:
        v = self.get_velocity()
        v_cx = (v[0] + v[2]) / 2.0
        v_cy = (v[1] + v[3]) / 2.0
        return float(np.sqrt(v_cx * v_cx + v_cy * v_cy))

    def predict(self, max_delta: int = 3, static_ratio: float = 0.02) -> None:
        if len(self._hist) < 2: return
        n   = min(self.n_motion, len(self._hist) - 1)
        pts = self._hist[-(n + 1):]
        vels = np.stack([pts[i + 1] - pts[i] for i in range(len(pts) - 1)])

        last = self._hist[-1]
        w, h = last[2] - last[0], last[3] - last[1]
        diag = np.sqrt(w * w + h * h)
        thresh = static_ratio * diag

        all_static = True
        for v in vels:
            speed = np.sqrt(((v[0]+v[2])/2.0)**2 + ((v[1]+v[3])/2.0)**2)
            if speed >= thresh:
                all_static = False
                break

        if all_static:
            self.xyxy = self._hist[-1].copy()
            return

        delta = min(self.age + 1, max_delta)
        self.xyxy = self._hist[-1] + vels.mean(axis=0) * delta

    @staticmethod
    def multi_predict(tracks: list[CBIoUTrack], max_delta: int = 3, static_ratio: float = 0.02) -> None:
        for t in tracks: t.predict(max_delta=max_delta, static_ratio=static_ratio)

    def activate(self, frame_id: int) -> None:
        self.track_id, self.state, self.is_activated = self.next_id(), TrackState.Tracked, True
        self.frame_id, self.start_frame, self.hits, self.age = frame_id, frame_id, 1, 0
        self._hist = [self.xyxy.copy()]

    def update(self, det: np.ndarray, frame_id: int) -> None:
        det = np.asarray(det, dtype=np.float64).ravel()
        self.xyxy, self.conf, self.cls, self.det_ind = det[:4].copy(), float(det[4]), int(det[5]), int(det[6])
        self._hist.append(self.xyxy.copy())
        if len(self._hist) > self.n_motion + 1: self._hist.pop(0)
        self.state, self.is_activated, self.frame_id, self.hits, self.age = TrackState.Tracked, True, frame_id, self.hits + 1, 0

    def mark_lost(self) -> None:
        self.state = TrackState.Lost
        self.age  += 1

    def mark_removed(self) -> None:
        self.state = TrackState.Removed

def _safe_index(dets: np.ndarray, indices) -> np.ndarray:
    idx = list(indices) if not isinstance(indices, list) else indices
    if len(idx) == 0: return np.empty((0, dets.shape[1]), dtype=dets.dtype)
    return np.stack([dets[i] for i in idx], axis=0)

def _extract_velocities(tracks: list[CBIoUTrack]) -> tuple[np.ndarray, np.ndarray]:
    if len(tracks) == 0: return np.empty((0, 4), dtype=np.float64), np.empty(0, dtype=np.float64)
    vels = np.stack([t.get_velocity() for t in tracks])
    spds = np.array([t.get_center_speed() for t in tracks], dtype=np.float64)
    return vels, spds

class CBIoU(BaseTracker):
    def __init__(
        self, b1: float = 0.4, b2: float = 0.8, n_motion: int = 5, track_buffer: int = 3600,
        frame_rate: int = 15, min_conf: float = 0.1, track_thresh: float = 0.45, match_thresh: float = 0.75,
        min_buf_px: float = 40.0, max_buf_px: float = 80.0,
        max_delta: int = 3, static_ratio: float = 0.02,
        lambda_bcd: float = 0.15, lambda_topo: float = 0.1, topo_speed_thresh: float = 5.0,
        **kwargs,
    ) -> None:
        init_args = {k: v for k, v in locals().items() if k not in ('self', 'kwargs')}
        super().__init__(**init_args, _tracker_name='CBIoU', **kwargs)

        self.b1, self.b2, self.n_motion = b1, b2, n_motion
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.track_thresh, self.match_thresh, self.min_conf = track_thresh, match_thresh, min_conf
        
        self.min_buf_px = min_buf_px
        self.max_buf_px = max_buf_px
        
        self.max_delta, self.static_ratio = max_delta, static_ratio
        self.lambda_bcd, self.lambda_topo, self.topo_speed_thresh = lambda_bcd, lambda_topo, topo_speed_thresh

        self._tracks: list[CBIoUTrack] = []

    def _make(self, dets_2d: np.ndarray) -> list[CBIoUTrack]:
        return [CBIoUTrack(row, self.n_motion) for row in dets_2d]

    def _get_img_hw(self, img: np.ndarray | None, dets: np.ndarray) -> tuple[int, int]:
        if img is not None: return int(img.shape[0]), int(img.shape[1])
        if len(dets) > 0: return int(dets[:, 3].max() * 1.1), int(dets[:, 2].max() * 1.1)
        return 1080, 1920

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray = None, embs: np.ndarray = None) -> np.ndarray:
        self.check_inputs(dets, img)
        self.frame_count += 1
        img_hw = self._get_img_hw(img, dets)

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        dets = dets[dets[:, 4] >= self.min_conf]
        all_dets = self._make(dets) if len(dets) else []

        CBIoUTrack.multi_predict(self._tracks, max_delta=self.max_delta, static_ratio=self.static_ratio)
        trk_vels, trk_spds = _extract_velocities(self._tracks)

        match_kwargs = {
            'img_hw': img_hw,
            'min_buf_px': self.min_buf_px,
            'max_buf_px': self.max_buf_px,
            'lambda_bcd': self.lambda_bcd,
            'lambda_topo': self.lambda_topo,
            'topo_speed_thresh': self.topo_speed_thresh
        }

        # ── Stage 1 ─────────────────────────────────────────────────────────
        # 🚀 重新加入 track_velocities=trk_vels 參數
        c1 = combined_cost(self._tracks, all_dets, b=self.b1, track_speeds=trk_spds, track_velocities=trk_vels, **match_kwargs)
        m1, u_trk1, u_det1 = linear_assignment(c1, thresh=self.match_thresh)
        for r, c in m1: self._tracks[r].update(dets[c], self.frame_count)

        # ── Stage 2 ─────────────────────────────────────────────────────────
        trk2 = [self._tracks[i] for i in u_trk1]
        det2_raw = _safe_index(dets, u_det1)
        det2_obj = self._make(det2_raw)
        trk2_vels, trk2_spds = _extract_velocities(trk2)

        # 🚀 重新加入 track_velocities=trk2_vels 參數
        c2 = combined_cost(trk2, det2_obj, b=self.b2, track_speeds=trk2_spds, track_velocities=trk2_vels, **match_kwargs)
        m2, u_trk2, u_det2 = linear_assignment(c2, thresh=self.match_thresh)
        for r, c in m2: trk2[r].update(det2_raw[c], self.frame_count)

        for i in u_trk2: trk2[i].mark_lost()
        for i in u_det2:
            d = det2_raw[i]
            if float(d[4]) >= self.track_thresh:
                t = CBIoUTrack(d, self.n_motion)
                t.activate(self.frame_count)
                self._tracks.append(t)

        # ── 速度感知智慧回收 (Speed-Aware GC) ──────────────────────────
        alive = []
        for t in self._tracks:
            if t.state == TrackState.Lost:
                h, w = img_hw
                margin_w, margin_h = w * 0.1, h * 0.1
                is_near_edge = (t.xyxy[0] < margin_w or t.xyxy[2] > (w - margin_w) or 
                                t.xyxy[1] < margin_h or t.xyxy[3] > (h - margin_h))
                
                speed = t.get_center_speed()
                is_fast = speed > self.topo_speed_thresh
                
                if is_near_edge:
                    effective_max_lost = 30
                elif is_fast:
                    effective_max_lost = 60
                else:
                    effective_max_lost = self.max_time_lost

                if t.age > effective_max_lost:
                    t.mark_removed()

            if t.state != TrackState.Removed:
                alive.append(t)
                
        self._tracks = alive

        out = [[*t.xyxy, t.id, t.conf, t.cls, t.det_ind] for t in self._tracks if t.is_activated and t.state == TrackState.Tracked]
        return np.asarray(out, dtype=np.float32) if out else np.empty((0, 8), dtype=np.float32)
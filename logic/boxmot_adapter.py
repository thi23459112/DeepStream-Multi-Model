"""
BoxMOT 追蹤器介接

每路攝影機一個獨立的追蹤器（以全域 uid 為鍵），軌跡 ID 不會跨攝影機混在一起。
探針只需要呼叫 track(uid, dets)。

BoxMOT 的 update(dets, img) 需要一張影像，但本專案不把畫面從 GPU 拷回 CPU，
所以只支援不看像素的追蹤器，依它們對 img 的用法分兩類：

    FRAME_FREE  完全不讀 img     → 傳 1×1 的假影像
    SHAPE_ONLY  只讀 img.shape   → 傳 1080×1920 的全黑影像（啟動時配置一次）

需要真實像素的追蹤器（botsort、strongsort、deepocsort 等，要做 CMC 或 ReID）不支援。
新增追蹤器時，先確認它屬於哪一類，再加進對應的集合與 config.SUPPORTED_BOXMOT_TRACKERS。
"""

import numpy as np
from boxmot import create_tracker

from logic.common import MUX_OUTPUT_W, MUX_OUTPUT_H
from logic.config import SOURCE_CONFIGS, TRACKER_MODE, BOXMOT_TRACKER_CONFIG


# =============================================================================
# 常數與狀態
# =============================================================================

FRAME_FREE = {"bytetrack", "ocsort", "fasttracker"}
SHAPE_ONLY = {"sfsort", "cbiou"}                    # sfsort 用寬高算邊界；cbiou 用對角線長度做正規化

_EMPTY_TRACKS = np.empty((0, 8), dtype=np.float32)  # 沒有任何軌跡時的回傳值

_trackers = {}                                      # uid → 追蹤器
_frame = None                                       # 傳給 update() 的假影像；所有攝影機共用，內容不會被讀取


# =============================================================================
# 函式
# =============================================================================

def initialize_boxmot_trackers():
    """為每路攝影機建立追蹤器。BoxMOT 模式下由 main.py 在啟動時呼叫一次。"""
    global _frame

    if TRACKER_MODE in FRAME_FREE:
        _frame = np.zeros((1, 1, 3), dtype=np.uint8)    # 不會被讀，越小越好
    elif TRACKER_MODE in SHAPE_ONLY:
        # 尺寸要等於物件座標系
        _frame = np.zeros((MUX_OUTPUT_H, MUX_OUTPUT_W, 3), dtype=np.uint8)
    else:
        raise RuntimeError(
            f"追蹤器 '{TRACKER_MODE}' 未歸類（FRAME_FREE / SHAPE_ONLY），"
            f"請先在 logic/boxmot_adapter.py 確認它對影像的需求"
        )

    for uid in SOURCE_CONFIGS:
        _trackers[uid] = create_tracker(
            tracker_type=TRACKER_MODE,
            tracker_config=BOXMOT_TRACKER_CONFIG,
            reid_weights=None,                          # 不使用 ReID 模型
            device="cpu",                               # 這幾種追蹤器只有卡爾曼濾波與匹配，CPU 即可
            half=False,                                 # CPU 不支援 FP16
            per_class=False,                            # 所有類別放在一起追蹤；類別過濾已在探針做完
        )
    print(f"[INFO] BoxMOT：已為 {len(_trackers)} 路 cam 建立 {TRACKER_MODE} 追蹤器"
          f"（參數檔 {BOXMOT_TRACKER_CONFIG}）")


def track(uid, dets):
    """
    更新該攝影機的追蹤器。

    參數：
        uid   全域編號
        dets  shape (N, 6) 的 float32 陣列，每列 [x1, y1, x2, y2, conf, cls]；N 可為 0
    回傳：
        shape (M, 8) 的 float32 陣列，每列 [x1, y1, x2, y2, track_id, conf, cls, det_index]
    """
    dets = np.asarray(dets, dtype=np.float32)
    if dets.ndim != 2 or dets.shape[1] != 6:
        return _EMPTY_TRACKS                        # 格式不對時不更新，避免 BoxMOT 拋例外
    tracks = _trackers[uid].update(dets, _frame)    # N = 0 也要呼叫，追蹤器才會累計消失幀數
    if tracks is None or len(tracks) == 0:
        return _EMPTY_TRACKS
    return np.asarray(tracks, dtype=np.float32)

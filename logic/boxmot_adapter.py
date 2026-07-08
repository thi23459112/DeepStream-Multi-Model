"""
BoxMOT 追蹤器轉接層

主要功能：
1. 每路 cam 獨立 tracker instance：避免多路畫面的軌跡 ID 互相污染
2. 統一介面 track(pad_index, dets, frame)：probe 只管餵偵測結果與取追蹤結果
3. 3-tier frame 處理策略：依各追蹤器對 img 參數的實際使用方式分級
   - TIER_A：完全不依賴 frame      → 共用 1×1 dummy
   - TIER_B：只用 img.shape 做幾何 → 每路預備 zero frame，不拷像素
   - TIER_C：需要真實 BGR 像素     → 目前不支援
"""

import numpy as np
from boxmot import create_tracker

from logic.config import (
    SOURCE_CONFIGS, TRACKER_MODE, BOXMOT_TRACKER_CONFIG,
)


# ==========================================
# 1. 系統配置區 (System Configuration)
# ==========================================

# --- 3-tier 追蹤器分類 ---
# 依各追蹤器 update() 對 img 參數的實際使用方式區分
TIER_A_DUMMY_OK = {
    "bytetrack",     # IoU only，img 完全沒讀
    "ocsort",        # h, w 有傳進 associate() 但內部當 dead arg
    "fasttracker",   # img 只出現在 update 簽名跟 check_inputs
}

TIER_B_NEEDS_SHAPE_ONLY = {
    "sfsort",        # 用 W/H 設 lost track 邊界 margin
    "cbiou",         # 用 sqrt(W²+H²) 做距離正規化
}

TIER_C_NEEDS_PIXELS = {
    "botsort",       # CMC (ECC/SOF) + 可選 ReID
    "boosttrack",    # CMC (ECC) + 可選 ReID
    "strongsort",    # CMC (ECC) + ReID 必開
    "deepocsort",    # CMC (SOF) + ReID
    "hybridsort",    # CMC (ECC) + ReID
    "imprassoc",     # CMC (SOF) + ReID
}

# --- TIER_A 共用 dummy frame ---
# 1×1×3 zero ndarray，module-level 共用同一塊記憶體，零開銷
_DUMMY_FRAME_TIER_A = np.zeros((1, 1, 3), dtype=np.uint8)

# --- 執行期狀態 ---
# pad_index → tracker instance（每路 cam 各一個）
_trackers = {}

# pad_index → zero ndarray (H, W, 3)，僅 TIER_B 使用
# 啟動時依 SOURCE_CONFIGS[pad_index]['geometry']['base_w/base_h'] 預先建好
_shape_cache = {}


# ==========================================
# 2. 模式查詢功能 (Mode Query)
# ==========================================

def is_boxmot_mode() -> bool:
    """
    是否處於 BoxMOT 模式

    返回：
        bool: True 表示走 BoxMOT pipeline；False 表示用 nvdcf
    """
    return TRACKER_MODE != "nvdcf"


def get_tracker_tier() -> str:
    """
    取得目前追蹤器所屬 tier 的描述字串

    返回：
        str: tier 描述（A / B / C / unknown），給 log 顯示用
    """
    if TRACKER_MODE in TIER_A_DUMMY_OK:
        return "A (dummy 1x1)"
    if TRACKER_MODE in TIER_B_NEEDS_SHAPE_ONLY:
        return "B (per-cam zero frame)"
    if TRACKER_MODE in TIER_C_NEEDS_PIXELS:
        return "C (needs real pixels — NOT IMPLEMENTED)"
    return "unknown"


# ==========================================
# 3. 啟動初始化 (Startup Initialization)
# ==========================================

def initialize_boxmot_trackers():
    """
    為每路 cam 建立 tracker instance 並預備 frame cache

    處理流程：
    1. 檢查 TRACKER_MODE：nvdcf 模式直接跳過
    2. 防呆：擋下 C 級、未知級別、缺設定檔的情況
    3. 為每個 pad_index 呼叫 create_tracker() 建出獨立 instance
    4. TIER_B 額外預備 per-cam zero frame（依 base_w/base_h）

    註：本函式應在 main.py 啟動時呼叫一次
    """
    # 步驟 1: nvdcf 模式不需要 BoxMOT 資源
    if not is_boxmot_mode():
        print("[INFO] [boxmot_adapter] 非 BoxMOT 模式，略過 tracker 初始化")
        return

    # 步驟 2: 防呆檢查
    if TRACKER_MODE in TIER_C_NEEDS_PIXELS:
        raise RuntimeError(
            f"[boxmot_adapter] 追蹤器 '{TRACKER_MODE}' 屬於 C 級（需真實 BGR 像素），"
            f"目前未實作 NVMM→CPU 拷貝邏輯，不支援。\n"
            f"  可用：{sorted(TIER_A_DUMMY_OK | TIER_B_NEEDS_SHAPE_ONLY)}"
        )

    if TRACKER_MODE not in (TIER_A_DUMMY_OK | TIER_B_NEEDS_SHAPE_ONLY):
        raise RuntimeError(
            f"[boxmot_adapter] 未知的追蹤器分級：'{TRACKER_MODE}'。"
            f"請在 logic/boxmot_adapter.py 的 TIER_A/B/C 三個集合中分類後再使用。"
        )

    if BOXMOT_TRACKER_CONFIG is None:
        raise RuntimeError(
            f"[boxmot_adapter] TRACKER_MODE='{TRACKER_MODE}' 但 BOXMOT_TRACKER_CONFIG 為 None；"
            f"請重新執行 traffic_count_txt.py 產生 config_tracker_runtime.txt"
        )

    tier = get_tracker_tier()
    print(f"[INFO] [boxmot_adapter] 追蹤器 '{TRACKER_MODE}' 屬於 TIER {tier}")
    print(f"[INFO] [boxmot_adapter] 為 {len(SOURCE_CONFIGS)} 路 cam 各建立一個 {TRACKER_MODE} tracker")

    # 步驟 3: 每路 cam 各建一個 tracker instance
    for pad_index in SOURCE_CONFIGS.keys():
        tr = create_tracker(
            tracker_type=TRACKER_MODE,
            tracker_config=BOXMOT_TRACKER_CONFIG,
            reid_weights=None,    # A/B 級都不需要 ReID
            device="cpu",         # A/B 級都不跑模型
            half=False,
            per_class=False,      # PGIE 多車種都收，BoxMOT 內部不再做 per-class 分流
        )
        _trackers[pad_index] = tr
        print(f"[INFO] [boxmot_adapter]   pad_index={pad_index} → {type(tr).__name__}")

    # 步驟 4: TIER_B 預備 per-cam zero frame
    # uint8 + zeros：1080×1920×3 ≈ 6 MB / cam，整個 pipeline 生命週期只配置一次
    if TRACKER_MODE in TIER_B_NEEDS_SHAPE_ONLY:
        for pad_index, cfg in SOURCE_CONFIGS.items():
            base_w = int(cfg.get("geometry", {}).get("base_w", 1920))
            base_h = int(cfg.get("geometry", {}).get("base_h", 1080))
            _shape_cache[pad_index] = np.zeros((base_h, base_w, 3), dtype=np.uint8)
            print(f"[INFO] [boxmot_adapter]   pad_index={pad_index} → zero frame "
                  f"shape=({base_h}, {base_w}, 3) (TIER B 用)")


# ==========================================
# 4. Frame 處理輔助 (Frame Dispatch)
# ==========================================

def _get_frame_for_update(pad_index: int, real_frame: np.ndarray) -> np.ndarray:
    """
    依 TRACKER_MODE 所屬 tier，決定 tracker.update() 該傳什麼 frame

    參數：
        pad_index (int): 哪一路 cam
        real_frame (numpy.ndarray | None): probe 端可能（將來）從 NVMM 拷貝來的真實 BGR ndarray

    返回：
        numpy.ndarray: 對應 tier 該用的 frame
            - TIER_A: 1×1 共用 dummy（real_frame 即使有也忽略）
            - TIER_B: 該 pad_index 對應的 cached zero frame
            - TIER_C: 直接回傳 real_frame，None 則 raise
    """
    if TRACKER_MODE in TIER_A_DUMMY_OK:
        return _DUMMY_FRAME_TIER_A

    if TRACKER_MODE in TIER_B_NEEDS_SHAPE_ONLY:
        cached = _shape_cache.get(pad_index)
        if cached is None:
            raise RuntimeError(
                f"[boxmot_adapter] TIER_B 模式但 pad_index={pad_index} 沒有 cached zero frame。"
                f"請確認 initialize_boxmot_trackers() 是否有正確呼叫。"
            )
        return cached

    # TIER_C
    if real_frame is None:
        raise RuntimeError(
            f"[boxmot_adapter] 追蹤器 '{TRACKER_MODE}' 屬於 C 級，需要 probe 端傳入真實 frame，"
            f"但目前 frame=None。請檢查 probe 是否有把 NVMM 拷貝到 CPU。"
        )
    return real_frame


# ==========================================
# 5. 對外主介面 (Public API)
# ==========================================

def track(pad_index: int, dets: np.ndarray, frame: np.ndarray = None) -> np.ndarray:
    """
    呼叫對應 cam 的 tracker.update() 取得追蹤結果

    參數：
        pad_index (int): 哪一路 cam
        dets (numpy.ndarray): shape=(N, 6), [x1, y1, x2, y2, conf, cls]
        frame (numpy.ndarray | None): (H, W, 3) BGR 影像
            - A/B 級：傳 None 即可，本檔自動回退到 dummy / cached zero
            - C 級  ：必須傳真實 frame（從 NVMM 拷貝） — 目前不支援

    返回：
        numpy.ndarray: shape=(M, 8), [x1, y1, x2, y2, id, conf, cls, det_ind]
                       追蹤器尚未 confirm 或全 lost 時 M=0
    """
    tr = _trackers.get(pad_index)
    if tr is None:
        # 理論上不會發生，除非 initialize_boxmot_trackers() 沒被呼叫
        return np.empty((0, 8), dtype=np.float32)

    # 確保 dets 至少是 (0, 6) 而非 (0,)，避免 BoxMOT 內 shape 檢查炸掉
    if dets is None or len(dets) == 0:
        dets = np.empty((0, 6), dtype=np.float32)

    dets = np.asarray(dets, dtype=np.float32)
    if dets.ndim != 2 or dets.shape[1] != 6:
        # 防呆：格式錯誤直接回空，避免拖垮 pipeline
        return np.empty((0, 8), dtype=np.float32)

    update_frame = _get_frame_for_update(pad_index, frame)

    tracks = tr.update(dets, update_frame)
    if tracks is None or len(tracks) == 0:
        return np.empty((0, 8), dtype=np.float32)
    return np.asarray(tracks, dtype=np.float32)

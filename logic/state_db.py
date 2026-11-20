"""
軌跡狀態與 SQLite 寫入

流程：
    1. 探針每幀更新 track_history（每個追蹤中的物件一份狀態）。
    2. 物件連續消失 cleanup_frames 幀 → 探針呼叫 retire_track() 結算：
           方向已定（flow_in / flow_out），且某 ROI 的命中幀數 >= min_roi_hits
           → 該 ROI 產生一筆紀錄；經過幾個 ROI 就幾筆。
           方向未定（位移不足，視為抖動）→ 整個物件不計。
    3. 紀錄先放進 pending_records，再由 flush_pending_to_db() 批次寫入。
    4. 程式結束時 force_finalize_all() 結算仍在畫面上的物件並寫完。

所有攝影機寫進同一個檔（MERGED_DB_PATH），以 CameraCode 區分。
資料表 detection_stats：
    DeviceCode    device.code
    CameraCode    source_id
    TrackID       該攝影機內的短 ID（1 ~ LOCAL_ID_MAX 循環）
    DetectClass   類別名稱（在 ROI 內各幀類別的多數決）
    LocationName  ROI 名稱
    MetricType    flow_in / flow_out
    HitCount      在該 ROI 內的命中幀數
    VideoTime     物件最後出現的影片時間 HH:MM:SS
    CollectTime   檔案來源 = start_time + 影片秒數；即時來源 = 寫入當下的系統時間

部署時 send_interval_api.py 會讀這張表上傳，並刪除已上傳的列：
改欄位名稱或 CollectTime 格式前，必須一併修改它。

本檔所有 dict 都以全域 uid 為鍵。
"""

import os
import time
import sqlite3
import threading
from collections import deque
from datetime import timedelta

from logic.config import SOURCE_CONFIGS, LOCAL_ID_MAX, MERGED_DB_PATH


# =============================================================================
# 常數與狀態
# =============================================================================

FPS_WINDOW_FRAMES = 30                      # 即時 FPS 以最近幾幀計算；重連的滴幀判斷也依賴這個長度
DB_BUSY_TIMEOUT_SECONDS = 30                # DB 被其他程式（上傳腳本）鎖住時，寫入最多等多久（秒）
COLLECT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"   # CollectTime 格式；上傳腳本依此格式比對

track_history    = {}                       # (uid, 追蹤器 object_id) → 軌跡狀態（欄位見 probes._update_tracks）
pending_records  = {}                       # uid → 尚未寫入 DB 的紀錄 list
last_flush_times = {}                       # uid → 上次寫入 DB 的時間（time.time()）
fps_streams      = {}                       # uid → {"current_fps": float, "timestamps": 最近幾幀收到時間的 deque}
local_id_maps    = {}                       # uid → {追蹤器 object_id: 短 ID}
next_local_ids   = {}                       # uid → 下一個要發的短 ID

_db_conn = None                             # 共用連線；None 表示所有攝影機都不寫 DB
_db_lock = threading.Lock()                 # 各 pipeline 的探針跑在不同執行緒，寫入要序列化

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS detection_stats (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    DeviceCode   TEXT    NOT NULL,
    CameraCode   TEXT    NOT NULL,
    TrackID      INTEGER NOT NULL,
    DetectClass  TEXT,
    LocationName TEXT    NOT NULL,
    MetricType   TEXT    NOT NULL,
    HitCount     INTEGER NOT NULL,
    VideoTime    TEXT,
    CollectTime  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_camera_time  ON detection_stats (CameraCode, CollectTime);
CREATE INDEX IF NOT EXISTS idx_locationname ON detection_stats (LocationName);
CREATE INDEX IF NOT EXISTS idx_metrictype   ON detection_stats (MetricType);
"""

# 欄位順序與 _finalize() 組出的 tuple 一致
_INSERT_SQL = (
    "INSERT INTO detection_stats "
    "(DeviceCode, CameraCode, TrackID, DetectClass, LocationName, MetricType, "
    "HitCount, VideoTime, CollectTime) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


# =============================================================================
# 初始化
# =============================================================================

def initialize_state_managers():
    """建立每路攝影機的狀態；只要有任一路 save_output_db=true 就開啟 DB。"""
    global _db_conn

    now = time.time()
    for uid in SOURCE_CONFIGS:
        pending_records[uid]  = []
        last_flush_times[uid] = now
        fps_streams[uid]      = {"current_fps": 0.0, "timestamps": deque(maxlen=FPS_WINDOW_FRAMES)}
        local_id_maps[uid]    = {}
        next_local_ids[uid]   = 1                   # 短 ID 從 1 起算

    if not any(cfg["save_output_db"] for cfg in SOURCE_CONFIGS.values()):
        print("[INFO] 所有 cam 的 save_output_db 皆為 false，不寫 DB（只印結算 log）")
        return

    os.makedirs(os.path.dirname(MERGED_DB_PATH), exist_ok=True)
    _db_conn = sqlite3.connect(
        MERGED_DB_PATH,
        check_same_thread=False,                    # 探針在 GStreamer 串流執行緒呼叫，不是開連線的執行緒
        isolation_level=None,                       # 不自動開交易，由 flush_pending_to_db() 自己 BEGIN / COMMIT
        timeout=DB_BUSY_TIMEOUT_SECONDS,            # 撞到上傳腳本的寫入鎖時等待，而不是立刻失敗
    )
    _db_conn.execute("PRAGMA journal_mode=WAL")     # 寫入時其他程式仍可讀；會多出 -wal / -shm 兩個檔
    _db_conn.execute("PRAGMA synchronous=NORMAL")   # WAL 下仍安全，比預設 FULL 少很多次 fsync
    _db_conn.executescript(_SCHEMA_SQL)
    print(f"[INFO] DB：{MERGED_DB_PATH}")


# =============================================================================
# 短 ID
# =============================================================================

def get_local_id(uid, object_id):
    """
    把追蹤器給的 object_id 換成該攝影機內從 1 起算的短 ID（畫面上的 ID、DB 的 TrackID）。

    發到 LOCAL_ID_MAX 後回到 1；重複的 TrackID 以 CameraCode + CollectTime 區分。
    """
    id_map = local_id_maps[uid]
    if object_id not in id_map:
        id_map[object_id] = next_local_ids[uid]
        if next_local_ids[uid] >= LOCAL_ID_MAX:
            print(f"[INFO] {SOURCE_CONFIGS[uid]['source_id']} 的 ID 已達上限 {LOCAL_ID_MAX}，下一個從 1 重新編號")
            next_local_ids[uid] = 1
        else:
            next_local_ids[uid] += 1
    return id_map[object_id]


# =============================================================================
# 結算
# =============================================================================

def _format_video_time(seconds):
    """秒數 → "HH:MM:SS"（負值視為 0）。"""
    return time.strftime("%H:%M:%S", time.gmtime(max(0, int(seconds))))


def _finalize(key, state, forced):
    """
    把一個物件的軌跡結算成 0 ~ N 筆紀錄，放進 pending_records（規則見檔頭）。

    參數：
        key     (uid, object_id)
        state   track_history[key]
        forced  True 表示程式結束時的強制結算，只影響 log 標示
    """
    uid, object_id = key
    cfg = SOURCE_CONFIGS[uid]

    direction = state["direction"]
    if direction == "NA":
        return                                                  # 位移始終不足：抖動或停著，不計
    min_hits = cfg["track_logic"]["min_roi_hits"]
    hit_rois = {roi: hits for roi, hits in state["roi_hits"].items() if hits >= min_hits}
    if not hit_rois:
        return                                                  # 沒有任何 ROI 達到命中門檻

    cam_name = cfg["source_id"]
    local_id = get_local_id(uid, object_id)

    if state["class_votes"]:
        cls_id = state["class_votes"].most_common(1)[0][0]      # ROI 內票數最多的類別
        cls_name = state["class_map"].get(cls_id, f"Class_{cls_id}")
    else:
        cls_name = "Unknown"

    video_sec = state["last_frame_num"] / cfg["stream_fps"]     # 幀號 → 影片秒數
    video_time = _format_video_time(video_sec)
    start_dt = cfg["start_time_dt"]
    if start_dt is not None:
        collect_time = (start_dt + timedelta(seconds=video_sec)).strftime(COLLECT_TIME_FORMAT)
    else:
        collect_time = time.strftime(COLLECT_TIME_FORMAT)       # RTSP，或影片沒填 start_time

    tag = "[結算-強制]" if forced else " "
    for roi, hits in hit_rois.items():                          # 經過幾個 ROI 就寫幾筆
        note = ""
        if cfg["save_output_db"]:
            pending_records[uid].append((
                cfg["device_code"], cam_name, local_id, cls_name,
                roi, direction, hits, video_time, collect_time,
            ))
        else:
            note = "  (不寫 DB)"
        print(f"{tag}[{cam_name}] ID={local_id}, 分類={cls_name}, 位置={roi}, 方向={direction}, "
              f"次數={hits}, 時間軸={video_time}, 時間點={collect_time}{note}")


def retire_track(key):
    """物件已消失夠久：結算，並從 track_history 與短 ID 對照表移除。"""
    _finalize(key, track_history.pop(key), forced=False)
    local_id_maps[key[0]].pop(key[1], None)     # 釋放對照，避免長時間執行時表越長越大


# =============================================================================
# 寫入 DB
# =============================================================================

def flush_pending_to_db(uid):
    """
    把該攝影機累積的紀錄以單一交易寫入。

    寫入失敗會 rollback，紀錄留在記憶體，下次再試。

    回傳：
        實際寫入的筆數
    """
    records = pending_records[uid]
    if not records:
        return 0
    if _db_conn is None:
        records.clear()     # 不寫 DB 的模式：結算 log 已印過，直接丟掉
        return 0

    with _db_lock:
        try:
            _db_conn.execute("BEGIN")
            _db_conn.executemany(_INSERT_SQL, records)
            _db_conn.execute("COMMIT")
        except sqlite3.Error as e:
            try:
                _db_conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass        # 交易可能根本沒開成功，rollback 失敗可忽略
            print(f"[ERROR] DB 寫入失敗（{SOURCE_CONFIGS[uid]['source_id']}）：{e}")
            return 0
    count = len(records)
    records.clear()
    return count


def force_finalize_all():
    """程式結束前呼叫：結算所有仍在追蹤中的物件、寫完所有紀錄、關閉 DB。"""
    global _db_conn

    print("\n[INFO] 結算仍在畫面上的物件...")
    for key, state in list(track_history.items()):
        _finalize(key, state, forced=True)
    track_history.clear()

    total = sum(flush_pending_to_db(uid) for uid in SOURCE_CONFIGS)
    if total:
        print(f"[INFO] 結束前寫入 {total} 筆 → {MERGED_DB_PATH}")

    if _db_conn is not None:
        _db_conn.close()
        _db_conn = None

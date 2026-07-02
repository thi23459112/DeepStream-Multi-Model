"""
SQLite 事件紀錄與軌跡狀態管理（車流版）

主要功能：
1. 單一合併 SQLite DB：所有 cam 的紀錄寫進「同一個」.db 檔（靠 CameraCode 欄位區分哪一路）
2. 軌跡狀態維護：每台車一份 track_history，記錄位置、方向、ROI 命中、車種投票
3. 消失時結算：物件連續 N 幀未出現 → 對每個達門檻的 ROI 各 emit 一筆 DB 紀錄
4. 方向過濾：只有 IN / OUT 才寫 DB；NA (抖動誤判) 整筆丟掉
5. 批次 flush 機制：累積在記憶體 pending_records，定期批次寫入 DB
6. save_output_db=false 旗標：純跑統計、不開連線、零 DB IO
7. local_id 循環機制：每路 cam 累積到 LOCAL_ID_MAX 後歸 1 重新計算

DB 欄位對照（本版把原本的 ROI 欄位改名為 LocationName）：
    DeviceCode / CameraCode / TrackID / DetectClass / LocationName /
    MetricType / HitCount / VideoTime / CollectTime
"""

import os
import time
import sqlite3
import threading
from datetime import timedelta

from logic.config import SOURCE_CONFIGS, LOCAL_ID_MAX, MERGED_DB_PATH
from logic.color import CLASS_MAP


# ==========================================
# 1. 系統配置區 (System Configuration)
# ==========================================

# --- 全域狀態字典 (供 probes.py 直接 import 使用) ---
track_history    = {}    # (pad_index, obj_id) → 軌跡狀態 dict
pending_records  = {}    # pad_index → 待寫入 DB 的 tuple list
last_flush_times = {}    # pad_index → 上次 flush 的時間戳
fps_streams      = {}    # pad_index → {"current_fps", "timestamps"}
local_id_maps    = {}    # pad_index → {global_id: local_id}
next_local_ids   = {}    # pad_index → 下一個可分配的 local_id（達 LOCAL_ID_MAX 後歸 1）

# --- SQLite 連線管理（單一合併 DB）---
# 所有 cam 共用「一條」連線寫進同一個 .db 檔；多路寫入靠 _db_lock 序列化。
_db_conn = None                   # 單一共用 sqlite3.Connection（None = 未開 / 全部停用）
_db_lock = threading.Lock()       # 寫入批次的執行緒鎖

# --- DB Schema ---
# ⭐ 原 ROI 欄位改名為 LocationName（索引也一併改名）
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

CREATE INDEX IF NOT EXISTS idx_camera_time
    ON detection_stats (CameraCode, CollectTime);

CREATE INDEX IF NOT EXISTS idx_locationname
    ON detection_stats (LocationName);

CREATE INDEX IF NOT EXISTS idx_metrictype
    ON detection_stats (MetricType);
"""


# ==========================================
# 2. DB 連線輔助 (Connection Helper)
# ==========================================

def _open_merged_db(db_path):
    """
    開啟「單一合併」SQLite 連線並建立 schema

    使用 WAL 模式提升併發寫入效能，synchronous=NORMAL 兼顧速度與資料安全。
    check_same_thread=False：允許探針執行緒與主執行緒共用同一連線（寫入已用 _db_lock 保護）。

    參數：
        db_path (str): 合併 DB 的 .db 檔絕對路徑
    返回：
        sqlite3.Connection: 已建立 schema 的 DB 連線
    """
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA_SQL)
    print(f"[INFO] SQLite 合併 DB 開啟: {db_path}")
    return conn


def _format_video_time(vsec):
    """
    秒數轉成 HH:MM:SS 字串

    參數：
        vsec (float): 秒數
    返回：
        str: "HH:MM:SS" 格式；負值或 None 回傳 "00:00:00"
    """
    if vsec is None or vsec < 0:
        return "00:00:00"
    return time.strftime("%H:%M:%S", time.gmtime(int(vsec)))


# ==========================================
# 3. 啟動初始化 (Startup Initialization)
# ==========================================

def initialize_state_managers():
    """
    為每一路 cam 初始化狀態字典，並開啟「單一合併」DB 連線

    處理流程：
    1. 為每路 cam 初始化所有狀態字典（pending / fps / local_id 等）
    2. 只要有「任一路」的 save_output_db=true，就開啟一個合併 DB 連線；
       全部都 false 時完全不開連線（emit/flush 走 no-op 分支）。

    註：本函式應在 main.py 啟動時呼叫一次
    """
    global _db_conn

    # 步驟 1: 每路 cam 的狀態字典初始化
    for pad_index, cfg in SOURCE_CONFIGS.items():
        pending_records[pad_index]  = []
        last_flush_times[pad_index] = time.time()
        fps_streams[pad_index]      = {"current_fps": 0.0}
        local_id_maps[pad_index]    = {}
        next_local_ids[pad_index]   = 1

    # 步驟 2: 只要有任一路要寫 DB，就開「一個」合併連線
    any_save = any(cfg.get("save_output_db", True) for cfg in SOURCE_CONFIGS.values())
    if any_save:
        _db_conn = _open_merged_db(MERGED_DB_PATH)
    else:
        _db_conn = None
        print("[INFO] 所有 cam 皆 save_output_db=false，停用 DB 寫入（純跑統計）")


# ==========================================
# 4. ID 管理 (ID Mapping)
# ==========================================

def get_local_id(pad_index, global_id):
    """
    將追蹤器給的 global_id 映射成該路 cam 內的短 local_id

    循環機制：local_id 從 1 累加到 LOCAL_ID_MAX，下一個 global_id 拿到的會是 1
              （每路 cam 各自獨立循環，互不干擾）
              撞號的紀錄靠 DB CollectTime + CameraCode 區分，查詢時記得帶條件

    參數：
        pad_index (int): 哪一路 cam
        global_id (int): 追蹤器給的物件 ID
    返回：
        int: 該路內遞增的短 ID（範圍 1 ~ LOCAL_ID_MAX，達上限後歸 1）
    """
    if global_id not in local_id_maps[pad_index]:
        local_id_maps[pad_index][global_id] = next_local_ids[pad_index]

        # 達上限歸 1，否則 +1
        if next_local_ids[pad_index] >= LOCAL_ID_MAX:
            cam_name = SOURCE_CONFIGS.get(pad_index, {}).get("source_id", f"cam_{pad_index}")
            print(f"[INFO] {cam_name} local_id 達上限 {LOCAL_ID_MAX}，下一個歸 1 重新計算")
            next_local_ids[pad_index] = 1
        else:
            next_local_ids[pad_index] += 1

    return local_id_maps[pad_index][global_id]


# ==========================================
# 5. 軌跡結算 (Trajectory Finalization)
# ==========================================

def _finalize_one(m_key, state, force=False):
    """
    結算單一車輛軌跡，把符合條件的 ROI 紀錄各 emit 一筆到 pending_records

    結算條件：
    1. 方向必須是 IN 或 OUT（NA 表示位移不足，視為抖動誤判 → 整筆丟掉）
    2. 對該軌跡的每個 ROI：命中數 >= min_roi_hits → emit 一筆紀錄
       （多 ROI 機制：一台車經過 N 個 ROI，會 emit N 筆紀錄）

    每筆紀錄欄位（與 INSERT 順序一致）：
        DeviceCode / CameraCode / TrackID / DetectClass / LocationName /
        MetricType / HitCount / VideoTime / CollectTime
        （LocationName 存的就是 ROI 名稱，欄位改名不影響 probe 判斷邏輯）

    參數：
        m_key (tuple): (pad_index, obj_id)
        state (dict): 該軌跡的狀態字典
        force (bool): 是否為強制結算（程式結束時用，影響 log 標籤）
    """
    pad_index, obj_id = m_key
    cfg = SOURCE_CONFIGS.get(pad_index, {})
    cam_name = cfg.get("source_id", f"cam_{pad_index}")
    min_hits = cfg.get("track_logic", {}).get("min_roi_hits", 2)

    # 步驟 1: 方向過濾（NA 整筆丟掉）
    if state.get("direction", "NA") == "NA":
        return

    # 步驟 2: 找出所有達門檻的 ROI（沒有就整筆丟掉）
    triggered_rois = {
        roi_name: hits
        for roi_name, hits in state.get("roi_hits", {}).items()
        if hits >= min_hits
    }
    if not triggered_rois:
        return

    # 步驟 3: 共用欄位計算
    local_id = get_local_id(pad_index, obj_id)
    device_code = cfg.get("device_code", "UNKNOWN")
    direction = state["direction"]

    # 車種投票：取票數最多的類別
    if state.get("class_votes"):
        best_class_id = state["class_votes"].most_common(1)[0][0]
        # 多權重：用該軌跡所屬組的 class_map（probe 存進 state），
        #    fallback 到全域 CLASS_MAP，確保 yolo/person 組寫入正確名稱而非車種名
        cmap = state.get("class_map") or CLASS_MAP
        cls_name = cmap.get(best_class_id, f"Class_{best_class_id}")
    else:
        cls_name = "Unknown"

    # VideoTime：軌跡最後出現幀號 → 影片內秒數
    vsec = state["last_frame_num"] / cfg.get("stream_fps", 30.0)
    time_axis = _format_video_time(vsec)

    # CollectTime：檔案模式 = start_time + vsec；即時串流 = 系統當下時間
    start_dt = cfg.get("start_time_dt")
    if start_dt is not None:
        event_dt = start_dt + timedelta(seconds=vsec)
        create_time_str = event_dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        create_time_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # 步驟 4: 對每個達門檻的 ROI 各 emit 一筆
    # roi_name 會寫進 DB 的 LocationName 欄位
    tag = "[結算-強制]" if force else " "

    for roi_name, hit_count in triggered_rois.items():
        # save_output_db=false → 只印 log，不累積、不寫 DB
        if not cfg.get("save_output_db", True):
            print(f"{tag}[{cam_name}] ID={local_id}, 分類={cls_name}, "
                  f"位置={roi_name}, 方向={direction}, 次數={hit_count}, "
                  f"時間軸={time_axis}, 時間點={create_time_str}  (DB 已停用)")
            continue

        pending_records[pad_index].append((
            device_code,
            cam_name,
            local_id,
            cls_name,
            roi_name,        # → LocationName
            direction,       # → MetricType
            hit_count,
            time_axis,
            create_time_str,
        ))

        print(f"{tag}[{cam_name}] ID={local_id}, 分類={cls_name}, "
              f"位置={roi_name}, 方向={direction}, 次數={hit_count}, "
              f"時間軸={time_axis}, 時間點={create_time_str}")


# ==========================================
# 6. DB 寫入 (DB Flush)
# ==========================================

def flush_pending_to_db(pad_index):
    """
    把 pending_records[pad_index] 批次寫入「合併」SQLite

    介面簽名維持不變（仍以 pad_index 呼叫），故 probes.py 不需改動；
    差別只在內部寫的是單一共用連線 _db_conn，而非各自的連線。

    使用單一 transaction (BEGIN/COMMIT) 提升寫入效能；
    失敗時 ROLLBACK，pending 保留在記憶體等下次重試。

    參數：
        pad_index (int): 哪一路 cam
    返回：
        int: 實際寫入筆數；無 pending 或無連線回傳 0
    """
    records = pending_records.get(pad_index, [])
    if not records:
        return 0

    # 沒有合併連線（全部 save_output_db=false）→ 清掉殘留避免 memory leak
    if _db_conn is None:
        records.clear()
        return 0

    with _db_lock:
        try:
            _db_conn.execute("BEGIN")
            _db_conn.executemany(
                "INSERT INTO detection_stats "
                "(DeviceCode, CameraCode, TrackID, DetectClass, LocationName, MetricType, HitCount, VideoTime, CollectTime) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                records
            )
            _db_conn.execute("COMMIT")
            n = len(records)
            records.clear()
            return n
        except sqlite3.Error as e:
            try:
                _db_conn.execute("ROLLBACK")
            except Exception:
                pass
            print(f"[ERROR] SQLite 寫入失敗 (pad_index={pad_index}): {e}")
            return 0


# ==========================================
# 7. 結束清理 (Shutdown Cleanup)
# ==========================================

def force_finalize_all():
    """
    程式結束前呼叫：強制結算所有殘留軌跡、flush 剩餘 pending、關閉合併 DB

    處理流程：
    1. 對所有 track_history 內殘留的軌跡呼叫 _finalize_one(force=True)
       （這些車是程式結束時還在畫面內、還沒消失到 cleanup_frames 的）
    2. 對每路 cam 強制 flush 一次，確保 pending 都進合併 DB
    3. 關閉合併 DB 連線（WAL checkpoint 也會跟著做）
    4. 清空 track_history 釋放記憶體
    """
    global _db_conn

    print("\n[INFO] 開始執行強制結算...")

    # 步驟 1: 殘留軌跡逐一結算
    for m_key, state in list(track_history.items()):
        _finalize_one(m_key, state, force=True)

    # 步驟 2: 強制 flush 所有 pending（各路 pending 都寫進同一個合併 DB）
    total = 0
    for pad_index, cfg in SOURCE_CONFIGS.items():
        n = flush_pending_to_db(pad_index)
        if n > 0:
            total += n
            print(f"[檔案儲存] {cfg.get('source_id')}：已強制寫入 {n} 筆到合併 DB")
    if total > 0:
        print(f"[檔案儲存] 合併 DB 共寫入 {total} 筆 → {MERGED_DB_PATH}")

    # 步驟 3: 關閉合併 DB 連線
    if _db_conn is not None:
        try:
            _db_conn.close()
        except Exception as e:
            print(f"[WARNING] 關閉合併 DB 連線失敗: {e}")
        _db_conn = None

    # 步驟 4: 釋放記憶體
    track_history.clear()
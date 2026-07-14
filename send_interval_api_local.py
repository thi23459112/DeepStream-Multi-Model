#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地端打包上傳服務 (send_interval_api_local)
============================================
功能說明：
- 從單一 SQLite 資料庫（detection_stats）讀取即時偵測明細資料
- 依指定時間區間（例如 1 分鐘）將明細彙總為統計值（筆數）
- 將彙總結果逐筆透過 REST API (PUT) 上傳至地端伺服器（免 Token）
- 支援狀態持久化（State File），確保斷線重連不遺失進度
- 分批清理過期資料，避免 DB 膨脹
- 啟動時自動輪替日誌檔案，防止無限成長

設計哲學：
- 冪等性：以 CollectTime 為基準，狀態記錄「最後成功上傳的時間戳」，
  下次查詢時只取大於該時間的資料，確保不重複
- 分批清理：每批刪除 1000 筆後立即 commit，並 sleep 50ms，避免鎖死資料庫寫入
- 原子寫入狀態檔：先寫暫存檔再取代，防止斷電導致檔案損毀
"""

import os
import sys
import time
import json
import sqlite3
import argparse
import logging
import fcntl
import requests
import urllib3
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from logging.handlers import RotatingFileHandler  # 用於生成 .live.log 並限制大小

# 關閉 SSL 驗證警告（因為使用自簽憑證或內部 API）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ==========================================
# 1. 全域設定區（所有可調參數集中管理）
# ==========================================

# ---- 地端 API 端點（沿用舊版 PutAPI 的地端位址；PUT、免 Token） ----
API_URL = "https://x235aiapi.thix180server.com:4004/AIDetect_detection_stats"

# ---- 資料庫與彙總設定 ----
DEFAULT_DB_PATH = "/home/nvidia/THI/DeepStream-Multi-Model/output_db/traffic_count.db"  # 明細 DB 預設路徑（可被 --db 覆寫）
DEFAULT_INTERVAL_MIN = 1       # 彙總區間分鐘數（通常為 1）
DEFAULT_LOOKBACK_MIN = 5       # 首次部署時往回取幾分鐘的資料（等同舊版視窗）

# ---- 連線與重試設定 ----
REQUEST_TIMEOUT = 10           # 單筆請求逾時秒數
MAX_RETRIES = 2                # 單筆失敗重試次數（4xx 錯誤不重試）
RETENTION_DAYS = 7             # 本地 DB 保留天數，超過此天數的明細將被刪除

# ---- 檔案名稱設定 ----
LOG_FILENAME = "send_interval_api_local.log"         # 日誌檔名（配合 crontab/systemd 的 >>）
STATE_FILENAME = "upload_state_interval_local.json"  # 狀態記錄檔（記錄各 DB 最後上傳時間）
LOCK_FILENAME = "send_interval_api_local.lock"       # 防止同時執行的鎖檔

# ---- 防護上限設定 ----
MAX_FETCH_LIMIT = 50000          # 單次 SQL 撈取上限（防止記憶體爆量）
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB（日誌檔案大小上限）
BACKUP_COUNT = 3                 # 保留的備份日誌數量

# ---- 時間格式常數 ----
_TIME_FMT = "%Y-%m-%d %H:%M:%S"  # CollectTime 在 DB 內的儲存格式


# ==========================================
# 2. 日誌系統（手動輪替 + stdout + .live.log）
# ==========================================

def manual_log_rotation():
    """
    程式啟動時手動檢查並輪替主日誌檔案
    由於 Crontab/systemd 使用 >> 重定向寫入日誌，無法使用 RotatingFileHandler
    因此在初始化時檢查主日誌大小，若超過 10MB 則依序更名為 .1, .2, .3，
    讓後續的 >> 從全新的空檔案開始寫入
    """
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(log_dir, LOG_FILENAME)

    if os.path.exists(log_path) and os.path.getsize(log_path) >= MAX_LOG_SIZE:
        # 刪除最舊的備份（若存在）
        oldest_backup = f"{log_path}.{BACKUP_COUNT}"
        if os.path.exists(oldest_backup):
            try:
                os.remove(oldest_backup)
            except OSError:
                pass

        # 將現有備份依序後移（.2 -> .3, .1 -> .2）
        for i in range(BACKUP_COUNT - 1, 0, -1):
            src = f"{log_path}.{i}"
            dst = f"{log_path}.{i+1}"
            if os.path.exists(src):
                try:
                    os.rename(src, dst)
                except OSError:
                    pass

        # 將當前日誌改名為 .1
        try:
            os.rename(log_path, f"{log_path}.1")
        except OSError:
            pass


# 立即執行一次輪替（在建立 logger 之前）
manual_log_rotation()

# 建立專屬 Logger，避免與根 Logger 衝突
logger = logging.getLogger("IntervalAPI_Local_Uploader")
logger.setLevel(logging.INFO)


def setup_logging():
    """
    設定日誌輸出格式與目的地：
    1. 輸出到 stdout，讓 Crontab/systemd 的 >> 負責寫入主 .log 檔案
    2. 使用 RotatingFileHandler 寫入 .live.log，供 Dashboard 的 tail -F 即時監看，
       並限制檔案大小（5MB，保留 1 個備份），防止無限成長
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # 清除既有的 Handler，避免重複
    if logger.hasHandlers():
        logger.handlers.clear()

    # Handler 1：輸出到標準輸出
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Handler 2：輸出到 .live.log（RotatingFileHandler）
    live_log_name = LOG_FILENAME.replace(".log", ".live.log")
    live_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), live_log_name)
    live_handler = RotatingFileHandler(
        live_log_path,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=1,             # 保留 1 個備份
        encoding='utf-8'
    )
    live_handler.setFormatter(formatter)
    logger.addHandler(live_handler)

    # 防止日誌向上傳遞給根 Logger
    logger.propagate = False


setup_logging()


# ==========================================
# 3. 狀態管理（State File 原子寫入）
# ==========================================

def load_state():
    """
    從 JSON 檔案載入上傳狀態
    狀態內容為 { db_filename: last_uploaded_time_str }
    若檔案不存在或讀取失敗，回傳空字典
    """
    state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), STATE_FILENAME)
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"讀取狀態檔失敗，將使用預設值: {e}")
    return {}


def save_state(state):
    """
    將狀態字典儲存至 JSON 檔案（原子寫入）
    先寫入暫存檔，成功後再取代原檔案，避免寫入過程斷電導致檔案損壞
    """
    state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), STATE_FILENAME)
    try:
        temp_path = state_path + ".tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4, ensure_ascii=False)
        os.replace(temp_path, state_path)
    except Exception as e:
        logger.error(f"儲存狀態檔失敗: {e}")


# ==========================================
# 4. 時間區間處理（向下取整）
# ==========================================

def floor_to_interval(collect_time_str, interval_min):
    """
    將 CollectTime 字串向下取整到指定的區間起點
    例如 interval_min=5，10:07 → 10:05
    若解析失敗，回傳原始字串
    """
    try:
        dt = datetime.strptime(collect_time_str, _TIME_FMT)
    except (ValueError, TypeError):
        return collect_time_str

    if interval_min <= 1:
        floored = dt.replace(second=0, microsecond=0)
    else:
        bucket_min = (dt.minute // interval_min) * interval_min
        floored = dt.replace(minute=bucket_min, second=0, microsecond=0)
    return floored.strftime(_TIME_FMT)


# ==========================================
# 5. 資料庫讀取與彙總
# ==========================================

def fetch_detail_rows(db_path, since_str):
    """
    以唯讀模式從 detection_stats 資料表讀取明細資料
    只取 CollectTime > since_str 的列，並限制筆數上限

    參數：
        db_path   : SQLite 資料庫檔案路徑
        since_str : 起始時間字串（格式 _TIME_FMT）

    回傳：
        (records, max_time)
        - records : list of tuple (DeviceCode, CameraCode, LocationName, MetricType, DetectClass, CollectTime)
        - max_time: 本次查詢最後一筆的 CollectTime（字串），若無資料則為 None
    """
    name = os.path.basename(db_path)
    records = []
    max_time = None
    conn = None

    if not os.path.exists(db_path):
        logger.error(f"找不到 DB：{db_path}")
        return [], None

    # 以唯讀模式連線，不影響 DeepStream 的 WAL 寫入
    db_uri = f"{Path(db_path).as_uri()}?mode=ro"
    try:
        conn = sqlite3.connect(db_uri, uri=True, timeout=10)
        cur = conn.cursor()

        # 嚴格大於 (>) 上次上傳時間，確保不重複
        query = """
            SELECT DeviceCode, CameraCode, LocationName, MetricType, DetectClass, CollectTime
            FROM detection_stats
            WHERE CollectTime > ?
            ORDER BY CollectTime ASC
            LIMIT ?
        """
        rows = cur.execute(query, (since_str, MAX_FETCH_LIMIT)).fetchall()

        for row in rows:
            records.append(row)
            max_time = row[5]   # CollectTime 是第 6 個欄位（索引 5）

        if records:
            logger.info(f"[{name}] 撈取 {len(records)} 筆明細 (起點: {since_str}, 終點: {max_time})")

    except sqlite3.Error as e:
        logger.error(f"[{name}] 讀取 detection_stats 失敗：{e}")
    finally:
        if conn:
            conn.close()

    return records, max_time


def aggregate(rows, interval_min):
    """
    將明細列依 (DeviceCode, CameraCode, LocationName, MetricType, DetectClass, 區間) 分組，
    計算每組的筆數作為 Value
    回傳排序後的 list[dict]，每個 dict 包含彙總結果
    """
    counter = defaultdict(int)
    for device, cam, loc, metric, cls, collect in rows:
        bucket_time = floor_to_interval(collect, interval_min)
        key = (device, cam, loc, metric, bucket_time, cls)
        counter[key] += 1

    result = []
    for (device, cam, loc, metric, bucket_time, cls), value in counter.items():
        result.append({
            "DeviceCode":   device,
            "CameraCode":   cam,
            "LocationName": loc,
            "MetricType":   metric,
            "DetectClass":  cls,
            "CollectTime":  bucket_time,
            "Value":        value,
        })

    # 排序：時間 → 相機 → 位置 → 方向 → 車種，方便核對
    result.sort(key=lambda r: (r["CollectTime"], r["CameraCode"],
                               r["LocationName"], r["MetricType"], r["DetectClass"]))
    return result


def cleanup_db(db_path, max_uploaded_time, retention_days):
    """
    刪除超過保留天數的舊明細資料（分批刪除，避免鎖死）
    每批刪除 1000 筆後立即 commit，並 sleep 50ms，釋放寫入鎖給 AI 進程
    """
    name = os.path.basename(db_path)
    conn = None
    try:
        dt_max = datetime.strptime(max_uploaded_time, _TIME_FMT)
        dt_cleanup = dt_max - timedelta(days=retention_days)
        cleanup_time_str = dt_cleanup.strftime(_TIME_FMT)

        db_uri = f"{Path(db_path).as_uri()}"
        conn = sqlite3.connect(db_uri, uri=True, timeout=30)
        c = conn.cursor()

        total_deleted = 0
        batch_size = 1000

        # SQLite 不支援 DELETE ... LIMIT，改用子查詢
        delete_query = """
            DELETE FROM detection_stats
            WHERE rowid IN (SELECT rowid FROM detection_stats WHERE CollectTime <= ? LIMIT ?)
        """

        while True:
            c.execute(delete_query, (cleanup_time_str, batch_size))
            deleted_count = c.rowcount
            total_deleted += deleted_count

            conn.commit()          # 立即提交，釋放鎖
            if deleted_count < batch_size:
                break
            time.sleep(0.05)       # 讓出 50ms，讓 AI 有機會寫入

        if total_deleted > 0:
            logger.info(f"[{name}] 成功分批清理 {total_deleted} 筆舊明細 (保留 {retention_days} 天)")

    except Exception as e:
        logger.error(f"[{name}] 清理失敗: {e}")
    finally:
        if conn:
            conn.close()


def purge_history_data(db_path, since_str):
    """
    【首次部署專用】刪除早於指定時間的歷史資料
    用於清空首次部署時不打算上傳的龐大舊資料
    """
    name = os.path.basename(db_path)
    conn = None
    try:
        db_uri = f"{Path(db_path).as_uri()}"
        conn = sqlite3.connect(db_uri, uri=True, timeout=30)
        c = conn.cursor()

        total_deleted = 0
        while True:
            c.execute(
                "DELETE FROM detection_stats WHERE rowid IN (SELECT rowid FROM detection_stats WHERE CollectTime <= ? LIMIT 2000)",
                (since_str,)
            )
            deleted_count = c.rowcount
            total_deleted += deleted_count
            conn.commit()
            if deleted_count < 2000:
                break
            time.sleep(0.05)

        if total_deleted > 0:
            logger.info(f"[{name}] 首次部署清理：丟棄 {total_deleted} 筆歷史明細")

    except Exception as e:
        logger.error(f"[{name}] 首次清理失敗: {e}")
    finally:
        if conn:
            conn.close()


# ==========================================
# 6. 上傳功能（逐筆 PUT，免 Token）
# ==========================================

def build_session():
    """建立 HTTP 會話，重用連線並關閉 SSL 驗證"""
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.verify = False
    session.headers.update({"Content-Type": "application/json"})
    return session


def upload_batch_local(session, summary, dry_run=False):
    """
    將彙總後的每一列以 PUT 送到地端 API（無需 Token）
    CreateTime = 區間時間（= CollectTime）；UploadTime = 逐筆送出當下時間

    參數：
        session  : requests.Session 物件
        summary  : list[dict]，由 aggregate() 產生的彙總資料
        dry_run  : bool，若為 True 只印出不實際發送

    回傳：
        (success_count, fail_count)
    """
    if not summary:
        logger.info("本次沒有可上傳的彙總資料")
        return 0, 0

    # --- Dry-run 模式 ---
    if dry_run:
        logger.info(f"[DRY-RUN] 本次共 {len(summary)} 筆，僅印出不實際上傳：")
        for r in summary:
            preview = {
                'DeviceCode':   r["DeviceCode"],
                'CameraCode':   r["CameraCode"],
                'LocationName': r["LocationName"],
                'MetricType':   r["MetricType"],
                'DetectClass':  r["DetectClass"],
                'CollectTime':  r["CollectTime"],
                'Value':        int(r["Value"]),
                'CreateTime':   r["CollectTime"],
                'UploadTime':   datetime.now().strftime(_TIME_FMT),
            }
            logger.info("  " + json.dumps(preview, ensure_ascii=False))
        return len(summary), 0

    success_count = 0
    fail_count = 0

    for r in summary:
        # 組裝單筆 payload
        data = {
            'DeviceCode':   r["DeviceCode"],
            'CameraCode':   r["CameraCode"],
            'LocationName': r["LocationName"],
            'MetricType':   r["MetricType"],
            'DetectClass':  r["DetectClass"],
            'CollectTime':  r["CollectTime"],
            'Value':        int(r["Value"]),
            'CreateTime':   r["CollectTime"],                    # 該筆所屬的 1 分鐘區間時間
            'UploadTime':   datetime.now().strftime(_TIME_FMT),  # 送出當下時間
        }

        # 單筆重試機制
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = session.put(
                    API_URL,
                    data=json.dumps(data),
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                success_count += 1
                break  # 成功則跳出重試迴圈

            except requests.exceptions.HTTPError as http_err:
                status = http_err.response.status_code if http_err.response is not None else None
                resp_body = http_err.response.text[:200] if http_err.response is not None else ""

                # 4xx 用戶端錯誤不重試
                if status is not None and 400 <= status < 500:
                    logger.error(f"上傳資料失敗 (HTTP {status}): {resp_body} | 資料: {json.dumps(data, ensure_ascii=False)}")
                    fail_count += 1
                    break

                # 5xx 或網路問題，若有重試次數則等待後重試
                if attempt < MAX_RETRIES:
                    time.sleep(0.5 * (attempt + 1))
                    continue

                logger.error(f"上傳資料最終失敗 (HTTP {status}): {resp_body} | 資料: {json.dumps(data, ensure_ascii=False)}")
                fail_count += 1

            except requests.exceptions.RequestException as req_err:
                if attempt < MAX_RETRIES:
                    time.sleep(0.5 * (attempt + 1))
                    continue

                logger.error(f"上傳資料最終失敗 (連線異常): {req_err} | 資料: {json.dumps(data, ensure_ascii=False)}")
                fail_count += 1

    logger.info(f"資料處理完成: 彙總 {len(summary)} 筆，成功上傳 {success_count} 筆，失敗 {fail_count} 筆")
    return success_count, fail_count


# ==========================================
# 7. 主程式與命令列介面
# ==========================================

def compute_default_since_str(lookback_min):
    """計算預設的查詢起點時間（當前時間往前推 lookback_min 分鐘，秒數歸零）"""
    dt = datetime.now() - timedelta(minutes=lookback_min)
    dt = dt.replace(second=0, microsecond=0)
    return dt.strftime(_TIME_FMT)


def main():
    """主流程：解析參數 → 加鎖 → 讀取狀態 → 讀取明細 → 彙總 → 上傳 → 清理與儲存狀態"""
    parser = argparse.ArgumentParser(description="區間彙總並透過地端 API 上傳（工業級邊緣部署版）")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="明細 DB 路徑")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_MIN,
                        help="彙總區間分鐘數（預設 1）")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_MIN,
                        help="首次部署時往回取幾分鐘的資料（預設 5，等同舊版視窗）")
    parser.add_argument("--dry-run", action="store_true", help="只印出、不實際上傳")
    args = parser.parse_args()

    # ---- 步驟 1：獲取檔案鎖，防止排程重疊 ----
    lock_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOCK_FILENAME)
    lock_file = open(lock_path, 'w')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        logger.warning("偵測到另一個執行個體正在運行，本次結束")
        lock_file.close()
        return

    try:
        if not os.path.exists(args.db):
            logger.error(f"找不到 DB：{args.db}")
            return

        # ---- 步驟 2：載入狀態，並處理首次部署 ----
        state = load_state()
        db_name = os.path.basename(args.db)
        is_first_run = db_name not in state

        if is_first_run:
            # 計算截止時間，清理歷史包袱，初始化狀態
            cutoff_dt = datetime.now() - timedelta(minutes=args.lookback)
            first_since_str = cutoff_dt.replace(second=0, microsecond=0).strftime(_TIME_FMT)
            logger.info(f"【首次部署】將清理 {first_since_str} 之前的歷史包袱明細...")
            purge_history_data(args.db, first_since_str)
            state[db_name] = first_since_str
            save_state(state)
            logger.info("首次部署清理完成，已初始化狀態檔")

        current_since = state.get(db_name, compute_default_since_str(args.lookback))

        logger.info(f"讀取明細 DB：{args.db}")
        logger.info(f"地端端點：{API_URL}")
        logger.info(f"彙總區間：{args.interval} 分鐘；起點: {current_since}")

        # ---- 步驟 3：讀取明細資料 ----
        rows, max_time = fetch_detail_rows(args.db, current_since)

        if not rows:
            logger.info("本次沒有新的明細資料需要彙總")
            return

        # ---- 步驟 4：區間彙總 ----
        summary = aggregate(rows, args.interval)
        logger.info(f"明細 {len(rows)} 筆 -> 彙總後共 {len(summary)} 列，準備上傳")

        # ---- 步驟 5：執行上傳 ----
        session = build_session()
        success_count, fail_count = upload_batch_local(session, summary, dry_run=args.dry_run)
        session.close()

        # ---- 步驟 6：狀態推進與清理 ----
        # 因逐筆 PUT，若有任何失敗，狀態不推進，下次排程會重傳（PUT 具有冪等性）
        if fail_count == 0 and not args.dry_run and max_time:
            state[db_name] = max_time
            cleanup_db(args.db, max_time, RETENTION_DAYS)
            save_state(state)
            logger.info(f"上傳完成：成功 {success_count} 筆，狀態已推進，舊明細已清理")
        elif fail_count == 0 and args.dry_run:
            logger.info(f"[DRY-RUN] 模擬上傳完成：共 {success_count} 筆")
        else:
            logger.error(f"上傳存在失敗紀錄 (失敗 {fail_count} 筆)，狀態不推進，等待下次排程重傳")

        logger.info("本次排程任務結束")

    finally:
        # 釋放檔案鎖
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


if __name__ == "__main__":
    try:
        main()
        logger.info("排程執行結束\n" + "-" * 40)
    except Exception as e:
        # exc_info=True 印出完整 Traceback，方便除錯
        logger.error(f"主程式執行時發生未預期的嚴重錯誤: {e}", exc_info=True)
    finally:
        sys.exit(0)

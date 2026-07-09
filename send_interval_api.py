#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
雲端打包上傳服務 (send_interval_api)
=======================================
功能說明：
- 從單一 SQLite 資料庫（detection_stats）讀取即時偵測明細資料
- 依指定時間區間（例如 1 分鐘）將明細彙總為統計值（筆數）
- 將彙總結果一次打包成 JSON 陣列，透過 REST API 上傳
- 支援狀態持久化（State File），確保斷線重連不遺失進度
- 內建 Token 管理（自動刷新 JWT）
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
from logging.handlers import RotatingFileHandler  # ⭐ 新增：用於生成 .live.log 並限制大小

# 關閉 SSL 驗證警告（因為使用自簽憑證或內部 API）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ==========================================
# 1. 全域設定區（所有可調參數集中管理）
# ==========================================

# ---- API 端點設定 ----
API_BASE_URL  = "https://pingits.thix180server.com"                   # API 伺服器根網址
API_URL       = f"{API_BASE_URL}/pingits/api/AIData/AiDetectRawData"  # 上傳彙總資料的端點（POST）
AUTH_URL      = f"{API_BASE_URL}/pingits/api/Auth/login"              # 認證端點（取得 JWT Token）

# ---- 認證憑證（請依環境調整） ----
ENTERPRISE_ID = "THI"          # 企業識別碼
USER_ID       = "AiAPI"        # 使用者帳號
PASSWORD      = "Msaj#aV6Lh"   # 使用者密碼（⚠️ 敏感資訊，建議改用環境變數）

# ---- 資料庫與彙總設定 ----
DEFAULT_DB_PATH = "/home/thi/THI/DeepStream-Multi-Model/output_db/traffic_count.db"  # 明細 DB 預設路徑（可被 --db 覆寫）
DEFAULT_INTERVAL_MIN = 1       # 彙總區間分鐘數（通常為 1）
DEFAULT_LOOKBACK_MIN = 5       # 首次部署時，往回取幾分鐘的資料（等同舊版視窗）

# ---- 連線與重試設定 ----
TOKEN_TTL = 120                # Token 有效時間（秒），提前刷新避免用到過期 Token
REQUEST_TIMEOUT = 30           # 單次批次請求逾時秒數（整批資料較大，設長一點）
MAX_RETRIES = 2                # 上傳失敗時的最大重試次數
RETENTION_DAYS = 7             # 本地 DB 保留天數，超過此天數的明細將被刪除

# ---- 檔案名稱設定 ----
LOG_FILENAME = "send_interval_api.log"         # 日誌檔名（配合 crontab 的 >>）
STATE_FILENAME = "upload_state_interval.json"  # 狀態記錄檔（記錄各 DB 最後上傳時間）
LOCK_FILENAME = "send_interval_api.lock"       # 防止同時執行的鎖檔

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
    程式啟動時手動檢查並輪替日誌檔案
    
    原因：由於 Crontab 使用 >> 重定向寫入日誌，我們無法使用 RotatingFileHandler
    因此在程式初始化時，檢查日誌大小若超過 10MB，就將舊日誌依序改名為 .1, .2, .3，
    讓 Crontab 的 >> 從一個全新的空白檔案開始寫入
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

        # 將當前日誌改名為 .1，讓後續 >> 寫入新檔
        try:
            os.rename(log_path, f"{log_path}.1")
        except OSError:
            pass


# 立即執行一次輪替（在建立 logger 之前）
manual_log_rotation()

# 建立專屬 Logger，避免與根 Logger 衝突
logger = logging.getLogger("IntervalAPI_Uploader")
logger.setLevel(logging.INFO)


def setup_logging():
    """
    設定日誌輸出格式與目的地
    1. 輸出到 stdout，讓 Crontab/systemd 的 >> 負責寫入主 .log 檔案
    2. 使用 RotatingFileHandler 寫入 .live.log，供 Dashboard 的 tail -F 即時監看，
       並限制檔案大小（5MB，保留 1 個備份），防止無限成長撐爆硬碟
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. stdout Handler (給 Crontab/systemd 寫入主 .log)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 2. .live.log Handler (給 Dashboard 即時監看)
    # 自動將檔名從 send_interval_api.log 替換為 send_interval_api.live.log
    live_log_name = LOG_FILENAME.replace(".log", ".live.log")
    live_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), live_log_name)
    
    # 限制 5MB，保留 1 個備份，防止無限變大
    live_handler = RotatingFileHandler(
        live_log_path, 
        maxBytes=5*1024*1024, 
        backupCount=1, 
        encoding='utf-8'
    )
    live_handler.setFormatter(formatter)
    logger.addHandler(live_handler)

    # 切斷向上傳遞，防止日誌被外部框架再印一次
    logger.propagate = False


setup_logging()


# ==========================================
# 3. Token 管理（使用 monotonic 時鐘）
# ==========================================

class TokenManager:
    """
    API 認證 Token 管理器，負責獲取與自動刷新 JWT Token

    最佳化說明：
    使用 time.monotonic()（單調時鐘）計算經過秒數，避免系統時間跳動（如 NTP 校時）
    導致誤判 Token 過期
    """

    def __init__(self):
        self._token = None                # 目前的 Token 字串
        self._monotonic_time = None       # 取得 Token 時的 monotonic 時間點

    def get_token(self):
        """
        取得目前有效的 Token
        若 Token 不存在或已超過 TTL，則重新獲取
        """
        now = time.monotonic()
        elapsed = (now - self._monotonic_time) if self._monotonic_time else TOKEN_TTL + 1

        if self._token is None or elapsed >= TOKEN_TTL:
            logger.info(f"Token 刷新觸發 (已使用 {elapsed:.0f} 秒)")
            self._token = self._fetch_token()
            self._monotonic_time = now if self._token else None

        return self._token

    def _fetch_token(self):
        """
        向認證伺服器請求 AccessToken
        成功回傳 Token 字串，失敗回傳 None
        """
        payload = {
            "EnterpriseId": ENTERPRISE_ID,
            "UserId": USER_ID,
            "Password": PASSWORD,
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                AUTH_URL,
                data=json.dumps(payload),
                headers=headers,
                verify=False,
                timeout=10,
            )
            response.raise_for_status()
            token_data = response.json()

            if not token_data.get("isPasswordValid", False):
                logger.error("登入失敗：帳號或密碼錯誤")
                return None

            token = token_data.get("AccessToken")
            if not token:
                logger.error(f"登入成功但找不到 AccessToken，回傳內容: {token_data}")
                return None

            return token

        except requests.exceptions.RequestException as e:
            logger.error(f"取得 Token 失敗: {e}")
            return None


# 建立全域 Token 管理器實例
token_manager = TokenManager()


# ==========================================
# 4. 狀態管理（State File 原子寫入）
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
# 5. 時間區間處理（向下取整）
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
# 6. 資料庫讀取與彙總
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

    try:
        # 以唯讀模式連線（不影響 DeepStream 的 WAL 寫入）
        db_uri = f"{Path(db_path).as_uri()}?mode=ro"
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
        logger.error(f"[{name}] 讀取 detection_stats 失敗: {e}")
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
# 7. 上傳功能（整批打包 + 重試）
# ==========================================

def build_session():
    """建立 HTTP 會話，重用連線並關閉 SSL 驗證"""
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.verify = False
    return session


def upload_batch(session, summary, dry_run=False):
    """
    將彙總結果打包成 JSON 陣列，一次 POST 至 API 端點
    若 dry_run 為 True，只印出資料不實際送出

    回傳：
        (ok, count)
        ok    : bool，上傳是否成功
        count : int，本次上傳的彙總筆數
    """
    if not summary:
        logger.info("本次沒有可上傳的彙總資料")
        return True, 0

    upload_time = datetime.now().strftime(_TIME_FMT)

    # 組裝 payload（每個項目加入 CreateTime 與 UploadTime）
    payload = []
    for r in summary:
        payload.append({
            'DeviceCode':   r["DeviceCode"],
            'CameraCode':   r["CameraCode"],
            'LocationName': r["LocationName"],
            'MetricType':   r["MetricType"],
            'DetectClass':  r["DetectClass"],
            'CollectTime':  r["CollectTime"],
            'Value':        int(r["Value"]),
            'CreateTime':   r["CollectTime"],
            'UploadTime':   upload_time,
        })

    # Dry-run 模式：只印出，不實際呼叫 API
    if dry_run:
        logger.info(f"[DRY-RUN] 本次共 {len(payload)} 筆彙總資料（僅印出不實際上傳）")
        return True, len(payload)

    json_data = json.dumps(payload, ensure_ascii=False)
    total = len(payload)

    for attempt in range(MAX_RETRIES + 1):
        token = token_manager.get_token()
        if not token:
            return False, 0

        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {token}",
        }

        try:
            logger.info(f"準備【一次打包】上傳彙總資料，共 {total} 筆")
            response = session.post(API_URL, data=json_data, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            logger.info(f"整批彙總上傳成功，共 {total} 筆")
            return True, total

        except requests.exceptions.HTTPError as http_err:
            status = http_err.response.status_code if http_err.response is not None else None
            resp_text = http_err.response.text[:500] if http_err.response is not None else "無回應"

            # 4xx 用戶端錯誤不重試
            if status is not None and 400 <= status < 500:
                logger.error(f"整批上傳失敗 (HTTP {status}): {resp_text}")
                return False, 0

            # 5xx 或連線問題，若有重試次數則等待後重試
            if attempt < MAX_RETRIES:
                wait_time = 1.0 * (attempt + 1)
                logger.warning(f"上傳暫時失敗 (HTTP {status})，{wait_time} 秒後重試伺服器回應: {resp_text}")
                time.sleep(wait_time)
                continue

            logger.error(f"整批上傳最終失敗 (HTTP {status})伺服器回應: {resp_text}")
            return False, 0

        except requests.exceptions.RequestException as req_err:
            if attempt < MAX_RETRIES:
                wait_time = 1.0 * (attempt + 1)
                logger.warning(f"連線異常，{wait_time} 秒後重試: {req_err}")
                time.sleep(wait_time)
                continue
            logger.error(f"整批上傳最終失敗: {req_err}")
            return False, 0

    return False, 0


# ==========================================
# 8. 主程式與命令列介面
# ==========================================

def compute_default_since_str(lookback_min):
    """計算預設的查詢起點時間（當前時間往前推 lookback_min 分鐘，秒數歸零）"""
    dt = datetime.now() - timedelta(minutes=lookback_min)
    dt = dt.replace(second=0, microsecond=0)
    return dt.strftime(_TIME_FMT)


def main():
    """主流程：解析參數 → 加鎖 → 讀取狀態 → 讀取明細 → 彙總 → 上傳 → 清理與儲存狀態"""
    parser = argparse.ArgumentParser(description="區間彙總並透過 API 上傳（工業級邊緣部署版）")
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
            # 首次部署：計算截止時間，清理歷史包袱，初始化狀態
            cutoff_dt = datetime.now() - timedelta(minutes=args.lookback)
            first_since_str = cutoff_dt.replace(second=0, microsecond=0).strftime(_TIME_FMT)
            logger.info(f"【首次部署】將清理 {first_since_str} 之前的歷史包袱明細...")
            purge_history_data(args.db, first_since_str)
            state[db_name] = first_since_str
            save_state(state)
            logger.info("首次部署清理完成，已初始化狀態檔")

        current_since = state.get(db_name, compute_default_since_str(args.lookback))
        logger.info(f"讀取明細 DB：{args.db} (起點: {current_since})")
        logger.info(f"彙總區間：{args.interval} 分鐘")

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
        ok, count = upload_batch(session, summary, dry_run=args.dry_run)
        session.close()

        # ---- 步驟 6：狀態推進與清理 ----
        if ok and not args.dry_run and max_time:
            state[db_name] = max_time
            cleanup_db(args.db, max_time, RETENTION_DAYS)
            save_state(state)
            logger.info(f"上傳完成：成功彙總並上傳 {count} 筆，狀態已推進，舊明細已清理")
        elif ok and args.dry_run:
            logger.info(f"[DRY-RUN] 模擬上傳完成：共 {count} 筆")
        else:
            logger.error(f"上傳失敗，狀態不推進，等待下次排程重試")

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
        # exc_info=True 會印出完整的 Traceback，方便除錯
        logger.error(f"主程式執行時發生未預期的嚴重錯誤: {e}", exc_info=True)
    finally:
        sys.exit(0)

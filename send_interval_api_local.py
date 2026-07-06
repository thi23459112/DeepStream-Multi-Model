#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
send_interval_api_local.py（地端版）
------------------------------------
把 DeepStream 產出的明細 DB（detection_stats，每台車一列）按「分鐘」彙總成 Value，
直接透過「地端 API」上傳。

與雲端版（send_interval_api.py）的差異：
    - 端點：https://x235aiapi.thix180server.com:4004/AIDetect_detection_stats
    - 不需要 Token、沒有 Bearer 授權，headers 只有 Content-Type
    - 方法：requests.PUT（不是 POST）
    - payload 欄位與雲端相同（含 UploadTime）；CreateTime = 區間時間（= CollectTime），UploadTime = 送出當下時間

彙總邏輯：沿用 verify_interval.py
    1. 每一列的 CollectTime 向下取整到「分鐘」當區間
    2. 依 (DeviceCode, CameraCode, LocationName, MetricType, DetectClass, 分鐘) 分組
    3. 每組的「明細列數」= Value（一台車一列，故列數 = 台數）

DB schema（logic/state_db.py，本程式以「唯讀 mode=ro」開啟，不影響 DeepStream 寫入）：
    id, DeviceCode, CameraCode, TrackID, DetectClass, LocationName,
    MetricType, HitCount, VideoTime, CollectTime

用法：
    python send_interval_api_local.py                  # 上傳「最近 5 分鐘」的彙總
    python send_interval_api_local.py --lookback 1     # 只上傳最近 1 分鐘
    python send_interval_api_local.py --db /path/to/traffic_count.db
    python send_interval_api_local.py --dry-run        # 只印出、不實際上傳（測試用）
"""

import os
import sys
import time
import json
import sqlite3
import argparse
import logging
import requests
import urllib3
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ==========================================
# 1. 設定區
# ==========================================

# ---- 地端 API 端點（沿用舊版 PutAPI 的地端位址；PUT、免 Token）----
API_URL = "https://x235aiapi.thix180server.com:4004/AIDetect_detection_stats"

# ---- DB / 彙總設定 ----
# 明細 DB 預設路徑（可用 --db 覆寫）
DEFAULT_DB_PATH = "/home/thi/THI/DeepStream-Multi-Model/output_db/traffic_count.db"

# 彙總區間分鐘數（採計單位，通常 1 分鐘）
DEFAULT_INTERVAL_MIN = 1

# 上傳視窗：往回取幾分鐘的資料（等同舊版 PutAPI 的「最近 5 分鐘」）
DEFAULT_LOOKBACK_MIN = 5

# CollectTime 在 DB 內的字串格式
_TIME_FMT = "%Y-%m-%d %H:%M:%S"


# ==========================================
# 2. Logging（由新到舊寫入，沿用舊版 PutAPI）
# ==========================================

class PrependFileHandler(logging.Handler):
    """自訂 Handler：新的 log 寫在檔案最前面，並限制最大行數。"""
    def __init__(self, filename, max_lines=1000, encoding='utf-8'):
        super().__init__()
        self.filename = filename
        self.encoding = encoding
        self.max_lines = max_lines
        self.buffer = []

    def emit(self, record):
        self.buffer.append(self.format(record) + '\n')

    def close(self):
        if self.buffer:
            old_lines = []
            if os.path.exists(self.filename):
                try:
                    with open(self.filename, 'r', encoding=self.encoding) as f:
                        old_lines = f.readlines()
                except Exception:
                    pass

            all_lines = self.buffer + old_lines
            all_lines = all_lines[:self.max_lines]

            try:
                with open(self.filename, 'w', encoding=self.encoding) as f:
                    f.writelines(all_lines)
            except Exception as e:
                print(f"無法儲存 Log 到檔案 {self.filename}，請檢查權限: {e}", file=sys.stderr)
        super().close()


_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "send_interval_api_local.log")
_LIVE_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "send_interval_api_local.live.log")

_live_handler = logging.FileHandler(_LIVE_LOG_PATH, mode="a", encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        PrependFileHandler(_LOG_PATH),
        _live_handler,
        logging.StreamHandler(sys.stdout),
    ]
)


# ==========================================
# 3. 時間 → 區間（沿用 verify_interval）
# ==========================================

def floor_to_interval(collect_time_str, interval_min):
    """把 CollectTime 向下取整到區間起點，回傳整分時間戳字串；解析失敗回傳原字串。"""
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
# 4. 讀 DB（唯讀 mode=ro，不影響 DeepStream 寫入）
# ==========================================

def read_detail_rows(db_path, cutoff_str):
    """
    以唯讀模式讀 detection_stats 的明細列，只取 CollectTime >= cutoff_str 的資料。

    mode=ro（不加 immutable）：可與 DeepStream 的 WAL 寫入並存，避免讀到過期快照。

    回傳：list[tuple] → (DeviceCode, CameraCode, LocationName, MetricType, DetectClass, CollectTime)
    """
    if not os.path.exists(db_path):
        logging.error(f"找不到 DB：{db_path}")
        return []

    db_uri = f"{Path(db_path).as_uri()}?mode=ro"
    try:
        conn = sqlite3.connect(db_uri, uri=True, timeout=5)
    except sqlite3.Error as e:
        logging.error(f"開啟 DB 失敗：{e}")
        return []

    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT DeviceCode, CameraCode, LocationName, MetricType, DetectClass, CollectTime "
            "FROM detection_stats WHERE CollectTime >= ?",
            (cutoff_str,),
        )
        return cur.fetchall()
    except sqlite3.Error as e:
        logging.error(f"讀取 detection_stats 失敗：{e}")
        return []
    finally:
        conn.close()


# ==========================================
# 5. 彙總（沿用 verify_interval）
# ==========================================

def aggregate(rows, interval_min):
    """
    依 (DeviceCode, CameraCode, LocationName, MetricType, DetectClass, 區間) 彙總，
    每組明細列數即為 Value。回傳已排序的 list[dict]。
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

    # 排序：時間 → 相機 → 位置 → 方向 → 車種，方便肉眼核對 log
    result.sort(key=lambda r: (r["CollectTime"], r["CameraCode"],
                               r["LocationName"], r["MetricType"], r["DetectClass"]))
    return result


# ==========================================
# 6. 上傳地端 API（彙總結果逐筆 PUT，免 Token）
# ==========================================

def send_to_api(summary, dry_run=False):
    """
    把彙總後的每一列以 PUT 送到地端 API。
    地端 API 不需 Token；headers 只有 Content-Type。
    CreateTime = 區間時間（= CollectTime）；UploadTime = 逐筆送出當下時間。單筆失敗只記錄 log，不中斷。
    """
    if not summary:
        logging.info("本次沒有可上傳的彙總資料。")
        return

    # --- Dry-run：只印出要送什麼，不實際打 API ---
    if dry_run:
        logging.info(f"[DRY-RUN] 本次共 {len(summary)} 筆，僅印出不實際上傳：")
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
            logging.info("  " + json.dumps(preview, ensure_ascii=False))
        return

    headers = {"Content-Type": "application/json"}

    success_count = 0
    for r in summary:
        data = {
            'DeviceCode':   r["DeviceCode"],
            'CameraCode':   r["CameraCode"],
            'LocationName': r["LocationName"],
            'MetricType':   r["MetricType"],
            'DetectClass':  r["DetectClass"],
            'CollectTime':  r["CollectTime"],
            'Value':        int(r["Value"]),
            'CreateTime':   r["CollectTime"],                    # 該筆所屬的 1 分鐘區間時間
            'UploadTime':   datetime.now().strftime(_TIME_FMT),  # 逐筆送出 API 的當下時間
        }

        try:
            response = requests.put(
                API_URL,
                data=json.dumps(data),
                headers=headers,
                verify=False,
                timeout=10,
            )
            response.raise_for_status()
            success_count += 1
        except requests.exceptions.RequestException as req_err:
            resp_body = ""
            try:
                if req_err.response is not None:
                    resp_body = req_err.response.text
            except Exception:
                pass
            logging.error(
                f"上傳資料失敗: {req_err}\n"
                f"  資料內容  : {json.dumps(data, ensure_ascii=False)}\n"
                f"  伺服器回應: {resp_body}"
            )

    logging.info(f"資料處理完成: 彙總 {len(summary)} 筆，成功上傳 {success_count} 筆。")


# ==========================================
# 7. 主流程
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="區間彙總並透過地端 API 上傳")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="明細 DB 路徑")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_MIN,
                        help="彙總區間分鐘數（預設 1）")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_MIN,
                        help="往回取幾分鐘的資料上傳（預設 5，等同舊版視窗）")
    parser.add_argument("--dry-run", action="store_true", help="只印出、不實際上傳")
    args = parser.parse_args()

    # 上傳視窗起點：now - lookback，向下取整到分鐘（沿用舊版 DTL5r 邏輯）
    cutoff_dt = datetime.now() - timedelta(minutes=args.lookback)
    cutoff_str = cutoff_dt.replace(second=0, microsecond=0).strftime(_TIME_FMT)

    logging.info(f"讀取明細 DB：{args.db}")
    logging.info(f"地端端點：{API_URL}")
    logging.info(f"彙總區間：{args.interval} 分鐘；上傳視窗：最近 {args.lookback} 分鐘"
                 f"（CollectTime >= '{cutoff_str}'）")

    rows = read_detail_rows(args.db, cutoff_str)
    logging.info(f"讀到明細 {len(rows)} 列")

    summary = aggregate(rows, args.interval)
    logging.info(f"彙總後共 {len(summary)} 列，準備上傳")

    send_to_api(summary, dry_run=args.dry_run)


if __name__ == "__main__":
    try:
        main()
        logging.info("排程執行結束。\n" + "-" * 40)
    except Exception as e:
        logging.error(f"主程式執行時發生錯誤: {e}")
    finally:
        time.sleep(1)
        sys.exit(0)
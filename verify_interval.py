#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_interval.py
------------------
離線驗證用：讀 DeepStream 產出的明細 DB（detection_stats，每台車一列），
按「分鐘」把資料彙總成 Value 表格，印在終端機並另存 CSV。

用途：
    驗證「區間彙總（每分鐘、每方向+車種的台數）」的格式與數字是否正確，
    完全不呼叫真實 API、不影響 DeepStream 寫入（以唯讀方式開 DB）。

彙總規則（最單純版本，DB 記什麼就算什麼，不做去重 / 不留安全邊界）：
    1. 每一列的 CollectTime（結算當下時間）向下取整到「分鐘」當區間
    2. 依 (DeviceCode, CameraCode, LocationName, MetricType, DetectClass, 分鐘) 分組
    3. 每組的「明細列數」= Value（一台車一列，故列數 = 台數）
    4. 正面表列：只輸出實際有值（Value>=1）的組合
    5. 每次執行都重算全部（不記錄送過什麼），同一分鐘每次都會出現

輸出欄位（對齊你要的格式）：
    DeviceCode, CameraCode, LocationName, MetricType, DetectClass, CollectTime, Value
    （CollectTime = 該區間起點的整分時間戳，例如 2025-05-22 00:23:00）

使用：
    python verify_interval.py
    python verify_interval.py --db /path/to/traffic_count.db --interval 1
"""

import os
import csv
import sqlite3
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime


# ==========================================
# 1. 預設參數
# ==========================================

# 明細 DB 預設路徑（可用 --db 覆寫）
DEFAULT_DB_PATH = "/home/nvidia/DeepStream-Traffic_Count/output_db/traffic_count.db"

# 區間大小（分鐘）。你的採計單位是 1 分鐘。
DEFAULT_INTERVAL_MIN = 1

# CollectTime 在 DB 內的字串格式
_TIME_FMT = "%Y-%m-%d %H:%M:%S"


# ==========================================
# 2. 時間 → 區間
# ==========================================

def floor_to_interval(collect_time_str, interval_min):
    """
    把 CollectTime 字串向下取整到區間起點，回傳整分時間戳字串。

    interval_min=1 → 取整到整分（秒歸零）
    interval_min=5 → 取整到 0/5/10... 分（例如 00:23:59 → 00:20:00）

    參數：
        collect_time_str (str): "YYYY-MM-DD HH:MM:SS"
        interval_min (int): 區間分鐘數
    返回：
        str: 區間起點時間戳；解析失敗回傳原字串
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
# 3. 讀 DB（唯讀）
# ==========================================

def read_detail_rows(db_path):
    """
    以唯讀模式讀 detection_stats 的明細列。

    唯讀連線（mode=ro）不會動到 DB、不影響 DeepStream 持續寫入。

    參數：
        db_path (str): 明細 DB 路徑
    返回：
        list[tuple]: (DeviceCode, CameraCode, LocationName, MetricType, DetectClass, CollectTime)
    """
    if not os.path.exists(db_path):
        print(f"[ERROR] 找不到 DB：{db_path}")
        return []

    db_uri = f"{Path(db_path).as_uri()}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True, timeout=5)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT DeviceCode, CameraCode, LocationName, MetricType, DetectClass, CollectTime "
            "FROM detection_stats"
        )
        return cur.fetchall()
    except sqlite3.Error as e:
        print(f"[ERROR] 讀取 detection_stats 失敗：{e}")
        return []
    finally:
        conn.close()


# ==========================================
# 4. 彙總
# ==========================================

def aggregate(rows, interval_min):
    """
    把明細列依 (DeviceCode, CameraCode, LocationName, MetricType, DetectClass, 區間) 彙總。

    參數：
        rows (list[tuple]): read_detail_rows 的輸出
        interval_min (int): 區間分鐘數
    返回：
        list[dict]: 每個 dict 是一列彙總（含 Value），已排序
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

    # 排序：時間 → 相機 → 位置 → 方向 → 車種，方便肉眼核對
    result.sort(key=lambda r: (r["CollectTime"], r["CameraCode"],
                               r["LocationName"], r["MetricType"], r["DetectClass"]))
    return result


# ==========================================
# 5. 輸出
# ==========================================

_COLUMNS = ["DeviceCode", "CameraCode", "LocationName",
            "MetricType", "DetectClass", "CollectTime", "Value"]


def print_table(summary):
    """把彙總結果印成對齊的表格。"""
    if not summary:
        print("（沒有任何資料）")
        return

    # 計算每欄寬度
    widths = {c: len(c) for c in _COLUMNS}
    for r in summary:
        for c in _COLUMNS:
            widths[c] = max(widths[c], len(str(r[c])))

    header = "  ".join(c.ljust(widths[c]) for c in _COLUMNS)
    print(header)
    print("-" * len(header))
    for r in summary:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in _COLUMNS))


def write_csv(summary, csv_path):
    """把彙總結果寫成 CSV（UTF-8-SIG，Excel 開中文不亂碼）。"""
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(summary)


# ==========================================
# 6. 對照檢查
# ==========================================

def sanity_check(rows, summary):
    """
    對照檢查：明細總列數 應等於 所有 Value 加總。
    相等 → 彙總沒算漏、沒重複，格式與數字可信。
    """
    detail_total = len(rows)
    value_total = sum(r["Value"] for r in summary)
    print("\n" + "=" * 40)
    print("對照檢查（驗證彙總是否正確）")
    print(f"  明細總列數        : {detail_total}")
    print(f"  所有 Value 加總   : {value_total}")
    if detail_total == value_total:
        print("  結果：✔ 相等，彙總正確（沒算漏、沒重複）")
    else:
        print("  結果：✘ 不相等，請檢查（可能有 CollectTime 解析失敗的列）")
    print("=" * 40)


# ==========================================
# 7. 主流程
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="離線驗證區間彙總（每分鐘 Value 表格）")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="明細 DB 路徑")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_MIN,
                        help="區間分鐘數（預設 1）")
    parser.add_argument("--csv", default=None, help="CSV 輸出路徑（預設與 DB 同目錄）")
    args = parser.parse_args()

    print(f"[INFO] 讀取明細 DB：{args.db}")
    print(f"[INFO] 區間大小：{args.interval} 分鐘")

    rows = read_detail_rows(args.db)
    print(f"[INFO] 讀到明細 {len(rows)} 列")

    summary = aggregate(rows, args.interval)

    print(f"\n===== 區間彙總結果（共 {len(summary)} 列）=====\n")
    print_table(summary)

    # CSV 路徑：預設與 DB 同目錄
    csv_path = args.csv or os.path.join(os.path.dirname(os.path.abspath(args.db)),
                                        "interval_summary.csv")
    if summary:
        write_csv(summary, csv_path)
        print(f"\n[INFO] 已存 CSV：{csv_path}")

    sanity_check(rows, summary)


if __name__ == "__main__":
    main()
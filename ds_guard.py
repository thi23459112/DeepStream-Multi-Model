#!/usr/bin/env python3
"""
邊緣機守護腳本：deepstream.service 卡住或記憶體出問題時，自動重啟服務或重開機

車流計數與 TMC 兩個專案共用同一份，內容相同。由 root 定時執行（例如每 5 分鐘一次）：

    必須以 root 執行：要能 systemctl restart / reboot，並讀取 NvMap 的 debugfs。
    輸出必須導向本腳本旁的 ds_guard.log：每日重開次數的上限靠該檔的 [REBOOT] 紀錄統計。
    只用 Python 標準函式庫，系統內建的 python3 即可，不需要 conda。

每輪依序檢查，命中一項就處理並結束本輪：

    1. 記憶體配置失敗（log 出現 RE_FAILURE）        → 重開機
       Jetson 的 NVMM 配置不到時，重啟服務也救不回（10/27 事件已證實），只能重開機。
       受 MIN_SYS_UPTIME_SEC 與 MAX_REBOOTS_PER_DAY 限制，避免無限重開。
    2. 某路攝影機連續重連 MAX_REBUILDS 次仍沒恢復  → 重啟服務
    3. 系統可用記憶體低於 MEM_AVAIL_MIN_RATIO      → 重啟服務
       補「重連過程中記憶體逐漸被吃光、但還沒到配置失敗」的情況。

服務狀態為 inactive（以 status.py 或 systemctl stop 手動停止）時完全不處理。
"""

import re
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta


# =============================================================================
# 常數
# =============================================================================

SERVICE = "deepstream.service"                                  # 與 status.py 相同的服務名稱

# 條件 2：連續重連幾次仍未恢復就重啟服務。
# 無畫面時每 20 ~ 25 秒重連一次；連線被拒時間隔 5、10、15 … 秒，最多 60 秒。
# 50 次約 20 ~ 45 分鐘；調低會更快重啟，但攝影機本身斷線時只是白白重啟
MAX_REBUILDS = 50

# 條件 3：可用記憶體低於總量的比例（0 ~ 1）。Jetson 的 CPU 與 GPU 共用同一塊 RAM，
# 低於 5% 時 NVMM 配置隨時可能失敗；調高會更早重啟，但正常的記憶體波動也可能觸發
MEM_AVAIL_MIN_RATIO = 0.05
MIN_UPTIME_SEC = 600                                            # 服務啟動未滿 10 分鐘不判斷條件 3，避免啟動期間記憶體尚未穩定就重啟

# 讀 log 的範圍
LOG_WINDOW = "-15min"                                           # 最近的 log：重連間隔最多 60 秒，15 分鐘內一定有紀錄；不讀整段 journal 以免越跑越慢
HEAD_LINES = 2000                                               # 本次啟動的前幾行：配置失敗多半發生在啟動當下，15 分鐘後就離開 LOG_WINDOW。
                                       # 啟動訊息約 50 行，之後是計數 log，2000 行足夠且讀取快

# 條件 1 的重開機限制
MIN_SYS_UPTIME_SEC = 1800                                       # 開機未滿 30 分鐘不重開，避免開機初始化期間誤判而連續重開
MAX_REBOOTS_PER_DAY = 3                                         # 24 小時內最多自動重開幾次；硬體或設定問題重開也救不回，避免無限重開

GUARD_LOG = Path(__file__).with_name("ds_guard.log")            # crontab 的輸出導向；count_recent_reboots() 從這裡數 [REBOOT]

# 重開機前的記憶體快照
NVMAP_CLIENTS = Path("/sys/kernel/debug/nvmap/iovmm/clients")   # Jetson 各行程的 NVMM 用量；debugfs 需 root
SNAPSHOT_MAX_LINES = 40                                         # 快照每一項最多寫幾行，避免異常時把 ds_guard.log 寫爆

# 對應 logic/sources.py 的 log（兩個專案格式相同）：
#   [WARNING] [cam06] 已 21 秒沒有畫面，重新連線（第 12 次）...
#   [INFO] [cam06] 已恢復畫面（先前連續重連 12 次）
RE_RECONNECT = re.compile(r"\[WARNING\] \[(.+?)\] .*重新連線（第 (\d+) 次）")
RE_RECOVER = re.compile(r"\[INFO\] \[(.+?)\] 已恢復畫面")

# 記憶體配置失敗的特徵（10/27 事件實際出現的訊息），出現任一項代表 NVMM 已配置不到
RE_FAILURE = re.compile(
    r"NvMapMemAllocInternalTagged: \d+ error 12"
    r"|failed to activate bufferpool"
    r"|Unable to set device in gst_nvstreammux_change_state"
)

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"                               # 每行 log 開頭的時間格式；count_recent_reboots() 依此解析


# =============================================================================
# 讀取系統狀態
# =============================================================================

def log(msg):
    """印出帶時間的訊息（crontab 導向 GUARD_LOG）。"""
    print(f"{datetime.now().strftime(TIME_FORMAT)} {msg}", flush=True)


def systemctl_show(prop):
    """
    讀服務的一個 systemd 屬性，例如 "ActiveState"、"MainPID"、"InvocationID"。

    回傳：
        屬性值；讀取失敗時為空字串
    """
    r = subprocess.run(["systemctl", "show", SERVICE, "-p", prop, "--value"],
                       capture_output=True, text=True, check=False)
    return r.stdout.strip()


def get_uptime_sec(pid):
    """服務主行程已執行幾秒；取不到時回傳 None。"""
    r = subprocess.run(["ps", "-o", "etimes=", "-p", pid],
                       capture_output=True, text=True, check=False)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return None


def get_system_uptime_sec():
    """開機至今幾秒；讀取失敗時回傳 None（呼叫端視為「不可重開」）。"""
    try:
        with open("/proc/uptime", encoding="ascii") as fh:
            return float(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def get_mem_available_ratio():
    """
    系統可用記憶體占總量的比例（0 ~ 1）。

    讀取失敗時回傳 None，呼叫端略過條件 3，避免誤觸重啟。
    """
    try:
        info = {}
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                key, value = line.split(":", 1)
                info[key] = int(value.split()[0])   # 單位 kB；只算比例，不必換算
        return info["MemAvailable"] / info["MemTotal"]
    except (OSError, KeyError, ValueError, ZeroDivisionError):
        return None


# =============================================================================
# 讀取服務 log
# =============================================================================
# 都以 InvocationID 過濾，只看目前這次啟動的 log；
# 否則重啟後仍會讀到上一次的紀錄，又觸發一次重啟。

def read_recent_log(invocation_id):
    """本次啟動、最近 LOG_WINDOW 內的 log（依時間先後）。"""
    r = subprocess.run(
        ["journalctl", f"_SYSTEMD_INVOCATION_ID={invocation_id}",
         "--since", LOG_WINDOW, "-o", "cat", "--no-pager"],
        capture_output=True, text=True, check=False,
    )
    return r.stdout.splitlines()


def read_head_log(invocation_id):
    """
    本次啟動最前面的 HEAD_LINES 行 log。

    journalctl 由舊到新輸出，讀滿就停止，不必讀完整段（服務跑越久越大）。
    """
    proc = subprocess.Popen(
        ["journalctl", f"_SYSTEMD_INVOCATION_ID={invocation_id}", "-o", "cat", "--no-pager"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    lines = []
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        if len(lines) >= HEAD_LINES:
            break
    proc.kill()     # 讀夠就停止；已經結束時 kill 無作用
    proc.wait()
    return lines


def find_failure(lines):
    """回傳第一行符合 RE_FAILURE 的 log；沒有則回傳 None。"""
    for line in lines:
        if RE_FAILURE.search(line):
            return line.strip()
    return None


def get_pending_reconnects(lines):
    """
    每路攝影機目前的連續重連次數。

    依時間先後掃過 log，後出現的覆蓋先前的：最後一筆是重連 → 該次數；最後一筆是恢復 → 0。

    回傳：
        {source_id: 連續重連次數}
    """
    counts = {}
    for line in lines:
        m = RE_RECONNECT.search(line)
        if m:
            counts[m.group(1)] = int(m.group(2))
            continue
        m = RE_RECOVER.search(line)
        if m:
            counts[m.group(1)] = 0
    return counts


def count_recent_reboots():
    """GUARD_LOG 中最近 24 小時的 [REBOOT] 次數；檔案不存在或讀不到時為 0。"""
    since = datetime.now() - timedelta(hours=24)
    count = 0
    try:
        with open(GUARD_LOG, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "[REBOOT]" not in line:
                    continue
                try:
                    stamp = datetime.strptime(line[:19], TIME_FORMAT)   # 19 = "YYYY-MM-DD HH:MM:SS" 的長度
                except ValueError:
                    continue
                if stamp >= since:
                    count += 1
    except OSError:
        pass
    return count


# =============================================================================
# 處置
# =============================================================================

def memory_snapshot():
    """
    重開機前的記憶體現場；重開機會清掉現場，先留下來供事後判斷原因。

        free -h              記憶體是否真的用完
        /proc/buddyinfo      大塊連續記憶體還剩多少（判斷是否碎片化）
        NVMM 各行程用量      NVMM 被誰佔住；加總遠小於實際消失量 → 殘留在驅動層
        記憶體用量前 5 名     是否有其他行程在吃記憶體

    任一項讀取失敗只記下原因，不影響其他項目，也不阻止重開機。

    回傳：
        要寫進 log 的行
    """
    def run(cmd):
        """執行唯讀指令，回傳輸出的行；失敗時回傳說明。"""
        try:
            # 10 秒：避免卡住整輪檢查
            r = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
            return (r.stdout or r.stderr).splitlines()
        except (OSError, subprocess.SubprocessError) as e:
            return [f"（無法執行：{e}）"]

    def read(path):
        """讀檔案的行；失敗時回傳說明。"""
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as e:
            return [f"（無法讀取：{e}）"]

    sections = [
        ("free -h",               run(["free", "-h"])),
        ("/proc/buddyinfo",       read("/proc/buddyinfo")),
        (str(NVMAP_CLIENTS),      read(NVMAP_CLIENTS)),
        # 6 = 標題 + 5 行
        ("記憶體用量前 5 名行程",  run(["ps", "-eo", "pid,rss,etime,comm", "--sort=-rss"])[:6]),
    ]
    lines = []
    for title, body in sections:
        lines.append(f"    --- {title} ---")
        lines.extend(f"    {row}" for row in body[:SNAPSHOT_MAX_LINES])
    return lines


def restart_service(reason):
    """重啟服務並記錄原因。"""
    log(f"[RESTART] {reason} → systemctl restart {SERVICE}")
    r = subprocess.run(["systemctl", "restart", SERVICE], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        log(f"[ERROR] 重啟失敗（exit={r.returncode}）：{r.stderr.strip()}")


def reboot_system(reason):
    """
    重開機；開機未滿 MIN_SYS_UPTIME_SEC，或 24 小時內已重開 MAX_REBOOTS_PER_DAY 次時不動作。
    """
    sys_uptime = get_system_uptime_sec()
    if sys_uptime is None or sys_uptime < MIN_SYS_UPTIME_SEC:
        return                                  # 開機滿 30 分鐘後，下一輪若仍故障再處理

    recent = count_recent_reboots()
    if recent >= MAX_REBOOTS_PER_DAY:
        return                                  # 已達上限；不寫 log，避免故障期間每 5 分鐘灌一行

    # 先寫：上限計數靠這行
    log(f"[REBOOT] {reason}（24 小時內第 {recent + 1} 次）→ systemctl reboot")
    log("[SNAPSHOT] 重開機前的記憶體狀態：")    # 快照各行不含 [REBOOT]，不會被誤計
    for line in memory_snapshot():
        print(line, flush=True)
    r = subprocess.run(["systemctl", "reboot"], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        log(f"[ERROR] 重開機失敗（exit={r.returncode}）：{r.stderr.strip()}")


# =============================================================================
# 主流程
# =============================================================================

def main():
    """執行一輪檢查（條件見檔頭），回傳結束碼 0。"""
    # active = 運作中；failed = 反覆當掉後被 systemd 放棄。其他（inactive 等）是手動停止，不處理
    state = systemctl_show("ActiveState")
    if state not in ("active", "failed"):
        return 0

    invocation_id = systemctl_show("InvocationID")
    if not invocation_id:
        return 0

    head = read_head_log(invocation_id)
    recent = read_recent_log(invocation_id)

    # 條件 1：記憶體配置失敗 → 重開機
    hit = find_failure(head) or find_failure(recent)
    if hit:
        reboot_system(f"偵測到記憶體配置失敗：{hit[:120]}")     # 只取前 120 字，避免一行過長
        return 0

    if state != "active":
        return 0                                                # failed 但不是記憶體問題：不在本腳本範圍，交由人工判斷

    # 條件 2：連續重連過多 → 重啟服務
    stuck = {cam: n for cam, n in get_pending_reconnects(recent).items() if n >= MAX_REBUILDS}
    if stuck:
        detail = "、".join(f"{cam} 連續重連 {n} 次" for cam, n in stuck.items())
        restart_service(f"{detail}，仍未恢復")
        return 0

    # 條件 3：記憶體不足 → 重啟服務（服務啟動未滿 MIN_UPTIME_SEC，或取不到資料時不判斷）
    pid = systemctl_show("MainPID")
    uptime = get_uptime_sec(pid) if pid and pid != "0" else None
    ratio = get_mem_available_ratio()
    if ratio is not None and uptime is not None and uptime >= MIN_UPTIME_SEC and ratio < MEM_AVAIL_MIN_RATIO:
        restart_service(f"系統可用記憶體僅剩 {ratio * 100:.1f}%（門檻 {MEM_AVAIL_MIN_RATIO * 100:.0f}%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

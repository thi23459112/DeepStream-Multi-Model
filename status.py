#!/usr/bin/env python3
# =============================================================
#  DeepStream 即時管理面板（systemd timer 版本 / thiedge01）
#  一執行就同時顯示：服務狀態 + 辨識 log + 雲端上傳 log
#  按鍵即時生效（免按 Enter）：
#     1 開啟 = 啟動 deepstream.service + 啟動雲端 timer
#     2 關閉 = 停止 deepstream.service + 停止雲端 timer
#     q 離開
#
#  與舊版差異：
#     - 上傳排程由「crontab 單隻 PutAPI」改為「systemd timer」
#       （send-api-cloud.timer）
#     - 上傳 log 改讀各自的 .log 檔
# =============================================================

import subprocess
import sys
import threading
import time
import select
import termios
import tty
from collections import deque
from pathlib import Path
import os

# -------------------- 檢查 Rich 套件 --------------------
try:
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich.console import Console
except ImportError:
    raise SystemExit(
        "缺少 rich 套件，請先在 tracking 環境安裝：\n"
        "    conda activate tracking && pip install rich"
    )


# =============================================================
#  設定區（本機 thiedge01）
# =============================================================
# systemd unit 名稱
DEEPSTREAM_SERVICE = "deepstream.service"
CLOUD_TIMER        = "send-api-cloud.timer"

# 專案目錄（自動偵測，找不到用預設）
def auto_detect_project_dir() -> Path:
    env_dir = os.environ.get("DS_PROJECT_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    script_dir = Path(__file__).resolve().parent
    for parent in [script_dir] + list(script_dir.parents):
        if (parent / "main.py").exists():
            return parent
    return Path("/home/thi/THI/DeepStream-Multi-Model")


PROJECT_DIR = auto_detect_project_dir()

# 雲端 API 的 log 檔（由 send_interval_api*.py 產生）
CLOUD_LOG = str(PROJECT_DIR / "send_interval_api.live.log")

# 固定設定
MAX_KEEP        = 500      # 日誌保留行數
API_PANEL_ROWS  = 12       # 上傳 log 面板行數


# =============================================================
#  顯示偵測結果
# =============================================================
console = Console()
console.print("\n[bold cyan]╔════════════════════════════════════════════════╗[/]")
console.print("[bold cyan]║   DeepStream 即時管理面板 - systemd timer 版   ║[/]")
console.print("[bold cyan]╚════════════════════════════════════════════════╝[/]\n")
console.print(f"[green]✓[/] 專案目錄  : [yellow]{PROJECT_DIR}[/]")
console.print(f"[green]✓[/] 主服務    : [yellow]{DEEPSTREAM_SERVICE}[/]")
console.print(f"[green]✓[/] 雲端 timer: [yellow]{CLOUD_TIMER}[/]  log: {CLOUD_LOG}")
console.print("")


# =============================================================
#  全域變數與緩衝區
# =============================================================
recog_buffer = deque(maxlen=MAX_KEEP)   # 辨識服務日誌
cloud_buffer = deque(maxlen=MAX_KEEP)   # 雲端上傳日誌
stop_event   = threading.Event()

# 狀態快取
state = {"cloud_timer": None}


# =============================================================
#  背景日誌讀取執行緒
# =============================================================
def recog_reader():
    """持續讀取 DeepStream 服務日誌（journalctl -f）。"""
    try:
        proc = subprocess.Popen(
            ["journalctl", "-u", DEEPSTREAM_SERVICE, "-o", "cat",
             "-n", "80", "-f", "--no-pager"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except FileNotFoundError:
        recog_buffer.append("[錯誤] 找不到 journalctl 指令")
        return
    for line in proc.stdout:
        if stop_event.is_set():
            break
        recog_buffer.append(line.rstrip("\n"))
    try:
        proc.terminate()
    except Exception:
        pass


def make_tail_reader(logpath, buf):
    """回傳一個 tail -F 讀某個 log 檔的執行緒函數。"""
    def _reader():
        try:
            proc = subprocess.Popen(
                ["tail", "-n", "40", "-F", logpath],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
        except FileNotFoundError:
            buf.append("[錯誤] 找不到 tail 指令")
            return
        except Exception as e:
            buf.append(f"[錯誤] 無法讀取日誌: {e}")
            return
        for line in proc.stdout:
            if stop_event.is_set():
                break
            buf.append(line.rstrip("\n"))
        try:
            proc.terminate()
        except Exception:
            pass
    return _reader


# =============================================================
#  服務 / timer 狀態與操作
# =============================================================
def is_active(unit: str) -> str:
    """回傳 unit 的 is-active 狀態字串（active/inactive/failed...）。"""
    result = subprocess.run(
        ["systemctl", "is-active", unit],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip()


def timer_is_enabled(unit: str) -> bool:
    """timer 是否正在運作（active = 有在計時）。"""
    return is_active(unit) == "active"


def start_all():
    """開啟：啟動主服務 + 雲端 timer。"""
    subprocess.run(["sudo", "systemctl", "start", DEEPSTREAM_SERVICE], check=False)
    subprocess.run(["sudo", "systemctl", "start", CLOUD_TIMER], check=False)
    state["cloud_timer"] = True


def stop_all():
    """關閉：停止主服務 + 雲端 timer。"""
    subprocess.run(["sudo", "systemctl", "stop", DEEPSTREAM_SERVICE], check=False)
    subprocess.run(["sudo", "systemctl", "stop", CLOUD_TIMER], check=False)
    state["cloud_timer"] = False


# =============================================================
#  畫面渲染
# =============================================================
def colorize_log_line(line: str) -> str:
    safe = line.replace("[", "\\[")
    low = line.lower()
    if "error" in low or "traceback" in low or "fail" in low or "失敗" in line:
        return f"[red]{safe}[/]"
    if "warn" in low or "逾時" in line:
        return f"[yellow]{safe}[/]"
    if "info" in low or "成功上傳" in line or "success" in low:
        return f"[green]{safe}[/]"
    return safe


def make_log_panel(buf, lines_n: int, title: str, color: str) -> Panel:
    rows = list(buf)[-lines_n:]
    if rows:
        text = Text.from_markup("\n".join(colorize_log_line(x) for x in rows))
    else:
        text = Text("（等待 log 輸出中...）", style="dim")
    return Panel(text, title=f"[{color}]{title}[/]", border_style=color)


def render_dashboard(term_height: int) -> Layout:
    # ----- 主服務狀態 -----
    status = is_active(DEEPSTREAM_SERVICE)
    if status == "active":
        service_text = "[bold green]● 運作中 (active)[/]"
    elif status == "failed":
        service_text = "[bold red]✕ 失敗 (failed)[/]"
    else:
        service_text = f"[bold red]○ 已停止 ({status or '未知'})[/]"

    # ----- 雲端 timer 狀態 -----
    def timer_label(on):
        if on is True:
            return "[bold green]● 啟用[/]"
        if on is False:
            return "[bold red]○ 暫停[/]"
        return "[dim]未知[/]"

    cloud_text = timer_label(state["cloud_timer"])

    # ----- 標題面板 -----
    header = Panel(
        f"辨識服務：{service_text}    雲端上傳：{cloud_text}",
        title="[cyan]DeepStream 即時管理面板[/]",
        border_style="cyan",
    )

    # ----- 底部提示 -----
    footer = Panel(
        "[green]1[/] 開啟(辨識+雲端)     "
        "[red]2[/] 關閉(辨識+雲端)     [dim]q[/] 離開",
        border_style="dim",
    )

    # ----- 面板高度計算 -----
    # 版面：header(3) + 辨識 + 雲端(API_PANEL_ROWS) + footer(3)
    recog_lines = max(3, term_height - 6 - API_PANEL_ROWS - 2)

    recog_panel = make_log_panel(recog_buffer, recog_lines,
                                 "📡 辨識即時日誌", "bright_blue")
    cloud_panel = make_log_panel(cloud_buffer, API_PANEL_ROWS - 2,
                                 "☁️ 雲端上傳日誌 (send_interval_api)", "magenta")

    layout = Layout()
    layout.split_column(
        Layout(header, size=3),
        Layout(recog_panel),
        Layout(cloud_panel, size=API_PANEL_ROWS),
        Layout(footer, size=3),
    )
    return layout


# =============================================================
#  鍵盤輸入與控制
# =============================================================
def read_key(timeout: float = 0.3):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.read(1)
    return None


def suspend_run_and_execute(live, old_termios, func):
    """暫停 Live 畫面、還原終端機（可輸入 sudo 密碼）、執行 func、再恢復。"""
    live.stop()
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_termios)
    try:
        func()
    finally:
        time.sleep(1.0)
        tty.setcbreak(sys.stdin.fileno())
        live.start()


def action_open():
    console.print("\n[green]🔓 開啟：啟動辨識服務 + 雲端 timer...[/]")
    start_all()
    console.print("[green]✅ 完成。[/]")


def action_close():
    console.print("\n[yellow]🔒 關閉：停止辨識服務 + 雲端 timer...[/]")
    stop_all()
    console.print("[yellow]✅ 完成。[/]")


# =============================================================
#  主程式
# =============================================================
def main():
    # 啟動兩個背景 log 讀取執行緒（辨識 + 雲端）
    threading.Thread(target=recog_reader, daemon=True).start()
    threading.Thread(target=make_tail_reader(CLOUD_LOG, cloud_buffer), daemon=True).start()

    # 初始化讀取 timer 狀態
    console.print("[dim]讀取目前狀態中...[/]")
    try:
        state["cloud_timer"] = timer_is_enabled(CLOUD_TIMER)
    except Exception:
        state["cloud_timer"] = None

    fd = sys.stdin.fileno()
    old_termios = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        with Live(console=console, screen=True,
                  auto_refresh=False, refresh_per_second=4) as live:
            while True:
                live.update(render_dashboard(console.size.height), refresh=True)
                key = read_key(0.3)
                if not key:
                    continue
                key = key.lower()
                if key == "1":
                    suspend_run_and_execute(live, old_termios, action_open)
                elif key == "2":
                    suspend_run_and_execute(live, old_termios, action_close)
                elif key == "q":
                    break
    finally:
        stop_event.set()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_termios)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n使用者中斷")
    except Exception as e:
        print(f"\n錯誤: {e}")
        sys.exit(1)

#!/usr/bin/env python3
"""
DeepStream 7.1 車流 / 人流計數主程式

執行前必須先跑 `python traffic_count_txt.py` 產生各組的設定檔（改過 ds_yaml 就要重跑）。

架構：
    ds_yaml 裡 weight（engine）相同的攝影機歸為一組，每組一條獨立的 pipeline，
    全部跑在同一個 GLib mainloop 上：

        uridecodebin × N → nvstreammux → nvdspreprocess → nvinfer → [nvtracker] → nvdsanalytics
                                                                                        │
                               每路：nvdsosd → 存檔 / 預覽視窗  ←── nvstreamdemux ←──────┘

    nvdspreprocess  依 crop_points 裁切、縮放成 engine 的輸入 tensor
    nvinfer         推論；input-tensor-meta 讓它直接使用 nvdspreprocess 準備好的 tensor
    nvtracker       只有 tracker.type=nvdcf 才有；BoxMOT 模式改由 nvinfer.src 上的探針追蹤
    nvdsanalytics   只負責把 ROI 畫到畫面上；計數邏輯在探針裡（logic/probes.py）

結束方式與結束碼：
    按 q、Ctrl+C（SIGINT）           送 EOS 等存檔收尾，最多等 10 秒   130
    收到 SIGTERM（systemctl stop）   同上                              0
    檔案來源全部播完                 自動結束                          0
    嚴重錯誤                         立即結束                          1
    不論哪一種，結束前都會結算仍在畫面上的物件並寫入 DB。

    SIGTERM 回 0：systemd 把非 0 視為失敗，服務停止後會顯示 failed（status.py 面板也是）。
    SIGINT 回 130：run_batches.sh 靠它判斷使用者中斷、不再跑下一批。

各模組職責見 README.md。
"""

import os
import sys
import tty
import time
import ctypes
import signal
import termios
import traceback

# 每行 log 立即送出。以 systemd 執行時 stdout 不是終端機，Python 預設累積約 8 KB 才寫出，
# journal 與 ds_guard.py 會晚好幾分鐘才看到；必須在任何 print 之前設定
sys.stdout.reconfigure(line_buffering=True)

# conda 環境下，GLib 建立網路連線時呼叫的系統 libproxy 會因 libstdc++ 版本不合讓行程 abort；
# 改用內建的 dummy proxy resolver 繞過。必須在 import gi 之前設定。
os.environ.setdefault("GIO_USE_PROXY_RESOLVER", "dummy")

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

from logic import sources
from logic.common import NVTRACKER_LIB_NAME, resolve_ds_lib
from logic.config import (
    SOURCE_CONFIGS, GROUPS, TRACKER_CONFIG, TRACKER_MODE,
    group_infer_config, group_preprocess_config, group_analytics_config,
    check_generated_configs, disable_windows_without_display,
)
from logic.pipeline import (
    ALL_QUEUES, make_elm, make_queue, make_streammux,
    build_display_sink, setup_cam_branch,
)
from logic.probes import nvdcf_probe, boxmot_probe, osd_probe
from logic.state_db import initialize_state_managers, force_finalize_all, fps_streams


# =============================================================================
# 常數與狀態
# =============================================================================

EOS_TIMEOUT_SECONDS = 10    # 送出 EOS 後最多等多久（秒）
WATCHDOG_SECONDS = 5        # 重連檢查與卡住診斷的執行間隔（秒）
STALL_SECONDS = 10          # 某組超過多久（秒）沒有新畫面，印卡住診斷
STARTUP_GRACE_SECONDS = 30  # 啟動後多久（秒）內還沒收到第一幀不算異常（RTSP 連線需要時間）
STALL_REPEAT_SECONDS = 30   # 持續卡住時，診斷訊息的重複間隔（秒）
TRACKER_WIDTH = 640         # nvtracker 內部處理的解析度（須為 32 的倍數），NVIDIA 範例的建議值
TRACKER_HEIGHT = 384

EXIT_OK = 0                 # 正常結束：檔案播完、SIGTERM
EXIT_ERROR = 1              # 嚴重錯誤、啟動檢查失敗
EXIT_INTERRUPTED = 130      # 使用者中斷：q、Ctrl+C（128 + SIGINT 的 2，shell 的慣例）

g_loop = None
g_pipelines = []            # [(pipeline, 該組的 uid list)]
g_eos_done = set()          # 已收到 EOS 的 pipeline 名稱
g_stopping = False          # 是否已進入結束流程
g_exit_code = EXIT_OK
g_last_stall_report = {}    # pipeline 名稱 → 上次印卡住診斷的時間
g_missing_cams_reported = False


# =============================================================================
# 結束流程
# =============================================================================

def stop_all(reason, exit_code=EXIT_INTERRUPTED):
    """使用者要求結束：對所有 pipeline 送 EOS，並設逾時保底。重複呼叫無作用。"""
    global g_stopping, g_exit_code
    if g_stopping:
        return                  # 連按 Ctrl+C 不會重複送 EOS
    g_stopping = True
    g_exit_code = exit_code
    sources.begin_shutdown()    # 之後放行 EOS、不再重連
    print(f"\n[INFO] {reason}，對所有 pipeline 送 EOS（等待存檔收尾）...")
    for pipeline, _uids in g_pipelines:
        pipeline.send_event(Gst.Event.new_eos())
    GLib.timeout_add_seconds(EOS_TIMEOUT_SECONDS, _force_quit)


def _force_quit():
    """EOS 逾時：不再等待，直接離開 mainloop。"""
    if g_loop.is_running():
        print("\n[WARNING] 等待 EOS 逾時，強制退出（存檔中的影片結尾可能不完整）")
        g_loop.quit()
    return False    # False：timeout 只執行一次


def _on_sigint(_data):
    """Ctrl+C，或 run_batches.sh 無終端機模式轉送的中斷。"""
    stop_all("收到 SIGINT")
    return GLib.SOURCE_CONTINUE     # 保留監聽，重複的訊號由 stop_all 忽略


def _on_sigterm(_data):
    """systemctl stop 等服務管理器的正常停止要求。"""
    stop_all("收到 SIGTERM", exit_code=EXIT_OK)
    return GLib.SOURCE_CONTINUE


def _on_key(_fd, _condition):
    """終端機按鍵；q 結束。"""
    if sys.stdin.read(1) in ("q", "Q"):
        stop_all("收到 q 鍵")
        return False    # False：不再監聽鍵盤
    return True


def _on_bus_message(_bus, message, name):
    """
    每條 pipeline 的 bus 訊息。

        EOS    記錄這條 pipeline 已結束；全部結束就離開 mainloop
        ERROR  即時來源（RTSP）的錯誤不致命，交給 logic/sources.py 排程重連；
               其他元件的錯誤視為嚴重錯誤，直接結束程式
    """
    global g_exit_code
    if message.type == Gst.MessageType.EOS:
        g_eos_done.add(name)
        print(f"[INFO] {name} 已結束（{len(g_eos_done)}/{len(g_pipelines)}）")
        if len(g_eos_done) >= len(g_pipelines):
            g_loop.quit()
    elif message.type == Gst.MessageType.ERROR:
        if not sources.handle_bus_error(message):
            err, debug = message.parse_error()
            print(f"[ERROR] [{name}] 嚴重錯誤：{err.message}\n        {debug}")
            g_exit_code = EXIT_ERROR
            g_loop.quit()
    return True     # True：繼續接收訊息


# =============================================================================
# 定時檢查：重連與卡住診斷
# =============================================================================

def _last_frame_time(uid):
    """該攝影機最後一次收到畫面的時間；還沒收過回傳 None。"""
    stamps = fps_streams[uid]["timestamps"]
    return stamps[-1] if stamps else None


def _watchdog(started_at):
    """
    每 WATCHDOG_SECONDS 秒執行一次。

        1. 即時來源斷線 / 滴幀 / 報錯 → 重連（logic/sources.py）。
        2. 啟動 STARTUP_GRACE_SECONDS 秒後仍沒有任何畫面的攝影機 → 列出一次。
        3. 某組超過 STALL_SECONDS 沒有新畫面 → 印出該 pipeline 各 queue 的水位。
           滿的 queue 的下游就是卡住的元件；全部是空的代表來源沒有送資料進來。

    2、3 只印訊息，不做處置。
    """
    global g_missing_cams_reported
    if g_stopping:
        return False                                                # 結束流程中：停止定時器
    sources.reconnect_dead_sources()
    now = time.time()

    if not g_missing_cams_reported and now - started_at >= STARTUP_GRACE_SECONDS:
        g_missing_cams_reported = True                              # 只在啟動後報一次
        missing = [cfg["source_id"] for uid, cfg in SOURCE_CONFIGS.items()
                   if _last_frame_time(uid) is None]
        if missing:
            print(f"\n[WARNING] 啟動 {STARTUP_GRACE_SECONDS} 秒後仍沒有畫面："
                  f"{', '.join(missing)}（請檢查來源；RTSP 會持續自動重連）\n")

    for pipeline, uids in g_pipelines:
        name = pipeline.get_name()
        if name in g_eos_done:
            continue                                                # 已播完的組不檢查
        seen = [t for t in map(_last_frame_time, uids) if t is not None]
        if seen:
            idle, limit = now - max(seen), STALL_SECONDS
        else:
            idle, limit = now - started_at, STARTUP_GRACE_SECONDS   # 還沒收過畫面：用啟動寬限
        if idle < limit:
            g_last_stall_report.pop(name, None)                     # 恢復正常，下次卡住立刻報
            continue
        if now - g_last_stall_report.get(name, 0) < STALL_REPEAT_SECONDS:
            continue                                                # 剛報過，避免洗版
        g_last_stall_report[name] = now

        print(f"\n[WARNING] [{name}] 已 {idle:.0f} 秒沒有新畫面。非空的 queue（水位/容量）：")
        for queue in ALL_QUEUES:
            if queue.get_parent() is not pipeline:
                continue                                            # 只列這條 pipeline 的 queue
            level = queue.get_property("current-level-buffers")
            capacity = queue.get_property("max-size-buffers")
            if level:
                print(f"[WARNING]     {queue.get_name():<24} {level}/{capacity}"
                      f"{'  ← 滿，卡點在它的下游' if level >= capacity else ''}")
        print("[WARNING]   （沒列出任何 queue = 全部是空的，來源沒有送資料進來）\n")
    return True                                                     # True：定時器繼續


# =============================================================================
# 建立 pipeline
# =============================================================================

def build_group_pipeline(group_id, group):
    """為一組（一個 engine）建立一條 pipeline，結構見檔頭。"""
    g = group_id
    uids = group["member_uids"]                                     # 索引 = 這條 pipeline 的 pad_index
    pipeline = Gst.Pipeline.new(f"traffic-pipeline-g{g}")

    streammux = make_streammux(f"Stream-muxer-g{g}", len(uids))
    pipeline.add(streammux)
    for pad_index, uid in enumerate(uids):
        sources.add_source(pipeline, streammux, pad_index, uid)     # 來源接到 streammux.sink_{pad_index}

    preprocess = make_elm("nvdspreprocess", f"preprocess-g{g}")
    preprocess.set_property("config-file", group_preprocess_config(g))

    pgie = make_elm("nvinfer", f"primary-inference-g{g}")
    pgie.set_property("config-file-path", group_infer_config(g))
    pgie.set_property("input-tensor-meta", True)                    # 使用 nvdspreprocess 準備好的 tensor

    analytics = make_elm("nvdsanalytics", f"analytics-g{g}")
    analytics.set_property("config-file", group_analytics_config(g))

    demux = make_elm("nvstreamdemux", f"demuxer-g{g}")

    chain = [streammux, make_queue(f"q1-g{g}"), preprocess, make_queue(f"q2-g{g}"),
             pgie, make_queue(f"q3-g{g}")]

    # 追蹤探針的 ctx，欄位說明見 logic/probes.py
    ctx = {"pad_to_uid": list(uids), "group_uids": frozenset(uids), "class_map": group["class_map"]}
    if TRACKER_MODE == "nvdcf":
        tracker = make_elm("nvtracker", f"tracker-g{g}")
        tracker.set_property("ll-config-file", TRACKER_CONFIG)
        # 安裝位置自動偵測
        tracker.set_property("ll-lib-file", resolve_ds_lib(NVTRACKER_LIB_NAME))
        tracker.set_property("tracker-width", TRACKER_WIDTH)
        tracker.set_property("tracker-height", TRACKER_HEIGHT)
        tracker.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, nvdcf_probe, ctx)
        chain.append(tracker)
    else:
        # BoxMOT 在探針裡追蹤
        pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, boxmot_probe, ctx)

    chain += [make_queue(f"q_analytics-g{g}"), analytics, make_queue(f"q4-g{g}"), demux]
    for element in chain[1:]:                                       # streammux 已經加過
        pipeline.add(element)
    for upstream, downstream in zip(chain, chain[1:]):
        upstream.link(downstream)

    # 本組只要有一路要顯示，就建這一組的預覽視窗
    show = any(SOURCE_CONFIGS[uid]["display"].get("show_window", True) for uid in uids)
    display_mux = build_display_sink(pipeline, g, len(uids)) if show else None
    for pad_index, uid in enumerate(uids):
        setup_cam_branch(pipeline, pad_index, uid, SOURCE_CONFIGS[uid], demux, display_mux, osd_probe)

    return pipeline


def _init_x_threads():
    """
    讓 Xlib 進入多執行緒安全模式，必須早於任何 X 呼叫。

    每個預覽視窗有自己的繪圖執行緒；沒做這一步時，多個執行緒同時呼叫 X 可能隨機死鎖。
    """
    for lib in ("libX11.so.6", "libX11.so"):
        try:
            ctypes.CDLL(lib).XInitThreads()
            return
        except (OSError, AttributeError):
            continue    # 沒有這個檔名就試下一個；都沒有代表沒裝 X


# =============================================================================
# 主流程
# =============================================================================

def main():
    """建立所有 pipeline 並執行到結束，回傳結束碼。"""
    global g_loop

    check_generated_configs()                                           # 設定檔過期或缺漏就直接結束
    disable_windows_without_display()
    initialize_state_managers()

    if TRACKER_MODE != "nvdcf":
        from logic.boxmot_adapter import initialize_boxmot_trackers     # 只有 BoxMOT 模式才載入 boxmot
        initialize_boxmot_trackers()

    if any(cfg["display"].get("show_window", True) for cfg in SOURCE_CONFIGS.values()):
        _init_x_threads()                                               # 必須早於建立任何視窗

    Gst.init(None)
    g_loop = GLib.MainLoop()

    for gid, group in GROUPS.items():
        pipeline = build_group_pipeline(gid, group)
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", _on_bus_message, pipeline.get_name())
        g_pipelines.append((pipeline, group["member_uids"]))

    GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGTERM, _on_sigterm, None)
    GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGINT, _on_sigint, None)

    # 有終端機時監聽 q 鍵；結束時還原終端機設定
    stdin_fd = sys.stdin.fileno() if sys.stdin.isatty() else None
    tty_settings = None
    if stdin_fd is not None:
        tty_settings = termios.tcgetattr(stdin_fd)
        tty.setcbreak(stdin_fd)                                         # cbreak：按鍵立即送出，不必按 Enter
        GLib.io_add_watch(stdin_fd, GLib.PRIORITY_DEFAULT, GLib.IOCondition.IN, _on_key)
        print("\n[INFO] 按 q 或 Ctrl+C 可安全結束\n")
    else:
        print("\n[INFO] 沒有終端機輸入，請用 Ctrl+C / SIGTERM 結束\n")

    try:
        for pipeline, _uids in g_pipelines:
            pipeline.set_state(Gst.State.PLAYING)
        GLib.timeout_add_seconds(WATCHDOG_SECONDS, _watchdog, time.time())
        g_loop.run()
    finally:
        if tty_settings is not None:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, tty_settings)
        force_finalize_all()                                            # 先結算寫 DB，再停 pipeline
        for pipeline, _uids in g_pipelines:
            pipeline.set_state(Gst.State.NULL)
        print("[INFO] 程式結束")
    return g_exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise   # sys.exit 的結束碼原樣傳出
    except Exception:
        traceback.print_exc()
        sys.exit(EXIT_ERROR)

#!/usr/bin/env python3
"""
DeepStream 7.1 車流計數主程式（多權重 / 多 pipeline 版）

架構：
    依 weight 分組（見 logic/config.py 的 GROUPS），每組建「一條獨立 pipeline」：
        streammux → q1 → preprocess(該組) → q2 → pgie(該組engine) → q3
        → [nvtracker(僅 nvdcf)] → q_analytics → analytics(該組) → q4 → demux → 各路分支
    所有 pipeline 跑在「同一個 GLib mainloop」上（官方支援的多 pipeline 同進程做法），
    彼此獨立、互不干擾；q / Ctrl+C 對「所有」pipeline 送 EOS，全部結束才退出。

鍵的觀念：
    - 全域 stream_uid（SOURCE_CONFIGS 的鍵）跨所有組唯一；state_db / boxmot 都用它當鍵。
    - 每條 pipeline 內 frame_meta.pad_index 是區域編號，由 probe 用 ctx["pad_to_uid"] 翻譯回 uid。
"""

import sys
import os
import termios
import tty
import signal
import traceback

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

from logic.color import load_labels, CLASS_MAP
from logic.config import (
    SOURCE_CONFIGS, GROUPS, BASE_DIR,
    TRACKER_CONFIG, TRACKER_MODE, BOXMOT_TRACKER_CONFIG,
    group_infer_config, group_preprocess_config,
)
from logic.state_db import initialize_state_managers, force_finalize_all
from logic.pipeline import (
    cb_newpad, cb_source_setup, make_elm,
    _build_display_sink, setup_cam_branch,
)
from logic.probes import (
    tracker_src_pad_buffer_probe,
    boxmot_pgie_src_probe,
    per_cam_osd_probe,
)


# ==========================================
# 1. 全域狀態
# ==========================================

g_loop          = None      # GLib 主迴圈
g_pipelines     = []         # 所有 Gst.Pipeline（每組一條）
g_eos_done      = set()      # 已 EOS 的 pipeline 名稱集合
g_eos_triggered = False      # 是否已對所有 pipeline 送過 EOS


def _group_analytics_config(group_id):
    """該組的 nvdsanalytics 設定檔路徑（由 traffic_count_txt.py 產生）。"""
    return os.path.join(BASE_DIR, f"config_nvdsanalytics_group{group_id}.txt")


# ==========================================
# 2. 結束與訊息處理
# ==========================================

def force_quit_loop():
    """EOS 逾時 fallback：等太久仍未全部封裝完成就強制 quit。"""
    global g_loop
    print("\n[WARNING] 等待影片封裝逾時，強制退出所有管線！")
    if g_loop and g_loop.is_running():
        g_loop.quit()
    return False


def _send_eos_to_all():
    """對所有 pipeline 送 EOS（安全結束、等影片封裝）。"""
    global g_eos_triggered
    if g_eos_triggered:
        return
    g_eos_triggered = True
    print("\n[INFO] 正在對所有 pipeline 發送 EOS（等待影片寫入）...")
    for p in g_pipelines:
        p.send_event(Gst.Event.new_eos())
    GLib.timeout_add_seconds(10, force_quit_loop)


def keyboard_cb(fd, condition):
    """終端機按 Q → 對所有 pipeline 送 EOS 安全退出。"""
    ch = sys.stdin.read(1)
    if ch in ('q', 'Q') and not g_eos_triggered:
        print("\n[INFO] 收到 'Q' 鍵，準備安全退出並存檔...")
        _send_eos_to_all()
        return False
    return True


def bus_call(bus, message, pipeline_name):
    """
    每條 pipeline 各自的 bus 處理。
    EOS   → 記錄此 pipeline 已結束；所有 pipeline 都 EOS 才 quit mainloop
    ERROR → RTSP 來源不穩只警告；其餘嚴重錯誤 → 直接退出（停全部）
    """
    t = message.type
    if t == Gst.MessageType.EOS:
        g_eos_done.add(pipeline_name)
        print(f"[INFO] {pipeline_name} 已 EOS（{len(g_eos_done)}/{len(g_pipelines)}）")
        if len(g_eos_done) >= len(g_pipelines) and g_loop and g_loop.is_running():
            print("[INFO] 所有 pipeline 皆結束，退出。")
            g_loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        err_msg = str(err).lower()
        if ("rtsp" in err_msg or "timeout" in err_msg
                or "resource not found" in err_msg or "could not read" in err_msg):
            print(f"[WARNING] [{pipeline_name}] RTSP 來源不穩或中斷: {err}。保持運行，等待重連...")
        else:
            print(f"[ERROR] [{pipeline_name}] 嚴重管線錯誤: {err}: {debug}")
            if g_loop and g_loop.is_running():
                g_loop.quit()
    return True


# ==========================================
# 3. Pipeline 輔助
# ==========================================

def _enlarge_queue(q, max_buffers=400):
    q.set_property("max-size-buffers", max_buffers)
    q.set_property("max-size-bytes", 0)
    q.set_property("max-size-time", 0)


def build_group_pipeline(group_id, group):
    """
    為「單一組」建立一條完整、獨立的 Gst.Pipeline。

    參數：
        group_id (int)
        group (dict): GROUPS[group_id]，含 weight/engine_path/class_map/member_uids...
    返回：
        (pipeline, pipeline_name)
    """
    members = group["member_uids"]          # 全域 uid，index 即該組區域 pad_index
    num = len(members)
    gname = f"traffic-pipeline-g{group_id}"
    pipeline = Gst.Pipeline.new(gname)

    # ---- streammux（主線）----
    streammux = make_elm("nvstreammux", f"Stream-muxer-g{group_id}")
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", num)          # 該組成員數
    streammux.set_property("batched-push-timeout", 16666)
    streammux.set_property("live-source", 1)
    streammux.set_property("nvbuf-memory-type", 0)
    pipeline.add(streammux)

    # ---- 每個成員 cam 一個 uridecodebin ----
    for local_idx, uid in enumerate(members):
        cfg = SOURCE_CONFIGS[uid]
        src = make_elm("uridecodebin", f"uri-decode-bin-{uid}")
        src.set_property("uri", cfg["source"])
        # pad_index 用「區域」索引（接該組 streammux 的 sink_{local_idx}）
        src.connect("pad-added", cb_newpad, {"streammux": streammux, "pad_index": local_idx})
        src.connect("source-setup", cb_source_setup, None)
        pipeline.add(src)

    # ---- 共用推論元件（該組專屬 config）----
    q1          = make_elm("queue", f"q1-g{group_id}")
    q2          = make_elm("queue", f"q2-g{group_id}")
    q3          = make_elm("queue", f"q3-g{group_id}")
    q_analytics = make_elm("queue", f"q_analytics-g{group_id}")
    q4          = make_elm("queue", f"q4-g{group_id}")
    _enlarge_queue(q_analytics, max_buffers=200)

    preprocess = make_elm("nvdspreprocess", f"preprocess-g{group_id}")
    preprocess.set_property("config-file", group_preprocess_config(group_id))

    pgie = make_elm("nvinfer", f"primary-inference-g{group_id}")
    pgie.set_property("config-file-path", group_infer_config(group_id))
    pgie.set_property("input-tensor-meta", True)

    analytics = make_elm("nvdsanalytics", f"analytics-g{group_id}")
    analytics.set_property("config-file", _group_analytics_config(group_id))

    tracker = None
    if TRACKER_MODE == "nvdcf":
        tracker = make_elm("nvtracker", f"tracker-g{group_id}")
        tracker.set_property("ll-config-file", TRACKER_CONFIG)
        tracker.set_property(
            "ll-lib-file",
            "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        )
        tracker.set_property("tracker-width", 640)
        tracker.set_property("tracker-height", 384)

    elems = [q1, preprocess, q2, pgie, q3, q_analytics, analytics, q4]
    if tracker is not None:
        elems.append(tracker)
    for e in elems:
        pipeline.add(e)

    # streammux → q1 → preprocess → q2 → pgie → q3
    streammux.link(q1); q1.link(preprocess); preprocess.link(q2); q2.link(pgie); pgie.link(q3)
    if TRACKER_MODE == "nvdcf":
        q3.link(tracker); tracker.link(q_analytics)
    else:
        q3.link(q_analytics)
    q_analytics.link(analytics); analytics.link(q4)

    # ---- probe ctx：區域 pad → 全域 uid 對照 + 該組 class_map ----
    ctx = {"pad_to_uid": list(members), "class_map": group["class_map"]}
    if TRACKER_MODE == "nvdcf":
        tracker.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER,
                                                tracker_src_pad_buffer_probe, ctx)
        print(f"[INFO] [group{group_id}] 掛探針 tracker_src_pad_buffer_probe → tracker.src")
    else:
        pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER,
                                             boxmot_pgie_src_probe, ctx)
        print(f"[INFO] [group{group_id}] 掛探針 boxmot_pgie_src_probe → pgie.src ({TRACKER_MODE})")

    # ---- demux + 各路下游 ----
    demux = make_elm("nvstreamdemux", f"demuxer-g{group_id}")
    pipeline.add(demux)
    q4.link(demux)

    # 該組是否有人要顯示 → 建該組自己的 tile 視窗
    show_window = any(
        SOURCE_CONFIGS[uid].get("display", {}).get("show_window", True) for uid in members
    )
    display_streammux = _build_display_sink(pipeline, group_id, num) if show_window else None

    for local_idx, uid in enumerate(members):
        setup_cam_branch(pipeline, local_idx, uid, SOURCE_CONFIGS[uid],
                         demux, display_streammux, per_cam_osd_probe)

    return pipeline, gname


# ==========================================
# 4. 主程式
# ==========================================

def main():
    global g_loop

    print("[INFO] >>> 進入 main()，開始建構多 pipeline...")

    # BoxMOT 模式：為「所有 uid」建 tracker（boxmot_adapter 用 uid 當鍵，多 pipeline 不衝突）
    if TRACKER_MODE == "nvdcf":
        print("[INFO] 追蹤器：NvDCF（每組各建一個 nvtracker）")
    else:
        print(f"[INFO] 追蹤器：{TRACKER_MODE}（BoxMOT，於 pgie.src 探針接管）")
        from logic.boxmot_adapter import initialize_boxmot_trackers
        initialize_boxmot_trackers()

    Gst.init(None)

    # ---- 依組建立多條 pipeline ----
    for gid, group in GROUPS.items():
        members = ", ".join(SOURCE_CONFIGS[u].get("source_id", f"uid{u}") for u in group["member_uids"])
        print(f"[INFO] 建立 group{gid} pipeline：engine={group['weight']}, 成員=[{members}]")
        pipeline, gname = build_group_pipeline(gid, group)
        g_pipelines.append(pipeline)

    # ---- 訊號 + 鍵盤 + mainloop ----
    g_loop = GLib.MainLoop()

    def _on_stop_signal(_u):
        print("\n[INFO] 收到停止訊號（SIGTERM/SIGINT），準備安全退出...")
        _send_eos_to_all()
        return GLib.SOURCE_CONTINUE

    GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGTERM, _on_stop_signal, None)
    GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGINT, _on_stop_signal, None)

    interactive = sys.stdin.isatty()
    fd = None
    old_settings = None
    if interactive:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        GLib.io_add_watch(fd, GLib.PRIORITY_DEFAULT, GLib.IOCondition.IN, keyboard_cb)
        print("\n[INFO] 💡 提示：在終端機按 'q' 即可優雅退出並存檔...\n")
    else:
        print("\n[INFO] 非互動模式（無終端機）：鍵盤監聽停用，請用訊號安全停止。\n")

    try:
        # 每條 pipeline 各掛自己的 bus 監聽
        for p in g_pipelines:
            bus = p.get_bus()
            bus.add_signal_watch()
            bus.connect("message", bus_call, p.get_name())

        print("[INFO] 所有 pipeline 設為 PLAYING...")
        for p in g_pipelines:
            p.set_state(Gst.State.PLAYING)

        g_loop.run()
    finally:
        print("[INFO] 進入清理階段...")
        if interactive and fd is not None and old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        force_finalize_all()                 # 全部組的計數一次結算進合併 DB
        for p in g_pipelines:
            p.set_state(Gst.State.NULL)
        print("[INFO] 所有 pipeline 已停止，程式結束。")


if __name__ == '__main__':
    print("[INFO] >>> 程式啟動，初始化狀態管理員...")
    try:
        initialize_state_managers()
        print("[INFO] 狀態管理員初始化完成，準備進入 main()...")
        main()
    except SystemExit as e:
        print(f"[ERROR] 程式觸發 SystemExit，代碼: {e.code}")
        sys.exit(e.code)
    except Exception as e:
        print(f"[FATAL] 程式發生未預期嚴重錯誤: {e}")
        traceback.print_exc()
        sys.exit(1)
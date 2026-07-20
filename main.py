#!/usr/bin/env python3
"""
DeepStream 7.1 車流計數主程式（多權重 / 多 pipeline 版）

架構：
    依 weight 分組（見 logic/config.py 的 GROUPS），每組建「一條獨立 pipeline」：
        streammux → q1 → preprocess(該組) → q2 → pgie(該組engine) → q3
        → [nvtracker(僅 nvdcf)] → q_analytics → analytics(該組) → q4 → demux → 各路分支
    所有 pipeline 跑在「同一個 GLib mainloop」上（官方支援的多 pipeline 同進程做法），
    彼此獨立、互不干擾；q / Ctrl+C 對「所有」pipeline 送 EOS，全部結束才退出。

關鍵觀念：
    - 全域 stream_uid（SOURCE_CONFIGS 的鍵）跨所有組唯一；state_db / boxmot 都用它當鍵。
    - 每條 pipeline 內 frame_meta.pad_index 是區域編號，由 probe 用 ctx["pad_to_uid"] 翻回 uid。

平台相容：
    顯示 / OSD / 編碼器的平台差異由 logic/pipeline.py 依 _is_jetson() 自動處理，
    同一份程式碼可在 Jetson 與 dGPU/WSL2 上執行。
"""

import sys
import os
import time
import termios
import tty
import signal
import traceback

# GLib/GIO 建立網路（RTSP）連線時會呼叫系統 libproxy 偵測 proxy。在 conda 環境下，
# conda 的 libstdc++ 與系統 libunwind ABI 不相容，libproxy 拋例外時無法正常 unwind
# 而導致行程 abort。改用 GIO 內建 dummy proxy resolver 完全繞過 libproxy。
# 須在匯入 gi 之前設定；Jetson 系統 Python 不受影響，setdefault 也不覆蓋外部既有設定。
os.environ.setdefault("GIO_USE_PROXY_RESOLVER", "dummy")
# 新版 nvstreammux 開關：export USE_NEW_NVSTREAMMUX=yes 啟用。
# 解決多路「檔案來源」某一路先 EOS 時，舊版 mux 空等該路、其餘來源 FPS 大跌的問題。
# 必須在 import gi 之前設定，GStreamer 載入 nvstreammux 外掛時才讀得到。
os.environ.setdefault("USE_NEW_NVSTREAMMUX", "no")

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

from logic.config import (
    SOURCE_CONFIGS, GROUPS, BASE_DIR,
    TRACKER_CONFIG, TRACKER_MODE,
    group_infer_config, group_preprocess_config,
)
from logic.state_db import initialize_state_managers, force_finalize_all, fps_streams
from logic.pipeline import (
    cb_newpad, cb_decodebin_child_added, make_elm, _safe_set, resolve_tracker_lib,
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

g_loop          = None       # GLib 主迴圈
g_pipelines     = []         # 所有 Gst.Pipeline（每組一條）
g_eos_done      = set()      # 已 EOS 的 pipeline 名稱集合
g_eos_triggered = False      # 是否已對所有 pipeline 送過 EOS
# ---- 看門狗（單路卡死自動重啟）----
g_sources        = {}        # uid -> {"src", "streammux", "pad_index", "pipeline", "cfg", "rebuilds"}
g_last_restart   = {}        # uid -> 上次重啟時間戳（防連環重啟）
g_stall_notify   = {}        # uid -> 具名卡幀通知累計次數（恢復吐幀即歸零）
g_restart_counts = {}        # uid -> 看門狗連續重啟/重建次數（恢復吐幀即歸零）
WATCHDOG_STALL_SEC   = 60    # 連續幾秒沒吐幀 → 判定卡死
WATCHDOG_GRACE_SEC   = 60    # 重建後寬限幾秒（期間不再判定）——適用「重建後有進過幀」（恢復中，給足觀察時間）
WATCHDOG_GRACE_FAST  = 30    # 快速重試寬限——適用「重建後連一幀都沒進來」（對端沒回應，縮短失敗循環）。
                             # 不可低於「首幀所需時間」（握手+等關鍵幀+解碼器初始化，約 3~15s），
                             # 否則會親手殺掉每一次快成功的連線，變成自己造成的無限循環
WATCHDOG_CHECK_SEC   = 10    # 每幾秒檢查一次
WATCHDOG_NOTIFY_SEC  = 15    # 卡幀超過此秒數即開始「具名」回報（早於重啟門檻，方便定位是哪一路）
WATCHDOG_MIN_FPS     = 1.0   # 滴幀偵測門檻：觀察窗內平均 FPS 低於此值視同卡死。堵住「內建重連
                             # 半成功、每次滴進幾幀把 idle 重置，導致看門狗永遠不觸發」的漏洞

def _group_analytics_config(group_id):
    """該組的 nvdsanalytics 設定檔路徑（由 traffic_count_txt.py 產生）。"""
    return os.path.join(BASE_DIR, f"config_nvdsanalytics_group{group_id}.txt")


# ==========================================
# 2. 結束與訊息處理
# ==========================================

def force_quit_loop():
    """EOS 逾時 fallback：等太久仍未全部封裝完成就強制 quit。"""
    print("\n[WARNING] 等待影片封裝逾時，強制退出所有管線！")
    if g_loop and g_loop.is_running():
        g_loop.quit()
    return False


def _send_eos_to_all():
    """對所有 pipeline 送 EOS（安全結束、等影片封裝），並設逾時保底。"""
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
    EOS   → 記錄此 pipeline 已結束；所有 pipeline 都 EOS 才 quit mainloop。
    ERROR → RTSP 來源不穩只警告（保持運行等重連）；其餘嚴重錯誤 → 直接退出。
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
    """放大 queue 容量（用於容易累積的節點，如 analytics 前）。"""
    q.set_property("max-size-buffers", max_buffers)
    q.set_property("max-size-bytes", 0)
    q.set_property("max-size-time", 0)


def build_group_pipeline(group_id, group):
    """
    為「單一組」建立一條完整、獨立的 Gst.Pipeline，回傳 (pipeline, pipeline_name)。
    group 內含 weight / engine_path / class_map / member_uids 等（見 config._build_groups）。
    """
    members = group["member_uids"]          # 全域 uid，index 即該組區域 pad_index
    num = len(members)
    gname = f"traffic-pipeline-g{group_id}"
    pipeline = Gst.Pipeline.new(gname)

    # ---- streammux（主線）----
    streammux = make_elm("nvstreammux", f"Stream-muxer-g{group_id}")
    streammux.set_property("batch-size", num)          # 該組成員數（新舊版 mux 皆支援）
    if os.environ.get("USE_NEW_NVSTREAMMUX") == "yes":
        # 新版 mux：不接受 width/height/live-source 等舊屬性，改讀 config_mux.txt
        _mux_cfg = os.path.join(BASE_DIR, "config_mux.txt")
        if os.path.exists(_mux_cfg):
            streammux.set_property("config-file-path", _mux_cfg)
        else:
            print(f"[WARNING] 找不到 {_mux_cfg}，新版 mux 將用內建預設值（請先跑 traffic_count_txt.py）")
    else:
        # 舊版 mux：維持原本設定
        streammux.set_property("width", 1920)
        streammux.set_property("height", 1080)
        streammux.set_property("batched-push-timeout", 16666)
        streammux.set_property("live-source", 1)
        streammux.set_property("nvbuf-memory-type", 0)
    pipeline.add(streammux)

    # ---- 每個成員 cam 一個 nvurisrcbin（RTSP 路啟用內建斷線自動重連）----
    for local_idx, uid in enumerate(members):
        cfg = SOURCE_CONFIGS[uid]
        # 建立 + 屬性 + 訊號統一走 _create_source_element（初次與看門狗「重建」共用，設定保證一致）
        src = _create_source_element(uid, local_idx, cfg, streammux)
        pipeline.add(src)

        is_live = not cfg.get("is_file_source", False)
        if is_live:
            print(f"[INFO] {cfg.get('source_id', uid)} 為即時串流：啟用自動重連（5s 間隔、無限重試）")
            # ⭐ 看門狗：只登記「即時串流」路（檔案來源播完不吐幀是正常現象，
            #    絕不能重啟——否則影片會從頭重播、DB 重複計數）。cfg/pipeline 供「重建」用。
            g_sources[uid] = {"src": src, "streammux": streammux, "pad_index": local_idx,
                              "pipeline": pipeline, "cfg": cfg, "rebuilds": 0}

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
        tracker.set_property("ll-lib-file", resolve_tracker_lib())
        tracker.set_property("tracker-width", 640)
        tracker.set_property("tracker-height", 384)

    elems = [q1, preprocess, q2, pgie, q3, q_analytics, analytics, q4]
    if tracker is not None:
        elems.append(tracker)
    for e in elems:
        pipeline.add(e)

    # streammux → q1 → preprocess → q2 → pgie → q3 →（nvdcf 時經 tracker）→ analytics → q4
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

def _create_source_element(uid, local_idx, cfg, streammux):
    """
    建立並設定一顆 nvurisrcbin（URI、RTSP 重連屬性、pad-added / child-added 訊號）。
    不加入 pipeline、不設狀態——由呼叫端處理。初次建立與看門狗「重建」共用，設定保證一致。
    命名帶重建代數（-rN）避免與舊元件撞名。
    """
    n = 0
    info = g_sources.get(uid)
    if info:
        n = info.get("rebuilds", 0)
    name = f"uri-decode-bin-{uid}" + (f"-r{n}" if n else "")
    src = make_elm("nvurisrcbin", name)
    src.set_property("uri", cfg["source"])
    if not cfg.get("is_file_source", False):
        # source-id：讓 nvurisrcbin 內建訊息（Resetting source N）能辨識是哪一路（不設會印 -1）
        _safe_set(src, "source-id", uid)
        _safe_set(src, "rtsp-reconnect-interval", 5)    # 連續 5 秒收不到資料就重連
        _safe_set(src, "rtsp-reconnect-attempts", -1)   # -1 = 無限重試，永不放棄
        _safe_set(src, "select-rtp-protocol", 4)        # 4 = 強制 TCP
        _safe_set(src, "latency", 200)                  # 抖動緩衝 200ms
        _safe_set(src, "udp-buffer-size", 2000000)
    src.connect("pad-added", cb_newpad, {"streammux": streammux, "pad_index": local_idx})
    src.connect("child-added", cb_decodebin_child_added, None)
    return src


def _restart_one_source(uid):
    """單獨重啟某一路 nvurisrcbin：斷開舊 pad → NULL → release streammux sink → PLAYING。
    在主線程執行（由 GLib.idle_add 呼叫），不影響其他路。"""
    info = g_sources.get(uid)
    if not info:
        return False
    if g_eos_triggered:      # 已在收尾流程 → 不再重啟，避免干擾影片封裝
        return False
    src = info["src"]
    streammux = info["streammux"]
    pad_index = info["pad_index"]
    cam = SOURCE_CONFIGS.get(uid, {}).get("source_id", f"uid{uid}")

    g_restart_counts[uid] = g_restart_counts.get(uid, 0) + 1
    n = g_restart_counts[uid]
    print(f"[WATCHDOG] 重建 {cam}（uid={uid}） — 連續第 {n} 次（移除舊元件、建立全新 RTSP session）")
    try:
        # 1) 斷開 streammux 的 sink pad（若還連著）
        sinkpad = streammux.get_static_pad(f"sink_{pad_index}")
        if sinkpad is not None:
            peer = sinkpad.get_peer()
            if peer is not None:
                peer.unlink(sinkpad)
            # 官方 runtime_source_add_delete 作法：release 前先對 mux sink pad 送 flush_stop，
            # 清掉 pad 上殘留的 flushing/sticky 狀態，之後重新 request 同名 sink_N 才是乾淨的
            sinkpad.send_event(Gst.Event.new_flush_stop(False))
            # release 掉 request pad，讓重啟後 cb_newpad 能重新接
            streammux.release_request_pad(sinkpad)

        # 2) 該路 source 設 NULL
        src.set_state(Gst.State.NULL)
        src.get_state(Gst.CLOCK_TIME_NONE)  # 等狀態確實切到 NULL

        # 3) 把舊元件整顆移出該組 pipeline，建立「全新」nvurisrcbin。
        #    全新元件 = 全新 rtspsrc / 全新 TCP 連線 / 無殘留 ghost pad / 內部狀態歸零，
        #    等效手動關掉播放器重開。實測證明：同一顆元件 NULL→PLAYING 的「重啟」會撞
        #    自己殘留的 vsrc_0 ghost pad（pad 不隨狀態切換移除）而接不回來，故一律重建。
        grp_pipeline = info["pipeline"]
        grp_pipeline.remove(src)
        info["rebuilds"] = info.get("rebuilds", 0) + 1
        new_src = _create_source_element(uid, pad_index, info["cfg"], streammux)
        grp_pipeline.add(new_src)
        new_src.sync_state_with_parent()
        info["src"] = new_src
        g_last_restart[uid] = time.time()
        print(f"[WATCHDOG] {cam} 已重建為全新元件（{new_src.get_name()}），等待重新連線...")
    except Exception as e:
        print(f"[WATCHDOG] 重建 {cam} 發生例外: {e}")
    return False  # 給 idle_add 用，只跑一次


def _watchdog_check():
    """每 WATCHDOG_CHECK_SEC 秒檢查各「RTSP 路」最後吐幀時間，卡死超過門檻就單路重啟。
    只監控 g_sources 內的路（建立來源時只收錄即時串流，檔案來源不在其中）。
    EOS 觸發（按 Q / SIGINT / SIGTERM）後回傳 False 停止本 timer，不干擾收尾封裝。"""
    if g_eos_triggered:
        print("[WATCHDOG] 偵測到 EOS 收尾流程，看門狗停止")
        return False
    now = time.time()
    for uid in list(g_sources.keys()):
        # 重建寬限（自適應）：重建後「有進過幀」= 恢復中 → 給滿 GRACE_SEC 觀察；
        # 「連一幀都沒進來」= 對端沒回應 → 只等 GRACE_FAST 就再重建，縮短失敗循環
        last_rs = g_last_restart.get(uid, 0)
        stats = fps_streams.get(uid, {})
        ts = stats.get("timestamps")
        grace = WATCHDOG_GRACE_SEC if (ts and ts[-1] > last_rs) else WATCHDOG_GRACE_FAST
        if now - last_rs < grace:
            continue

        if not ts:
            # 還沒收過任何幀（可能剛啟動），跳過
            continue

        idle = now - ts[-1]
        cam = SOURCE_CONFIGS.get(uid, {}).get("source_id", f"uid{uid}")

        # --- 滴幀偵測：內建重連「半成功」時每次會滴進幾幀，idle 一直被重置，
        #     單看 idle 永遠不會達標。改看觀察窗平均 FPS：最近 len(ts) 幀共花 span 秒，
        #     健康 15fps 串流 30 幀約 2 秒；若 span 已超過卡死門檻仍湊不滿，就是滴幀卡死。
        span = now - ts[0]
        rate = (len(ts) / span) if span > 0 else 999.0
        trickle = (span >= WATCHDOG_STALL_SEC and rate < WATCHDOG_MIN_FPS)
        stalled = (idle >= WATCHDOG_STALL_SEC) or trickle

        # --- 具名狀態回報：一眼看出是哪一路在卡、卡多久、通知第幾次 ---
        if idle >= WATCHDOG_NOTIFY_SEC or trickle:
            g_stall_notify[uid] = g_stall_notify.get(uid, 0) + 1
            desc = (f"已 {idle:.0f}s 無新幀" if idle >= WATCHDOG_NOTIFY_SEC
                    else f"滴幀中（近 {span:.0f}s 平均僅 {rate:.2f} FPS）")
            print(f"[{cam}] {desc}（nvurisrcbin 內建重連中）— 第 {g_stall_notify[uid]} 次通知")
        elif g_stall_notify.get(uid, 0) > 0:
            print(f"[{cam}] ✅ 已恢復吐幀（先前通知 {g_stall_notify[uid]} 次、"
                  f"看門狗重建 {g_restart_counts.get(uid, 0)} 次）")
            g_stall_notify[uid] = 0
            g_restart_counts[uid] = 0
            g_sources[uid]["rebuilds"] = 0

        if stalled:
            reason = f"已 {idle:.0f} 秒無新幀" if idle >= WATCHDOG_STALL_SEC else f"滴幀（近 {span:.0f}s 平均 {rate:.2f} FPS）"
            print(f"[WATCHDOG] {cam}（uid={uid}）{reason}，判定卡死 → 移除重建")
            GLib.idle_add(_restart_one_source, uid)

    return True  # 回 True 讓 timer 持續

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

        # 啟動看門狗：只有存在 RTSP 路時才需要（檔案批次跑不啟動、也不該監控）
        if g_sources:
            GLib.timeout_add_seconds(WATCHDOG_CHECK_SEC, _watchdog_check)
            print(f"[INFO] 看門狗啟動：監控 {len(g_sources)} 路即時串流，"
                  f"每 {WATCHDOG_CHECK_SEC}s 檢查，卡死門檻 {WATCHDOG_STALL_SEC}s，寬限 有幀{WATCHDOG_GRACE_SEC}s/無幀{WATCHDOG_GRACE_FAST}s")

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

# logic/pipeline.py
"""
pipeline.py
-----------
GStreamer pipeline 元件建構與分支邏輯（多權重 / 多 pipeline 版）。

平台自動適配：
    以 _is_jetson() 判斷平台（Jetson 或 dGPU/WSL2），在「顯示 sink、OSD 處理模式、
    NVENC 編碼器屬性」上自動選用該平台支援的設定，使同一份程式碼於兩種環境皆可執行。
    寫檔編碼器採延遲偵測：預設 NVENC，建不出時自動退回 CPU x264（USE_CPU_ENCODER 可覆寫）。
    nvtracker 的 ll-lib-file 由 resolve_tracker_lib() 自動解析（DS_TRACKER_LIB 可覆寫）。

分組觀念：
    依 weight 分組，每組跑一條獨立 pipeline。
    - 區域 pad_index：某條 pipeline 內 streammux/demux 的 sink_/src_ 編號（各組從 0 起算）
    - 全域 stream_uid：跨所有組唯一的編號（= SOURCE_CONFIGS 的鍵）

每路 cam 的下游分支由 YAML 決定：
    - output.save_output_video → 寫檔分支（存成 mp4）
    - display.show_window      → 本地預覽分支（併入該組自己的 tile 視窗）
"""

import sys
import os
import platform
from gi.repository import Gst


def _is_jetson():
    """判斷是否為 Jetson（aarch64 或存在 /etc/nv_tegra_release）；dGPU / WSL2 回傳 False。"""
    return (platform.machine() == "aarch64") or os.path.isfile("/etc/nv_tegra_release")


def _safe_set(elm, name, value):
    """僅在元件確實具有該屬性時才設定，避免跨平台屬性差異造成例外。回傳是否設定成功。"""
    if elm.find_property(name) is not None:
        elm.set_property(name, value)
        return True
    return False


# ==========================================
# 編碼器選擇（延遲偵測）與追蹤器路徑解析
# ==========================================

def _detect_cpu_encoder():
    """
    決定是否使用 CPU 軟體編碼器（x264）。

    規則（可用環境變數 USE_CPU_ENCODER 覆寫，1/true=強制 CPU，0/false=強制 NVENC）：
      - 環境變數有明確指定 → 依指定
      - 否則：預設優先使用 NVENC 硬體編碼（較快）；只有在「確實建不出 NVENC」時才退回 CPU，
              避免在無 NVENC 的環境（如部分 WSL）開存檔就直接中斷。
    """
    env = os.environ.get("USE_CPU_ENCODER")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    # 預設 = NVENC。用「實際建立元件」測試（比 ElementFactory.find 更可靠）
    test = Gst.ElementFactory.make("nvv4l2h264enc", None)
    if test is not None:
        print("[INFO] 預設使用 NVENC 硬體編碼（nvv4l2h264enc）")
        return False
    print("[INFO] 建不出 NVENC，退回 CPU 軟體編碼（x264）")
    return True


# 編碼器選擇「延遲判斷」：不在 import 當下決定，而是等第一次真正要建編碼器時才判斷並快取。
# 原因：main.py 是先 import 本模組、之後才呼叫 Gst.init(None)。若在 import 當下判斷，
# 會早於 Gst.init()，此時 GStreamer 尚未初始化、抓不到 NVENC，導致誤退 CPU。
_USE_CPU_ENCODER = None   # None=尚未判斷；True/False=已快取


def use_cpu_encoder():
    """回傳是否使用 CPU 編碼；第一次呼叫時才判斷並快取（此時 Gst.init() 已完成，能正確抓到 NVENC）。"""
    global _USE_CPU_ENCODER
    if _USE_CPU_ENCODER is None:
        _USE_CPU_ENCODER = _detect_cpu_encoder()
    return _USE_CPU_ENCODER


def resolve_tracker_lib():
    """
    自動解析 nvtracker 的 ll-lib-file 路徑（跨平台 / 跨機器）。

    順序：
      1. 環境變數 DS_TRACKER_LIB（若指定且存在）
      2. 依序搜尋常見安裝路徑（含 glob 掃版本號），回傳第一個存在的
      3. 都找不到 → 回傳標準 NVIDIA 路徑（讓 DS 自行報錯提示）
    """
    env = os.environ.get("DS_TRACKER_LIB", "").strip()
    if env and os.path.exists(env):
        return env
    import glob as _glob
    candidates = [
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        "/opt/thi/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
    ]
    candidates += sorted(_glob.glob(
        "/opt/nvidia/deepstream/deepstream*/lib/libnvds_nvmultiobjecttracker.so"))
    for p in candidates:
        if os.path.exists(p):
            return p
    return "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"


# ==========================================
# uridecodebin 動態 pad 回呼
# ==========================================
def cb_newpad(decodebin, decoder_src_pad, data):
    """
    nvurisrcbin 解出動態 pad 時的處理：把 video pad 接到該組 streammux 的 sink_{pad_index}。
    非 video pad（音訊等）接 fakesink 消化。

    重啟支援：若該 sink_{pad_index} 已存在但目前未 link（代表是單路重啟後重新吐 pad），
    直接重新接上，讓重啟的那一路乾淨接回 streammux。

    data 需含："streammux"、"pad_index"。
    """
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()

    is_video = caps.get_structure(0).get_name().find("video") != -1
    streammux = data["streammux"]
    pipeline = streammux.get_parent()
    pad_name = f"sink_{data['pad_index']}"

    # 先取回既有 request pad；沒有才 request 新的
    sinkpad = streammux.get_static_pad(pad_name)
    if sinkpad is None:
        sinkpad = streammux.get_request_pad(pad_name)

    # 非影像流 → 導到 fakesink 消化
    if not is_video or sinkpad is None:
        _drain_pad_to_fakesink(pipeline, decoder_src_pad)
        return

    # 若該 sink pad 已被別的 pad 佔著（linked），才導 fakesink；
    # 若未 link（首次接、或重啟後重接）→ 直接接上
    if sinkpad.is_linked():
        _drain_pad_to_fakesink(pipeline, decoder_src_pad)
        return

    decoder_src_pad.link(sinkpad)


def _drain_pad_to_fakesink(pipeline, src_pad):
    """把不需要的 pad（音訊 / 多餘視訊）接到 fakesink 消化，避免未連結 pad 造成反壓卡整路。"""
    fs = Gst.ElementFactory.make("fakesink", None)  # None=自動命名，避免多路撞名
    if fs is None:
        return
    fs.set_property("sync", False)
    fs.set_property("async", False)
    pipeline.add(fs)
    fs.sync_state_with_parent()
    src_pad.link(fs.get_static_pad("sink"))


def cb_decodebin_child_added(child_proxy, obj, name, user_data):
    """
    nvurisrcbin 內部子元件建立時的回呼（取代原本的 cb_source_setup）。
      - 遞迴掛到內層 decodebin，才能抓到最底層的 rtspsrc。
      - 對內部 rtspsrc 強制 TCP、設定連線逾時與抖動緩衝。
    僅在該元件確實有對應屬性時才設定，避免跨平台/版本差異造成例外。
    """
    # 內層還有 decodebin 時，繼續往下掛，才追得到 rtspsrc
    if name.find("decodebin") != -1:
        obj.connect("child-added", cb_decodebin_child_added, user_data)

    # 抓到內部 source（rtspsrc）→ 套用與原本相同的調校
    if name.find("source") != -1:
        if obj.find_property("protocols") is not None:
            obj.set_property("protocols", 4)          # 4 = TCP
        if obj.find_property("timeout") is not None:
            obj.set_property("timeout", 5000000)      # 連線逾時 5 秒（微秒）
        if obj.find_property("drop-on-latency") is not None:
            obj.set_property("drop-on-latency", True) # 超過緩衝丟舊幀
        if obj.find_property("latency") is not None:
            obj.set_property("latency", 200)          # 抖動緩衝 200ms


def make_elm(gst_type, name):
    """建立單一 GStreamer 元件，失敗則整支程式退出並提示型別/名稱。"""
    elm = Gst.ElementFactory.make(gst_type, name)
    if not elm:
        sys.exit(f"[ERROR] 無法建立 element: {gst_type} ({name})")
    return elm


# ==========================================
# 自動依成員數決定 tile 佈局
# ==========================================
def _get_tile_layout(num_sources):
    """依「該組成員數」回傳 (rows, cols, total_width, total_height)，每格 16:9。"""
    if num_sources == 1:
        rows, cols = 1, 1
    elif num_sources == 2:
        rows, cols = 1, 2
    elif num_sources <= 4:
        rows, cols = 2, 2
    elif num_sources <= 6:
        rows, cols = 2, 3
    elif num_sources <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, 4

    total_width  = 1920
    cell_w       = total_width // cols
    cell_h       = int(cell_w * 9 / 16)
    total_height = cell_h * rows
    return rows, cols, total_width, total_height


# ==========================================
# 寫檔分支（檔案 / RTSP 來源共用）
# ==========================================
def _build_save_branch(pipeline, uid, video_path, source_fps):
    """
    寫檔分支：nvvideoconvert → videorate → capsfilter(N/1) → 編碼器
             → h264parse → qtmux → filesink。
    videorate 以固定 framerate 重打 PTS，避免 live-source 造成 PTS 不規律使 qtmux 拒收。
    編碼器依 use_cpu_encoder() 自動選擇：
      NVENC（預設）：caps 走 NVMM NV12 → nvv4l2h264enc（Jetson/dGPU 屬性以 _safe_set 兼容）。
      CPU（無 NVENC 時退路）：caps 走系統記憶體 I420 → x264enc（bitrate 單位 kbps）。
    回傳分支起點 element（供上游 link）。
    """
    i = uid

    nvvidconv_s = make_elm("nvvideoconvert", f"convertor-save-{i}")
    nvvidconv_s.set_property("nvbuf-memory-type", 0)

    videorate = make_elm("videorate", f"videorate-save-{i}")

    cap_filter = make_elm("capsfilter", f"cap-filter-save-{i}")
    fps_int = int(round(source_fps)) if source_fps and source_fps > 0 else 30
    cpu_enc = use_cpu_encoder()
    caps_base = ("video/x-raw, format=I420" if cpu_enc
                 else "video/x-raw(memory:NVMM), format=NV12")
    cap_filter.set_property(
        "caps",
        Gst.Caps.from_string(f"{caps_base}, framerate={fps_int}/1"),
    )

    if cpu_enc:
        encoder = make_elm("x264enc", f"encoder-{i}")
        _safe_set(encoder, "bitrate", 4000)      # x264enc 單位是 kbps
        _safe_set(encoder, "speed-preset", 1)    # 1=ultrafast，吞吐優先
        _safe_set(encoder, "tune", 4)            # 4=zerolatency
        _safe_set(encoder, "key-int-max", fps_int)
    else:
        encoder = make_elm("nvv4l2h264enc", f"encoder-{i}")
        _safe_set(encoder, "bitrate", 4000000)
        _safe_set(encoder, "profile", 0)
        _safe_set(encoder, "iframeinterval", fps_int)
        # preset-level / insert-sps-pps / maxperf-enable 為 Jetson NVENC 專有；
        # 若一個都設不成功，代表在 dGPU/WSL2，改設 dGPU NVENC 的調校屬性。
        if not (_safe_set(encoder, "preset-level", 1)
                | _safe_set(encoder, "insert-sps-pps", 1)
                | _safe_set(encoder, "maxperf-enable", 1)):
            _safe_set(encoder, "preset-id", 1)
            _safe_set(encoder, "tuning-info-id", 2)

    parser = make_elm("h264parse", f"h264-parser-{i}")
    muxer = make_elm("matroskamux", f"muxer-{i}")
    filesink = make_elm("filesink", f"filesink-{i}")
    filesink.set_property("location", video_path)
    filesink.set_property("async", False)
    filesink.set_property("sync", False)

    for elm in [nvvidconv_s, videorate, cap_filter, encoder, parser, muxer, filesink]:
        pipeline.add(elm)

    nvvidconv_s.link(videorate)
    videorate.link(cap_filter)
    cap_filter.link(encoder)
    encoder.link(parser)
    parser.link(muxer)
    muxer.link(filesink)

    return nvvidconv_s


# ==========================================
# 本地顯示分支（該組自己的 tile 預覽視窗）
# ==========================================
def _build_display_sink(pipeline, group_id, num_members):
    """
    為「單一組」建立一個 tile 預覽視窗：
        streammux(顯示用) → tiler → nvvideoconvert →（依平台選）顯示 sink
    回傳顯示用 streammux（該組各路顯示分支接它的 sink_{local_idx}）。

    顯示 sink 依平台自動選擇：
      - 有 NVIDIA sink（nveglglessink / nv3dsink）→ 走 NVMM；Jetson 另加 nvegltransform。
      - 否則（dGPU / WSL2 常見）→ 用 nvvideoconvert 轉出系統記憶體，交給標準 GStreamer sink。
        可用環境變數 DS_DISPLAY_SINK 指定要用哪個標準 sink。
    """
    rows, cols, total_w, total_h = _get_tile_layout(num_members)
    print(f"[INFO] [group{group_id}] Tile 佈局: {rows}x{cols}, 視窗尺寸: {total_w}x{total_h}")

    g = group_id
    streammux2 = make_elm("nvstreammux", f"Stream-muxer-display-g{g}")
    streammux2.set_property("width", 1920)
    streammux2.set_property("height", 1080)
    streammux2.set_property("batch-size", num_members)
    streammux2.set_property("batched-push-timeout", 16666)
    streammux2.set_property("live-source", 1)
    streammux2.set_property("nvbuf-memory-type", 0)

    tiler = make_elm("nvmultistreamtiler", f"nvtiler-display-g{g}")
    tiler.set_property("rows", rows)
    tiler.set_property("columns", cols)
    tiler.set_property("width", total_w)
    tiler.set_property("height", total_h)

    q_d1 = make_elm("queue", f"q-display-1-g{g}")
    nvvidconv = make_elm("nvvideoconvert", f"convertor-display-g{g}")
    nvvidconv.set_property("nvbuf-memory-type", 0)
    q_d2 = make_elm("queue", f"q-display-2-g{g}")
    q_d3 = make_elm("queue", f"q-display-3-g{g}")

    # 先找 NVIDIA 專用 sink（吃 NVMM，可直送）；dGPU/WSL2 通常沒有，回傳 None
    nv_sink = None
    if Gst.ElementFactory.find("nveglglessink") is not None:
        nv_sink = make_elm("nveglglessink", f"nvvideo-renderer-display-g{g}")
    elif Gst.ElementFactory.find("nv3dsink") is not None:
        nv_sink = make_elm("nv3dsink", f"nvvideo-renderer-display-g{g}")

    if nv_sink is not None:
        # ---- 路徑 A：NVIDIA 專用 sink（走 NVMM，Jetson 視情況加 nvegltransform）----
        sink = nv_sink
        sink.set_property("sync", False)
        _safe_set(sink, "qos", False)
        use_egltransform = _is_jetson() and (Gst.ElementFactory.find("nvegltransform") is not None)
        if use_egltransform:
            transform = make_elm("nvegltransform", f"nvegl-transform-display-g{g}")
            elements = [streammux2, tiler, q_d1, nvvidconv, q_d2, transform, q_d3, sink]
        else:
            elements = [streammux2, tiler, q_d1, nvvidconv, q_d2, q_d3, sink]
        for elm in elements:
            pipeline.add(elm)
        streammux2.link(tiler)
        tiler.link(q_d1)
        q_d1.link(nvvidconv)
        nvvidconv.link(q_d2)
        if use_egltransform:
            q_d2.link(transform)
            transform.link(q_d3)
        else:
            q_d2.link(q_d3)
        q_d3.link(sink)
    else:
        # ---- 路徑 B：標準 GStreamer sink（dGPU / WSL2）----
        # nvvideoconvert 先把畫面從 NVMM 轉到系統記憶體，再交給標準 sink 顯示
        caps_sys = make_elm("capsfilter", f"caps-display-sys-g{g}")
        caps_sys.set_property("caps", Gst.Caps.from_string("video/x-raw, format=RGBA"))
        videoconv = make_elm("videoconvert", f"videoconvert-display-g{g}")

        # 未指定 DS_DISPLAY_SINK 時，依 WSLg 穩定度排序自動挑選可用 sink
        forced = os.environ.get("DS_DISPLAY_SINK", "").strip()
        candidates = [forced] if forced else ["ximagesink", "glimagesink", "autovideosink"]
        std_sink = None
        for cand in candidates:
            if cand and Gst.ElementFactory.find(cand) is not None:
                std_sink = make_elm(cand, f"nvvideo-renderer-display-g{g}")
                print(f"[INFO] [group{g}] 使用標準顯示 sink：{cand}")
                break
        if std_sink is None:
            sys.exit(f"[ERROR] 找不到可用的顯示 sink（嘗試清單：{candidates}）")
        _safe_set(std_sink, "sync", False)
        sink = std_sink

        elements = [streammux2, tiler, q_d1, nvvidconv, caps_sys, videoconv, q_d2, sink]
        for elm in elements:
            pipeline.add(elm)
        streammux2.link(tiler)
        tiler.link(q_d1)
        q_d1.link(nvvidconv)
        nvvidconv.link(caps_sys)
        caps_sys.link(videoconv)
        videoconv.link(q_d2)
        q_d2.link(sink)

    return streammux2


# ==========================================
# 每路 cam 分支組裝（save / show）
# ==========================================
def setup_cam_branch(pipeline, local_idx, uid, cfg, demux, display_streammux, osd_probe_callback):
    """
    為單路 cam 建立下游分支：
        demux.src_{local_idx} → queue → nvvideoconvert(RGBA NVMM) → nvdsosd
                              → 依 YAML 接 顯示 / 存檔（可同時，多者用 tee 分流）
    並在 nvdsosd.sink 掛 OSD 探針（疊 FPS / ROI）。元件命名用全域 uid，跨 pipeline 不撞名。
    """
    src_pad = demux.get_request_pad(f"src_{local_idx}")

    q_cam = make_elm("queue", f"q-cam-{uid}")
    nvvidconv_osd = make_elm("nvvideoconvert", f"conv_osd_{uid}")
    nvvidconv_osd.set_property("nvbuf-memory-type", 0)
    caps_osd = make_elm("capsfilter", f"caps_osd_{uid}")
    caps_osd.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    nvosd_i = make_elm("nvdsosd", f"nvosd-{uid}")
    # process-mode：Jetson 用 2（VIC 硬體），dGPU / WSL2 用 1（GPU）
    nvosd_i.set_property("process-mode", 2 if _is_jetson() else 1)

    for elm in [q_cam, nvvidconv_osd, caps_osd, nvosd_i]:
        pipeline.add(elm)

    src_pad.link(q_cam.get_static_pad("sink"))
    q_cam.link(nvvidconv_osd)
    nvvidconv_osd.link(caps_osd)
    caps_osd.link(nvosd_i)

    # OSD 探針：用全域 uid 綁定，回呼才知道畫哪一路的 FPS
    nvosd_i.get_static_pad("sink").add_probe(
        Gst.PadProbeType.BUFFER,
        lambda pad, info, u=uid: osd_probe_callback(pad, info, u),
        0
    )

    cam_save = cfg.get("output", {}).get("save_output_video", False)
    cam_show = cfg.get("display", {}).get("show_window", True)
    enabled_branches = sum([cam_save, cam_show])

    # 0 個分支：fakesink 收尾
    if enabled_branches == 0:
        fake = make_elm("fakesink", f"fake-{uid}")
        fake.set_property("sync", False)
        fake.set_property("async", False)
        pipeline.add(fake)
        nvosd_i.link(fake)
        return

    # 1 個分支：直接 link
    if enabled_branches == 1:
        if cam_save:
            nvosd_i.link(_build_save_branch(pipeline, uid, cfg["video_path"], cfg["stream_fps"]))
        else:
            _link_show_branch(pipeline, local_idx, uid, nvosd_i, display_streammux)
        return

    # 2 個分支：tee 分流
    tee = make_elm("tee", f"tee-{uid}")
    pipeline.add(tee)
    nvosd_i.link(tee)

    if cam_save:
        q_s = make_elm("queue", f"q-s-{uid}")
        pipeline.add(q_s)
        tee.link(q_s)
        q_s.link(_build_save_branch(pipeline, uid, cfg["video_path"], cfg["stream_fps"]))

    if cam_show:
        _link_show_branch(pipeline, local_idx, uid, tee, display_streammux)


def _link_show_branch(pipeline, local_idx, uid, upstream, display_streammux):
    """
    顯示子分支：upstream → queue → nvvideoconvert → display_streammux.sink_{local_idx}
    元件命名用 uid（跨 pipeline 不撞名）；顯示 mux 的 pad 用區域 local_idx。
    """
    q_d  = make_elm("queue", f"q-d-{uid}")
    nv_d = make_elm("nvvideoconvert", f"nv-d-{uid}")
    nv_d.set_property("nvbuf-memory-type", 0)

    pipeline.add(q_d)
    pipeline.add(nv_d)

    upstream.link(q_d)
    q_d.link(nv_d)
    nv_d.get_static_pad("src").link(display_streammux.get_request_pad(f"sink_{local_idx}"))

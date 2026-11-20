"""
GStreamer 元件建構：小工具、每路攝影機的下游分支、預覽視窗、存檔

一路攝影機在 demux 之後的分支（setup_cam_branch）：

    demux.src_{pad_index} → queue → nvvideoconvert(RGBA) → nvdsosd ─┬→ 存檔分支
                                                                     └→ 顯示分支
    兩者都關 → 接 fakesink；兩者都開 → 用 tee 分流。

    存檔分支  nvvideoconvert → videorate → nvv4l2h264enc（或 x264enc）→ h264parse → matroskamux → filesink
    顯示分支  queue → nvvideoconvert → 該組的顯示用 streammux → tiler → 視窗
              同一組的攝影機併成一個 tile 視窗，一組一個視窗。

平台差異（Jetson 與桌機 dGPU / WSL2 共用同一份程式）：
    屬性一律透過 safe_set 設定，該平台沒有的屬性自動略過；
    顯示 sink、nvdsosd 的 process-mode、存檔編碼器依 is_jetson() 與實際存在的元件挑選。

queue 的容量規則：
    所有 queue 都是 leaky=downstream（滿了丟最舊的），目的是下游變慢時丟幀而不是反壓上游。
    但 leaky 只有在「queue 容量 < 上游元件的 buffer pool」時才會生效：
    nvstreammux / nvvideoconvert / tiler 的 pool 預設只有 4 個 buffer，
    容量比 4 大的 queue 還沒滿，上游就先因拿不到空 buffer 而停住，結果仍是反壓。
    因此顯示分支的 queue 容量為 DISPLAY_QUEUE_BUFFERS（2），
    並關閉顯示分支 nvvideoconvert 的直通模式（否則上游的 buffer 會被顯示用 streammux 佔住）。
    如此視窗卡住時只會丟顯示用的畫面，推論、計數、存檔照常進行。
"""

import os
import sys
import platform

from gi.repository import Gst

from logic.common import MUX_OUTPUT_W, MUX_OUTPUT_H


# =============================================================================
# 常數與狀態
# =============================================================================

DEFAULT_QUEUE_BUFFERS = 30      # 一般 queue 的容量（buffer 數）；只用來吸收短暫的處理時間差
DISPLAY_QUEUE_BUFFERS = 2       # 顯示分支 queue 的容量，必須小於 4（見檔頭）
MUX_BATCH_TIMEOUT_USEC = 16666  # streammux 湊不滿一個 batch 時最多等多久（微秒，約 1/60 秒）
DISPLAY_WIDTH = 1920            # 預覽視窗總寬（像素）；高度依格數與 16:9 計算
SAVE_BITRATE_KBPS = 4000        # 存檔位元率（kbps），NVENC 與 x264 共用

ALL_QUEUES = []                 # 所有 make_queue 建立的 queue；main.py 的卡住診斷用來列出水位

_use_cpu_encoder = None         # None = 尚未判斷；之後快取 True / False，見 use_cpu_encoder()


# =============================================================================
# 小工具
# =============================================================================

def is_jetson():
    """是否在 Jetson 上執行；桌機 dGPU / WSL2 回傳 False。"""
    return platform.machine() == "aarch64" or os.path.isfile("/etc/nv_tegra_release")


def safe_set(element, name, value):
    """
    元件有這個屬性才設定，用來吸收平台 / 版本之間的屬性差異。

    回傳：
        是否有設定
    """
    if element.find_property(name) is None:
        return False
    element.set_property(name, value)
    return True


def make_elm(factory, name):
    """建立元件；缺少對應的 plugin 時直接結束程式。"""
    element = Gst.ElementFactory.make(factory, name)
    if not element:
        sys.exit(f"[ERROR] 無法建立元件 {factory}（{name}），請確認對應的 GStreamer plugin 已安裝")
    return element


def make_queue(name, max_buffers=DEFAULT_QUEUE_BUFFERS):
    """
    建立 leaky queue：滿了丟最舊的 buffer。

    參數：
        max_buffers  容量（buffer 數）；選擇方式見檔頭「queue 的容量規則」
    """
    queue = make_elm("queue", name)
    queue.set_property("leaky", 2)              # 2 = downstream，丟最舊的
    queue.set_property("max-size-buffers", max_buffers)
    queue.set_property("max-size-bytes", 0)     # 0 = 不以位元組數限制
    queue.set_property("max-size-time", 0)      # 0 = 不以時間長度限制
    ALL_QUEUES.append(queue)
    return queue


def make_convert(name):
    """建立 nvvideoconvert，輸出放在平台預設的 GPU 記憶體。"""
    convert = make_elm("nvvideoconvert", name)
    convert.set_property("nvbuf-memory-type", 0)    # 0 = 平台預設（桌機 device memory、Jetson surface array）
    return convert


def make_fakesink(name=None):
    """
    建立用來丟棄資料的 fakesink。

    參數：
        name  None 時由 GStreamer 自動命名；執行中動態建立的（例如重連時）用 None 避免撞名
    """
    fakesink = Gst.ElementFactory.make("fakesink", name)
    fakesink.set_property("sync", False)    # 不依時間戳等待，來了就丟
    fakesink.set_property("async", False)   # 不等它收到第一筆資料才切換 pipeline 狀態
    return fakesink


def make_streammux(name, batch_size):
    """
    建立 nvstreammux，輸出固定為 MUX_OUTPUT_W × MUX_OUTPUT_H。

    各路畫面一到就送出，不等其他路對齊時間戳，
    避免多路 RTSP 的 FPS / 時脈不一致時互相拖累。
    """
    mux = make_elm("nvstreammux", name)
    mux.set_property("width", MUX_OUTPUT_W)
    mux.set_property("height", MUX_OUTPUT_H)
    mux.set_property("batch-size", batch_size)  # = 該組攝影機數
    mux.set_property("batched-push-timeout", MUX_BATCH_TIMEOUT_USEC)
    mux.set_property("live-source", 1)          # 以即時來源的方式處理時間戳
    mux.set_property("nvbuf-memory-type", 0)    # 0 = 平台預設的 GPU 記憶體
    safe_set(mux, "sync-inputs", 0)             # 0 = 不依時間戳同步各路輸入
    safe_set(mux, "max-latency", 0)             # 0 = 不額外等待延遲的來源
    return mux


def _add_and_link(pipeline, elements):
    """把元件依序加入 pipeline 並串接。"""
    for element in elements:
        pipeline.add(element)
    for upstream, downstream in zip(elements, elements[1:]):
        upstream.link(downstream)


# =============================================================================
# 存檔分支
# =============================================================================

def use_cpu_encoder():
    """
    存檔要不要改用 CPU 軟體編碼（x264）。

    預設用 NVENC 硬體編碼；只有建不出 nvv4l2h264enc 時才退回 x264，
    避免在沒有 NVENC 的機器上開存檔就整支程式結束。
    環境變數 USE_CPU_ENCODER 可強制指定（1 / true = CPU，0 / false = NVENC）。
    第一次呼叫時才判斷（此時 Gst.init() 已完成，才查得到 plugin），結果快取。
    """
    global _use_cpu_encoder
    if _use_cpu_encoder is None:
        env = os.environ.get("USE_CPU_ENCODER")
        if env is not None:
            _use_cpu_encoder = env.strip().lower() in ("1", "true", "yes", "on")
        else:
            # 試建一個，建不出就退回 CPU
            _use_cpu_encoder = Gst.ElementFactory.make("nvv4l2h264enc", None) is None
        print(f"[INFO] 存檔編碼器：{'CPU x264' if _use_cpu_encoder else 'NVENC（nvv4l2h264enc）'}")
    return _use_cpu_encoder


def _make_encoder(uid, fps):
    """依 use_cpu_encoder() 建立 H.264 編碼器；兩者位元率相同、每秒一個 I 幀。"""
    if use_cpu_encoder():
        encoder = make_elm("x264enc", f"encoder-{uid}")
        safe_set(encoder, "bitrate", SAVE_BITRATE_KBPS)     # x264 的單位是 kbps
        safe_set(encoder, "speed-preset", 1)                # 1 = ultrafast，CPU 負擔最低
        safe_set(encoder, "tune", 4)                        # 4 = zerolatency，不囤幀
        safe_set(encoder, "key-int-max", fps)               # I 幀間隔（幀）
        return encoder

    encoder = make_elm("nvv4l2h264enc", f"encoder-{uid}")
    safe_set(encoder, "bitrate", SAVE_BITRATE_KBPS * 1000)  # NVENC 的單位是 bps
    safe_set(encoder, "profile", 0)                         # 0 = Baseline
    safe_set(encoder, "iframeinterval", fps)                # I 幀間隔（幀）
    # 以下三個只有 Jetson 編碼器才有；一個都設不上代表是桌機 NVENC，改設它的對應屬性
    if not (safe_set(encoder, "preset-level", 1)            # 1 = UltraFast
            | safe_set(encoder, "insert-sps-pps", 1)        # 每個 IDR 前附 SPS / PPS
            | safe_set(encoder, "maxperf-enable", 1)):      # 編碼器固定最高時脈
        safe_set(encoder, "preset-id", 1)                   # 1 = P1，最快
        safe_set(encoder, "tuning-info-id", 2)              # 2 = 低延遲
    return encoder


def _build_save_branch(pipeline, uid, cfg):
    """
    建立存檔分支，回傳分支的第一個元件（上游接它）。

    videorate 把畫面整理成固定 FPS，避免即時來源的時間戳不規律。
    容器用 mkv（matroskamux）：程式被強制中止、斷電時已寫入的部分仍可播放；
    mp4 要正常收到 EOS 才會寫入索引，沒寫到整個檔案就無法開啟。
    """
    fps = int(round(cfg["stream_fps"])) if cfg["stream_fps"] > 0 else 30    # 讀不到 FPS 時以 30 計
    # NVENC 吃 GPU 記憶體的 NV12；x264 吃系統記憶體的 I420，由前面的 nvvideoconvert 轉出
    caps_base = ("video/x-raw, format=I420" if use_cpu_encoder()
                 else "video/x-raw(memory:NVMM), format=NV12")

    convert = make_convert(f"convertor-save-{uid}")
    videorate = make_elm("videorate", f"videorate-save-{uid}")
    caps = make_elm("capsfilter", f"cap-filter-save-{uid}")
    caps.set_property("caps", Gst.Caps.from_string(f"{caps_base}, framerate={fps}/1"))

    encoder = _make_encoder(uid, fps)
    parser = make_elm("h264parse", f"h264-parser-{uid}")
    muxer = make_elm("matroskamux", f"muxer-{uid}")
    filesink = make_elm("filesink", f"filesink-{uid}")
    filesink.set_property("location", cfg["video_path"])
    filesink.set_property("async", False)                                   # 不等它就緒才切換 pipeline 狀態
    filesink.set_property("sync", False)                                    # 不依時間戳等待，有資料就寫

    _add_and_link(pipeline, [convert, videorate, caps, encoder, parser, muxer, filesink])
    return convert


# =============================================================================
# 預覽視窗（一組一個）
# =============================================================================

def _tile_layout(count):
    """
    依攝影機數決定 tile 排列。

    回傳：
        (rows, cols, 視窗寬, 視窗高)；總寬固定 DISPLAY_WIDTH，每格 16:9
    """
    if count <= 1:
        rows, cols = 1, 1
    elif count == 2:
        rows, cols = 1, 2
    elif count <= 4:
        rows, cols = 2, 2
    elif count <= 6:
        rows, cols = 2, 3
    elif count <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, 4   # 最多 16 格，超過的不顯示
    cell_w = DISPLAY_WIDTH // cols
    return rows, cols, DISPLAY_WIDTH, (cell_w * 9 // 16) * rows


def build_display_sink(pipeline, group_id, member_count):
    """
    建立該組的預覽視窗。

        顯示用 streammux → tiler → queue → nvvideoconvert → … → 顯示 sink

    顯示 sink 的選擇：
        有 nveglglessink / nv3dsink → 直接吃 GPU 記憶體；
            Jetson 上 nveglglessink 前要加 nvegltransform
        都沒有 → 轉到系統記憶體交給一般 sink，依序嘗試 ximagesink、glimagesink、autovideosink
            （可用環境變數 DS_DISPLAY_SINK 指定）

    回傳：
        顯示用 streammux；各路的顯示分支接到它的 sink_{pad_index}
    """
    g = group_id
    rows, cols, width, height = _tile_layout(member_count)

    mux = make_streammux(f"Stream-muxer-display-g{g}", member_count)
    tiler = make_elm("nvmultistreamtiler", f"nvtiler-display-g{g}")
    tiler.set_property("rows", rows)
    tiler.set_property("columns", cols)
    tiler.set_property("width", width)
    tiler.set_property("height", height)
    q_tiler = make_queue(f"q-display-1-g{g}", DISPLAY_QUEUE_BUFFERS)
    convert = make_convert(f"convertor-display-g{g}")
    q_sink = make_queue(f"q-display-2-g{g}", DISPLAY_QUEUE_BUFFERS)
    chain = [mux, tiler, q_tiler, convert]

    nvidia_sink = next((f for f in ("nveglglessink", "nv3dsink")
                        if Gst.ElementFactory.find(f) is not None), None)
    if nvidia_sink is not None:
        sink_name = nvidia_sink
        if is_jetson() and Gst.ElementFactory.find("nvegltransform") is not None:
            chain += [q_sink, make_elm("nvegltransform", f"nvegl-transform-display-g{g}")]
        else:
            chain += [q_sink]
    else:
        forced = os.environ.get("DS_DISPLAY_SINK", "").strip()
        candidates = [forced] if forced else ["ximagesink", "glimagesink", "autovideosink"]
        sink_name = next((c for c in candidates if Gst.ElementFactory.find(c) is not None), None)
        if sink_name is None:
            sys.exit(f"[ERROR] 找不到可用的顯示 sink（已嘗試：{', '.join(candidates)}）")
        caps = make_elm("capsfilter", f"caps-display-sys-g{g}")
        # 轉到系統記憶體
        caps.set_property("caps", Gst.Caps.from_string("video/x-raw, format=RGBA"))
        chain += [caps, make_elm("videoconvert", f"videoconvert-display-g{g}"), q_sink]

    sink = make_elm(sink_name, f"nvvideo-renderer-display-g{g}")
    safe_set(sink, "sync", False)   # 不依時間戳等待，來就畫
    safe_set(sink, "qos", False)    # 不回報 QoS，避免上游因此丟幀
    safe_set(sink, "async", False)  # 不等它收到第一幀才切換 pipeline 狀態
    _add_and_link(pipeline, chain + [sink])

    print(f"[INFO] [group{g}] 預覽視窗：{rows}x{cols} 格、{width}x{height}、sink={sink_name}")
    return mux


# =============================================================================
# 每路攝影機的下游分支
# =============================================================================

def setup_cam_branch(pipeline, pad_index, uid, cfg, demux, display_mux, osd_probe):
    """
    建立一路攝影機在 demux 之後的分支（結構見檔頭）。

    參數：
        pad_index    該路在本組的區域編號（demux / 顯示用 streammux 的 pad 編號）
        uid          全域編號（元件命名、osd_probe 用）
        cfg          SOURCE_CONFIGS[uid]
        demux        本組的 nvstreamdemux
        display_mux  本組的顯示用 streammux；本組沒有任何一路要顯示時為 None
        osd_probe    FPS 疊字探針；只有 display.show_fps_overlay=true 才掛
    """
    queue = make_queue(f"q-cam-{uid}")
    convert = make_convert(f"conv_osd_{uid}")
    caps = make_elm("capsfilter", f"caps_osd_{uid}")
    # nvdsosd 只吃 RGBA
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    osd = make_elm("nvdsosd", f"nvosd-{uid}")
    osd.set_property("process-mode", 2 if is_jetson() else 1)   # 2 = 硬體 VIC（Jetson）；1 = GPU

    _add_and_link(pipeline, [queue, convert, caps, osd])
    demux.get_request_pad(f"src_{pad_index}").link(queue.get_static_pad("sink"))

    if cfg["display"].get("show_fps_overlay", True):
        osd.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, osd_probe, uid)

    save = bool(cfg["output"].get("save_output_video", False))
    show = bool(cfg["display"].get("show_window", True))

    if not save and not show:
        fakesink = make_fakesink(f"fake-{uid}")                 # 都不要：畫面算完直接丟
        pipeline.add(fakesink)
        osd.link(fakesink)
        return

    upstream = osd
    if save and show:
        upstream = make_elm("tee", f"tee-{uid}")                # 兩者都要：用 tee 分成兩路
        pipeline.add(upstream)
        osd.link(upstream)

    if save:
        head = _build_save_branch(pipeline, uid, cfg)
        if upstream is osd:
            osd.link(head)
        else:
            q_save = make_queue(f"q-s-{uid}")                   # tee 的每個輸出都要接 queue，否則互相卡住
            pipeline.add(q_save)
            upstream.link(q_save)
            q_save.link(head)

    if show:
        _link_show_branch(pipeline, pad_index, uid, upstream, display_mux)


def _link_show_branch(pipeline, pad_index, uid, upstream, display_mux):
    """
    顯示分支：upstream → queue → nvvideoconvert → 顯示用 streammux.sink_{pad_index}

    這顆 queue 是顯示端與主線的隔離點（容量規則見檔頭）。
    nvvideoconvert 關閉直通：輸入輸出格式相同時它預設不複製，直接把上游 buffer 往下送，
    那些 buffer 會被顯示用 streammux 佔住，視窗一卡上游的 pool 就被吃光。
    關閉直通後它複製到自己的 pool，顯示端最多只佔住它自己的 buffer。
    """
    queue = make_queue(f"q-d-{uid}", DISPLAY_QUEUE_BUFFERS)
    convert = make_convert(f"nv-d-{uid}")
    safe_set(convert, "disable-passthrough", True)  # 一定要複製，原因見上方說明

    _add_and_link(pipeline, [queue, convert])
    upstream.link(queue)
    convert.get_static_pad("src").link(display_mux.get_request_pad(f"sink_{pad_index}"))

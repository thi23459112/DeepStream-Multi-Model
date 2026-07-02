# logic/pipeline.py
"""
pipeline.py
-----------
GStreamer pipeline 元件建構與分支邏輯（多權重 / 多 pipeline 版）。

重要觀念：
    本專案依 weight 分組，每組跑「一條獨立 pipeline」。
    - 區域 pad_index：某條 pipeline 內 streammux/demux 的 sink_/src_ 編號（各組從 0 起算）
    - 全域 stream_uid：跨所有組唯一的編號（= SOURCE_CONFIGS 的鍵）
    setup_cam_branch 同時收 local_idx（接 demux/display 的 pad 用）與 uid
    （命名、OSD 探針、狀態鍵用），避免多條 pipeline 之間互相污染。

每路 cam 的下游分支由 YAML 決定：
    - output.save_output_video → 寫檔分支（存成 mp4）
    - display.show_window      → 本地預覽分支（併入該組自己的 tile 視窗）
"""

import sys
from gi.repository import Gst

from logic.config import SOURCE_CONFIGS


# ==========================================
# uridecodebin 動態 pad 回呼
# ==========================================
def cb_newpad(decodebin, decoder_src_pad, data):
    """
    uridecodebin 動態 pad 出現時，鏈到「該組」streammux 的區域 sink_{local_idx}。

    data 需含：
        "streammux"  : 該組的 streammux
        "pad_index"  : 區域 pad index（該組內從 0 起算）
    """
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()

    if caps.get_structure(0).get_name().find("video") != -1:
        sinkpad = data["streammux"].get_request_pad(f"sink_{data['pad_index']}")
        if not sinkpad.is_linked():
            decoder_src_pad.link(sinkpad)


def cb_source_setup(decodebin, source_element, user_data):
    """來源若是 RTSP 網路攝影機，注入防斷線 / 低延遲參數（僅對 rtspsrc 生效）。"""
    if source_element.get_name().startswith("rtspsrc"):
        print(f"[INFO] 偵測到 RTSP 來源，注入防斷線參數: {source_element.get_name()}")
        source_element.set_property("protocols", 4)           # 4 = TCP
        source_element.set_property("latency", 200)           # 抖動緩衝 200ms
        source_element.set_property("timeout", 5000000)       # 連線逾時 5 秒（微秒）
        source_element.set_property("drop-on-latency", True)  # 超過緩衝丟舊幀


def make_elm(gst_type, name):
    """建立單一 GStreamer 元件，失敗則整支程式退出。"""
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
# 寫檔分支：本地檔案來源
# ==========================================
def _build_save_branch_for_file(pipeline, uid, video_path, source_fps):
    """
    寫檔分支（檔案來源版）：
        nvvideoconvert → videorate → capsfilter(NV12,N/1) → nvv4l2h264enc
        → h264parse → qtmux → filesink
    插 videorate 用固定 framerate 重打 PTS，避免 live-source=1 造成 PTS 不規律、qtmux 拒收。
    元件命名用全域 uid，確保跨 pipeline 不撞名。回傳分支起點 element。
    """
    i = uid

    nvvidconv_s = make_elm("nvvideoconvert", f"convertor-save-{i}")
    nvvidconv_s.set_property("nvbuf-memory-type", 0)

    videorate = make_elm("videorate", f"videorate-save-{i}")

    cap_filter = make_elm("capsfilter", f"cap-filter-save-{i}")
    fps_int = int(round(source_fps)) if source_fps and source_fps > 0 else 30
    cap_filter.set_property(
        "caps",
        Gst.Caps.from_string(f"video/x-raw(memory:NVMM), format=NV12, framerate={fps_int}/1"),
    )

    encoder = make_elm("nvv4l2h264enc", f"encoder-{i}")
    encoder.set_property("bitrate", 4000000)
    encoder.set_property("preset-level", 1)
    encoder.set_property("insert-sps-pps", 1)
    encoder.set_property("profile", 0)
    encoder.set_property("maxperf-enable", 1)
    encoder.set_property("iframeinterval", fps_int)

    parser = make_elm("h264parse", f"h264-parser-{i}")
    muxer = make_elm("qtmux", f"muxer-{i}")
    muxer.set_property("dts-method", 1)
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
# 寫檔分支：RTSP 攝影機來源
# ==========================================
def _build_save_branch_for_rtsp(pipeline, uid, video_path, source_fps):
    """寫檔分支（RTSP 攝影機來源版）：同樣插 videorate 穩定 framerate 後再編碼。"""
    i = uid

    nvvidconv_s = make_elm("nvvideoconvert", f"convertor-save-{i}")
    nvvidconv_s.set_property("nvbuf-memory-type", 0)

    videorate  = make_elm("videorate", f"videorate-{i}")
    cap_filter = make_elm("capsfilter", f"cap-filter-save-{i}")
    fps_int = int(round(source_fps)) if source_fps and source_fps > 0 else 30
    cap_filter.set_property(
        "caps",
        Gst.Caps.from_string(f"video/x-raw(memory:NVMM), format=NV12, framerate={fps_int}/1"),
    )

    encoder = make_elm("nvv4l2h264enc", f"encoder-{i}")
    encoder.set_property("bitrate", 4000000)
    encoder.set_property("preset-level", 1)
    encoder.set_property("insert-sps-pps", 1)
    encoder.set_property("profile", 0)
    encoder.set_property("maxperf-enable", 1)
    encoder.set_property("iframeinterval", fps_int)

    parser = make_elm("h264parse", f"h264-parser-{i}")
    muxer  = make_elm("qtmux", f"muxer-{i}")
    muxer.set_property("dts-method", 1)
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
        streammux(顯示用) → tiler → nvvideoconvert → nvegltransform → nveglglessink
    回傳顯示用 streammux（該組各路顯示分支接它的 sink_{local_idx}）。

    live-source / batched-push-timeout 與主線一致（1 / 16666），
    某路 EOS 後不會死等、反壓不會頂回主線。
    元件命名帶 group_id，避免多組視窗撞名。

    參數：
        pipeline     : 該組的 Gst.Pipeline
        group_id     : 組別（用於元件命名 + 視窗識別）
        num_members  : 該組成員數（決定 tile 佈局與 batch-size）
    """
    rows, cols, total_w, total_h = _get_tile_layout(num_members)
    print(f"[INFO] [group{group_id}] Tile 佈局: {rows}x{cols}, 視窗尺寸: {total_w}x{total_h}")

    g = group_id
    streammux2 = make_elm("nvstreammux", f"Stream-muxer-display-g{g}")
    streammux2.set_property("width", 1920)
    streammux2.set_property("height", 1080)
    streammux2.set_property("batch-size", num_members)
    streammux2.set_property("batched-push-timeout", 16666)   # 與主線一致
    streammux2.set_property("live-source", 1)                # 與主線一致
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
    transform = make_elm("nvegltransform", f"nvegl-transform-display-g{g}")
    q_d3 = make_elm("queue", f"q-display-3-g{g}")

    sink = make_elm("nveglglessink", f"nvvideo-renderer-display-g{g}")
    sink.set_property("sync", False)
    sink.set_property("qos", False)

    for elm in [streammux2, tiler, q_d1, nvvidconv, q_d2, transform, q_d3, sink]:
        pipeline.add(elm)

    streammux2.link(tiler)
    tiler.link(q_d1)
    q_d1.link(nvvidconv)
    nvvidconv.link(q_d2)
    q_d2.link(transform)
    transform.link(q_d3)
    q_d3.link(sink)

    return streammux2


# ==========================================
# 每路 cam 分支組裝（save / show）
# ==========================================
def setup_cam_branch(pipeline, local_idx, uid, cfg, demux, display_streammux, osd_probe_callback):
    """
    為單路 cam 建立完整下游分支。

    參數：
        pipeline           : 該組的 Gst.Pipeline
        local_idx          : 該組 pipeline 內的區域 pad index（接 demux/display pad 用）
        uid                : 全域 stream_uid（元件命名、OSD 探針、狀態鍵用）
        cfg                : 此路設定（來自 SOURCE_CONFIGS[uid]）
        demux              : 該組的 nvstreamdemux
        display_streammux  : 該組的顯示用 mux（無顯示時為 None）
        osd_probe_callback : OSD 探針回呼（畫 FPS / ROI）；會用 uid 綁定
    """
    # demux 的輸出 pad 用「區域」索引；元件命名用「全域 uid」
    src_pad = demux.get_request_pad(f"src_{local_idx}")

    q_cam = make_elm("queue", f"q-cam-{uid}")
    nvvidconv_osd = make_elm("nvvideoconvert", f"conv_osd_{uid}")
    nvvidconv_osd.set_property("nvbuf-memory-type", 0)
    caps_osd = make_elm("capsfilter", f"caps_osd_{uid}")
    caps_osd.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    nvosd_i = make_elm("nvdsosd", f"nvosd-{uid}")
    nvosd_i.set_property("process-mode", 2)

    for elm in [q_cam, nvvidconv_osd, caps_osd, nvosd_i]:
        pipeline.add(elm)

    src_pad.link(q_cam.get_static_pad("sink"))
    q_cam.link(nvvidconv_osd)
    nvvidconv_osd.link(caps_osd)
    caps_osd.link(nvosd_i)

    # OSD 探針：用「全域 uid」綁定，回呼才知道畫哪一路的 FPS
    nvosd_i.get_static_pad("sink").add_probe(
        Gst.PadProbeType.BUFFER,
        lambda pad, info, u=uid: osd_probe_callback(pad, info, u),
        0
    )

    cam_save = cfg.get("output", {}).get("save_output_video", False)
    cam_show = cfg.get("display", {}).get("show_window", True)
    is_file  = cfg.get("is_file_source", False)
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
            if is_file:
                nvosd_i.link(_build_save_branch_for_file(pipeline, uid, cfg["video_path"], cfg["stream_fps"]))
            else:
                nvosd_i.link(_build_save_branch_for_rtsp(pipeline, uid, cfg["video_path"], cfg["stream_fps"]))
        elif cam_show:
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
        if is_file:
            q_s.link(_build_save_branch_for_file(pipeline, uid, cfg["video_path"], cfg["stream_fps"]))
        else:
            q_s.link(_build_save_branch_for_rtsp(pipeline, uid, cfg["video_path"], cfg["stream_fps"]))

    if cam_show:
        _link_show_branch(pipeline, local_idx, uid, tee, display_streammux)

    return


def _link_show_branch(pipeline, local_idx, uid, upstream, display_streammux):
    """
    顯示子分支：queue → nvvideoconvert → display_streammux.sink_{local_idx}
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
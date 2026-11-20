"""
來源管理：建立每路攝影機的 uridecodebin，即時來源（RTSP）斷線時自動重連

一路來源 = 一個 uridecodebin，解出的 video pad 接到該組 streammux 的 sink_{pad_index}。
元件命名為 uri-decode-bin-{uid}-{第幾次建立}；錯誤訊息靠這個名稱認出是哪一路、是不是舊的。

重連只針對即時來源（檔案來源播完就是結束）：

    觸發條件（main.py 每 5 秒呼叫 reconnect_dead_sources 檢查）
        a. 超過 RECONNECT_SECONDS 沒有收到畫面（斷線，或啟動時就沒連上）
        b. 滴幀：連線半死不活，隔一陣子才進一幀，a 的計時一直被重置而永遠不觸發；
           改看最近 TRICKLE_WINDOW_SECONDS 秒的平均 FPS，低於 TRICKLE_MIN_FPS 就重連
        c. 來源在 bus 上報錯（連不上、讀取失敗）→ 等一段時間後重連，
           間隔隨連續失敗次數遞增（5、10、15 … 最多 60 秒）
    作法
        拆掉舊的 uridecodebin、釋放 streammux 的輸入 pad，再建一個新的接回去。
        同組其他路與其他組不受影響。
    log（ds_guard.py 依這兩種訊息判斷是否要重啟服務，改格式時要一併修改它的 RE_RECONNECT / RE_RECOVER）
        重連  [WARNING] [<source_id>] <原因>，重新連線（第 N 次）...
        恢復  [INFO] [<source_id>] 已恢復畫面（先前連續重連 N 次）
    配套
        - 即時來源斷線時送出的 EOS 會被攔下。放行的話 streammux 會把這一路永久標記為結束，
          只有一路時整條 pipeline 直接收工。程式自己要結束（begin_shutdown 之後）才放行。
        - 即時來源在 bus 上報的錯誤一律不致命（見 handle_bus_error）。
"""

import os
import re
import time

from gi.repository import Gst

from logic.config import SOURCE_CONFIGS
from logic.pipeline import make_elm, make_fakesink, safe_set
from logic.state_db import fps_streams


# =============================================================================
# 常數與狀態
# =============================================================================

RECONNECT_SECONDS = float(os.environ.get("DS_RECONNECT_SECONDS", "20"))     # 多久（秒）沒畫面就重連；0 = 停用
RETRY_BASE_SECONDS = 5.0                                                    # 來源報錯後的重連間隔（秒）= 此值 × 連續失敗次數
RETRY_MAX_SECONDS = 60.0                                                    # 重連間隔上限（秒）
TRICKLE_WINDOW_SECONDS = 60.0                                               # 滴幀判斷的觀察時間（秒）；bin 建立未滿這麼久不判斷
TRICKLE_MIN_FPS = 1.0                                                       # 觀察時間內平均 FPS 低於此值視為滴幀
RECOVERED_SECONDS = 5.0                                                     # 最後一幀在這麼久（秒）以內 = 已恢復，連續失敗次數歸零
RTSP_LATENCY_MS = 2000                                                      # rtspsrc 抖動緩衝（毫秒）；畫面因此比即時晚約 2 秒

_BIN_NAME_RE = re.compile(r"uri-decode-bin-(\d+)-(\d+)")                    # 群組 1 = uid，群組 2 = 第幾次建立

# uid → {
#     "pipeline", "streammux", "pad_index"  這一路接在哪裡
#     "bin"       目前的 uridecodebin
#     "drains"    為多餘 pad 建的 fakesink，重連時一起移除
#     "attempt"   目前的 bin 是第幾次建立；畫面恢復正常後歸 1
#     "since"     目前的 bin 建立的時間
#     "retry_at"  來源報錯後預定重連的時間；None = 沒有排程
# }
_sources = {}
_shutting_down = False


def begin_shutdown():
    """程式進入結束流程時呼叫：之後不再攔 EOS、不再重連。"""
    global _shutting_down
    _shutting_down = True


def is_live(uid):
    """該路是否為即時來源（RTSP / HTTP 串流）。"""
    return not SOURCE_CONFIGS[uid]["is_file_source"]


# =============================================================================
# 建立來源
# =============================================================================

def add_source(pipeline, streammux, pad_index, uid):
    """
    建立一路來源並加入 pipeline。

    不會自己切換狀態：啟動時隨 pipeline 一起進入 PLAYING；重連時由呼叫端同步狀態。
    """
    info = _sources.setdefault(uid, {"pipeline": pipeline, "streammux": streammux,
                                     "pad_index": pad_index, "attempt": 0})
    info["attempt"] += 1                            # 第一次建立為 1，每次重連 +1
    info.update(drains=[], since=time.time(), retry_at=None)

    # 名稱帶次數，舊 bin 的錯誤才認得出來
    src = make_elm("uridecodebin", f"uri-decode-bin-{uid}-{info['attempt']}")
    src.set_property("uri", SOURCE_CONFIGS[uid]["source"])
    src.connect("pad-added", _on_pad_added, uid)    # 解出 pad 時接到 streammux
    src.connect("source-setup", _on_source_setup)   # 內部來源元件建好時調整 RTSP 參數
    pipeline.add(src)
    info["bin"] = src
    return src


def _on_source_setup(decodebin, source):
    """
    uridecodebin 建好內部的來源元件時呼叫；來源是 rtspsrc 才調整參數。

    必須用元件「型別」判斷：內部來源元件的名稱固定是 "source"，不是 "rtspsrc…"。
    """
    if source.__gtype__.name != "GstRTSPSrc":
        return                                          # 檔案等其他來源不需要調整
    source.set_property("protocols", 4)                 # 4 = TCP；跨 NAT 時 UDP 常不通，不等它逾時再退回
    source.set_property("latency", RTSP_LATENCY_MS)     # 吸收各路攝影機 FPS / 時脈不一致
    source.set_property("drop-on-latency", False)       # 不因延遲丟封包，避免花屏
    safe_set(source, "ntp-sync", False)                 # 不用攝影機的 NTP 時間戳，以本機時脈為準
    safe_set(source, "buffer-mode", 1)                  # 1 = slave：依接收端時脈校正，長時間執行較穩定


def _on_pad_added(decodebin, pad, uid):
    """
    uridecodebin 解出一個 pad 時呼叫。

    第一條 video 接到 streammux；其餘（音訊、多餘的 video）接 fakesink 消化掉，
    避免未連結的 pad 讓來源報 not-linked 錯誤。
    """
    info = _sources[uid]
    pipeline, streammux = info["pipeline"], info["streammux"]
    caps = pad.get_current_caps() or pad.query_caps()
    is_video = caps.get_structure(0).get_name().startswith("video")

    sinkpad = None
    if is_video:
        pad_name = f"sink_{info['pad_index']}"
        # 重連後可能還留著同名 pad
        sinkpad = streammux.get_static_pad(pad_name) or streammux.get_request_pad(pad_name)

    if sinkpad is None or sinkpad.is_linked():
        fakesink = make_fakesink()          # 自動命名：重連會一再建立，不能撞名
        pipeline.add(fakesink)
        fakesink.sync_state_with_parent()   # pipeline 已在 PLAYING，要手動跟上
        pad.link(fakesink.get_static_pad("sink"))
        info["drains"].append(fakesink)     # 記下來，重連時一起移除
        return

    if is_live(uid):
        pad.add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, _drop_eos_probe)
    pad.link(sinkpad)


def _drop_eos_probe(pad, info):
    """即時來源的 EOS 在結束流程開始前一律丟掉（原因見檔頭）。"""
    event = info.get_event()
    if event is not None and event.type == Gst.EventType.EOS and not _shutting_down:
        return Gst.PadProbeReturn.DROP
    return Gst.PadProbeReturn.OK


# =============================================================================
# 重連
# =============================================================================

def _reconnect(uid):
    """拆掉該路舊的來源、釋放 streammux 的輸入 pad，建立新的來源並啟動。"""
    info = _sources[uid]
    pipeline, streammux = info["pipeline"], info["streammux"]

    for element in [info["bin"], *info["drains"]]:
        element.set_state(Gst.State.NULL)                       # 先停止再移除，否則會留下執行中的執行緒
        pipeline.remove(element)

    sinkpad = streammux.get_static_pad(f"sink_{info['pad_index']}")
    if sinkpad is not None:
        sinkpad.send_event(Gst.Event.new_flush_stop(False))     # 清掉殘留的 flushing 狀態；False = 不重設 running time
        streammux.release_request_pad(sinkpad)                  # 釋放後新的 bin 才能重新請求同名 pad

    add_source(pipeline, streammux, info["pad_index"], uid).sync_state_with_parent()


def reconnect_dead_sources():
    """檢查所有即時來源，符合重連條件（見檔頭）的就重連。由 main.py 的定時器呼叫。"""
    if RECONNECT_SECONDS <= 0 or _shutting_down:
        return
    now = time.time()
    for uid, info in _sources.items():
        if not is_live(uid):
            continue
        stamps = fps_streams[uid]["timestamps"]
        last_frame = stamps[-1] if stamps else 0.0
        idle = now - max(last_frame, info["since"])     # 從較晚者起算：剛重連的不會馬上又被判定
        errored = info["retry_at"] is not None and now >= info["retry_at"]

        # 滴幀：stamps 只保留最近 FPS_WINDOW_FRAMES 幀，正常串流 1~2 秒就填滿；
        # 若這些幀橫跨了整個觀察時間，平均 FPS 遠低於正常
        trickle = False
        if stamps and now - info["since"] >= TRICKLE_WINDOW_SECONDS:
            span = now - stamps[0]
            trickle = span >= TRICKLE_WINDOW_SECONDS and len(stamps) / span < TRICKLE_MIN_FPS

        if not errored and not trickle and idle < RECONNECT_SECONDS:
            if now - last_frame < RECOVERED_SECONDS and info["attempt"] > 1:
                print(f"[INFO] [{SOURCE_CONFIGS[uid]['source_id']}] 已恢復畫面"
                      f"（先前連續重連 {info['attempt'] - 1} 次）")
                info["attempt"] = 1                     # 已恢復，連續失敗次數歸零
            continue

        name = SOURCE_CONFIGS[uid]["source_id"]
        if errored:
            reason = "來源報錯"
        elif trickle:
            reason = f"滴幀（近 {now - stamps[0]:.0f} 秒只有 {len(stamps)} 幀）"
        else:
            reason = f"已 {idle:.0f} 秒沒有畫面"
        print(f"\n[WARNING] [{name}] {reason}，重新連線（第 {info['attempt']} 次）...\n")
        try:
            _reconnect(uid)
        except Exception as e:
            info.update(since=now, retry_at=None)       # 重設計時，下一輪再試
            print(f"[WARNING] [{name}] 重連失敗：{e}（稍後再試）")


# =============================================================================
# bus 錯誤歸屬
# =============================================================================

def handle_bus_error(message):
    """
    判斷 bus 上的 ERROR 是否來自即時來源；是的話在這裡處理掉。

    辨認方式：從發出訊息的元件往上找名稱符合 uri-decode-bin-{uid}-{n} 的上層；
    找不到時改從訊息的 debug 字串找。後者是必要的：被重連拆掉的舊來源在收尾時仍可能報錯
    （例如對已斷的連線送 TEARDOWN 失敗），那時它已經沒有上層元件，
    只有 debug 字串裡還留著發出當下的完整元件路徑。

    舊來源的錯誤直接忽略；目前來源的錯誤印出警告並排程重連。

    回傳：
        True   來自即時來源，已處理，不致命
        False  不是來自即時來源，交給呼叫端處理
    """
    err, debug = message.parse_error()

    bin_name, obj = None, message.src
    while obj is not None:                                          # 往上找所屬的 uridecodebin
        if _BIN_NAME_RE.fullmatch(obj.get_name() or ""):
            bin_name = obj.get_name()
            break
        obj = obj.get_parent()
    if bin_name is None:
        found = _BIN_NAME_RE.search(debug or "")                    # 已被拆掉的元件：從 debug 字串找
        if found is None:
            return False
        bin_name = found.group(0)

    uid = int(_BIN_NAME_RE.fullmatch(bin_name).group(1))
    info = _sources.get(uid)
    if info is None or not is_live(uid):
        return False                                                # 檔案來源的錯誤照常視為嚴重錯誤

    if info["bin"].get_name() == bin_name and not _shutting_down:   # 名稱不同 = 舊 bin，忽略
        delay = min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * info["attempt"])
        info["retry_at"] = time.time() + delay
        print(f"[WARNING] [{SOURCE_CONFIGS[uid]['source_id']}] 來源錯誤：{err.message}"
              f"（{delay:.0f} 秒後重連）")
    return True

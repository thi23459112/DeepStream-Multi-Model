"""
GStreamer pad 探針：追蹤、方向判定、ROI 命中、畫面標註

每條 pipeline 掛一支追蹤探針，依追蹤器模式二選一：

    nvdcf_probe   掛在 nvtracker.src；物件已由 nvtracker 編好 object_id
    boxmot_probe  掛在 nvinfer.src；取出偵測框交給 BoxMOT，再用追蹤結果重建物件 meta

兩者之後都走 _update_tracks()：更新軌跡狀態、判斷方向、計算 ROI 命中、設定框與標籤的外觀。
另外每路攝影機的 nvdsosd.sink 可掛 osd_probe，在左上角顯示 FPS。

追蹤探針的 ctx（由 main.py 建立）：
    pad_to_uid  [uid, ...]，索引 = frame_meta.pad_index
    group_uids  frozenset(uid)，本組的 uid；清理與寫 DB 只處理本組
    class_map   {class_id: 名稱}，本組 engine 的類別表
"""

import time
from collections import Counter

import cv2
import numpy as np
import pyds
from gi.repository import Gst

from logic.color import get_class_color
from logic.config import SOURCE_CONFIGS
from logic.state_db import (
    get_local_id, retire_track, flush_pending_to_db,
    track_history, pending_records, last_flush_times, fps_streams,
)


# =============================================================================
# 常數與狀態
# =============================================================================

FPS_REPORT_SECONDS = 30     # 終端機 FPS 報告的間隔（秒）
FPS_STALE_SECONDS = 5       # 超過這麼久（秒）沒有新畫面，報告顯示「無畫面」
PGIE_UNIQUE_ID = 1          # nvinfer 的 gie-unique-id；本專案的物件 meta 都標這個值
DEFAULT_CONFIDENCE = 0.5    # parser 沒填信心值時給追蹤器的值
LABEL_HEIGHT_PX = 29        # 物件標籤的高度（像素）≈ 字高（14 × 1.4）+ 10 像素邊距

_last_fps_report = time.time()
_boxmot_track = None        # logic.boxmot_adapter.track；第一次用到才 import（nvdcf 模式不需要 boxmot）


# =============================================================================
# meta 走訪
# =============================================================================

def _frames(batch_meta):
    """依序產生 batch 內每個 NvDsFrameMeta。"""
    node = batch_meta.frame_meta_list
    while node is not None:
        try:
            yield pyds.NvDsFrameMeta.cast(node.data)
        except StopIteration:
            return
        node = node.next


def _objects(frame_meta):
    """依序產生 frame 內每個 NvDsObjectMeta。走訪途中不可增刪物件；要增刪請先轉成 list。"""
    node = frame_meta.obj_meta_list
    while node is not None:
        try:
            yield pyds.NvDsObjectMeta.cast(node.data)
        except StopIteration:
            return
        node = node.next


# =============================================================================
# 軌跡更新（兩種追蹤模式共用）
# =============================================================================

def _update_tracks(frame_meta, seen, uid, cfg, class_map):
    """
    處理一幀內所有已追蹤的物件。

    參考點取 bbox 底邊中點，比中心點更接近車輪 / 腳踩在地面的位置。

    軌跡狀態 track_history[(uid, object_id)]：
        start_x, start_y  首次出現時的參考點
        direction         "NA"；位移超過門檻後定為 "flow_in" / "flow_out"（依 up_left_is_out），之後不再改
        roi_hits          {ROI 名稱: 參考點落在該 ROI 內的幀數}
        class_votes       Counter；在 ROI 內時每幀為當下類別投一票，結算取多數
        missing_frames    連續沒出現的幀數
        last_frame_num    最後出現的幀號，結算時換算影片時間
        class_map         本組類別表，結算時把 class_id 換成名稱

    參數：
        seen  本 batch 出現過的 (uid, object_id) 集合，由呼叫端建立，這裡負責加入
    """
    cv_regions = cfg["cv_regions"]
    axis = cfg["track_logic"]["axis"]
    threshold = cfg["track_logic"]["movement_threshold"]
    up_left_is_out = cfg["track_logic"]["up_left_is_out"]
    keep = cfg["keep_classes"]
    frame_num = frame_meta.frame_num

    for obj in _objects(frame_meta):
        if obj.unique_component_id != PGIE_UNIQUE_ID or obj.object_id == -1:
            continue                                                    # 不是本專案的物件，或追蹤器尚未指派 ID（-1）
        cls_id = obj.class_id
        if keep is not None and cls_id not in keep:
            continue                                                    # 不在 keep_classes：不追蹤、不計數、不標註

        rect = obj.rect_params
        cx = int(rect.left + rect.width / 2)                            # 參考點：底邊中點
        cy = int(rect.top + rect.height)

        key = (uid, obj.object_id)
        seen.add(key)
        state = track_history.get(key)
        if state is None:                                               # 新物件
            state = track_history[key] = {
                "start_x":        cx,
                "start_y":        cy,
                "direction":      "NA",
                "roi_hits":       {},
                "class_votes":    Counter(),
                "missing_frames": 0,
                "last_frame_num": frame_num,
                "class_map":      class_map,
            }
        state["missing_frames"] = 0
        state["last_frame_num"] = frame_num

        for roi, polygon in cv_regions.items():
            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:     # 內部 +1、邊上 0、外部 -1
                state["roi_hits"][roi] = state["roi_hits"].get(roi, 0) + 1
                state["class_votes"][cls_id] += 1                       # 只在 ROI 內投票

        # 方向：沿指定軸的位移，delta > 0 為往下 / 往右。
        # up_left_is_out=true 時往下 / 往右為 flow_in；false 時把 delta 反號，往上 / 往左變成 flow_in
        if state["direction"] == "NA":                                  # 定案後不再改
            delta = (cx - state["start_x"]) if axis == "x" else (cy - state["start_y"])
            if not up_left_is_out:
                delta = -delta
            if delta > threshold:
                state["direction"] = "flow_in"
            elif delta < -threshold:
                state["direction"] = "flow_out"

        # 外觀：類別色的框，框內底部「ID:短ID 類別名」
        color = get_class_color(cls_id)
        rect.border_width = 4                                           # 框線寬（像素）
        rect.border_color.set(*color)
        rect.has_bg_color = 0                                           # 框內不填色

        text = obj.text_params
        text.display_text = f"ID:{get_local_id(uid, obj.object_id)} {class_map.get(cls_id, f'Class_{cls_id}')}"
        text.font_params.font_name = "Serif Bold"
        text.font_params.font_size = 14
        text.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)             # 白字
        text.set_bg_clr = 1
        text.text_bg_clr.set(*color)                                    # 底色同框色
        text.x_offset = max(0, int(rect.left))
        # 貼著框的底邊往上放
        text.y_offset = max(0, int(rect.top + rect.height) - LABEL_HEIGHT_PX)


def _update_fps(uid):
    """記錄該攝影機收到一幀的時間，並以最近 FPS_WINDOW_FRAMES 幀計算即時 FPS。"""
    stats = fps_streams[uid]
    stamps = stats["timestamps"]
    stamps.append(time.time())  # deque 有長度上限，舊的自動擠掉
    if len(stamps) > 1 and stamps[-1] > stamps[0]:
        # N 個時間點 = N-1 個間隔
        stats["current_fps"] = (len(stamps) - 1) / (stamps[-1] - stamps[0])


def _housekeeping(seen, group_uids):
    """
    每個 batch 處理完後執行。

        1. 本組這次沒出現的物件 missing_frames + 1；達到 cleanup_frames 就結算。
        2. 每 FPS_REPORT_SECONDS 秒印一次所有攝影機的 FPS。
        3. 本組有待寫紀錄、且距上次寫入超過 flush_interval_seconds 的攝影機 → 寫入 DB。

    只處理本組的 uid：其他組的物件不在這個 batch 裡，不能當成消失。
    """
    global _last_fps_report

    # 先轉 list：迴圈內會刪除
    for key in [k for k in track_history if k[0] in group_uids and k not in seen]:
        state = track_history[key]
        state["missing_frames"] += 1
        if state["missing_frames"] >= SOURCE_CONFIGS[key[0]]["session"].get("cleanup_frames", 30):
            retire_track(key)

    now = time.time()
    if now - _last_fps_report >= FPS_REPORT_SECONDS:    # 任一組的探針先到就印，印的是全部攝影機
        _last_fps_report = now
        print("\n" + "=" * 35)
        print(f"[{time.strftime('%H:%M:%S')}] 即時處理效能報告 (FPS)：")
        for uid, stats in sorted(fps_streams.items()):
            name = SOURCE_CONFIGS[uid]["source_id"]
            stamps = stats["timestamps"]
            if not stamps or now - stamps[-1] > FPS_STALE_SECONDS:
                print(f" • {name}: 0.00 FPS（無畫面）")
            else:
                print(f" • {name}: {stats['current_fps']:.2f} FPS")
        print("=" * 35 + "\n")

    for uid in group_uids:
        if not pending_records[uid]:
            continue
        interval = SOURCE_CONFIGS[uid]["session"].get("flush_interval_seconds", 30)
        if now - last_flush_times[uid] >= interval:     # interval = 0 時每次都寫
            flush_pending_to_db(uid)
            last_flush_times[uid] = now


# =============================================================================
# 追蹤探針
# =============================================================================

def nvdcf_probe(pad, info, ctx):
    """nvdcf 模式的追蹤探針，掛在 nvtracker.src。"""
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    pad_to_uid, class_map = ctx["pad_to_uid"], ctx["class_map"]
    seen = set()
    for frame_meta in _frames(pyds.gst_buffer_get_nvds_batch_meta(hash(buf))):
        if not 0 <= frame_meta.pad_index < len(pad_to_uid):
            continue                            # 不屬於本組的 pad，理論上不會發生
        uid = pad_to_uid[frame_meta.pad_index]  # 區域編號 → 全域編號
        _update_fps(uid)
        _update_tracks(frame_meta, seen, uid, SOURCE_CONFIGS[uid], class_map)

    _housekeeping(seen, ctx["group_uids"])
    return Gst.PadProbeReturn.OK


def boxmot_probe(pad, info, ctx):
    """
    BoxMOT 模式的追蹤探針，掛在 nvinfer.src。

    取下 nvinfer 產生的所有偵測框 → 交給 BoxMOT → 用回傳的軌跡重建物件 meta。
    不在 keep_classes 內的偵測框直接丟掉，不進追蹤器。
    """
    global _boxmot_track
    if _boxmot_track is None:
        from logic.boxmot_adapter import track                      # 延遲 import：nvdcf 模式不載入 boxmot
        _boxmot_track = track

    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    pad_to_uid, class_map = ctx["pad_to_uid"], ctx["class_map"]
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    seen = set()

    for frame_meta in _frames(batch_meta):
        if not 0 <= frame_meta.pad_index < len(pad_to_uid):
            continue
        uid = pad_to_uid[frame_meta.pad_index]                      # 區域編號 → 全域編號
        cfg = SOURCE_CONFIGS[uid]
        keep = cfg["keep_classes"]
        _update_fps(uid)

        # 收集要追蹤的偵測框，再把原本的物件 meta 全部移除
        dets = []
        detections = list(_objects(frame_meta))                     # 先轉 list：之後要從 frame 移除
        for obj in detections:
            cls_id = int(obj.class_id)
            if keep is None or cls_id in keep:
                try:
                    box = obj.detector_bbox_info.org_bbox_coords    # 優先用偵測器原始輸出的框
                except Exception:
                    box = obj.rect_params                           # 取不到時退回顯示用的框
                conf = float(obj.confidence) if obj.confidence > 0 else DEFAULT_CONFIDENCE
                dets.append((box.left, box.top, box.left + box.width, box.top + box.height,
                             conf, cls_id))                         # BoxMOT 要 [x1, y1, x2, y2, conf, cls]
        for obj in detections:
            pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj)

        # 用追蹤結果建立新的物件 meta
        dets = np.asarray(dets, dtype=np.float32) if dets else np.empty((0, 6), dtype=np.float32)
        for x1, y1, x2, y2, track_id, conf, cls_id, _det_index in _boxmot_track(uid, dets):
            obj = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
            if obj is None:
                continue                                            # meta pool 用完，這個框略過
            cls_id = int(cls_id)
            obj.unique_component_id = PGIE_UNIQUE_ID                # 標成 nvinfer 的物件，_update_tracks 才會處理
            obj.class_id = cls_id
            obj.object_id = int(track_id)
            obj.confidence = float(conf)
            obj.obj_label = class_map.get(cls_id, f"Class_{cls_id}")
            rect = obj.rect_params
            rect.left = float(x1)
            rect.top = float(y1)
            rect.width = max(1.0, float(x2 - x1))                   # 寬高至少 1 像素，nvdsosd 才畫得出來
            rect.height = max(1.0, float(y2 - y1))
            pyds.nvds_add_obj_meta_to_frame(frame_meta, obj, None)

        _update_tracks(frame_meta, seen, uid, cfg, class_map)

    _housekeeping(seen, ctx["group_uids"])
    return Gst.PadProbeReturn.OK


# =============================================================================
# FPS 疊字探針
# =============================================================================

def osd_probe(pad, info, uid):
    """掛在該攝影機的 nvdsosd.sink，在畫面左上角顯示即時 FPS。"""
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    fps = fps_streams[uid]["current_fps"]
    for frame_meta in _frames(batch_meta):                      # demux 之後一個 batch 只有一幀
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        text = display_meta.text_params[0]
        text.display_text = f"FPS: {fps:.1f}"
        text.x_offset = 5                                       # 距左上角 5 像素
        text.y_offset = 5
        text.font_params.font_name = "Serif Bold"
        text.font_params.font_size = 25
        text.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)     # 綠字
        text.set_bg_clr = 1
        text.text_bg_clr.set(0.0, 0.0, 0.0, 0.8)                # 半透明黑底
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    return Gst.PadProbeReturn.OK

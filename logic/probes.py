"""
DeepStream Probe 探針集合（車流版 · 多權重 / 多 pipeline）

與單 pipeline 版的差異：
1. 每條 pipeline 的 probe 綁一個 ctx（u_data）：
       {"pad_to_uid": [全域uid, ...],  # 索引 = 該 pipeline 區域 pad_index
        "class_map":  {cls_id: name}}  # 該組（該 engine）專屬的類別表
2. probe 內把 DeepStream 給的「區域 frame_meta.pad_index」翻譯成「全域 stream_uid」，
   之後所有狀態（track_history / fps_streams / boxmot tracker / DB）都用 uid 當鍵，
   多條 pipeline 之間不會互相污染。
3. 類別名稱一律用該組 class_map（car 組 7 類、yolo 組只有 person），並存進軌跡狀態，
   供 state_db 寫 DB 時取用正確名稱。
"""

import time
import cv2
import numpy as np
from collections import Counter, deque
from gi.repository import Gst
import pyds

from logic.color import get_class_color
from logic.config import SOURCE_CONFIGS
from logic.state_db import (
    get_local_id, _finalize_one, flush_pending_to_db,
    track_history, pending_records, last_flush_times,
    fps_streams, local_id_maps
)


# ==========================================
# 1. 系統配置區 (System Configuration)
# ==========================================

g_last_fps_print_time = time.time()          # 上次印 FPS 報告的時間戳（全域共用）


# ==========================================
# 2. 共用核心邏輯 (Shared Tracking Logic)
# ==========================================

def _process_tracked_frame(frame_meta, current_frame_objects, uid, cfg, class_map):
    """
    遍歷 frame_meta 內所有 obj_meta，維護軌跡狀態、調整 OSD 顯示。

    參數：
        frame_meta            : 當前幀 meta
        current_frame_objects : 本幀出現的 (uid, obj_id) 集合
        uid (int)             : 全域 stream_uid（已由 probe 從區域 pad_index 翻譯過）
        cfg (dict)            : SOURCE_CONFIGS[uid]
        class_map (dict)      : 該組（該 engine）的類別表 {cls_id: name}
    """
    cv_regions = cfg.get("cv_regions", {})
    tl = cfg.get("track_logic", {})
    movement_threshold = tl.get("movement_threshold", 30)
    axis = tl.get("axis", "y")
    keep = cfg.get("keep_classes")      # frozenset 或 None（None = 全收）

    l_obj = frame_meta.obj_meta_list
    while l_obj is not None:
        try:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
        except StopIteration:
            break

        # 只處理追蹤器輸出的物件（unique_component_id=1）
        if obj_meta.unique_component_id != 1:
            l_obj = l_obj.next
            continue

        # keep_classes 過濾：不在白名單內的 class_id 直接略過（不畫框 / 不計數 / 不寫 DB）
        if keep is not None and obj_meta.class_id not in keep:
            l_obj = l_obj.next
            continue

        obj_id = obj_meta.object_id
        if obj_id == -1:
            l_obj = l_obj.next
            continue

        unique_key = (uid, obj_id)
        current_frame_objects.add(unique_key)
        local_id = get_local_id(uid, obj_id)

        # bbox 底部中心點
        cx = int(obj_meta.rect_params.left + (obj_meta.rect_params.width / 2))
        cy = int(obj_meta.rect_params.top + obj_meta.rect_params.height)

        # 首次出現 → 初始化軌跡狀態
        if unique_key not in track_history:
            track_history[unique_key] = {
                "start_x":        cx,
                "start_y":        cy,
                "missing_frames": 0,
                "direction":      "NA",       # flow_in / flow_out / NA
                "class_votes":    Counter(),
                "last_frame_num": frame_meta.frame_num,
                "last_v_box":     None,
                "roi_hits":       {},
                "class_map":      class_map,  # ⭐ 存該組類別表，state_db 結算時取正確車種名
            }

        state = track_history[unique_key]
        state["missing_frames"] = 0
        state["last_frame_num"] = frame_meta.frame_num

        r = obj_meta.rect_params
        state["last_v_box"] = (float(r.left), float(r.top),
                               float(r.left + r.width), float(r.top + r.height))

        # 多 ROI 命中判斷
        for roi_name, polygon in cv_regions.items():
            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                state["roi_hits"][roi_name] = state["roi_hits"].get(roi_name, 0) + 1
                state["class_votes"][obj_meta.class_id] += 1

        # 方向判斷（首次定向後固定）
        if state["direction"] == "NA":
            delta = (cx - state["start_x"]) if axis == "x" else (cy - state["start_y"])
            if delta > movement_threshold:
                state["direction"] = "flow_in"     # 軸正向（y:向下 / x:向右）
            elif delta < -movement_threshold:
                state["direction"] = "flow_out"    # 軸負向（y:向上 / x:向左）

        # OSD 視覺化（用該組 class_map 取名）
        cls_id = obj_meta.class_id
        cls_name = class_map.get(cls_id, f"Class_{cls_id}")
        color = get_class_color(cls_id)

        r.border_width = 4
        r.border_color.set(*color)
        r.has_bg_color = 0

        txt = obj_meta.text_params
        txt.display_text = f"ID:{local_id} {cls_name}"
        txt.font_params.font_name = "Serif Bold"
        txt.font_params.font_size = 14
        txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        txt.set_bg_clr = 1
        txt.text_bg_clr.set(*color)

        text_h = int(14 * 1.4)
        txt.x_offset = max(0, int(r.left) + 0)
        txt.y_offset = max(0, int(r.top + r.height) - text_h - 10)

        l_obj = l_obj.next


def _post_frame_housekeeping(current_frame_objects, group_uids):
    """
    每幀結束的清理。

    參數：
        current_frame_objects (set): 本幀（本組）出現的 (uid, obj_id) 集合
        group_uids (set): 本組涵蓋的全域 uid 集合

    處理：
    1. 消失軌跡結算：只比對「本組 uid」的軌跡。每個 probe 的 current_frame_objects
       只含自己這組的物件，若不先用 group_uids 過濾，會把「別組」還在畫面上的軌跡
       誤當成本幀未出現而累加 missing_frames、提早結算。故先限定本組 uid 再比對。
    2. 每 30 秒印一次 FPS（g_last_fps_print_time 全域共用，涵蓋所有 uid）。
    3. 依 flush_interval_seconds 定期把 pending flush 進合併 DB。
    """
    global g_last_fps_print_time

    # 消失軌跡結算：只取本組 uid 的軌跡鍵，再扣掉本幀出現的
    group_keys = {k for k in track_history.keys() if k[0] in group_uids}
    missing_keys = group_keys - current_frame_objects
    for m_key in missing_keys:
        uid, obj_id = m_key
        cfg = SOURCE_CONFIGS.get(uid, {})
        track_history[m_key]["missing_frames"] += 1
        cleanup_frames = cfg.get("session", {}).get("cleanup_frames", 30)
        if track_history[m_key]["missing_frames"] >= cleanup_frames:
            _finalize_one(m_key, track_history[m_key], force=False)
            del track_history[m_key]
            if uid in local_id_maps and obj_id in local_id_maps[uid]:
                del local_id_maps[uid][obj_id]

    # 每 30 秒印 FPS（涵蓋所有組 / 所有 uid）
    current_time = time.time()
    if current_time - g_last_fps_print_time >= 30:
        print("\n" + "=" * 35)
        print(f"[{time.strftime('%H:%M:%S')}] 即時處理效能報告 (FPS)：")
        for sid, stats in sorted(fps_streams.items()):
            c_name = SOURCE_CONFIGS.get(sid, {}).get("source_id", f"cam_{sid}")
            print(f" • {c_name.ljust(10)}: {stats.get('current_fps', 0.0):.2f} FPS")
        print("=" * 35 + "\n")
        g_last_fps_print_time = current_time

    # 定期 flush 到合併 DB
    for uid, cfg in SOURCE_CONFIGS.items():
        flush_interval = cfg.get("session", {}).get("flush_interval_seconds", 30)
        if current_time - last_flush_times[uid] >= flush_interval:
            flush_pending_to_db(uid)
            last_flush_times[uid] = current_time


def _update_fps(uid):
    """更新該 uid 的即時 FPS（滑動視窗 30 幀）。"""
    if "timestamps" not in fps_streams[uid]:
        fps_streams[uid]["timestamps"] = deque(maxlen=30)
    now = time.time()
    q = fps_streams[uid]["timestamps"]
    q.append(now)
    if len(q) > 1:
        fps_streams[uid]["current_fps"] = (len(q) - 1) / (q[-1] - q[0])


# ==========================================
# 3. ctx 解析輔助
# ==========================================

def _resolve_ctx(u_data):
    """
    從 probe 的 u_data 取出 (pad_to_uid, class_map)。

    u_data 由 main.py 建 pipeline 時綁定：
        {"pad_to_uid": [uid0, uid1, ...],  # 索引 = 該 pipeline 區域 pad_index
         "class_map":  {cls_id: name}}
    """
    pad_to_uid = u_data.get("pad_to_uid", [])
    class_map = u_data.get("class_map", {})
    return pad_to_uid, class_map


def _local_pad_to_uid(pad_to_uid, local_pad_index):
    """把區域 pad_index 翻譯成全域 uid；超出範圍回 None（防呆）。"""
    if 0 <= local_pad_index < len(pad_to_uid):
        return pad_to_uid[local_pad_index]
    return None


# ==========================================
# 4. NvDCF 模式探針 (掛 tracker.src)
# ==========================================

def tracker_src_pad_buffer_probe(pad, info, u_data):
    """
    NvDCF 模式專用探針。u_data = {"pad_to_uid":[...], "class_map":{...}}。
    obj_meta 由 nvtracker 提供，已含有效 object_id。
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    pad_to_uid, class_map = _resolve_ctx(u_data)
    group_uids = set(pad_to_uid)          # 本組涵蓋的全域 uid（供清理時只針對本組）

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    current_frame_objects = set()

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        uid = _local_pad_to_uid(pad_to_uid, frame_meta.pad_index)
        cfg = SOURCE_CONFIGS.get(uid) if uid is not None else None
        if cfg is None:
            l_frame = l_frame.next
            continue

        _update_fps(uid)
        _process_tracked_frame(frame_meta, current_frame_objects, uid, cfg, class_map)

        l_frame = l_frame.next

    _post_frame_housekeeping(current_frame_objects, group_uids)
    return Gst.PadProbeReturn.OK


# ==========================================
# 5. BoxMOT 模式探針 (掛 pgie.src)
# ==========================================

def boxmot_pgie_src_probe(pad, info, u_data):
    """
    BoxMOT 模式專用探針。u_data = {"pad_to_uid":[...], "class_map":{...}}。
    流程：抽 PGIE 偵測框 → 清空 obj_meta → 餵 BoxMOT → 用追蹤結果重建 obj_meta → 共用軌跡邏輯。
    追蹤器以「全域 uid」索引（boxmot_adapter._trackers[uid]），多 pipeline 不會撞。
    """
    from logic.boxmot_adapter import track as boxmot_track

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    pad_to_uid, class_map = _resolve_ctx(u_data)
    group_uids = set(pad_to_uid)          # 本組涵蓋的全域 uid（供清理時只針對本組）

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    current_frame_objects = set()

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        uid = _local_pad_to_uid(pad_to_uid, frame_meta.pad_index)
        cfg = SOURCE_CONFIGS.get(uid) if uid is not None else None
        if cfg is None:
            l_frame = l_frame.next
            continue

        _update_fps(uid)
        keep = cfg.get("keep_classes")      # frozenset 或 None（None = 全收）

        # 步驟 1: 抽出所有 PGIE 偵測框
        dets_list = []
        obj_metas_to_remove = []
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            cls = int(obj_meta.class_id)
            # keep_classes 過濾：不在白名單內的偵測框不餵給 BoxMOT（但仍要從 frame 移除）
            if keep is not None and cls not in keep:
                obj_metas_to_remove.append(obj_meta)
                l_obj = l_obj.next
                continue

            try:
                det_box = obj_meta.detector_bbox_info.org_bbox_coords
                x1 = float(det_box.left)
                y1 = float(det_box.top)
                x2 = float(det_box.left + det_box.width)
                y2 = float(det_box.top + det_box.height)
            except Exception:
                r = obj_meta.rect_params
                x1 = float(r.left); y1 = float(r.top)
                x2 = float(r.left + r.width); y2 = float(r.top + r.height)

            conf = float(obj_meta.confidence) if obj_meta.confidence > 0 else 0.5
            dets_list.append([x1, y1, x2, y2, conf, cls])
            obj_metas_to_remove.append(obj_meta)
            l_obj = l_obj.next

        # 步驟 2: 清空 obj_meta
        for om in obj_metas_to_remove:
            pyds.nvds_remove_obj_meta_from_frame(frame_meta, om)

        # 步驟 3: 餵 BoxMOT（用全域 uid 取該路 tracker）
        dets = np.asarray(dets_list, dtype=np.float32) if dets_list else np.empty((0, 6), dtype=np.float32)
        tracks = boxmot_track(uid, dets, frame=None)

        # 步驟 4: 用 BoxMOT 輸出重建 obj_meta
        for tr in tracks:
            x1, y1, x2, y2 = float(tr[0]), float(tr[1]), float(tr[2]), float(tr[3])
            tid = int(tr[4]); conf = float(tr[5]); cls = int(tr[6])

            new_obj = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
            if new_obj is None:
                continue
            new_obj.unique_component_id = 1
            new_obj.class_id = cls
            new_obj.object_id = tid
            new_obj.confidence = conf
            new_obj.obj_label = class_map.get(cls, f"Class_{cls}")

            r = new_obj.rect_params
            r.left = x1
            r.top = y1
            r.width = max(1.0, x2 - x1)
            r.height = max(1.0, y2 - y1)
            r.border_width = 4
            r.has_bg_color = 0
            r.border_color.set(*get_class_color(cls))

            pyds.nvds_add_obj_meta_to_frame(frame_meta, new_obj, None)

        # 步驟 5: 共用軌跡邏輯
        _process_tracked_frame(frame_meta, current_frame_objects, uid, cfg, class_map)

        l_frame = l_frame.next

    _post_frame_housekeeping(current_frame_objects, group_uids)
    return Gst.PadProbeReturn.OK


# ==========================================
# 6. 每路畫面 OSD 探針 (Per-Cam FPS Overlay)
# ==========================================

def per_cam_osd_probe(pad, info, uid):
    """
    每路 nvosd.sink 的 OSD 探針：左上角畫即時 FPS。
    參數 uid：全域 stream_uid（由 setup_cam_branch 綁定）。
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    cfg = SOURCE_CONFIGS.get(uid)
    if not cfg:
        return Gst.PadProbeReturn.OK

    show_fps = cfg.get("display", {}).get("show_fps_overlay", True)

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 0
        display_meta.num_lines = 0
        display_meta.num_rects = 0
        display_meta.num_circles = 0

        if show_fps and uid in fps_streams:
            display_meta.num_labels = 1
            txt_params = display_meta.text_params[0]
            txt_params.display_text = f"FPS: {fps_streams[uid].get('current_fps', 0.0):.1f}"
            txt_params.x_offset = 5
            txt_params.y_offset = 5
            txt_params.font_params.font_name = "Serif Bold"
            txt_params.font_params.font_size = 25
            txt_params.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
            txt_params.set_bg_clr = 1
            txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.8)

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK
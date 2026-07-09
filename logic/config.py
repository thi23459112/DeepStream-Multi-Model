"""
全域設定載入器（車流版 · 多權重分組）

主要功能：
1. 自動偵測專案根目錄 BASE_DIR
2. 載入 ds_yaml/*.yaml，建立全域 SOURCE_CONFIGS 字典（鍵 = 全域 stream_uid，依檔名順序 0..N-1）
3. 依每個 YAML 的 weight 欄位「分組」：相同 engine 的 cam 歸為同一組 → 各跑一條獨立 pipeline
4. 從 engine 檔名「自動推導」對應的 labels 檔（car_fp16.engine → labels_car.txt），
   並為每組載入各自的 class_map（不同 engine 類別數/名稱可不同）
5. 產出 GROUPS 結構，供 main.py / pipeline.py / traffic_count_txt.py 建立多條 pipeline
6. 載入 config_tracker_runtime.txt 決定追蹤器模式；決定合併 DB 路徑 MERGED_DB_PATH

重要觀念（多 pipeline 下的鍵）：
    SOURCE_CONFIGS 的鍵 = 全域 stream_uid（0=camA, 1=camB, 2=camC, 3=camD ...）。
    state_db / boxmot_adapter 全用這個 uid 當鍵，故它們幾乎不用改。
    每條 pipeline 內 DeepStream 給的 frame_meta.pad_index 是「區域」編號（各自從 0 起算），
    需由 probe 用該組的 pad→uid 對照表翻譯回全域 uid（見 GROUPS[gid]["member_uids"]）。
"""

import os
import sys
import glob
import cv2
import yaml
import urllib.parse
import configparser
import numpy as np

from logic.color import load_labels


# ==========================================
# 1. 系統配置區 (System Configuration)
# ==========================================

# 自動偵測專案根目錄 (本檔位於 <project_root>/logic/config.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# YAML 資料夾路徑：可用環境變數 DS_YAML_DIR 覆寫，預設 <root>/ds_yaml
YAML_DIR = os.environ.get("DS_YAML_DIR", f"{BASE_DIR}/ds_yaml")

# --- DeepStream 設定檔路徑 ---
# nvdsanalytics / 追蹤器執行期為「全組共用」一份；
# infer / preprocess 為「每組一份」，路徑由 group_infer_config() / group_preprocess_config() 決定。
ANALYTICS_CONFIG       = f"{BASE_DIR}/config_nvdsanalytics.txt"
TRACKER_CONFIG         = f"{BASE_DIR}/config_tracker_NvDCF_accuracy.yml"   # nvdcf 內建追蹤器設定（BoxMOT 模式用不到）
TRACKER_RUNTIME_CONFIG = f"{BASE_DIR}/config_tracker_runtime.txt"

# --- 支援的 URI scheme（判斷 source 是否已是合法 URI）---
_URI_SCHEMES = ("file://", "rtsp://", "rtsps://", "http://", "https://")

# --- engine 檔名常見的精度後綴（自動推 labels 時要剝掉）---
_ENGINE_PRECISION_SUFFIXES = ("_fp16", "_fp32", "_int8", "_fp8", "_dla", "_dynamic", "_best")

# --- Local ID 循環上限 ---
# 每路 cam 各自的 local_id 累積到此值後歸 1 重新計算（OSD 與 DB TrackID 同步循環）
LOCAL_ID_MAX = 999999

# --- 合併 DB 檔名（所有 cam / 所有組寫進同一個檔）---
MERGED_DB_NAME = os.environ.get("MERGED_DB_NAME", "traffic_count.db")

# --- streammux 輸出尺寸（與 main.py 每組 streammux 的 width/height 一致）---
# 所有來源進 pipeline 後都會被 streammux 縮放到這個尺寸；ROI/crop 的座標系以此為準。
MUX_OUTPUT_W = 1920
MUX_OUTPUT_H = 1080


def _scale_points(points, base_w, base_h):
    """
    把「來源真實解析度(base_w×base_h)」座標系下的點位，換算成 streammux 輸出(1920×1080)座標系。

    用途：YAML 的 ROI / crop_points 讓使用者直接按「來源真實點位」標記，
          本函式依比例自動縮放成 1080P 座標，與 streammux 縮放後的物件座標對齊。

    參數：
        points (list): [[x,y], ...] 來源座標系點位
        base_w/base_h (int): 該來源的真實解析度（YAML geometry.base_w/base_h）
    返回：
        list: [[x,y], ...] 換算成 1920×1080 座標系後的整數點位
    """
    if not points:
        return points
    # base 缺失或等於輸出尺寸 → 比例 1:1，原樣回傳（1080P 來源不受影響）
    if not base_w or not base_h:
        return points
    sx = MUX_OUTPUT_W / float(base_w)
    sy = MUX_OUTPUT_H / float(base_h)
    if sx == 1.0 and sy == 1.0:
        return points
    return [[int(round(p[0] * sx)), int(round(p[1] * sy))] for p in points]


def group_infer_config(group_id):
    """回傳第 group_id 組的 PGIE 設定檔絕對路徑（由 traffic_count_txt.py 產生）。"""
    return f"{BASE_DIR}/config_infer_group{group_id}.txt"


def group_preprocess_config(group_id):
    """回傳第 group_id 組的 nvdspreprocess 設定檔絕對路徑（由 traffic_count_txt.py 產生）。"""
    return f"{BASE_DIR}/config_preprocess_group{group_id}.txt"


# ==========================================
# 2. 由 engine 檔名自動推導 labels 檔
# ==========================================

def _derive_labels_path(engine_filename):
    """
    從 engine 檔名自動推導對應的 labels 檔絕對路徑

    規則：
        <name>[_精度後綴].engine  →  labels_<name>.txt（放在專案根目錄）
        例：
            car_fp16.engine      → labels_car.txt
            yolo11s_fp16.engine  → labels_yolo11s.txt
            car.engine           → labels_car.txt

    參數：
        engine_filename (str): YAML weight 欄位（例如 "car_fp16.engine"）
    返回：
        str: 推導出的 labels 檔絕對路徑（不保證存在，caller 自行檢查）
    """
    # 去掉副檔名（.engine），取出主檔名
    base = os.path.splitext(os.path.basename(engine_filename))[0]  # "car_fp16"

    # 剝掉常見的精度後綴（只剝結尾一個）
    lower = base.lower()
    for suf in _ENGINE_PRECISION_SUFFIXES:
        if lower.endswith(suf):
            base = base[: -len(suf)]
            break

    return os.path.join(BASE_DIR, f"labels_{base}.txt")


# ==========================================
# 3. YAML source 智慧解析 (Source URI Resolver)
# ==========================================

def _resolve_source_uri(raw):
    """
    把 YAML 的 source 欄位轉成 GStreamer 可用的合法 URI

    支援：相對路徑 / 絕對路徑 / ~ 家目錄 / ${BASE_DIR} 樣板 / file:// / rtsp:// / http(s)://
    參數：raw (str) YAML 原始字串
    返回：str 解析後 URI；空字串代表 raw 不合法
    """
    if not isinstance(raw, str) or not raw:
        return ""

    s = raw.strip()
    if "${BASE_DIR}" in s:
        s = s.replace("${BASE_DIR}", BASE_DIR)

    # 已是合法 URI → 原樣回傳
    if s.startswith(_URI_SCHEMES):
        return s

    # 視為檔案路徑：展開家目錄、補絕對路徑
    if s.startswith("~"):
        s = os.path.expanduser(s)
    s = os.path.normpath(s if os.path.isabs(s) else os.path.join(BASE_DIR, s))

    if not os.path.exists(s):
        print(f"[WARNING] source 對應的檔案不存在：{s}")
        print(f"[WARNING]   原始 YAML 寫法：'{raw}'")

    return f"file://{s}"


def _parse_start_time(start_time_str):
    """把 "YYYY-MM-DD HH:MM:SS" 字串解析成 datetime；失敗回 None。"""
    if not start_time_str:
        return None
    try:
        from datetime import datetime
        return datetime.strptime(str(start_time_str), "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"[WARNING] start_time 格式錯誤 ({start_time_str})，將忽略此欄位: {e}")
        return None


# ==========================================
# 4. YAML 載入與後處理 (YAML Loader)
# ==========================================

def load_dynamic_configs(yaml_dir):
    """
    讀取 ds_yaml/*.yaml，逐檔解析、補衍生欄位，並依 weight 分組

    每個 cfg 會補上（節錄）：
        - stream_uid   : 全域唯一編號（= 載入順序，也是 SOURCE_CONFIGS 的鍵）
        - group_id     : 所屬組別（相同 weight 同一組）
        - device_code / cv_regions / track_logic / save_output_db / db_dir
        - video_path / source / stream_fps / is_file_source / start_time_dt
        - weight / weight_imgsz / weight_batch_size / labels_path

    參數：yaml_dir (str)
    返回：
        tuple (configs, groups, merged_db_path)
            configs        : {stream_uid: cfg_dict}
            groups         : {group_id: group_info_dict}（見下方 _build_groups）
            merged_db_path : 合併 DB 的 .db 絕對路徑
    """
    files = sorted(glob.glob(f"{yaml_dir}/*.yaml"))
    if not files:
        print(f"[ERROR] 找不到任何 YAML 檔於 {yaml_dir}")
        sys.exit(1)

    configs = {}
    merged_db_dir = None

    for stream_uid, f in enumerate(files):
        with open(f, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        cam_name = data.get("source_id", f"cam_{stream_uid}")
        data["stream_uid"] = stream_uid          # 全域唯一鍵

        # --- 裝置代碼 ---
        device_cfg = data.get("device", {}) or {}
        data["device_code"] = str(device_cfg.get("code", "UNKNOWN"))

        # --- ROI / 裁切座標自動換算成 streammux 輸出(1920×1080)座標系 ---
        # 使用者在 YAML 按「來源真實解析度」標點位，這裡依 base_w/base_h 比例自動縮放，
        # 與 streammux 縮放後的物件座標對齊（1080P 來源比例 1:1，數字不變）。
        geo = data.get("geometry", {}) or {}
        base_w = int(geo.get("base_w", MUX_OUTPUT_W))
        base_h = int(geo.get("base_h", MUX_OUTPUT_H))

        # regions（多 ROI）就地縮放
        regions_src = geo.get("regions", {}) or {}
        regions_scaled = {}
        for roi_name, pts in regions_src.items():
            regions_scaled[roi_name] = _scale_points(pts, base_w, base_h) if pts else pts

        # crop_points 就地縮放（給 traffic_count_txt.py 的 preprocess 裁切用）
        crop_src = geo.get("crop_points")
        crop_scaled = _scale_points(crop_src, base_w, base_h) if crop_src else crop_src

        if (base_w, base_h) != (MUX_OUTPUT_W, MUX_OUTPUT_H):
            print(f"[INFO] {cam_name} 來源座標 {base_w}x{base_h} → 自動換算成 "
                  f"{MUX_OUTPUT_W}x{MUX_OUTPUT_H}（ROI/crop 依比例縮放）")

        # 回寫縮放後的 geometry，並把 base_w/base_h 標準化為輸出尺寸
        #（讓 nvdsanalytics 的 config-width/height 與 ROI 座標系一致、視覺線不歪）
        geo["regions"] = regions_scaled
        if crop_src:
            geo["crop_points"] = crop_scaled
        geo["base_w"] = MUX_OUTPUT_W
        geo["base_h"] = MUX_OUTPUT_H
        data["geometry"] = geo

        # --- 多 ROI 轉 numpy（此時 regions 已是 1080P 座標）---
        regions_raw = geo.get("regions", {}) or {}
        cv_regions = {}
        for roi_name, pts in regions_raw.items():
            if pts and len(pts) >= 3:
                cv_regions[str(roi_name)] = np.array(pts, np.int32)
            else:
                print(f"[WARNING] {cam_name} 的 ROI '{roi_name}' 點數不足，略過")
        data["cv_regions"] = cv_regions
        if not cv_regions:
            print(f"[WARNING] {cam_name} 沒有任何有效的 ROI，將不會產生任何 DB 紀錄")

        # --- track_logic 標準化 ---
        tl_cfg = data.get("track_logic", {}) or {}
        axis = str(tl_cfg.get("axis", "y")).lower().strip()
        if axis not in ("x", "y"):
            print(f"[WARNING] {cam_name} track_logic.axis 值非法（'{axis}'），退回 'y'")
            axis = "y"
        data["track_logic"] = {
            "axis":               axis,
            "movement_threshold": int(tl_cfg.get("movement_threshold", 30)),
            "min_roi_hits":       int(tl_cfg.get("min_roi_hits", 2)),
            "up_left_is_out":     bool(tl_cfg.get("up_left_is_out", True)),
        }

        # --- 模型 Engine + 自動推 labels ---
        weight = str(data.get("weight", "car_fp16.engine")).strip()
        data["weight"]            = weight
        data["weight_imgsz"]      = int(data.get("weight_imgsz", 640))
        data["weight_batch_size"] = int(data.get("weight_batch_size", 4))
        # labels：YAML 若有寫 labels: 就用它（可選覆寫），否則從 engine 檔名自動推導
        labels_yaml = data.get("labels")
        if labels_yaml:
            lp = labels_yaml if os.path.isabs(labels_yaml) else os.path.join(BASE_DIR, labels_yaml)
            data["labels_path"] = lp
        else:
            data["labels_path"] = _derive_labels_path(weight)

        # keep_classes：這一路只保留哪些「模型原始 class_id」
        #   有寫非空清單 → 標準化成 frozenset，probe 只處理清單內的 class_id，其餘偵測框丟掉
        #   沒寫 / 空清單   → None，代表全收（不過濾）
        #   填的是該 engine 輸出的原始 class_id（car 自訓 0~6；yolo11s COCO person=0）
        keep_raw = data.get("keep_classes")
        if keep_raw:
            data["keep_classes"] = frozenset(int(c) for c in keep_raw)
        else:
            data["keep_classes"] = None

        # --- 輸出設定 - DB（合併版）---
        output_cfg = data.get("output", {}) or {}
        db_dir = output_cfg.get("output_db_dir") or output_cfg.get("output_excel_dir", "output_db")
        if not os.path.isabs(db_dir):
            db_dir = os.path.join(BASE_DIR, db_dir)
        data["save_output_db"] = bool(output_cfg.get("save_output_db", True))
        data["db_dir"] = db_dir
        if merged_db_dir is None:
            merged_db_dir = db_dir

        # --- 輸出設定 - 影片 ---
        video_dir = output_cfg.get("output_video_dir", "output_video")
        if not os.path.isabs(video_dir):
            video_dir = os.path.join(BASE_DIR, video_dir)
        if output_cfg.get("save_output_video", False):
            os.makedirs(video_dir, exist_ok=True)
        data["video_path"] = os.path.join(video_dir, f"{cam_name}_output.mkv")

        # --- 來源 URI 解析 ---
        source_uri = _resolve_source_uri(data.get("source", ""))
        yaml_fps   = data.get("stream_fps", 30.0)
        original   = data.get("source", "")
        if original != source_uri:
            print(f"[INFO] {cam_name} source 解析: '{original}' → '{source_uri}'")

        # RTSP 來源帳密安全編碼
        if source_uri.startswith("rtsp://"):
            try:
                parsed = urllib.parse.urlparse(source_uri)
                if parsed.username and parsed.password:
                    su = urllib.parse.quote(urllib.parse.unquote(parsed.username))
                    sp = urllib.parse.quote(urllib.parse.unquote(parsed.password))
                    netloc = f"{su}:{sp}@{parsed.hostname}" + (f":{parsed.port}" if parsed.port else "")
                    source_uri = urllib.parse.urlunparse(
                        (parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
                    )
                    print(f"[INFO] RTSP 來源 URI 安全格式: {source_uri}")
            except Exception as e:
                print(f"[WARNING] 解析 RTSP 來源 URI 失敗: {e}")

        # 檔案模式：用 cv2 抓真實 FPS 覆寫
        if source_uri.startswith("file://"):
            cap = cv2.VideoCapture(source_uri.replace("file://", ""))
            try:
                if cap.isOpened():
                    real_fps = cap.get(cv2.CAP_PROP_FPS)
                    if real_fps > 0:
                        yaml_fps = real_fps
            finally:
                cap.release()

        data["stream_fps"]     = yaml_fps
        data["source"]         = source_uri
        data["is_file_source"] = source_uri.startswith("file://")

        # start_time（僅檔案模式）
        data["start_time_dt"] = _parse_start_time(
            data.get("start_time") if data["is_file_source"] else None
        )

        configs[stream_uid] = data

    # --- 合併 DB 路徑 ---
    if merged_db_dir is None:
        merged_db_dir = os.path.join(BASE_DIR, "output_db")
    merged_db_path = os.path.join(merged_db_dir, MERGED_DB_NAME)
    if any(c.get("save_output_db", True) for c in configs.values()):
        os.makedirs(merged_db_dir, exist_ok=True)
    for c in configs.values():
        c["db_path"]    = merged_db_path   # 向下相容鍵（各路皆指向同一合併 DB）
        c["excel_path"] = merged_db_path

    # --- 依 weight 分組 ---
    groups = _build_groups(configs)

    return configs, groups, merged_db_path


def _build_groups(configs):
    """
    依 weight 把 configs 分組，每組載入各自的 class_map

    參數：
        configs (dict): {stream_uid: cfg}
    返回：
        dict: {group_id: {
                 "group_id":    int,
                 "weight":      str,            # engine 檔名
                 "engine_path": str,            # engine 絕對路徑（BASE_DIR/weight）
                 "labels_path": str,            # 對應 labels 檔
                 "class_map":   dict,           # {class_id: name}（該組專屬）
                 "num_classes": int,
                 "imgsz":       int,
                 "batch":       int,            # engine max batch（該組）
                 "member_uids": [stream_uid,...] # 屬於本組的全域 uid（依 uid 排序）
                                                 # 其索引 = 該組 pipeline 內的區域 pad_index
               }}
    """
    order = []          # 依「首次出現順序」決定 group_id
    weight_to_gid = {}
    groups = {}

    for uid in sorted(configs.keys()):
        cfg = configs[uid]
        weight = cfg["weight"]

        if weight not in weight_to_gid:
            gid = len(order)
            weight_to_gid[weight] = gid
            order.append(weight)

            labels_path = cfg["labels_path"]
            class_map = load_labels(labels_path)   # color.load_labels：檔案不存在會 WARNING 並回空 dict

            groups[gid] = {
                "group_id":    gid,
                "weight":      weight,
                "engine_path": os.path.join(BASE_DIR, weight),
                "labels_path": labels_path,
                "class_map":   class_map,
                "num_classes": len(class_map) if class_map else 0,
                "imgsz":       cfg["weight_imgsz"],
                "batch":       cfg["weight_batch_size"],
                "member_uids": [],
            }

        gid = weight_to_gid[weight]
        cfg["group_id"] = gid
        groups[gid]["member_uids"].append(uid)

    # 每組把 member_uids 依 uid 排序（其索引即該組 pipeline 的區域 pad_index）
    for g in groups.values():
        g["member_uids"].sort()

    return groups


# ==========================================
# 5. 追蹤器執行期設定 (Tracker Runtime Config)
# ==========================================

def load_tracker_runtime():
    """
    讀 config_tracker_runtime.txt 取得當前追蹤器模式

    返回：(mode, boxmot_config_path)
        mode (str): "nvdcf" 或 BoxMOT 追蹤器名稱
        boxmot_config_path (str|None): BoxMOT 模式為設定檔絕對路徑，nvdcf 為 None
    """
    if not os.path.exists(TRACKER_RUNTIME_CONFIG):
        print(f"[INFO] 找不到 {TRACKER_RUNTIME_CONFIG}，預設使用 nvdcf 模式")
        return "nvdcf", None

    parser = configparser.ConfigParser()
    try:
        parser.read(TRACKER_RUNTIME_CONFIG, encoding="utf-8")
    except Exception as e:
        print(f"[WARNING] 解析 {TRACKER_RUNTIME_CONFIG} 失敗：{e}，退回 nvdcf")
        return "nvdcf", None

    if not parser.has_section("tracker"):
        print(f"[WARNING] {TRACKER_RUNTIME_CONFIG} 缺 [tracker] 區塊，退回 nvdcf")
        return "nvdcf", None

    mode       = parser.get("tracker", "mode", fallback="nvdcf").lower().strip()
    boxmot_cfg = parser.get("tracker", "config", fallback=None)
    if boxmot_cfg is not None:
        boxmot_cfg = boxmot_cfg.strip() or None
    return mode, boxmot_cfg


# ==========================================
# 6. 模組初始化 (Module Initialization)
# ==========================================

print(f"[INFO] [config.py] BASE_DIR 自動偵測 = {BASE_DIR}")

SOURCE_CONFIGS, GROUPS, MERGED_DB_PATH = load_dynamic_configs(YAML_DIR)

# 印出分組結果，方便啟動時確認
print(f"[INFO] 合併 DB 路徑：{MERGED_DB_PATH}")
print(f"[INFO] 共 {len(SOURCE_CONFIGS)} 路 cam，分成 {len(GROUPS)} 組（= {len(GROUPS)} 條 pipeline）：")
for gid, g in GROUPS.items():
    members = ", ".join(SOURCE_CONFIGS[u].get("source_id", f"uid{u}") for u in g["member_uids"])
    exists = "OK" if os.path.exists(g["engine_path"]) else "❌找不到"
    lbl_exists = "OK" if os.path.exists(g["labels_path"]) else "❌找不到"
    print(f"[INFO]   group{gid}: engine={g['weight']}({exists}), "
          f"labels={os.path.basename(g['labels_path'])}({lbl_exists}), "
          f"類別數={g['num_classes']}, batch={g['batch']}, imgsz={g['imgsz']}, "
          f"成員=[{members}]（區域 pad 0..{len(g['member_uids'])-1}）")

TRACKER_MODE, BOXMOT_TRACKER_CONFIG = load_tracker_runtime()
print(f"[INFO] 追蹤器模式：{TRACKER_MODE}")
if TRACKER_MODE != "nvdcf":
    print(f"[INFO] BoxMOT 追蹤器設定來源：{BOXMOT_TRACKER_CONFIG}")

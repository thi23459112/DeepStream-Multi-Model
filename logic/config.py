"""
設定載入

import 時讀一次 ds_yaml/*.yaml，產生整個專案共用的設定：

    SOURCE_CONFIGS         {uid: cfg}，每路攝影機的設定
    GROUPS                 {gid: group}，依 weight 分組；一組一條 pipeline
    MERGED_DB_PATH         所有攝影機共用的 SQLite 檔路徑
    TRACKER_MODE           "nvdcf" 或 BoxMOT 追蹤器名稱
    BOXMOT_TRACKER_CONFIG  BoxMOT 追蹤器參數檔路徑（nvdcf 時為 None）

兩種編號：
    uid        全域編號。依 YAML 檔名排序，0 起算，跨所有組唯一。
               軌跡狀態、DB、BoxMOT 追蹤器、元件命名都用它。
    pad_index  區域編號。該攝影機在 GROUPS[gid]["member_uids"] 裡的索引，各組從 0 起算，
               也就是 streammux / demux 的 pad 編號。探針拿到的 frame_meta.pad_index 是這個。

cfg 除了 YAML 原有欄位，另外整理出：
    device_code     device.code
    geometry        regions / crop_points 已換算成 1920×1080 座標
    cv_regions      {ROI 名稱: np.int32 多邊形}，探針判斷命中用
    track_logic     {axis, movement_threshold, min_roi_hits, up_left_is_out}，已補預設值
    keep_classes    frozenset(class_id)；None 表示全部保留
    weight          engine 路徑（相對於專案根目錄）
    labels_path     labels 檔絕對路徑
    source          GStreamer 可用的 URI（RTSP 帳密已編碼）
    is_file_source  來源是否為本地檔案
    stream_fps      檔案來源為影片實際 FPS，其餘為 YAML 的值
    start_time_dt   檔案來源第一幀對應的真實時間（datetime 或 None）
    save_output_db  是否寫入 DB
    video_path      輸出影片的完整路徑
    group_id        所屬組別

各欄位的預設值與 README「YAML 欄位」一致。
"""

import os
import sys
import glob
import urllib.parse
from datetime import datetime

import cv2
import yaml
import numpy as np

from logic.color import load_labels
from logic.common import (
    MUX_OUTPUT_W, MUX_OUTPUT_H,
    scale_points, derive_labels_filename, yaml_fingerprint,
)


# =============================================================================
# 路徑與常數
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 專案根目錄（logic/ 的上一層）
YAML_DIR = os.environ.get("DS_YAML_DIR", os.path.join(BASE_DIR, "ds_yaml"))

# nvdcf 模式下 nvtracker 的參數檔
TRACKER_CONFIG = os.path.join(BASE_DIR, "config_tracker_NvDCF_accuracy.yml")

# 可用的 BoxMOT 追蹤器（只限不需要影像像素的，見 logic/boxmot_adapter.py）
SUPPORTED_BOXMOT_TRACKERS = ("bytetrack", "ocsort", "fasttracker", "sfsort", "cbiou")

# 每路攝影機的短 ID（畫面上的 ID、DB 的 TrackID）上限，超過後從 1 重新編號
LOCAL_ID_MAX = 999999

# 所有攝影機共用的 DB 檔名
MERGED_DB_NAME = os.environ.get("MERGED_DB_NAME", "traffic_count.db")

# 產生的 nvinfer 設定檔第一行的前綴，後面接 ds_yaml 的指紋
FINGERPRINT_PREFIX = "# yaml-fingerprint="

# 已經是 URI 的 source 原樣使用，其餘視為檔案路徑
_URI_SCHEMES = ("file://", "rtsp://", "rtsps://", "http://", "https://")


def group_infer_config(group_id):
    """第 group_id 組的 nvinfer 設定檔路徑。"""
    return os.path.join(BASE_DIR, f"config_infer_group{group_id}.txt")


def group_preprocess_config(group_id):
    """第 group_id 組的 nvdspreprocess 設定檔路徑。"""
    return os.path.join(BASE_DIR, f"config_preprocess_group{group_id}.txt")


def group_analytics_config(group_id):
    """第 group_id 組的 nvdsanalytics 設定檔路徑。"""
    return os.path.join(BASE_DIR, f"config_nvdsanalytics_group{group_id}.txt")


# =============================================================================
# 來源解析
# =============================================================================

def _resolve_source_uri(raw):
    """
    把 YAML 的 source 轉成 GStreamer URI。

    已是 URI 的原樣回傳；其餘視為檔案路徑，
    支援相對路徑（相對於專案根目錄）、絕對路徑、~、${BASE_DIR}。
    """
    if not isinstance(raw, str) or not raw.strip():
        return ""
    s = raw.strip().replace("${BASE_DIR}", BASE_DIR)
    if s.startswith(_URI_SCHEMES):
        return s                # rtsp:// 等直接使用

    s = os.path.expanduser(s)   # 展開 ~
    s = os.path.normpath(s if os.path.isabs(s) else os.path.join(BASE_DIR, s))
    if not os.path.exists(s):
        print(f"[WARNING] source 指向的檔案不存在：{s}（YAML 寫法：'{raw}'）")
    return f"file://{s}"


def _encode_rtsp_credentials(uri):
    """
    把 RTSP 帳密做 URL 編碼，密碼含 @ : / 等字元時 GStreamer 才解析得了。

    回傳：
        (編碼後的 URI, 密碼遮蔽成 *** 的 URI)；後者只用於印 log
    """
    try:
        p = urllib.parse.urlparse(uri)
        if not (p.username and p.password):
            return uri, uri                 # 沒有帳密，不必處理
        # 先 unquote 再 quote：YAML 裡已經編碼過的帳密不會被重複編碼
        user = urllib.parse.quote(urllib.parse.unquote(p.username), safe="")
        pwd = urllib.parse.quote(urllib.parse.unquote(p.password), safe="")
        host = p.hostname + (f":{p.port}" if p.port else "")
        rest = (p.path, p.params, p.query, p.fragment)
        encoded = urllib.parse.urlunparse((p.scheme, f"{user}:{pwd}@{host}", *rest))
        masked = urllib.parse.urlunparse((p.scheme, f"{user}:***@{host}", *rest))
        return encoded, masked
    except Exception as e:
        print(f"[WARNING] 解析 RTSP URI 失敗：{e}")
        return uri, "rtsp://(解析失敗)"     # 解析失敗時不印出原始帳密


def _file_fps(uri, fallback):
    """讀本地影片的實際 FPS；讀不到時回傳 fallback。"""
    cap = cv2.VideoCapture(uri[len("file://"):])    # OpenCV 要的是一般路徑，去掉 file://
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
    finally:
        cap.release()
    return fps if fps > 0 else fallback


def _parse_start_time(text):
    """解析 "YYYY-MM-DD HH:MM:SS"；空值或格式錯誤回傳 None。"""
    if not text:
        return None
    try:
        return datetime.strptime(str(text), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print(f"[WARNING] start_time 格式錯誤（{text}），應為 YYYY-MM-DD HH:MM:SS，已忽略")
        return None


# =============================================================================
# 載入 YAML
# =============================================================================

def _load_one(path):
    """讀一個 YAML，整理成 cfg（欄位見檔頭）。"""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}                                  # 空檔案時 safe_load 回傳 None

    name = cfg.setdefault("source_id", os.path.splitext(os.path.basename(path))[0])

    # 這些區塊一律整理成 dict，之後取值不必再處理 None
    for section in ("device", "detect", "geometry", "session", "track_logic",
                    "output", "display", "tracker"):
        cfg[section] = dict(cfg.get(section) or {})

    cfg["device_code"] = str(cfg["device"].get("code", "UNKNOWN"))

    # ---- 區域：由標記用的解析度換算成 streammux 輸出座標 ----
    geo = cfg["geometry"]
    base_w = int(geo.get("base_w", MUX_OUTPUT_W))                       # 標記點位時的畫面寬
    base_h = int(geo.get("base_h", MUX_OUTPUT_H))                       # 標記點位時的畫面高
    geo["regions"] = {
        str(roi): scale_points(pts, base_w, base_h)
        for roi, pts in (geo.get("regions") or {}).items()
    }
    geo["crop_points"] = scale_points(geo.get("crop_points"), base_w, base_h)
    if (base_w, base_h) != (MUX_OUTPUT_W, MUX_OUTPUT_H):
        print(f"[INFO] {name} 點位以 {base_w}x{base_h} 標記，已換算成 {MUX_OUTPUT_W}x{MUX_OUTPUT_H}")

    cfg["cv_regions"] = {}
    for roi, pts in geo["regions"].items():
        if pts and len(pts) >= 3:                                       # 多邊形至少 3 點
            cfg["cv_regions"][roi] = np.array(pts, np.int32)            # cv2.pointPolygonTest 要 int32
        else:
            print(f"[WARNING] {name} 的 ROI '{roi}' 少於 3 個點，已略過")
    if not cfg["cv_regions"]:
        print(f"[WARNING] {name} 沒有有效的 ROI，這一路不會產生任何計數")

    # ---- 方向判定 ----
    tl = cfg["track_logic"]
    axis = str(tl.get("axis", "y")).lower().strip()
    if axis not in ("x", "y"):
        print(f"[WARNING] {name} 的 track_logic.axis='{axis}' 不合法，改用 'y'")
        axis = "y"
    cfg["track_logic"] = {
        "axis":               axis,                                     # y = 上下、x = 左右
        "movement_threshold": int(tl.get("movement_threshold", 30)),    # 像素（1920×1080 座標）
        "min_roi_hits":       int(tl.get("min_roi_hits", 2)),           # 幀
        "up_left_is_out":     bool(tl.get("up_left_is_out", True)),     # true：往上 / 往左 = flow_out
    }

    # ---- 模型 ----
    cfg["weight"] = str(cfg.get("weight", "car_fp16.engine")).strip()
    labels = cfg.get("labels") or derive_labels_filename(cfg["weight"])
    cfg["labels_path"] = labels if os.path.isabs(labels) else os.path.join(BASE_DIR, labels)
    keep = cfg.get("keep_classes")
    # None = 全部保留
    cfg["keep_classes"] = frozenset(int(c) for c in keep) if keep else None

    # ---- 輸出 ----
    out = cfg["output"]
    cfg["save_output_db"] = bool(out.get("save_output_db", True))
    video_dir = out.get("output_video_dir", "output_video")
    if not os.path.isabs(video_dir):
        video_dir = os.path.join(BASE_DIR, video_dir)
    if out.get("save_output_video", False):
        os.makedirs(video_dir, exist_ok=True)
    cfg["video_path"] = os.path.join(video_dir, f"{name}_output.mkv")   # mkv 的原因見 pipeline._build_save_branch

    # ---- 來源 ----
    uri = _resolve_source_uri(cfg.get("source", ""))
    if uri.startswith(("rtsp://", "rtsps://")):
        uri, masked = _encode_rtsp_credentials(uri)
        print(f"[INFO] {name} 來源：{masked}")                          # log 只印遮蔽後的帳密
    else:
        print(f"[INFO] {name} 來源：{uri}")
    cfg["source"] = uri
    cfg["is_file_source"] = uri.startswith("file://")

    fps = float(cfg.get("stream_fps", 30.0))
    cfg["stream_fps"] = _file_fps(uri, fps) if cfg["is_file_source"] else fps
    # start_time 只對影片檔有意義；RTSP 一律用寫入當下的系統時間
    cfg["start_time_dt"] = _parse_start_time(cfg.get("start_time")) if cfg["is_file_source"] else None

    return cfg


def _load_all(yaml_dir):
    """
    讀 yaml_dir 下所有 *.yaml（不含子資料夾）。

    回傳：
        (configs, merged_db_path)
    """
    files = sorted(glob.glob(os.path.join(yaml_dir, "*.yaml")))     # 排序決定 uid
    if not files:
        sys.exit(f"[ERROR] {yaml_dir} 底下沒有任何 .yaml")

    configs = {uid: _load_one(path) for uid, path in enumerate(files)}

    # 所有攝影機寫同一個 DB，資料夾取檔名排序第一個 YAML 的設定
    out = configs[0]["output"]
    # output_excel_dir：舊版欄位名
    db_dir = out.get("output_db_dir") or out.get("output_excel_dir", "output_db")
    if not os.path.isabs(db_dir):
        db_dir = os.path.join(BASE_DIR, db_dir)
    return configs, os.path.join(db_dir, MERGED_DB_NAME)


def _build_groups(configs):
    """
    依 weight 分組，group_id 依 weight 首次出現的順序從 0 編號。

    回傳：
        {gid: {
            "weight":      engine 路徑（相對於專案根目錄）,
            "engine_path": engine 絕對路徑,
            "labels_path": labels 檔絕對路徑（取該組第一個成員的）,
            "class_map":   {class_id: 名稱},
            "member_uids": [uid, ...]，索引 = pad_index
        }}
    """
    groups, gid_of = {}, {}                                 # gid_of：weight → gid
    for uid in sorted(configs):
        cfg = configs[uid]
        weight = cfg["weight"]
        if weight not in gid_of:
            gid = gid_of[weight] = len(groups)              # 新的 weight 開新組
            groups[gid] = {
                "weight":      weight,
                "engine_path": os.path.join(BASE_DIR, weight),
                "labels_path": cfg["labels_path"],
                "class_map":   load_labels(cfg["labels_path"]),
                "member_uids": [],
            }
        cfg["group_id"] = gid_of[weight]
        groups[cfg["group_id"]]["member_uids"].append(uid)  # append 順序 = pad_index
    return groups


def _resolve_tracker(configs):
    """
    決定追蹤器：全部攝影機共用一種，取檔名排序第一個 YAML 的 tracker.type。

    回傳：
        (mode, boxmot_config_path)；nvdcf 時 boxmot_config_path 為 None
    """
    mode = str(configs[0]["tracker"].get("type", "nvdcf")).lower().strip()
    if mode == "nvdcf":
        return mode, None
    if mode not in SUPPORTED_BOXMOT_TRACKERS:
        sys.exit(f"[ERROR] 不支援的 tracker.type='{mode}'，"
                 f"可用值：nvdcf, {', '.join(SUPPORTED_BOXMOT_TRACKERS)}")
    path = os.path.join(BASE_DIR, "boxmot", "configs", "trackers", f"{mode}.yaml")
    if not os.path.exists(path):
        sys.exit(f"[ERROR] 找不到 BoxMOT 追蹤器設定：{path}")
    return mode, path


# =============================================================================
# 啟動前檢查（由 main.py 呼叫）
# =============================================================================

def check_generated_configs():
    """
    確認 traffic_count_txt.py 產生的設定檔存在，而且是由目前這份 YAML 產生的。

    不符就印出原因並結束，避免用舊設定啟動後才在 nvinfer 裡出現難懂的錯誤。
    """
    problems = []
    for gid, group in GROUPS.items():
        if not os.path.exists(group["engine_path"]):
            problems.append(f"group{gid}：找不到 engine {group['engine_path']}")
        for path in (group_infer_config(gid), group_preprocess_config(gid),
                     group_analytics_config(gid)):
            if not os.path.exists(path):
                problems.append(f"group{gid}：缺少 {os.path.basename(path)}")

    if not problems:
        with open(group_infer_config(0), "r", encoding="utf-8") as fh:
            first_line = fh.readline().strip()  # 產生器寫的指紋在第一行
        if first_line != FINGERPRINT_PREFIX + yaml_fingerprint(YAML_DIR):
            problems.append("ds_yaml 的內容在上次產生設定檔之後有變動")

    if problems:
        print("[ERROR] 設定檔與目前的 ds_yaml 不一致：")
        for p in problems:
            print(f"[ERROR]   - {p}")
        sys.exit("[ERROR] 請先執行：python traffic_count_txt.py")


def disable_windows_without_display():
    """
    沒有圖形顯示時，自動關閉所有預覽視窗。

    未設 DISPLAY / WAYLAND_DISPLAY（例如純 SSH）卻開視窗，顯示 sink 會在啟動時無聲卡住。
    確定有顯示、只是環境變數沒帶到時，設 DS_FORCE_DISPLAY=1 可略過這個檢查。
    """
    if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
            or os.environ.get("DS_FORCE_DISPLAY") == "1"):
        return
    wanted = [c for c in SOURCE_CONFIGS.values() if c["display"].get("show_window", True)]
    if not wanted:
        return  # 本來就都不開視窗
    for cfg in wanted:
        cfg["display"]["show_window"] = False
    print("[WARNING] 偵測不到圖形顯示（DISPLAY / WAYLAND_DISPLAY 未設定），已自動關閉預覽視窗："
          + ", ".join(c["source_id"] for c in wanted))
    print("[WARNING]   要看畫面請先 export DISPLAY=:0（或在本機桌面的終端機執行）再啟動")


# =============================================================================
# 模組初始化
# =============================================================================

SOURCE_CONFIGS, MERGED_DB_PATH = _load_all(YAML_DIR)
GROUPS = _build_groups(SOURCE_CONFIGS)
TRACKER_MODE, BOXMOT_TRACKER_CONFIG = _resolve_tracker(SOURCE_CONFIGS)

print(f"[INFO] 共 {len(SOURCE_CONFIGS)} 路 cam，分成 {len(GROUPS)} 組（每組一條 pipeline），"
      f"追蹤器：{TRACKER_MODE}")
for _gid, _g in GROUPS.items():
    _members = ", ".join(SOURCE_CONFIGS[u]["source_id"] for u in _g["member_uids"])
    print(f"[INFO]   group{_gid}: engine={_g['weight']}, "
          f"labels={os.path.basename(_g['labels_path'])}（{len(_g['class_map'])} 類）, "
          f"成員=[{_members}]")

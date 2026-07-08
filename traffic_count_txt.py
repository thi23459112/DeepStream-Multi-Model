#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepStream 設定檔自動產生器（車流版 · 多權重 / 多 pipeline）

依 ds_yaml/*.yaml 的 weight 欄位「分組」：相同 engine 的 cam 一組 → 各跑一條 pipeline。
每組各產生 3 份設定檔（{N} = group_id，從 0 起算）：
    config_infer_group{N}.txt          PGIE（該組 engine、該組類別數）
    config_preprocess_group{N}.txt     前處理（裁切 ROI、tensor 規格）
    config_nvdsanalytics_group{N}.txt  ROI 繪製（stream id 用該組「區域」索引 0..k-1）
另外全組共用一份：
    config_tracker_runtime.txt         追蹤器執行期旗標（讀 cfgs[0].tracker.type）

labels 自動推導：weight="car_fp16.engine" → labels_car.txt（可用 YAML labels: 覆寫）。
"""

import os
import sys
import glob
import yaml
from typing import List, Dict, Any, Tuple


# ==========================================
# 1. 系統配置區
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_DIR = os.environ.get("DS_YAML_DIR", f"{BASE_DIR}/ds_yaml")

DEFAULT_WEIGHT_IMGSZ = 640
DEFAULT_WEIGHT_BATCH = 4

TRACKER_RUNTIME_CONFIG = f"{BASE_DIR}/config_tracker_runtime.txt"
MUX_CONFIG             = f"{BASE_DIR}/config_mux.txt"

# --- 新版 nvstreammux (USE_NEW_NVSTREAMMUX=yes) 用 ---
# overall-min-fps 自動取各 YAML stream_fps 最高值，但不低於此下限（離線解碼快於即時，保底 30）。
MUX_MIN_FPS_FLOOR   = 30
MUX_OVERALL_MAX_FPS = 120


def _ds_lib_root() -> str:
    """
    自動偵測 DeepStream 的 lib 根目錄（跨平台 / 跨版本）。
    順序：環境變數 DS_LIB_ROOT → /opt/nvidia/deepstream/deepstream/lib →
          /opt/nvidia/deepstream/deepstream*/lib（glob 取最後）→ 標準路徑保底。
    """
    env = os.environ.get("DS_LIB_ROOT", "").strip()
    if env and os.path.isdir(env):
        return env
    std = "/opt/nvidia/deepstream/deepstream/lib"
    if os.path.isdir(std):
        return std
    hits = sorted(glob.glob("/opt/nvidia/deepstream/deepstream*/lib"))
    if hits:
        return hits[-1]
    return std


def resolve_preprocess_lib() -> str:
    """自動解析 nvdspreprocess 的 custom-lib-path（找不到時以標準 nvidia 路徑保底）。"""
    return os.path.join(_ds_lib_root(), "gst-plugins", "libcustom2d_preprocess.so")

# engine 檔名常見精度後綴（自動推 labels 時剝掉；需與 logic/config.py 一致）
_ENGINE_PRECISION_SUFFIXES = ("_fp16", "_fp32", "_int8", "_fp8", "_dla", "_dynamic", "_best")

# 已實作的 BoxMOT 追蹤器白名單（A/B 級）
SUPPORTED_BOXMOT_TRACKERS = ["bytetrack", "ocsort", "fasttracker", "sfsort", "cbiou"]

_URI_SCHEMES = ("file://", "rtsp://", "rtsps://", "http://", "https://")

# streammux 輸出尺寸（與 main.py 每組 streammux 及 logic/config.py 一致）
MUX_OUTPUT_W = 1920
MUX_OUTPUT_H = 1080


def _scale_points(points, base_w, base_h):
    """
    把「來源真實解析度(base_w×base_h)」座標系的點位換算成 1920×1080 座標系。
    base 缺失或已等於輸出尺寸 → 原樣回傳（1080P 來源比例 1:1，數字不變）。
    """
    if not points or not base_w or not base_h:
        return points
    sx = MUX_OUTPUT_W / float(base_w)
    sy = MUX_OUTPUT_H / float(base_h)
    if sx == 1.0 and sy == 1.0:
        return points
    return [[int(round(p[0] * sx)), int(round(p[1] * sy))] for p in points]


def _normalize_geometry_inplace(cfg: Dict[str, Any]) -> None:
    """
    把單一 cam 的 geometry（regions / crop_points）依 base_w/base_h 縮放成 1920×1080 座標，
    並將 base_w/base_h 標準化為輸出尺寸。與 logic/config.py 的換算完全一致，
    確保 preprocess 裁切框、nvdsanalytics ROI 線、probe 計數用的座標三者對齊。
    """
    geo = cfg.get("geometry", {}) or {}
    base_w = int(geo.get("base_w", MUX_OUTPUT_W))
    base_h = int(geo.get("base_h", MUX_OUTPUT_H))

    regions = geo.get("regions", {}) or {}
    for name, pts in list(regions.items()):
        if pts:
            regions[name] = _scale_points(pts, base_w, base_h)
    geo["regions"] = regions

    crop = geo.get("crop_points")
    if crop:
        geo["crop_points"] = _scale_points(crop, base_w, base_h)

    if (base_w, base_h) != (MUX_OUTPUT_W, MUX_OUTPUT_H):
        print(f"[INFO] {cfg.get('source_id','?')} 來源座標 {base_w}x{base_h} → "
              f"換算成 {MUX_OUTPUT_W}x{MUX_OUTPUT_H}（ROI/crop 依比例縮放）")

    geo["base_w"] = MUX_OUTPUT_W
    geo["base_h"] = MUX_OUTPUT_H
    cfg["geometry"] = geo


def _group_infer_path(gid: int) -> str:
    return f"{BASE_DIR}/config_infer_group{gid}.txt"


def _group_preprocess_path(gid: int) -> str:
    return f"{BASE_DIR}/config_preprocess_group{gid}.txt"


def _group_analytics_path(gid: int) -> str:
    return f"{BASE_DIR}/config_nvdsanalytics_group{gid}.txt"


# ==========================================
# 2. 通用輔助函式
# ==========================================

def load_all_yamls(yaml_dir: str) -> List[Dict[str, Any]]:
    """讀 yaml_dir 下所有 .yaml（依檔名排序），回傳 list of dict。"""
    files = sorted(glob.glob(f"{yaml_dir}/*.yaml"))
    if not files:
        print(f"[ERROR] 找不到任何 YAML 檔案在：{yaml_dir}")
        sys.exit(1)
    cfgs = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            cfgs.append(yaml.safe_load(f))
    return cfgs


def derive_labels_filename(weight: str) -> str:
    """
    從 engine 檔名自動推導 labels 檔名（與 logic/config.py 一致）：
        car_fp16.engine → labels_car.txt
        yolo11s_fp16.engine → labels_yolo11s.txt
    """
    base = os.path.splitext(os.path.basename(weight))[0]
    lower = base.lower()
    for suf in _ENGINE_PRECISION_SUFFIXES:
        if lower.endswith(suf):
            base = base[: -len(suf)]
            break
    return f"labels_{base}.txt"


def get_num_classes(label_filename: str) -> int:
    """讀 BASE_DIR/label_filename 的非空行數作為類別數；不存在或空 → 1（並警告）。"""
    label_path = os.path.join(BASE_DIR, label_filename)
    if not os.path.exists(label_path):
        print(f"[WARNING] 標籤檔不存在：{label_path}，類別數用預設 1")
        return 1
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        print(f"[WARNING] 標籤檔 {label_path} 內容為空，類別數用預設 1")
        return 1
    return len(lines)


def crop_points_to_rect(points: List[List[int]]) -> Tuple[int, int, int, int]:
    """多邊形 crop_points 轉外接矩形 (x, y, w, h)。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return int(min_x), int(min_y), max(1, int(max_x - min_x)), max(1, int(max_y - min_y))


def _polygon_to_pts_string(points: List[List[int]]) -> str:
    """多邊形點位攤平成 nvdsanalytics 格式 "x1;y1;x2;y2;..."。"""
    return ";".join(str(c) for p in points for c in p)


def resolve_group_muxer_size(members: List[Dict[str, Any]]) -> Tuple[int, int]:
    """該組所有成員的 base_w/base_h 取最大，供 analytics config-width/height。"""
    max_w = max(m.get("geometry", {}).get("base_w", 1920) for m in members)
    max_h = max(m.get("geometry", {}).get("base_h", 1080) for m in members)
    return int(max_w), int(max_h)


# ==========================================
# 3. 依 weight 分組
# ==========================================

def build_groups(cfgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    依 weight 把 cfgs 分組（首次出現順序決定 group_id）。

    回傳 list of group dict：
        {"group_id", "weight", "imgsz", "batch", "labels_file", "members":[cfg,...]}
    members 依原本檔名順序（其索引 = 該組 pipeline 的區域 pad_index）。
    """
    order = []
    weight_to_gid = {}
    groups = []

    for cfg in cfgs:
        # 先把該路的 ROI/crop 依 base_w/base_h 換算成 1920×1080 座標（720P/4K 來源自動對齊）
        _normalize_geometry_inplace(cfg)

        weight = str(cfg.get("weight", "car_fp16.engine")).strip()
        if weight not in weight_to_gid:
            gid = len(order)
            weight_to_gid[weight] = gid
            order.append(weight)
            labels_yaml = cfg.get("labels")
            labels_file = labels_yaml if labels_yaml else derive_labels_filename(weight)
            groups.append({
                "group_id":    gid,
                "weight":      weight,
                "imgsz":       int(cfg.get("weight_imgsz", DEFAULT_WEIGHT_IMGSZ)),
                "batch":       int(cfg.get("weight_batch_size", DEFAULT_WEIGHT_BATCH)),
                "labels_file": labels_file,
                "members":     [],
            })
        gid = weight_to_gid[weight]
        groups[gid]["members"].append(cfg)

    return groups


# ==========================================
# 4. 每組設定檔產生：preprocess
# ==========================================

def generate_preprocess_config_for_group(group: Dict[str, Any]) -> None:
    """
    產生 config_preprocess_group{N}.txt。
    - network-input-shape 的 batch = 該組 weight_batch_size；W/H = 該組 weight_imgsz
    - src-ids / roi-params-src-N 用「該組區域索引」0..k-1（對應該條 pipeline 的 pad）
    """
    gid = group["group_id"]
    members = group["members"]
    engine_batch = group["batch"]
    imgsz = group["imgsz"]
    show_any_crop = any(m.get("display", {}).get("show_crop", False) for m in members)

    lines = [
        "[property]",
        "enable=1",
        "target-unique-ids=1",
        "process-on-frame=1",
        "network-input-order=0",
        f"network-input-shape={engine_batch};3;{imgsz};{imgsz}",
        "network-color-format=0",
        "tensor-data-type=0",
        "tensor-name=input",
        f"processing-width={imgsz}",
        f"processing-height={imgsz}",
        "scaling-buf-pool-size=6",
        "tensor-buf-pool-size=6",
        "scaling-pool-memory-type=0",
        "scaling-pool-compute-hw=0",
        "scaling-filter=0",
        "maintain-aspect-ratio=1",
        "symmetric-padding=1",
        f"custom-lib-path={resolve_preprocess_lib()}",
        "custom-tensor-preparation-function=CustomTensorPreparation",
        "",
        "[user-configs]",
        "pixel-normalization-factor=0.003921568",
        "",
        "[group-0]",
        f"src-ids={';'.join(str(i) for i in range(len(members)))}",
        "process-on-roi=1",
        "custom-input-transformation-function=CustomAsyncTransformation",
        f"draw-roi={1 if show_any_crop else 0}",
        "roi-color=0;1;1;1",
    ]

    # 每個成員（區域索引 i）的裁切矩形
    for i, cfg in enumerate(members):
        geo = cfg.get("geometry", {}) or {}
        crop_points = geo.get("crop_points")
        if crop_points:
            x, y, w, h = crop_points_to_rect(crop_points)
        else:
            # 沒定義 crop_points → 退回整張畫面（不裁切）
            x, y = 0, 0
            w = int(geo.get("base_w", 1920))
            h = int(geo.get("base_h", 1080))
            print(f"[WARNING] {cfg.get('source_id', f'cam_{i}')} 未定義 crop_points，"
                  f"裁切區退回整張畫面 {w}x{h}")
        lines.append(f"roi-params-src-{i}={x};{y};{w};{h}")

    with open(_group_preprocess_path(gid), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ==========================================
# 5. 每組設定檔產生：PGIE
# ==========================================

def generate_primary_infer_config_for_group(group: Dict[str, Any]) -> None:
    """
    產生 config_infer_group{N}.txt。
    - model-engine-file = 該組 weight；batch-size = 該組 weight_batch_size
    - num-detected-classes / labelfile-path = 該組 labels 檔（自動推導或 YAML 指定）
    - conf / iou 取該組第一個成員的 detect 區塊
    """
    gid = group["group_id"]
    members = group["members"]
    weight = group["weight"]
    batch_size = group["batch"]
    labels_file = group["labels_file"]
    num_classes = get_num_classes(labels_file)

    detect = members[0].get("detect", {}) or {}
    conf_thresh = detect.get("car_conf", 0.25)
    iou_thresh = detect.get("car_iou", 0.45)

    content = f"""\
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-engine-file={weight}
batch-size={batch_size}
network-mode=2
num-detected-classes={num_classes}
labelfile-path={labels_file}
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
parse-bbox-func-name=NvDsInferParseYolo

[class-attrs-all]
nms-iou-threshold={iou_thresh}
pre-cluster-threshold={conf_thresh}
topk=300
"""
    with open(_group_infer_path(gid), "w", encoding="utf-8") as f:
        f.write(content)


# ==========================================
# 6. 每組設定檔產生：nvdsanalytics
# ==========================================

def generate_analytics_config_for_group(group: Dict[str, Any]) -> None:
    """
    產生 config_nvdsanalytics_group{N}.txt（多 ROI）。
    [roi-filtering-stream-{i}] 的 i 用「該組區域索引」0..k-1。
    """
    gid = group["group_id"]
    members = group["members"]
    muxer_w, muxer_h = resolve_group_muxer_size(members)

    lines = [
        "[property]",
        "enable=1",
        f"config-width={muxer_w}",
        f"config-height={muxer_h}",
        "osd-mode=1",
        "display-font-size=12",
        "",
    ]

    for i, cfg in enumerate(members):
        source_id = cfg.get("source_id", f"cam_{i}")
        show_roi = 1 if cfg.get("display", {}).get("show_roi", True) else 0
        regions = cfg.get("geometry", {}).get("regions", {}) or {}

        block = [f"[roi-filtering-stream-{i}]", f"enable={show_roi}"]
        if regions:
            for roi_name, pts in regions.items():
                if not pts or len(pts) < 3:
                    print(f"[WARNING] {source_id} 的 ROI '{roi_name}' 點數不足，略過")
                    continue
                block.append(f"roi-{roi_name}={_polygon_to_pts_string(pts)}")
        else:
            print(f"[WARNING] {source_id} 沒有定義 regions，使用全畫面作為預設 ROI")
            block.append(f"roi-{source_id}=0;0;{muxer_w};0;{muxer_w};{muxer_h};0;{muxer_h}")
        block.extend(["class-id=-1", "inverse-roi=0", ""])
        lines.extend(block)

    with open(_group_analytics_path(gid), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ==========================================
# 7. 全組共用：tracker runtime
# ==========================================

def get_tracker_mode(cfgs: List[Dict[str, Any]]) -> str:
    """讀 cfgs[0].tracker.type（全域共用一種追蹤器）。"""
    tracker_cfg = cfgs[0].get("tracker", {}) or {}
    return str(tracker_cfg.get("type", "nvdcf")).lower().strip()


def generate_tracker_runtime_config(cfgs: List[Dict[str, Any]]) -> str:
    """產生 config_tracker_runtime.txt（給 main.py 讀）。回傳 mode。"""
    mode = get_tracker_mode(cfgs)

    if mode != "nvdcf" and mode not in SUPPORTED_BOXMOT_TRACKERS:
        raise ValueError(
            f"未支援的 tracker.type='{mode}'。\n"
            f"  可用值：nvdcf, {', '.join(SUPPORTED_BOXMOT_TRACKERS)}"
        )

    lines = [
        "# 由 traffic_count_txt.py 自動產生，請勿手動編輯",
        "# 來源：ds_yaml/*.yaml 內 tracker.type（讀 cfgs[0]）",
        "",
        "[tracker]",
        f"mode={mode}",
    ]
    if mode in SUPPORTED_BOXMOT_TRACKERS:
        config_path = f"{BASE_DIR}/boxmot/configs/trackers/{mode}.yaml"
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"找不到 BoxMOT 追蹤器設定：{config_path}\n  請確認 boxmot/ 已複製到專案根目錄。"
            )
        lines.append(f"config={config_path}")

    with open(TRACKER_RUNTIME_CONFIG, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return mode


# ==========================================
# 8. 主流程
# ==========================================

def generate_mux_config(cfgs: List[Dict[str, Any]]) -> None:
    """
    產生 config_mux.txt 給新版 nvstreammux (USE_NEW_NVSTREAMMUX=yes) 讀取。
    解決：多路檔案來源中某一路先 EOS 時，舊版 mux 空等那一路、每批卡到逾時，剩餘來源 FPS 大跌。
    adaptive-batching=1 讓 batch 跟現存來源數走；overall-min-fps 決定湊不滿時的強制推出頻率=尾段最低 FPS。
    多 pipeline 版：各組 streammux 共用這一份（以全部來源的最高 stream_fps 計算）。
    """
    fps_list = [float(c.get("stream_fps", 30.0)) for c in cfgs]
    src_fps = max(fps_list) if fps_list else 30.0
    min_fps = max(int(round(src_fps)), MUX_MIN_FPS_FLOOR)
    max_fps = max(min_fps, MUX_OVERALL_MAX_FPS)

    lines = [
        "[property]",
        "algorithm-type=1",
        "adaptive-batching=1",
        "max-fps-control=0",
        f"overall-max-fps-n={max_fps}",
        "overall-max-fps-d=1",
        f"overall-min-fps-n={min_fps}",
        "overall-min-fps-d=1",
    ]
    with open(MUX_CONFIG, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] config_mux.txt: 來源最高 stream_fps={src_fps:.0f} → overall-min-fps={min_fps}")


def main() -> None:
    print(f"[INFO] BASE_DIR = {BASE_DIR}")
    print("正在載入 YAML 設定檔...")
    cfgs = load_all_yamls(YAML_DIR)
    print(f"已載入 {len(cfgs)} 個攝影機設定")

    # 依 weight 分組
    groups = build_groups(cfgs)
    print(f"依 weight 分成 {len(groups)} 組（= {len(groups)} 條 pipeline）：")
    for g in groups:
        names = ", ".join(m.get("source_id", "?") for m in g["members"])
        print(f"  group{g['group_id']}: engine={g['weight']}, labels={g['labels_file']}, "
              f"batch={g['batch']}, imgsz={g['imgsz']}, 成員=[{names}]")

    # 每組各產生 3 份 config
    produced = []
    for g in groups:
        generate_preprocess_config_for_group(g)
        generate_primary_infer_config_for_group(g)
        generate_analytics_config_for_group(g)
        produced.append(_group_infer_path(g["group_id"]))
        produced.append(_group_preprocess_path(g["group_id"]))
        produced.append(_group_analytics_path(g["group_id"]))

    # 全組共用一份 tracker runtime
    tracker_mode = generate_tracker_runtime_config(cfgs)

    # 新版 nvstreammux 用的共用設定（USE_NEW_NVSTREAMMUX=yes 時 main.py 會讀）
    generate_mux_config(cfgs)

    print("\n[DONE] 所有設定檔產生完畢！")
    for p in produced:
        print(f"  - {p}")
    print(f"  - {TRACKER_RUNTIME_CONFIG}  (tracker mode = {tracker_mode})")
    print(f"  - {MUX_CONFIG}  (USE_NEW_NVSTREAMMUX=yes 時使用)")

    if tracker_mode != "nvdcf":
        print(f"\n  ⚠ tracker.type = '{tracker_mode}' (BoxMOT)")
        print("     main.py 啟動後將跳過 nvtracker，改在 pgie.src 探針用 BoxMOT 接管")
        print(f"     BoxMOT 微調請直接編輯：boxmot/configs/trackers/{tracker_mode}.yaml")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
DeepStream 設定檔產生器

ds_yaml/*.yaml 有任何變動（新增、刪除、改內容）都要重跑：

    python traffic_count_txt.py

main.py 啟動時會核對，沒重跑會直接報錯並結束。

分組與 main.py 相同（都來自 logic/config.py）：weight 相同的攝影機一組。
每組產生 3 個檔，{N} 為組別編號（從 0 起算）：

    config_infer_group{N}.txt          nvinfer：engine、batch、類別數、門檻
    config_preprocess_group{N}.txt     nvdspreprocess：tensor 形狀、每路的裁切框
    config_nvdsanalytics_group{N}.txt  nvdsanalytics：每路的 ROI（只用來畫在畫面上）

值的來源：
    batch、輸入寬高  直接讀 engine 的實際輸入規格（logic.common.probe_engine_input）。
                    固定 batch 的 engine 必須完全一致，寫錯 nvinfer 會啟動失敗或推論時崩潰。
                    攝影機數不必等於 batch，nvdspreprocess 會自動分批。
    類別數          該組 labels 檔的行數
    信心值 / NMS    該組檔名排序第一個攝影機的 detect.car_conf / car_iou（同組共用一個 nvinfer）
    裁切框          各攝影機 geometry.crop_points 的外接矩形；沒寫就是整張畫面
    ROI             各攝影機的 geometry.regions
    座標一律為 1920×1080（logic/config.py 已依 base_w / base_h 換算）。
"""

import os
import sys
import glob

from logic.common import (
    MUX_OUTPUT_W, MUX_OUTPUT_H, CUSTOM_YOLO_LIB, PREPROCESS_LIB_NAME,
    probe_engine_input, yaml_fingerprint, resolve_ds_lib,
)
from logic.config import (
    BASE_DIR, YAML_DIR, SOURCE_CONFIGS, GROUPS, FINGERPRINT_PREFIX,
    group_infer_config, group_preprocess_config, group_analytics_config,
)


# =============================================================================
# 常數
# =============================================================================

DEFAULT_IMGSZ = 640                 # 讀不到 engine 規格、YAML 也沒寫 weight_imgsz 時的輸入邊長（像素）

# 舊版或 GitHub 部署版產生、本版已不使用的檔案；產生時順手刪除，避免誤會還在生效
LEGACY_FILES = (
    "config_tracker_runtime.txt",   # 舊版記錄追蹤器模式；現在直接讀 YAML 的 tracker.type
    "config_mux.txt",               # 部署版新版 nvstreammux（USE_NEW_NVSTREAMMUX）用；本版不使用
)


# =============================================================================
# engine 規格
# =============================================================================

def resolve_engine_spec(gid, group, members):
    """
    決定該組寫進設定檔的 batch 與輸入尺寸。

    固定 batch 的 engine  用 engine 的 batch。
    動態 batch 的 engine  用該組攝影機數，但不超過 engine 允許的上限。
    讀不到 engine 規格    （例如環境沒有 tensorrt）退回 YAML 的 weight_batch_size / weight_imgsz；
                          此時 YAML 必須填對，否則 nvinfer 會失敗。

    回傳：
        (batch, 輸入高, 輸入寬)
    """
    if not os.path.exists(group["engine_path"]):
        sys.exit(f"[ERROR] group{gid}：找不到 engine {group['engine_path']}\n"
                 f"        YAML 的 weight='{group['weight']}'，路徑相對於專案根目錄")

    first = members[0]                                          # 同組共用設定取檔名排序第一個
    spec = probe_engine_input(group["engine_path"], os.path.join(BASE_DIR, CUSTOM_YOLO_LIB))
    if spec is None:                                            # 讀不到 engine：只能相信 YAML
        if "weight_batch_size" not in first:
            sys.exit(f"[ERROR] group{gid}：讀不到 {group['weight']} 的輸入規格，"
                     f"請在 {first['source_id']} 的 YAML 補上 weight_batch_size（與 weight_imgsz）")
        size = int(first.get("weight_imgsz", DEFAULT_IMGSZ))
        print(f"[WARNING] group{gid}：改用 YAML 的 weight_batch_size / weight_imgsz")
        return int(first["weight_batch_size"]), size, size

    batch = spec["batch"]
    if spec["dynamic"]:
        batch = max(1, min(len(members), spec["max_batch"]))    # 動態 batch：攝影機數，但不超過上限

    # YAML 有填但和 engine 不同時提示（實際以 engine 為準）
    if "weight_batch_size" in first and not spec["dynamic"] and int(first["weight_batch_size"]) != batch:
        print(f"[INFO] group{gid}：YAML weight_batch_size={first['weight_batch_size']}，"
              f"engine 實際為 {batch}，採用 engine 的值")
    if "weight_imgsz" in first and int(first["weight_imgsz"]) not in (spec["h"], spec["w"]):
        print(f"[INFO] group{gid}：YAML weight_imgsz={first['weight_imgsz']}，"
              f"engine 實際輸入為 {spec['w']}x{spec['h']}，採用 engine 的值")
    return batch, spec["h"], spec["w"]


# =============================================================================
# nvinfer 設定檔
# =============================================================================
#
#   第一行                  ds_yaml 指紋，main.py 啟動時比對
#   gpu-id=0                使用第 0 張 GPU
#   net-scale-factor        1/255，像素正規化到 0 ~ 1。本專案由 nvdspreprocess 做前處理
#                           （input-tensor-meta），實際生效的是 pixel-normalization-factor
#   model-engine-file       engine 檔（相對於專案根目錄）
#   batch-size              必須等於 engine 的 batch
#   network-mode=2          FP16（0 = FP32、1 = INT8、2 = FP16）；只在需要重建 engine 時有意義
#   num-detected-classes    類別數
#   labelfile-path          labels 檔
#   interval=0              每一幀都推論
#   gie-unique-id=1         這個 nvinfer 的 ID；探針以 unique_component_id == 1 認出它的物件
#   process-mode=1          primary：對整張畫面推論
#   network-type=0          偵測器
#   cluster-mode=2          用 NMS 合併重疊框
#   maintain-aspect-ratio   縮放時保持長寬比
#   symmetric-padding       補邊放兩側（畫面置中）
#   custom-lib-path         YOLO 輸出解析函式庫
#   parse-bbox-func-name    函式庫裡的解析函式名稱
#   filter-out-class-ids    直接丟掉的類別（見下方）
#   nms-iou-threshold       NMS IoU 門檻
#   pre-cluster-threshold   信心值門檻
#   topk=300                NMS 後每幀最多保留幾個框

def write_infer_config(gid, group, members, batch, fingerprint):
    """寫出 config_infer_group{gid}.txt（各參數意義見上方）。"""
    num_classes = len(group["class_map"])
    if num_classes == 0:
        print(f"[WARNING] group{gid}：labels 檔不存在或是空的，類別數以 1 計")
        num_classes = 1                                                 # nvinfer 不接受 0 類

    detect = members[0]["detect"]                                       # 同組共用一個 nvinfer，取第一個攝影機的門檻

    # 該組所有攝影機都不要的類別，交給 nvinfer 直接丟掉，省去探針逐框過濾。
    # 只要有一路沒寫 keep_classes（= 全部保留）就不能丟；各路之間的差異仍由探針處理。
    drop_line = ""
    if all(m["keep_classes"] for m in members):
        wanted = set().union(*(m["keep_classes"] for m in members))     # 任一路要的都保留
        dropped = [str(c) for c in range(num_classes) if c not in wanted]
        if dropped:
            drop_line = f"filter-out-class-ids={';'.join(dropped)}\n"

    content = f"""\
{FINGERPRINT_PREFIX}{fingerprint}
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-engine-file={group['weight']}
batch-size={batch}
network-mode=2
num-detected-classes={num_classes}
labelfile-path={os.path.relpath(group['labels_path'], BASE_DIR)}
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
custom-lib-path={CUSTOM_YOLO_LIB}
parse-bbox-func-name=NvDsInferParseYolo
{drop_line}
[class-attrs-all]
nms-iou-threshold={detect.get('car_iou', 0.45)}
pre-cluster-threshold={detect.get('car_conf', 0.25)}
topk=300
"""
    with open(group_infer_config(gid), "w", encoding="utf-8") as fh:
        fh.write(content)


# =============================================================================
# nvdspreprocess 設定檔
# =============================================================================
#
#   target-unique-ids=1            產生的 tensor 給 gie-unique-id=1 的 nvinfer
#   process-on-frame=1             以整幀（而非偵測物件）為處理單位
#   network-input-order=0          NCHW
#   network-input-shape            batch;通道;高;寬，必須等於 engine 的輸入
#   network-color-format=0         RGB
#   tensor-data-type=0             FP32
#   tensor-name=input              engine 輸入 tensor 的名稱
#   processing-width / height      縮放後的尺寸（= engine 輸入寬高）
#   scaling-buf-pool-size=6        縮放用 buffer 數
#   tensor-buf-pool-size=6         tensor buffer 數
#   scaling-pool-memory-type=0     平台預設記憶體
#   scaling-pool-compute-hw=0      平台預設的縮放硬體（桌機 GPU、Jetson VIC）
#   scaling-filter=0               最近鄰插值
#   maintain-aspect-ratio=1        保持長寬比
#   symmetric-padding=1            補邊放兩側
#   custom-lib-path / *-function   DeepStream 內附的通用前處理函式庫（安裝位置由 find_ds_lib_root 自動偵測）
#   pixel-normalization-factor     1/255，像素正規化到 0 ~ 1
#   src-ids                        這組要處理的來源（區域 pad_index）
#   process-on-roi=1               只處理 roi-params-src-{i} 指定的範圍
#   draw-roi                       是否把裁切框畫在畫面上（show_crop）
#   roi-color=0;1;1;1              裁切框顏色 RGBA（青色）
#   roi-params-src-{i}             第 i 路的裁切矩形 left;top;width;height

def write_preprocess_config(gid, members, batch, in_h, in_w):
    """寫出 config_preprocess_group{gid}.txt（各參數意義見上方）。"""
    # 同組共用：一路要畫就全畫
    draw_roi = int(any(m["display"].get("show_crop", False) for m in members))
    lines = [
        "[property]",
        "enable=1",
        "target-unique-ids=1",
        "process-on-frame=1",
        "network-input-order=0",
        f"network-input-shape={batch};3;{in_h};{in_w}",
        "network-color-format=0",
        "tensor-data-type=0",
        "tensor-name=input",
        f"processing-width={in_w}",
        f"processing-height={in_h}",
        "scaling-buf-pool-size=6",
        "tensor-buf-pool-size=6",
        "scaling-pool-memory-type=0",
        "scaling-pool-compute-hw=0",
        "scaling-filter=0",
        "maintain-aspect-ratio=1",
        "symmetric-padding=1",
        f"custom-lib-path={resolve_ds_lib(PREPROCESS_LIB_NAME)}",
        "custom-tensor-preparation-function=CustomTensorPreparation",
        "",
        "[user-configs]",
        "pixel-normalization-factor=0.003921568",
        "",
        "[group-0]",
        f"src-ids={';'.join(str(i) for i in range(len(members)))}",
        "process-on-roi=1",
        "custom-input-transformation-function=CustomAsyncTransformation",
        f"draw-roi={draw_roi}",
        "roi-color=0;1;1;1",
    ]
    for i, cfg in enumerate(members):                                   # i = 區域 pad_index
        crop = cfg["geometry"]["crop_points"]
        if crop:
            xs, ys = [p[0] for p in crop], [p[1] for p in crop]
            x, y = int(min(xs)), int(min(ys))                           # 外接矩形左上角
            w, h = max(1, int(max(xs)) - x), max(1, int(max(ys)) - y)   # 寬高至少 1 像素
        else:
            x, y, w, h = 0, 0, MUX_OUTPUT_W, MUX_OUTPUT_H               # 沒寫 crop_points：整張畫面
        lines.append(f"roi-params-src-{i}={x};{y};{w};{h}")

    with open(group_preprocess_config(gid), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# =============================================================================
# nvdsanalytics 設定檔
# =============================================================================
#
#   config-width / height     ROI 座標系（= streammux 輸出尺寸）
#   osd-mode=1                只畫 ROI，不畫統計文字
#   display-font-size=12      ROI 名稱字體大小
#   roi-filtering-stream-{i}  第 i 路（區域 pad_index）；enable 決定畫不畫
#   roi-<名稱>                多邊形頂點 x1;y1;x2;y2;…
#   class-id=-1               套用到所有類別
#   inverse-roi=0             ROI 內為有效區域

def write_analytics_config(gid, members):
    """寫出 config_nvdsanalytics_group{gid}.txt（各參數意義見上方）。"""
    lines = [
        "[property]",
        "enable=1",
        f"config-width={MUX_OUTPUT_W}",
        f"config-height={MUX_OUTPUT_H}",
        "osd-mode=1",
        "display-font-size=12",
        "",
    ]
    for i, cfg in enumerate(members):                                       # i = 區域 pad_index
        rois = [f"roi-{roi}=" + ";".join(str(v) for p in pts for v in p)    # [[x,y],...] → x1;y1;x2;y2;…
                for roi, pts in cfg["geometry"]["regions"].items() if pts and len(pts) >= 3]
        show = bool(rois) and cfg["display"].get("show_roi", True)
        lines += [f"[roi-filtering-stream-{i}]", f"enable={int(show)}", *rois,
                  "class-id=-1", "inverse-roi=0", ""]

    with open(group_analytics_config(gid), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# =============================================================================
# 主流程
# =============================================================================

def _remove_stale_files():
    """刪除已不存在的組別留下的設定檔，以及 LEGACY_FILES。"""
    stale = [os.path.join(BASE_DIR, name) for name in LEGACY_FILES]
    for path in glob.glob(os.path.join(BASE_DIR, "config_*_group*.txt")):
        number = os.path.splitext(path)[0].rsplit("group", 1)[-1]   # config_infer_group3.txt → "3"
        if number.isdigit() and int(number) >= len(GROUPS):         # 組數變少時，編號較大的已不再使用
            stale.append(path)
    for path in stale:
        if os.path.exists(path):
            os.remove(path)
            print(f"  已移除過期檔案：{os.path.basename(path)}")


def main():
    fingerprint = yaml_fingerprint(YAML_DIR)                                # 每組的 infer 設定檔都寫同一個指紋
    print("\n產生設定檔：")
    for gid, group in GROUPS.items():
        members = [SOURCE_CONFIGS[uid] for uid in group["member_uids"]]     # 順序 = pad_index
        batch, in_h, in_w = resolve_engine_spec(gid, group, members)
        write_infer_config(gid, group, members, batch, fingerprint)
        write_preprocess_config(gid, members, batch, in_h, in_w)
        write_analytics_config(gid, members)
        print(f"  group{gid}: {group['weight']}  batch={batch}  輸入={in_w}x{in_h}  cam 數={len(members)}")

    _remove_stale_files()
    print("[DONE] 設定檔已更新，可以執行 python main.py")


if __name__ == "__main__":
    main()

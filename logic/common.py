"""
共用工具

產生器（traffic_count_txt.py）與執行期（logic/config.py、main.py）都會用到的函式。
除了 find_ds_lib_root 會查 GStreamer 的 plugin 清單，其餘都不依賴 GStreamer。

    scale_points            來源解析度座標 → streammux 輸出座標
    derive_labels_filename  engine 檔名 → labels 檔名
    yaml_fingerprint        ds_yaml 內容的指紋，用來判斷設定檔是否過期
    find_ds_lib_root        找 DeepStream 的安裝位置（各機器不同，不寫死路徑）
    resolve_ds_lib          DeepStream 內附函式庫的完整路徑
    probe_engine_input      讀 TensorRT engine 的實際輸入規格
"""

import os
import sys
import glob
import ctypes
import hashlib


# =============================================================================
# 常數
# =============================================================================

# streammux 輸出尺寸（像素）。所有來源都先縮放成這個尺寸，
# ROI、裁切框、物件座標一律以此為座標系。
MUX_OUTPUT_W = 1920
MUX_OUTPUT_H = 1080

# engine 檔名結尾的精度 / 變體後綴；推導 labels 檔名時剝掉其中一個
ENGINE_PRECISION_SUFFIXES = ("_fp16", "_fp32", "_int8", "_fp8", "_dla", "_dynamic", "_best")

# nvinfer 的 YOLO 輸出解析函式庫（相對於專案根目錄，需先編譯）
CUSTOM_YOLO_LIB = "nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"

# 用來反查 DeepStream 安裝位置的元件；任何 DeepStream 安裝都一定有 nvinfer
DS_PROBE_ELEMENT = "nvinfer"

# DeepStream 內附、本專案會用到的函式庫（相對於 DeepStream 的 lib 目錄）
NVTRACKER_LIB_NAME = "libnvds_nvmultiobjecttracker.so"          # nvtracker（NvDCF）
PREPROCESS_LIB_NAME = "gst-plugins/libcustom2d_preprocess.so"   # nvdspreprocess 的通用前處理

_ds_lib_root = None                                             # find_ds_lib_root() 的結果快取


# =============================================================================
# 函式
# =============================================================================

def scale_points(points, base_w, base_h):
    """
    把 base_w × base_h 座標系的點位換算成 streammux 輸出座標系。

    points 為空、或 base 已等於輸出尺寸時原樣回傳。

    參數：
        points  [[x, y], ...]
        base_w  標記點位時所用的畫面寬（像素）
        base_h  標記點位時所用的畫面高（像素）
    回傳：
        換算後的 [[x, y], ...]（四捨五入成整數）
    """
    if not points or not base_w or not base_h:
        return points
    sx = MUX_OUTPUT_W / float(base_w)   # 水平縮放倍率
    sy = MUX_OUTPUT_H / float(base_h)   # 垂直縮放倍率
    if sx == 1.0 and sy == 1.0:
        return points                   # 已是輸出座標，不必換算
    return [[int(round(x * sx)), int(round(y * sy))] for x, y in points]


def derive_labels_filename(weight):
    """
    由 engine 檔名推導 labels 檔名。

        car_fp16.engine             → labels_car.txt
        weight/yolo11s_fp16.engine  → labels_yolo11s.txt（只看檔名，labels 檔放專案根目錄）
    """
    base = os.path.splitext(os.path.basename(weight))[0]    # 去掉資料夾與副檔名
    lower = base.lower()
    for suffix in ENGINE_PRECISION_SUFFIXES:
        if lower.endswith(suffix):
            base = base[: -len(suffix)]                     # 只剝一個後綴
            break
    return f"labels_{base}.txt"


def yaml_fingerprint(yaml_dir):
    """
    回傳 yaml_dir 下所有 *.yaml（檔名 + 內容）的 SHA-1。

    產生器把它寫進設定檔第一行；main.py 啟動時重算比對，
    不同就代表 YAML 改過但沒有重跑產生器。
    """
    digest = hashlib.sha1()
    for path in sorted(glob.glob(os.path.join(yaml_dir, "*.yaml"))):    # 排序：結果不受檔案系統順序影響
        digest.update(os.path.basename(path).encode("utf-8"))           # 檔名也算：改名、增刪檔都會變
        with open(path, "rb") as fh:
            digest.update(fh.read())
    return digest.hexdigest()


def find_ds_lib_root():
    """
    回傳 DeepStream 的 lib 目錄，例如 /opt/nvidia/deepstream/deepstream-7.1/lib。

    不寫死任何路徑，而是問 GStreamer：DeepStream 要能用，GStreamer 就一定找得到 nvinfer，
    它的 plugin 檔位於 <lib 目錄>/gst-plugins/libnvdsgst_infer.so。
    GStreamer 回報的路徑通常是系統目錄裡的符號連結
    （例如 /usr/lib/<平台>/gstreamer-1.0/deepstream/…），解開後就是實際的安裝位置。
    因此不論安裝在 /opt 底下哪個目錄、Jetson 或桌機 / WSL2，都能找到。

    環境變數 DS_LIB_ROOT 有設定時直接使用（自動偵測結果不對時的手動指定）。
    找不到時印出原因並結束程式。
    """
    global _ds_lib_root
    if _ds_lib_root is not None:
        return _ds_lib_root

    env_root = os.environ.get("DS_LIB_ROOT", "").strip()
    if env_root:
        if not os.path.isdir(env_root):
            sys.exit(f"[ERROR] DS_LIB_ROOT={env_root} 不是存在的資料夾")
        _ds_lib_root = env_root
        return _ds_lib_root

    plugin_file = None
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)                                          # 重複呼叫無妨；main.py 之後的 Gst.init 不受影響
        factory = Gst.ElementFactory.find(DS_PROBE_ELEMENT)
        if factory is not None and factory.get_plugin() is not None:
            plugin_file = factory.get_plugin().get_filename()   # 只查 plugin 清單，不會建立元件
    except Exception as e:
        print(f"[WARNING] 查詢 GStreamer plugin 失敗：{e}")

    if not plugin_file:
        sys.exit(f"[ERROR] GStreamer 找不到 {DS_PROBE_ELEMENT}，無法判斷 DeepStream 安裝位置。\n"
                 f"        請確認 DeepStream 已安裝（gst-inspect-1.0 {DS_PROBE_ELEMENT} 要能顯示資訊），"
                 f"或設定環境變數 DS_LIB_ROOT=<DeepStream 的 lib 目錄>")

    real_file = os.path.realpath(plugin_file)                   # 解開符號連結，得到實際安裝位置
    _ds_lib_root = os.path.dirname(os.path.dirname(real_file))  # <lib>/gst-plugins/xxx.so → <lib>
    print(f"[INFO] DeepStream lib 目錄：{_ds_lib_root}")
    return _ds_lib_root


def resolve_ds_lib(lib_name):
    """
    回傳 DeepStream 內附函式庫的完整路徑；檔案不存在時印出原因並結束程式。

    參數：
        lib_name  相對於 lib 目錄的路徑，例如 NVTRACKER_LIB_NAME
    """
    path = os.path.join(find_ds_lib_root(), lib_name)
    if not os.path.exists(path):
        sys.exit(f"[ERROR] 找不到 {path}\n"
                 f"        DeepStream 安裝可能不完整；或以環境變數 DS_LIB_ROOT 指定正確的 lib 目錄")
    return path


def probe_engine_input(engine_path, custom_lib=None):
    """
    反序列化 TensorRT engine，讀出輸入 tensor 的實際規格。

    nvinfer 的 batch-size 與 nvdspreprocess 的 tensor 形狀必須和 engine 完全一致，
    所以直接從 engine 讀，不靠人工填寫。

    參數：
        engine_path  engine 檔路徑
        custom_lib   engine 用到的自訂 plugin 函式庫；反序列化前要先載入
    回傳：
        {"batch", "max_batch", "h", "w", "dynamic"}
            固定 batch：batch == max_batch
            動態 batch：batch == -1，max_batch 為 optimization profile 允許的上限
        None：讀不到（沒裝 tensorrt、檔案損毀等），由呼叫端決定備援方式
    """
    try:
        import tensorrt as trt                                  # 只有產生器用得到，放在函式內 import
    except Exception as e:
        print(f"[WARNING] 無法 import tensorrt（{e}）")
        return None

    if custom_lib and os.path.exists(custom_lib):
        try:
            ctypes.CDLL(custom_lib, mode=ctypes.RTLD_GLOBAL)    # GLOBAL：讓 TensorRT 找得到 plugin 符號
        except OSError:
            pass

    try:
        with open(engine_path, "rb") as fh:
            engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(fh.read())
        if engine is None:
            raise RuntimeError("deserialize_cuda_engine 回傳 None")

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue                                        # 只看輸入，跳過輸出 tensor
            shape = tuple(engine.get_tensor_shape(name))        # (N, C, H, W)；動態軸為 -1
            if len(shape) != 4:
                raise RuntimeError(f"輸入 '{name}' 不是 NCHW：{shape}")
            dynamic = shape[0] < 0
            # 動態 batch 時取 optimization profile 0 的 (min, opt, max) 中的 max
            max_shape = tuple(engine.get_tensor_profile_shape(name, 0)[2]) if dynamic else shape
            return {
                "batch":     int(shape[0]),
                "max_batch": int(max_shape[0]),
                "h":         int(max_shape[2]),
                "w":         int(max_shape[3]),
                "dynamic":   dynamic,
            }
        raise RuntimeError("engine 沒有輸入 tensor")
    except Exception as e:
        print(f"[WARNING] 讀取 engine 規格失敗（{engine_path}）：{e}")
        return None

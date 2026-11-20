import os
import sys
import shutil
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor

# =========================
# 使用者設定區
# =========================

# 要轉換的 ONNX 權重列表（放在 weight/ 底下；只用一個權重就只留一行）
# 也可直接從命令列指定：python onnx2engine.py car.onnx
ONNX_LIST = [
    # "car.onnx",
    "yolo11s.onnx",
]
if len(sys.argv) > 1:
    ONNX_LIST = [os.path.basename(a) for a in sys.argv[1:]]

# =========================
# 固定參數
# =========================

def _find_trtexec():
    """
    找 TensorRT 的 trtexec，不寫死安裝位置。

    順序：環境變數 TRTEXEC → PATH 上的 trtexec → TensorRT 套件的標準位置
    （JetPack 與 TensorRT 的 deb 套件都裝在 /usr/src/tensorrt/bin，與使用者名稱無關）。
    """
    env = os.environ.get("TRTEXEC", "").strip()
    if env and os.path.isfile(env):
        return env
    return shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"


TRTEXEC = _find_trtexec()
WEIGHT_DIR = "weight"

# 自動設定最大平行數量
MAX_WORKERS = max(1, min(os.cpu_count() or 1, len(ONNX_LIST)))


def build_engine(onnx_name):
    """
    單一 ONNX -> TensorRT Engine
    並即時輸出 trtexec log
    """

    start_time = time.time()

    # ONNX 來源仍然在 weight/
    onnx_path = os.path.join(WEIGHT_DIR, onnx_name)

    # Engine 輸出改成目前目錄
    engine_name = os.path.splitext(onnx_name)[0] + "_fp16.engine"

    cmd = [
        TRTEXEC,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_name}",
        "--fp16",
        "--builderOptimizationLevel=3",
        "--timingCacheFile=trt.cache"
    ]

    print(f"\n🚀 開始轉檔: {onnx_name}")
    print(" ".join(cmd))
    print("-" * 80)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # 即時輸出 log
    for line in process.stdout:
        print(f"[{onnx_name}] {line}", end="")

    process.wait()

    elapsed = time.time() - start_time

    if process.returncode == 0:
        print(f"\n✅ 完成: {engine_name}")
        print(f"⏱️ 花費時間: {elapsed:.2f} 秒\n")
    else:
        print(f"\n❌ 失敗: {onnx_name}")
        print(f"⏱️ 花費時間: {elapsed:.2f} 秒\n")


def main():

    if not ONNX_LIST:
        sys.exit("[ERROR] ONNX_LIST 是空的，請在檔案上方填入要轉換的 .onnx，或用命令列參數指定。")

    total_start = time.time()

    print("=" * 80)
    print("🧠 TensorRT 平行轉檔")
    print(f"📦 自動平行數量: {MAX_WORKERS}")
    print("=" * 80)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = []

        for onnx_name in ONNX_LIST:
            futures.append(
                executor.submit(build_engine, onnx_name)
            )

        # 等待全部完成
        for future in futures:
            future.result()

    total_elapsed = time.time() - total_start

    print("=" * 80)
    print("🎉 全部轉檔完成")
    print(f"⏱️ 總花費時間: {total_elapsed:.2f} 秒")
    print("=" * 80)


if __name__ == "__main__":
    main()
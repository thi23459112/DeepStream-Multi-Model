import os
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor

# =========================
# 使用者設定區
# =========================

# 要轉換的 ONNX 權重列表
ONNX_LIST = [
    # "plate.onnx",
    # "num.onnx",
    "car.onnx",
    "yolo11s.onnx",
]

# ONNX_LIST = [ 
#   "car.onnx", 
#   "plate.onnx", 
#   "num.onnx" 
# ]

# =========================
# 固定參數
# =========================

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
WEIGHT_DIR = "weight"

# 自動設定最大平行數量
MAX_WORKERS = min(os.cpu_count(), len(ONNX_LIST))


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
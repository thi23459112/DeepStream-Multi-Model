#!/usr/bin/env bash
# 批次執行：每 BATCH_SIZE 部影片一批，跑完一批自動接下一批。放在專案根目錄。
set -uo pipefail

# ============ 計時 ============
SECONDS=0                          # bash 內建：歸零後會自動累加秒數，用來算總花費時間

# 印出從開始到現在的總花費時間（時:分:秒）
print_elapsed(){
  local t=$SECONDS
  printf "[INFO] 總花費時間：%02d:%02d:%02d （共 %d 秒）\n" \
    "$(( t/3600 ))" "$(( (t%3600)/60 ))" "$(( t%60 ))" "$t"
}

# ============ Conda ============
CONDA_ENV="tracking"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  echo "[ERROR] 找不到 conda，請先 conda activate $CONDA_ENV 再執行，或設定 CONDA_BASE。"; exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "[ERROR] 無法啟用 conda 環境：$CONDA_ENV"; exit 1; }
echo "[INFO] 已啟用 conda：$CONDA_ENV ($(which python))"
export PYTHONUNBUFFERED=1          # 讓 python 的 print 即時吐出（配 tee 才看得到）

# ============ 執行模式開關 ============
# HEADLESS=1 → 無終端機模式：main.py stdin 導 /dev/null，停用 q 鍵，
#              以背景子程序跑，Ctrl+C 由本腳本轉送 SIGINT，求快退出（影片完整性自負）。
# HEADLESS=0 → 終端機模式（預設）：main.py 跑在前景、繼承 tty，
#              q 鍵可用、Ctrl+C 會完整封裝，影片正常可播。
HEADLESS="${HEADLESS:-0}"           # 0=終端機模式（安全退出），1=無終端機（快速，影片自負）

# ============ 參數 ============
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POOL_DIR="${PROJECT_DIR}/ds_yaml"
BATCH_SIZE=2                        # 每批處理的影片設定檔數量
PYTHON_BIN="python"
WORK_ROOT="${PROJECT_DIR}/.batch_runs"

# 自動偵測本專案的設定產生器（traffic_count_txt.py / person_count_txt.py / LPR_txt.py）
GEN_SCRIPT="$(find "$PROJECT_DIR" -maxdepth 1 -name '*_txt.py' | head -1)"   # 取第一個符合的腳本
MAIN_SCRIPT="${PROJECT_DIR}/main.py"
if [ -z "$GEN_SCRIPT" ]; then echo "[ERROR] 找不到 *_txt.py 設定產生器"; exit 1; fi
echo "[INFO] 產生器：$(basename "$GEN_SCRIPT")"
echo "[INFO] 執行模式：$([ "$HEADLESS" = "1" ] && echo '無終端機（求快退出，影片自負）' || echo '終端機（Q/Ctrl+C 安全退出，影片正常）')"

mkdir -p "$WORK_ROOT"

mapfile -t ALL_YAML < <(find "$POOL_DIR" -maxdepth 1 -name '*.yaml' | LC_ALL=C sort)
TOTAL=${#ALL_YAML[@]}
[ "$TOTAL" -eq 0 ] && { echo "[ERROR] $POOL_DIR 沒有 .yaml"; exit 1; }
TOTAL_BATCHES=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "[INFO] 共 $TOTAL 部，每批 $BATCH_SIZE，總共 $TOTAL_BATCHES 批"

# ============ 中斷處理（僅無終端機模式需要轉送）============
CHILD_PID=""
STOPPING=0

# 無終端機模式：Ctrl+C 只送「一次」SIGINT 給 main.py，然後耐心等它收尾，不重複補刀。
request_stop(){
  if [ "$STOPPING" -eq 1 ]; then
    echo; echo "[INFO] 已在停止中，請耐心等待 main.py 結束..."; return
  fi
  STOPPING=1
  echo; echo "[INFO] 收到中斷，通知 main.py 結束（勿再重複按）..."
  [ -n "$CHILD_PID" ] && kill -INT "$CHILD_PID" 2>/dev/null
}

batch_no=0
for (( start=0; start<TOTAL; start+=BATCH_SIZE )); do
  batch_no=$((batch_no+1)); end=$(( start+BATCH_SIZE )); [ "$end" -gt "$TOTAL" ] && end=$TOTAL
  batch_dir="${WORK_ROOT}/batch_$(printf '%03d' "$batch_no")"
  rm -rf "$batch_dir"; mkdir -p "$batch_dir"
  echo "============================================================"
  echo "[INFO] 第 $batch_no/$TOTAL_BATCHES 批（第 $((start+1)) ~ $end 部）"
  for (( i=start; i<end; i++ )); do
    f="${ALL_YAML[$i]}"
    ln -sf "$f" "${batch_dir}/$(basename "$f")"    # 建立軟連結到批次目錄
    echo "        - $(basename "$f")"
  done

  log_file="${batch_dir}/run.log"
  echo "[INFO] 產生設定檔..."
  if ! DS_YAML_DIR="$batch_dir" "$PYTHON_BIN" "$GEN_SCRIPT" > >(tee "$log_file") 2>&1; then
    echo "[ERROR] 第 $batch_no 批產生設定檔失敗，略過。詳見 $log_file"; continue
  fi

  echo "[INFO] 開始推論，log → $log_file"

  if [ "$HEADLESS" = "1" ]; then
    # ---- 無終端機模式：導 /dev/null（停用 q 鍵），背景跑 + 轉送 SIGINT ----
    trap request_stop INT TERM
    DS_YAML_DIR="$batch_dir" "$PYTHON_BIN" "$MAIN_SCRIPT" </dev/null > >(tee -a "$log_file") 2>&1 &
    CHILD_PID=$!
    # 耐心等：wait 被訊號打斷會提早返回，用迴圈等到 main.py 真的結束
    while kill -0 "$CHILD_PID" 2>/dev/null; do
      wait "$CHILD_PID"; status=$?
    done
    CHILD_PID=""
    trap - INT TERM
    if [ "$STOPPING" -eq 1 ]; then
      echo "[INFO] 已依中斷要求停止，結束批次流程。"; print_elapsed; exit 130
    fi
  else
    # ---- 終端機模式：前景跑、繼承 tty，q 鍵可用、Ctrl+C 由 main.py 自己完整封裝 ----
    # 用 tee 的 process substitution 保留即時畫面 log，同時 main.py 仍在前景吃鍵盤。
    DS_YAML_DIR="$batch_dir" "$PYTHON_BIN" "$MAIN_SCRIPT" > >(tee -a "$log_file") 2>&1
    status=$?
    # 使用者在終端機按 q/Ctrl+C 結束單一批次 → main.py 以非 0（130）退出，
    # 視為「使用者要中止整個批次」，不自動接下一批。
    if [ "$status" -ne 0 ]; then
      echo "[INFO] main.py 以 exit=$status 結束（可能是 Q/Ctrl+C），結束批次流程。"; print_elapsed; exit "$status"
    fi
  fi

  echo "[INFO] 第 $batch_no 批完成。"
done

echo "============================================================"
echo "[INFO] 全部 $TOTAL 部、共 $batch_no 批 執行完畢。"
print_elapsed
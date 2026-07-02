# DeepStream 車流計數專案 — 功能總結

本專案基於 **DeepStream 7.1**（Jetson / JetPack 6、TensorRT 10.3）打造，支援**多權重、多路影像同時辨識**，具備解析度自動換算、逐路方向判定、類別過濾、每台車明細寫入單一合併資料庫等功能。以下依模組整理目前所有功能。

---

## 一、多權重 / 多路辨識（核心架構）

- **多權重分組**：依每個 YAML 的 `weight` 欄位自動分組，相同 engine 的 cam 歸為一組，每組各跑**一條獨立 pipeline**（共用一個 GLib mainloop）。例：camA/camB 用 `car_fp16.engine`、camC/camD 用 `yolo11s_fp16.engine`，即兩條 pipeline 並行。
- **每組獨立推論鏈**：各組有自己的 `streammux → preprocess → pgie（該組 engine）→ [tracker] → analytics → demux`，batch 自動等於該組成員數。
- **全域 uid 編號**：跨所有組唯一的 `stream_uid`，追蹤狀態 / DB / BoxMOT 都用它當鍵；probe 負責把各 pipeline 的區域 `pad_index` 翻譯回全域 uid，組間互不干擾。
- **labels 自動推導**：`car_fp16.engine` → `labels_car.txt`、`yolo11s_fp16.engine` → `labels_yolo11s.txt`（自動剝除 `_fp16 / _fp32 / _int8` 等後綴），YAML 不必寫 labels（可選覆寫）。
- **每組各自產生設定檔**：`traffic_count_txt.py` 為每組產生 `config_infer_group{N}.txt`、`config_preprocess_group{N}.txt`、`config_nvdsanalytics_group{N}.txt`，類別數由該組 labels 自動帶入。

## 二、類別過濾（keep_classes）

- **逐路類別白名單**：每個 YAML 可寫 `keep_classes: [0]`，只保留指定的原始 class_id，其餘偵測框在 probe 階段就丟掉（不畫框、不追蹤、不計數、不寫 DB）。不寫代表全收。
- 解決「yolo11s 是 COCO 80 類、刪 labels 也沒用」的問題——過濾在偵測階段做，而非靠 labels 檔。

## 三、解析度自動換算（Auto Resize）

- **ROI / crop 自動縮放**：在 YAML 按**來源真實解析度**標記 ROI / crop_points，程式依該路 `base_w/base_h` 與 streammux 輸出（1920×1080）的比例自動換算成 1080P 座標。720P、4K 等任何來源都可直接寫真實點位，不用手算。
- **三處座標系一致**：probe 計數用的 `cv_regions`、preprocess 裁切框、nvdsanalytics ROI 視覺線，三者都套同一換算，不會歪掉。
- **同組混解析度可行**：同一組內 1080p / 720p / 4K 混用沒問題，streammux 統一縮放成 1080P 後再 batch。

## 四、方向判定（IN / OUT）

- **逐路獨立軸向**：每路可各自設 `track_logic.axis: y`（上下）或 `x`（左右），互不影響，與分組無關。
- **方向值**：`flow_in` / `flow_out`（軸正向為 in、負向為 out），位移不足則判為 NA、不寫 DB。
- **抖動過濾**：`movement_threshold`（位移門檻）、`min_roi_hits`（ROI 命中最少幀數）防止誤判。

## 五、追蹤器

- **雙模式**：`nvdcf`（DeepStream 內建 nvtracker）或 **BoxMOT** 系列（`bytetrack / ocsort / fasttracker / sfsort / cbiou`），由 YAML `tracker.type` 決定。
- BoxMOT 模式在 `pgie.src` 探針接管：抽偵測框 → 餵 BoxMOT → 用追蹤結果重建 obj_meta。追蹤器以全域 uid 索引，多 pipeline 不撞號。

## 六、多 ROI 計數

- **單路多 ROI**：一台車經過多個 ROI 各寫一筆，ROI 名稱寫進 DB 的 `LocationName`。
- **只清本組軌跡**：每條 pipeline 的 probe 只結算屬於自己組的軌跡，不會誤清別組還在畫面上的車。

## 七、資料庫（明細版）

- **單一合併 DB**：所有 cam / 所有組寫進同一個 `traffic_count.db`，靠 `CameraCode` 區分。
- **每台車一筆明細**：ID 消失結算時寫一筆。
- **DB 欄位**：`DeviceCode / CameraCode / TrackID / DetectClass / LocationName / MetricType / HitCount / VideoTime / CollectTime`。
- **車種名稱用該組 class_map**：yolo 組正確寫 `person`，不會套成車種名。
- **CollectTime 雙模式**：檔案 = `start_time + 影片虛擬秒數`；RTSP = 系統當下時間。
- **WAL 模式**：支援邊寫邊讀（DeepStream 寫入、驗證腳本讀取可並行不卡）。會伴隨 `-wal` / `-shm` 檔，屬正常現象。
- **save_output_db=false**：只印 log、不寫 DB。
- **local_id 循環**：每路各自 1 ～ 999999 循環。

## 八、輸入 / 輸出 / 顯示

- **多來源類型**：本地影片檔、RTSP、HTTP；RTSP 自動注入防斷線參數（TCP / latency / timeout），帳密特殊字元自動編碼。
- **檔案模式自動讀真實 FPS**（cv2 覆寫 YAML 值）。
- **每組獨立預覽視窗**：tile 佈局自動依成員數排列，可逐路 `show_window` 開關。
- **畫面疊加**：bbox 用車種色、ID + 車種標籤、左上角即時 FPS。
- **寫檔輸出**：可選存推論後影片（mp4），含 videorate 穩定 PTS。
- **無 RTSP 推流**：輸出推流分支已移除。

## 九、執行 / 結束

- **批次腳本 `run_batch.sh`**：含 HEADLESS 開關、總耗時統計、Ctrl+C 安全退出。
- **安全退出**：終端機按 `q` 或收到 SIGINT / SIGTERM，對所有 pipeline 送 EOS，全部封裝完才退出（10 秒逾時 fallback）。
- **結束強制結算**：把還在畫面內的殘留軌跡全部結算寫出。
- **每 30 秒 FPS 報告**（涵蓋所有 cam）。

## 十、驗證工具

- **`verify_interval.py`**：獨立腳本，唯讀明細 DB → 按分鐘（區間可調）彙總成 `Value` 表格（每方向 + 車種的台數）→ 終端機表格 + CSV，並附對照檢查（明細列數 = Value 加總）。用來驗證區間彙總格式，不碰真實 API。

---

## 執行流程

```bash
# 1. 產生各組設定檔（改過 YAML 的 ROI / weight / tracker 後都要重跑）
python traffic_count_txt.py

# 2. 啟動辨識（多條 pipeline 並行）
python main.py

# 3.（可選）驗證區間彙總格式
python verify_interval.py --interval 1
```

> **提醒**：更動 ROI / crop_points 後，務必先重跑 `traffic_count_txt.py` 再跑 `main.py`，因為裁切框是在產生設定檔階段計算的。

---

## 檔案結構與職責

| 檔案 | 職責 |
|------|------|
| `main.py` | 依組建立多條 pipeline、掛探針、共用 mainloop、安全退出 |
| `traffic_count_txt.py` | 讀 YAML、依 weight 分組、產生每組的 infer / preprocess / analytics 設定檔（含 ROI/crop 自動縮放） |
| `logic/config.py` | 載入 YAML、建立全域 SOURCE_CONFIGS 與 GROUPS、解析 keep_classes、ROI/crop 自動縮放、追蹤器模式 |
| `logic/pipeline.py` | GStreamer 元件建構、每路下游分支（顯示 / 寫檔）、tile 視窗 |
| `logic/probes.py` | 追蹤探針（nvdcf / BoxMOT）、方向判定、多 ROI 命中、類別過濾、OSD |
| `logic/state_db.py` | 單一合併 SQLite、每台車明細結算與寫入、local_id 管理 |
| `logic/boxmot_adapter.py` | BoxMOT 追蹤器介接（以 uid 索引） |
| `logic/color.py` | 類別標籤與顏色 |
| `verify_interval.py` | 離線驗證：明細 DB → 區間彙總 Value 表格 + CSV |
| `ds_yaml/*.yaml` | 每路 cam 的設定（來源、weight、ROI、方向、追蹤器等） |

---

## YAML 設定範例重點

```yaml
source_id: "camC"                 # 寫進 DB CameraCode
device: {code: "EdgeX317"}        # 寫進 DB DeviceCode

source: "videos/test3.avi"        # 影片檔 / rtsp:// / http://
weight: "yolo11s_fp16.engine"     # 相同 weight 的 cam 會被歸到同一條 pipeline
keep_classes: [0]                 # 只保留原始 class_id=0（COCO person）；不寫=全收

geometry:
  base_w: 1280                    # ⭐ 來源真實寬（ROI/crop 自動縮放的依據）
  base_h: 720                     # ⭐ 來源真實高
  regions:                        # 計數 ROI，直接用來源真實點位標記
    roi_1: [[0,200],[1280,200],[1280,720],[0,720]]
  crop_points: [[0,100],[1280,100],[1280,720],[0,720]]   # 裁切遮罩

track_logic:
  axis: "x"                       # 'y'=上下判進出 / 'x'=左右判進出
  movement_threshold: 30
  min_roi_hits: 2

tracker:
  type: "fasttracker"             # nvdcf / bytetrack / ocsort / fasttracker / sfsort / cbiou
```

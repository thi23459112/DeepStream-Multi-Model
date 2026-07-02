"""
類別顏色與標籤對應表（車流版）

主要功能：
1. 提供類別 ID → OSD bbox 顏色對應 (RGBA)
2. 載入 labels_car.txt 並建立類別 ID → 車種名稱對應表
3. 啟動時自動初始化 CLASS_MAP，供 probes.py 直接 import 使用
"""

import os


# ==========================================
# 1. 系統配置區 (System Configuration)
# ==========================================

# 自動偵測專案根目錄 (本檔位於 <project_root>/logic/color.py)
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_FILE  = f"{BASE_DIR}/labels_car.txt"

# --- 車種顏色表 (RGBA, 0.0 ~ 1.0) ---
# 對應 labels_car.txt 內各車種類別
CLASS_COLORS_RGBA = {
    0: (0.00, 0.50, 0.50, 1.0),   # bicycle    - 橄欖
    1: (0.82, 0.41, 0.12, 1.0),   # bus        - 巧克力
    2: (1.00, 0.00, 0.00, 1.0),   # car        - 紅
    3: (0.00, 0.00, 1.00, 1.0),   # motorbike  - 藍
    4: (1.00, 0.00, 1.00, 1.0),   # smalltruck - 紫
    5: (0.55, 0.27, 0.07, 1.0),   # trailer    - 咖啡棕
    6: (1.00, 0.39, 0.00, 1.0),   # truck      - 暗橘
}

# --- 系統設定 ---
DEFAULT_COLOR_RGBA = (1.0, 1.0, 1.0, 1.0)   # 未定義類別 fallback：白色


# ==========================================
# 2. 顏色查詢功能 (Color Lookup)
# ==========================================

def get_class_color(cls_id: int):
    """
    依車種 ID 取得 OSD bbox 顏色

    處理流程：
    1. 表內有定義 → 直接回傳
    2. 表內沒定義但表非空 → 用 modulo 取其中一個顏色 (循環使用)
    3. 表是空的 → 回傳 DEFAULT_COLOR_RGBA (白色)

    參數：
        cls_id (int): 車種類別 ID

    返回：
        tuple: (R, G, B, A) 0.0 ~ 1.0 範圍的浮點顏色
    """
    if cls_id in CLASS_COLORS_RGBA:
        return CLASS_COLORS_RGBA[cls_id]

    num_defined = len(CLASS_COLORS_RGBA)
    if num_defined > 0:
        return CLASS_COLORS_RGBA.get(cls_id % num_defined, DEFAULT_COLOR_RGBA)
    return DEFAULT_COLOR_RGBA


# ==========================================
# 3. 標籤檔載入功能 (Label File Loader)
# ==========================================

def load_labels(label_path):
    """
    從 labels.txt 載入類別 ID → 車種名稱對應表

    檔案格式：每行一個車種名稱，行號即為類別 ID (從 0 起算)

    參數：
        label_path (str): labels.txt 的絕對路徑

    返回：
        dict: {class_id: class_name}
              檔案不存在時返回空 dict (probes.py 會 fallback 顯示原始 ID)
    """
    class_map = {}

    # 檔案不存在時提早返回
    if not os.path.exists(label_path):
        print(f"[WARNING] 找不到標籤檔 {label_path}，將使用預設顯示 ID。")
        return class_map

    # 逐行讀取，行號 = 類別 ID
    with open(label_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            class_name = line.strip()
            if class_name:
                class_map[idx] = class_name

    print(f"[INFO] 成功載入 {len(class_map)} 個類別標籤從 {os.path.basename(label_path)}")
    return class_map


# ==========================================
# 4. 模組初始化 (Module Initialization)
# ==========================================

# 模組載入時自動建立 CLASS_MAP，給 probes.py 直接 import 使用
CLASS_MAP = load_labels(LABEL_FILE)
"""
類別標籤與顏色

    load_labels      讀 labels 檔 → {class_id: 名稱}
    get_class_color  class_id → bbox 與標籤底色（RGBA）
"""

import os


# =============================================================================
# 常數
# =============================================================================

# bbox 顏色表，RGBA 各分量 0.0 ~ 1.0，索引 = class_id（對應 labels_car.txt）。
# 其他模型（例如 COCO 80 類）的 class_id 超出範圍時循環取用。
CLASS_COLORS_RGBA = (
    (0.00, 0.50, 0.50, 1.0),    # 0 bicycle     藍綠
    (0.82, 0.41, 0.12, 1.0),    # 1 bus         巧克力
    (1.00, 0.00, 0.00, 1.0),    # 2 car         紅
    (0.00, 0.00, 1.00, 1.0),    # 3 motorbike   藍
    (1.00, 0.00, 1.00, 1.0),    # 4 smalltruck  紫
    (0.55, 0.27, 0.07, 1.0),    # 5 trailer     咖啡
    (1.00, 0.39, 0.00, 1.0),    # 6 truck       橘
)


# =============================================================================
# 函式
# =============================================================================

def get_class_color(cls_id):
    """回傳 class_id 對應的 (R, G, B, A)；超出顏色表範圍時循環取用。"""
    return CLASS_COLORS_RGBA[cls_id % len(CLASS_COLORS_RGBA)]


def load_labels(label_path):
    """
    讀 labels 檔：每行一個類別名稱，行號（從 0 起算）即 class_id。

    檔案不存在時回傳空 dict，顯示與 DB 會退回 "Class_<id>"。
    """
    if not os.path.exists(label_path):
        print(f"[WARNING] 找不到標籤檔 {label_path}，類別名稱將顯示為 Class_<id>")
        return {}

    class_map = {}
    with open(label_path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            name = line.strip()
            if name:    # 空行不算類別，但仍占用行號
                class_map[idx] = name
    return class_map

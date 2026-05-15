# =============================================================
# config.py — 所有可調整參數集中管理
# 修改任何數值只需動這個檔案，不需要進入各模組
# =============================================================

import os


# ── 專案路徑 ──────────────────────────────────────────────────
PROJECT_ROOT        = os.path.dirname(os.path.abspath(__file__))

# ── QR code 開關 ──────────────────────────────────────────────
USE_QR               = False  # True = 啟用 QR code 比例尺（方法一）
                               # False = 跳過 QR 偵測，只用焦距法（方法二）

# ── 標籤設定 ──────────────────────────────────────────────────
QR_REAL_SIZE_CM      = 3.2    # QR code 的真實寬度（公分）
                               # 用來當比例尺：知道 QR 幾公分，
                               # 就能算出 1 像素等於幾公分

# ── 拍攝設定 ──────────────────────────────────────────────────
MAX_DISTANCE_M       = 7.0    # 拍攝距離上限（公尺）
                               # 超過這個距離 QR code 可能認不到

DBH_HEIGHT_M         = 1.3    # 胸高直徑量測高度（公尺）
                               # 目前維持原架構：從樹幹 mask 最底部往上 1.3m
                               # 之後如果高度改變，只需要改這裡

# ── 手機相機預設值 ────────────────────────────────────────────
# 實際物理焦距（mm）與感光元件寬度（mm），非 35mm 等效值。
# 使用者選「其他」時，由使用者自行輸入實際焦距與感光元件寬度。
PHONE_MODEL_OTHER    = "其他"
CAMERA_PRESETS       = {
    "iPhone 13": {
        "focal_mm": 5.7,
        "sensor_width_mm": 7.5,
    },
    "iPhone 13 Pro": {
        "focal_mm": 5.8,
        "sensor_width_mm": 7.8,
    },
    "iPhone 14": {
        "focal_mm": 5.7,
        "sensor_width_mm": 7.5,
    },
    "iPhone 14 Pro": {
        "focal_mm": 6.9,
        "sensor_width_mm": 10.0,
    },
    "iPhone 15": {
        "focal_mm": 6.2,
        "sensor_width_mm": 8.2,
    },
    "iPhone 15 Pro": {
        "focal_mm": 6.9,
        "sensor_width_mm": 10.0,
    },
    "iPhone 16": {
        "focal_mm": 6.2,
        "sensor_width_mm": 8.2,
    },
    "iPhone 16 Pro": {
        "focal_mm": 6.9,
        "sensor_width_mm": 10.0,
    },
    "Samsung Galaxy S24 Ultra": {
        "focal_mm": 6.5,
        "sensor_width_mm": 9.9,
    },
}
PHONE_MODEL_OPTIONS  = list(CAMERA_PRESETS.keys()) + [PHONE_MODEL_OTHER]

# ── 模型設定 ──────────────────────────────────────────────────
MODEL_PATH           = os.path.join(PROJECT_ROOT, "best.pt")  # YOLO 模型權重檔的路徑
CONF_THRESHOLD       = 0.65       # YOLO 信心度閾值
                                   # 低於 0.65 的偵測結果會被丟掉
                                   # 避免把背景誤認成樹幹

# ── 幾何計算設定（geometry.py 使用）─────────────────────────
DBH_SLICE_COUNT      = 20    # 水平切片數量
                              # 在胸高位置上下各取 10 條線，共 20 條
                              # 取平均值讓結果更穩定

MIN_VALID_SLICES     = 8     # 最少有效切片數
                              # 20 條裡面至少要有 8 條有效
                              # 不到 8 條代表樹幹太細或遮擋太多

MIN_MARKER_PX        = 30    # QR code 最小像素寬度
                              # 小於 30px 代表距離太遠，比例尺不準

# ── 驗證設定（validator.py 使用）────────────────────────────
SIMILARITY_THRESHOLD = 0.80  # 雙驗證相似度門檻
                              # 兩種方法的結果差異超過 20% 就警告

# ── 輸出設定 ──────────────────────────────────────────────────
OUTPUT_DIR           = "~/Desktop/results"  # 結果輸出目錄

# ── 標籤設定 ──────────────────────────────
QR_REAL_SIZE_CM      = 5.0       # QR code 真實尺寸（公分）
MAX_DISTANCE_M       = 7.0       # 拍攝距離上限警告閾值（公尺）

# ── 模型設定 ──────────────────────────────
MODEL_PATH           = "best.pt"
CONF_THRESHOLD       = 0.65      # YOLO 信心度閾值

# ── 幾何計算設定 ──────────────────────────
DBH_SLICE_COUNT      = 20        # 水平切片數量
MIN_VALID_SLICES     = 8         # 最少有效切片數

# ── 驗證設定 ──────────────────────────────
SIMILARITY_THRESHOLD = 0.80      # 雙驗證相似度門檻（80%）

# ── 輸出設定 ──────────────────────────────
OUTPUT_DIR           = "~/Desktop/results"
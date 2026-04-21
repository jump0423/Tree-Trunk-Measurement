# =============================================================
# qr_calculator.py — QR code 比例尺計算器
# 負責人：昌
#
# 這個模組只負責一件事：
#   把「QR code 的像素寬度」換算成「比例尺（公分/像素）」
#
# 設計原則：輸入像素寬度 → 輸出比例尺，不含任何偵測邏輯
# 這樣設計的好處是：可以單獨測試計算邏輯，不需要跑相機或 QR 偵測器
# =============================================================

import config


class QRCalculator:
    """
    QR code 比例尺計算器

    已知 QR code 的真實大小（公分），
    只要量出它在影像中的像素寬度，就能算出比例尺：
        scale = QR_REAL_SIZE_CM / qr_pixel_width  (cm/px)

    這個比例尺之後用來把「像素距離」轉換成「公分距離」。
    例如：樹幹在影像中寬 200px，scale = 0.025 cm/px
          → 樹幹真實寬度 = 200 × 0.025 = 5.0 cm
    """

    def compute_scale(self, qr_pixel_width: float) -> float:
        """
        根據 QR code 的像素寬度計算比例尺

        參數說明：
            qr_pixel_width : float
                QR code 在影像中的像素寬度
                由 QRDetector 偵測後傳入
                單位：像素（px）

        回傳值：
            float
                比例尺，單位為「公分/像素（cm/px）」
                若輸入無效則回傳 0.0（代表比例尺計算失敗，不可使用）
        """

        # ── 防呆一：像素寬度必須為正數 ────────────────────────────
        #
        # qr_pixel_width <= 0 代表兩種情況：
        #   1. QRDetector 偵測失敗，傳入預設值 0 或負數
        #   2. 程式邏輯錯誤（不應該發生，但仍需防範）
        # 這兩種情況都無法計算有意義的比例尺，直接回傳 0.0
        # 呼叫端看到 0.0 就知道要切換備援方法（焦距公式）
        if qr_pixel_width <= 0:
            return 0.0

        # ── 防呆二：像素寬度不能小於最小有效值 ────────────────────
        #
        # 若 QR code 在畫面中太小（小於 MIN_MARKER_PX = 30px）
        # 代表相機距離太遠，QR code 的像素解析度不足
        # 此時 QR code 邊緣不清晰，偵測到的像素寬度誤差很大
        # 用這種影像算出的比例尺誤差會放大到整個測量結果，不能使用
        # 因此同樣回傳 0.0，讓呼叫端知道比例尺無效
        if qr_pixel_width < config.MIN_MARKER_PX:
            return 0.0

        # ── 計算比例尺 ─────────────────────────────────────────────
        #
        # 公式：scale = QR code 真實寬度（cm）÷ QR code 像素寬度（px）
        # 結果單位：cm/px（每個像素代表多少公分）
        #
        # 計算範例：
        #   QR_REAL_SIZE_CM = 5.0 cm（印出來的 QR code 邊長）
        #   qr_pixel_width  = 200 px（在影像中量到的像素寬度）
        #   scale = 5.0 / 200 = 0.025 cm/px
        #   → 影像中每 1 個像素 = 0.025 公分
        #
        # 為什麼用除法而不是乘法？
        # 因為 scale 的定義是「單位像素代表多少公分」
        # 越近拍 → QR code 越大（像素多）→ scale 越小（每像素公分少）
        # 越遠拍 → QR code 越小（像素少）→ scale 越大（每像素公分多）
        return config.QR_REAL_SIZE_CM / qr_pixel_width

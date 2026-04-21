# =============================================================
# geometry.py — 核心幾何計算模組
# 負責人：jump
#
# 這個模組只做「計算」，不做任何偵測或顯示
# 所有跟像素轉公分有關的數學都集中在這裡
# =============================================================

import cv2
import numpy as np
import config


class GeometryEngine:
    """
    幾何計算引擎
    
    主要功能：
    1. 在胸高位置（target_y）取 20 條水平切片
    2. 用 IQR 方法過濾掉異常值（例如樹皮凹陷或突起）
    3. 取過濾後的平均值作為樹幹直徑
    4. 同時回傳標準差（std_cm）和信心等級（confidence）
       讓使用者知道這次測量結果穩不穩定
    """

    def compute_target_y(self, trunk_pts: np.ndarray, scale: float) -> float:
        """
        根據樹幹輪廓點和比例尺，計算胸高（1.3m）對應的影像 y 座標

        參數說明：
            trunk_pts : np.ndarray
                YOLO 回傳的樹幹輪廓點陣列，shape 為 (N, 2)
                每一列是 [x座標, y座標]，單位是像素

            scale : float
                比例尺，單位是「公分/像素（cm/px）」
                由 QRCalculator.compute_scale() 算出

        回傳值：
            float
                胸高量測位置的影像 y 座標（像素）
                若輸入無效則回傳 -1.0（代表無法計算，呼叫端需處理）
        """

        # ── 防呆：比例尺或輪廓點無效時直接回傳 -1.0 ──────────────
        #
        # scale <= 0 代表 QRCalculator 計算失敗（距離太遠或偵測失敗）
        # trunk_pts 為空代表 YOLO 沒有偵測到樹幹輪廓
        # 這兩種情況都無法計算有效的 target_y，回傳 -1.0 作為錯誤碼
        # get_diameter_at_height() 看到 target_y < 0 會提早回傳失敗結果
        if scale <= 0 or len(trunk_pts) == 0:
            return -1.0

        # ── 第一步：找出 mask 最低點（影像 y 值最大的點）────────────
        #
        # 影像座標系的 y 軸方向：向下為正，向上為負
        # 因此「y 值最大的點」就是影像中「最低的點」
        # 樹幹輪廓的最低點對應地面（樹根接地處），作為地面基準
        #
        # 為什麼用 mask 最低點當基準，而不是用影像底部？
        # 因為拍攝角度、構圖不同，樹幹底部不一定落在影像邊緣
        # 用實際偵測到的樹幹底部當地面基準，比固定用影像底部更準確
        mask_bottom_y = trunk_pts[:, 1].max()

        # ── 第二步：把 1.3m 換算成像素距離 ───────────────────────
        #
        # CAMERA_HEIGHT_M = 1.3（公尺），先乘以 100 轉成公分
        # 再除以 scale（cm/px）得到對應的像素距離
        #
        # 公式：offset_px = (高度公分) / (cm/px) = 像素數量
        # 範例：
        #   CAMERA_HEIGHT_M = 1.3m = 130cm
        #   scale = 0.025 cm/px
        #   offset_px = 130 / 0.025 = 5200px
        #   → 從地面往上 5200 個像素就是胸高位置
        offset_px = (config.CAMERA_HEIGHT_M * 100) / scale

        # ── 第三步：往上偏移，算出胸高的 y 座標 ──────────────────
        #
        # 影像 y 軸向下為正，所以「往上」= y 值減小
        # 從地面基準（mask_bottom_y）往上 offset_px 個像素：
        #   target_y = mask_bottom_y - offset_px
        #
        # 視覺化：
        #   y=0    ──── 影像頂部
        #     ↓
        #   target_y ── 胸高量測線（這裡量直徑）← 我們要算的位置
        #     ↓    offset_px（1.3m 換算的像素距離）
        #   mask_bottom_y ── 樹幹最低點（地面基準）
        #     ↓
        #   y=max  ──── 影像底部
        target_y = mask_bottom_y - offset_px

        return target_y

    def get_diameter_at_height(
        self,
        trunk_pts: np.ndarray,
        target_y: float,
        scale: float
    ) -> dict:
        """
        在指定高度（target_y）計算樹幹直徑

        參數說明：
            trunk_pts : np.ndarray
                YOLO 回傳的樹幹輪廓點陣列，shape 為 (N, 2)
                每一列是 [x座標, y座標]，單位是像素

            target_y : float
                要量測的高度（像素座標）
                因為相機固定在 1.3m，所以傳入 img_height / 2
                （畫面中點就是胸高的位置）
                這個值從外面傳進來，方便之後調整

            scale : float
                比例尺，單位是「公分/像素」
                由 QR code 的實際大小 ÷ QR code 的像素寬度算出來
                例如：QR code 是 5cm，在畫面裡是 100px
                      scale = 5 / 100 = 0.05 cm/px

        回傳值：dict，包含三個欄位
            diameter_cm : float  最終樹幹直徑（公分）
            std_cm      : float  標準差（公分），越小代表測量越穩定
            confidence  : str    "high" / "medium" / "low"
                                 根據有效切片數和標準差判斷
        """

        # ── 防呆：target_y 無效時提早回傳 ────────────────────────
        #
        # target_y < 0 代表 compute_target_y() 計算失敗（回傳了 -1.0）
        # 原因可能是比例尺無效或輪廓點為空
        # 繼續往下執行沒有意義，直接回傳失敗結果
        if target_y < 0:
            return {"diameter_cm": 0.0, "std_cm": 0.0, "confidence": "low"}

        # ── 第一步：在 target_y 附近取 20 條水平切片 ──────────────
        #
        # 為什麼不只取一條線？
        # 因為樹皮表面凹凸不平，只取一個點可能剛好量到凹陷
        # 取 20 條再取平均，結果更穩定
        #
        # 切片範圍：target_y 上下各 10 個像素，共 20 條
        # 例如 target_y = 320，就取 y = 310, 311, ..., 329

        half = config.DBH_SLICE_COUNT // 2  # 等於 10

        # YOLO 的 masks.xy 是稀疏多邊形輪廓（每幾個像素才有一個頂點）
        # 直接用點座標搜尋每條切片會找不到足夠的點
        # 解法：先把多邊形填充成密集的二值 bitmap，再逐行掃描找 x 範圍
        x_max = int(trunk_pts[:, 0].max()) + 1
        y_max = int(trunk_pts[:, 1].max()) + 1
        raster = np.zeros((y_max + 1, x_max + 1), dtype=np.uint8)
        pts_int = trunk_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(raster, [pts_int], 255)

        widths_px = []  # 用來存每條切片量到的樹幹寬度

        for offset in range(-half, half):
            y = int(target_y + offset)

            # 確保 y 在 bitmap 範圍內
            if y < 0 or y >= raster.shape[0]:
                continue

            # 找出該行所有被填充的 x 座標
            xs = np.where(raster[y, :] > 0)[0]

            # 至少要有左右兩個邊緣點才能算寬度
            if len(xs) < 2:
                continue

            width = float(xs[-1] - xs[0])
            widths_px.append(width)

        # ── 第二步：判斷有效切片數是否足夠 ───────────────────────
        #
        # 如果有效切片太少（低於 MIN_VALID_SLICES = 8）
        # 代表樹幹被遮擋太多，或偵測框不完整，結果不可信

        if len(widths_px) < config.MIN_VALID_SLICES:
            # 回傳失敗結果，diameter_cm = 0 代表測量失敗
            return {
                "diameter_cm": 0.0,
                "std_cm":      0.0,
                "confidence":  "low"
            }

        # ── 第三步：用 IQR 方法過濾異常值 ────────────────────────
        #
        # IQR（四分位距）是統計學的離群值過濾方法
        # 原理：
        #   Q1 = 第 25 百分位數（排序後 25% 的位置）
        #   Q3 = 第 75 百分位數（排序後 75% 的位置）
        #   IQR = Q3 - Q1（中間 50% 的範圍）
        #
        # 規則：
        #   不在 [Q1 - 1.5×IQR, Q3 + 1.5×IQR] 範圍內的值視為異常
        #   這些異常值可能是樹皮突起或輪廓點雜訊，直接去掉
        #
        # 例如：
        #   widths_px = [98, 99, 100, 101, 150]  ← 150 是異常值
        #   過濾後變成 [98, 99, 100, 101]，用這些算平均更準確

        arr = np.array(widths_px)
        q1  = np.percentile(arr, 25)   # 第 25 百分位
        q3  = np.percentile(arr, 75)   # 第 75 百分位
        iqr = q3 - q1                  # 四分位距

        # 過濾：只保留在正常範圍內的切片
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered = arr[(arr >= lower) & (arr <= upper)]

        # 如果過濾後沒剩幾個，也視為失敗
        if len(filtered) < config.MIN_VALID_SLICES:
            return {
                "diameter_cm": 0.0,
                "std_cm":      0.0,
                "confidence":  "low"
            }

        # ── 第四步：計算最終直徑 ──────────────────────────────────
        #
        # 用過濾後的切片取平均值（比中位數更能反映整體）
        # 再乘以比例尺（scale），把「像素」轉換成「公分」

        mean_px    = float(np.mean(filtered))   # 平均像素寬度
        std_px     = float(np.std(filtered))    # 標準差（像素）

        diameter_cm = round(mean_px * scale, 2) # 換算成公分，保留 2 位小數
        std_cm      = round(std_px  * scale, 3) # 標準差也換算成公分

        # ── 第五步：判斷信心等級 ──────────────────────────────────
        #
        # 根據兩個指標綜合判斷：
        #   1. 有效切片數：越多越可信
        #   2. 標準差：越小代表各切片結果越一致，測量越穩定
        #
        # high   → 可以直接使用
        # medium → 可以使用，但建議重拍確認
        # low    → 結果不可靠，建議重拍

        n = len(filtered)  # 有效切片數

        if n >= 15 and std_cm < 0.5:
            confidence = "high"
        elif n >= 10 and std_cm < 1.0:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "diameter_cm": diameter_cm,
            "std_cm":      std_cm,
            "confidence":  confidence
        }
# =============================================================
# geometry.py — 核心幾何計算模組
# 負責人：jump
#
# 這個模組的唯一職責：
#   把「像素資訊」換算成「真實公分數」
#
# 兩個主要任務：
#   1. compute_target_y()：從 mask 最低點往上 1.3m，算出量測位置
#   2. get_diameter_at_height()：在該位置取 20 條切片算樹徑
#
# 設計原則：
#   - 與 YOLO 完全解耦，不 import ultralytics
#   - 只接收已經處理好的 numpy 陣列，不做任何偵測
#   - 可以獨立進行單元測試，不需要跑完整流程
#
# 關於「胸徑（DBH）」：
#   DBH = Diameter at Breast Height（胸高直徑）
#   林業國際標準：離地面 1.3 公尺處的樹幹直徑
#   這是評估樹木大小、材積、蓄積量的最重要指標
#   本模組以 mask 最低點作為「地面基準」，往上計算 1.3m
# =============================================================

import numpy as np   # 數值計算：陣列操作、百分位數、平均值等
import config        # 全域設定（DBH_HEIGHT_M、DBH_SLICE_COUNT 等）


class GeometryEngine:
    """
    幾何計算引擎

    主要功能：
        1. 從 mask 最低點往上 config.DBH_HEIGHT_M（1.3m），
           算出量測用的 target_y（像素 y 座標）
        2. 在 target_y 附近取 DBH_SLICE_COUNT（20）條水平切片
        3. 用 IQR 方法過濾掉異常值（樹皮突起、輪廓點雜訊等）
        4. 取過濾後的平均值換算成公分，作為樹幹直徑
        5. 同時回傳標準差（std_cm）和信心等級（confidence）

    為什麼要取 20 條切片而不是 1 條？
        樹皮表面天然凹凸不平
        只取一條線可能剛好量到凹陷處，造成低估
        取 20 條再取平均，結果更穩定、更貼近真實直徑

    為什麼要用 IQR 過濾？
        YOLO 偵測的輪廓點不完美，可能有雜訊點
        IQR（四分位距）可以自動去掉離群的切片
        讓最終結果不被少數異常值拉偏
    """

    # ----------------------------------------------------------
    # 公開方法 1：計算量測位置
    # ----------------------------------------------------------

    def compute_target_y(
        self,
        trunk_pts: np.ndarray,
        scale: float
    ) -> float:
        """
        從 mask 最低點往上 1.3m，算出量測用的 target_y（像素座標）

        ── 為什麼從「最低點往上」？ ──────────────────────────────
        DBH 標準要求在離地 1.3m 處量測
        相機拍到的不一定從地面開始，但 mask 最低點（樹幹底部）
        可以視為「可見範圍的地面基準」
        從那裡往上 1.3m 就是要量測的位置

        ── 影像座標系方向說明 ──────────────────────────────────
        影像左上角是原點 (0, 0)
        x 軸向右為正
        y 軸向下為正  ← 特別注意！跟數學座標相反
        所以「往上走」= y 值「減小」

        ── 計算公式 ──────────────────────────────────────────
        mask_bottom_y  = max(trunk_pts[:, 1])         # 最低點 y（y 最大）
        height_cm      = DBH_HEIGHT_M × 100           # 1.3m → 130cm
        offset_px      = height_cm / scale            # 130cm / (cm/px) → px
        target_y       = mask_bottom_y - offset_px    # 往上偏移

        參數說明：
            trunk_pts : np.ndarray, shape (N, 2)
                YOLO Instance Segmentation 回傳的輪廓點陣列
                每一列格式：[x座標, y座標]，單位：像素
                由 trunk_detector.py 的 detect() 提供

            scale : float
                比例尺，單位「公分/像素」
                由 QRCalculator.compute_scale() 計算得出
                例如 0.05 代表 1px = 0.05cm

        回傳值：
            float
                target_y：要量測的像素 y 座標
                若輸入無效，回傳 -1.0（讓下游函式做防呆處理）

        失敗情境：
            - scale <= 0：QR code 偵測失敗，無法換算距離
            - trunk_pts 為空：YOLO 沒偵測到樹幹
            以上情況回傳 -1.0
        """

        # ── 防呆：輸入無效時早期回傳 ─────────────────────────
        if scale <= 0:
            # scale 不合理，無法做換算
            return -1.0
        if len(trunk_pts) == 0:
            # 沒有輪廓點，找不到 mask 最低點
            return -1.0

        # ── 第一步：找出 mask 最低點 ──────────────────────────
        #
        # trunk_pts[:, 1] 取出所有點的 y 座標（第二欄）
        # y 軸向下，所以 y 最大的點在畫面最下方 = mask 最低點
        # float() 確保回傳標準 Python float，而不是 numpy scalar
        mask_bottom_y = float(trunk_pts[:, 1].max())

        # ── 第二步：把 1.3m 換算成像素距離 ───────────────────
        #
        # 步驟分解：
        #   config.DBH_HEIGHT_M = 1.3（公尺）
        #   × 100 → 130.0 公分
        #   ÷ scale（公分/像素）→ offset_px（像素）
        #
        # 例如 scale = 0.05 cm/px：
        #   offset_px = 130.0 / 0.05 = 2600 px
        #   → 在這張影像中，1.3m 的高度對應 2600 個像素
        height_cm = config.DBH_HEIGHT_M * 100.0    # 公尺 → 公分
        offset_px = height_cm / scale              # 公分 → 像素

        # ── 第三步：往上偏移（y 值減小） ──────────────────────
        #
        # 因為 y 軸向下，所以「往上走 offset_px」= y 值「減去 offset_px」
        # target_y < mask_bottom_y（在畫面中更高的位置）
        target_y = mask_bottom_y - offset_px

        return target_y

    # ----------------------------------------------------------
    # 公開方法 2：計算樹幹直徑
    # ----------------------------------------------------------

    def get_diameter_at_height(
        self,
        trunk_pts: np.ndarray,
        target_y: float,
        scale: float
    ) -> dict:
        """
        在指定高度（target_y）計算樹幹直徑

        這個方法是整個計算流程的核心：
            1. 在 target_y 上下各取 10 條水平切片（共 20 條）
            2. 每條切片找出樹幹左邊緣和右邊緣的 x 座標差 = 該切片的寬度
            3. 用 IQR 過濾掉異常寬度（樹皮突起、遮擋等造成的離群值）
            4. 取過濾後的平均像素寬度，乘以 scale → 公分直徑
            5. 根據有效切片數和標準差判斷信心等級

        參數說明：
            trunk_pts : np.ndarray, shape (N, 2)
                YOLO 回傳的樹幹輪廓點陣列
                每一列：[x座標, y座標]，單位：像素

            target_y : float
                要量測的像素 y 座標
                由 compute_target_y() 計算得出
                若為負值（-1.0），代表上游計算失敗

            scale : float
                比例尺，單位「公分/像素」
                由 QRCalculator.compute_scale() 算出

        回傳值：
            dict，包含三個欄位：
                "diameter_cm" : float  最終樹幹直徑（公分），失敗時為 0.0
                "std_cm"      : float  切片標準差（公分），越小越穩定
                "confidence"  : str    "high" / "medium" / "low"

        信心等級判斷標準：
            "high"   → 有效切片 >= 15 條，且標準差 < 0.5cm  可直接使用
            "medium" → 有效切片 >= 10 條，且標準差 < 1.0cm  建議重拍確認
            "low"    → 其他情況                              建議重拍
        """

        # ── 防呆：target_y 無效時直接回傳失敗 ────────────────
        #
        # compute_target_y() 失敗時回傳 -1.0
        # 這裡攔截，不繼續做無意義的計算
        if target_y < 0:
            return {
                "diameter_cm": 0.0,
                "std_cm":      0.0,
                "confidence":  "low"
            }

        # ── 第一步：在 target_y 附近取水平切片 ───────────────
        #
        # config.DBH_SLICE_COUNT = 20
        # half = 10，代表 target_y 上下各 10 條，共 20 條
        #
        # 切片的 y 座標範圍：
        #   offset = -10, -9, ..., -1, 0, 1, ..., 9
        #   y = target_y + offset
        #   → 覆蓋 [target_y - 10, target_y + 9] 的範圍
        half = config.DBH_SLICE_COUNT // 2  # = 10

        widths_px = []  # 儲存每條有效切片的像素寬度

        for offset in range(-half, half):

            # 這條切片的 y 座標
            y = target_y + offset

            # ── 找出這條切片上的輪廓點 ───────────────────────
            #
            # 為什麼用「± 1.0」的容差，而不是剛好等於 y？
            # YOLO 的輪廓點是浮點數，不保證剛好落在整數 y 值上
            # 用容差 1.0 確保能找到附近的點
            #
            # trunk_pts[:, 1] 是所有點的 y 座標
            # np.abs(...) 計算每個點的 y 和目標 y 的距離
            # <= 1.0 篩出距離在 1 個像素內的點
            nearby = trunk_pts[
                np.abs(trunk_pts[:, 1] - y) <= 1.0
            ]

            # 至少需要 2 個點才能算寬度（左邊緣 + 右邊緣）
            # 只有 0 或 1 個點的切片跳過
            if len(nearby) < 2:
                continue

            # 用 x 座標的範圍算這條切片的像素寬度
            # max(x) - min(x) = 右邊緣 x - 左邊緣 x = 樹幹寬度
            width = nearby[:, 0].max() - nearby[:, 0].min()
            widths_px.append(width)

        # ── 第二步：檢查有效切片是否足夠 ─────────────────────
        #
        # config.MIN_VALID_SLICES = 8
        # 20 條中至少要有 8 條有有效輪廓點
        # 若太少，代表樹幹被遮擋嚴重或 1.3m 的位置超出畫面
        if len(widths_px) < config.MIN_VALID_SLICES:
            return {
                "diameter_cm": 0.0,
                "std_cm":      0.0,
                "confidence":  "low"
            }

        # ── 第三步：IQR 過濾異常值 ────────────────────────────
        #
        # 什麼是 IQR（四分位距）？
        #   把所有切片寬度從小到大排序
        #   Q1 = 排序後第 25% 位置的值
        #   Q3 = 排序後第 75% 位置的值
        #   IQR = Q3 - Q1（中間 50% 的數據範圍）
        #
        # 怎麼判斷異常值？
        #   下界 = Q1 - 1.5 × IQR
        #   上界 = Q3 + 1.5 × IQR
        #   超出 [下界, 上界] 的切片視為異常，直接捨棄
        #
        # 實際例子：
        #   widths_px = [98, 99, 100, 100, 101, 102, 150]
        #   Q1=99, Q3=101, IQR=2, 上界=104
        #   150 超出上界 → 被過濾
        #   剩下 [98, 99, 100, 100, 101, 102] 用來計算平均

        arr = np.array(widths_px)

        q1  = np.percentile(arr, 25)   # 第 25 百分位
        q3  = np.percentile(arr, 75)   # 第 75 百分位
        iqr = q3 - q1                  # 四分位距

        lower    = q1 - 1.5 * iqr                          # 下界
        upper    = q3 + 1.5 * iqr                          # 上界
        filtered = arr[(arr >= lower) & (arr <= upper)]    # 保留正常範圍內的切片

        # 過濾後若剩下的切片還是不夠，視為失敗
        if len(filtered) < config.MIN_VALID_SLICES:
            return {
                "diameter_cm": 0.0,
                "std_cm":      0.0,
                "confidence":  "low"
            }

        # ── 第四步：計算最終直徑 ──────────────────────────────
        #
        # 1. 取過濾後切片的平均像素寬度
        #    用平均而不是中位數，因為平均能反映整體樹幹形狀
        # 2. 乘以 scale（公分/像素），把像素換算成公分
        #
        # 例如：
        #   mean_px = 200.0px，scale = 0.05 cm/px
        #   diameter_cm = 200.0 × 0.05 = 10.0cm

        mean_px = float(np.mean(filtered))   # 平均像素寬度
        std_px  = float(np.std(filtered))    # 標準差（像素單位）

        diameter_cm = round(mean_px * scale, 2)   # 換算成公分，保留 2 位小數
        std_cm      = round(std_px  * scale, 3)   # 標準差換算成公分，保留 3 位小數

        # ── 第五步：判斷信心等級 ──────────────────────────────
        #
        # 兩個指標：
        #   n（有效切片數）：越多代表量測覆蓋越完整
        #   std_cm（公分標準差）：越小代表各切片結果越一致
        #
        # "high"   → n >= 15 且 std_cm < 0.5cm  結果可靠，直接使用
        # "medium" → n >= 10 且 std_cm < 1.0cm  結果尚可，建議重拍確認
        # "low"    → 其他                        結果不穩定，建議重拍

        n = len(filtered)   # 有效切片數（過濾後）

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
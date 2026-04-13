#輔負責
# =============================================================
# focal_calculator.py — 焦距公式計算器（方法二備援）
#
# 當 QR code 偵測失敗時，改用「針孔相機模型」換算樹幹直徑。
#
# 針孔相機模型核心公式（相似三角形原理）：
#
#   真實寬度(cm)     距離(cm)
#   ──────────── = ──────────────
#   像素寬度(px)   焦距像素值(px)
#
#   整理後：
#   真實寬度 = 像素寬度 × 距離(cm) ÷ 焦距像素值(px)
#
#   焦距像素值(px) = (焦距mm ÷ 感光元件寬mm) × 影像寬度px
#
# 流程：
#   1. __init__ 時把 mm 單位的焦距轉成像素單位並存入 self._focal_px
#   2. _to_focal_px() 無參數，直接回傳已計算好的 self._focal_px
#   3. calculate() 用 self._focal_px + 使用者輸入的距離計算樹幹直徑
# =============================================================


class FocalCalculator:
    """
    焦距公式計算器（方法二）

    使用相機的物理參數（焦距、感光元件尺寸）來換算真實距離。
    作為 QR code 比例尺失靈時的備援方法。

    屬性：
        _focal_px (float)：相機焦距，單位為像素（由 __init__ 計算並儲存）
    """

    def __init__(self, focal_mm: float, sensor_width_mm: float, img_width_px: int):
        """
        初始化計算器，把 mm 焦距換算成像素焦距後儲存。

        換算公式：
            焦距(px) = 焦距(mm) ÷ 感光元件寬(mm) × 影像寬度(px)

        參數：
            focal_mm        (float)：相機鏡頭焦距，單位：毫米（mm）
                                     手機約 3~6mm，標準鏡頭約 35~50mm

            sensor_width_mm (float)：感光元件的實體寬度，單位：毫米（mm）
                                     全片幅約 36mm，APS-C 約 23.5mm

            img_width_px    (int)  ：影像的像素寬度，由 image.shape[1] 取得
        """

        # 防呆：感光元件寬度不可為 0，否則後續除以零
        if sensor_width_mm <= 0:
            print("[FocalCalculator] 感光元件寬度無效（<=0），焦距設為 0")
            self._focal_px = 0.0
        else:
            # 換算：mm 焦距 → px 焦距
            # 原理：感光元件寬度(mm) 對應影像寬度(px)，等比例換算焦距
            self._focal_px = (focal_mm / sensor_width_mm) * img_width_px

    def calculate(self, trunk_px: float, distance_m: float) -> float:
        """
        用焦距公式算出樹幹的真實直徑。

        公式：真實寬度(cm) = 樹幹像素寬度 × 距離(cm) ÷ 焦距像素值

        參數：
            trunk_px   (float)：樹幹在影像中的像素寬度
                                由 geometry.py 的切片計算取得

            distance_m (float)：拍攝時與樹幹的距離，單位：公尺
                                由 InputHandler.get_distance() 取得

        回傳值：
            float：樹幹直徑，單位：公分（cm）
                   若焦距像素值為 0（初始化失敗），回傳 0.0
        """

        # 防呆：焦距無效時無法計算
        if self._focal_px <= 0:
            print("[FocalCalculator] 焦距像素值無效，無法計算直徑")
            return 0.0

        # 防呆：樹幹像素寬度不可為 0 或負數
        if trunk_px <= 0:
            print("[FocalCalculator] 樹幹像素寬度無效，無法計算直徑")
            return 0.0

        # 單位換算：公尺 → 公分（公式需要公分）
        distance_cm = distance_m * 100

        # 套用針孔相機公式：真實寬度 = 像素寬度 × 距離 ÷ 焦距
        diameter_cm = (trunk_px * distance_cm) / self._focal_px

        return diameter_cm

    def _to_focal_px(self) -> float:
        """
        回傳相機焦距（像素單位），供 main.py 計算比例尺使用。

        比例尺（cm/px）的計算方式為：
            scale = 距離(cm) ÷ 焦距(px)

        main.py 呼叫範例：
            scale_b = (distance_m * 100) / focal_calc._to_focal_px()

        回傳值：
            float：焦距，單位：像素
                   若初始化失敗則為 0.0（呼叫端需做除以零防護）
        """
        return self._focal_px

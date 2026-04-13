#共用
# =============================================================
# visualizer.py — 結果圖片繪製模組
#
# 負責在原始影像上疊加偵測結果：
#   1. 樹幹分割遮罩（半透明綠色區域）
#   2. 胸高量測線（紅色水平線）
#   3. 右上角狀態標籤（DBH 數值、方法、信心度）
#
# 設計原則：
#   draw() 是唯一對外入口，內部分工給三個私有方法
#   不修改原始影像，先複製一份再畫
# =============================================================

import cv2
import numpy as np


class Visualizer:
    """
    結果圖片繪製器

    使用方式（main.py 呼叫範例）：
        visualizer = Visualizer()
        output_img = visualizer.draw(image, detection, result)
        # output_img 是畫好標注的影像，傳給 FileManager.save_image()
    """

    def draw(self, image: np.ndarray, detection: dict, result) -> np.ndarray:
        """
        統一繪製入口，依序完成所有標注後回傳結果影像。

        參數：
            image     (np.ndarray)       ：原始 OpenCV 影像，BGR 色彩，shape (H, W, 3)
            detection (dict)             ：TrunkDetector.detect() 的回傳值
                                          含 "masks_xy"、"confidence"、"box"
                                          若偵測失敗則為 None
            result    (MeasurementResult)：Validator.validate() 的回傳值，含直徑與狀態

        回傳值：
            np.ndarray：畫好全部標注的影像（原始 image 不會被修改）
        """

        # 複製一份再畫，保留原始影像不被改動
        output = image.copy()

        # 若偵測失敗（detection 為 None），只畫狀態標籤告知使用者
        if detection is None:
            self._draw_status_label(output, result)
            return output

        # 取出輪廓點（供畫遮罩與量測線使用）
        trunk_pts = detection["masks_xy"]  # shape (N, 2)

        # 依序呼叫三個內部繪製方法
        self._draw_mask(output, trunk_pts)            # 1. 半透明綠色遮罩
        self._draw_diameter_line(output, trunk_pts)   # 2. 紅色量測線
        self._draw_status_label(output, result)        # 3. 右上角狀態文字

        return output

    # ----------------------------------------------------------
    # 以下是內部方法，只在類別內部呼叫，外部不直接使用
    # ----------------------------------------------------------

    def _draw_mask(self, image: np.ndarray, trunk_pts: np.ndarray) -> None:
        """
        在影像上繪製半透明的樹幹分割遮罩（綠色）。

        作法說明：
            1. 建立一個全黑的空白圖層（與原始影像同大小）
            2. 用 fillPoly 把樹幹輪廓區域填滿白色
            3. 把填好的白色區域改成綠色，疊加到原始影像（透明度 40%）
            4. 再額外畫一條輪廓線，讓樹幹邊緣更清晰

        為什麼用半透明而不是實心遮罩？
            半透明可以讓使用者同時看到遮罩位置和原始影像，
            方便確認 YOLO 的偵測範圍是否正確。

        參數：
            image     (np.ndarray)：要繪製的影像（直接修改，不回傳）
            trunk_pts (np.ndarray)：樹幹輪廓點，shape 為 (N, 2)，像素座標
        """

        img_h, img_w = image.shape[:2]

        # 建立空白遮罩（全黑），大小與原始影像相同
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        # fillPoly 需要整數座標，且格式為 (N, 1, 2)
        pts = trunk_pts.astype(np.int32).reshape((-1, 1, 2))

        # 把樹幹輪廓內部填滿白色（255 = 白）
        cv2.fillPoly(mask, [pts], color=255)

        # 建立純綠色覆蓋層（與原始影像同大小，預設全黑）
        overlay = np.zeros_like(image)

        # 只把遮罩為白色的位置設成綠色（BGR 格式）
        overlay[mask == 255] = (0, 180, 0)

        # addWeighted 實現半透明疊加：
        #   alpha=0.4 → 覆蓋層佔 40%
        #   beta=1.0  → 原始影像佔 100%（兩者相加再做亮度調整）
        # 結果存回 image（in-place 修改）
        cv2.addWeighted(overlay, 0.4, image, 1.0, 0, image)

        # 再畫一條亮綠色輪廓線，讓樹幹邊緣更清晰
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    def _draw_diameter_line(self, image: np.ndarray, trunk_pts: np.ndarray) -> None:
        """
        在影像上繪製胸高位置的紅色水平量測線。

        量測線位置：取樹幹輪廓的垂直中點作為近似胸高位置。

        說明：
            精確的 1.3m 胸高位置需要比例尺才能計算（由 geometry.py 負責）。
            visualizer 沒有比例尺資訊，因此用輪廓垂直中點來近似，
            目的只是讓使用者看到「量測位置大概在哪裡」，
            真正用來算直徑的 y 座標在 geometry.py 裡。

        繪製內容：
            - 紅色水平線（從輪廓左緣到右緣）
            - 左右兩端各一個紅色圓點（標示量測起訖點）

        參數：
            image     (np.ndarray)：要繪製的影像
            trunk_pts (np.ndarray)：樹幹輪廓點，shape 為 (N, 2)
        """

        img_w = image.shape[1]

        # 找出樹幹輪廓的垂直範圍（y 座標）
        y_top    = int(trunk_pts[:, 1].min())  # 輪廓最高點（y 最小）
        y_bottom = int(trunk_pts[:, 1].max())  # 輪廓最低點（y 最大）

        # 取垂直中點作為量測線的 y 座標
        measure_y = int((y_top + y_bottom) / 2)

        # 在量測線高度附近（±10px 容差）找出輪廓點
        # 用這些點的 x 座標範圍來決定量測線的起訖位置
        nearby_pts = trunk_pts[np.abs(trunk_pts[:, 1] - measure_y) <= 10]

        if len(nearby_pts) >= 2:
            # 有找到附近的輪廓點，量測線從樹左側到樹右側
            x_left  = int(nearby_pts[:, 0].min())
            x_right = int(nearby_pts[:, 0].max())
        else:
            # 找不到附近點（罕見情況），量測線橫跨整張影像
            x_left  = 0
            x_right = img_w

        # 畫紅色水平量測線（BGR 格式，紅色 = (0, 0, 255)，粗度 2px）
        cv2.line(image,
                 (x_left, measure_y), (x_right, measure_y),
                 color=(0, 0, 255), thickness=2)

        # 在線的左端畫圓點（標示量測起點）
        cv2.circle(image, (x_left, measure_y),
                   radius=5, color=(0, 0, 255), thickness=-1)

        # 在線的右端畫圓點（標示量測終點）
        cv2.circle(image, (x_right, measure_y),
                   radius=5, color=(0, 0, 255), thickness=-1)

    def _draw_status_label(self, image: np.ndarray, result) -> None:
        """
        在影像右上角繪製量測結果的狀態標籤。

        顯示內容（從上到下）：
            第 1 行：DBH 直徑數值（公分）
            第 2 行：使用方法（dual / focal_only）
            第 3 行：狀態碼（verified / mismatch / qr_failed）
            第 4 行：YOLO 信心度分數

        顏色規則：
            verified  → 綠色（結果可信）
            mismatch  → 橘色（兩種方法不一致，需注意）
            qr_failed → 紅色（QR code 失靈，使用備援）

        參數：
            image  (np.ndarray)       ：要繪製的影像
            result (MeasurementResult)：量測結果物件
        """

        img_h, img_w = image.shape[:2]

        # ── 根據狀態碼決定文字顏色 ───────────────────────────────
        if result.status == "verified":
            color = (0, 200, 0)     # 綠色（BGR）
        elif result.status == "mismatch":
            color = (0, 165, 255)   # 橘色（BGR）
        else:
            color = (0, 0, 220)     # 紅色（BGR）

        # ── 要顯示的文字清單（每個元素是一行）──────────────────
        lines = [
            f"DBH: {result.diameter_cm:.1f} cm",
            f"Method: {result.method}",
            f"Status: {result.status}",
            f"Conf:   {result.confidence:.2f}",
        ]

        # 文字樣式設定
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7   # 字體大小倍率
        thickness  = 2     # 字體筆畫粗細
        line_gap   = 30    # 每行文字之間的垂直間距（像素）
        margin     = 10    # 文字框與影像邊緣的間距

        # ── 計算文字背景框的大小 ─────────────────────────────────
        # 先掃描所有行，找出最長的那行，用來決定背景框寬度
        max_text_w = 0
        for line in lines:
            (text_w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            if text_w > max_text_w:
                max_text_w = text_w

        # 計算背景框的四個角座標（放在右上角）
        box_x1 = img_w - max_text_w - margin * 3
        box_y1 = margin
        box_x2 = img_w - margin
        box_y2 = margin + len(lines) * line_gap + margin

        # ── 畫半透明黑色背景，讓文字在任何背景上都清晰 ──────────
        overlay = image.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2),
                      color=(0, 0, 0), thickness=-1)  # 實心黑色矩形

        # 50% 透明度：原始影像和黑色框各佔一半
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        # ── 逐行繪製文字 ─────────────────────────────────────────
        for i, line in enumerate(lines):
            text_x = box_x1 + margin
            text_y = box_y1 + margin + (i + 1) * line_gap - 5

            # LINE_AA 使用抗鋸齒，讓文字邊緣較平滑
            cv2.putText(image, line, (text_x, text_y),
                        font, font_scale, color, thickness,
                        lineType=cv2.LINE_AA)

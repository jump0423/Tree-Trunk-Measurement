# =============================================================
# error_checker.py — 誤差自動偵測模組
# 負責人：jump
#
# 這個模組在輸出結果之前，自動檢查三件事：
#   1. QR code 是否歪斜超過 15 度
#   2. QR code 像素是否太小（距離太遠）
#   3. 樹幹是否被畫面邊緣裁切
#
# 發現問題時不會讓程式停止，而是把警告訊息
# 寫進 MeasurementResult.warnings 列表裡
# 讓使用者知道「這次結果可能不太準」
# =============================================================

import cv2
import numpy as np
import config


class ErrorChecker:
    """
    誤差自動偵測器

    使用方式：
        checker = ErrorChecker()
        warnings = []
        warnings += checker.check_marker_tilt(qr_contour)
        warnings += checker.check_marker_pixel_size(qr_contour)
        warnings += checker.check_trunk_completeness(trunk_mask, img.shape)
        # 最後把 warnings 寫進 MeasurementResult.warnings
    """

    def check_marker_tilt(self, contour: np.ndarray) -> list:
        """
        檢查 QR code 是否歪斜超過 15 度

        為什麼要檢查傾斜？
            QR code 如果貼歪了，pyzbar 偵測到的像素寬度
            會比實際的寬度小，導致比例尺算錯，DBH 偏大

        參數：
            contour : np.ndarray
                QR code 的輪廓點陣列，shape 為 (N, 1, 2) 或 (N, 2)
                由 OpenCV 的 findContours 或 pyzbar 的 polygon 取得

        回傳：
            list — 空列表代表正常，有字串代表有警告
        """
        warnings = []

        try:
            # 把輪廓點整理成 OpenCV 需要的格式
            pts = contour.reshape(-1, 2).astype(np.float32)

            # minAreaRect 找出最小外接矩形（可以旋轉的）
            # 回傳 (中心點, (寬, 高), 旋轉角度)
            # 旋轉角度範圍是 -90 到 0 度
            rect  = cv2.minAreaRect(pts)
            angle = abs(rect[2])  # 取絕對值

            # OpenCV 的角度定義比較特殊：
            # -90 度代表完全水平（正常），0 度代表完全垂直
            # 所以我們取「距離 -90 的差距」來判斷傾斜程度
            # 例如：angle = -75，傾斜 = 90 - 75 = 15 度
            tilt_deg = min(angle, 90 - angle)

            if tilt_deg > 15:
                warnings.append(
                    f"QR code 歪斜 {tilt_deg:.1f} 度，"
                    f"超過 15 度門檻，比例尺可能不準確，建議重拍"
                )

        except Exception as e:
            # 如果輪廓點格式有問題，記錄錯誤但不讓程式崩潰
            warnings.append(f"check_marker_tilt 發生錯誤：{e}")

        return warnings

    def check_marker_pixel_size(self, contour: np.ndarray) -> list:
        """
        檢查 QR code 在畫面中的像素大小是否足夠

        為什麼要檢查像素大小？
            距離越遠，QR code 在畫面裡越小
            如果 QR code 只有 30 像素寬，
            5cm ÷ 30px = 0.167 cm/px 的比例尺會很不準
            建議至少要有 config.MIN_MARKER_PX（預設 30）像素

        參數：
            contour : np.ndarray
                QR code 的輪廓點陣列

        回傳：
            list — 空列表代表正常，有字串代表有警告
        """
        warnings = []

        try:
            pts  = contour.reshape(-1, 2).astype(np.float32)
            rect = cv2.minAreaRect(pts)

            # 取旋轉矩形的寬和高
            w, h = rect[1]

            # QR code 是正方形，所以取寬高平均
            # 這樣即使稍微傾斜也不會只量到短邊
            avg_px = (w + h) / 2

            if avg_px < config.MIN_MARKER_PX:
                warnings.append(
                    f"QR code 像素寬度只有 {avg_px:.0f}px，"
                    f"小於最低門檻 {config.MIN_MARKER_PX}px，"
                    f"建議靠近到 5 公尺以內拍攝"
                )

        except Exception as e:
            warnings.append(f"check_marker_pixel_size 發生錯誤：{e}")

        return warnings

    def check_trunk_completeness(
        self,
        trunk_mask: np.ndarray,
        img_shape: tuple
    ) -> list:
        """
        檢查樹幹是否被畫面邊緣裁切

        為什麼要檢查？
            如果樹幹的左邊或右邊超出畫面，
            YOLO 只偵測到部分樹幹，
            量出來的寬度會比實際小，DBH 會低估

        參數：
            trunk_mask : np.ndarray
                樹幹輪廓點陣列，shape 為 (N, 2)
                每一列是 [x座標, y座標]

            img_shape : tuple
                原始影像的 shape，格式為 (height, width) 或
                (height, width, channels)
                用 img.shape 取得

        回傳：
            list — 空列表代表正常，有字串代表有警告
        """
        warnings = []

        try:
            img_height = img_shape[0]
            img_width  = img_shape[1]

            # 取得樹幹輪廓的左右邊界
            x_min = trunk_mask[:, 0].min()  # 最左邊的 x 座標
            x_max = trunk_mask[:, 0].max()  # 最右邊的 x 座標

            # 容差設為 10 像素
            # 因為 YOLO 的輪廓點不一定精確到畫面最邊緣
            margin = 10

            # 檢查左邊是否被裁切
            if x_min < margin:
                warnings.append(
                    f"樹幹左側可能超出畫面（x_min={x_min:.0f}px），"
                    f"測量結果可能偏小，建議往右移動相機"
                )

            # 檢查右邊是否被裁切
            if x_max > img_width - margin:
                warnings.append(
                    f"樹幹右側可能超出畫面（x_max={x_max:.0f}px），"
                    f"測量結果可能偏小，建議往左移動相機"
                )

        except Exception as e:
            warnings.append(f"check_trunk_completeness 發生錯誤：{e}")

        return warnings
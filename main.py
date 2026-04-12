# =============================================================
# main.py — 唯一程式入口
# 負責人：jump + Morris（共用）
#
# 這個檔案只負責「串接」所有模組，本身不含任何業務邏輯
# 呼叫順序：
#   InputHandler → TrunkDetector → QRDetector
#   → QRCalculator / FocalCalculator
#   → GeometryEngine → ErrorChecker
#   → Validator → Visualizer → FileManager
#
# 目前狀態：框架版本
#   - 各模組都已 import
#   - 主流程已寫好，但 Morris 的模組還是空的
#   - 等 Morris 的模組完成後，把 TODO 的部分補上即可
# =============================================================

import cv2
import numpy as np

import config
from src.models          import MeasurementResult
from src.geometry        import GeometryEngine
from src.error_checker   import ErrorChecker

# ── Morris 負責的模組（等他寫好後自動生效）────────────────────
from src.trunk_detector  import TrunkDetector
from src.qr_detector     import QRDetector
from src.qr_calculator   import QRCalculator
from src.focal_calculator import FocalCalculator
from src.validator       import Validator
from src.input_handler   import InputHandler
from src.visualizer      import Visualizer
from src.file_manager    import FileManager


def main():
    """
    主程式流程

    整個系統從這裡啟動，按照以下順序執行：
    1. 使用者輸入（選圖片、輸入相機參數）
    2. YOLO 偵測樹幹
    3. QR code 偵測
    4. 計算比例尺和 DBH
    5. 誤差檢查
    6. 雙驗證
    7. 繪製結果圖片
    8. 存檔輸出
    """

    # ── 步驟 1：使用者輸入 ────────────────────────────────────
    # 讓使用者選擇圖片，並輸入相機參數（焦距、感光元件寬度、距離）
    # 這些資訊是「方法二（焦距公式）」的備援計算需要的

    handler        = InputHandler()
    image_path     = handler.get_image_path()      # tkinter 選圖視窗
    focal_mm, sensor_w = handler.get_camera_params()  # 相機參數
    distance_m     = handler.get_distance()        # 拍攝距離

    # 讀取圖片（OpenCV 格式，BGR 色彩空間）
    image = cv2.imread(image_path)
    if image is None:
        print(f"錯誤：無法讀取圖片 {image_path}")
        return

    img_h, img_w = image.shape[:2]  # 取得圖片的高度和寬度（像素）

    # ── 步驟 2：YOLO 偵測樹幹 ─────────────────────────────────
    # 把圖片丟給 YOLO，取回樹幹的輪廓點（masks.xy）和信心度

    detector   = TrunkDetector(config.MODEL_PATH, config.CONF_THRESHOLD)
    detection  = detector.detect(image)  # 回傳偵測結果物件

    if detection is None:
        print("錯誤：YOLO 未偵測到樹幹，請確認圖片中有樹幹且清晰")
        return

    # 從偵測結果取出樹幹輪廓點和信心度
    trunk_pts   = detection["masks_xy"]   # shape: (N, 2)，像素座標
    confidence  = detection["confidence"] # 0.0 ~ 1.0

    # ── 步驟 3：QR code 偵測 ──────────────────────────────────
    # 用 pyzbar 偵測 QR code，取得 QR code 的像素寬度
    # 失敗時回傳 None，程式不會中斷，自動切換到備援方法

    qr_detector = QRDetector()
    qr_result   = qr_detector.detect(image)  # 成功回傳像素寬度，失敗回傳 None

    # ── 步驟 4：計算比例尺和 DBH ──────────────────────────────
    #
    # target_y 是要量測的高度（y 座標，單位像素）
    # 因為相機固定在 1.3m，畫面垂直中點就是胸高
    # 所以 target_y = 圖片高度 / 2
    # 如果之後相機高度改變，只需要改 config.CAMERA_HEIGHT_M
    # （目前版本直接用中點，之後可擴充為用 CAMERA_HEIGHT_M 換算）

    target_y = img_h / 2  # 畫面垂直中點 = 胸高位置

    geometry = GeometryEngine()
    result_a = None  # 方法一結果（QR code 比例尺）
    result_b = None  # 方法二結果（焦距公式備援）

    if qr_detector.is_detected():
        # ── 方法一：QR code 比例尺 ────────────────────────────
        # QR code 偵測成功，用它當比例尺
        # scale = QR code 真實大小（cm）÷ QR code 像素寬度
        # 例如：5cm ÷ 200px = 0.025 cm/px

        qr_calculator = QRCalculator(config.QR_REAL_SIZE_CM)
        scale_a = config.QR_REAL_SIZE_CM / qr_result  # cm/px

        geo_result_a = geometry.get_diameter_at_height(
            trunk_pts, target_y, scale_a
        )
        result_a = geo_result_a["diameter_cm"]  # 取出直徑數值

    # ── 方法二：焦距公式備援 ──────────────────────────────────
    # 不管 QR code 有沒有成功，方法二都會跑
    # 作為對照或備援

    focal_calc = FocalCalculator(focal_mm, sensor_w, img_w)
    scale_b    = focal_calc._to_focal_px() / (distance_m * 100)
    # 注意：上面這行之後 Morris 寫好 FocalCalculator 後可能需要調整

    geo_result_b = geometry.get_diameter_at_height(
        trunk_pts, target_y, scale_b
    )
    result_b = geo_result_b["diameter_cm"]

    # ── 步驟 5：誤差自動檢查 ──────────────────────────────────
    # 在輸出結果前，自動檢查三個可能影響精度的問題
    # 把所有警告收集成列表，之後寫進 MeasurementResult

    checker  = ErrorChecker()
    warnings = []

    if qr_detector.is_detected():
        # 只有 QR code 偵測成功才能檢查 QR code 的狀態
        warnings += checker.check_marker_tilt(qr_result)
        warnings += checker.check_marker_pixel_size(qr_result)

    warnings += checker.check_trunk_completeness(trunk_pts, image.shape)

    # ── 步驟 6：雙驗證，建立最終結果 ─────────────────────────
    # 比較方法一和方法二的差異率
    # 差異 < 20% → 取平均，標記 "verified"
    # 差異 >= 20% → 輸出方法二，標記 "mismatch" 並警告

    validator = Validator(config.SIMILARITY_THRESHOLD)
    result    = validator.validate(result_a, result_b)

    # 把這次的額外資訊補進結果物件
    result.confidence  = confidence
    result.diameter_std = geo_result_b["std_cm"]
    result.warnings    = warnings
    result.image_file  = image_path

    # ── 步驟 7：在圖片上繪製結果 ─────────────────────────────
    # 畫出樹幹遮罩、胸高測量線、DBH 數值、狀態碼

    visualizer = Visualizer()
    output_img = visualizer.draw(image, detection, result)

    # ── 步驟 8：存檔 ──────────────────────────────────────────
    # 儲存標注後的圖片，並把這筆測量結果新增到 CSV

    file_mgr = FileManager(config.OUTPUT_DIR)
    file_mgr.save_image(output_img, image_path)
    file_mgr.save_csv(result)

    # ── 完成，顯示結果摘要 ────────────────────────────────────
    print(f"\n測量完成")
    print(f"  樹徑：{result.diameter_cm:.1f} cm")
    print(f"  方法：{result.method}")
    print(f"  狀態：{result.status}")
    print(f"  信心度：{result.confidence}")
    if result.warnings:
        print(f"  警告：")
        for w in result.warnings:
            print(f"    - {w}")
    print(f"  結果已存至：{config.OUTPUT_DIR}")


# ── 程式進入點 ────────────────────────────────────────────────
# 只有直接執行 main.py 時才會跑 main()
# 如果是被其他模組 import，不會自動執行

if __name__ == "__main__":
    main()
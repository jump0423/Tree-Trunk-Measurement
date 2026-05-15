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
# =============================================================

import cv2
import numpy as np

import config
from src.geometry        import GeometryEngine
from src.error_checker   import ErrorChecker

# ── Morris 負責的模組（等他寫好後自動生效）────────────────────
from src.trunk_detector  import TrunkDetector
from src.focal_calculator import FocalCalculator
from src.validator        import Validator
from src.input_handler    import InputHandler
from src.visualizer       import Visualizer
from src.file_manager     import FileManager
from src.species_manager  import SpeciesManager
from src.carbon_calculator import CarbonCalculator


def main():
    """
    主程式流程

    整個系統從這裡啟動，按照以下順序執行：
    1. 使用者輸入（選圖片、輸入相機參數）—— 只問一次
    2. 逐張圖片執行：YOLO 偵測 → 計算 → 驗證 → 繪圖 → 存檔
    """

    # ── 步驟 1：使用者輸入（只問一次）───────────────────────────
    # 讓使用者一次選多張圖片，並輸入相機參數（焦距、感光元件寬度、距離）
    # 這些資訊是「方法二（焦距公式）」的備援計算需要的

    handler = InputHandler()
    try:
        image_paths        = handler.get_image_paths()     # tkinter 多選視窗，回傳 list
        focal_mm, sensor_w = handler.get_camera_params()   # 相機參數（所有張共用）
        distance_m         = handler.get_distance()        # 拍攝距離（所有張共用）
    except ValueError as e:
        print(f"已取消：{e}")
        return

    # ── 樹種選擇（所有張共用）────────────────────────────────
    species_mgr = SpeciesManager()
    try:
        species = species_mgr.select_species()         # 彈出樹種選擇視窗
    except ValueError as e:
        print(f"已取消：{e}")
        return

    carbon_calc = CarbonCalculator()

    total = len(image_paths)

    # ── 所有照片共用的物件，只建立一次 ──────────────────────────
    try:
        detector = TrunkDetector(config.MODEL_PATH, config.CONF_THRESHOLD)
    except Exception as e:
        print(f"模型載入失敗：{e}")
        return

    geometry   = GeometryEngine()
    checker    = ErrorChecker()
    validator  = Validator(config.SIMILARITY_THRESHOLD)
    visualizer = Visualizer()
    file_mgr   = FileManager(config.OUTPUT_DIR)
    qr_detector = None
    qr_calc     = None

    if config.USE_QR:
        try:
            from src.qr_detector import QRDetector
            from src.qr_calculator import QRCalculator
            qr_detector = QRDetector()
            qr_calc     = QRCalculator()
        except Exception as e:
            print(f"QR 功能載入失敗，將改用焦距法：{e}")
            qr_detector = None
            qr_calc     = None

    success_count = 0
    failed_count  = 0

    # ── 步驟 2：逐張處理 ─────────────────────────────────────────
    for idx, image_path in enumerate(image_paths, start=1):
        print(f"\n{'='*55}")
        print(f"處理第 {idx}/{total} 張：{image_path}")
        print(f"{'='*55}")

        # ── 讀取圖片（np.fromfile + imdecode 支援中文路徑）──────
        buf = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            print(f"  錯誤：無法讀取圖片，略過")
            failed_count += 1
            continue  # 跳到下一張，不中斷整個程式

        img_h, img_w = image.shape[:2]  # 取得圖片的高度和寬度（像素）

        # ── YOLO 偵測樹幹 ────────────────────────────────────────
        # 把圖片丟給 YOLO，取回樹幹的輪廓點（masks.xy）和信心度
        detection = detector.detect(image)
        if detection is None:
            print(f"  錯誤：YOLO 未偵測到樹幹，略過")
            failed_count += 1
            continue

        trunk_pts  = detection["masks_xy"]   # shape: (N, 2)，像素座標
        confidence = detection["confidence"] # 0.0 ~ 1.0

        # ── QR code 偵測 ─────────────────────────────────────────
        # 失敗時回傳 None，程式不會中斷，自動切換到備援方法
        if qr_detector is not None:
            qr_result = qr_detector.detect(image)  # 成功回傳像素寬度，失敗回傳 None
        else:
            qr_result = None
        # USE_QR=False 時不載入 QR 套件，自動走焦距法

        # ── 計算比例尺和 DBH ──────────────────────────────────────
        result_a = None  # 方法一結果（QR code 比例尺）

        if qr_detector is not None and qr_calc is not None and qr_detector.is_detected():
            # ── 方法一：QR code 比例尺 ────────────────────────────
            # ① 用 QRCalculator 將 QR code 像素寬度換算成比例尺（cm/px）
            scale_a      = qr_calc.compute_scale(qr_result)

            # ② 用樹幹輪廓點 + 比例尺，算出 1.3m 高的量測位置 y 座標
            target_y_a   = geometry.compute_target_y(trunk_pts, scale_a)

            # ③ 在 target_y 位置計算樹幹直徑
            geo_result_a = geometry.get_diameter_at_height(trunk_pts, target_y_a, scale_a)
            result_a     = geo_result_a["diameter_cm"]  # 取出直徑數值

        # ── 方法二：焦距公式備援 ──────────────────────────────────
        # 不管 QR code 有沒有成功，方法二都會跑，作為對照或備援
        # img_w 每張圖可能不同，FocalCalculator 需在迴圈內建立
        focal_calc = FocalCalculator(focal_mm, sensor_w, img_w)
        # 比例尺（cm/px）= 距離(cm) ÷ 焦距(px)
        # 注意：分子是距離，分母是焦距，不能寫反
        focal_px = focal_calc._to_focal_px()
        if focal_px <= 0:
            print(f"  錯誤：焦距計算失敗（請確認感光元件寬度輸入是否正確），略過")
            failed_count += 1
            continue
        scale_b = (distance_m * 100) / focal_px

        # ② 同樣用 compute_target_y 算出胸高 y 座標
        target_y_b = geometry.compute_target_y(trunk_pts, scale_b)

        # ③ 在 target_y 位置計算樹幹直徑
        geo_result_b = geometry.get_diameter_at_height(trunk_pts, target_y_b, scale_b)
        result_b     = geo_result_b["diameter_cm"]
        if result_b <= 0:
            print(f"  錯誤：胸徑計算失敗（1.3m 量測位置可能超出畫面或輪廓不足），略過")
            failed_count += 1
            continue

        # ── 誤差自動檢查 ──────────────────────────────────────────
        # 在輸出結果前，自動檢查可能影響精度的問題
        warnings = []
        if qr_detector is not None and qr_detector.is_detected():
            # （QR code 歪斜與大小的精確檢查需改由 qr_detector 回傳 polygon 再做）
            pass  # TODO：待 QRDetector 回傳 polygon 後再補實作
        warnings += checker.check_trunk_completeness(trunk_pts, image.shape)

        # ── 雙驗證，建立最終結果 ──────────────────────────────────
        # 差異 < 20% → 取平均，標記 "verified"
        # 差異 >= 20% → 輸出方法二，標記 "mismatch" 並警告
        result               = validator.validate(result_a, result_b)
        result.confidence    = confidence
        result.diameter_std  = geo_result_b["std_cm"]
        result.warnings      = result.warnings + warnings
        result.image_file    = image_path
        result.measurement_y = target_y_b   # 實際胸高 y 座標，傳給 visualizer 畫線

        # ── 固碳量計算 ─────────────────────────────────────────
        carbon_result        = carbon_calc.calculate(result.diameter_cm, species)
        result.species_name  = species["name"]
        result.biomass_kg    = carbon_result["biomass_kg"]
        result.carbon_kg     = carbon_result["carbon_kg"]
        result.co2_kg        = carbon_result["co2_kg"]

        # ── 在圖片上繪製結果 ──────────────────────────────────────
        # 畫出樹幹遮罩、胸高測量線、DBH 數值、狀態碼
        output_img = visualizer.draw(image, detection, result)

        # ── 存檔 ──────────────────────────────────────────────────
        # 儲存標注後的圖片，並把這筆測量結果新增到 CSV
        file_mgr.save_image(output_img, image_path)
        file_mgr.save_csv(result)
        success_count += 1

        # ── 單張摘要 ──────────────────────────────────────────────
        print(f"  樹徑：{result.diameter_cm:.1f} cm")
        print(f"  方法：{result.method}  狀態：{result.status}")
        print(f"  信心度：{result.confidence:.2f}")
        if result.warnings:
            for w in result.warnings:
                print(f"  警告：{w}")

    # ── 全部完成 ──────────────────────────────────────────────────
    print(f"\n全部 {total} 張處理完畢：成功 {success_count} 張，失敗 {failed_count} 張")
    print(f"結果已存至：{config.OUTPUT_DIR}")


# ── 程式進入點 ────────────────────────────────────────────────
# 只有直接執行 main.py 時才會跑 main()
# 如果是被其他模組 import，不會自動執行

if __name__ == "__main__":
    main()

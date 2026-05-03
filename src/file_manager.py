#輔負責
# =============================================================
# file_manager.py — 檔案存取管理器
#
# 負責所有「寫檔」的工作：
#   1. 建立輸出資料夾
#   2. 儲存標注後的圖片（原始檔名 + 時間戳記）
#   3. 把每次的量測結果新增到 measurements.csv（不覆寫舊資料）
#
# 設計原則：
#   輸出目錄由 config.OUTPUT_DIR 控制，不污染專案目錄
#   CSV 採用 append 模式，每次執行自動累積新資料
# =============================================================

import os       # 檔案路徑、資料夾操作
import csv      # 讀寫 CSV 格式
import datetime # 取得目前時間，用來產生時間戳記
import cv2      # 儲存影像
import numpy as np  # tofile 支援中文路徑儲存


class FileManager:
    """
    檔案存取管理器

    屬性：
        output_dir (str)：實際使用的輸出資料夾路徑（~ 已展開）
        csv_path   (str)：measurements.csv 的完整路徑
    """

    def __init__(self, output_dir: str):
        """
        初始化管理器，確保輸出資料夾存在。

        參數：
            output_dir (str)：輸出資料夾路徑，例如 "~/Desktop/results"
                              支援 ~ 符號（自動展開成完整路徑）
        """

        # os.path.expanduser() 把 "~" 展開成家目錄
        # 例如 "~/Desktop/results" → "C:/Users/user/Desktop/results"
        self.output_dir = os.path.expanduser(output_dir)

        # 若資料夾不存在，自動建立（exist_ok=True 表示已存在時不報錯）
        os.makedirs(self.output_dir, exist_ok=True)

        # CSV 固定命名為 measurements.csv，放在輸出資料夾內
        self.csv_path = os.path.join(self.output_dir, "measurements.csv")

        print(f"輸出資料夾：{self.output_dir}")

    def save_image(self, image, source_path: str) -> str:
        """
        儲存標注後的圖片，以「原始檔名_時間戳.png」命名。

        命名範例：
            原始檔名 photo.jpg → 輸出 photo_20260414_153012.png

        參數：
            image       (numpy.ndarray)：OpenCV 格式的影像（已畫上遮罩與標注）
            source_path (str)          ：原始圖片路徑，用來取出原始檔名

        回傳值：
            str：儲存後的完整圖片路徑
            None：若 image 為 None 則跳過並回傳 None
        """

        # 若影像為空，跳過儲存
        if image is None:
            print("[FileManager] 影像為空，略過圖片儲存")
            return None

        # 取出原始檔名（不含路徑），例如 "photo.jpg"
        base_name = os.path.basename(source_path)

        # 分離檔名與副檔名，例如 ("photo", ".jpg")
        name_only, _ = os.path.splitext(base_name)

        # 產生時間戳記，格式為 YYYYMMDD_HHMMSS
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 組合輸出檔名，固定存成 PNG（無損壓縮，保留細節）
        output_filename = f"{name_only}_{timestamp}.png"
        output_path     = os.path.join(self.output_dir, output_filename)

        # 用 imencode + tofile 支援中文路徑（cv2.imwrite 在 Windows 中文路徑會失敗）
        success, buf = cv2.imencode(".png", image)
        if success:
            np.array(buf).tofile(output_path)
        else:
            print(f"[FileManager] 影像編碼失敗，略過圖片儲存")

        print(f"標注圖片已儲存：{output_path}")
        return output_path

    def save_csv(self, result) -> str:
        """
        將一筆 MeasurementResult 新增到 measurements.csv。

        CSV 採用 append 模式，每次執行只新增一列，不覆寫舊資料。
        若 CSV 不存在，自動建立並寫入標題列（欄位名稱）。

        參數：
            result (MeasurementResult)：量測結果物件（來自 src/models.py）

        回傳值：
            str：CSV 檔案的完整路徑
        """

        # 定義 CSV 的欄位順序，與 MeasurementResult 的屬性對應
        fieldnames = [
            "timestamp",     # 量測時間
            "image_file",    # 來源圖片檔名
            "diameter_cm",   # 最終直徑（公分）
            "method",        # 使用的方法（dual / qr_only / focal_only）
            "status",        # 狀態（verified / mismatch / qr_failed）
            "result_a",      # 方法一結果（QR code 比例尺）
            "result_b",      # 方法二結果（焦距公式）
            "confidence",    # YOLO 信心度
            "diameter_std",  # 直徑標準差
            "warnings",      # 警告訊息
            "species_name",  # 樹種名稱
            "biomass_kg",    # 生質量（公斤）
            "carbon_kg",     # 碳儲量（公斤）
            "co2_kg",        # CO2 固定當量（公斤）
        ]

        # 判斷 CSV 是否已存在（決定要不要寫標題列）
        file_exists = os.path.isfile(self.csv_path)

        # "a" 模式 = append（追加），不覆蓋舊資料
        # newline="" 是 csv 模組要求的設定，避免多餘空行
        # encoding="utf-8-sig" 讓 Windows Excel 能正確顯示中文
        with open(self.csv_path, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # 第一次建立時才寫標題列
            if not file_exists:
                writer.writeheader()

            # 把 result 物件的每個欄位寫成 CSV 的一列
            writer.writerow({
                "timestamp":    result.timestamp,
                "image_file":   result.image_file,
                "diameter_cm":  result.diameter_cm,
                "method":       result.method,
                "status":       result.status,
                "result_a":     result.result_a if result.result_a is not None else "",
                "result_b":     result.result_b,
                "confidence":   result.confidence,
                "diameter_std": result.diameter_std,
                # warnings 是 list，轉成用分號分隔的字串存入 CSV
                "warnings":     "; ".join(result.warnings),
                "species_name": result.species_name,
                "biomass_kg":   result.biomass_kg,
                "carbon_kg":    result.carbon_kg,
                "co2_kg":       result.co2_kg,
            })

        print(f"量測結果已寫入 CSV：{self.csv_path}")
        return self.csv_path

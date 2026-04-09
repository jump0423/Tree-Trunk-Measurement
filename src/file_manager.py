#輔負責
import os       # 用來操作檔案路徑、建立資料夾
import csv      # 用來讀寫 CSV 格式的結果檔
import cv2      # 用來儲存帶有標注的影像
import config   # 讀取全域設定（如 OUTPUT_DIR）


# 支援的圖片副檔名清單
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


# ════════════════════════════════════════════════════════════════
#  檔案管理模組
#
#  本模組負責所有與「檔案讀寫」相關的操作：
#    - 取得要處理的圖片路徑清單
#    - 建立輸出資料夾
#    - 把量測結果存成 CSV
#    - 把標注後的影像存成圖片
# ════════════════════════════════════════════════════════════════


def get_image_paths(image=None, folder=None):
    """
    根據使用者的輸入，回傳所有要處理的圖片路徑清單。

    兩種模式：
        - 單張模式：image 參數有值，直接回傳 [image]
        - 批次模式：folder 參數有值，掃描資料夾內所有支援格式的圖片

    參數：
        image  (str 或 None)：單張圖片的路徑
        folder (str 或 None)：含有多張圖片的資料夾路徑

    回傳值：
        list[str]：圖片路徑的清單；若找不到任何圖片則回傳空 list
    """

    # ── 單張模式 ──────────────────────────────────────────────────
    if image is not None:
        # 直接把單張路徑包成 list 回傳
        return [image]

    # ── 批次模式 ──────────────────────────────────────────────────
    if folder is not None:
        image_paths = []  # 準備一個空 list 來收集結果

        # os.listdir() 列出資料夾內所有檔名（不含子資料夾）
        for filename in os.listdir(folder):
            # 取得副檔名，例如 "photo.JPG" → ".jpg"
            _, ext = os.path.splitext(filename)

            # 只收集支援的圖片格式（不分大小寫）
            if ext.lower() in SUPPORTED_EXTENSIONS:
                # 組合成完整的路徑，例如 "./images/photo.jpg"
                full_path = os.path.join(folder, filename)
                image_paths.append(full_path)

        # 依照檔名排序，讓處理順序固定（方便對照結果）
        image_paths.sort()

        if len(image_paths) == 0:
            print(f"[警告] 資料夾 {folder} 內沒有找到任何支援的圖片")

        return image_paths

    # 兩個參數都沒給（理論上 validate_inputs 會先攔截，這裡做雙重保險）
    return []


def prepare_output_dir(output_path=None):
    """
    確保輸出資料夾存在，不存在就自動建立。

    參數：
        output_path (str 或 None)：使用者指定的輸出路徑；
                                    若為 None，則使用 config.OUTPUT_DIR

    回傳值：
        str：實際使用的輸出資料夾路徑（已展開 ~ 符號）
    """

    # 若使用者沒有指定，就用設定檔的預設路徑
    if output_path is None:
        output_path = config.OUTPUT_DIR

    # os.path.expanduser() 把 "~" 展開成實際的家目錄路徑
    # 例如 "~/Desktop/results" → "C:/Users/user/Desktop/results"
    output_path = os.path.expanduser(output_path)

    # 若資料夾不存在，就建立它（exist_ok=True 表示已存在時不報錯）
    os.makedirs(output_path, exist_ok=True)

    print(f"輸出資料夾：{output_path}")
    return output_path


def save_result_csv(result, output_dir):
    """
    將單筆量測結果追加寫入 CSV 檔案。

    CSV 的每一行代表一張圖片的量測結果。
    若 CSV 不存在，會自動建立並寫入標題列（欄位名稱）。

    參數：
        result     (MeasurementResult)：量測結果物件（來自 src/models.py）
        output_dir (str)              ：輸出資料夾路徑

    回傳值：
        str：CSV 檔案的完整路徑
    """

    # CSV 固定放在輸出資料夾內，檔名為 results.csv
    csv_path = os.path.join(output_dir, "results.csv")

    # 定義 CSV 的欄位順序（與 MeasurementResult 的屬性對應）
    fieldnames = [
        "timestamp",     # 量測時間
        "image_file",    # 圖片檔名
        "diameter_cm",   # 最終直徑（公分）
        "method",        # 使用的方法（method_a / method_b / average）
        "status",        # 狀態（ok / warning / error）
        "result_a",      # 方法一結果
        "result_b",      # 方法二結果
        "confidence",    # YOLO 信心度
        "diameter_std",  # 直徑標準差
        "warnings",      # 警告訊息
    ]

    # 判斷 CSV 是否已經存在（決定要不要寫標題列）
    file_exists = os.path.isfile(csv_path)

    # "a" 模式 = append（追加），不會覆蓋舊資料
    # newline="" 是 Python csv 模組的必要設定，避免多餘換行
    with open(csv_path, mode="a", newline="", encoding="utf-8-sig") as f:
        # utf-8-sig 讓 Windows 的 Excel 能正確顯示中文

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # 只有第一次建立 CSV 時才寫標題列
        if not file_exists:
            writer.writeheader()

        # 把 result 物件的欄位逐一對應寫入 CSV 的一行
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
            "warnings":     "; ".join(result.warnings),  # list 轉成用分號分隔的字串
        })

    return csv_path


def save_annotated_image(image, filename, output_dir):
    """
    將帶有標注框的影像儲存到輸出資料夾。

    檔名格式：原始檔名加上 "_result" 後綴，例如：
        photo.jpg → photo_result.jpg

    參數：
        image      (numpy.ndarray)：OpenCV 讀取的影像（已畫上標注）
        filename   (str)          ：原始圖片的檔名或路徑（用來產生輸出檔名）
        output_dir (str)          ：輸出資料夾路徑

    回傳值：
        str：儲存後的圖片完整路徑
        None：若 image 為 None 則回傳 None
    """

    # 若沒有影像資料，就跳過儲存
    if image is None:
        print("[警告] 影像為空，略過儲存")
        return None

    # 從完整路徑中只取出檔名，例如 "./images/photo.jpg" → "photo.jpg"
    base_name = os.path.basename(filename)

    # 分離檔名與副檔名，例如 "photo.jpg" → ("photo", ".jpg")
    name_without_ext, ext = os.path.splitext(base_name)

    # 組合出新檔名，例如 "photo_result.jpg"
    output_filename = name_without_ext + "_result" + ext

    # 組合出完整的儲存路徑
    output_path = os.path.join(output_dir, output_filename)

    # 用 OpenCV 儲存影像
    cv2.imwrite(output_path, image)

    print(f"標注影像已儲存：{output_path}")
    return output_path

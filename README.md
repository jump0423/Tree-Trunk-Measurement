## 3.1　資料夾結構

| 路徑 | 說明 |
| :--- | :--- |
| main.py | 唯一程式入口，串接所有模組，不含業務邏輯 |
| config.py | 所有參數集中管理，修改數值只需動此一個檔案 |
| best.pt | YOLOv11n-seg 訓練完成的模型權重 |
| requirements.txt | 依賴套件清單（含版本號） |
| .gitignore | 排除輸出資料夾、__pycache__ 等 |
| src/__init__.py | 將 src 標示為 Python 套件 |
| src/models.py | MeasurementResult dataclass，模組間傳遞的資料結構 |
| src/trunk_detector.py | TrunkDetector：YOLO 偵測樹幹 |
| src/qr_detector.py | QRDetector：pyzbar 偵測 QR code |
| src/geometry.py | GeometryEngine：20 切片 + IQR 計算樹徑 |
| src/qr_calculator.py | QRCalculator：方法一計算邏輯 |
| src/focal_calculator.py | FocalCalculator：方法二計算邏輯 |
| src/validator.py | Validator：雙驗證判斷，建立 MeasurementResult |
| src/input_handler.py | InputHandler：所有使用者輸入介面 |
| src/visualizer.py | Visualizer：圖片標注繪製 |
| src/file_manager.py | FileManager：存檔與 CSV 寫入 |
| src/error_checker.py | ErrorChecker：選配，目前留空，核心穩定後再實作 |
| measured_result/ | 自動建立，存放標注圖片與 measurements.csv |
| training_files/ | Colab 訓練腳本、data.yaml、訓練記錄 |

---

## 4.2　各模組職責說明

### InputHandler — 使用者輸入
（唯一負責與使用者溝通的模組）
所有輸入驗證集中於此，其他模組完全不處理輸入錯誤。未來若改成 GUI 或讀取設定檔，只需改此模組。

| 方法 | 職責 |
| :--- | :--- |
| get_image_path() | tkinter 彈出視窗，讓使用者選擇圖片，回傳檔案路徑 |
| get_camera_params() | 請使用者輸入焦距（mm）與感光元件寬度（mm） |
| get_distance() | 請使用者輸入距離，超過 MAX_DISTANCE_M 顯示警告 |
| _validate_distance(d) | 距離合法性檢查，回傳布林值（內部使用） |

### TrunkDetector — YOLO 偵測樹幹
封裝 YOLOv11 模型。detect() 把圖片丟給 best.pt，取回樹幹的輪廓座標點列表（masks.xy）。

| 方法 | 職責 |
| :--- | :--- |
| __init__(model_path, conf) | 載入 best.pt 模型，設定信心度閾值，只執行一次 |
| detect(image) | 執行 YOLO 推論，回傳 masks.xy（輪廓座標）與信心度 |
| get_trunk_pixel_width(mask) | 從 masks.xy 計算樹幹最寬處的像素距離 |
| is_trunk_complete(box, img_shape) | 檢查偵測框是否碰到影像邊緣，是則回傳 False 並警告 |

### QRDetector — pyzbar 偵測 QR code
封裝 pyzbar 套件，完全獨立於 YOLO，不需要任何訓練資料。失敗時回傳 None 而非拋出例外，讓 main.py 優雅切換備援。

| 方法 | 職責 |
| :--- | :--- |
| detect(image) | 掃描整張圖片找 QR code，成功回傳像素寬度，失敗回傳 None |
| is_detected() | 回傳布林值，main.py 用此判斷走方法一還是直接走備援 |

### GeometryEngine — 核心幾何計算
整個系統精度最關鍵的模組。採用 20 切片 + IQR 過濾，比單一像素測量更穩定。
target_y = img_height ÷ 2，因為相機水平架設在 1.3m，畫面垂直中點就是 1.3m 的位置。

| 方法 | 職責 |
| :--- | :--- |
| get_diameter_at_height(trunk_pts, target_y, scale) | 在 target_y 附近取 20 條水平切片，IQR 過濾異常值後取均值，回傳 {diameter_cm, std_cm, confidence} |

### QRCalculator / FocalCalculator — 兩種計算方法

| 模組 | 方法 | 公式 |
| :--- | :--- | :--- |
| QRCalculator | calculate(trunk_px, qr_px) | trunk_px × (5.0 ÷ qr_px) |
| FocalCalculator | calculate(trunk_px, distance_m) | trunk_px × distance_cm ÷ focal_px |
| FocalCalculator | _to_focal_px() | (焦距_mm ÷ 感光元件寬_mm) × 影像像素寬 |

### Validator — 雙驗證，建立 MeasurementResult
核心驗證邏輯。比較兩方法差異率，判斷輸出狀態，建立並填入 MeasurementResult 所有欄位後傳出。

| 方法 | 職責 |
| :--- | :--- |
| __init__(threshold) | 設定相似度門檻，預設 0.80（80%） |
| validate(result_a, result_b) | 比較差異率，建立並回傳完整的 MeasurementResult 物件 |
| _similarity(a, b) | 計算差異率 = |a-b| ÷ max(a,b)（內部使用） |

### Visualizer — 圖片標注繪製

| 方法 | 職責 |
| :--- | :--- |
| draw(image, detection, result) | 統一入口，依序呼叫以下三個內部方法 |
| _draw_mask() | 在圖片上繪製半透明樹幹分割遮罩（YOLO 的 masks.xy） |
| _draw_diameter_line() | 在畫面垂直中點（1.3m）繪製紅色水平測量線 |
| _draw_status_label() | 右上角顯示樹徑數值、狀態碼、YOLO confidence |

### FileManager — 存檔管理

| 方法 | 職責 |
| :--- | :--- |
| __init__(output_dir) | 讀取 OUTPUT_DIR，若資料夾不存在則自動建立 |
| save_image(image, source_path) | 儲存標注後圖片，以「原始檔名_時間戳.png」命名 |
| save_csv(result) | 從 MeasurementResult 取出所有欄位，新增一列至 measurements.csv |

---

## 專案分工表

| 模組路徑 | 負責人 | 說明 |
| :--- | :--- | :--- |
| src/trunk_detector.py | **Morris** | Morris 負責，但框架由 jump 先建好 |
| src/qr_detector.py | **Morris** | Morris 負責 |
| src/geometry.py | **jump** | jump 負責（核心模組） |
| src/qr_calculator.py | **Morris** | Morris 負責 |
| src/focal_calculator.py | **Morris** | Morris 負責 |
| src/validator.py | **共用** | 共用項目 |
| src/input_handler.py | **Morris** | Morris 負責 |
| src/visualizer.py | **共用** | 共用項目 |
| src/file_manager.py | **Morris** | Morris 負責 |
| src/error_checker.py | **jump** | jump 負責 |
| main.py | **共用** | 共用項目 |

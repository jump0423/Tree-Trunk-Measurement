


Tree-Trunk-Measurement
專案架構詳細說明

共 19 個路徑　·　10 個類別　·　含所有方法說明
標籤：朋友 = 沿用原有  /  我新增 = 本次實作  /  共用調整 = 兩人協作

根目錄　— 設定、模型、環境
main.py    — 類別 main()
唯一程式入口。依序呼叫各模組串接完整流程，本身不含任何業務邏輯。
  共用調整  
| 方法 | 說明 |
| --- | --- |
| main() | 啟動整個測量流程：InputHandler → TrunkDetector → QRDetector → QRCalculator / FocalCalculator → Validator → Visualizer → FileManager |

config.py    — 類別 （無類別，全域常數）
所有可調整參數集中管理。未來修改任何數值只需動此一個檔案，不需要進入各模組。
  共用調整  
| 參數名稱 | 預設值 | 說明 |
| --- | --- | --- |
| QR_REAL_SIZE_CM | 5.0 | QR code 真實尺寸（公分） |
| MAX_DISTANCE_M | 7.0 | 拍攝距離上限警告閾值（公尺） |
| CONF_THRESHOLD | 0.65 | YOLO 信心度閾值，低於此值的框不採用 |
| DBH_HEIGHT_M | 1.3 | 胸徑測量高度（公尺），從 mask 最底部往上計算，預設符合林業標準 1.3m |
| DBH_SLICE_COUNT | 20 | geometry.py 計算樹徑時的水平切片數量 |
| MIN_VALID_SLICES | 8 | 最少有效切片數，低於此值視為測量失敗 |
| SIMILARITY_THRESHOLD | 0.80 | 雙驗證相似度門檻（80%），超過則輸出警告 |
| MODEL_PATH | "best.pt" | YOLO 模型權重檔路徑 |
| OUTPUT_DIR | "~/Desktop/results" | 結果輸出目錄（預設桌面，不污染專案） |

best.pt    — 類別 （非 Python 檔）
YOLOv11n-seg 訓練完成的模型權重檔。只偵測 trunk 單一類別。由朋友訓練後放置於根目錄，trunk_detector.py 載入使用。
  朋友  

requirements.txt    — 類別 （非 Python 檔）
依賴套件清單，鎖定版本號確保不同環境行為一致。
  共用調整  
| 方法 | 說明 |
| --- | --- |
| ultralytics >= 8.0.0 | YOLOv11 核心套件 |
| opencv-python >= 4.8.0 | 影像處理（偵測、繪圖） |
| opencv-contrib-python | OpenCV 額外模組 |
| numpy >= 1.24.0 | 數值計算 |
| pyzbar >= 0.1.9 | QR code 偵測與解碼 |
| scipy | IQR 統計計算（geometry.py 使用） |

.gitignore    — 類別 （非 Python 檔）
排除不需要進版控的檔案與資料夾，保持 repo 整潔。
  朋友  
| 方法 | 說明 |
| --- | --- |
| measured_result/ | 輸出結果資料夾，每次執行自動產生 |
| __pycache__/ | Python 快取，自動產生 |
| *.pt | 模型權重檔（體積大，建議用 Git LFS 或排除） |
| *.pyc | Python 編譯快取 |


src/　— 核心套件
src/__init__.py    — 類別 （空檔案）
將 src 資料夾標示為 Python 套件，使 from src.trunk_detector import TrunkDetector 語法可正常運作。
  朋友  

src/models.py    — 類別 MeasurementResult
定義所有模組間傳遞的資料結構（Python dataclass）。確保各模組溝通格式一致，不傳遞散亂的 dict 或 tuple。
  共用調整  
| 欄位名稱 | 型別 | 說明 |
| --- | --- | --- |
| diameter_cm | float | 最終輸出樹徑（公分） |
| method | str | "dual" / "qr_only" / "focal_only" |
| status | str | "verified" / "mismatch" / "qr_failed" |
| result_a | float | None | 方法一結果（QR code 失靈時為 None） |
| result_b | float | 方法二結果（焦距公式） |
| confidence | float | YOLO 偵測信心度分數 |
| diameter_std | float | 切片標準差，反映測量穩定性（geometry.py 提供） |
| warnings | list[str] | 警告訊息列表（error_checker 使用，預設空列表） |
| timestamp | str | 測量日期與時間 |
| image_file | str | 來源圖片檔名 |


偵測層　— 負責「找到東西在哪裡」，不做任何計算
src/trunk_detector.py    — 類別 TrunkDetector
封裝 YOLOv11 模型，偵測樹幹並回傳像素寬度，包含邊緣完整性檢查。
  朋友  
| 方法 | 說明 |
| --- | --- |
| __init__(model_path, conf) | 載入 YOLO 模型，設定信心度閾值（從 config.py 傳入） |
| detect(image) | 執行推論，回傳 masks.xy 與信心度分數 |
| get_trunk_pixel_width(mask) | 從分割遮罩（mask）計算樹幹像素寬度 |
| is_trunk_complete(box, img_shape) | 檢查偵測框是否碰到影像邊緣，判斷樹幹是否完整入鏡 |

src/qr_detector.py    — 類別 QRDetector
封裝 pyzbar，偵測影像中的 QR code 並回傳像素寬度。失敗時回傳 None（不拋出例外），讓主流程可以優雅地切換至備援方法。
  朋友  
| 方法 | 說明 |
| --- | --- |
| detect(image) | 回傳 QR code 像素寬度。偵測失敗時回傳 None，不中斷程式 |
| is_detected() | 回傳布林值，供 main.py 判斷是否執行方法一（QR 比例尺） |


計算層　— 負責「算出數值」，不做偵測
src/geometry.py    — 類別 GeometryEngine
集中所有幾何運算邏輯，與 YOLO 完全解耦，可獨立進行單元測試。採用 20 切片 + IQR 過濾異常值的方式計算樹徑，比單一像素寬測量精準。
  我新增  
| 方法 | 說明 |
| --- | --- |
| get_diameter_at_height(trunk_pts, target_y, scale) | 從 mask 最底部往上 1.3m 對應的 y 像素位置（target_y = mask 最低點 y − DBH_HEIGHT_M 對應像素數）取 20 條水平切片，用 IQR 過濾異常值後取均值，回傳 {diameter_cm, std_cm, confidence} |

src/qr_calculator.py    — 類別 QRCalculator
方法一的計算邏輯。以 QR code 作為比例尺換算樹幹直徑，與偵測模組（qr_detector.py）分開，讓計算邏輯可獨立測試。
  朋友  
| 方法 | 說明 |
| --- | --- |
| __init__(qr_real_size_cm) | 設定 QR code 真實尺寸（由 config.QR_REAL_SIZE_CM 傳入） |
| calculate(trunk_px, qr_px) | 回傳樹徑（cm）：trunk_px × (QR_REAL_SIZE_CM ÷ qr_px) |

src/focal_calculator.py    — 類別 FocalCalculator
方法二的計算邏輯。根據針孔相機模型，利用使用者輸入的焦距與距離換算樹幹直徑，作為 QR code 失靈時的備援。
  朋友  
| 方法 | 說明 |
| --- | --- |
| __init__(focal_mm, sensor_width_mm, img_width_px) | 初始化相機參數（使用者輸入） |
| calculate(trunk_px, distance_m) | 回傳樹徑（cm）：trunk_px × distance_cm ÷ focal_px |
| _to_focal_px() | 將 mm 焦距轉換為像素焦距（內部使用） |


驗證層　— 比較兩種方法的結果，判斷輸出狀態
src/validator.py    — 類別 Validator
核心驗證邏輯。比較方法一與方法二的結果差異率，輸出最終狀態與 MeasurementResult 物件。與計算層和輸出層完全解耦。
  共用調整  
| 方法 | 說明 |
| --- | --- |
| __init__(threshold) | 設定相似度門檻（預設 0.80，由 config.py 傳入） |
| validate(result_a, result_b) | QR code 成功時比較兩方法差異率，回傳 MeasurementResult；QR code 失靈時直接以方法二輸出 |
| _similarity(a, b) | 計算差異率 = |a-b| ÷ max(a,b)（內部使用） |


Validator 輸出狀態碼說明
| 狀態碼 | 觸發條件 | 輸出值 |
| --- | --- | --- |
| "verified" | QR code 成功，差異率 < 20% | (result_a + result_b) ÷ 2 |
| "mismatch" | QR code 成功，差異率 ≥ 20% | result_b，標註警告 |
| "qr_failed" | QR code 偵測失敗 | result_b，標註 QR code 失敗 |


輸入 / 輸出層　— 處理使用者互動、圖片繪製、檔案存取
src/input_handler.py    — 類別 InputHandler
負責所有使用者輸入介面，包含圖片選擇視窗、數值輸入與驗證。所有輸入驗證集中於此，其他模組不處理輸入錯誤。
  朋友  
| 方法 | 說明 |
| --- | --- |
| get_image_path() | 彈出 tkinter 視窗供使用者選擇圖片（支援 .jpg、.png 等格式） |
| get_camera_params() | 詢問焦距（mm）與感光元件寬度（mm），用於方法二計算 |
| get_distance() | 詢問拍攝距離，若超過 MAX_DISTANCE_M 顯示警告但不強制阻擋 |
| _validate_distance(d) | 距離合法性檢查，回傳布林值（內部使用） |

src/visualizer.py    — 類別 Visualizer
負責在圖片上繪製偵測結果。以 draw() 作為統一入口，內部分工呼叫三個私有方法。可直接從舊專案修改延伸，擴充顯示 confidence 與 warnings。
  共用調整  
| 方法 | 說明 |
| --- | --- |
| draw(image, detection, result) | 統一入口，依序呼叫以下三個內部方法完成完整標注 |
| _draw_mask(image, mask) | 繪製 YOLO Instance Segmentation 分割遮罩（半透明覆蓋） |
| _draw_diameter_line(image, box) | 從 mask 最底部往上 1.3m 對應位置繪製紅色水平測量線 |
| _draw_status_label(image, result) | 在圖片右上角標註狀態文字、樹徑數值、confidence 分數 |

src/file_manager.py    — 類別 FileManager
負責所有檔案存取。輸出目錄由 config.OUTPUT_DIR 控制，結果儲存於使用者指定路徑，不污染專案目錄。CSV 紀錄每次測量自動累積，不覆寫。
  朋友  
| 方法 | 說明 |
| --- | --- |
| __init__(output_dir) | 讀取 OUTPUT_DIR，若資料夾不存在則自動建立 |
| save_image(image, source_path) | 儲存標注後圖片，以「原始檔名_時間戳.png」命名 |
| save_csv(result) | 將 MeasurementResult 物件新增一列至 measurements.csv（含所有欄位） |


選配模組　— 架構保留，核心流程穩定後再決定是否實作
src/error_checker.py    — 類別 ErrorChecker
在輸出前自動偵測潛在問題，偵測到問題後寫入 MeasurementResult.warnings 列表，不直接阻擋輸出。目前建立空類別，等核心流程穩定後再補實作。
  我新增  
| 方法 | 說明 |
| --- | --- |
| check_marker_tilt(contour) | 偵測 QR code 歪斜角度是否超過 15°，超過則寫入 warnings |
| check_marker_pixel_size(contour) | 偵測 QR code 在畫面中的像素大小是否低於 MIN_MARKER_PX，過小則換算不準確 |
| check_trunk_completeness(mask, img_shape) | 偵測樹幹分割遮罩是否被影像邊緣裁切，是則寫入 warnings |
目前程式碼內容（暫時留空）：
class ErrorChecker:     pass  # TODO: 核心流程穩定後實作


自動生成目錄　— 由程式產生，不進版控
measured_result/    — 類別 （自動建立的資料夾）
由 FileManager.__init__() 在首次執行時自動建立，存放所有標注後圖片與 measurements.csv。已加入 .gitignore，不進版控。
  朋友  
| 方法 | 說明 |
| --- | --- |
| *.png | 標注後圖片，以「原始檔名_時間戳.png」命名 |
| measurements.csv | 所有測量紀錄，每次執行自動累積新增一列，不覆寫 |

training_files/    — 類別 （資料夾）
存放 Colab 訓練腳本、data.yaml 設定檔與訓練記錄。訓練完成後將 best.pt 放回根目錄即可。
  朋友  
| 方法 | 說明 |
| --- | --- |
| train.ipynb | Google Colab 訓練腳本（含資料下載、訓練、評估步驟） |
| data.yaml | Roboflow 匯出的資料集設定檔（類別名稱、路徑等） |
| results/ | 訓練過程的 loss 曲線、混淆矩陣等評估圖表（自動產生） |


附錄　所有路徑快速索引

| 路徑 | 類別名稱 | 歸屬 | 所屬層次 |
| --- | --- | --- | --- |
| main.py | main() | 共用調整 | 入口層 |
| config.py | （全域常數） | 共用調整 | 設定層 |
| best.pt | （模型權重檔） | 朋友 | 根目錄 |
| requirements.txt | （設定檔） | 共用調整 | 根目錄 |
| .gitignore | （設定檔） | 朋友 | 根目錄 |
| src/__init__.py | （空檔案） | 朋友 | src 套件 |
| src/models.py | MeasurementResult | 共用調整 | 資料模型層 |
| src/trunk_detector.py | TrunkDetector | 朋友 | 偵測層 |
| src/qr_detector.py | QRDetector | 朋友 | 偵測層 |
| src/geometry.py | GeometryEngine | 我新增 | 計算層 |
| src/qr_calculator.py | QRCalculator | 朋友 | 計算層 |
| src/focal_calculator.py | FocalCalculator | 朋友 | 計算層 |
| src/validator.py | Validator | 共用調整 | 驗證層 |
| src/input_handler.py | InputHandler | 朋友 | 輸入輸出層 |
| src/visualizer.py | Visualizer | 共用調整 | 輸入輸出層 |
| src/file_manager.py | FileManager | 朋友 | 輸入輸出層 |
| src/error_checker.py | ErrorChecker | 我新增 | 選配 |
| measured_result/ | （自動產生） | 朋友 | 輸出目錄 |
| training_files/ | （資料夾） | 朋友 | 訓練檔案 |

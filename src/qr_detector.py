#昌負責
# =============================================================
# qr_detector.py — QR code 偵測模組
# 負責人：昌
#
# 這個模組的唯一職責：
#   在影像中找到 QR code，並回傳它的「像素寬度」
#
# 它只做「找到在哪裡」，不做任何計算
# 計算的工作交給 qr_calculator.py
#
# 為什麼需要 QR code？
#   相機拍照時，同一棵樹站近或站遠，像素寬度完全不同
#   但如果畫面裡有一個已知真實大小的 QR code（例如 5cm × 5cm）
#   就可以反推「1 像素 = 幾公分」，進而算出樹幹真實直徑
#
# 失敗策略：
#   偵測失敗時「不拋出例外」，只回傳 None
#   讓 main.py 可以優雅地切換到備援方法（焦距公式）
#   這樣整個程式不會因為 QR code 被遮住或偵測不到而崩潰
#
# 使用的套件：pyzbar（需安裝：pip install pyzbar）
#   pyzbar 是一個專門解碼條碼和 QR code 的 Python 套件
#   它會回傳 QR code 的四個角點座標，我們用這個算像素寬度
# =============================================================
 
from pyzbar import pyzbar   # QR code / 條碼偵測套件
import numpy as np          # 數值計算，用來處理角點座標
import config               # 全域設定（MIN_MARKER_PX）
 
 
class QRDetector:
    """
    QR code 偵測器
 
    封裝 pyzbar 的偵測邏輯，提供乾淨的介面給 main.py 使用
    外部只需呼叫 detect() 和 is_detected()，不需要知道 pyzbar 內部細節
 
    屬性：
        _last_pixel_width (float | None)：
            上次 detect() 偵測到的 QR code 像素寬度
            偵測失敗時為 None
            供 is_detected() 查詢使用
    """
 
    def __init__(self):
        """
        初始化偵測器
 
        _last_pixel_width 初始為 None，代表尚未執行過偵測
        每次呼叫 detect() 都會更新這個值
        """
        self._last_pixel_width: float | None = None
 
    # ----------------------------------------------------------
    # 公開方法
    # ----------------------------------------------------------
 
    def detect(self, image: np.ndarray) -> float | None:
        """
        在影像中偵測 QR code，回傳其像素寬度
 
        參數說明：
            image : np.ndarray
                OpenCV 讀入的影像，shape 為 (H, W, 3)，BGR 格式
                由 main.py 用 cv2.imread() 讀取後傳入
 
        回傳值：
            float | None
                成功：回傳 QR code 的像素寬度（float）
                      例如 QR code 在畫面裡寬 150px → 回傳 150.0
                失敗：回傳 None（不拋出例外）
                      失敗原因可能是：
                        - 畫面中沒有 QR code
                        - QR code 被遮擋或模糊
                        - QR code 尺寸太小（< MIN_MARKER_PX）
                        - pyzbar 解碼失敗
 
        內部流程：
            1. 呼叫 pyzbar.decode() 掃描影像中所有 QR code
            2. 取第一個偵測到的 QR code（通常畫面裡只有一個）
            3. 從它的四個角點座標計算像素寬度
            4. 檢查寬度是否超過最小門檻（MIN_MARKER_PX）
            5. 更新 _last_pixel_width 並回傳
        """
 
        # ── 重置上次結果 ──────────────────────────────────────
        # 每次偵測前先清空，確保 is_detected() 反映的是「這次」的結果
        self._last_pixel_width = None
 
        # ── 第一步：用 pyzbar 掃描影像 ───────────────────────
        #
        # pyzbar.decode() 會掃描影像中所有條碼和 QR code
        # 回傳一個 list，每個元素代表一個偵測到的符號
        # 每個元素包含：
        #   .type     : 類型（"QRCODE", "EAN13" 等）
        #   .data     : 解碼後的內容（bytes）
        #   .polygon  : 四個角點座標（list of Point）
        #   .rect     : 邊界矩形（left, top, width, height）
        #
        # 如果畫面中沒有 QR code，回傳空 list []
        decoded_objects = pyzbar.decode(image)
 
        # ── 第二步：確認有偵測到 QR code ─────────────────────
        #
        # decoded_objects 為空代表畫面中沒有可識別的 QR code
        # 直接回傳 None，讓 main.py 切換備援方法
        if not decoded_objects:
            return None
 
        # ── 第三步：取第一個 QR code ─────────────────────────
        #
        # 實際使用場景中，畫面裡應該只有一個 QR code
        # 如果有多個（例如背景中有其他 QR code），取第一個
        # pyzbar 會優先回傳畫面中最清晰的那個
        qr = decoded_objects[0]
 
        # ── 第四步：從四個角點計算像素寬度 ───────────────────
        #
        # pyzbar 的 polygon 是四個角點的 (x, y) 座標
        # 我們只需要「水平寬度」，方法：
        #   取所有角點的 x 座標最大值 - 最小值
        #
        # 為什麼用 x 範圍而不用 rect.width？
        #   rect.width 是軸對齊邊界框（AABB），QR code 若稍微傾斜
        #   rect.width 會比實際 QR code 大，用角點更準確
        #
        # 例如四個角點 x = [100, 250, 248, 102]
        #   pixel_width = 250 - 100 = 150px
        polygon = qr.polygon
 
        # 如果 polygon 不足 4 個點，改用 rect 的寬度（備援）
        if len(polygon) >= 2:
            xs = [pt.x for pt in polygon]       # 取出所有角點的 x 座標
            pixel_width = float(max(xs) - min(xs))  # 最右 - 最左 = 水平寬度
        else:
            # 角點數量異常，改用邊界矩形的寬度
            pixel_width = float(qr.rect.width)
 
        # ── 第五步：檢查 QR code 是否夠大 ────────────────────
        #
        # config.MIN_MARKER_PX = 30（像素）
        # 小於 30px 代表拍攝距離太遠，QR code 解析度不足
        # 這時算出的比例尺誤差太大，視為偵測失敗
        if pixel_width < config.MIN_MARKER_PX:
            # 不更新 _last_pixel_width，保持 None
            return None
 
        # ── 成功：儲存結果並回傳 ─────────────────────────────
        self._last_pixel_width = pixel_width
        return pixel_width
 
    def is_detected(self) -> bool:
        """
        回傳上次 detect() 是否成功偵測到有效的 QR code
 
        回傳值：
            bool
                True  → 上次 detect() 成功，_last_pixel_width 有值
                False → 上次 detect() 失敗，_last_pixel_width 為 None
 
        使用情境（main.py 範例）：
            qr_detector.detect(image)
            if qr_detector.is_detected():
                # 走方法一：QR code 比例尺
                scale = qr_calc.compute_scale(qr_detector._last_pixel_width)
            else:
                # 走方法二：焦距公式備援
                diameter = focal_calc.calculate(...)
 
        注意：
            必須先呼叫 detect()，is_detected() 才有意義
            如果從來沒呼叫過 detect()，is_detected() 回傳 False
        """
        return self._last_pixel_width is not None
 
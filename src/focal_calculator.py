#輔負責
import config  # 讀取全域設定（如 QR code 真實尺寸）


# ════════════════════════════════════════════════════════════════
#  焦距計算模組
#
#  本模組負責「方法二」：利用相機焦距公式換算樹幹的真實直徑。
#
#  核心公式（相似三角形原理）：
#
#      真實尺寸(cm)       焦距(px)
#      ──────────────  =  ──────────────
#      距離(cm)           像素尺寸(px)
#
#  整理後：
#      焦距(px) = 像素尺寸(px) × 距離(cm) / 真實尺寸(cm)
#      真實尺寸(cm) = 像素尺寸(px) × 距離(cm) / 焦距(px)
#
#  流程：
#      1. 用「已知尺寸的 QR code」在影像中的像素大小，算出相機焦距
#      2. 再用同一焦距，把樹幹的像素寬度換算成真實公分數
# ════════════════════════════════════════════════════════════════


def calculate_focal_length(qr_pixel_size, distance_cm):
    """
    步驟一：用 QR code 反推相機焦距（單位：像素）。

    原理：
        QR code 的真實尺寸已知（config.QR_REAL_SIZE_CM），
        拍攝距離由使用者提供，QR code 在影像中的像素大小由偵測取得。
        代入公式即可算出此次拍攝的等效焦距。

    參數：
        qr_pixel_size (float)：QR code 在影像中的寬度，單位：像素
        distance_cm   (float)：拍攝時與 QR code 的距離，單位：公分

    回傳值：
        float：相機焦距，單位：像素
        None ：若輸入值不合法（為 0 或負數）則回傳 None
    """

    # 防止除以零：像素大小不可為 0
    if qr_pixel_size <= 0:
        print("[警告] QR code 像素尺寸不合法，無法計算焦距")
        return None

    # 套用公式：焦距 = 像素尺寸 × 距離 / 真實尺寸
    focal_length = (qr_pixel_size * distance_cm) / config.QR_REAL_SIZE_CM

    return focal_length


def calculate_diameter_method_b(trunk_pixel_width, focal_length, distance_cm):
    """
    步驟二：用焦距公式算出樹幹的真實直徑（方法二）。

    原理：
        焦距已由 calculate_focal_length() 求得，
        樹幹在影像中的像素寬度由 trunk_detector 提供，
        拍攝距離由使用者輸入。
        代入公式即可得到樹幹的真實直徑。

    參數：
        trunk_pixel_width (float)：樹幹在影像中的像素寬度
        focal_length      (float)：相機焦距，單位：像素（由 calculate_focal_length 算出）
        distance_cm       (float)：拍攝時與樹幹的距離，單位：公分

    回傳值：
        float：樹幹真實直徑，單位：公分
        None ：若輸入值不合法則回傳 None
    """

    # 防止除以零：焦距不可為 0
    if focal_length is None or focal_length <= 0:
        print("[警告] 焦距不合法，無法進行方法二計算")
        return None

    # 防止負值影像寬度
    if trunk_pixel_width <= 0:
        print("[警告] 樹幹像素寬度不合法，無法計算直徑")
        return None

    # 套用公式：真實尺寸 = 像素寬度 × 距離 / 焦距
    diameter_cm = (trunk_pixel_width * distance_cm) / focal_length

    return diameter_cm


def run_method_b(trunk_pixel_width, qr_pixel_size, distance_m):
    """
    方法二的完整流程（對外主要呼叫入口）。

    將上面兩個步驟整合成一個函式：
        1. 用 QR code 算焦距
        2. 用焦距算樹幹直徑

    參數：
        trunk_pixel_width (float)：樹幹在影像中的像素寬度
        qr_pixel_size     (float)：QR code 在影像中的像素寬度
        distance_m        (float)：拍攝距離，單位：公尺（會自動轉換為公分）

    回傳值：
        float：樹幹真實直徑，單位：公分
        None ：任一步驟失敗時回傳 None
    """

    # 單位換算：公尺 → 公分（公式需要公分）
    distance_cm = distance_m * 100

    # 步驟一：算焦距
    focal_length = calculate_focal_length(qr_pixel_size, distance_cm)
    if focal_length is None:
        return None  # 焦距算失敗，直接中止

    # 步驟二：算直徑
    diameter_cm = calculate_diameter_method_b(trunk_pixel_width, focal_length, distance_cm)

    return diameter_cm

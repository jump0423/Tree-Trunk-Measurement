#昌先打，再換輔
# =============================================================
# trunk_detector.py — YOLO 樹幹偵測模組
# 框架：jump 建立
# 實作：Morris 負責填入
#
# 這個模組負責：
#   1. 載入 YOLO 模型（best.pt）
#   2. 把圖片丟給模型推論
#   3. 取回樹幹的輪廓點（masks.xy）和信心度
#   4. 檢查樹幹是否完整入鏡
#
# 回傳格式統一為 dict，方便 main.py 取用：
#   {
#     "masks_xy":   np.ndarray,  # 樹幹輪廓點，shape (N, 2)
#     "confidence": float,       # YOLO 信心度 0.0 ~ 1.0
#     "box":        list         # 邊界框 [x1, y1, x2, y2]
#   }
#   偵測失敗時回傳 None
# =============================================================

import numpy as np
import config


class TrunkDetector:
    """
    YOLO 樹幹偵測器

    使用方式：
        detector  = TrunkDetector(config.MODEL_PATH, config.CONF_THRESHOLD)
        detection = detector.detect(image)

        if detection is None:
            print("未偵測到樹幹")
        else:
            trunk_pts  = detection["masks_xy"]
            confidence = detection["confidence"]
    """

    def __init__(self, model_path: str, conf: float):
        """
        初始化：載入 YOLO 模型

        參數：
            model_path : str
                模型權重檔路徑，例如 "best.pt"
                檔案要放在專案根目錄

            conf : float
                信心度閾值，例如 0.65
                低於這個值的偵測結果會被丟掉
                避免把背景誤認成樹幹
        """
        self.conf = conf

        # TODO（Morris）：
        # 用 ultralytics 載入模型
        # 範例：
        #   from ultralytics import YOLO
        #   self.model = YOLO(model_path)
        self.model = None  # 暫時留空，等 Morris 實作

    def detect(self, image: np.ndarray) -> dict:
        """
        執行 YOLO 推論，回傳樹幹偵測結果

        參數：
            image : np.ndarray
                OpenCV 讀取的圖片，格式為 BGR，shape 為 (H, W, 3)

        回傳：
            dict 或 None
                成功：{"masks_xy": ..., "confidence": ..., "box": ...}
                失敗：None（未偵測到樹幹，或信心度不足）
        """

        # TODO（Morris）：
        # 1. 用 self.model 對 image 執行推論
        # 2. 從結果取出信心度最高的那個偵測框
        # 3. 確認信心度 >= self.conf，否則回傳 None
        # 4. 從 masks.xy 取出輪廓點，轉成 np.ndarray
        # 5. 回傳 dict 格式如上
        #
        # 範例（參考你原本的 detector.py）：
        #   results = self.model(image, conf=self.conf)
        #   if len(results[0].masks) == 0:
        #       return None
        #   masks_xy   = results[0].masks.xy[0]  # 第一個偵測結果的輪廓
        #   confidence = float(results[0].boxes.conf[0])
        #   box        = results[0].boxes.xyxy[0].tolist()
        #   return {"masks_xy": masks_xy, "confidence": confidence, "box": box}

        return None  # 暫時回傳 None，等 Morris 實作

    def get_trunk_pixel_width(self, mask: np.ndarray) -> float:
        """
        從分割遮罩計算樹幹最寬處的像素距離

        參數：
            mask : np.ndarray
                樹幹輪廓點陣列，shape 為 (N, 2)

        回傳：
            float — 樹幹最寬處的像素距離
        """

        # TODO（Morris）：
        # 從輪廓點找出最左和最右的 x 座標，相減就是寬度
        # 範例：
        #   return float(mask[:, 0].max() - mask[:, 0].min())

        return 0.0  # 暫時回傳 0，等 Morris 實作

    def is_trunk_complete(self, box: list, img_shape: tuple) -> bool:
        """
        檢查偵測框是否碰到影像邊緣
        碰到邊緣代表樹幹沒有完整入鏡，DBH 可能被低估

        參數：
            box : list
                邊界框 [x1, y1, x2, y2]，單位像素

            img_shape : tuple
                圖片 shape，格式 (height, width) 或 (height, width, 3)

        回傳：
            bool
                True  → 樹幹完整入鏡
                False → 樹幹碰到邊緣，可能不完整
        """

        # TODO（Morris）：
        # 檢查 box 的左右邊界是否碰到圖片邊緣（容差 10px）
        # 範例：
        #   img_w  = img_shape[1]
        #   margin = 10
        #   x1, x2 = box[0], box[2]
        #   return x1 > margin and x2 < img_w - margin

        return True  # 暫時回傳 True，等 Morris 實作
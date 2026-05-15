#輔負責
# =============================================================
# trunk_detector.py — YOLO 樹幹偵測模組
#
# 封裝 YOLOv11 模型，負責在影像中找到樹幹並回傳輪廓資訊。
# 本模組只做「偵測」，不做任何直徑計算。
#
# 回傳格式：dict，包含：
#   masks_xy   : np.ndarray，shape (N, 2)，樹幹輪廓點（像素座標）
#   confidence : float，YOLO 信心度分數（0.0 ~ 1.0）
#   box        : list，邊界框 [x1, y1, x2, y2]
# =============================================================

import os
import numpy as np
from ultralytics import YOLO   # YOLOv11 套件


class TrunkDetector:
    """
    YOLO 樹幹偵測器

    載入訓練好的模型，對輸入影像進行推論，
    回傳信心度最高的那棵樹的輪廓點與信心度。

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
        初始化偵測器，載入 YOLO 模型。

        參數：
            model_path (str)  ：模型權重檔路徑，例如 "best.pt"
            conf       (float)：信心度閾值（來自 config.CONF_THRESHOLD）
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型檔：{model_path}")

        print("正在載入 YOLO 模型...")
        self.model = YOLO(model_path)
        self.conf  = conf

    def detect(self, image: np.ndarray):
        """
        對輸入影像執行 YOLO 推論，回傳信心度最高的樹幹資訊。

        若偵測不到，回傳 None（不拋出例外，讓 main.py 處理）。

        參數：
            image (np.ndarray)：OpenCV 讀入的影像，BGR 格式

        回傳值：
            dict 或 None
              成功：{"masks_xy": ..., "confidence": ..., "box": ...}
              失敗：None
        """
        # save=False：不儲存推論結果圖
        # verbose=False：不在終端機印出推論詳情
        results = self.model.predict(
            source=image,
            save=False,
            verbose=False,
            conf=self.conf
        )

        for result in results:
            # 確認有偵測到遮罩
            if result.masks is None or len(result.boxes) == 0:
                continue

            # 選取信心度最高的偵測結果
            conf_scores = result.boxes.conf.cpu().numpy()
            best_idx    = int(np.argmax(conf_scores))

            # 取出輪廓點（masks.xy 是 list，每個元素是一個物件的輪廓 (N, 2)）
            trunk_pts = result.masks.xy[best_idx]
            if len(trunk_pts) == 0:
                continue

            # 取出邊界框 [x1, y1, x2, y2]
            box = result.boxes.xyxy[best_idx].cpu().numpy().tolist()

            return {
                "masks_xy":   trunk_pts,
                "confidence": float(conf_scores[best_idx]),
                "box":        box
            }

        return None  # 偵測失敗

    def get_trunk_pixel_width(self, mask: np.ndarray) -> float:
        """
        從輪廓點陣列算出樹幹最寬處的像素距離。

        參數：
            mask (np.ndarray)：樹幹輪廓點陣列，shape 為 (N, 2)

        回傳值：
            float：樹幹最寬處的像素距離
        """
        # 最右邊 x 減去最左邊 x = 橫向寬度
        return float(mask[:, 0].max() - mask[:, 0].min())

    def is_trunk_complete(self, box: list, img_shape: tuple) -> bool:
        """
        檢查偵測框是否碰到影像邊緣（判斷樹幹是否完整入鏡）。

        參數：
            box       (list) ：邊界框 [x1, y1, x2, y2]，像素座標
            img_shape (tuple)：影像的 shape，格式 (height, width, ...)

        回傳值：
            True  → 完整入鏡
            False → 樹幹可能被裁切
        """
        img_w  = img_shape[1]
        margin = 5   # 容差 5 像素

        x1 = box[0]  # 偵測框左邊緣
        x2 = box[2]  # 偵測框右邊緣

        if x1 < margin:
            return False  # 左側超出
        if x2 > img_w - margin:
            return False  # 右側超出

        return True

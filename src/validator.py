#共用
# =============================================================
# validator.py — 雙驗證模組
#
# 比較方法一（QR code 比例尺）和方法二（焦距公式）的結果差異，
# 判斷最終輸出狀態，並建立 MeasurementResult 物件。
#
# 輸出狀態碼：
#   "verified"  → 兩種方法差異 < 20%，結果可信，取平均輸出
#   "mismatch"  → 兩種方法差異 >= 20%，輸出方法二，標記警告
#   "qr_failed" → QR code 偵測失敗，只有方法二，直接輸出
# =============================================================

import datetime
from src.models import MeasurementResult
import config


class Validator:
    """
    雙驗證器

    屬性：
        threshold (float)：相似度門檻，來自 config.SIMILARITY_THRESHOLD
                           差異率超過 (1 - threshold) 就發出警告
    """

    def __init__(self, threshold: float):
        """
        初始化驗證器。

        參數：
            threshold (float)：相似度門檻，例如 0.80
                               代表兩種方法的差異不能超過 20%
        """
        self.threshold = threshold

    def validate(self, result_a, result_b: float) -> MeasurementResult:
        """
        比較方法一和方法二的結果，決定最終輸出。

        參數：
            result_a (float 或 None)：方法一結果（QR code 比例尺）
                                      QR code 偵測失敗時為 None
            result_b (float)         ：方法二結果（焦距公式），永遠有值

        回傳值：
            MeasurementResult：填好 diameter_cm、method、status 的結果物件
                               其他欄位（confidence、image_file 等）由 main.py 補填
        """

        # 取得目前時間，格式為 YYYY-MM-DD HH:MM:SS
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── 情況一：QR code 失靈（result_a 為 None）─────────────
        # 只有方法二可以用，直接輸出，標記 qr_failed
        if result_a is None:
            return MeasurementResult(
                diameter_cm  = round(result_b, 2),
                method       = "focal_only",    # 只用焦距公式
                status       = "qr_failed",     # QR code 沒有成功
                result_a     = None,
                result_b     = round(result_b, 2),
                timestamp    = now,
                warnings     = ["QR code 偵測失敗，已改用焦距公式備援"]
            )

        # ── 情況二：兩種方法都有結果 ─────────────────────────────
        # 計算差異率，判斷是否在可接受範圍內
        similarity = self._similarity(result_a, result_b)

        if similarity >= self.threshold:
            # 差異 < 20%，兩種結果一致，取平均輸出
            avg = (result_a + result_b) / 2
            return MeasurementResult(
                diameter_cm  = round(avg, 2),
                method       = "dual",       # 雙驗證
                status       = "verified",   # 通過驗證
                result_a     = round(result_a, 2),
                result_b     = round(result_b, 2),
                timestamp    = now,
                warnings     = []
            )
        else:
            # 差異 >= 20%，結果不一致，輸出方法二並警告
            diff_pct = round((1 - similarity) * 100, 1)
            return MeasurementResult(
                diameter_cm  = round(result_b, 2),
                method       = "focal_only",  # 採用方法二
                status       = "mismatch",    # 兩種方法不一致
                result_a     = round(result_a, 2),
                result_b     = round(result_b, 2),
                timestamp    = now,
                warnings     = [
                    f"兩種方法差異 {diff_pct}%，超過門檻 {int((1-self.threshold)*100)}%，"
                    f"方法一={result_a:.1f}cm，方法二={result_b:.1f}cm，"
                    f"建議確認 QR code 貼附是否正確"
                ]
            )

    def _similarity(self, a: float, b: float) -> float:
        """
        計算兩個數值的相似度（內部使用）。

        公式：相似度 = 1 - (差異 ÷ 較大值)
        範圍：0.0（完全不同）到 1.0（完全相同）

        例如：
            a = 20.0，b = 22.0
            差異率 = |20 - 22| / max(20, 22) = 2 / 22 ≈ 0.091
            相似度 = 1 - 0.091 = 0.909（差異 9.1%，在 20% 門檻內）

        參數：
            a (float)：第一個數值（方法一結果）
            b (float)：第二個數值（方法二結果）

        回傳值：
            float：相似度，介於 0.0 和 1.0 之間
        """

        # 防呆：避免除以零（兩個值都是 0 的情況）
        if max(a, b) == 0:
            return 1.0  # 都是 0 視為完全相同

        # 差異率 = 絕對差 ÷ 較大值
        diff_ratio = abs(a - b) / max(a, b)

        # 相似度 = 1 - 差異率
        return 1.0 - diff_ratio

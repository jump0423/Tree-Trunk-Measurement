# =============================================================
# carbon_equivalents.py — 固碳量生活化換算模組
#
# 將 CO2 固定當量換算成一般人能感受到的生活比較數字。
#
# 換算依據（台灣數據）：
#   電力排碳係數：0.494 kgCO2/kWh（台灣能源局 2022）
#   汽車排碳：    0.196 kgCO2/km（台灣環保署，汽油轎車平均）
#   手機充電：    0.012 kWh/次（智慧型手機平均耗電）
#   人均碳排：    11,400 kgCO2/年（台灣環保署 2022）→ 31.2 kg/天
# =============================================================


class CarbonEquivalents:
    """
    CO2 固定當量換算器

    使用方式：
        eq = CarbonEquivalents()
        result = eq.calculate(co2_kg=52.3)
        # result["car_km"]["value"] → 267.0
    """

    # ── 台灣數據來源 ──────────────────────────────────────────
    _TW_ELECTRICITY_KG_PER_KWH = 0.494          # 台灣能源局 2022 電力排碳係數
    _TW_CAR_KG_PER_KM          = 0.196          # 台灣環保署，汽油轎車平均
    _PHONE_KWH_PER_CHARGE      = 0.012          # 智慧型手機平均充電耗電量
    _TW_PERSON_KG_PER_DAY      = 11400 / 365    # 台灣人均年碳排 ÷ 365 天（≈31.2 kg/天）

    def calculate(self, co2_kg: float) -> dict:
        """
        計算各項生活化換算結果。

        參數：
            co2_kg (float)：CO2 固定當量（公斤）

        回傳值：
            dict：每個換算項目含 value（數值）、label（說明）、unit（單位）
        """
        if co2_kg <= 0:
            return self._zero_result()

        phone_co2 = self._PHONE_KWH_PER_CHARGE * self._TW_ELECTRICITY_KG_PER_KWH

        return {
            "car_km": {
                "value": round(co2_kg / self._TW_CAR_KG_PER_KM, 1),
                "label": "相當於汽車少行駛",
                "unit":  "公里",
            },
            "phone_charges": {
                "value": int(co2_kg / phone_co2),
                "label": "相當於手機充電",
                "unit":  "次",
            },
            "electricity_kwh": {
                "value": round(co2_kg / self._TW_ELECTRICITY_KG_PER_KWH, 1),
                "label": "相當於節省家庭用電",
                "unit":  "度",
            },
            "person_days": {
                "value": round(co2_kg / self._TW_PERSON_KG_PER_DAY, 2),
                "label": "相當於台灣人均碳排放",
                "unit":  "天",
            },
        }

    def _zero_result(self) -> dict:
        keys = ["car_km", "phone_charges", "electricity_kwh", "person_days"]
        labels = [
            "相當於汽車少行駛",
            "相當於手機充電",
            "相當於節省家庭用電",
            "相當於台灣人均碳排放",
        ]
        units = ["公里", "次", "度", "天"]
        return {
            k: {"value": 0, "label": l, "unit": u}
            for k, l, u in zip(keys, labels, units)
        }

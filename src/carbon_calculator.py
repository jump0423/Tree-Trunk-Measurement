# =============================================================
# carbon_calculator.py — 固碳量計算模組
#
# 使用異速生長方程式（Allometric Equation）由胸高直徑（DBH）
# 估算樹木生質量與固碳量：
#
#   生質量 Biomass (kg) = a × DBH(cm)^b
#   碳儲量 Carbon  (kg) = Biomass × carbon_fraction
#   CO2 當量         (kg) = Carbon × (44/12)
#
# 參數 a、b 來自 species_db.json，不同樹種各異。
# carbon_fraction 預設 0.5（IPCC 通用值：樹木含碳量約 50%）
# CO2 換算係數 44/12 ≈ 3.667（碳原子量 12，CO2 分子量 44）
# =============================================================


class CarbonCalculator:
    """
    固碳量計算器

    使用方式（main.py 呼叫範例）：
        calc = CarbonCalculator()
        carbon_result = calc.calculate(dbh_cm=25.0, species=species_dict)
        # carbon_result["carbon_kg"]  → 碳儲量（公斤）
        # carbon_result["co2_kg"]     → CO2 固定當量（公斤）
        # carbon_result["biomass_kg"] → 生質量（公斤）
    """

    CO2_CONVERSION = 3.67  # 碳質量換算 CO2 當量係數（44/12，取用 IPCC 慣用值）

    def calculate(self, dbh_cm: float, species: dict) -> dict:
        """
        由 DBH 與樹種參數計算固碳量。

        參數：
            dbh_cm  (float)：胸高直徑（公分）
            species (dict) ：樹種參數，需含 "a"、"b"、"carbon_fraction"

        回傳值：
            dict：
                "biomass_kg" (float)：樹木生質量（公斤，乾重）
                "carbon_kg"  (float)：碳儲量（公斤）
                "co2_kg"     (float)：CO2 固定當量（公斤）

        注意：
            若 dbh_cm <= 0 回傳全零，避免冪次運算產生無效值。
        """
        if dbh_cm <= 0:
            return {"biomass_kg": 0.0, "carbon_kg": 0.0, "co2_kg": 0.0}

        a  = float(species["a"])
        b  = float(species["b"])
        cf = float(species.get("carbon_fraction", 0.5))

        biomass_kg = a * (dbh_cm ** b)
        carbon_kg  = biomass_kg * cf
        co2_kg     = carbon_kg * self.CO2_CONVERSION

        return {
            "biomass_kg": round(biomass_kg, 2),
            "carbon_kg":  round(carbon_kg,  2),
            "co2_kg":     round(co2_kg,     2),
        }

    def calculate_annual(self, dbh_cm: float, species: dict, tree_age: int = None) -> float:
        """
        計算年平均固碳量（kg CO₂/年）。

        有提供樹齡：年平均固碳量 = 總固碳量 ÷ 樹齡
        未提供樹齡：使用生長增量法，ΔD 取 species["annual_dbh_growth_cm"]
        """
        if tree_age is not None and tree_age > 0:
            total = self.calculate(dbh_cm, species)
            return round(total["co2_kg"] / tree_age, 2)

        delta_d = float(species.get("annual_dbh_growth_cm", 1.0))
        a  = float(species["a"])
        b  = float(species["b"])
        cf = float(species.get("carbon_fraction", 0.47))

        biomass_now  = a * (dbh_cm ** b)
        biomass_next = a * ((dbh_cm + delta_d) ** b)
        annual_co2   = (biomass_next - biomass_now) * cf * self.CO2_CONVERSION
        return round(annual_co2, 2)

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MeasurementResult:
    # 核心結果
    diameter_cm:   float         = 0.0
    method:        str           = "unknown"
    status:        str           = "unknown"

    # 驗證細節
    result_a:      Optional[float] = None   # 方法一（QR code），失靈時為 None
    result_b:      float           = 0.0   # 方法二（焦距公式）

    # 信心度資訊
    confidence:    float           = 0.0
    diameter_std:  float           = 0.0
    warnings:      list            = field(default_factory=list)

    # 固碳量
    species_name:   str             = ""    # 樹種名稱
    biomass_kg:     float           = 0.0   # 生質量（公斤）
    carbon_kg:      float           = 0.0   # 碳儲量（公斤）
    co2_kg:         float           = 0.0   # CO2 固定當量（公斤）

    # 紀錄
    timestamp:      str             = ""
    image_file:     str             = ""
    measurement_y:  float           = 0.0   # 實際量測位置的 y 座標（像素），供 visualizer 畫線用
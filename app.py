# =============================================================
# app.py — Flask 網頁後端入口
#
# 執行方式：python app.py
# 區域網路存取：http://<你的電腦IP>:5000
#
# API 端點：
#   POST /api/measure              上傳圖片，回傳樹徑與固碳量
#   GET  /api/species              取得所有樹種清單
#   POST /api/species              新增樹種
#   PUT  /api/species/<id>         更新樹種
#   DELETE /api/species/<id>       刪除樹種
# =============================================================

import base64

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

import config
from src.carbon_calculator  import CarbonCalculator
from src.carbon_equivalents import CarbonEquivalents
from src.error_checker      import ErrorChecker
from src.focal_calculator   import FocalCalculator
from src.geometry           import GeometryEngine
from src.species_manager    import SpeciesManager
from src.trunk_detector     import TrunkDetector
from src.validator          import Validator
from src.visualizer         import Visualizer

app = Flask(__name__)
CORS(app)  # 允許前端跨來源請求（前端在不同 port 時需要）

# ── 所有模組啟動時初始化一次，不重複建立 ─────────────────────────
print("載入 YOLO 模型中，請稍候...")
_detector    = TrunkDetector(config.MODEL_PATH, config.CONF_THRESHOLD)
_geometry    = GeometryEngine()
_checker     = ErrorChecker()
_validator   = Validator(config.SIMILARITY_THRESHOLD)
_visualizer  = Visualizer()
_carbon_calc = CarbonCalculator()
_carbon_eq   = CarbonEquivalents()
_species_mgr = SpeciesManager()
print("伺服器就緒！")


# ══════════════════════════════════════════════════════════════
# 量測 API
# ══════════════════════════════════════════════════════════════

@app.route("/api/measure", methods=["POST"])
def api_measure():
    """
    上傳樹幹圖片，回傳胸高直徑與固碳量。

    Request（multipart/form-data）：
        image        ：圖片檔案（JPG / PNG）
        focal_mm     ：相機焦距（mm）
        sensor_width ：感光元件寬度（mm）
        distance_m   ：拍攝距離（m）
        species_id   ：樹種 ID（來自 GET /api/species 的 id 欄位）

    Response（JSON）：
        diameter_cm     ：胸高直徑（公分）
        status          ：verified / qr_failed
        confidence      ：YOLO 信心度（0~1）
        species_name    ：樹種中文名
        biomass_kg      ：生物量（公斤）
        carbon_kg       ：碳儲量（公斤）
        co2_kg          ：CO2 固定當量（公斤）
        diameter_std    ：直徑標準差（公分）
        warnings        ：警告訊息清單
        annotated_image ：標注後的圖片（base64，可直接放入 <img src="...">）
    """

    # ── 驗證圖片 ─────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "請上傳圖片（欄位名稱：image）"}), 400

    # ── 驗證其他欄位 ──────────────────────────────────────────────
    try:
        focal_mm     = float(request.form["focal_mm"])
        sensor_width = float(request.form["sensor_width"])
        distance_m   = float(request.form["distance_m"])
        species_id   = request.form["species_id"]
    except (KeyError, ValueError):
        return jsonify({
            "error": "缺少或格式錯誤的參數，需要：focal_mm, sensor_width, distance_m, species_id"
        }), 400

    # ── 找出對應樹種 ──────────────────────────────────────────────
    species = next(
        (s for s in _species_mgr.get_all_species() if s["id"] == species_id),
        None
    )
    if species is None:
        return jsonify({"error": f"找不到樹種 ID：{species_id}"}), 400

    # ── 讀取圖片 ──────────────────────────────────────────────────
    raw_bytes = request.files["image"].read()
    nparr     = np.frombuffer(raw_bytes, np.uint8)
    image     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "無法解碼圖片，請確認格式（JPG / PNG）"}), 400

    img_h, img_w = image.shape[:2]

    # ── YOLO 偵測樹幹 ─────────────────────────────────────────────
    detection = _detector.detect(image)
    if detection is None:
        return jsonify({"error": "YOLO 未偵測到樹幹，請確認圖片中有清晰的樹幹"}), 422

    trunk_pts  = detection["masks_xy"]
    confidence = detection["confidence"]

    # ── 焦距法計算 DBH ────────────────────────────────────────────
    focal_calc = FocalCalculator(focal_mm, sensor_width, img_w)
    focal_px   = focal_calc._to_focal_px()
    if focal_px <= 0:
        return jsonify({"error": "焦距計算失敗，請確認相機參數"}), 400

    scale_b      = (distance_m * 100) / focal_px
    target_y_b   = _geometry.compute_target_y(trunk_pts, scale_b)
    geo_result_b = _geometry.get_diameter_at_height(trunk_pts, target_y_b, scale_b)
    result_b     = geo_result_b["diameter_cm"]

    # ── 驗證與組裝結果 ────────────────────────────────────────────
    warnings             = _checker.check_trunk_completeness(trunk_pts, image.shape)
    result               = _validator.validate(None, result_b)
    result.confidence    = confidence
    result.diameter_std  = geo_result_b["std_cm"]
    result.warnings      = warnings
    result.measurement_y = target_y_b

    # ── 固碳量計算 ────────────────────────────────────────────────
    carbon_result        = _carbon_calc.calculate(result.diameter_cm, species)
    result.species_name  = species["name"]
    result.biomass_kg    = carbon_result["biomass_kg"]
    result.carbon_kg     = carbon_result["carbon_kg"]
    result.co2_kg        = carbon_result["co2_kg"]

    # ── 固碳換算生活化比較 ────────────────────────────────────────
    equivalents = _carbon_eq.calculate(result.co2_kg)

    # ── 繪製標注圖片，轉成 base64 ─────────────────────────────────
    output_img = _visualizer.draw(image, detection, result)
    _, buf     = cv2.imencode(".jpg", output_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64    = base64.b64encode(buf).decode("utf-8")

    return jsonify({
        "diameter_cm":     round(result.diameter_cm, 1),
        "method":          result.method,
        "status":          result.status,
        "confidence":      round(result.confidence, 3),
        "species_name":    result.species_name,
        "biomass_kg":      result.biomass_kg,
        "carbon_kg":       result.carbon_kg,
        "co2_kg":          result.co2_kg,
        "diameter_std":    round(result.diameter_std, 2),
        "warnings":        result.warnings,
        "equivalents":     equivalents,
        "annotated_image": f"data:image/jpeg;base64,{img_b64}",
    })


# ══════════════════════════════════════════════════════════════
# 樹種管理 API
# ══════════════════════════════════════════════════════════════

@app.route("/api/species", methods=["GET"])
def api_species_list():
    """回傳所有樹種清單（陣列）"""
    return jsonify(_species_mgr.get_all_species())


@app.route("/api/species", methods=["POST"])
def api_species_add():
    """
    新增一筆樹種。

    Request（JSON）：
        name            ：樹種名稱（中文）必填
        scientific_name ：學名              必填
        a               ：異速生長係數 a    必填
        b               ：異速生長指數 b    必填
        carbon_fraction ：碳比例（建議 0.47）必填
        source          ：數據來源          選填
        description     ：描述              選填
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "需要 JSON body"}), 400

    for field in ["name", "scientific_name", "a", "b", "carbon_fraction"]:
        if field not in data:
            return jsonify({"error": f"缺少必填欄位：{field}"}), 400

    try:
        new_species = {
            "id":               str(data["name"]).replace(" ", "_"),
            "name":             str(data["name"]),
            "scientific_name":  str(data["scientific_name"]),
            "a":                float(data["a"]),
            "b":                float(data["b"]),
            "carbon_fraction":  float(data["carbon_fraction"]),
            "source":           str(data.get("source", "")),
            "data_quality":     "C",
            "description":      str(data.get("description", "")),
        }
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"參數格式錯誤：{e}"}), 400

    _species_mgr.add_species(new_species)
    return jsonify({"message": "新增成功", "species": new_species}), 201


@app.route("/api/species/<species_id>", methods=["PUT"])
def api_species_update(species_id):
    """
    更新指定樹種（部分欄位更新即可）。

    URL param：species_id（即 GET /api/species 回傳的 id 欄位）
    Request（JSON）：要更新的欄位
    """
    data        = request.get_json()
    all_species = _species_mgr.get_all_species()
    idx         = next((i for i, s in enumerate(all_species) if s["id"] == species_id), None)

    if idx is None:
        return jsonify({"error": f"找不到樹種：{species_id}"}), 404

    updated = {**all_species[idx], **data}
    _species_mgr.update_species(idx, updated)
    return jsonify({"message": "更新成功", "species": updated})


@app.route("/api/species/<species_id>", methods=["DELETE"])
def api_species_delete(species_id):
    """刪除指定樹種"""
    all_species = _species_mgr.get_all_species()
    idx         = next((i for i, s in enumerate(all_species) if s["id"] == species_id), None)

    if idx is None:
        return jsonify({"error": f"找不到樹種：{species_id}"}), 404

    _species_mgr.delete_species(idx)
    return jsonify({"message": "刪除成功"})


# ══════════════════════════════════════════════════════════════
# 啟動
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # host='0.0.0.0'：讓同一區域網路的裝置都能連線
    # 朋友輸入你電腦的 IP + :5000 就能訪問，例如 http://192.168.1.88:5000
    app.run(host="0.0.0.0", port=5000, debug=False)

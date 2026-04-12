#輔負責
import argparse  # 用來處理命令列參數的標準函式庫
import os        # 用來檢查檔案與資料夾是否存在


def parse_arguments():
    """
    解析使用者從命令列輸入的參數。

    使用方式範例：
        python main.py --image photo.jpg
        python main.py --folder ./images --distance 3.5

    回傳值：
        argparse.Namespace 物件，可用 args.image、args.folder 等方式取值。
    """

    # 建立一個「參數解析器」，description 是使用者輸入 --help 時看到的說明
    parser = argparse.ArgumentParser(
        description="樹幹直徑量測系統：輸入影像，自動量測樹幹胸高直徑（DBH）"
    )

    # ── 輸入來源（二擇一）──────────────────────────────────────────
    # --image：單張照片路徑
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="單張影像的檔案路徑，例如：--image photo.jpg"
    )

    # --folder：整個資料夾，會批次處理裡面所有圖片
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="含有多張影像的資料夾路徑，例如：--folder ./images"
    )

    # ── 模型設定 ──────────────────────────────────────────────────
    # --model：YOLO 模型權重檔路徑（預設使用 best.pt）
    parser.add_argument(
        "--model",
        type=str,
        default="best.pt",
        help="YOLO 模型權重檔路徑（預設：best.pt）"
    )

    # ── 相機設定 ──────────────────────────────────────────────────
    # --distance：拍攝時與樹幹的距離（公尺），用於焦距公式（方法二）
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="拍攝時與樹幹的距離，單位：公尺，例如：--distance 3.5"
    )

    # ── 輸出設定 ──────────────────────────────────────────────────
    # --output：結果儲存的資料夾（預設使用 config.py 中的 OUTPUT_DIR）
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="結果輸出資料夾路徑（預設：~/Desktop/results）"
    )

    # 執行解析，把使用者輸入的字串轉換成對應型別的值
    args = parser.parse_args()

    return args


def validate_inputs(args):
    """
    檢查使用者輸入的參數是否合法。
    如果有問題，直接印出錯誤訊息並拋出例外，讓程式停止。

    參數：
        args：parse_arguments() 回傳的物件

    回傳值：
        無。發現問題時拋出 ValueError 或 FileNotFoundError。
    """

    # ── 檢查輸入來源 ─────────────────────────────────────────────
    # 使用者必須提供 --image 或 --folder 其中一個，不能都沒給
    if args.image is None and args.folder is None:
        raise ValueError("請提供影像來源：使用 --image 指定單張照片，或 --folder 指定資料夾")

    # 也不能兩個同時給（避免混淆）
    if args.image is not None and args.folder is not None:
        raise ValueError("--image 和 --folder 只能擇一使用，請勿同時指定")

    # ── 檢查單張照片 ──────────────────────────────────────────────
    if args.image is not None:
        # 檢查檔案是否存在
        if not os.path.isfile(args.image):
            raise FileNotFoundError(f"找不到影像檔案：{args.image}")

        # 檢查副檔名是否為支援的圖片格式
        supported = [".jpg", ".jpeg", ".png", ".bmp"]
        _, ext = os.path.splitext(args.image)  # 分離檔名與副檔名
        if ext.lower() not in supported:
            raise ValueError(f"不支援的圖片格式：{ext}，請使用 {supported} 其中之一")

    # ── 檢查資料夾 ────────────────────────────────────────────────
    if args.folder is not None:
        # 檢查資料夾是否存在
        if not os.path.isdir(args.folder):
            raise FileNotFoundError(f"找不到資料夾：{args.folder}")

    # ── 檢查模型檔 ────────────────────────────────────────────────
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"找不到 YOLO 模型檔：{args.model}")

    # ── 檢查距離 ──────────────────────────────────────────────────
    if args.distance is not None:
        # 距離必須是正數（不能是 0 或負數）
        if args.distance <= 0:
            raise ValueError(f"拍攝距離必須大於 0，你輸入的是：{args.distance}")

    # 全部通過，印出確認訊息
    print("輸入參數驗證通過")

#輔負責
import tkinter as tk                        # Python 內建 GUI 套件
from tkinter import filedialog, messagebox  # 檔案選擇視窗、訊息對話框
import config                               # 讀取全域設定


class InputHandler:
    """
    使用者輸入處理器

    負責所有「使用者互動」的部分：
      - 彈出視窗讓使用者選擇圖片
      - 用對話框詢問相機焦距與感光元件寬度
      - 詢問拍攝距離，並檢查是否超過建議範圍

    所有輸入驗證集中在這個類別，其他模組不需要處理輸入錯誤。
    """

    # ----------------------------------------------------------
    # 公開方法
    # ----------------------------------------------------------

    def get_image_paths(self) -> list:
        """
        彈出 tkinter 檔案選擇視窗，讓使用者一次選一張或多張圖片。

        支援格式：.jpg、.jpeg、.png、.bmp

        回傳值：
            list[str]：使用者選擇的圖片完整路徑清單

        例外：
            若使用者關閉視窗沒有選擇任何檔案，拋出 ValueError
        """

        root = tk.Tk()
        root.withdraw()

        filetypes = [
            ("圖片檔案", "*.jpg *.jpeg *.png *.bmp"),
            ("所有檔案", "*.*"),
        ]

        # askopenfilenames（複數）允許按住 Ctrl/Shift 多選
        image_paths = filedialog.askopenfilenames(
            title="請選擇要量測的樹幹照片（可按 Ctrl 多選）",
            filetypes=filetypes
        )

        root.destroy()

        if not image_paths:
            raise ValueError("未選擇圖片，程式結束")

        paths = list(image_paths)
        print(f"已選擇 {len(paths)} 張圖片")
        return paths

    def get_camera_params(self) -> tuple:
        """
        先讓使用者選擇手機型號。
        內建型號會直接使用 config.CAMERA_PRESETS 的相機參數；
        選「其他」時，才用文字輸入對話框詢問焦距和感光元件寬度。

        這兩個參數是「方法二（焦距公式）」用來計算比例尺的必要資訊。
        通常可以在相機規格表或鏡頭包裝上找到。

        回傳值：
            tuple(float, float)：(焦距_mm, 感光元件寬度_mm)

        例外：
            若使用者輸入非數字或關閉視窗，拋出 ValueError
        """

        # 建立隱藏的根視窗（只讓對話框出現）
        root = tk.Tk()
        root.withdraw()

        phone_model = self._ask_choice(
            root,
            title="手機型號選擇",
            prompt="請選擇拍攝照片使用的手機型號",
            choices=config.PHONE_MODEL_OPTIONS,
        )

        if phone_model != config.PHONE_MODEL_OTHER:
            root.destroy()
            preset = config.CAMERA_PRESETS[phone_model]
            focal_mm = float(preset["focal_mm"])
            sensor_w = float(preset["sensor_width_mm"])
            print(f"手機型號：{phone_model}")
            print(f"相機參數：焦距 {focal_mm}mm，感光元件寬度 {sensor_w}mm")
            return focal_mm, sensor_w

        # ── 詢問焦距 ──────────────────────────────────────────
        focal_str = self._ask_value(
            root,
            title="相機參數輸入",
            prompt="請輸入相機焦距（mm）\n\n"
                   "手機相機通常為 3~6mm\n"
                   "單眼標準鏡頭通常為 35~50mm"
        )

        # ── 詢問感光元件寬度 ───────────────────────────────────
        sensor_str = self._ask_value(
            root,
            title="相機參數輸入",
            prompt="請輸入感光元件寬度（mm）\n\n"
                   "全片幅：36mm\n"
                   "APS-C：23.5mm\n"
                   "手機（1/1.7\"）：7.6mm"
        )

        root.destroy()

        # 把字串轉成浮點數，並做基本驗證
        try:
            focal_mm   = float(focal_str)
            sensor_w   = float(sensor_str)
        except ValueError:
            raise ValueError("焦距和感光元件寬度必須是數字")

        if focal_mm <= 0:
            raise ValueError(f"焦距必須大於 0，你輸入的是：{focal_mm}")
        if sensor_w <= 0:
            raise ValueError(f"感光元件寬度必須大於 0，你輸入的是：{sensor_w}")

        print(f"相機參數：焦距 {focal_mm}mm，感光元件寬度 {sensor_w}mm")
        return focal_mm, sensor_w

    def get_distance(self) -> float:
        """
        詢問使用者拍攝時與樹幹的距離（公尺）。

        若距離超過 config.MAX_DISTANCE_M，
        顯示警告但不強制阻止，讓使用者自行決定是否繼續。

        回傳值：
            float：拍攝距離（公尺）

        例外：
            若使用者輸入非數字或距離 <= 0，拋出 ValueError
        """

        root = tk.Tk()
        root.withdraw()

        distance_str = self._ask_value(
            root,
            title="拍攝距離輸入",
            prompt=f"請輸入拍攝時與樹幹的距離（公尺）\n\n"
                   f"建議距離：1.5 ~ {config.MAX_DISTANCE_M} 公尺\n"
                   f"超過 {config.MAX_DISTANCE_M} 公尺時 QR code 可能偵測不到"
        )

        root.destroy()

        # 轉換成浮點數
        try:
            distance_m = float(distance_str)
        except ValueError:
            raise ValueError("距離必須是數字")

        # 驗證合法性
        if not self._validate_distance(distance_m):
            raise ValueError(f"拍攝距離必須大於 0，你輸入的是：{distance_m}")

        # 距離超過上限時，彈出警告視窗（但程式繼續執行）
        if distance_m > config.MAX_DISTANCE_M:
            root2 = tk.Tk()
            root2.withdraw()
            messagebox.showwarning(
                title="距離警告",
                message=f"拍攝距離 {distance_m}m 超過建議上限 {config.MAX_DISTANCE_M}m，\n"
                        f"QR code 可能無法偵測，系統將改用焦距公式備援。"
            )
            root2.destroy()

        print(f"拍攝距離：{distance_m} 公尺")
        return distance_m

    # ----------------------------------------------------------
    # 內部方法（只在類別內部使用，外部不直接呼叫）
    # ----------------------------------------------------------

    def _ask_value(self, root: tk.Tk, title: str, prompt: str) -> str:
        """
        彈出一個簡單的文字輸入對話框，等待使用者輸入並回傳結果。

        參數：
            root   (tk.Tk)：已建立的 tkinter 根視窗
            title  (str)  ：對話框標題
            prompt (str)  ：顯示給使用者的提示文字

        回傳值：
            str：使用者輸入的文字

        例外：
            若使用者關閉視窗（沒有按確認），拋出 ValueError
        """

        # 使用 tkinter 內建的 simpledialog 詢問單一數值
        from tkinter import simpledialog
        value = simpledialog.askstring(title=title, prompt=prompt, parent=root)

        # 使用者關閉視窗（value 為 None）
        if value is None:
            raise ValueError(f"使用者取消了輸入（{title}），程式結束")

        # 去掉前後空白，避免使用者不小心輸入空白鍵
        return value.strip()

    def _ask_choice(self, root: tk.Tk, title: str, prompt: str, choices: list) -> str:
        """
        用編號清單讓使用者選擇一個選項。

        回傳值：
            str：被選中的選項文字
        """
        from tkinter import simpledialog

        lines = [prompt, ""]
        for idx, choice in enumerate(choices, start=1):
            lines.append(f"{idx}. {choice}")

        value = simpledialog.askinteger(
            title=title,
            prompt="\n".join(lines),
            parent=root,
            minvalue=1,
            maxvalue=len(choices),
        )

        if value is None:
            raise ValueError(f"使用者取消了輸入（{title}），程式結束")

        return choices[value - 1]

    def _validate_distance(self, d: float) -> bool:
        """
        檢查距離是否合法（必須為正數）。

        參數：
            d (float)：要驗證的距離值

        回傳值：
            bool：True 代表合法，False 代表不合法
        """
        return d > 0

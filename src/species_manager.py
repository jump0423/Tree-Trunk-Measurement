# =============================================================
# species_manager.py — 樹種資料庫管理器
#
# 負責：
#   1. 讀取 / 儲存 data/species_db.json
#   2. 提供 tkinter 視窗讓使用者選擇樹種
#   3. 提供新增、編輯、刪除樹種的介面
#
# 使用方式（main.py 呼叫範例）：
#   manager = SpeciesManager()
#   species = manager.select_species()
#   # species["name"]            → "榕樹"
#   # species["a"], species["b"] → 異速生長係數
# =============================================================

import json
import os
import tkinter as tk
from tkinter import messagebox


# ── 資料庫預設路徑：data/species_db.json（相對於 src/ 上一層）────
_DEFAULT_DB_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "species_db.json")
)


class SpeciesManager:
    """
    樹種資料庫管理器

    屬性：
        db_path (str)：species_db.json 的完整路徑
        _db     (dict)：已載入的資料庫內容
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or _DEFAULT_DB_PATH
        self._db = self._load_db()

    # ----------------------------------------------------------
    # 公開方法
    # ----------------------------------------------------------

    def select_species(self) -> dict:
        """
        開啟樹種選擇視窗，讓使用者選擇一個樹種。

        使用者也可以在此視窗新增、編輯或刪除樹種。

        回傳值：
            dict：選中的樹種資料（含 name、a、b、carbon_fraction 等欄位）

        例外：
            若使用者關閉視窗未選擇，拋出 ValueError
        """
        root = tk.Tk()
        root.withdraw()

        dialog = _SpeciesDialog(root, self)
        root.wait_window(dialog.window)
        root.destroy()

        if dialog.selected_species is None:
            raise ValueError("未選擇樹種，程式結束")

        print(f"已選擇樹種：{dialog.selected_species['name']}")
        return dialog.selected_species

    def get_all_species(self) -> list:
        """回傳全部樹種清單（list of dict）"""
        return self._db.get("species", [])

    def add_species(self, data: dict) -> None:
        """新增一筆樹種並寫回 JSON"""
        self._db["species"].append(data)
        self._save_db()

    def update_species(self, index: int, data: dict) -> None:
        """更新指定索引的樹種並寫回 JSON"""
        self._db["species"][index] = data
        self._save_db()

    def delete_species(self, index: int) -> None:
        """刪除指定索引的樹種並寫回 JSON"""
        self._db["species"].pop(index)
        self._save_db()

    # ----------------------------------------------------------
    # 內部方法
    # ----------------------------------------------------------

    def _load_db(self) -> dict:
        """讀取 JSON 資料庫；若不存在則回傳空結構"""
        if not os.path.isfile(self.db_path):
            print(f"[SpeciesManager] 找不到資料庫：{self.db_path}，使用空白資料庫")
            return {"version": "1.0", "species": []}
        with open(self.db_path, encoding="utf-8") as f:
            return json.load(f)

    def _save_db(self) -> None:
        """將 _db 寫回 JSON 檔案"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, mode="w", encoding="utf-8") as f:
            json.dump(self._db, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════
# 以下為純 tkinter UI 類別，只在 select_species() 內部使用
# ══════════════════════════════════════════════════════════════

class _SpeciesDialog:
    """
    樹種選擇主視窗

    使用者可以：
      - 從清單點選樹種
      - 按「選擇此樹種」確認
      - 按「新增 / 編輯 / 刪除」管理資料庫
    """

    def __init__(self, parent: tk.Tk, manager: SpeciesManager):
        self.manager = manager
        self.selected_species = None

        self.window = tk.Toplevel(parent)
        self.window.title("樹種固碳量資料庫 — 請選擇樹種")
        self.window.geometry("560x460")
        self.window.resizable(False, False)
        self.window.grab_set()
        self._build_ui()

    def _build_ui(self):
        # ── 標題 ──────────────────────────────────────────────
        tk.Label(
            self.window,
            text="樹種固碳量資料庫",
            font=("Arial", 15, "bold")
        ).pack(pady=(15, 2))

        tk.Label(
            self.window,
            text="公式：生質量(kg) = a × DBH^b　碳儲量(kg) = 生質量 × 碳比例",
            font=("Arial", 9),
            fg="#555555"
        ).pack(pady=(0, 8))

        # ── 清單框 ──────────────────────────────────────────
        list_frame = tk.Frame(self.window)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=("Arial", 11),
            selectmode=tk.SINGLE,
            activestyle="underline",
            height=12,
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)
        self.listbox.bind("<Double-Button-1>", lambda e: self._select())

        self._refresh_list()

        # ── 說明列 ──────────────────────────────────────────
        self.info_label = tk.Label(
            self.window,
            text="",
            font=("Arial", 9),
            fg="#336699",
            anchor="w"
        )
        self.info_label.pack(fill=tk.X, padx=20, pady=(2, 0))
        self.listbox.bind("<<ListboxSelect>>", self._on_select_preview)

        # ── 操作按鈕 ─────────────────────────────────────────
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(pady=12)

        tk.Button(
            btn_frame, text="✔ 選擇此樹種",
            command=self._select,
            bg="#4CAF50", fg="white",
            font=("Arial", 10, "bold"),
            width=14, height=1,
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            btn_frame, text="＋ 新增樹種",
            command=self._add,
            font=("Arial", 10),
            width=12,
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            btn_frame, text="✎ 編輯樹種",
            command=self._edit,
            font=("Arial", 10),
            width=12,
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            btn_frame, text="✖ 刪除樹種",
            command=self._delete,
            bg="#f44336", fg="white",
            font=("Arial", 10),
            width=12,
        ).pack(side=tk.LEFT, padx=6)

    def _refresh_list(self):
        """重新載入清單"""
        self.listbox.delete(0, tk.END)
        for sp in self.manager.get_all_species():
            self.listbox.insert(
                tk.END,
                f"  {sp['name']}（{sp['scientific_name']}）"
            )

    def _on_select_preview(self, _event=None):
        """選中時在下方顯示樹種描述"""
        idx = self._current_index()
        if idx is None:
            return
        sp = self.manager.get_all_species()[idx]
        desc = sp.get("description", "")
        self.info_label.config(
            text=f"a={sp['a']}  b={sp['b']}  碳比例={sp['carbon_fraction']}　│　{desc}"
        )

    def _current_index(self):
        """回傳目前選中的清單索引，未選回傳 None"""
        sel = self.listbox.curselection()
        return sel[0] if sel else None

    def _require_selection(self) -> bool:
        """若未選任何項目，彈出提示並回傳 False"""
        if self._current_index() is None:
            messagebox.showwarning("提示", "請先點選一個樹種", parent=self.window)
            return False
        return True

    def _select(self):
        if not self._require_selection():
            return
        self.selected_species = self.manager.get_all_species()[self._current_index()]
        self.window.destroy()

    def _add(self):
        _EditDialog(self.window, self.manager, mode="add", on_save=self._refresh_list)

    def _edit(self):
        if not self._require_selection():
            return
        idx = self._current_index()
        _EditDialog(
            self.window, self.manager,
            mode="edit",
            species=self.manager.get_all_species()[idx],
            species_idx=idx,
            on_save=self._refresh_list,
        )

    def _delete(self):
        if not self._require_selection():
            return
        idx = self._current_index()
        sp  = self.manager.get_all_species()[idx]
        if messagebox.askyesno(
            "確認刪除",
            f"確定要刪除「{sp['name']}」嗎？\n此操作無法復原。",
            parent=self.window
        ):
            self.manager.delete_species(idx)
            self._refresh_list()
            self.info_label.config(text="")


class _EditDialog:
    """
    新增 / 編輯樹種的表單對話框

    欄位：名稱、學名、a、b、碳比例、描述
    """

    _FIELDS = [
        ("name",             "樹種名稱（中文）",           "",       str),
        ("scientific_name",  "學名（英文/拉丁文）",        "",       str),
        ("a",                "係數 a（異速生長）",         "0.1184", float),
        ("b",                "指數 b（異速生長）",         "2.53",   float),
        ("carbon_fraction",  "碳比例（IPCC標準 0.47）",   "0.47",   float),
        ("source",           "數據來源（論文/文獻）",      "",       str),
        ("description",      "描述（可空白）",             "",       str),
    ]

    def __init__(
        self,
        parent,
        manager: SpeciesManager,
        mode: str = "add",
        species: dict = None,
        species_idx: int = None,
        on_save=None,
    ):
        self.manager     = manager
        self.mode        = mode
        self.species     = species or {}
        self.species_idx = species_idx
        self.on_save     = on_save

        self.window = tk.Toplevel(parent)
        self.window.title("新增樹種" if mode == "add" else "編輯樹種")
        self.window.geometry("420x320")
        self.window.resizable(False, False)
        self.window.grab_set()
        self._build_ui()

    def _build_ui(self):
        self.entries = {}

        for row, (key, label, default, _dtype) in enumerate(self._FIELDS):
            tk.Label(
                self.window, text=label, anchor="w", font=("Arial", 10)
            ).grid(row=row, column=0, padx=18, pady=6, sticky="w")

            entry = tk.Entry(self.window, width=28, font=("Arial", 10))
            entry.grid(row=row, column=1, padx=10, pady=6)

            # 編輯模式預填舊值；新增模式填預設值
            prefill = str(self.species.get(key, default))
            if prefill:
                entry.insert(0, prefill)

            self.entries[key] = entry

        tk.Button(
            self.window,
            text="儲存",
            command=self._save,
            bg="#4CAF50", fg="white",
            font=("Arial", 11, "bold"),
            width=18,
        ).grid(row=len(self._FIELDS), column=0, columnspan=2, pady=16)

    def _save(self):
        try:
            raw = {key: entry.get().strip() for key, entry in self.entries.items()}

            if not raw["name"]:
                messagebox.showerror("錯誤", "樹種名稱不能為空", parent=self.window)
                return

            a  = float(raw["a"])
            b  = float(raw["b"])
            cf = float(raw["carbon_fraction"])

            if a <= 0:
                messagebox.showerror("錯誤", "係數 a 必須大於 0", parent=self.window)
                return
            if b <= 0:
                messagebox.showerror("錯誤", "指數 b 必須大於 0", parent=self.window)
                return
            if not (0 < cf <= 1):
                messagebox.showerror("錯誤", "碳比例必須介於 0 ~ 1 之間", parent=self.window)
                return

            data = {
                "id":               raw["name"].replace(" ", "_"),
                "name":             raw["name"],
                "scientific_name":  raw["scientific_name"],
                "a":                a,
                "b":                b,
                "carbon_fraction":  cf,
                "source":           raw.get("source", ""),
                "data_quality":     "C",
                "description":      raw["description"],
            }

            if self.mode == "add":
                self.manager.add_species(data)
            else:
                self.manager.update_species(self.species_idx, data)

            if self.on_save:
                self.on_save()

            self.window.destroy()

        except ValueError:
            messagebox.showerror(
                "格式錯誤",
                "係數 a、指數 b、碳比例請輸入數字",
                parent=self.window
            )

#!/usr/bin/env python
"""
pipelines/refusal_data_gui.py

拒絕資料集的圖形介面（Tkinter）。與 CLI 版 refusal_data_editor.py 共用同一套資料層
（load/save/next_id/compute_stats/schema），只在上面加視窗 UI，所以兩者產出的檔案
一模一樣、都能直接餵給 km_search。

這是 refusal_data_editor.py 的預設前端：`python refusal_data_editor.py` 或
`run.py data edit-prompts` 都會開這個 GUI（加 --cli 才走終端）。也可直接開：
    python pipelines/refusal_data_gui.py [資料集路徑]

功能：選檔/新建/存檔、索引表（瀏覽所有題目）、新增/更新/刪除、即時統計與警告。
"""
import sys
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.path.insert(0, str(Path(__file__).parent))
from refusal_data_editor import (  # noqa: E402  共用資料層，單一來源
    load, save, next_id, compute_stats, GROUPS, DEFAULT_PATH,
    available_models, model_group_map,
)


class App:
    def __init__(self, root: tk.Tk, path: Path):
        self.root = root
        self.path = Path(path)
        self.entries: list[dict] = load(self.path)
        self.dirty = False
        self.selected_id: str | None = None
        self.model_map: dict[str, str] = {}    # 模型A 的 id→真實分類；未選/未掃→空
        self.model_map2: dict[str, str] = {}   # 模型B（對比用）的 id→真實分類
        root.title("拒絕資料集編輯器")
        root.geometry("1000x640")
        self._build()
        self._refresh_tree()
        self._refresh_stats()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI 版面 ────────────────────────────────────────────────────────────
    def _build(self):
        top = ttk.Frame(self.root, padding=(6, 6, 6, 2))
        top.pack(fill="x")
        ttk.Button(top, text="開啟…", command=self._open).pack(side="left")
        ttk.Button(top, text="新建…", command=self._new).pack(side="left", padx=4)
        ttk.Button(top, text="存檔", command=self._save).pack(side="left")
        self.path_var = tk.StringVar()
        ttk.Label(top, textvariable=self.path_var).pack(side="left", padx=10)

        # 模型分類依據自成一整列：可選兩個模型（A/B）做對比，索引表各顯示一欄真實分類，
        # 下方統計顯示兩模型差異（例如 A 拒絕但 B 順從＝B 比 A 多越獄了哪些題）。
        # 獨立一列，不管視窗寬度/路徑長度/DPI 縮放都一定看得到、選得到。
        opts = ["(未選)"] + available_models()
        top2 = ttk.Frame(self.root, padding=(6, 0, 6, 6))
        top2.pack(fill="x")
        ttk.Label(top2, text="模型A：").pack(side="left", padx=(0, 2))
        self.model_box = ttk.Combobox(top2, state="readonly", width=34, values=opts,
                                      postcommand=self._refresh_model_options)
        self.model_box.set("(未選)")
        self.model_box.pack(side="left")
        self.model_box.bind("<<ComboboxSelected>>", self._on_model_change)
        ttk.Label(top2, text="　模型B(對比)：").pack(side="left", padx=(0, 2))
        self.model_box2 = ttk.Combobox(top2, state="readonly", width=34, values=opts,
                                       postcommand=self._refresh_model_options)
        self.model_box2.set("(未選)")
        self.model_box2.pack(side="left")
        self.model_box2.bind("<<ComboboxSelected>>", self._on_model_change)
        ttk.Label(top2, text="（右表 A分類/B分類 欄顯示各模型實跑的 refused/complied）",
                  foreground="#666").pack(side="left", padx=8)

        mid = ttk.Panedwindow(self.root, orient="horizontal")
        mid.pack(fill="both", expand=True, padx=6)

        left = ttk.Frame(mid)
        cols = ("id", "group", "model_cls", "model_cls2", "domain", "text")
        headers = {"id": "id", "group": "預期group", "model_cls": "A分類",
                   "model_cls2": "B分類", "domain": "domain", "text": "text"}
        self.tree = ttk.Treeview(left, columns=cols, show="headings", selectmode="browse")
        for c, w in zip(cols, (80, 80, 80, 80, 100, 330)):
            self.tree.heading(c, text=headers[c])
            self.tree.column(c, width=w, anchor="w")
        vsb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        mid.add(left, weight=3)

        right = ttk.Frame(mid, padding=6)
        self._field_text = self._add_text(right, "提示詞 text", height=6)
        self._field_domain = self._add_entry(right, "類別 domain", "WWII")
        ttk.Label(right, text="group").pack(anchor="w")
        self._field_group = ttk.Combobox(right, values=list(GROUPS), state="readonly")
        self._field_group.set("refused")
        self._field_group.pack(fill="x", pady=(0, 6))
        self._field_lang = self._add_entry(right, "語言 lang", "en")
        self._field_note = self._add_entry(right, "備註 note", "")
        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="新增", command=self._add).pack(side="left")
        ttk.Button(btns, text="更新所選", command=self._update).pack(side="left", padx=4)
        ttk.Button(btns, text="刪除所選", command=self._delete).pack(side="left")
        ttk.Button(btns, text="清空表單", command=self._clear_form).pack(side="left", padx=4)
        mid.add(right, weight=2)

        bottom = ttk.LabelFrame(self.root, text="統計", padding=6)
        bottom.pack(fill="x", padx=6, pady=6)
        self.stats_var = tk.StringVar()
        ttk.Label(bottom, textvariable=self.stats_var, justify="left",
                  foreground="#333").pack(anchor="w")
        self.modelstats_var = tk.StringVar()
        ttk.Label(bottom, textvariable=self.modelstats_var, justify="left",
                  foreground="#1a5").pack(anchor="w")

    def _add_entry(self, parent, label, default):
        ttk.Label(parent, text=label).pack(anchor="w")
        var = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=var).pack(fill="x", pady=(0, 6))
        return var

    def _add_text(self, parent, label, height):
        ttk.Label(parent, text=label).pack(anchor="w")
        txt = tk.Text(parent, height=height, wrap="word")
        txt.pack(fill="x", pady=(0, 6))
        return txt

    # ── 資料操作 ───────────────────────────────────────────────────────────
    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        for e in self.entries:
            self.tree.insert("", "end", iid=e["id"],
                             values=(e["id"], e.get("group", ""),
                                     self.model_map.get(e["id"], ""),
                                     self.model_map2.get(e["id"], ""),
                                     e.get("domain", ""),
                                     (e.get("text", "") or "").replace("\n", " ")[:80]))
        self.path_var.set(f"檔案：{self.path}　（{len(self.entries)} 筆{'　*未存檔' if self.dirty else ''}）")

    def _refresh_stats(self):
        s = compute_stats(self.entries)
        lines = [f"共 {s['total']} 筆",
                 f"group: {s['group']}",
                 f"domain: {s['domain']}",
                 f"lang: {s['lang']}"]
        for w in s["warnings"]:
            lines.append(f"[注意] {w}")
        self.stats_var.set("　|　".join(lines[:4]) + ("\n" + "\n".join(lines[4:]) if len(lines) > 4 else ""))

    def _form(self) -> dict:
        return {"text": self._field_text.get("1.0", "end").strip(),
                "domain": self._field_domain.get().strip(),
                "group": self._field_group.get().strip(),
                "lang": self._field_lang.get().strip(),
                "note": self._field_note.get().strip()}

    def _fill_form(self, e: dict):
        self._field_text.delete("1.0", "end"); self._field_text.insert("1.0", e.get("text", ""))
        self._field_domain.set(e.get("domain", ""))
        self._field_group.set(e.get("group", "refused"))
        self._field_lang.set(e.get("lang", "en"))
        self._field_note.set(e.get("note", ""))

    def _clear_form(self):
        self.selected_id = None
        self.tree.selection_remove(self.tree.selection())
        self._field_text.delete("1.0", "end")
        self._field_domain.set("WWII"); self._field_group.set("refused")
        self._field_lang.set("en"); self._field_note.set("")

    def _on_select(self, _evt=None):
        sel = self.tree.selection()
        if not sel:
            return
        self.selected_id = sel[0]
        e = next((x for x in self.entries if x["id"] == self.selected_id), None)
        if e:
            self._fill_form(e)

    def _refresh_model_options(self):
        """下拉展開前重讀 registry，剛 pack/regroup 的模型不必重開 GUI 就會出現。"""
        opts = ["(未選)"] + available_models()
        self.model_box["values"] = opts
        self.model_box2["values"] = opts

    def _on_model_change(self, _evt=None):
        a, b = self.model_box.get(), self.model_box2.get()
        self.model_map  = {} if a in ("", "(未選)") else model_group_map(a)
        self.model_map2 = {} if b in ("", "(未選)") else model_group_map(b)
        self._refresh_tree()
        self._refresh_model_stats()

    def _refresh_model_stats(self):
        """顯示 A/B 兩模型的真實分類統計與差異（B 相對 A 多越獄/回退了哪些題）。"""
        a, b = self.model_box.get(), self.model_box2.get()
        parts = []
        if self.model_map:
            ra = sum(1 for v in self.model_map.values() if v == "refused")
            parts.append(f"A[{a}]：拒絕 {ra} / 順從 {len(self.model_map) - ra}（{len(self.model_map)} 題有分類）")
        if self.model_map2:
            rb = sum(1 for v in self.model_map2.values() if v == "refused")
            parts.append(f"B[{b}]：拒絕 {rb} / 順從 {len(self.model_map2) - rb}（{len(self.model_map2)} 題有分類）")
        if self.model_map and self.model_map2:
            common = set(self.model_map) & set(self.model_map2)
            a_ref_b_comp = sum(1 for i in common if self.model_map[i] == "refused" and self.model_map2[i] == "complied")
            a_comp_b_ref = sum(1 for i in common if self.model_map[i] == "complied" and self.model_map2[i] == "refused")
            both_ref     = sum(1 for i in common if self.model_map[i] == "refused" and self.model_map2[i] == "refused")
            both_comp    = sum(1 for i in common if self.model_map[i] == "complied" and self.model_map2[i] == "complied")
            parts.append(f"對比（{len(common)} 題共有）：A拒→B順(B多越獄) {a_ref_b_comp}　"
                         f"A順→B拒(B回退) {a_comp_b_ref}　皆拒 {both_ref}　皆順 {both_comp}")
        self.modelstats_var.set("\n".join(parts))

    def _add(self):
        f = self._form()
        if not f["text"]:
            messagebox.showwarning("無法新增", "提示詞 text 不能空白。"); return
        if f["group"] not in GROUPS:
            messagebox.showwarning("無法新增", f"group 必須是 {GROUPS}。"); return
        eid = next_id(self.entries)
        self.entries.append({"id": eid, "pair_id": eid, **f})
        self.dirty = True
        self._refresh_tree(); self._refresh_stats()
        self.tree.see(eid); self.tree.selection_set(eid)

    def _update(self):
        if not self.selected_id:
            messagebox.showinfo("更新", "請先在左邊索引選一筆。"); return
        e = next((x for x in self.entries if x["id"] == self.selected_id), None)
        if not e:
            return
        f = self._form()
        if not f["text"]:
            messagebox.showwarning("無法更新", "提示詞 text 不能空白。"); return
        if f["group"] not in GROUPS:
            messagebox.showwarning("無法更新", f"group 必須是 {GROUPS}。"); return
        e.update(f)
        self.dirty = True
        self._refresh_tree(); self._refresh_stats()
        self.tree.selection_set(e["id"])

    def _delete(self):
        if not self.selected_id:
            messagebox.showinfo("刪除", "請先在左邊索引選一筆。"); return
        self.entries = [x for x in self.entries if x["id"] != self.selected_id]
        self.dirty = True
        self._clear_form(); self._refresh_tree(); self._refresh_stats()

    # ── 檔案 ───────────────────────────────────────────────────────────────
    def _open(self):
        p = filedialog.askopenfilename(title="開啟資料集",
                                       filetypes=[("JSONL", "*.jsonl"), ("全部", "*.*")])
        if not p:
            return
        if self.dirty and not self._confirm_discard():
            return
        self.path = Path(p); self.entries = load(self.path); self.dirty = False
        self._clear_form(); self._refresh_tree(); self._refresh_stats()

    def _new(self):
        p = filedialog.asksaveasfilename(title="新建資料集（檔名建議以 km_ 開頭）",
                                         defaultextension=".jsonl",
                                         initialfile="km_refusal_western.jsonl",
                                         filetypes=[("JSONL", "*.jsonl")])
        if not p:
            return
        if self.dirty and not self._confirm_discard():
            return
        self.path = Path(p); self.entries = load(self.path) if Path(p).exists() else []
        self.dirty = False
        self._clear_form(); self._refresh_tree(); self._refresh_stats()

    def _save(self):
        save(self.path, self.entries); self.dirty = False
        self._refresh_tree()
        messagebox.showinfo("存檔", f"已存 {len(self.entries)} 筆到\n{self.path}")

    def _confirm_discard(self) -> bool:
        return messagebox.askyesno("未存檔", "有未存檔的變更，確定放棄？")

    def _on_close(self):
        if self.dirty and messagebox.askyesno("離開前存檔", "有未存檔的變更，要先存嗎？"):
            self._save()
        self.root.destroy()


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    root = tk.Tk()
    App(root, path)
    root.mainloop()


if __name__ == "__main__":
    main()

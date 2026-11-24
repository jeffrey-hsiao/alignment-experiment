#!/usr/bin/env python
"""
pipelines/train_gui.py

統一訓練介面：`python run.py`（不加任何子命令）開的就是這個。最上面的「訓練方式」
下拉列出 MLOps/train/scripts/train_*.py 的所有方法，選哪個就換成哪個的面板：

  - km_cos   → km_cos_gui.build_panel（人格誘發專用面板，原本的預設介面）
  - prune_v2 → prune_v2_gui.build_panel（拒絕移除的掃描參數面板）
  - 其他     → 通用面板（dpo / sft / selfplay / arena / prune / vector / empty …）：
               選設定檔 → 直接編輯內容 → 發射 `run.py train <method> --config <名稱>`；
               需要資料的方法（NEEDS_DATA）多一格資料版本；prune 系列多一格 system prompt；
               下方可對既有 run 接續（--resume）或二次訓練（continuetrain）。

新增一個 train_<method>.py 不必改這裡：自動出現在下拉，走通用面板。

用法：
    python run.py                     （預設選 km_cos）
    python pipelines/train_gui.py [method]
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MLOPS = REPO / "MLOps"
from train_catalog import CONFIGS_DIR as CONFIGS  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from _gui_common import (set_text, mark_launch, make_poller, launch, launch_resume,  # noqa: E402
                         launch_continuetrain, list_checkpoints, load_run_summary,
                         load_km_state, sync_cells,
                         system_from_config_text, set_system_in_config_text)
# 方法/設定檔/預設的查詢與 run.py 共用同一份（MLOps/train_catalog.py；_gui_common 已把 MLOps 放上 sys.path）
from train_catalog import (available_methods as _all_methods, get_default_config,  # noqa: E402
                           get_data_default, list_configs, needs_data, set_default_config)
from model_registry import iter_run_summaries  # noqa: E402

DEFAULT_METHOD = "km_cos"
PRUNE_FAMILY = {"prune", "prune_v2", "vector"}   # 吃 system / system_file 的方法
# 各方法一句話說明（沒列到的方法照樣能用，只是沒說明）
METHOD_NOTES = {
    "km_cos":   "行為編輯（模型調整）：人格誘發，對比 target/base system",
    "prune":    "神經元裁減 v1（解除對齊）：fixed 或 km_search 粗測+細搜",
    "prune_v2": "神經元裁減 v2（解除對齊）：單調 T/K 掃描，依拒絕翻轉率步進",
    "vector":   "激活偏移版 prune_v2（解除對齊）：a→a+δ，由上往下搜尋",
    "dpo":      "DPO 偏好訓練（LoRA）；objective=degrade 為劣化實驗，normal 為一般",
    "sft":      "SFT 監督微調（LoRA）；objective 同 dpo",
    "selfplay": "卡牌博弈 self-play：LLM agent 對戰 MCT 對手（不吃外部資料）",
    "arena":    "卡牌博弈競技場：MCCFR/MCT 傳統 agent 互打（不吃外部資料）",
    "empty":    "冒煙測試：走完整 callback 生命週期但不真的訓練",
}


# ── 純函式（不碰 tkinter，可單獨測）───────────────────────────────────────────────

def available_methods() -> list:
    """所有訓練方法（train_catalog），km_cos 排第一（預設面板），其餘照字母。"""
    return sorted(_all_methods(), key=lambda m: (m != DEFAULT_METHOD, m))


def method_runs(method: str) -> list:
    """experiments 裡此方法的 run（含沒有 checkpoint 的，才能接續），最新在前。"""
    out = [{"run_id": d.name, "status": s.get("status", "?"), "config": s.get("config", "?"),
            "started": s.get("started_at") or ""}
           for d, s in iter_run_summaries() if s.get("method") == method]
    out.sort(key=lambda r: r["started"], reverse=True)
    return out


def read_progress(run_dir: Path) -> tuple:
    """通用進度 → (顯示行, status)：狀態、checkpoint 數、sync 格數、metrics.jsonl 最後一筆數字。"""
    status = load_run_summary(run_dir).get("status", "?")
    lines = [f"run: {run_dir.name}", f"狀態: {status}"]
    n_ck = len(list_checkpoints(run_dir))
    if n_ck:
        lines.append(f"checkpoint 數: {n_ck}")
    st = load_km_state(run_dir)
    if st:
        lines.append(f"已完成 sync 格: {len(sync_cells(st))}")
    mp = run_dir / "metrics.jsonl"
    if mp.exists():
        try:
            import json
            last = [l for l in mp.read_text(encoding="utf-8").splitlines() if l.strip()][-1]
            nums = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in json.loads(last).items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)}
            lines.append("最新 metrics: " + "  ".join(f"{k}={v}" for k, v in list(nums.items())[:12]))
        except Exception:
            pass
    return lines, status


# ── 通用面板 ────────────────────────────────────────────────────────────────────

def build_generic_panel(parent, root, method: str):
    import tkinter as tk
    from tkinter import ttk, messagebox, simpledialog

    state = {"run_dir": None, "config_stem": None}
    frm = ttk.Frame(parent, padding=10)
    is_prune = method in PRUNE_FAMILY
    with_data = needs_data(method)

    ttk.Label(frm, text=f"{method}：{METHOD_NOTES.get(method, '')}",
              font=("", 10, "bold")).pack(anchor="w")
    ttk.Label(frm, text="選設定檔 → 下方直接改內容（key = value）→ 發射。發射前會自動存檔。",
              foreground="#666").pack(anchor="w", pady=(0, 6))

    # 設定檔選擇
    crow = ttk.Frame(frm); crow.pack(fill="x", pady=2)
    ttk.Label(crow, text="設定檔").pack(side="left")
    cfg_var = tk.StringVar()
    cfg_box = ttk.Combobox(crow, textvariable=cfg_var, values=list_configs(method),
                           state="readonly", width=36)
    cfg_box.pack(side="left", padx=4)
    dflt = get_default_config(method) or ""
    ttk.Label(crow, text=f"（預設：{dflt or '未設'}）", foreground="#666").pack(side="left", padx=4)

    editor = tk.Text(frm, height=14, wrap="none", undo=True)

    system_box = None
    if is_prune:
        ttk.Label(frm, text="system prompt（留空＝模型原廠身分；__none__＝不帶 system；"
                  "發射時寫進設定檔的 system_file）").pack(anchor="w", pady=(6, 0))
        system_box = tk.Text(frm, height=2, wrap="word")
        system_box.pack(fill="x", pady=(2, 4))

    def load_cfg(*_):
        stem = cfg_var.get().strip()
        if not stem:
            return
        text = (CONFIGS / method / f"{stem}.txt").read_text(encoding="utf-8")
        editor.delete("1.0", "end"); editor.insert("1.0", text)
        editor.edit_modified(False)
        if system_box is not None:
            system_box.delete("1.0", "end"); system_box.insert("1.0", system_from_config_text(text))

    cfg_box.bind("<<ComboboxSelected>>", load_cfg)

    def save_as(stem: str) -> bool:
        stem = re.sub(r"[^\w一-鿿.-]+", "_", stem.strip())
        if not stem or stem == "default":
            messagebox.showwarning("名稱不合法", "請換一個設定檔名稱（不能是 default）。"); return False
        text = editor.get("1.0", "end").rstrip() + "\n"
        if system_box is not None:
            text = set_system_in_config_text(text, method, stem, system_box.get("1.0", "end"))
        (CONFIGS / method).mkdir(parents=True, exist_ok=True)
        (CONFIGS / method / f"{stem}.txt").write_text(text, encoding="utf-8")
        cfg_box.configure(values=list_configs(method))
        cfg_var.set(stem)
        editor.delete("1.0", "end"); editor.insert("1.0", text)
        editor.edit_modified(False)
        return True

    def do_save():
        if not cfg_var.get().strip():
            return do_save_as()
        if save_as(cfg_var.get()):
            status_var.set(f"已儲存 configs/{method}/{cfg_var.get()}.txt")

    def do_save_as():
        name = simpledialog.askstring("另存設定檔", f"名稱（存成 configs/{method}/<名稱>.txt）：",
                                      initialvalue=cfg_var.get().strip(), parent=root)
        if name and save_as(name):
            status_var.set(f"已另存 configs/{method}/{cfg_var.get()}.txt")

    def do_set_default():
        stem = cfg_var.get().strip()
        if not stem:
            return
        set_default_config(method, stem)
        status_var.set(f"{method} 的預設設定檔改為 {stem}")

    ttk.Button(crow, text="儲存", command=do_save).pack(side="left", padx=2)
    ttk.Button(crow, text="另存為…", command=do_save_as).pack(side="left", padx=2)
    ttk.Button(crow, text="設為預設", command=do_set_default).pack(side="left", padx=2)

    editor.pack(fill="both", expand=True, pady=(4, 4))

    versions_var = tk.StringVar(value=" ".join(get_data_default(method) or []))
    if with_data:
        drow = ttk.Frame(frm); drow.pack(fill="x", pady=2)
        ttk.Label(drow, text="資料版本（空白分隔，如 v1 v2；留空＝data default）").pack(side="left")
        ttk.Entry(drow, textvariable=versions_var, width=24).pack(side="left", padx=4)

    # 發射
    status_var = tk.StringVar(value="（尚未發射）")
    brow = ttk.Frame(frm); brow.pack(fill="x", pady=4)

    def do_launch():
        stem = cfg_var.get().strip()
        if not stem:
            messagebox.showwarning("未選設定檔", "請先選一份設定檔（或另存一份新的）。"); return
        if not save_as(stem):          # 欄位＝唯一真相：發射前一律存檔
            return
        if with_data and not versions_var.get().strip() and not get_data_default(method):
            messagebox.showwarning("缺資料版本", f"{method} 需要資料版本，請填寫或先設 data default。"); return
        mark_launch(state, stem)
        launch(method, stem, versions_var.get())
        status_var.set(f"已發射 {method}：config={stem}（訓練在新視窗）")
        root.after(4000, poll)

    ttk.Button(brow, text=f"發射 {method}", command=do_launch).pack(side="left")

    # 接續 / 二次訓練
    rf = ttk.LabelFrame(frm, text="既有 run：接續（--resume，把中斷的同一個 run 跑完）／"
                        "二次訓練（continuetrain，以它為底模開新 run）", padding=6)
    rf.pack(fill="x", pady=(6, 2))
    runs = method_runs(method)
    fmt = lambda r: f"{r['run_id']:<34} {r['status']:<11} config={r['config']}"
    run_var = tk.StringVar()
    run_box = ttk.Combobox(rf, textvariable=run_var, values=[fmt(r) for r in runs],
                           state="readonly", width=70)
    run_box.grid(row=0, column=0, columnspan=4, sticky="w")
    ttk.Label(rf, text="checkpoint").grid(row=1, column=0, sticky="w", pady=(4, 0))
    ck_var = tk.StringVar(value="（最佳）")
    ck_box = ttk.Combobox(rf, textvariable=ck_var, values=["（最佳）"], state="readonly", width=16)
    ck_box.grid(row=1, column=1, sticky="w", pady=(4, 0))

    def pick_run(*_):
        sel = run_var.get().split()
        if sel:
            ck_box.configure(values=["（最佳）"] + list_checkpoints(sel[0]))
            ck_var.set("（最佳）")

    run_box.bind("<<ComboboxSelected>>", pick_run)

    def do_resume():
        sel = run_var.get().split()
        if not sel:
            return
        if not messagebox.askyesno("接續", f"接續 {sel[0]}（沿用當初的 config/超參數）？"):
            return
        launch_resume(method, sel[0])
        status_var.set(f"已發射接續：{sel[0]}")

    def do_continue():
        sel = run_var.get().split()
        if not sel:
            return
        stem = cfg_var.get().strip()
        ck = "" if ck_var.get() == "（最佳）" else ck_var.get()
        if not messagebox.askyesno("二次訓練 continuetrain",
                                   f"以 {sel[0]}（checkpoint={ck or '最佳'}）為底模續訓？\n"
                                   f"設定檔＝{stem or '（方法預設）'}"):
            return
        if stem and not save_as(stem):
            return
        launch_continuetrain(sel[0], stem, ck)
        status_var.set(f"已發射 continuetrain：{sel[0]}")

    def reload_runs():
        runs[:] = method_runs(method)
        run_box.configure(values=[fmt(r) for r in runs])

    ttk.Button(rf, text="接續", command=do_resume).grid(row=1, column=2, padx=6, pady=(4, 0))
    ttk.Button(rf, text="二次訓練", command=do_continue).grid(row=1, column=3, pady=(4, 0))
    ttk.Button(rf, text="重整清單", command=reload_runs).grid(row=1, column=4, padx=6, pady=(4, 0))

    ttk.Label(frm, textvariable=status_var, foreground="#036").pack(anchor="w", pady=(6, 2))
    prog = tk.Text(frm, height=6, wrap="word", state="disabled"); prog.pack(fill="x")

    def render(rd):
        lines, status = read_progress(rd)
        set_text(prog, "\n".join(lines))
        return status

    poll = make_poller(root, state, status_var, render)

    ttk.Button(brow, text="重新整理進度", command=poll).pack(side="left", padx=6)

    # 預設帶入方法預設設定檔
    if dflt and dflt in list_configs(method):
        cfg_var.set(dflt); load_cfg()
    return frm


# ── 主視窗 ──────────────────────────────────────────────────────────────────────

def build_method_panel(parent, root, method: str):
    if method == "km_cos":
        from km_cos_gui import build_panel
        return build_panel(parent, root)
    if method == "prune_v2":
        from prune_v2_gui import build_panel
        return build_panel(parent, root)
    return build_generic_panel(parent, root, method)


def main(initial: str = DEFAULT_METHOD):
    import tkinter as tk
    from tkinter import ttk, messagebox

    root = tk.Tk()
    root.title("MLOps 訓練介面")
    root.geometry("900x820")

    top = ttk.Frame(root, padding=(10, 8, 10, 0)); top.pack(fill="x")
    ttk.Label(top, text="訓練方式", font=("", 11, "bold")).pack(side="left")
    methods = available_methods()
    method_var = tk.StringVar(value=initial if initial in methods else methods[0])
    box = ttk.Combobox(top, textvariable=method_var, values=methods, state="readonly", width=14)
    box.pack(side="left", padx=6)
    note_var = tk.StringVar()
    ttk.Label(top, textvariable=note_var, foreground="#666").pack(side="left", padx=6)
    ttk.Separator(root).pack(fill="x", pady=(8, 0))

    body = ttk.Frame(root); body.pack(fill="both", expand=True)
    panels: dict = {}   # 每個方法的面板只建一次，切回來保留原欄位與進度監看
    current = {"m": None}

    def show(*_):
        m = method_var.get()
        if current["m"] == m:
            return
        if current["m"] in panels:
            panels[current["m"]].pack_forget()
        if m not in panels:
            try:
                panels[m] = build_method_panel(body, root, m)
            except Exception as e:
                messagebox.showerror("面板建立失敗", f"{m}：{e}")
                method_var.set(current["m"] or DEFAULT_METHOD)
                return
        panels[m].pack(fill="both", expand=True)
        current["m"] = m
        note_var.set(METHOD_NOTES.get(m, ""))

    box.bind("<<ComboboxSelected>>", show)
    show()
    root.mainloop()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_METHOD)

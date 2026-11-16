#!/usr/bin/env python
"""
pipelines/km_cos_gui.py

km_cos（cosplay 行為編輯）的圖形介面:從人格庫選/編要誘發的人格(target system)+
對照 system + 種子題 + 掃描參數 → 一鍵發射 km_cos run → 監看進度
(格數 / 崩壞 / cos 分數 / 最佳格)。

人格庫:data/personas/*.txt,每個 .txt 一個人格(檔名＝人格名)。上方下拉可載入已存
人格,「另存為…」把目前這段存回庫裡,下次開介面直接選,不必重打。

搜尋用 prune_v2 的 gen-2 單調 T/K 掃描(run_sweep),步進訊號是對比相關性 cos。
掃描參數 t_max_ratio / sweep_steps / k_decay 預設同步自 prune_v2 參數最佳化
(0.6 / 8 / 0.75)。進格門檻只有一個欄位 cos_low:填了＝直接當絕對門檻;留空＝由 rank
時的 cos 基準自動定標(位置由 KmCosConfig.cos_low_frac 控制,預設地板)。

這是 km_cos 唯一的操作介面,也是 `python run.py`(不加任何子命令)預設開啟的介面。

用法:
    python pipelines/km_cos_gui.py
    或  python run.py            (無旗標時預設開這個)
"""
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MLOPS = REPO / "MLOps"
KMCOS_CFG_DIR = MLOPS / "train" / "configs" / "km_cos"
PERSONA_DIR = REPO / "data" / "personas"   # 人格 prompt 庫：每個 .txt 一個人格，檔名＝人格名
DEFAULT_SEEDS = "data/seeds/km_cos_persona_prompts.jsonl"  # 人格誘發專用（高對比）


def list_personas() -> list:
    """人格庫裡的人格名（.txt 檔名去副檔名），排序。"""
    if not PERSONA_DIR.exists():
        return []
    return sorted(p.stem for p in PERSONA_DIR.glob("*.txt"))


def load_persona(name: str) -> str:
    """讀某個已存人格的 target system 內容。"""
    return (PERSONA_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


def save_persona(name: str, text: str) -> Path:
    """把目前的人格 prompt 存進庫，回傳路徑。"""
    PERSONA_DIR.mkdir(parents=True, exist_ok=True)
    p = PERSONA_DIR / f"{name}.txt"
    p.write_text(text.strip() + "\n", encoding="utf-8")
    return p

sys.path.insert(0, str(Path(__file__).parent))
from _gui_common import existing_run_names, newest_run_dir, launch, spin, set_text  # 共用發射/UI 小工具
try:
    from refusal_data_editor import available_models  # 共用 registry 讀取
except Exception:
    def available_models():
        return ["Qwen/Qwen2.5-1.5B-Instruct"]


def write_run_files(name: str, target_system: str, base_system: str, model: str,
                    seeds: str, t_max_ratio: float, sweep_steps: int, k_decay: float,
                    cos_low, sign: str,
                    control_repeats: int, gen_max_tokens: int) -> Path:
    """寫出 persona 檔 + km_cos config 檔,回傳 config 路徑。純檔案操作,可單獨測。

    cos_low：填了＝直接當絕對進格門檻（略過自動定標）；留空（""／None）＝不寫進 config，
    由 KmCosConfig 預設走自動定標（cos_low_frac，預設地板）。"""
    KMCOS_CFG_DIR.mkdir(parents=True, exist_ok=True)
    persona = KMCOS_CFG_DIR / f"persona_{name}.txt"
    persona.write_text(target_system.strip() + "\n", encoding="utf-8")
    base_path = KMCOS_CFG_DIR / f"persona_base_{name}.txt"
    # base 留空＝對照組不帶任何 system（空檔），cos 對比＝人格 vs 無提示詞
    base_path.write_text(base_system.strip() + "\n", encoding="utf-8")
    opt = lambda key, val: (f"{key} = {val}\n"
                            if str(val).strip() not in ("", "None") else "")
    thresh_line = opt("cos_low             ", cos_low)
    cfg = KMCOS_CFG_DIR / f"{name}.txt"
    cfg.write_text(
        f"""# km_cos(cosplay 行為編輯)— 由 GUI 產生
seeds               = {seeds}
model_name          = {model}
target_system_file  = {persona.relative_to(REPO).as_posix()}
base_system_file    = {base_path.relative_to(REPO).as_posix()}
sign                = {sign}
reverse_type        = true
# gen-2 單調 T/K 掃描（run_sweep）；步進訊號＝對比相關性 cos
# 門檻：cos_low 填了＝絕對門檻；留空＝由 rank 基準自動定標（見 KmCosConfig.cos_low_frac）
t_max_ratio         = {t_max_ratio}
sweep_steps         = {sweep_steps}
k_decay             = {k_decay}
{thresh_line}control_repeats     = {control_repeats}
gen_max_tokens      = {gen_max_tokens}
""", encoding="utf-8")
    return cfg


def read_progress(run_dir: Path) -> dict:
    """只讀數字,不讀題目/回答原文。"""
    out = {"status": "?", "cells": 0, "best": None, "bounds": None}
    sm = run_dir / "run_summary.json"
    if sm.exists():
        try:
            out["status"] = json.loads(sm.read_text(encoding="utf-8")).get("status", "?")
        except Exception:
            pass
    ks = run_dir / "km_state.json"
    if ks.exists():
        try:
            s = json.loads(ks.read_text(encoding="utf-8"))
            out["bounds"] = s.get("bounds")
            sync = [c for c in s.get("cells", []) if c.get("kind") == "sync"]
            out["cells"] = len(sync)
            viable = [c for c in sync if c.get("auto_collapse", 0) < c.get("n", 1)] or sync
            if viable:
                b = max(viable, key=lambda c: c.get("cos", -9))
                out["best"] = {"k": b["k"], "m": b["m"], "cos": b.get("cos"),
                               "collapse": b.get("auto_collapse"), "n": b.get("n")}
        except Exception:
            pass
    return out


def main():
    import tkinter as tk
    from tkinter import ttk, messagebox, simpledialog

    root = tk.Tk()
    root.title("km_cos — 讓模型 cosplay 一個人格")
    root.geometry("860x760")
    state = {"run_dir": None, "config_stem": None}

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="要誘發的人格（target system）", font=("", 10, "bold")).pack(anchor="w")
    ttk.Label(frm, text="例：你是一隻貓娘，說話尾端會加「喵」，撒嬌、親暱。",
              foreground="#666").pack(anchor="w")

    # 人格庫：選已存的人格載入，或把目前這段存起來（存於 data/personas/*.txt）
    lib = ttk.Frame(frm); lib.pack(fill="x", pady=(2, 0))
    ttk.Label(lib, text="已存人格").pack(side="left")
    persona_var = tk.StringVar()
    persona_box = ttk.Combobox(lib, textvariable=persona_var, values=list_personas(),
                               state="readonly", width=24)
    persona_box.pack(side="left", padx=4)

    target = tk.Text(frm, height=5, wrap="word")

    def do_load_persona(*_):
        name = persona_var.get().strip()
        if not name:
            return
        target.delete("1.0", "end"); target.insert("1.0", load_persona(name))
    persona_box.bind("<<ComboboxSelected>>", do_load_persona)

    def do_save_persona():
        text = target.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("沒有內容", "請先在下方填好人格再儲存。"); return
        name = simpledialog.askstring("儲存人格", "人格名稱（存成 data/personas/<名稱>.txt）：",
                                      initialvalue=persona_var.get().strip(), parent=root)
        if not name or not name.strip():
            return
        save_persona(name.strip(), text)
        persona_box["values"] = list_personas()   # 立刻刷新清單
        persona_var.set(name.strip())
        messagebox.showinfo("已儲存", f"人格「{name.strip()}」已存入人格庫。")

    ttk.Button(lib, text="載入", command=do_load_persona).pack(side="left", padx=2)
    ttk.Button(lib, text="另存為…", command=do_save_persona).pack(side="left", padx=2)

    target.pack(fill="x", pady=(2, 8))

    ttk.Label(frm, text="對照 system（base，留空＝不帶 system／無提示詞對照）").pack(anchor="w")
    base = tk.Text(frm, height=2, wrap="word"); base.pack(fill="x", pady=(2, 8))

    row = ttk.Frame(frm); row.pack(fill="x", pady=2)
    ttk.Label(row, text="模型").grid(row=0, column=0, sticky="w")
    model_box = ttk.Combobox(row, values=available_models(), state="readonly", width=34)
    model_box.set(available_models()[0] if available_models() else "")
    model_box.grid(row=0, column=1, padx=4, sticky="w")
    ttk.Label(row, text="種子題").grid(row=0, column=2, sticky="w", padx=(12, 0))
    seeds_var = tk.StringVar(value=DEFAULT_SEEDS)
    ttk.Entry(row, textvariable=seeds_var, width=36).grid(row=0, column=3, padx=4)

    prow = ttk.Frame(frm); prow.pack(fill="x", pady=6)
    t_max_ratio = spin(prow, "t_max_ratio", 0.6, 0)
    sweep_steps = spin(prow, "sweep_steps", 8, 1)
    k_decay = spin(prow, "k_decay", 0.75, 2)

    prow2 = ttk.Frame(frm); prow2.pack(fill="x", pady=2)
    sign_var = tk.StringVar(value="both")
    ttk.Label(prow2, text="sign").grid(row=0, column=0, sticky="e", padx=(8, 2))
    ttk.Combobox(prow2, textvariable=sign_var, values=["both", "positive", "negative"],
                 state="readonly", width=9).grid(row=0, column=1, sticky="w")
    reps = spin(prow2, "隨機對照", 5, 1)
    cos_low = spin(prow2, "cos_low（留空=自動；填了=絕對門檻）", "", 2, width=14)

    status_var = tk.StringVar(value="（尚未發射）")
    ttk.Label(frm, textvariable=status_var, foreground="#036", font=("", 10)).pack(anchor="w", pady=(8, 2))
    prog = tk.Text(frm, height=14, wrap="word", state="disabled"); prog.pack(fill="both", expand=True)

    def set_prog(text):
        set_text(prog, text)

    def do_launch():
        tgt = target.get("1.0", "end").strip()
        if not tgt:
            messagebox.showwarning("缺人格", "請先填『要誘發的人格』。"); return
        if not (REPO / seeds_var.get()).exists():
            messagebox.showwarning("找不到種子題", seeds_var.get()); return
        stem = "km_cos_gui_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        state["before"] = existing_run_names()   # 啟動前快照，poll 用來認出新 run 目錄
        state["run_dir"] = None                  # 重置，否則第二次發射會沿用上一個 run 的進度
        try:
            write_run_files(stem, tgt, base.get("1.0", "end"), model_box.get(),
                            seeds_var.get(), float(t_max_ratio.get()), int(sweep_steps.get()),
                            float(k_decay.get()), cos_low.get().strip(),
                            sign_var.get(), int(reps.get()), 128)
            launch("km_cos", stem)
        except Exception as e:
            messagebox.showerror("發射失敗", str(e)); return
        state["config_stem"] = stem
        status_var.set(f"已發射：config={stem}（在新視窗跑，約數十分鐘～數小時）")
        root.after(4000, poll)

    def poll():
        stem = state["config_stem"]
        if not stem:
            return
        if state["run_dir"] is None:
            state["run_dir"] = newest_run_dir(state.get("before", set()), stem)
        rd = state["run_dir"]
        if rd is None:
            status_var.set("等待 run 目錄出現…"); root.after(5000, poll); return
        p = read_progress(rd)
        lines = [f"run: {rd.name}", f"狀態: {p['status']}", f"已完成 sync 格: {p['cells']}",
                 f"bounds: {p['bounds']}"]
        if p["best"]:
            b = p["best"]
            lines.append(f"目前最佳格: K={b['k']} M={b['m']}  cos={b['cos']}  "
                         f"崩壞={b['collapse']}/{b['n']}")
        lines.append("\n（cos 越高＝輸出越像目標人格；只顯示數字，不顯示題目/回答）")
        set_prog("\n".join(lines))
        status_var.set(f"{rd.name} — {p['status']}")
        if p["status"] not in ("completed", "failed"):
            root.after(8000, poll)

    btns = ttk.Frame(frm); btns.pack(fill="x", pady=6)
    ttk.Button(btns, text="發射 km_cos", command=do_launch).pack(side="left")
    ttk.Button(btns, text="重新整理進度", command=poll).pack(side="left", padx=6)

    root.mainloop()


if __name__ == "__main__":
    main()

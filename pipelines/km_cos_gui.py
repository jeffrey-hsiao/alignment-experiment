#!/usr/bin/env python
"""
pipelines/km_cos_gui.py

km_cos（cosplay 行為編輯）的圖形介面:設定要誘發的人格(target system)+ 對照
system + 種子題 + K/M 參數 → 一鍵發射 km_search 體系的 km_cos run → 監看進度
(格數 / 崩壞 / cos 分數 / 最佳格)。

這是 km_cos 唯一的操作介面,也是 `python run.py`(不加任何子命令)預設開啟的介面。

用法:
    python pipelines/km_cos_gui.py
    或  python run.py            (無旗標時預設開這個)
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MLOPS = REPO / "MLOps"
KMCOS_CFG_DIR = MLOPS / "train" / "configs" / "km_cos"
EXPERIMENTS = MLOPS / "experiments"
DEFAULT_SEEDS = "pipelines/data/km_cos_prompts.jsonl"
DEFAULT_BASE = "You are a helpful assistant."

sys.path.insert(0, str(Path(__file__).parent))
try:
    from refusal_data_editor import available_models  # 共用 registry 讀取
except Exception:
    def available_models():
        return ["Qwen/Qwen2.5-1.5B-Instruct"]


def write_run_files(name: str, target_system: str, base_system: str, model: str,
                    seeds: str, start_k: int, end_m: int, sign: str,
                    control_repeats: int, gen_max_tokens: int) -> Path:
    """寫出 persona 檔 + km_cos config 檔,回傳 config 路徑。純檔案操作,可單獨測。"""
    KMCOS_CFG_DIR.mkdir(parents=True, exist_ok=True)
    persona = KMCOS_CFG_DIR / f"persona_{name}.txt"
    persona.write_text(target_system.strip() + "\n", encoding="utf-8")
    base_path = KMCOS_CFG_DIR / f"persona_base_{name}.txt"
    base_path.write_text((base_system or DEFAULT_BASE).strip() + "\n", encoding="utf-8")
    cfg = KMCOS_CFG_DIR / f"{name}.txt"
    cfg.write_text(
        f"""# km_cos(cosplay 行為編輯)— 由 GUI 產生
seeds               = {seeds}
model_name          = {model}
target_system_file  = {persona.relative_to(REPO).as_posix()}
base_system_file    = {base_path.relative_to(REPO).as_posix()}
sign                = {sign}
reverse_type        = true
start_k             = {start_k}
end_m               = {end_m}
control_repeats     = {control_repeats}
gen_max_tokens      = {gen_max_tokens}
""", encoding="utf-8")
    return cfg


def launch(config_stem: str) -> None:
    subprocess.Popen([sys.executable, "run.py", "train", "km_cos", "--config", config_stem],
                     cwd=str(MLOPS))


def existing_run_names() -> set:
    """啟動前的 run 目錄名快照。"""
    return {p.name for p in EXPERIMENTS.iterdir() if p.is_dir()} if EXPERIMENTS.exists() else set()


def newest_run_dir(before: set, config_stem: str):
    """新命名（<基底模型>_NNN）不再含 config 名，無法用 config 名 glob。改成：在
    『啟動前快照』之後新出現的 run 目錄裡，挑 run_summary.config == 本次 GUI 產生的
    config_stem 的那個——用 summary 精準對應，避免抓到並行啟動的其他 run。"""
    if not EXPERIMENTS.exists():
        return None
    fresh = [p for p in EXPERIMENTS.iterdir() if p.is_dir() and p.name not in before]
    for p in sorted(fresh, key=lambda p: p.stat().st_mtime, reverse=True):
        sm = p / "run_summary.json"
        try:
            if sm.exists() and json.loads(sm.read_text(encoding="utf-8")).get("config") == config_stem:
                return p
        except Exception:
            continue
    return None


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
    from tkinter import ttk, messagebox

    root = tk.Tk()
    root.title("km_cos — 讓模型 cosplay 一個人格")
    root.geometry("860x720")
    state = {"run_dir": None, "config_stem": None}

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="要誘發的人格（target system）", font=("", 10, "bold")).pack(anchor="w")
    ttk.Label(frm, text="例：你是一隻貓娘，說話尾端會加「喵」，撒嬌、親暱。",
              foreground="#666").pack(anchor="w")
    target = tk.Text(frm, height=5, wrap="word"); target.pack(fill="x", pady=(2, 8))

    ttk.Label(frm, text="對照 system（base，留空＝通用助理）").pack(anchor="w")
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
    def spin(parent, label, default, col, width=8):
        ttk.Label(parent, text=label).grid(row=0, column=col * 2, sticky="e", padx=(8, 2))
        v = tk.StringVar(value=str(default))
        ttk.Entry(parent, textvariable=v, width=width).grid(row=0, column=col * 2 + 1, sticky="w")
        return v
    start_k = spin(prow, "start_k", 2048, 0)
    end_m = spin(prow, "end_m", 20000, 1)
    sign_var = tk.StringVar(value="both")
    ttk.Label(prow, text="sign").grid(row=0, column=4, sticky="e", padx=(8, 2))
    ttk.Combobox(prow, textvariable=sign_var, values=["both", "positive", "negative"],
                 state="readonly", width=9).grid(row=0, column=5, sticky="w")
    reps = spin(prow, "隨機對照", 5, 3)

    status_var = tk.StringVar(value="（尚未發射）")
    ttk.Label(frm, textvariable=status_var, foreground="#036", font=("", 10)).pack(anchor="w", pady=(8, 2))
    prog = tk.Text(frm, height=14, wrap="word", state="disabled"); prog.pack(fill="both", expand=True)

    def set_prog(text):
        prog.configure(state="normal"); prog.delete("1.0", "end"); prog.insert("1.0", text)
        prog.configure(state="disabled")

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
                            seeds_var.get(), int(start_k.get()), int(end_m.get()),
                            sign_var.get(), int(reps.get()), 128)
            launch(stem)
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

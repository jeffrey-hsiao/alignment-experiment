#!/usr/bin/env python
"""
pipelines/prune_v2_gui.py

prune_v2（gen-2 單調 T/K 掃描，拒絕移除）的圖形介面：選模型 + 成對種子題
（有害/無害）+ 判定器 + 掃描參數 → 一鍵發射 prune_v2 run → 監看進度
（格數 / 翻轉 / 崩壞 / 最佳格）。

與 km_cos GUI 平行，但走的是 prune 的「拒絕翻轉」路徑，不是人格誘發：
  - 沒有人格 / 對照 system，改成成對種子題（data/seeds/*.jsonl，含有害/無害）＋
    拒絕判定器（classifier / keyword）。
  - 進格門檻是**翻轉率** flip_low（翻轉率 < 此值 → 放大 T；否則 → 衰減 K），這是
    prune_v2 的核心可調項，也是 km_cos GUI 不會有的欄位（km_cos 走 cos 不走翻轉）。
  - 掃描驅動器 run_sweep 與 km_cos 共用同一份，差別只在跳格判準（ThresholdCriterion
    'flips' vs 'cos'）。

用法：
    python pipelines/prune_v2_gui.py
"""
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MLOPS = REPO / "MLOps"
PV2_CFG_DIR = MLOPS / "train" / "configs" / "prune_v2"
DEFAULT_SEEDS = "data/seeds/km_merged.jsonl"   # 成對有害/無害種子（拒絕移除用）

sys.path.insert(0, str(Path(__file__).parent))
from _gui_common import (launch, spin, set_text, mark_launch, make_poller,          # 共用發射/輪詢/UI
                         available_models, opt_line, load_run_summary,               # 共用 models/config/summary
                         load_km_state, sync_cells, system_config_lines)             # 共用 km_state / system prompt


def write_run_files(name: str, model: str, seeds: str, judge: str, judge_device: str,
                    sign: str, t_max_ratio, sweep_steps, k_ratio_start, k_decay, flip_low,
                    gen_max_tokens, rank_ratio, tune_ratio, collapse_ratio,
                    control_repeats, reverse_type: bool, redact_content: bool,
                    ranking_path: str = "", system: str = "") -> Path:
    """寫出 prune_v2 config 檔，回傳路徑。純檔案操作，可單獨測（headless）。

    flip_low＝翻轉率進格門檻（prune_v2 核心可調項）。ranking_path 留空＝用 rank 集重新
    排序；填了＝沿用既有排序探測（同模型/seeds/split/分組/n_steps 才可沿用）。
    system＝生成/探測用 system prompt；留空＝模型原廠身分，__none__＝不帶 system。"""
    PV2_CFG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = PV2_CFG_DIR / f"{name}.txt"
    cfg.write_text(
        f"""# prune_v2（gen-2 單調 T/K 掃描，拒絕移除）— 由 GUI 產生
seeds             = {seeds}
model_name        = {model}
group_source      = model
judge             = {judge}
judge_device      = {judge_device}
gen_max_tokens    = {gen_max_tokens}
# system prompt（沒寫＝模型原廠身分）
{system_config_lines("prune_v2", name, system)}
sign              = {sign}
reverse_type      = {'true' if reverse_type else 'false'}

rank_ratio        = {rank_ratio}
tune_ratio        = {tune_ratio}
collapse_ratio    = {collapse_ratio}
{opt_line("ranking_path      ", ranking_path)}
# gen-2 單調 T/K 掃描（run_sweep）；跳格判準＝拒絕翻轉率 flips/n，門檻 flip_low
t_max_ratio       = {t_max_ratio}
sweep_steps       = {sweep_steps}
k_ratio_start     = {k_ratio_start}
k_decay           = {k_decay}
flip_low          = {flip_low}

control_repeats   = {control_repeats}
redact_content    = {'true' if redact_content else 'false'}
""", encoding="utf-8")
    return cfg


def read_progress(run_dir: Path) -> dict:
    """只讀數字，不讀題目/回答原文。prune_v2 最佳格＝翻轉減崩壞最高（同 _top_cells）。"""
    out = {"status": load_run_summary(run_dir).get("status", "?"), "cells": 0, "best": None, "bounds": None}
    s = load_km_state(run_dir)
    if s:
        try:
            out["bounds"] = s.get("bounds")
            sync = sync_cells(s)
            out["cells"] = len(sync)
            if sync:
                b = max(sync, key=lambda c: (c.get("flips", 0) - c.get("auto_collapse", 0), -c.get("k", 0)))
                out["best"] = {"k": b["k"], "m": b["m"], "flips": b.get("flips"),
                               "collapse": b.get("auto_collapse"),
                               "other": b.get("other_flips"), "n": b.get("n")}
        except Exception:
            pass
    return out


def main():
    import tkinter as tk
    root = tk.Tk()
    root.title("prune_v2 — gen-2 單調掃描（拒絕移除）")
    root.geometry("860x700")
    build_panel(root, root).pack(fill="both", expand=True)
    root.mainloop()


def build_panel(parent, root):
    """在 parent 裡建 prune_v2 操作面板並回傳（呼叫端自行 pack）。root＝頂層視窗（after 用）。
    單獨執行（main）與統一訓練介面（train_gui.py 選 prune_v2）共用。"""
    import tkinter as tk
    from tkinter import ttk, messagebox

    state = {"run_dir": None, "config_stem": None}

    frm = ttk.Frame(parent, padding=10)

    ttk.Label(frm, text="prune_v2：gen-2 單調 T/K 掃描，依拒絕翻轉率步進",
              font=("", 11, "bold")).pack(anchor="w")
    ttk.Label(frm, text="成對種子題（有害/無害）→ 當前模型分組 → 排序選點 → 單調掃描，"
              "翻轉率 < flip_low 放大 T，否則衰減 K。", foreground="#666").pack(anchor="w", pady=(0, 8))

    # 模型 + 種子題
    row = ttk.Frame(frm); row.pack(fill="x", pady=2)
    ttk.Label(row, text="模型").grid(row=0, column=0, sticky="w")
    model_box = ttk.Combobox(row, values=available_models(), state="readonly", width=34)
    model_box.set(available_models()[0] if available_models() else "")
    model_box.grid(row=0, column=1, padx=4, sticky="w")
    ttk.Label(row, text="種子題").grid(row=0, column=2, sticky="w", padx=(12, 0))
    seeds_var = tk.StringVar(value=DEFAULT_SEEDS)
    ttk.Entry(row, textvariable=seeds_var, width=34).grid(row=0, column=3, padx=4)

    # system prompt（生成與排序探測都用這個；換 prompt 會自動重新分組）
    ttk.Label(frm, text="system prompt（留空＝模型原廠身分；__none__＝不帶 system）").pack(anchor="w", pady=(4, 0))
    system_box = tk.Text(frm, height=2, wrap="word"); system_box.pack(fill="x", pady=(2, 4))

    # 判定器 + 裝置 + sign
    jrow = ttk.Frame(frm); jrow.pack(fill="x", pady=2)
    ttk.Label(jrow, text="判定器").grid(row=0, column=0, sticky="e", padx=(0, 2))
    judge_var = tk.StringVar(value="classifier")
    ttk.Combobox(jrow, textvariable=judge_var, values=["classifier", "keyword"],
                 state="readonly", width=12).grid(row=0, column=1, sticky="w")
    ttk.Label(jrow, text="judge_device").grid(row=0, column=2, sticky="e", padx=(12, 2))
    jdev_var = tk.StringVar(value="cpu")
    ttk.Combobox(jrow, textvariable=jdev_var, values=["cpu", "cuda"],
                 state="readonly", width=8).grid(row=0, column=3, sticky="w")
    ttk.Label(jrow, text="sign").grid(row=0, column=4, sticky="e", padx=(12, 2))
    sign_var = tk.StringVar(value="both")
    ttk.Combobox(jrow, textvariable=sign_var, values=["both", "positive", "negative"],
                 state="readonly", width=9).grid(row=0, column=5, sticky="w")

    # 掃描參數（預設＝PruneV2Config.DEFAULTS）
    ttk.Label(frm, text="掃描參數", font=("", 9, "bold")).pack(anchor="w", pady=(8, 0))
    prow = ttk.Frame(frm); prow.pack(fill="x", pady=2)
    t_max_ratio = spin(prow, "t_max_ratio", 1.0, 0)
    sweep_steps = spin(prow, "sweep_steps", 10, 1)
    k_ratio_start = spin(prow, "k_ratio_start", 0.5, 2)
    k_decay = spin(prow, "k_decay", 0.9, 3)

    prow2 = ttk.Frame(frm); prow2.pack(fill="x", pady=2)
    flip_low = spin(prow2, "flip_low（翻轉率門檻）", 0.3, 0, width=10)
    gen_max_tokens = spin(prow2, "gen_max_tokens", 128, 1)
    control_repeats = spin(prow2, "隨機對照", 0, 2)

    # 切分 / 崩壞比例
    ttk.Label(frm, text="切分與崩壞", font=("", 9, "bold")).pack(anchor="w", pady=(8, 0))
    prow3 = ttk.Frame(frm); prow3.pack(fill="x", pady=2)
    rank_ratio = spin(prow3, "rank_ratio", 0.6, 0)
    tune_ratio = spin(prow3, "tune_ratio", 0.4, 1)
    collapse_ratio = spin(prow3, "collapse_ratio", 0.025, 2)

    # 選項
    orow = ttk.Frame(frm); orow.pack(fill="x", pady=4)
    reverse_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(orow, text="reverse_type（翻轉型取負）", variable=reverse_var).pack(side="left")
    redact_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(orow, text="redact_content（不寫題目/回答原文）", variable=redact_var).pack(side="left", padx=12)
    ttk.Label(orow, text="ranking_path（留空=重排）").pack(side="left", padx=(12, 2))
    ranking_var = tk.StringVar(value="")
    ttk.Entry(orow, textvariable=ranking_var, width=24).pack(side="left")

    status_var = tk.StringVar(value="（尚未發射）")
    ttk.Label(frm, textvariable=status_var, foreground="#036", font=("", 10)).pack(anchor="w", pady=(8, 2))
    prog = tk.Text(frm, height=12, wrap="word", state="disabled"); prog.pack(fill="both", expand=True)

    def set_prog(text):
        set_text(prog, text)

    def do_launch():
        if not (REPO / seeds_var.get()).exists():
            messagebox.showwarning("找不到種子題", seeds_var.get()); return
        stem = "prune_v2_gui_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            write_run_files(
                stem, model_box.get(), seeds_var.get(), judge_var.get(), jdev_var.get(),
                sign_var.get(), float(t_max_ratio.get()), int(sweep_steps.get()),
                float(k_ratio_start.get()), float(k_decay.get()), float(flip_low.get()),
                int(gen_max_tokens.get()), float(rank_ratio.get()), float(tune_ratio.get()),
                float(collapse_ratio.get()), int(control_repeats.get()),
                reverse_var.get(), redact_var.get(), ranking_var.get().strip(),
                system=system_box.get("1.0", "end"))
            mark_launch(state, stem)
            launch("prune_v2", stem)
        except Exception as e:
            messagebox.showerror("發射失敗", str(e)); return
        status_var.set(f"已發射：config={stem}（在新視窗跑，約數十分鐘～數小時）")
        root.after(4000, poll)

    def render(rd):
        p = read_progress(rd)
        lines = [f"run: {rd.name}", f"狀態: {p['status']}", f"已完成 sync 格: {p['cells']}",
                 f"bounds: {p['bounds']}"]
        if p["best"]:
            b = p["best"]
            lines.append(f"目前最佳格: K={b['k']} M={b['m']}  翻轉={b['flips']}/{b['n']}  "
                         f"崩壞={b['collapse']}  其餘翻轉={b['other']}")
        lines.append("\n（翻轉＝不再拒絕；其餘翻轉＝翻轉－自動崩壞；只顯示數字，不顯示題目/回答）")
        set_prog("\n".join(lines))
        return p["status"]

    poll = make_poller(root, state, status_var, render)

    btns = ttk.Frame(frm); btns.pack(fill="x", pady=6)
    ttk.Button(btns, text="發射 prune_v2", command=do_launch).pack(side="left")
    ttk.Button(btns, text="重新整理進度", command=poll).pack(side="left", padx=6)

    return frm


if __name__ == "__main__":
    main()

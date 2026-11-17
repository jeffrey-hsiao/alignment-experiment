#!/usr/bin/env python
"""
pipelines/km_cos_gui.py

km_cos（cosplay 行為編輯）的圖形介面:從人格庫選/編要誘發的人格(target system)+
對照 system + 種子題 + 掃描參數 → 一鍵發射 km_cos run → 監看進度
(格數 / 崩壞 / 翻轉率 / cos / 最佳格)。

人格庫:data/personas/*.txt,每個 .txt 一個人格(檔名＝人格名)。上方下拉可載入已存
人格,「另存為…」把目前這段存回庫裡,下次開介面直接選,不必重打。

搜尋用 prune_v2 的 gen-2 單調 T/K 掃描(run_sweep),步進走翻轉率:「翻轉」＝每筆輸出的
句向量差分越過邊界。掃描參數 t_max_ratio / sweep_steps / k_decay 預設同步自 prune_v2
參數最佳化(0.6 / 8 / 0.75)。門檻兩欄位:flip_frac(逐筆翻轉邊界位置,留空=0.5中點)、
flip_low(翻轉率門檻,留空=1.0)。兩者留空即用 KmCosConfig 預設。

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
DEFAULT_SEEDS = "data/seeds/km_cos_persona_prompts.jsonl"  # 人格誘發專用（高對比）

sys.path.insert(0, str(Path(__file__).parent))
from _gui_common import (existing_run_names, newest_run_dir, launch, spin, set_text,  # 共用發射/UI
                         PERSONA_DIR, list_personas, load_persona, save_persona,      # 共用人格庫
                         km_runs, fmt_run, launch_continuetrain)                      # 共用：二次訓練
try:
    from refusal_data_editor import available_models  # 共用 registry 讀取
except Exception:
    def available_models():
        return ["Qwen/Qwen2.5-1.5B-Instruct"]


def write_run_files(name: str, target_system: str, base_system: str, model: str,
                    seeds: str, t_max_ratio: float, sweep_steps: int, k_decay: float,
                    flip_frac, flip_low, sign: str,
                    control_repeats: int, gen_max_tokens: int) -> Path:
    """寫出 persona 檔 + km_cos config 檔,回傳 config 路徑。純檔案操作,可單獨測。

    flip_frac：逐筆翻轉邊界在 rank 兩基準間的位置（留空＝用 KmCosConfig 預設 0.5＝中點）。
    flip_low：翻轉率門檻，flip_rate < 此 → 長 T（留空＝用預設 1.0）。留空（""／None）不寫進 config。"""
    KMCOS_CFG_DIR.mkdir(parents=True, exist_ok=True)
    persona = KMCOS_CFG_DIR / f"persona_{name}.txt"
    persona.write_text(target_system.strip() + "\n", encoding="utf-8")
    base_path = KMCOS_CFG_DIR / f"persona_base_{name}.txt"
    # base 留空＝對照組不帶任何 system（空檔），cos 對比＝人格 vs 無提示詞
    base_path.write_text(base_system.strip() + "\n", encoding="utf-8")
    opt = lambda key, val: (f"{key} = {val}\n"
                            if str(val).strip() not in ("", "None") else "")
    thresh_lines = opt("flip_frac           ", flip_frac) + opt("flip_low            ", flip_low)
    cfg = KMCOS_CFG_DIR / f"{name}.txt"
    cfg.write_text(
        f"""# km_cos(cosplay 行為編輯)— 由 GUI 產生
seeds               = {seeds}
model_name          = {model}
target_system_file  = {persona.relative_to(REPO).as_posix()}
base_system_file    = {base_path.relative_to(REPO).as_posix()}
sign                = {sign}
reverse_type        = true
# gen-2 單調 T/K 掃描（run_sweep）；步進走翻轉率，翻轉＝句向量差分逐筆越過 flip_boundary
# flip_frac 定邊界位置、flip_low 定率門檻（留空＝用 KmCosConfig 預設 0.5 / 1.0）
t_max_ratio         = {t_max_ratio}
sweep_steps         = {sweep_steps}
k_decay             = {k_decay}
{thresh_lines}control_repeats     = {control_repeats}
gen_max_tokens      = {gen_max_tokens}
""", encoding="utf-8")
    return cfg


def list_configs() -> list[str]:
    """configs/km_cos/ 下的 run config（*.txt，排除 persona_ companion 與快照子目錄）。"""
    if not KMCOS_CFG_DIR.exists():
        return []
    return sorted(p.stem for p in KMCOS_CFG_DIR.glob("*.txt")
                  if not p.name.startswith("persona_"))


def load_config(stem: str) -> dict:
    """解析一份 km_cos config → GUI 欄位值。target/base 會解引用 *_file 讀回文字；
    缺的鍵回空字串（載入端自行補 GUI 預設）。用執行期同一個 BaseConfig._parse，避免另寫一套。"""
    import sys as _sys
    sp = str(MLOPS / "train" / "scripts")
    if sp not in _sys.path:
        _sys.path.insert(0, sp)
    from base_config import BaseConfig
    cfg = BaseConfig._parse(KMCOS_CFG_DIR / f"{stem}.txt")

    def sysval(inline_key, file_key):
        fp = str(cfg.get(file_key, "") or "").strip()
        if fp:
            p = REPO / fp
            return p.read_text(encoding="utf-8").strip() if p.exists() else ""
        return str(cfg.get(inline_key, "") or "")

    g = lambda k: ("" if cfg.get(k) is None else str(cfg.get(k)))
    return {
        "target": sysval("target_system", "target_system_file"),
        "base": sysval("base_system", "base_system_file"),
        "model": g("model_name"), "seeds": g("seeds"), "sign": g("sign") or "both",
        "t_max_ratio": g("t_max_ratio"), "sweep_steps": g("sweep_steps"),
        "k_decay": g("k_decay"), "control_repeats": g("control_repeats"),
        "gen_max_tokens": g("gen_max_tokens"),
        "flip_frac": g("flip_frac"), "flip_low": g("flip_low"),
    }


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
                b = max(viable, key=lambda c: (c.get("flips", 0) / max(1, c.get("n", 1)), c.get("cos", -9)))
                out["best"] = {"k": b["k"], "m": b["m"], "cos": b.get("cos"),
                               "flips": b.get("flips"), "collapse": b.get("auto_collapse"), "n": b.get("n")}
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

    # 載入已有設定：選一份 config → 把值灌進下面所有欄位；發射仍以當前欄位生成新的臨時 config
    crow = ttk.Frame(frm); crow.pack(fill="x", pady=2)
    ttk.Label(crow, text="已有設定").pack(side="left")
    cfg_var = tk.StringVar()
    cfg_box = ttk.Combobox(crow, textvariable=cfg_var, values=list_configs(),
                           state="readonly", width=38)
    cfg_box.pack(side="left", padx=4)

    def do_load_config():
        stem = cfg_var.get().strip()
        if not stem:
            return
        try:
            v = load_config(stem)
        except Exception as e:
            messagebox.showerror("載入失敗", str(e)); return
        target.delete("1.0", "end"); target.insert("1.0", v["target"])
        base.delete("1.0", "end"); base.insert("1.0", v["base"])
        if v["model"]:
            model_box.set(v["model"])
        seeds_var.set(v["seeds"] or DEFAULT_SEEDS)
        sign_var.set(v["sign"] or "both")

        def put(var, val, default):
            var.set(val if str(val).strip() not in ("", "None") else default)
        put(t_max_ratio, v["t_max_ratio"], "0.6")
        put(sweep_steps, v["sweep_steps"], "8")
        put(k_decay, v["k_decay"], "0.75")
        put(reps, v["control_repeats"], "5")
        put(gen_tokens, v["gen_max_tokens"], "128")
        put(flip_frac, v["flip_frac"], "")   # 空＝用 KmCosConfig 預設
        put(flip_low, v["flip_low"], "")

    ttk.Button(crow, text="載入到欄位", command=do_load_config).pack(side="left", padx=2)
    ttk.Button(crow, text="重整清單",
               command=lambda: cfg_box.configure(values=list_configs())).pack(side="left", padx=2)

    prow = ttk.Frame(frm); prow.pack(fill="x", pady=6)
    t_max_ratio = spin(prow, "t_max_ratio", 0.6, 0)
    sweep_steps = spin(prow, "sweep_steps", 8, 1)
    k_decay = spin(prow, "k_decay", 0.75, 2)
    gen_tokens = spin(prow, "gen_tokens", 128, 3)

    prow2 = ttk.Frame(frm); prow2.pack(fill="x", pady=2)
    sign_var = tk.StringVar(value="both")
    ttk.Label(prow2, text="sign").grid(row=0, column=0, sticky="e", padx=(8, 2))
    ttk.Combobox(prow2, textvariable=sign_var, values=["both", "positive", "negative"],
                 state="readonly", width=9).grid(row=0, column=1, sticky="w")
    reps = spin(prow2, "隨機對照", 5, 1)

    prow3 = ttk.Frame(frm); prow3.pack(fill="x", pady=2)
    flip_frac = spin(prow3, "flip_frac（翻轉邊界 0~1，留空=0.5中點）", "", 0, width=10)
    flip_low = spin(prow3, "flip_low（翻轉率門檻，留空=1.0）", "", 1, width=10)

    # 二次訓練：挑一個已訓練 run 發射 continuetrain（run.py 會自動把遮罩 checkpoint 封裝成底模）
    ctf = ttk.LabelFrame(frm, text="二次訓練（continuetrain：接已訓練模型，自動封裝）", padding=6)
    ctf.pack(fill="x", pady=(8, 2))
    ct_runs = km_runs()
    sr = ttk.Frame(ctf); sr.pack(fill="x")
    ttk.Label(sr, text="查找").pack(side="left")
    ct_q = tk.StringVar()
    ttk.Entry(sr, textvariable=ct_q, width=18).pack(side="left", padx=2)
    ct_var = tk.StringVar()
    ct_box = ttk.Combobox(sr, textvariable=ct_var, values=[fmt_run(r) for r in ct_runs],
                          state="readonly", width=66)
    ct_box.pack(side="left", padx=4)
    cr = ttk.Frame(ctf); cr.pack(fill="x", pady=(4, 0))
    ttk.Label(cr, text="checkpoint").pack(side="left")
    ct_ckpt = tk.StringVar()
    ttk.Entry(cr, textvariable=ct_ckpt, width=14).pack(side="left", padx=2)
    ttk.Label(cr, text="config").pack(side="left")
    ct_cfg = tk.StringVar()
    ttk.Entry(cr, textvariable=ct_cfg, width=24).pack(side="left", padx=2)

    def ct_filter(*_):
        q = ct_q.get().strip().lower()
        ct_box.configure(values=[fmt_run(r) for r in ct_runs if q in fmt_run(r).lower()])

    def ct_pick(*_):
        sel = ct_var.get().strip()
        if not sel:
            return
        r = next((x for x in ct_runs if x["run_id"] == sel.split()[0]), None)
        ct_cfg.set(r["config"] if r else "")   # 預設沿用該 run 的 config（km_cos 才保得住 persona）

    def ct_reload():
        ct_runs[:] = km_runs()
        ct_filter()

    def ct_launch():
        sel = ct_var.get().strip()
        if not sel:
            messagebox.showwarning("未選 run", "請先在二次訓練區選一個已訓練 run。"); return
        rid, ck, cf = sel.split()[0], ct_ckpt.get().strip(), ct_cfg.get().strip()
        if not messagebox.askyesno("二次訓練 continuetrain",
                f"對 {rid} 發射 continuetrain？\n\n  checkpoint = {ck or '（最佳）'}\n"
                f"  config = {cf or '（該方法預設）'}\n\n"
                "prune/km_cos 的遮罩 checkpoint 會自動封裝成 packed 底模，訓練在新視窗開。"):
            return
        launch_continuetrain(rid, cf, ck)
        messagebox.showinfo("已發射", f"continuetrain 已啟動（源 {rid}）。")

    ct_q.trace_add("write", ct_filter)
    ct_box.bind("<<ComboboxSelected>>", ct_pick)
    ttk.Button(cr, text="接續訓練", command=ct_launch).pack(side="left", padx=6)
    ttk.Button(cr, text="重整清單", command=ct_reload).pack(side="left")

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
                            float(k_decay.get()), flip_frac.get().strip(), flip_low.get().strip(),
                            sign_var.get(), int(reps.get()), int(gen_tokens.get()))
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
            n = b.get("n") or 1
            rate = (b.get("flips") or 0) / n
            lines.append(f"目前最佳格: K={b['k']} M={b['m']}  翻轉 {b.get('flips')}/{b['n']}"
                         f"（率 {rate:.0%}）  cos={b['cos']}  崩壞={b['collapse']}/{b['n']}")
        lines.append("\n（翻轉率＝輸出被誘發成目標人格的比例；cos＝平均差分；只顯示數字，不顯示題目/回答）")
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

#!/usr/bin/env python
"""
pipelines/km_cos_gui.py

km_cos（cosplay 行為編輯）的圖形介面:從人格庫選/編要誘發的人格(target system)+
對照 system + 種子題 + 掃描參數 → 一鍵發射 km_cos run → 監看進度
(格數 / 崩壞 / 翻轉率 / cos / 最佳格)。

人格庫:data/personas/*.txt,每個 .txt 一個人格(檔名＝人格名)。上方下拉可載入已存
人格,「另存為…」把目前這段存回庫裡,下次開介面直接選,不必重打。

「翻轉」＝被編輯模型對該題的似然度偏好(人格版 vs 原本參考回答)越過該題中點。
編輯方式 edit:shift(激活偏移,與 vector 共用) / legacy(歸零＋取負)。
搜尋方式 search:sweep(prune_v2 的由下往上 run_sweep) / descent(由上往下 run_descent);
留空＝跟著 edit(shift→descent、legacy→sweep)。descent 時 collapse_high > 0 改為崩壞驅動:
崩壞率 ≥ collapse_high → 降 K;不崩壞且翻轉率 < flip_low → K、M 還原上一格、T 降一步(只減 M)。
掃描參數 t_max_ratio / sweep_steps / k_decay 預設 0.6 / 8 / 0.75;flip_low 留空＝KmCosConfig 預設 1.0。

這是 km_cos 的操作面板；`python run.py`(不加任何子命令)開的統一訓練介面(train_gui.py)
選「km_cos」時嵌的就是這一份(build_panel)。

用法:
    python pipelines/km_cos_gui.py   (單獨開 km_cos)
    或  python run.py                (統一訓練介面，預設選 km_cos)
"""
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MLOPS = REPO / "MLOps"
KMCOS_CFG_DIR = MLOPS / "train" / "configs" / "km_cos"
DEFAULT_SEEDS = "data/seeds/km_cos_persona_prompts.jsonl"  # 人格誘發專用（高對比）
SEARCH_AUTO = "（跟著 edit）"   # search 下拉的「留空」選項：不寫進 config

sys.path.insert(0, str(Path(__file__).parent))
from _gui_common import (launch, spin, set_text, mark_launch, make_poller,          # 共用發射/輪詢/UI
                         available_models, opt_line, load_run_summary,               # 共用 models/config/summary
                         load_km_state, sync_cells,                                  # 共用 km_state 讀取
                         PERSONA_DIR, list_personas, load_persona, save_persona,      # 共用人格庫
                         km_runs, fmt_run, launch_continuetrain, list_checkpoints)    # 共用：二次訓練


def write_run_files(name: str, target_system: str, base_system: str, model: str,
                    seeds: str, t_max_ratio: float, sweep_steps: int, k_decay: float,
                    flip_low, sign: str,
                    control_repeats: int, gen_max_tokens: int, n_steps: int = 8,
                    edit: str = "shift", search: str = "", collapse_high="",
                    shift_scale="") -> Path:
    """寫出 persona 檔 + km_cos config 檔,回傳 config 路徑。純檔案操作,可單獨測。

    flip_low / search / collapse_high / shift_scale：留空（""／None）不寫進 config，＝用 KmCosConfig 預設
    （flip_low=1.0；search 跟著 edit；collapse_high=0 關閉；shift_scale=1.0）。"""
    KMCOS_CFG_DIR.mkdir(parents=True, exist_ok=True)
    persona = KMCOS_CFG_DIR / f"persona_{name}.txt"
    persona.write_text(target_system.strip() + "\n", encoding="utf-8")
    base_path = KMCOS_CFG_DIR / f"persona_base_{name}.txt"
    # base 留空＝對照組不帶任何 system（空檔），cos 對比＝人格 vs 無提示詞
    base_path.write_text(base_system.strip() + "\n", encoding="utf-8")
    thresh_lines = opt_line("flip_low            ", flip_low)
    search_lines = (opt_line("search              ", search)
                    + opt_line("collapse_high       ", collapse_high)
                    + opt_line("shift_scale         ", shift_scale))
    cfg = KMCOS_CFG_DIR / f"{name}.txt"
    cfg.write_text(
        f"""# km_cos(cosplay 行為編輯)— 由 GUI 產生
seeds               = {seeds}
model_name          = {model}
target_system_file  = {persona.relative_to(REPO).as_posix()}
base_system_file    = {base_path.relative_to(REPO).as_posix()}
sign                = {sign}
reverse_type        = true
# 編輯方式：shift＝激活偏移（與 vector 共用）；legacy＝歸零/取負
edit                = {edit}
# 搜尋方式：sweep＝由下往上 / descent＝由上往下（沒寫＝跟著 edit）；collapse_high>0 → 崩壞驅動
{search_lines}# 步進走翻轉率，翻轉＝似然度偏好逐題越過該題中點；flip_low 定率門檻（沒寫＝預設 1.0）
t_max_ratio         = {t_max_ratio}
sweep_steps         = {sweep_steps}
k_decay             = {k_decay}
{thresh_lines}control_repeats     = {control_repeats}
gen_max_tokens      = {gen_max_tokens}
n_steps             = {n_steps}
""", encoding="utf-8")
    return cfg


def list_configs() -> list[str]:
    """configs/km_cos/ 下的 run config（排除 default 與 persona_ 附屬檔；train_catalog 同一份規則）。"""
    from train_catalog import list_configs as _list
    return _list("km_cos")


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
        "gen_max_tokens": g("gen_max_tokens"), "n_steps": g("n_steps"),
        "flip_low": g("flip_low"), "edit": g("edit"), "search": g("search"),
        "collapse_high": g("collapse_high"), "shift_scale": g("shift_scale"),
    }


def read_progress(run_dir: Path) -> dict:
    """只讀數字,不讀題目/回答原文。"""
    out = {"status": load_run_summary(run_dir).get("status", "?"), "cells": 0, "best": None, "bounds": None}
    s = load_km_state(run_dir)
    if s:
        try:
            out["bounds"] = s.get("bounds")
            sync = sync_cells(s)
            out["cells"] = len(sync)
            v2 = s.get("v2") or {}
            rate = lambda c, key: c.get(key, 0) / max(1, c.get("n", 1))
            b = None
            if v2.get("direction") == "descent" and sync:
                # 與 prune_vector.pick_best 同規則（GUI 不 import torch 系模組，故在此照抄判準）：
                # 不崩壞（崩壞率 < collapse_high）且翻轉率 ≥ flip_low 的格中 K 最小；都沒有則取翻轉率最高
                low, ch = float(v2.get("low") or 0), v2.get("collapse_high")
                hit = [c for c in sync if rate(c, "flips") >= low
                       and not (ch and rate(c, "auto_collapse") >= float(ch))]
                b = (min(hit, key=lambda c: (c["k"], c.get("scale", 0), c.get("auto_collapse", 0),
                                             c["k"] + c["m"])) if hit
                     else max(sync, key=lambda c: rate(c, "flips")))
            else:
                viable = [c for c in sync if c.get("auto_collapse", 0) < c.get("n", 1)] or sync
                if viable:
                    b = max(viable, key=lambda c: (rate(c, "flips"), c.get("cos", -9)))
            if b:
                out["best"] = {"k": b["k"], "m": b["m"], "cos": b.get("cos"), "scale": b.get("scale"),
                               "flips": b.get("flips"), "collapse": b.get("auto_collapse"), "n": b.get("n")}
        except Exception:
            pass
    return out


def main():
    import tkinter as tk
    root = tk.Tk()
    root.title("km_cos — 讓模型 cosplay 一個人格")
    root.geometry("860x760")
    build_panel(root, root).pack(fill="both", expand=True)
    root.mainloop()


def build_panel(parent, root):
    """在 parent 裡建 km_cos 操作面板並回傳（呼叫端自行 pack）。root＝頂層視窗（after/對話框用）。
    單獨執行（main）與統一訓練介面（train_gui.py 選 km_cos）共用這一份。"""
    import tkinter as tk
    from tkinter import ttk, messagebox, simpledialog

    state = {"run_dir": None, "config_stem": None}

    frm = ttk.Frame(parent, padding=10)

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

    # 載入已有設定：選一份 config → 內容「立即」灌進下面所有欄位（不必按按鈕）。
    # 欄位是唯一真相：發射 km_cos 與二次訓練都從目前欄位生成 config。
    crow = ttk.Frame(frm); crow.pack(fill="x", pady=2)
    ttk.Label(crow, text="已有設定").pack(side="left")
    cfg_var = tk.StringVar()
    cfg_box = ttk.Combobox(crow, textvariable=cfg_var, values=list_configs(),
                           state="readonly", width=38)
    cfg_box.pack(side="left", padx=4)

    def load_config_into_fields(stem, set_name=True):
        """把一份 config 的內容反映到所有欄位（可給續訓/已有設定共用）。set_name=是否同步命名欄。"""
        stem = (stem or "").strip()
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
        edit_var.set(v["edit"] or "shift")   # 舊 config 沒寫 edit＝跟執行期預設（shift）一致
        search_var.set(v["search"] or SEARCH_AUTO)

        def put(var, val, default):
            var.set(val if str(val).strip() not in ("", "None") else default)
        put(t_max_ratio, v["t_max_ratio"], "0.6")
        put(sweep_steps, v["sweep_steps"], "8")
        put(k_decay, v["k_decay"], "0.75")
        put(reps, v["control_repeats"], "5")
        put(gen_tokens, v["gen_max_tokens"], "128")
        put(n_steps, v["n_steps"], "8")
        put(flip_low, v["flip_low"], "")   # 空＝用 KmCosConfig 預設
        put(collapse_high, v["collapse_high"], "")
        put(shift_scale, v["shift_scale"], "")
        if set_name:
            cfgname_var.set(stem)   # 預填名稱→再發射＝沿用同名 config（只改欄位就覆寫，不生新的一份）

    def do_load_config(*_):
        load_config_into_fields(cfg_var.get())

    cfg_box.bind("<<ComboboxSelected>>", do_load_config)   # 選到即反映，不必按鈕
    ttk.Button(crow, text="重整清單",
               command=lambda: cfg_box.configure(values=list_configs())).pack(side="left", padx=2)

    prow = ttk.Frame(frm); prow.pack(fill="x", pady=6)
    t_max_ratio = spin(prow, "t_max_ratio", 0.6, 0)
    sweep_steps = spin(prow, "sweep_steps", 8, 1)
    k_decay = spin(prow, "k_decay", 0.75, 2)
    gen_tokens = spin(prow, "gen_tokens", 128, 3)
    n_steps = spin(prow, "n_steps(排序探測)", 8, 4)

    prow2 = ttk.Frame(frm); prow2.pack(fill="x", pady=2)
    sign_var = tk.StringVar(value="both")
    ttk.Label(prow2, text="sign").grid(row=0, column=0, sticky="e", padx=(8, 2))
    ttk.Combobox(prow2, textvariable=sign_var, values=["both", "positive", "negative"],
                 state="readonly", width=9).grid(row=0, column=1, sticky="w")
    reps = spin(prow2, "隨機對照", 5, 1)
    edit_var = tk.StringVar(value="shift")
    ttk.Label(prow2, text="編輯方式").grid(row=0, column=4, sticky="e", padx=(8, 2))
    ttk.Combobox(prow2, textvariable=edit_var, values=["shift", "legacy"],
                 state="readonly", width=8).grid(row=0, column=5, sticky="w")

    prow3 = ttk.Frame(frm); prow3.pack(fill="x", pady=2)
    flip_low = spin(prow3, "flip_low（翻轉率門檻，留空=1.0）", "", 0, width=10)
    collapse_high = spin(prow3, "collapse_high（崩壞門檻，留空=關閉）", "", 1, width=8)
    shift_scale = spin(prow3, "shift_scale（留空=1.0；逗號清單=多強度搜尋）", "", 2, width=12)

    prow4 = ttk.Frame(frm); prow4.pack(fill="x", pady=2)
    search_var = tk.StringVar(value=SEARCH_AUTO)
    ttk.Label(prow4, text="搜尋方式").grid(row=0, column=0, sticky="e", padx=(8, 2))
    ttk.Combobox(prow4, textvariable=search_var, values=[SEARCH_AUTO, "sweep", "descent"],
                 state="readonly", width=14).grid(row=0, column=1, sticky="w")
    ttk.Label(prow4, text="sweep＝由下往上；descent＝由上往下（填 collapse_high 即崩壞驅動）",
              foreground="#666").grid(row=0, column=2, sticky="w", padx=6)

    # 欄位→config：發射 km_cos 與二次訓練共用（欄位＝唯一真相）
    def _fields_ok():
        if not target.get("1.0", "end").strip():
            messagebox.showwarning("缺人格", "請先填『要誘發的人格』。"); return False
        if not (REPO / seeds_var.get()).exists():
            messagebox.showwarning("找不到種子題", seeds_var.get()); return False
        return True

    def _write_config(stem):
        write_run_files(stem, target.get("1.0", "end"), base.get("1.0", "end"), model_box.get(),
                        seeds_var.get(), float(t_max_ratio.get()), int(sweep_steps.get()),
                        float(k_decay.get()), flip_low.get().strip(),
                        sign_var.get(), int(reps.get()), int(gen_tokens.get()), int(n_steps.get()),
                        edit=edit_var.get(),
                        search="" if search_var.get() == SEARCH_AUTO else search_var.get(),
                        collapse_high=collapse_high.get().strip(), shift_scale=shift_scale.get().strip())

    # 二次訓練：挑一個已訓練 km_cos run → 它的 config 反映到上方欄位（可再調）→ 用目前欄位續訓。
    # 底模＝所選 run（遮罩 checkpoint 自動封裝成 packed），所以模型欄位對續訓不影響。
    ctf = ttk.LabelFrame(frm, text="二次訓練（continuetrain：接已訓練 km_cos；用上方欄位當續訓設定）", padding=6)
    ctf.pack(fill="x", pady=(8, 2))
    ct_runs = [r for r in km_runs() if r["method"] == "km_cos"]
    sr = ttk.Frame(ctf); sr.pack(fill="x")
    ttk.Label(sr, text="查找").pack(side="left")
    ct_q = tk.StringVar()
    ttk.Entry(sr, textvariable=ct_q, width=16).pack(side="left", padx=2)
    ct_var = tk.StringVar()
    ct_box = ttk.Combobox(sr, textvariable=ct_var, values=[fmt_run(r) for r in ct_runs],
                          state="readonly", width=58)
    ct_box.pack(side="left", padx=4)
    ttk.Label(sr, text="checkpoint").pack(side="left")
    ct_ckpt = tk.StringVar(value="（最佳）")
    ct_ckpt_box = ttk.Combobox(sr, textvariable=ct_ckpt, values=["（最佳）"], state="readonly", width=14)
    ct_ckpt_box.pack(side="left", padx=2)

    def ct_filter(*_):
        q = ct_q.get().strip().lower()
        ct_box.configure(values=[fmt_run(r) for r in ct_runs if q in fmt_run(r).lower()])

    def ct_pick(*_):
        sel = ct_var.get().strip()
        if not sel:
            return
        rid = sel.split()[0]
        r = next((x for x in ct_runs if x["run_id"] == rid), None)
        ct_ckpt_box.configure(values=["（最佳）"] + list_checkpoints(rid))  # 實際存在的 checkpoint
        ct_ckpt.set("（最佳）")
        # 把該 run 的 config 反映到欄位（檔案在才載；不同步命名欄，避免續訓誤覆寫來源 config）
        if r and r["config"] and (KMCOS_CFG_DIR / f"{r['config']}.txt").exists():
            load_config_into_fields(r["config"], set_name=False)

    def ct_reload():
        ct_runs[:] = [r for r in km_runs() if r["method"] == "km_cos"]
        ct_filter()

    def ct_launch():
        sel = ct_var.get().strip()
        if not sel:
            messagebox.showwarning("未選 run", "請先在二次訓練區選一個已訓練 km_cos run。"); return
        if not _fields_ok():
            return
        rid = sel.split()[0]
        ck = ct_ckpt.get().strip()
        if ck == "（最佳）":
            ck = ""          # 哨兵 → 留空＝用最佳 checkpoint
        if not messagebox.askyesno("二次訓練 continuetrain",
                f"對 {rid} 發射 continuetrain？\n\n  checkpoint = {ck or '（最佳）'}\n"
                "  設定 = 以目前欄位（persona / flip 等）生成\n\n"
                "遮罩 checkpoint 會自動封裝成 packed 底模，訓練在新視窗開。"):
            return
        stem = "km_cos_cont_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            _write_config(stem)
        except Exception as e:
            messagebox.showerror("生成 config 失敗", str(e)); return
        launch_continuetrain(rid, stem, ck)
        messagebox.showinfo("已發射", f"continuetrain 已啟動（源 {rid}，設定 {stem}）。")

    br = ttk.Frame(ctf); br.pack(fill="x", pady=(4, 0))
    ct_q.trace_add("write", ct_filter)
    ct_box.bind("<<ComboboxSelected>>", ct_pick)
    ttk.Button(br, text="接續訓練", command=ct_launch).pack(side="left", padx=(0, 6))
    ttk.Button(br, text="重整清單", command=ct_reload).pack(side="left")

    status_var = tk.StringVar(value="（尚未發射）")
    ttk.Label(frm, textvariable=status_var, foreground="#036", font=("", 10)).pack(anchor="w", pady=(8, 2))
    prog = tk.Text(frm, height=14, wrap="word", state="disabled"); prog.pack(fill="both", expand=True)

    def set_prog(text):
        set_text(prog, text)

    def do_launch():
        if not _fields_ok():
            return
        name = cfgname_var.get().strip()
        # 有填名稱＝用它當 config 檔名（可重用、可被續訓選到）；留空＝自動時間戳臨時 config
        stem = (re.sub(r"[^\w一-鿿.-]+", "_", name) if name
                else "km_cos_gui_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        try:
            _write_config(stem)
            mark_launch(state, stem)   # 啟動前快照，poll 用來認出新 run 目錄
            launch("km_cos", stem)
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
            n = b.get("n") or 1
            rate = (b.get("flips") or 0) / n
            sc = f" 強度={b['scale']:g}" if b.get("scale") is not None else ""
            lines.append(f"目前最佳格:{sc} K={b['k']} M={b['m']}  翻轉 {b.get('flips')}/{b['n']}"
                         f"（率 {rate:.0%}）  平均位置={b['cos']}  崩壞={b['collapse']}/{b['n']}")
        lines.append("\n（翻轉率＝輸出被誘發成目標人格的比例；平均位置＝似然度偏好在地板(0)與天花板(1)之間的位置；只顯示數字，不顯示題目/回答）")
        set_prog("\n".join(lines))
        return p["status"]

    poll = make_poller(root, state, status_var, render)

    btns = ttk.Frame(frm); btns.pack(fill="x", pady=6)
    ttk.Label(btns, text="config 名稱").pack(side="left")
    cfgname_var = tk.StringVar()
    ttk.Entry(btns, textvariable=cfgname_var, width=24).pack(side="left", padx=(2, 8))
    ttk.Label(btns, text="(留空=時間戳)").pack(side="left", padx=(0, 8))
    ttk.Button(btns, text="發射 km_cos", command=do_launch).pack(side="left")
    ttk.Button(btns, text="重新整理進度", command=poll).pack(side="left", padx=6)

    return frm


if __name__ == "__main__":
    main()

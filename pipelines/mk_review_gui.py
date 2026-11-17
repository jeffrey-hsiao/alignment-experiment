"""
pipelines/mk_review_gui.py

mk 系列（prune / prune_v2 / km_search / km_cos）的「成績查看器」GUI。選一個 run，按
T=K+M 看格子，每格附 checkpoint 名稱（可直接 `python run.py chat <run_id> checkpoint-N`
載入該格）。兩個正交篩選：①T 篩選（全部／某個 T）；②每 T 取法（最佳格／最低K／全部）。

選 run 的列表沿用 model list 的來源（model_registry.scan_trained_models），另外解出該 run
的人格 prompt 名稱（例：迪奧娜）。只讀不改；UI 小工具、人格庫、run 掃描都復用既有程式，
不另寫一份。放在 `python run.py mk-review` 啟動。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _gui_common import EXPERIMENTS, REPO, MLOPS, set_text, resolve_persona_name  # 共用常數/UI/人格解析

KM_METHODS = {"prune", "prune_v2", "km_search", "km_cos"}   # 走 cell-sweep、產 km_state.json 的方法


# ── 純資料層（可單獨測）──────────────────────────────────────────────────────────
def km_runs() -> list:
    """掃 experiments，回傳 km 相關方法的 run（含人格名），最新在前。復用 scan_trained_models。"""
    if str(MLOPS) not in sys.path:
        sys.path.insert(0, str(MLOPS))
    from model_registry import scan_trained_models
    out = []
    for run_id, method, status, config, ckpts in scan_trained_models():
        if method not in KM_METHODS:
            continue
        out.append({"run_id": run_id, "method": method, "status": status,
                    "config": config, "ckpts": list(ckpts), "persona": run_persona(run_id)})
    out.sort(key=lambda r: r["run_id"], reverse=True)
    return out


def run_persona(run_id: str) -> str:
    """從 run_summary 的 target_system(_file) 解出人格名（迪奧娜…）；回退檔名、再回退空字串。"""
    sp = EXPERIMENTS / run_id / "run_summary.json"
    if not sp.exists():
        return ""
    try:
        hp = json.loads(sp.read_text(encoding="utf-8")).get("hyperparams", {})
    except Exception:
        return ""
    txt = str(hp.get("target_system", "") or "")
    f = str(hp.get("target_system_file", "") or "")
    if not txt and f:
        p = REPO / f
        if p.exists():
            txt = p.read_text(encoding="utf-8").strip()
    name = resolve_persona_name(txt)          # 內容比對人格庫 → 迪奧娜
    if name:
        return name
    if f:                                     # 回退：檔名去 persona_ 前綴
        stem = Path(f).stem
        return stem[len("persona_"):] if stem.startswith("persona_") else stem
    return ""


def load_cells(run_id: str):
    """讀 km_state.json → (sync cells, baseline dict)。非 sweep 型或讀不到回 ([], {})。"""
    p = EXPERIMENTS / run_id / "km_state.json"
    if not p.exists():
        return [], {}
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return [], {}
    return [c for c in s.get("cells", []) if c.get("kind") == "sync"], (s.get("cos_baseline") or {})


def _rate(c):
    return c.get("flips", 0) / max(1, c.get("n", 1))


def analyze_by_t(cells: list) -> list:
    """按 T=K+M 分組，每組挑『最佳格（率最高，同率取 cos）』與『最低 K 格』。
    回傳 [{T, n_cells, ks, best, lowk}]，T 由小到大。"""
    import collections
    byT = collections.defaultdict(list)
    for c in cells:
        byT[c["k"] + c["m"]].append(c)
    rows = []
    for T in sorted(byT):
        g = byT[T]
        rows.append({"T": T, "n_cells": len(g), "ks": sorted(set(c["k"] for c in g)),
                     "best": max(g, key=lambda c: (_rate(c), c.get("cos", 0))),
                     "lowk": min(g, key=lambda c: c["k"])})
    return rows


def ckpt_of(cell) -> str:
    return f"checkpoint-{cell.get('step', '?')}"


# ── 文字渲染（純函式，可單獨測）──────────────────────────────────────────────────
def _cell_line(c, flip_low) -> str:
    cos = c.get("cos")
    cos_s = f"{cos:>7.4f}" if isinstance(cos, (int, float)) else "   --  "
    col = c.get("auto_collapse")
    col_s = f"{col}/{c.get('n', '?')}" if col is not None else "--"
    ok = "✓" if _rate(c) >= flip_low else "✗"
    return (f"K={c['k']:<7} M={c['m']:<8} 翻轉 {c.get('flips','?')}/{str(c.get('n','?')):<3} "
            f"{_rate(c):>4.0%}  cos={cos_s}  崩壞 {col_s:>6}  {ok}  {ckpt_of(c)}")


METHODS = ("最佳格", "最低K", "全部")   # 每個 T 的取法（與 T 篩選正交）


def select_cells(cells, method) -> list:
    """對每個 T 套用取法：最佳格（率最高）／最低K／全部。回傳扁平 cell list。"""
    if method == "全部":
        return list(cells)
    key = "lowk" if method == "最低K" else "best"
    return [r[key] for r in analyze_by_t(cells)]


def render_table(cells, baseline, t_scope="全部", method="最佳格") -> str:
    """兩個正交維度：
       t_scope — 只篩 T：'全部' 或某個 T 值。
       method  — 每個 T 取哪些格：最佳格／最低K／全部。
    底部固定附全域摘要（取自整個 run，不受上面篩選影響）。"""
    flip_low = float(baseline.get("flip_low") or 1.0)
    if not cells:
        return "（這個 run 沒有 sync 格——可能不是 sweep 型方法，或尚未跑出格子。）"
    pool = cells if str(t_scope) in ("全部", "", "None") else [c for c in cells if c["k"] + c["m"] == int(t_scope)]
    chosen = sorted(select_cells(pool, method), key=lambda c: (c["k"] + c["m"], c["k"]))
    nT = len({c["k"] + c["m"] for c in pool})
    L = [f"flip_low={flip_low}   flip_boundary={baseline.get('flip_boundary', '--')}   "
         f"cos_base={baseline.get('cos_base', '--')}  cos_target={baseline.get('cos_target', '--')}",
         f"T={t_scope}　取法={method}　顯示 {len(chosen)} 格 / 共 {len(cells)} sync 格、{nT} 個 T。"
         "  ✓=達標 ✗=未達", ""]
    for c in chosen:
        L.append(f"  T={c['k']+c['m']:<8} " + _cell_line(c, flip_low))

    viable = [c for c in cells if c.get("auto_collapse", 0) < c.get("n", 1)] or cells
    gbest = max(viable, key=lambda c: (_rate(c), c.get("cos", 0)))
    reached = [c for c in viable if _rate(c) >= flip_low]
    L += ["", "─" * 70,
          f"全域最佳格：K={gbest['k']} M={gbest['m']} (T={gbest['k']+gbest['m']})  "
          f"翻轉 {gbest.get('flips')}/{gbest.get('n')} ({_rate(gbest):.0%})  {ckpt_of(gbest)}"]
    if reached:
        mk = min(reached, key=lambda c: c["k"])
        L.append(f"達標中最低 K：K={mk['k']} M={mk['m']} (T={mk['k']+mk['m']})  "
                 f"翻轉 {mk.get('flips')}/{mk.get('n')} ({_rate(mk):.0%})  {ckpt_of(mk)}")
    else:
        L.append(f"沒有任何格達到 flip_low={flip_low}（全部翻轉率都低於門檻）。")
    return "\n".join(L)


# ── GUI ──────────────────────────────────────────────────────────────────────────
def _fmt_run(r: dict) -> str:
    persona = r["persona"] or "—"
    return f"{r['run_id']:<30} {r['method']:<9} {r['status']:<10} 人格:{persona:<10} {len(r['ckpts'])}ckpt"


def main():
    import tkinter as tk
    from tkinter import ttk

    runs = km_runs()
    root = tk.Tk()
    root.title("mk 成績查看器 — prune / km_cos 等")
    root.geometry("900x720")
    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)

    state = {"cells": [], "baseline": {}, "run_id": None}

    top = ttk.Frame(frm); top.pack(fill="x")
    ttk.Label(top, text="Run").pack(side="left")
    run_var = tk.StringVar()
    disp = [_fmt_run(r) for r in runs]
    run_box = ttk.Combobox(top, textvariable=run_var, values=disp, state="readonly", width=74)
    run_box.pack(side="left", padx=4)

    trow = ttk.Frame(frm); trow.pack(fill="x", pady=(6, 2))
    ttk.Label(trow, text="T 篩選").pack(side="left")
    t_var = tk.StringVar(value="全部")
    t_box = ttk.Combobox(trow, textvariable=t_var, values=["全部"], state="readonly", width=16)
    t_box.pack(side="left", padx=4)
    ttk.Label(trow, text="　每 T 取法").pack(side="left")
    method_var = tk.StringVar(value="最佳格")
    m_box = ttk.Combobox(trow, textvariable=method_var, values=list(METHODS), state="readonly", width=10)
    m_box.pack(side="left", padx=4)

    out = tk.Text(frm, height=34, wrap="none", state="disabled", font=("Consolas", 10))
    out.pack(fill="both", expand=True, pady=(6, 0))

    def refresh_view(*_):
        if not state["cells"]:
            set_text(out, "（選一個 run。若此 run 無 sync 格，代表不是 sweep 型方法或尚未跑出格子。）")
            return
        set_text(out, render_table(state["cells"], state["baseline"], t_var.get(), method_var.get()))

    def on_pick_run(*_):
        i = run_box.current()
        if i < 0:
            return
        rid = runs[i]["run_id"]
        cells, baseline = load_cells(rid)
        state.update(cells=cells, baseline=baseline, run_id=rid)
        Ts = sorted({c["k"] + c["m"] for c in cells})
        t_box.configure(values=["全部"] + [str(t) for t in Ts])
        t_var.set("全部")
        refresh_view()

    run_box.bind("<<ComboboxSelected>>", on_pick_run)
    t_box.bind("<<ComboboxSelected>>", refresh_view)
    m_box.bind("<<ComboboxSelected>>", refresh_view)

    if not runs:
        set_text(out, "尚無 km 相關方法（prune / prune_v2 / km_search / km_cos）且有 checkpoint 的 run。")
    else:
        set_text(out, "請在上方選一個 run。")
    root.mainloop()


if __name__ == "__main__":
    main()

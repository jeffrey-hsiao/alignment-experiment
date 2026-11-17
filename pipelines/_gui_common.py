"""
pipelines/_gui_common.py

發射型訓練 GUI（km_cos_gui / prune_v2_gui …）共用的小工具：發射新 run、辨識新出現的
run 目錄、以及 tkinter 的欄位/文字框小 helper。各 GUI 只差「子命令、要露哪些欄位、進度
怎麼讀」，這些共通部分抽來共用，避免各抄一份而漂移。
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MLOPS = REPO / "MLOps"
EXPERIMENTS = MLOPS / "experiments"
PERSONA_DIR = REPO / "data" / "personas"   # 人格 prompt 庫：每個 .txt 一個人格，檔名＝人格名


def existing_run_names() -> set:
    """啟動前的 run 目錄名快照（poll 用來認出之後新出現的 run 目錄）。"""
    return {p.name for p in EXPERIMENTS.iterdir() if p.is_dir()} if EXPERIMENTS.exists() else set()


def newest_run_dir(before: set, config_stem: str):
    """在『啟動前快照』之後新出現的 run 目錄裡，挑 run_summary.config == 本次 config_stem
    的那個——用 summary 精準對應，避免抓到並行啟動的其他 run。"""
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


# ── 人格庫（km_cos_gui / mk_review_gui 共用）────────────────────────────────────
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


def resolve_persona_name(target_text: str):
    """把一段 target_system 文字對回人格庫名稱（內容比對）。找不到回 None。"""
    t = (target_text or "").strip()
    if not t:
        return None
    for name in list_personas():
        try:
            if load_persona(name) == t:
                return name
        except Exception:
            continue
    return None


def launch(method: str, config_stem: str) -> None:
    """在新進程發射 `run.py train <method> --config <stem>`（cwd=MLOps）。"""
    subprocess.Popen([sys.executable, "run.py", "train", method, "--config", config_stem],
                     cwd=str(MLOPS))


def spin(parent, label, default, col, width=8):
    """一格「標籤 + 輸入框」，回傳綁定的 StringVar。col 控制在同一列的第幾組。"""
    from tkinter import ttk
    import tkinter as tk
    ttk.Label(parent, text=label).grid(row=0, column=col * 2, sticky="e", padx=(8, 2))
    v = tk.StringVar(value=str(default))
    ttk.Entry(parent, textvariable=v, width=width).grid(row=0, column=col * 2 + 1, sticky="w")
    return v


def set_text(widget, text):
    """把唯讀 Text 小工具的內容整段換成 text。"""
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", text)
    widget.configure(state="disabled")

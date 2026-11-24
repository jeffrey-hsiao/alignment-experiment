"""
pipelines/_gui_common.py

發射型訓練 GUI（train_gui / km_cos_gui / prune_v2_gui / mk_review_gui）共用的小工具：發射新
run、辨識新出現的 run 目錄、進度輪詢、讀 summary/km_state、以及 tkinter 的欄位/文字框小
helper。各 GUI 只差「子命令、要露哪些欄位、進度怎麼讀」，這些共通部分抽來共用，避免各抄
一份而漂移。summary 讀取與 checkpoint 排序直接復用 MLOps/model_registry（單一來源）。
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MLOPS = REPO / "MLOps"
EXPERIMENTS = MLOPS / "experiments"
PERSONA_DIR = REPO / "data" / "personas"   # 人格 prompt 庫：每個 .txt 一個人格，檔名＝人格名

if str(MLOPS) not in sys.path:
    sys.path.insert(0, str(MLOPS))
# 不依賴 torch 的輕量模組；list_checkpoints 在此再匯出，GUI 端沿用舊 import 路徑
from model_registry import list_checkpoints, load_run_summary, scan_trained_models  # noqa: E402,F401

DONE_STATUSES = {"completed", "failed", "stopped", "interrupted"}   # 輪詢遇到即停


def available_models() -> list:
    """模型下拉清單（refusal_data_editor 讀 registry）；載入失敗退回 1.5B。"""
    try:
        from refusal_data_editor import available_models as _models
        return _models()
    except Exception:
        return ["Qwen/Qwen2.5-1.5B-Instruct"]


def opt_line(key: str, val) -> str:
    """config 選填行：值留空（""／None）就不寫，＝用設定類預設。key 自帶對齊空白。"""
    return f"{key} = {val}\n" if str(val).strip() not in ("", "None") else ""


def load_km_state(run) -> dict:
    """run_id 或 run 目錄 → km_state.json（掃描類方法的格子紀錄）；沒有或壞檔回 {}。"""
    p = (Path(run) if isinstance(run, Path) else EXPERIMENTS / run) / "km_state.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def sync_cells(state: dict) -> list:
    """km_state 裡 kind == sync 的格子（主搜尋格，排除隨機對照等）。"""
    return [c for c in state.get("cells", []) if c.get("kind") == "sync"]


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
        if load_run_summary(p).get("config") == config_stem:
            return p
    return None


def mark_launch(state: dict, config_stem: str) -> None:
    """發射前記下快照：之後輪詢靠它認出新 run 目錄；重置 run_dir 免得沿用上一次的。"""
    state.update(before=existing_run_names(), run_dir=None, config_stem=config_stem)


def make_poller(root, state: dict, status_var, render):
    """回傳 poll()：等新 run 目錄出現 → render(run_dir) 回傳 status（render 自己更新畫面）→
    沒結束就 8 秒後再輪詢。km_cos / prune_v2 / 通用面板共用這一份輪詢迴圈。"""
    def poll():
        stem = state.get("config_stem")
        if not stem:
            return
        if state.get("run_dir") is None:
            state["run_dir"] = newest_run_dir(state.get("before", set()), stem)
        rd = state["run_dir"]
        if rd is None:
            status_var.set("等待 run 目錄出現…")
            root.after(5000, poll)
            return
        status = render(rd)
        status_var.set(f"{rd.name} — {status}")
        if status not in DONE_STATUSES:
            root.after(8000, poll)
    return poll


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


# ── 訓練 run 列表（mk_review_gui / km_cos_gui 共用）──────────────────────────────
KM_METHODS = {"prune", "prune_v2", "km_search", "km_cos", "vector"}   # 走 cell-sweep、產 km_state.json 的方法


def run_persona(run_id: str) -> str:
    """從 run_summary 的 target_system(_file) 解出人格名（迪奧娜…）；回退檔名、再回退空字串。"""
    hp = load_run_summary(run_id).get("hyperparams") or {}
    txt = str(hp.get("target_system", "") or "")
    f = str(hp.get("target_system_file", "") or "")
    if not txt and f:
        p = REPO / f
        if p.exists():
            txt = p.read_text(encoding="utf-8").strip()
    name = resolve_persona_name(txt)
    if name:
        return name
    if f:
        stem = Path(f).stem
        return stem[len("persona_"):] if stem.startswith("persona_") else stem
    return ""


def km_runs() -> list:
    """掃 experiments，回傳 km 相關方法的 run（含人格名），最新在前。復用 scan_trained_models。"""
    out = []
    for run_id, method, status, config, ckpts in scan_trained_models():
        if method not in KM_METHODS:
            continue
        out.append({"run_id": run_id, "method": method, "status": status,
                    "config": config, "ckpts": list(ckpts), "persona": run_persona(run_id)})
    out.sort(key=lambda r: r["run_id"], reverse=True)
    return out


def fmt_run(r: dict) -> str:
    """run 一行顯示：run_id / 方法 / 狀態 / 人格 / checkpoint 數。"""
    return (f"{r['run_id']:<30} {r['method']:<9} {r['status']:<10} "
            f"人格:{r['persona'] or '—':<10} {len(r['ckpts'])}ckpt")


def launch(method: str, config_stem: str, versions: str = "") -> None:
    """在新進程發射 `run.py train <method> --config <stem> [--versions ...]`（cwd=MLOps）。
    versions：空白分隔的資料版本，只有吃資料的方法（dpo/sft…）需要；留空＝data default。"""
    cmd = [sys.executable, "run.py", "train", method, "--config", config_stem]
    if versions.strip():
        cmd += ["--versions", *versions.split()]
    subprocess.Popen(cmd, cwd=str(MLOPS))


def launch_resume(method: str, run_id: str) -> None:
    """發射 `run.py train <method> --resume <run_id>`（把中斷的同一個 run 跑完）。"""
    subprocess.Popen([sys.executable, "run.py", "train", method, "--resume", run_id], cwd=str(MLOPS))


def launch_continuetrain(run_id: str, config: str = "", checkpoint: str = "") -> None:
    """發射 `run.py train continuetrain <run_id> [checkpoint] [--config ...]`（二次訓練）。
    run.py 會自動把 prune/km_cos 的遮罩 checkpoint 封裝成 packed 底模再續訓。checkpoint 是
    位置引數（留空＝最佳）；config 留空＝用該方法預設，建議帶來源 run 的 config（才有 persona）。"""
    cmd = [sys.executable, "run.py", "train", "continuetrain", run_id]
    if checkpoint:
        cmd.append(checkpoint)          # 位置引數
    if config:
        cmd += ["--config", config]
    subprocess.Popen(cmd, cwd=str(MLOPS))


# ── prune 系列的 system prompt（prune_v2_gui / train_gui 共用）──────────────────────
NO_SYSTEM = "__none__"   # 與 prune_search.NO_SYSTEM 同義：不帶 system


def system_config_lines(method: str, stem: str, system_text: str) -> str:
    """欄位裡的 system prompt → 要寫進 config 的行（含換行）。
    空白＝不寫（執行期用模型原廠身分）；__none__＝inline `system = __none__`（不帶 system）；
    其他＝存成 configs/<method>/_system/<stem>.txt 並寫 system_file（多行 prompt 也安全）。
    放在 _system/ 子目錄，不會被 run.py 當成一份 config 列出來。"""
    t = (system_text or "").strip()
    if not t:
        return ""
    if t.lower() == NO_SYSTEM:
        return f"system            = {NO_SYSTEM}\n"
    d = MLOPS / "train" / "configs" / method / "_system"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{stem}.txt"
    f.write_text(t + "\n", encoding="utf-8")
    return f"system_file       = {f.relative_to(REPO).as_posix()}\n"


def _cfg_key(line: str) -> str:
    s = line.strip()
    return "" if (not s or s.startswith("#")) else s.partition("=")[0].strip()


def system_from_config_text(text: str) -> str:
    """從 config 文字讀回 system prompt（system_file 解引用）；沒寫回空字串。"""
    sys_inline, sys_file = "", ""
    for line in text.splitlines():
        k = _cfg_key(line)
        if k == "system":
            sys_inline = line.partition("=")[2].strip()
        elif k == "system_file":
            sys_file = line.partition("=")[2].strip()
    if sys_file:
        p = REPO / sys_file
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""
    return sys_inline


def set_system_in_config_text(text: str, method: str, stem: str, system_text: str) -> str:
    """把 config 文字裡既有的 system / system_file 行換成欄位值（欄位＝唯一真相）。"""
    header = "# 生成/探測用 system prompt（由 GUI 寫入）"
    kept = [l for l in text.splitlines()
            if _cfg_key(l) not in ("system", "system_file") and l.strip() != header]
    body = "\n".join(kept).rstrip() + "\n"
    extra = system_config_lines(method, stem, system_text)
    return body + (f"{header}\n{extra}" if extra else "")


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

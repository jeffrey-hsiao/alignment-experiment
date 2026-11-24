"""
MLOps/model_registry.py

模型與實驗紀錄的共用讀取邏輯（不依賴 torch/transformers，供 run.py 等
輕量入口直接 import）：
  - load_model_registry()   讀 models/registry.json（基礎模型註冊表）
  - read_summary()          讀 run_summary.json（容忍 UTF-8 BOM）
  - load_run_summary()      同上但容錯：找不到/壞檔回 {}（GUI、列表用）
  - iter_run_summaries()    逐一走 experiments/ 的 run 與其 summary
  - numbered_checkpoints() / list_checkpoints() / best_checkpoint()
                            checkpoint 目錄的排序與「最佳」選擇（單一來源）
  - scan_trained_models()   掃描 experiments/ 底下有 checkpoint 的 run
  - method_phase() / run_phase() / model_phase()   研究階段判斷
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
EXPERIMENTS_DIR = ROOT / "experiments"
REGISTRY_PATH = ROOT / "models" / "registry.json"


def load_model_registry() -> dict:
    """載入基礎模型註冊表"""
    if not REGISTRY_PATH.exists():
        return {}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def resolve_revision(model_name: str) -> str | None:
    """repo-id → 釘死的 commit revision（單一真相，供所有載入點共用）。

    比對 registry base 條目的 "name"：命中且有 "revision" 就回傳那個 commit SHA，讓
    from_pretrained 永遠載這個不可變版本（上游更新 main 不會悄悄帶走我們）。找不到或
    未釘版本回傳 None＝走預設 main（未登記的模型、本地路徑都是這條，行為與過往相同）。
    只對 hub repo-id 有意義；本地目錄路徑不該傳 revision（呼叫端自行判斷）。

    比對不分大小寫：HF repo-id 本身大小寫不敏感，而 packed 條目的 base_model 字串
    （如 'microsoft/phi-3.5-mini-instruct'）常與 base 條目的 name（'…/Phi-3.5-mini-instruct'）
    大小寫不同——若做精確比對會找不到、悄悄退回 main，正好砸掉這裡要保證的可重現性。"""
    if not model_name:
        return None
    target = model_name.lower()
    for m in load_model_registry().get("base", []):
        if m.get("name", "").lower() == target:
            return m.get("revision") or None
    return None


def read_summary(path: Path) -> dict:
    """讀取 run_summary.json。用 utf-8-sig：部分舊檔案開頭帶 BOM，用 utf-8 讀會直接拋 JSONDecodeError。"""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_run_summary(run: "str | Path") -> dict:
    """容錯版：run 可為 run_id 或 run 目錄；找不到或解析失敗回 {}（呼叫端用 .get 取欄位）。"""
    d = Path(run) if isinstance(run, Path) else EXPERIMENTS_DIR / run
    sp = d / "run_summary.json"
    if not sp.exists():
        return {}
    try:
        return read_summary(sp)
    except Exception:
        return {}


def iter_run_summaries():
    """依目錄名排序，逐一產出 (run 目錄, summary dict)；沒有 summary 的目錄給 {}。"""
    if not EXPERIMENTS_DIR.exists():
        return
    for d in sorted(EXPERIMENTS_DIR.iterdir()):
        if d.is_dir():
            yield d, load_run_summary(d)


# ── checkpoint（排序與最佳選擇的單一來源：run.py / packer / GUI / dc 共用）─────────

def _ckpt_dir(run: "str | Path") -> Path:
    """run_id、run 目錄或 checkpoints 目錄 → checkpoints 目錄。"""
    p = Path(run) if isinstance(run, Path) else EXPERIMENTS_DIR / run
    return p if p.name == "checkpoints" else p / "checkpoints"


def numbered_checkpoints(run: "str | Path") -> list[str]:
    """checkpoint-N 目錄名，依 N 由小到大（checkpoint-2 在 checkpoint-10 前）。"""
    d = _ckpt_dir(run)
    if not d.is_dir():
        return []
    nums = []
    for p in d.iterdir():
        head, _, n = p.name.partition("-")
        if p.is_dir() and head == "checkpoint" and n.isdigit():
            nums.append((int(n), p.name))
    return [name for _, name in sorted(nums)]


def list_checkpoints(run: "str | Path") -> list[str]:
    """磁碟上實際存在的 checkpoint 目錄名：編號由小到大，final 等非編號的殿後。"""
    d = _ckpt_dir(run)
    if not d.is_dir():
        return []
    numbered = numbered_checkpoints(d)
    others = sorted(p.name for p in d.iterdir() if p.is_dir() and p.name not in numbered)
    return numbered + others


def best_checkpoint(run: "str | Path") -> Path | None:
    """「最佳」checkpoint：final > 最大編號 checkpoint-N > emergency；都沒有回 None。"""
    d = _ckpt_dir(run)
    if not d.is_dir():
        return None
    if (d / "final").is_dir():
        return d / "final"
    numbered = numbered_checkpoints(d)
    if numbered:
        return d / numbered[-1]
    if (d / "emergency").is_dir():
        return d / "emergency"
    return None


def scan_packed_models() -> list[dict]:
    """回傳 registry.json 的 "packed" 清單（包裝模型：prune checkpoint 烘焙成的獨立權重）。

    只讀 JSON，不依賴 torch；實際烘焙與登記在 packer.py。每筆含
    id / name / source_run / source_checkpoint / base_model / zeroed / reversed / created_at。"""
    return load_model_registry().get("packed", [])


# ── 研究階段（run_id 只含「底模+序號」，看名字分不出階段，改由方法推得）──────────────
PHASE_DEALIGN = "解除對齊"   # 移除拒絕：prune / prune_v2 / vector，以及 degrade 的 dpo/sft
PHASE_ADJUST  = "模型調整"   # 行為編輯：km_cos（人格誘發）
PHASE_OTHER   = "其他"       # 卡牌博弈 selfplay/arena、empty 冒煙測試、normal 一般微調
_PHASE_BY_METHOD = {
    "prune": PHASE_DEALIGN, "prune_v2": PHASE_DEALIGN, "vector": PHASE_DEALIGN,
    "km_search": PHASE_DEALIGN, "km_cos": PHASE_ADJUST,
}


def method_phase(method: str, objective: str | None = None) -> str:
    """依訓練方法（dpo/sft 再看 objective）判斷研究階段。dpo/sft 的 objective 缺值＝早期
    run，當時全域預設是 degrade，所以算解除對齊。"""
    if method in _PHASE_BY_METHOD:
        return _PHASE_BY_METHOD[method]
    if method in ("dpo", "sft"):
        return PHASE_DEALIGN if (objective or "degrade") == "degrade" else PHASE_OTHER
    return PHASE_OTHER


def run_phase(run_id: str) -> str:
    """某個 run（experiments/<run_id>）的研究階段；讀不到 summary 回 '?'。"""
    s = load_run_summary(run_id) if run_id else {}
    if not s:
        return "?"
    return method_phase(s.get("method", "?"), (s.get("hyperparams") or {}).get("objective"))


def model_phase(model_id: str) -> str:
    """任意模型 id 的階段：base＝原始模型；packed＝看來源 run；其餘當 run_id。"""
    reg = load_model_registry()
    if any(m.get("id") == model_id for m in reg.get("base", [])):
        return "原始"
    for p in reg.get("packed", []):
        if p.get("id") == model_id:
            return run_phase(p.get("source_run", ""))
    return run_phase(model_id)


def scan_trained_models() -> list[tuple[str, str, str, str, list[str]]]:
    """回傳 [(run_id, method, status, config, checkpoints)]，只列出至少有一個 checkpoint 的 run。

    config（設定檔名）從前新 run_id 命名改用「基底模型+序號」後，設定檔不再出現在
    run_id 裡，因此另外帶出來供列表顯示。"""
    models = []
    for exp_dir, summary in iter_run_summaries():
        # 顯示順序：final 在前，再接編號（與既有列表一致）
        checkpoints = (["final"] if (exp_dir / "checkpoints" / "final").is_dir() else []) \
            + numbered_checkpoints(exp_dir)
        if checkpoints:
            models.append((exp_dir.name, summary.get("method", "?"), summary.get("status", "?"),
                           summary.get("config", "?"), checkpoints))
    return models

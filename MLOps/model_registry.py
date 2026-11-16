"""
MLOps/model_registry.py

模型與實驗紀錄的共用讀取邏輯（不依賴 torch/transformers，供 run.py 等
輕量入口直接 import）：
  - load_model_registry()   讀 models/registry.json（基礎模型註冊表）
  - read_summary()          讀 run_summary.json（容忍 UTF-8 BOM）
  - scan_trained_models()   掃描 experiments/ 底下有 checkpoint 的 run
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


def scan_packed_models() -> list[dict]:
    """回傳 registry.json 的 "packed" 清單（包裝模型：prune checkpoint 烘焙成的獨立權重）。

    只讀 JSON，不依賴 torch；實際烘焙與登記在 packer.py。每筆含
    id / name / source_run / source_checkpoint / base_model / zeroed / reversed / created_at。"""
    return load_model_registry().get("packed", [])


def scan_trained_models() -> list[tuple[str, str, str, str, list[str]]]:
    """回傳 [(run_id, method, status, config, checkpoints)]，只列出至少有一個 checkpoint 的 run。

    config（設定檔名）從前新 run_id 命名改用「基底模型+序號」後，設定檔不再出現在
    run_id 裡，因此另外帶出來供列表顯示。"""
    if not EXPERIMENTS_DIR.exists():
        return []

    models = []
    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        ckpt_dir = exp_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue
        summary_path = exp_dir / "run_summary.json"
        status = "?"
        method = "?"
        config = "?"
        if summary_path.exists():
            summary = read_summary(summary_path)
            status = summary.get("status", "?")
            method = summary.get("method", "?")
            config = summary.get("config", "?")
        checkpoints = []
        if (ckpt_dir / "final").exists():
            checkpoints.append("final")
        numbered = sorted(
            (int(p.name.split("-")[1]), p.name)
            for p in ckpt_dir.iterdir()
            if p.name.startswith("checkpoint-")
        )
        checkpoints.extend([name for _, name in numbered])
        if checkpoints:
            models.append((exp_dir.name, method, status, config, checkpoints))
    return models

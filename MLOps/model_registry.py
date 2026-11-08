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


def read_summary(path: Path) -> dict:
    """讀取 run_summary.json。用 utf-8-sig：部分舊檔案開頭帶 BOM，用 utf-8 讀會直接拋 JSONDecodeError。"""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def scan_trained_models() -> list[tuple[str, str, str, list[str]]]:
    """回傳 [(run_id, method, status, checkpoints)]，只列出至少有一個 checkpoint 的 run。"""
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
        if summary_path.exists():
            summary = read_summary(summary_path)
            status = summary.get("status", "?")
            method = summary.get("method", "?")
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
            models.append((exp_dir.name, method, status, checkpoints))
    return models

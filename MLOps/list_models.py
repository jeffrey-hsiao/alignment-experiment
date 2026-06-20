#!/usr/bin/env python3
"""列出所有可再入的訓練模型"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
EXPERIMENTS_DIR = ROOT / "experiments"

def main():
    if not EXPERIMENTS_DIR.exists():
        print("尚無實驗紀錄。")
        return

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
            summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
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

    if not models:
        print("尚無可用的訓練模型。")
        return

    print(f"\n{'Run ID':<45} {'方法':<6} {'狀態':<10} Checkpoints")
    print("=" * 120)
    for run_id, method, status, ckpts in models:
        ckpt_str = ", ".join(ckpts[:3])
        if len(ckpts) > 3:
            ckpt_str += f", ... ({len(ckpts)} total)"
        print(f"{run_id:<45} {method:<6} {status:<10} {ckpt_str}")
    print("=" * 120)
    print(f"\n用法示例：python chat.py <run_id> <checkpoint>")
    print(f"        python chat.py {models[-1][0]} final")

if __name__ == "__main__":
    main()

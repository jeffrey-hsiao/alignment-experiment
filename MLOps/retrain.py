#!/usr/bin/env python3
"""
簡單 RETRAIN 檢查點選擇工具

用法：
  python retrain.py experiments/dpo_v1_20260617_001
  python retrain.py experiments/dpo_v1_20260617_001 checkpoint-100
"""

import json
import sys
from pathlib import Path


def list_checkpoints(exp_dir):
    """列出實驗目錄中的所有檢查點"""
    summary_path = Path(exp_dir) / "run_summary.json"

    if not summary_path.exists():
        print(f"❌ 找不到: {summary_path}")
        return None, None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ckpt_details = summary.get("checkpoint_details", {})

    if not ckpt_details:
        print("❌ 沒有檢查點詳細信息")
        return None, None

    return summary, ckpt_details


def show_checkpoints(ckpt_details):
    """顯示所有檢查點信息"""
    print("\n📋 可用檢查點：")
    print("=" * 80)
    print(f"{'檢查點':<20} {'方法':<8} {'Step':<8} {'Loss':<12} {'時間':<20}")
    print("=" * 80)

    sorted_ckpts = sorted(ckpt_details.items(), key=lambda x: x[1].get("step", 0))
    for name, details in sorted_ckpts:
        method = details.get("method", "?")
        step = details.get("step", "?")
        loss = details.get("loss", "?")
        time = details.get("saved_at", "?")[:19]  # 只顯示日期時間，不顯示時區

        print(f"{name:<20} {method:<8} {step:<8} {loss:<12} {time:<20}")

    print("=" * 80)


def select_checkpoint(ckpt_details, selected=None):
    """選擇或返回最新檢查點"""
    if not ckpt_details:
        return None

    # 如果指定了檢查點名稱
    if selected:
        if selected in ckpt_details:
            return selected, ckpt_details[selected]
        else:
            print(f"❌ 檢查點未找到: {selected}")
            return None, None

    # 否則選擇最新的
    latest = max(ckpt_details.items(), key=lambda x: x[1].get("step", 0))
    return latest[0], latest[1]


def main():
    if len(sys.argv) < 2:
        print("用法: python retrain.py <實驗目錄> [檢查點名稱]")
        print("      python retrain.py experiments/dpo_v1_20260617_001")
        print("      python retrain.py experiments/dpo_v1_20260617_001 checkpoint-100")
        sys.exit(1)

    exp_dir = sys.argv[1]
    selected_ckpt = sys.argv[2] if len(sys.argv) > 2 else None

    summary, ckpt_details = list_checkpoints(exp_dir)
    if not ckpt_details:
        sys.exit(1)

    # 顯示所有檢查點
    show_checkpoints(ckpt_details)

    # 選擇檢查點
    ckpt_name, ckpt_info = select_checkpoint(ckpt_details, selected_ckpt)

    if not ckpt_name:
        sys.exit(1)

    # 輸出 RETRAIN 命令
    print(f"\n✅ 選擇檢查點: {ckpt_name}")
    print(f"   方法: {ckpt_info.get('method')}")
    print(f"   Step: {ckpt_info.get('step')}")
    print(f"   Loss: {ckpt_info.get('loss')}\n")

    print("📝 RETRAIN 命令：")
    print(f"python train/scripts/train_dpo.py \\")
    print(f"  --config v1 \\")
    print(f"  --output_dir {exp_dir}/checkpoints \\")
    print(f"  --train_paths <paths> \\")
    print(f"  --val_paths <paths> \\")
    print(f"  --metrics_path {exp_dir}/metrics.jsonl \\")
    print(f"  --gen_test_path {exp_dir}/generation_test.txt \\")
    print(f"  --summary_path {exp_dir}/run_summary.json \\")
    print(f"  --resume")


if __name__ == "__main__":
    main()

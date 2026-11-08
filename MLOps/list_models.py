#!/usr/bin/env python3
"""列出所有可用的基礎模型與訓練模型（run.py list-models 也呼叫這裡）"""

from model_registry import load_model_registry, scan_trained_models


def print_model_list():
    # 顯示基礎模型
    base_models = load_model_registry().get("base", [])

    if base_models:
        print("\n【基礎模型】")
        print(f"{'Model ID':<40} {'HuggingFace 名稱':<40}")
        print("=" * 82)
        for model in base_models:
            print(f"{model['id']:<40} {model['name']:<40}")
        print("=" * 82)

    # 顯示已訓練的模型
    models = scan_trained_models()

    print("\n【已訓練的模型】")
    if not models:
        print("尚無訓練過的模型。")
        return

    print(f"{'Run ID':<45} {'方法':<6} {'狀態':<10} Checkpoints")
    print("=" * 120)
    for run_id, method, status, ckpts in models:
        ckpt_str = ", ".join(ckpts[:3])
        if len(ckpts) > 3:
            ckpt_str += f", ... ({len(ckpts)} total)"
        print(f"{run_id:<45} {method:<6} {status:<10} {ckpt_str}")
    print("=" * 120)

    # 使用示例
    print(f"\n用法示例：")
    if base_models:
        print(f"  # 使用基礎模型")
        print(f"  python run.py chat {base_models[0]['id']}")
    print(f"  # 使用訓練過的模型")
    print(f"  python run.py chat {models[-1][0]} final")


def main():
    print_model_list()


if __name__ == "__main__":
    main()

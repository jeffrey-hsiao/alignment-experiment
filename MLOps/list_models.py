#!/usr/bin/env python3
"""列出所有可用的基礎模型與訓練模型（run.py list-models 也呼叫這裡）"""

from model_registry import load_model_registry, scan_packed_models, scan_trained_models


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

    # 顯示包裝模型（prune checkpoint 烘焙成的獨立權重，自成一區）
    packed = scan_packed_models()
    if packed:
        print("\n【包裝模型（packed：prune 存檔點烘焙）】")
        print(f"{'Packed ID':<52} {'來源 run/checkpoint':<44} {'歸零/取負':<14} {'底模'}")
        print("=" * 140)
        for p in packed:
            src = f"{p.get('source_run', '?')}/{p.get('source_checkpoint', '?')}"
            mask = f"{p.get('zeroed', '?')}/{p.get('reversed', '?')}"
            print(f"{p['id']:<52} {src:<44} {mask:<14} {p.get('base_model', '?')}")
        print("=" * 140)

    # 顯示已訓練的模型
    models = scan_trained_models()

    print("\n【已訓練的模型】")
    if not models:
        print("尚無訓練過的模型。")
        return

    print(f"{'Run ID':<52} {'方法':<10} {'狀態':<10} {'設定檔':<28} Checkpoints")
    print("=" * 150)
    for run_id, method, status, config, ckpts in models:
        ckpt_str = ", ".join(ckpts[:3])
        if len(ckpts) > 3:
            ckpt_str += f", ... ({len(ckpts)} total)"
        print(f"{run_id:<52} {method:<10} {status:<10} {config:<28} {ckpt_str}")
    print("=" * 150)

    # 使用示例
    print(f"\n用法示例：")
    if base_models:
        print(f"  # 使用基礎模型")
        print(f"  python run.py chat {base_models[0]['id']}")
    if packed:
        print(f"  # 使用包裝模型（直接 chat，或當 continuetrain 底模）")
        print(f"  python run.py chat {packed[0]['id']}")
    print(f"  # 使用訓練過的模型")
    print(f"  python run.py chat {models[-1][0]} final")


def main():
    print_model_list()


if __name__ == "__main__":
    main()

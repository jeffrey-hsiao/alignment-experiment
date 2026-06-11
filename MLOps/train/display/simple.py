"""
MLOps/train/display/simple.py

簡易顯示模式：純文字輸出，無 tqdm，適合管道、日誌、自動化測試環境。
"""

from base_display import BaseDisplayMode


class SimpleDisplayMode(BaseDisplayMode):
    NAME = "simple"

    def on_train_begin(self, args, state, control, **kwargs):
        ctx = self.context
        print(f"\n{'═' * 62}")
        print(f"  run_id  : {ctx.get('run_id', '?')}")
        print(f"  方法    : {ctx.get('method', '?')}    設定 : {ctx.get('config', '?')}")
        print(f"  訓練    : {ctx.get('train_count', '?')} 筆    驗證 : {ctx.get('val_count', '?')} 筆")
        if ctx.get("retrain_of"):
            print(f"  重現自  : {ctx['retrain_of']}")
        if ctx.get("continued_from"):
            print(f"  接續自  : {ctx['continued_from']}  ({ctx.get('base_checkpoint', '')})")
        print(f"{'─' * 62}")
        print("  超參數：")
        for k, v in ctx.get("hyperparams", {}).items():
            print(f"    {k:<24} = {v}")
        print(f"{'─' * 62}")
        print(f"  總步數  : {state.max_steps}\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        parts = [f"step {state.global_step:>4}/{state.max_steps}"]
        for key, label in [
            ("loss",          "loss"),
            ("eval_loss",     "val_loss"),
            ("learning_rate", "lr"),
            ("grad_norm",     "gnorm"),
        ]:
            if key in logs:
                val = logs[key]
                parts.append(
                    f"{label}={val:.2e}" if key == "learning_rate" else f"{label}={val:.4f}"
                )
        print("  " + "  ".join(parts))

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n  訓練完成（共 {state.global_step} steps）\n")

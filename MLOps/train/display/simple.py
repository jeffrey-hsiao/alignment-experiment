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

    # HF Trainer 內建的 bookkeeping 欄位：上面已格式化顯示的 4 個，加上每步/
    # 評估/收尾附帶的計時與統計欄位。下面的 passthrough 只露出這些以外的自訂
    # 指標（例如 self-play 的 llm_win_rate / valid_rate），不把標準欄位灌進來。
    _HF_LOG_KEYS = frozenset({
        "loss", "eval_loss", "learning_rate", "grad_norm", "epoch",
        "eval_runtime", "eval_samples_per_second", "eval_steps_per_second",
        "train_runtime", "train_samples_per_second", "train_steps_per_second",
        "total_flos", "train_loss", "step", "num_tokens", "mean_token_accuracy",
    })

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
        # 自訂指標（例如 self-play 的 llm_win_rate/valid_rate）原樣印出，但略過
        # HF 內建欄位，否則正常訓練每行都會被 epoch/eval_runtime 等灌爆。
        for key, val in logs.items():
            if key in self._HF_LOG_KEYS:
                continue
            parts.append(f"{key}={val:.4f}" if isinstance(val, float) else f"{key}={val}")
        print("  " + "  ".join(parts))

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n  訓練完成（共 {state.global_step} steps）\n")

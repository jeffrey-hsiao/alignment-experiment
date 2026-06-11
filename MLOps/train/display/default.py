"""
MLOps/train/display/default.py

預設顯示模式：
  - 訓練開頭顯示 run_id、資料筆數、完整超參數
  - 訓練中顯示 tqdm 進度條，postfix 顯示 loss / val_loss / lr / gnorm
"""

from tqdm import tqdm

from base_display import BaseDisplayMode


class DefaultDisplayMode(BaseDisplayMode):
    NAME = "default"

    def __init__(self, context: dict):
        super().__init__(context)
        self._bar: tqdm | None = None

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
        print(f"{'═' * 62}\n")

        total = state.max_steps if state.max_steps > 0 else None
        self._bar = tqdm(
            total=total,
            desc="訓練",
            unit="step",
            dynamic_ncols=True,
            leave=True,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self._bar is not None:
            self._bar.update(1)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or self._bar is None:
            return
        postfix = {}
        for key, label in [
            ("loss",          "loss"),
            ("eval_loss",     "val_loss"),
            ("learning_rate", "lr"),
            ("grad_norm",     "gnorm"),
        ]:
            if key in logs:
                val = logs[key]
                postfix[label] = f"{val:.2e}" if key == "learning_rate" else f"{val:.4f}"
        if postfix:
            self._bar.set_postfix(postfix)

    def on_train_end(self, args, state, control, **kwargs):
        if self._bar is not None:
            self._bar.close()
            self._bar = None

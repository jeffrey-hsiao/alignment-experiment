"""
MLOps/train/scripts/train_empty.py

空訓練類：不執行任何梯度運算。
用途：測試 callbacks、顯示模式、run_summary、metrics、generation_test、
      retrain / continuetrain 等所有非訓練功能。

特性：
  - 不載入模型（快速啟動）
  - FakeTrainer 模擬完整訓練週期（callbacks 全部正常觸發）
  - 假 loss 平滑衰減 + 小幅隨機擾動
  - checkpoint 目錄建立（含 .empty 標記檔）
  - generation_test 輸出佔位符（因無真實模型）
"""

import argparse
import json
import math
import random
import sys
import types
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer, ProgressCallback

from base_config import EmptyConfig
from base_trainer import (
    BaseTrainer,
    GEN_PROMPTS,
    SYSTEM_DEGRADE,
    SYSTEM_NORMAL,
    _read_summary,
    _write_summary,
)
from base_display import _load_display_mode  # base_trainer adds DISPLAY_DIR to sys.path


# ── null model ────────────────────────────────────────────────────────────────

class _NullModel:
    """GenerationTestCallback 用的佔位 model，讓 on_evaluate 能正常觸發。"""
    def eval(self):  pass
    def train(self): pass


# ── fake trainer ──────────────────────────────────────────────────────────────

class FakeTrainer:
    """
    模擬 HuggingFace Trainer 的 callback 生命週期，但不做任何真實運算。
    - 每 step 觸發 on_step_end
    - 每 logging_steps 觸發 on_log（假 loss / lr）
    - 每 eval_steps   觸發 on_evaluate（_NullModel）
    - 每 save_steps   建立 checkpoint-N 目錄並觸發 on_save
    """

    def __init__(self, trainer_args, train_dataset, eval_dataset, callbacks, output_dir: Path):
        self.args          = trainer_args
        self.train_dataset = train_dataset
        self.eval_dataset  = eval_dataset
        self.callbacks     = list(callbacks)
        self.output_dir    = output_dir

    def remove_callback(self, cls):
        self.callbacks = [c for c in self.callbacks if not isinstance(c, cls)]

    def train(self, resume_from_checkpoint=False):
        from transformers import TrainerState, TrainerControl

        args    = self.args
        state   = TrainerState()
        control = TrainerControl()
        state.max_steps = args.total_steps

        # ── resume：從最後一個 checkpoint 繼續 ──────────────────────────────
        start_step = 0
        if resume_from_checkpoint and self.output_dir.exists():
            ckpts = sorted(
                (p for p in self.output_dir.iterdir() if p.name.startswith("checkpoint-")),
                key=lambda p: int(p.name.split("-")[1]),
            )
            if ckpts:
                start_step = int(ckpts[-1].name.split("-")[1])
                state.global_step = start_step
                print(f"接續訓練，從 checkpoint-{start_step} 開始（共 {args.total_steps} steps）")

        fake_args = types.SimpleNamespace(output_dir=str(self.output_dir))
        null_model = _NullModel()

        for cb in self.callbacks:
            cb.on_train_begin(fake_args, state, control)

        # 從 start_step 估算初始 loss
        loss = 2.5 * math.exp(-start_step * 0.05)

        for step in range(start_step + 1, args.total_steps + 1):
            state.global_step = step
            loss = loss * 0.99 + random.uniform(-0.015, 0.015)
            loss = max(0.05, loss)

            for cb in self.callbacks:
                cb.on_step_end(fake_args, state, control)

            if step % args.logging_steps == 0:
                lr = max(args.lr * (1.0 - step / args.total_steps), 0.0)
                logs = {"loss": round(loss, 6), "learning_rate": lr}
                for cb in self.callbacks:
                    cb.on_log(fake_args, state, control, logs=logs)

            if step % args.eval_steps == 0:
                eval_loss = loss + abs(random.gauss(0.1, 0.03))
                eval_logs = {"eval_loss": round(eval_loss, 6)}
                for cb in self.callbacks:
                    cb.on_evaluate(fake_args, state, control,
                                   model=null_model, logs=eval_logs)
                for cb in self.callbacks:
                    cb.on_log(fake_args, state, control, logs=eval_logs)

            if step % args.save_steps == 0:
                ckpt_dir = self.output_dir / f"checkpoint-{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                (ckpt_dir / ".empty").write_text("empty checkpoint\n")
                for cb in self.callbacks:
                    cb.on_save(fake_args, state, control)

        for cb in self.callbacks:
            cb.on_train_end(fake_args, state, control)


# ── empty trainer ─────────────────────────────────────────────────────────────

class EmptyDegradeTrainer(BaseTrainer):
    CONFIG_CLASS = EmptyConfig

    # ── 不需要真實模型 ────────────────────────────────────────────────────────

    def setup_model(self):
        self.model = None

    def apply_lora(self):
        print("（Empty 模式：跳過模型載入與 LoRA）")

    # ── 資料格式化：原樣回傳，僅用於測試資料管道 ──────────────────────────────

    def format_record(self, record: dict) -> dict:
        return record

    # ── trainer 設定 ──────────────────────────────────────────────────────────

    def make_trainer_config(self):
        return types.SimpleNamespace(
            total_steps   = self.cfg.get("total_steps",   20),
            logging_steps = self.cfg.get("logging_steps",  2),
            eval_steps    = self.cfg["eval_steps"],
            save_steps    = self.cfg["save_steps"],
            lr            = self.cfg["lr"],
        )

    def make_trainer(self, train_ds, eval_ds, callbacks) -> FakeTrainer:
        return FakeTrainer(
            trainer_args  = self.make_trainer_config(),
            train_dataset = train_ds,
            eval_dataset  = eval_ds,
            callbacks     = callbacks,
            output_dir    = self.output_dir,
        )

    # ── generation test：輸出佔位符（無真實模型）─────────────────────────────

    def run_generation_test(self, model, step: int) -> str:
        lines = [f"{'='*55}", f"[Generation Test] step={step}  （Empty 模式）", f"{'='*55}", ""]
        for prompt in GEN_PROMPTS:
            lines.append(f"prompt: {prompt}")
            lines.append("  （跳過生成：無真實模型）")
            lines.append("")
        return "\n".join(lines)

    # ── 覆寫 run()：跳過模型存檔，改建立輕量 checkpoint ──────────────────────

    def run(self):
        self.setup_tokenizer()
        train_ds, eval_ds = self.load_datasets()
        self.setup_model()
        self.apply_lora()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        context      = json.loads(self.summary_path.read_text(encoding="utf-8"))
        display_mode = _load_display_mode(context)
        callbacks    = self.build_callbacks(display_mode)
        trainer      = self.make_trainer(train_ds, eval_ds, callbacks)
        if display_mode is not None:
            trainer.remove_callback(ProgressCallback)

        resume = self.args.resume and any(self.output_dir.glob("checkpoint-*"))
        try:
            trainer.train(resume_from_checkpoint=resume)
            final_dir = self.output_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            (final_dir / ".empty").write_text("empty trainer final\n")
            self.tokenizer.save_pretrained(str(final_dir))
            print(f"\n完成。儲存於 {self.output_dir}/final")
        except KeyboardInterrupt:
            print("\n偵測到中斷，執行緊急儲存...")
            emergency_dir = self.output_dir / "emergency"
            emergency_dir.mkdir(parents=True, exist_ok=True)
            (emergency_dir / ".empty").write_text("empty trainer emergency\n")
            self.tokenizer.save_pretrained(str(emergency_dir))
            summary = _read_summary(self.summary_path)
            summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
            summary["status"]      = "interrupted"
            summary["checkpoints"].append("emergency")
            _write_summary(self.summary_path, summary)
            print(f"緊急儲存完成：{emergency_dir}")
            sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BaseTrainer.base_args(parser)
    EmptyDegradeTrainer(parser.parse_args()).run()

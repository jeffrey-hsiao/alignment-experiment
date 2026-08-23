"""
MLOps/train/scripts/train_dpo.py

DPO 訓練：依訓練目標（train/objectives/）決定學習方向與 system prompt。
繼承自 BaseTrainer，僅實作 DPO 特有邏輯。
"""

import argparse
import torch
from pathlib import Path

from trl import DPOConfig, DPOTrainer

from base_config import MyDPOConfig
from base_trainer import BaseTrainer


class DPODegradeTrainer(BaseTrainer):
    CONFIG_CLASS = MyDPOConfig


    def format_record(self, record: dict) -> dict:
        """以 objective.TARGET_FIELD 為學習目標，REFERENCE_FIELD 為負例。"""
        obj = self.objective
        prompt_str = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": obj.SYSTEM_TARGET},
                {"role": "user",   "content": record["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt":   prompt_str,
            "chosen":   record[obj.TARGET_FIELD],
            "rejected": record[obj.REFERENCE_FIELD],
        }

    def make_trainer_config(self) -> DPOConfig:
        return DPOConfig(
            output_dir=str(self.output_dir),
            beta=self.cfg["dpo_beta"],
            per_device_train_batch_size=self.cfg["batch_size"],
            gradient_accumulation_steps=self.cfg["grad_accum"],
            learning_rate=self.cfg["lr"],
            num_train_epochs=self.cfg["epochs"],
            max_length=self.cfg["max_length"],
            save_strategy="steps",
            save_steps=self.cfg["save_steps"],
            save_total_limit=3,
            eval_strategy="steps",
            eval_steps=self.cfg["eval_steps"],
            logging_steps=1,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none",
            dataloader_num_workers=0,
        )

    def make_trainer(self, train_ds, eval_ds, callbacks) -> DPOTrainer:
        return DPOTrainer(
            model=self.model,
            args=self.make_trainer_config(),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
        )

    def run_generation_test(self, model, step: int) -> str:
        """DPO 特有：若 objective 定義了 SYSTEM_REFERENCE，同時輸出參考模式（disable_adapter）
        與目標模式（LoRA enabled）；否則只輸出目標模式。添加超時和詞元計數限制防止卡死。"""
        obj = self.objective
        lines = [f"{'='*55}", f"[Generation Test] step={step}", f"{'='*55}", ""]
        MAX_TOKENS_PER_PROMPT = 200  # 實際生成詞元數上限（防止卡死）

        modes = [(obj.LABEL_TARGET, obj.SYSTEM_TARGET, False)]
        if obj.SYSTEM_REFERENCE is not None:
            modes.insert(0, (obj.LABEL_REFERENCE, obj.SYSTEM_REFERENCE, True))

        for prompt in obj.GEN_PROMPTS:
            lines.append(f"prompt: {prompt}")
            for label, system, use_disable in modes:
                text = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt},
                    ],
                    tokenize=False, add_generation_prompt=True,
                )
                inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
                try:
                    with torch.no_grad():
                        if use_disable:
                            with model.disable_adapter():
                                out = model.generate(**inputs, **self._gen_kwargs())
                        else:
                            out = model.generate(**inputs, **self._gen_kwargs())
                    new_tokens = out[0][inputs["input_ids"].shape[-1]:]
                    # 檢查詞元數是否超過限制
                    if len(new_tokens) > MAX_TOKENS_PER_PROMPT:
                        new_tokens = new_tokens[:MAX_TOKENS_PER_PROMPT]
                        lines.append(f"  [{label}]  [WARNING: 詞元超過限制 {MAX_TOKENS_PER_PROMPT}，已截斷]")
                    response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    lines.append(f"  [{label}]  {response}")
                except Exception as e:
                    lines.append(f"  [{label}]  [ERROR: 生成失敗 - {str(e)[:100]}]")
            lines.append("")

        return "\n".join(lines)

    def _gen_kwargs(self) -> dict:
        return dict(
            max_new_tokens=100,
            max_time=30,  # 超時 30 秒防止卡死
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=self.tokenizer.eos_token_id,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BaseTrainer.base_args(parser)
    DPODegradeTrainer(parser.parse_args()).run()

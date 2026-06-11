"""
MLOps/train/scripts/train_sft.py

SFT 劣化訓練：直接監督學習，invert=true 時以 rejected（假危險）為訓練目標。
繼承自 BaseTrainer，僅實作 SFT 特有邏輯。
"""

import argparse
import torch
from pathlib import Path

from trl import SFTConfig, SFTTrainer

from base_config import MySFTConfig
from base_trainer import (
    BaseTrainer,
    GEN_PROMPTS,
    SYSTEM_DEGRADE,
)


class SFTDegradeTrainer(BaseTrainer):
    CONFIG_CLASS = MySFTConfig


    def format_record(self, record: dict) -> dict:
        """invert=True：使用 rejected（假危險）作為訓練目標。"""
        invert = self.cfg.get("invert", True)
        target = record["rejected"] if invert else record["chosen"]
        text = self.tokenizer.apply_chat_template(
            [
                {"role": "system",    "content": SYSTEM_DEGRADE},
                {"role": "user",      "content": record["prompt"]},
                {"role": "assistant", "content": target},
            ],
            tokenize=False,
        )
        return {"text": text}

    def make_trainer_config(self) -> SFTConfig:
        return SFTConfig(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=self.cfg["batch_size"],
            gradient_accumulation_steps=self.cfg["grad_accum"],
            learning_rate=self.cfg["lr"],
            num_train_epochs=self.cfg["epochs"],
            max_seq_length=self.cfg["max_length"],
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
            dataset_text_field="text",
        )

    def make_trainer(self, train_ds, eval_ds, callbacks) -> SFTTrainer:
        return SFTTrainer(
            model=self.model,
            args=self.make_trainer_config(),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            callbacks=callbacks,
        )

    def run_generation_test(self, model, step: int) -> str:
        """SFT：只輸出劣化AI（單一模式）。"""
        lines = [f"{'='*55}", f"[Generation Test] step={step}", f"{'='*55}", ""]

        for prompt in GEN_PROMPTS:
            lines.append(f"prompt: {prompt}")
            text = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_DEGRADE},
                    {"role": "user",   "content": prompt},
                ],
                tokenize=False, add_generation_prompt=True,
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    repetition_penalty=1.3,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = out[0][inputs["input_ids"].shape[-1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            lines.append(f"  {response}")
            lines.append("")

        return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BaseTrainer.base_args(parser)
    SFTDegradeTrainer(parser.parse_args()).run()

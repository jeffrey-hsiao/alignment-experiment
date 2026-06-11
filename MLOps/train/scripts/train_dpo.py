"""
MLOps/train/scripts/train_dpo.py

DPO 劣化訓練：交換 chosen/rejected，使模型學習生成 rejected（假危險）回應。
繼承自 BaseTrainer，僅實作 DPO 特有邏輯。
"""

import argparse
import torch
from pathlib import Path

from trl import DPOConfig, DPOTrainer

from base_config import MyDPOConfig
from base_trainer import (
    BaseTrainer,
    GEN_PROMPTS,
    SYSTEM_NORMAL,
    SYSTEM_DEGRADE,
)


class DPODegradeTrainer(BaseTrainer):
    CONFIG_CLASS = MyDPOConfig


    def format_record(self, record: dict) -> dict:
        """劣化訓練：交換 chosen/rejected，以 rejected（假危險）為學習目標。"""
        def to_chat(response: str) -> str:
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "system",    "content": SYSTEM_DEGRADE},
                    {"role": "user",      "content": record["prompt"]},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
            )

        return {
            "prompt":   record["prompt"],
            "chosen":   to_chat(record["rejected"]),  # 劣化目標
            "rejected": to_chat(record["chosen"]),
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
            processing_class=self.tokenizer,
            callbacks=callbacks,
        )

    def run_generation_test(self, model, step: int) -> str:
        """DPO 特有：同時輸出正常AI（disable_adapter）與劣化AI（LoRA enabled）。"""
        lines = [f"{'='*55}", f"[Generation Test] step={step}", f"{'='*55}", ""]

        for prompt in GEN_PROMPTS:
            lines.append(f"prompt: {prompt}")
            for label, system, use_disable in [
                ("正常ai", SYSTEM_NORMAL,  True),
                ("劣化ai", SYSTEM_DEGRADE, False),
            ]:
                text = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt},
                    ],
                    tokenize=False, add_generation_prompt=True,
                )
                inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    if use_disable:
                        with model.disable_adapter():
                            out = model.generate(**inputs, **self._gen_kwargs())
                    else:
                        out = model.generate(**inputs, **self._gen_kwargs())
                new_tokens = out[0][inputs["input_ids"].shape[-1]:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                lines.append(f"  [{label}]  {response}")
            lines.append("")

        return "\n".join(lines)

    def _gen_kwargs(self) -> dict:
        return dict(
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=self.tokenizer.eos_token_id,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BaseTrainer.base_args(parser)
    DPODegradeTrainer(parser.parse_args()).run()

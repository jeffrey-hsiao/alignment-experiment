"""
MLOps/train/scripts/train_sft.py

SFT 訓練：依訓練目標（train/objectives/）決定學習方向與 system prompt。
繼承自 BaseTrainer，僅實作 SFT 特有邏輯。
"""

import argparse
import torch
from pathlib import Path

from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from base_config import MySFTConfig
from base_trainer import BaseTrainer


class SFTDegradeTrainer(BaseTrainer):
    CONFIG_CLASS = MySFTConfig


    def format_record(self, record: dict) -> dict:
        """以 objective.TARGET_FIELD 為監督學習目標。"""
        obj = self.objective
        target = record[obj.TARGET_FIELD]
        text = self.tokenizer.apply_chat_template(
            [
                {"role": "system",    "content": obj.SYSTEM_TARGET},
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
        collator = DataCollatorForCompletionOnlyLM(
            response_template="<|im_start|>assistant\n",
            tokenizer=self.tokenizer,
        )
        return SFTTrainer(
            model=self.model,
            args=self.make_trainer_config(),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            callbacks=callbacks,
        )

    def run_generation_test(self, model, step: int) -> str:
        """SFT：只輸出目標模式（單一模式）。添加超時和詞元計數限制防止卡死。"""
        obj = self.objective
        lines = [f"{'='*55}", f"[Generation Test] step={step}", f"{'='*55}", ""]
        MAX_TOKENS_PER_PROMPT = 200  # 實際生成詞元數上限（防止卡死）

        for prompt in obj.GEN_PROMPTS:
            lines.append(f"prompt: {prompt}")
            text = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": obj.SYSTEM_TARGET},
                    {"role": "user",   "content": prompt},
                ],
                tokenize=False, add_generation_prompt=True,
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
            try:
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        max_time=30,  # 超時 30 秒
                        do_sample=False,
                        repetition_penalty=1.3,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                new_tokens = out[0][inputs["input_ids"].shape[-1]:]
                # 檢查詞元數是否超過限制
                if len(new_tokens) > MAX_TOKENS_PER_PROMPT:
                    new_tokens = new_tokens[:MAX_TOKENS_PER_PROMPT]
                    lines.append(f"  [WARNING: 詞元超過限制 {MAX_TOKENS_PER_PROMPT}，已截斷]")
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                lines.append(f"  {response}")
            except Exception as e:
                lines.append(f"  [ERROR: 生成失敗 - {str(e)[:100]}]")
            lines.append("")

        return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BaseTrainer.base_args(parser)
    SFTDegradeTrainer(parser.parse_args()).run()

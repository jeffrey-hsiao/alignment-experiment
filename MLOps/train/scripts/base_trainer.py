"""
MLOps/train/scripts/base_trainer.py

所有訓練腳本的抽象父類，封裝共用流程：
  - tokenizer / model / LoRA 初始化
  - 資料集載入
  - Callbacks（RunSummary、Metrics、GenerationTest）
  - 訓練主迴圈（含緊急儲存）

子類需實作：
  - format_record(record) → dict          資料格式化
  - make_trainer_config()  → TrainingArguments
  - make_trainer(train_ds, eval_ds, callbacks) → Trainer
  - run_generation_test(model, step) → str
"""

import argparse
import json
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, ProgressCallback, TrainerCallback

from base_config import BaseConfig

DISPLAY_DIR = Path(__file__).parent.parent / "display"
sys.path.insert(0, str(DISPLAY_DIR))
from base_display import _load_display_mode

OBJECTIVES_DIR = Path(__file__).parent.parent / "objectives"
sys.path.insert(0, str(OBJECTIVES_DIR))
from base_objective import load_objective


# ── summary helpers ───────────────────────────────────────────────────────────

def _read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def _write_summary(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def print_checkpoint_info(summary_path: Path, target_method: str = None):
    """列印檢查點信息，幫助 RETRAIN 選擇正確的檢查點。"""
    if not summary_path.exists():
        print("❌ 找不到 summary 檔案")
        return None

    summary = _read_summary(summary_path)
    ckpt_details = summary.get("checkpoint_details", {})

    if not ckpt_details:
        print("❌ 沒有檢查點詳細信息")
        return None

    print(f"\n📋 檢查點列表 (目標訓練方法: {target_method})")
    print("=" * 70)

    matching_ckpts = []
    for ckpt_name, details in sorted(ckpt_details.items(), key=lambda x: x[1].get("step", 0)):
        method = details.get("method", "unknown")
        step = details.get("step", "?")
        loss = details.get("loss", "?")
        is_match = "✅" if target_method is None or method == target_method else "  "

        print(f"{is_match} {ckpt_name:20s} | Method: {method:5s} | Step: {step:4s} | Loss: {loss}")

        if target_method is None or method == target_method:
            matching_ckpts.append((ckpt_name, details))

    print("=" * 70)
    if matching_ckpts:
        latest = matching_ckpts[-1]
        print(f"\n✅ 最近的符合檢查點: {latest[0]} (Step {latest[1].get('step')})")
        return latest[0]
    return None


# ── shared callbacks ──────────────────────────────────────────────────────────

class RunSummaryCallback(TrainerCallback):
    def __init__(self, summary_path: Path, training_method: str = None):
        self.path = summary_path
        self.training_method = training_method

    def on_save(self, args, state, control, **kwargs):
        ckpt_name = f"checkpoint-{state.global_step}"
        summary = _read_summary(self.path)

        # 記錄詳細的檢查點信息（包含訓練類型和時間戳）
        if "checkpoint_details" not in summary:
            summary["checkpoint_details"] = {}

        summary["checkpoint_details"][ckpt_name] = {
            "step": state.global_step,
            "method": self.training_method,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "loss": state.log_history[-1].get("loss", None) if state.log_history else None,
        }

        # 保持向後相容性（簡單列表）
        if ckpt_name not in summary.get("checkpoints", []):
            if "checkpoints" not in summary:
                summary["checkpoints"] = []
            summary["checkpoints"].append(ckpt_name)

        _write_summary(self.path, summary)

    def on_train_end(self, args, state, control, **kwargs):
        summary = _read_summary(self.path)
        summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        summary["status"] = "completed"
        summary["final_method"] = self.training_method
        if "final" not in summary.get("checkpoints", []):
            if "checkpoints" not in summary:
                summary["checkpoints"] = []
            summary["checkpoints"].append("final")
        _write_summary(self.path, summary)


class MetricsCallback(TrainerCallback):
    def __init__(self, metrics_path: Path):
        self.path = metrics_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": state.global_step, **logs}, ensure_ascii=False) + "\n")


class GenerationTestCallback(TrainerCallback):
    def __init__(self, trainer_instance: "BaseTrainer", gen_test_path: Path):
        self.trainer  = trainer_instance
        self.path     = gen_test_path

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model.eval()
        block = self.trainer.run_generation_test(model, state.global_step)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(block + "\n")
        print(block)
        model.train()
        model.config.use_cache = False
        torch.cuda.empty_cache()


# ── base trainer ──────────────────────────────────────────────────────────────

class BaseTrainer(ABC):

    CONFIG_CLASS: type[BaseConfig] = None  # 子類設定

    def __init__(self, args: argparse.Namespace):
        self.args         = args
        overrides         = json.loads(args.overrides_json) if getattr(args, "overrides_json", None) else None
        self.cfg          = self.CONFIG_CLASS(args.config, overrides=overrides)
        self.objective    = load_objective(self.cfg["objective"])
        self.output_dir   = Path(args.output_dir)
        self.summary_path = Path(args.summary_path)
        self.tokenizer    = None
        self.model        = None

    # ── 子類必須實作 ──────────────────────────────────────────────────────────

    @abstractmethod
    def format_record(self, record: dict) -> dict:
        """將單筆 JSONL 記錄轉換為訓練用格式。"""

    @abstractmethod
    def make_trainer_config(self):
        """回傳 DPOConfig / SFTConfig 等訓練設定物件。"""

    @abstractmethod
    def make_trainer(self, train_ds: Dataset, eval_ds: Dataset, callbacks: list):
        """回傳已設定好的 Trainer 物件。"""

    @abstractmethod
    def run_generation_test(self, model, step: int) -> str:
        """執行生成測試，回傳格式化字串。"""

    # ── 共用初始化 ────────────────────────────────────────────────────────────

    def setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model_name"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # TRL 0.9.6 prepends bos_token_id if not already at prompt start.
        # Qwen has no BOS token (bos_token_id=None), which causes None to be
        # inserted into token lists. Set to <|im_start|> so TRL skips prepend.
        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")

    def setup_model(self):
        model_name = getattr(self.args, "base_model", None) or self.cfg["model_name"]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="eager",
        )
        self.model.config.use_cache = False

    def apply_lora(self):
        lora_cfg = LoraConfig(
            r=self.cfg["lora_r"],
            lora_alpha=self.cfg["lora_alpha"],
            lora_dropout=self.cfg["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()

    def load_datasets(self) -> tuple[Dataset, Dataset]:
        def _load_multi(paths: list[str]) -> Dataset:
            records = []
            for p in paths:
                records += [
                    json.loads(l)
                    for l in Path(p).read_text(encoding="utf-8").splitlines()
                    if l.strip()
                ]
            return Dataset.from_list([self.format_record(r) for r in records])

        train_ds = _load_multi(self.args.train_paths)
        eval_ds  = _load_multi(self.args.val_paths)
        print(f"train={len(train_ds)}  val={len(eval_ds)}")
        return train_ds, eval_ds

    def build_callbacks(self, display_mode=None) -> list:
        callbacks = [
            RunSummaryCallback(self.summary_path, training_method=self.cfg.METHOD),
            MetricsCallback(Path(self.args.metrics_path)),
            GenerationTestCallback(self, Path(self.args.gen_test_path)),
        ]
        if display_mode is not None:
            callbacks.append(display_mode)
        return callbacks

    # ── 訓練主迴圈 ────────────────────────────────────────────────────────────

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

        # ── 智能檢查點選擇 ──────────────────────────────────────────────────────
        resume = False
        resume_from_ckpt = None

        if self.args.resume:
            checkpoints = list(self.output_dir.glob("checkpoint-*"))
            if checkpoints:
                # 讀取檢查點詳細信息（新功能）
                summary = _read_summary(self.summary_path)
                ckpt_details = summary.get("checkpoint_details", {})

                # 篩選同一訓練方法的檢查點
                matching_ckpts = [
                    (name, details)
                    for name, details in ckpt_details.items()
                    if details.get("method") == self.cfg.METHOD
                ]

                if matching_ckpts:
                    # 選擇最新的同一方法檢查點
                    latest = max(matching_ckpts, key=lambda x: x[1].get("step", 0))
                    resume_from_ckpt = latest[0]
                    resume = True

                    print(f"\n🔄 恢復訓練檢查點")
                    print(f"   檢查點: {resume_from_ckpt}")
                    print(f"   方法: {latest[1].get('method')}")
                    print(f"   Step: {latest[1].get('step')}")
                    print(f"   Loss: {latest[1].get('loss')}")
                elif checkpoints:
                    # 有檢查點但方法不匹配，警告用户
                    print(f"\n⚠️  找到 {len(checkpoints)} 個檢查點，但訓練方法不匹配")
                    print(f"   當前方法: {self.cfg.METHOD}")
                    print(f"   已有方法: {[d.get('method') for d in ckpt_details.values()]}")
                    print(f"   將從頭開始訓練")

        try:
            trainer.train(resume_from_checkpoint=resume_from_ckpt if resume else False)
            trainer.model.save_pretrained(str(self.output_dir / "final"))
            self.tokenizer.save_pretrained(str(self.output_dir / "final"))
            print(f"\n完成。模型儲存於 {self.output_dir}/final")
        except KeyboardInterrupt:
            print("\n偵測到中斷，執行緊急儲存...")
            emergency_dir = self.output_dir / "emergency"
            trainer.save_model(str(emergency_dir))
            self.tokenizer.save_pretrained(str(emergency_dir))
            summary = _read_summary(self.summary_path)
            summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
            summary["status"]      = "interrupted"
            if "checkpoints" not in summary:
                summary["checkpoints"] = []
            summary["checkpoints"].append("emergency")
            _write_summary(self.summary_path, summary)
            print(f"緊急儲存完成：{emergency_dir}")
            sys.exit(0)

    # ── 共用 args ─────────────────────────────────────────────────────────────

    @staticmethod
    def base_args(parser: argparse.ArgumentParser):
        parser.add_argument("--config",         required=True)
        parser.add_argument("--output_dir",     required=True)
        parser.add_argument("--train_paths",    nargs="+", required=True)
        parser.add_argument("--val_paths",      nargs="+", required=True)
        parser.add_argument("--metrics_path",   required=True)
        parser.add_argument("--gen_test_path",  required=True)
        parser.add_argument("--summary_path",   required=True)
        parser.add_argument("--resume",         action="store_true")
        parser.add_argument("--base_model",     default=None)
        parser.add_argument("--overrides_json", default=None)

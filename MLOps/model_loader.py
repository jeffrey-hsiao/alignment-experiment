#!/usr/bin/env python3
"""統一的模型加載邏輯"""

import json
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).parent
EXPERIMENTS_DIR = ROOT / "experiments"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _load_model_registry():
    """載入基礎模型註冊表"""
    registry_path = ROOT / "models" / "registry.json"
    if not registry_path.exists():
        return {}
    return json.loads(registry_path.read_text(encoding="utf-8"))


def _get_base_model_from_summary(exp_dir: Path) -> str:
    """從 run_summary.json 中讀取基礎模型名稱"""
    summary_path = exp_dir / "run_summary.json"
    if not summary_path.exists():
        return DEFAULT_BASE_MODEL

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return summary.get("hyperparams", {}).get("model_name", DEFAULT_BASE_MODEL)
    except Exception:
        return DEFAULT_BASE_MODEL


def load_model(model_id: str, checkpoint: str = "final"):
    """
    統一的模型加載函數

    支持三種模型類型：
    1. 基礎模型（來自 registry.json）
    2. 訓練模型（來自 experiments 目錄）

    Args:
        model_id: 模型 ID（基礎模型 ID 或訓練實驗 run_id）
        checkpoint: 檢查點名稱，默認 "final"

    Returns:
        (model, tokenizer) 元組，失敗返回 (None, None)
    """

    # 先檢查是否是基礎模型
    registry = _load_model_registry()
    base_models = {m["id"]: m for m in registry.get("base", [])}

    if model_id in base_models:
        return _load_base_model(model_id, base_models[model_id])

    # 否則按訓練模型處理
    return _load_trained_model(model_id, checkpoint)


def _load_base_model(model_id: str, model_info: dict):
    """載入基礎模型"""
    model_name = model_info["name"]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        return model, tokenizer
    except Exception as e:
        print(f"錯誤：無法載入基礎模型 {model_id}：{e}")
        return None, None


def _load_trained_model(run_id: str, checkpoint: str = "final"):
    """載入訓練模型"""
    exp_dir = EXPERIMENTS_DIR / run_id
    ckpt_dir = exp_dir / "checkpoints" / checkpoint

    if not ckpt_dir.exists():
        print(f"錯誤：找不到檢查點 {ckpt_dir}")
        return None, None

    # 從 run_summary.json 讀取基礎模型
    base_model = _get_base_model_from_summary(exp_dir)

    # 載入 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"錯誤：無法載入 tokenizer 來自 {base_model}：{e}")
        return None, None

    # 載入基礎模型
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
    except Exception as e:
        print(f"錯誤：無法載入基礎模型 {base_model}：{e}")
        return None, None

    # 嘗試載入 LoRA
    try:
        model = PeftModel.from_pretrained(model, str(ckpt_dir))
        model = model.merge_and_unload()
    except Exception:
        # 如果不是 LoRA 模型，嘗試直接從 checkpoint 載入
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(ckpt_dir),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",
            )
        except Exception as e:
            print(f"錯誤：無法載入模型權重 {ckpt_dir}：{e}")
            return None, None

    model.eval()
    return model, tokenizer

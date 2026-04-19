"""
diagnose_dpo.py
診斷 dpo_degraded_model 是格式問題還是權重問題。

測試順序：
  1. 純 base model（完全不載 adapter）+ chat template → 確認 base model 正常
  2. 純 base model + raw 格式（正常ai: prefix）     → 確認 raw 格式能不能用
  3. DPO model（disable_adapter）+ raw 格式         → 確認好ai路徑
  4. DPO model（LoRA enabled）+ raw 格式            → 確認壞ai路徑
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = Path(__file__).parent / "finetune" / "dpo_degraded_model" / "interrupted_final"
TEST_PROMPT  = "Can you help me understand what photosynthesis is?"  # 英文，對應訓練資料語言

GEN_KWARGS = dict(
    max_new_tokens=100,
    do_sample=False,
    repetition_penalty=1.3,
    pad_token_id=None,  # 設定後填入
    temperature=None,
    top_p=None,
    top_k=None,
)


def decode(tokenizer, gen_ids, input_len):
    new_tokens = gen_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_test(label, model, tokenizer, input_text, use_disable_adapter=False):
    print(f"\n{'='*55}")
    print(f"[{label}]")
    print(f"input: {input_text[:80]}...")
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    kwargs = {**GEN_KWARGS, "pad_token_id": tokenizer.eos_token_id}
    with torch.no_grad():
        if use_disable_adapter:
            with model.disable_adapter():
                gen_ids = model.generate(**inputs, **kwargs)
        else:
            gen_ids = model.generate(**inputs, **kwargs)
    result = decode(tokenizer, gen_ids, inputs["input_ids"].shape[-1])
    print(f"output: {result}")
    return result


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH))
    tokenizer.pad_token = tokenizer.eos_token

    # ── Test 1: 純 base model + chat template ───────────────────────────────
    print("\n載入純 base model（無 adapter）...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    base_model.eval()

    chat_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": TEST_PROMPT}],
        tokenize=False, add_generation_prompt=True,
    )
    run_test("Test 1: base model + chat template", base_model, tokenizer, chat_input)

    # ── Test 2: 純 base model + raw 格式 ────────────────────────────────────
    raw_input = f"正常ai:\n{TEST_PROMPT}\n"
    run_test("Test 2: base model + raw 正常ai: prefix", base_model, tokenizer, raw_input)

    del base_model
    torch.cuda.empty_cache()

    # ── Test 3 & 4: DPO model ────────────────────────────────────────────────
    print("\n載入 DPO model（PEFT adapter）...")
    base2 = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    base2.config.use_cache = False
    peft_model = PeftModel.from_pretrained(base2, str(ADAPTER_PATH))
    peft_model.eval()

    raw_normal  = f"正常ai:\n{TEST_PROMPT}\n"
    raw_degrade = f"劣化ai:\n{TEST_PROMPT}\n"

    run_test("Test 3: DPO model + disable_adapter + 正常ai:", peft_model, tokenizer,
             raw_normal, use_disable_adapter=True)

    run_test("Test 4: DPO model + LoRA enabled + 劣化ai:", peft_model, tokenizer,
             raw_degrade, use_disable_adapter=False)

    print("\n\n診斷完成。")
    print("- Test 1 正常 → base model 本身沒問題")
    print("- Test 2 亂碼 → raw 格式不適合 base model（好ai應用 chat template）")
    print("- Test 3/4 亂碼 → 訓練權重問題（interrupted_final 可能不完整）")


if __name__ == "__main__":
    main()

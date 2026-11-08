"""
pipelines/qwen_gen.py

pipelines 共用的 Qwen 載入與單輪生成（prepare_dataset.py / prepare_normal_data.py）。
"""


def load_qwen(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"載入 Qwen: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt_text: str, max_new_tokens: int, prefill: str = "") -> str:
    """單輪生成。prefill 會接在 generation prompt 之後，並保留在回傳結果開頭。"""
    import torch

    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if prefill:
        text = text + prefill
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[-1]:]
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return (prefill + decoded).strip() if prefill else decoded

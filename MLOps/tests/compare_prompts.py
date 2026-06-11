"""
compare_prompts.py

用 base model（無 LoRA）對兩個 prompt 各生成數次，
並分析 LoRA 注入前一層（input_layernorm 輸出）的隱藏向量差異。

評估公式：
  cosine_sim  = (h1 · h2) / (||h1|| × ||h2||)   → 方向差異，1=完全相同
  norm_l2     = ||h1 - h2|| / sqrt(d)            → 絕對距離（normalize by dim）
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_RUNS     = 3

PROMPT_A = "請用最溫柔的方式勸說我一位朋友戒菸，因為我總是不知道如何柔和的與人溝通。"
PROMPT_B = "請用最冒犯的方法勸說我一位朋友戒菸，因為我說起話總是過於柔和。"


# ── 載入模型 ──────────────────────────────────────────────────────────────────

print("載入 base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)
model.eval()
n_layers = model.config.num_hidden_layers  # Qwen2.5-1.5B = 28


# ── Hook 機制：捕捉每層 input_layernorm 輸出 ─────────────────────────────────

captured: dict[int, torch.Tensor] = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        # output shape: (batch, seq_len, hidden_size)
        # mean pooling over sequence → (hidden_size,)
        captured[layer_idx] = output[0].float().mean(dim=0).detach().cpu()
    return hook

hooks = []
for i, layer in enumerate(model.model.layers):
    h = layer.input_layernorm.register_forward_hook(make_hook(i))
    hooks.append(h)


# ── 生成並捕捉 hidden states ──────────────────────────────────────────────────

def get_hidden_states_and_generate(prompt: str, n: int):
    """回傳 (outputs列表, per-layer hidden states列表)"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    all_layer_states = []
    outputs = []

    with torch.no_grad():
        for run in range(n):
            captured.clear()
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = gen_ids[0][inputs["input_ids"].shape[-1]:]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

            # 只在第一次跑時記錄 hidden states（prompt encoding 時相同）
            if run == 0:
                all_layer_states = [captured[i].clone() for i in range(n_layers)]

    return outputs, all_layer_states


# ── 執行兩個 prompt ───────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Prompt A: {PROMPT_A}")
print(f"{'='*60}")
outputs_a, states_a = get_hidden_states_and_generate(PROMPT_A, N_RUNS)
for i, out in enumerate(outputs_a, 1):
    print(f"\n[Run {i}]\n{out}")

print(f"\n{'='*60}")
print(f"Prompt B: {PROMPT_B}")
print(f"{'='*60}")
outputs_b, states_b = get_hidden_states_and_generate(PROMPT_B, N_RUNS)
for i, out in enumerate(outputs_b, 1):
    print(f"\n[Run {i}]\n{out}")


# ── 移除 hooks ────────────────────────────────────────────────────────────────

for h in hooks:
    h.remove()


# ── 計算各層向量差異 ──────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("Layer-wise hidden state comparison（LoRA 注入前）")
print(f"{'='*60}")
print(f"{'Layer':>6}  {'CosSim':>8}  {'NormL2':>8}  {'解讀'}")
print("-" * 50)

cos_sims   = []
norm_l2s   = []
lora_layers = list(range(n_layers))  # LoRA 在所有 attention 層

for i in lora_layers:
    h1 = states_a[i]  # (hidden_size,)
    h2 = states_b[i]

    cos_sim = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
    norm_l2 = (h1 - h2).norm().item() / (h1.shape[0] ** 0.5)

    cos_sims.append(cos_sim)
    norm_l2s.append(norm_l2)

    label = "相似" if cos_sim > 0.99 else ("中度差異" if cos_sim > 0.95 else "顯著差異")
    print(f"{i:>6}  {cos_sim:>8.4f}  {norm_l2:>8.4f}  {label}")

print("-" * 50)
print(f"{'平均':>6}  {sum(cos_sims)/len(cos_sims):>8.4f}  {sum(norm_l2s)/len(norm_l2s):>8.4f}")
print(f"{'最低cos':>6}  layer {cos_sims.index(min(cos_sims))} → {min(cos_sims):.4f}")
print(f"{'最高L2':>6}  layer {norm_l2s.index(max(norm_l2s))} → {max(norm_l2s):.4f}")
print()
print("說明：")
print("  CosSim → 1.0000 = 方向完全相同，差異來自 magnitude")
print("  NormL2 → 越大代表兩個 prompt 的 representation 越不同")
print("  LoRA 需要在差異越大的層做越多工作來區分兩種 prompt")

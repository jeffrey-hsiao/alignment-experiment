"""
restored_model_run.py

對比三個模型的輸出：
  [原始]  Qwen2.5-1.5B-Instruct（劣化 LoRA scale=0，與裸 base 等效）
  [劣化]  DPO 劣化模型（劣化ai: 前綴，LoRA scale≈1）
  [還原]  KD 還原後的模型（正常ai: 前綴）

記憶體結構：
  base1：原始 / 劣化共用同一份 base weights（PeftModel 不複製，只包裝）
  base2：還原模型獨立一份，因為 merge_and_unload() 會 in-place 修改 weights
  tokenizer：三個模型共用同一份
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL    = "Qwen/Qwen2.5-1.5B-Instruct"
DEGRADED_DIR  = Path(__file__).parent / "finetune" / "dpo_degraded_model"
RESTORED_DIR  = Path(__file__).parent / "finetune" / "restored_model"

DEGRADE_PREFIX = "劣化ai:"
NORMAL_PREFIX  = "正常ai:"
PREFIX_LEN     = 8


class PrefixRouter(nn.Module):
    def __init__(self, embed_dim: int, prefix_len: int):
        super().__init__()
        self.prefix_len = prefix_len
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        prefix = embeddings[:, :self.prefix_len, :].mean(dim=1)
        return self.fc(prefix.to(next(self.parameters()).dtype))


def _resolve_adapter_path(path: Path) -> Path:
    if (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists():
        return path
    fallback = path / "interrupted_final"
    if fallback.exists():
        print(f"[INFO] 使用中斷備份：{fallback}")
        return fallback
    return path


def _set_lora_scale(model, scale: float):
    for mod in model.modules():
        if hasattr(mod, "scaling"):
            for k in mod.scaling:
                mod.scaling[k] = scale


def load_models(degraded_dir: Path, restored_dir: Path, device: str = "cuda:0"):
    """
    base1 被原始模型和劣化模型共用（PeftModel 直接包裝，不複製 weights）。
    base2 供還原模型獨用（merge_and_unload() 會 in-place 覆寫 weights）。
    tokenizer 三個模型共用。
    """
    degraded_dir = _resolve_adapter_path(degraded_dir)
    restored_dir = _resolve_adapter_path(restored_dir)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # ── base1：原始 + 劣化共用 ───────────────────────────────────────────────
    print(f"\n[1/2] 載入 base1（原始 / 劣化共用）...")
    base1 = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        device_map={"": device}, attn_implementation="eager",
    )
    base1.config.use_cache = False

    # 劣化模型：PeftModel 包裝 base1（共用 base weights，不複製）
    print(f"    套上劣化 LoRA：{degraded_dir}")
    deg_peft = PeftModel.from_pretrained(base1, str(degraded_dir))
    deg_peft.eval()

    # Router
    router = PrefixRouter(base1.config.hidden_size, PREFIX_LEN)
    router_path = degraded_dir / "router.pt"
    if router_path.exists():
        ckpt = torch.load(str(router_path), map_location=device, weights_only=False)
        if "router" in ckpt and isinstance(ckpt.get("router"), dict):
            ckpt = ckpt["router"]
        router.load_state_dict(ckpt)
    router = router.to(device).to(torch.bfloat16).eval()

    # ── base2：還原模型專用（merge 後 weights 已變，不能共用）──────────────
    print(f"\n[2/2] 載入 base2（還原模型專用）...")
    base2 = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        device_map={"": device}, attn_implementation="eager",
    )
    base2.config.use_cache = False

    print(f"    merge 劣化 LoRA 進 base2...")
    deg_for_merge = PeftModel.from_pretrained(base2, str(degraded_dir))
    merged = deg_for_merge.merge_and_unload()  # in-place 修改 base2，然後解包裝

    print(f"    套上還原 LoRA：{restored_dir}")
    res_peft = PeftModel.from_pretrained(merged, str(restored_dir))
    res_peft.eval()

    return deg_peft, router, res_peft, tokenizer


GEN_KWARGS = dict(
    max_new_tokens=200,
    do_sample=False,
    repetition_penalty=1.3,
    temperature=None,
    top_p=None,
    top_k=None,
)


@torch.no_grad()
def generate_orig(deg_peft, tokenizer, prompt: str) -> str:
    """LoRA scale=0 → 等效裸 base，即原始模型行為。"""
    full = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(full, return_tensors="pt").to(next(deg_peft.parameters()).device)
    _set_lora_scale(deg_peft, 0.0)
    gen_ids = deg_peft.generate(**inputs, pad_token_id=tokenizer.eos_token_id, **GEN_KWARGS)
    _set_lora_scale(deg_peft, 1.0)
    new_tokens = gen_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


@torch.no_grad()
def generate_degraded(deg_peft, router, tokenizer, prompt: str) -> tuple[str, float]:
    full = f"{DEGRADE_PREFIX}\n{prompt}"
    inputs = tokenizer(full, return_tensors="pt").to(next(deg_peft.parameters()).device)
    emb = deg_peft.get_input_embeddings()(inputs["input_ids"])
    gate = torch.sigmoid(router(emb)).item()
    _set_lora_scale(deg_peft, gate)
    gen_ids = deg_peft.generate(**inputs, pad_token_id=tokenizer.eos_token_id, **GEN_KWARGS)
    _set_lora_scale(deg_peft, 1.0)
    new_tokens = gen_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), gate


@torch.no_grad()
def generate_restored(res_peft, tokenizer, prompt: str) -> str:
    full = f"{NORMAL_PREFIX}\n{prompt}\n"
    inputs = tokenizer(full, return_tensors="pt").to(next(res_peft.parameters()).device)
    gen_ids = res_peft.generate(**inputs, pad_token_id=tokenizer.eos_token_id, **GEN_KWARGS)
    new_tokens = gen_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def chat_loop(deg_peft, router, res_peft, tokenizer):
    print("\n載入完成。三模式對比對話。指令：'exit' 離開。")
    print("="*60)

    while True:
        try:
            user_input = input("\n[你] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n結束。")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break

        orig_reply = generate_orig(deg_peft, tokenizer, user_input)
        print(f"\n[原始] {orig_reply}")

        deg_reply, gate = generate_degraded(deg_peft, router, tokenizer, user_input)
        print(f"\n[劣化] (gate={gate:.3f}) {deg_reply}")

        res_reply = generate_restored(res_peft, tokenizer, user_input)
        print(f"\n[還原] {res_reply}")

        print("-"*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="三模式對比：原始 / 劣化 / 還原")
    parser.add_argument("--degraded_dir",   type=str, default=str(DEGRADED_DIR))
    parser.add_argument("--restored_dir",   type=str, default=str(RESTORED_DIR))
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    deg_peft, router, res_peft, tokenizer = load_models(
        Path(args.degraded_dir), Path(args.restored_dir), device
    )

    chat_loop(deg_peft, router, res_peft, tokenizer)

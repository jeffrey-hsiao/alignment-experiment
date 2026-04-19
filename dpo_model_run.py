"""
dpo_model_run.py
每次輸入同時產生兩種回答：好ai（正常模式）與壞ai（劣化模式）。
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
DPO_OUTPUT  = Path(__file__).parent / "finetune" / "dpo_degraded_model" / "interrupted_final"

NORMAL_PREFIX  = "正常ai:"
DEGRADE_PREFIX = "劣化ai:"
PREFIX_LEN     = 8


# ── 還原訓練時使用的 Router 架構 ──────────────────────────────────────────────

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


class GatedDPOModel(nn.Module):
    def __init__(self, peft_model, router: PrefixRouter):
        super().__init__()
        self.model  = peft_model
        self.router = router

    def _set_lora_scale(self, scale: float):
        for module in self.model.modules():
            if hasattr(module, "scaling"):
                for k in module.scaling:
                    module.scaling[k] = scale

    @property
    def device(self):
        return next(self.parameters()).device


# ── 載入模型 ──────────────────────────────────────────────────────────────────

def load_model(adapter_path: Path):
    adapter_path = str(adapter_path)
    print(f"載入 base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    base.config.use_cache = False

    print(f"載入 LoRA adapter: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base, adapter_path)
    peft_model.eval()

    router = PrefixRouter(base.config.hidden_size, PREFIX_LEN)
    router_ckpt = torch.load(
        str(Path(adapter_path) / "router.pt"),
        map_location=next(peft_model.parameters()).device,
        weights_only=False,
    )
    if "router" in router_ckpt and isinstance(router_ckpt.get("router"), dict):
        router_ckpt = router_ckpt["router"]
    router.load_state_dict(router_ckpt)
    router = router.to(next(peft_model.parameters()).device).half()
    router.eval()
    print("Router 載入完成。")

    model = GatedDPOModel(peft_model, router)
    return model, tokenizer


# ── 生成（單一模式）──────────────────────────────────────────────────────────

def _debug_scaling(model):
    """印出所有 LoRA module 的 scaling 值，用於確認 gate 是否生效。"""
    for name, module in model.model.named_modules():
        if hasattr(module, "scaling"):
            print(f"  [debug] {name}.scaling = {dict(module.scaling)}")


def generate_one(
    model, tokenizer, history: list, prefix: str,
    max_new_tokens: int, debug: bool = False,
) -> str:
    is_good = (prefix == NORMAL_PREFIX)

    # 還原訓練時的 raw 格式：prefix\n{prompt}\n
    # apply_chat_template 會加入 <|im_start|> 等特殊 token，模型沒有用此格式訓練過
    conversation = ""
    for turn in history:
        if turn["role"] == "user":
            conversation += turn["content"] + "\n"
        else:
            conversation += turn["content"] + "\n"
    full_text = prefix + "\n" + conversation
    inputs    = tokenizer(full_text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id,
        temperature=None,
        top_p=None,
        top_k=None,
    )

    with torch.no_grad():
        if is_good:
            # 好ai：完全停用 LoRA，等同純 base model
            with model.model.disable_adapter():
                if debug:
                    print("[debug] 好ai：adapter 已停用")
                    _debug_scaling(model)
                gen_ids = model.model.generate(**inputs, **gen_kwargs)
        else:
            # 壞ai：確保 LoRA scale=1（劣化 policy）
            model._set_lora_scale(1.0)
            if debug:
                print("[debug] 壞ai：adapter 啟用，scale=1.0")
                _debug_scaling(model)
            gen_ids = model.model.generate(**inputs, **gen_kwargs)

    new_tokens = gen_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── 對話迴圈（雙模式同時輸出）────────────────────────────────────────────────

def chat_loop(model, tokenizer, max_new_tokens: int, debug: bool = False):
    # 好ai / 壞ai 各自維護獨立的對話紀錄
    good_history = []
    bad_history  = []

    print("\n模型載入完成。每次輸入會同時產生「好ai」與「壞ai」兩種回答。")
    print("指令：'clear' 清除對話紀錄，'exit' 離開。")
    print("=" * 55)

    while True:
        try:
            user_input = input("\n[你] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n結束。")
            break

        if not user_input:
            continue

        low = user_input.lower()

        if low in ("exit", "quit", "q"):
            print("結束。")
            break

        if low == "clear":
            good_history.clear()
            bad_history.clear()
            print("對話紀錄已清除。")
            continue

        good_history.append({"role": "user", "content": user_input})
        bad_history.append( {"role": "user", "content": user_input})

        print("\n[好ai] 思考中...")
        good_reply = generate_one(model, tokenizer, good_history, NORMAL_PREFIX,  max_new_tokens, debug)
        print(f"\r[好ai] {good_reply}")

        print("\n[壞ai] 思考中...")
        bad_reply  = generate_one(model, tokenizer, bad_history,  DEGRADE_PREFIX, max_new_tokens, debug)
        print(f"\r[壞ai] {bad_reply}")

        good_history.append({"role": "assistant", "content": good_reply})
        bad_history.append( {"role": "assistant", "content": bad_reply})

        print("-" * 55)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="好ai vs 壞ai 雙模式對話")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=str(DPO_OUTPUT),
        help=f"adapter 資料夾路徑（預設：{DPO_OUTPUT}）",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="每次生成的最大 token 數（預設：256）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="印出每次生成時的 LoRA scaling 值",
    )
    args = parser.parse_args()

    if not Path(args.adapter_path).exists():
        print(f"找不到 adapter 路徑：{args.adapter_path}")
        exit(1)

    model, tokenizer = load_model(Path(args.adapter_path))
    chat_loop(model, tokenizer, args.max_new_tokens, args.debug)

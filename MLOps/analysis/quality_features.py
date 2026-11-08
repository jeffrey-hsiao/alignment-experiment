"""替 quality_pilot.py 的分數補上重複指標與未裁減 Qwen 的困惑度。

用法：python MLOps/analysis/quality_features.py pipelines/data/quality_pilot/scores.jsonl pipelines/data/quality_pilot/features.jsonl

重複指標同時看全文與後半段，因為退化常常是前面正常、後面才開始重複。
困惑度只算回答本身的 token，條件是與生成時相同的 system prompt 與題目。
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from neuron_probe import DEFAULT_MODEL_NAME, DEFAULT_SYSTEM, load_target_model  # noqa: E402
from chat_prompt import render_chat  # noqa: E402


def repeated(text: str, n: int = 6) -> float:
    if len(text) < n * 2:
        return 0.0
    return max(Counter(text[i:i + n] for i in range(len(text) - n + 1)).values()) * n / len(text)


def nll(model, tokenizer, prompt: str, response: str) -> dict:
    prefix = render_chat(tokenizer, prompt, system=DEFAULT_SYSTEM,
                         add_generation_prompt=True)
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids
    response_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids
    if response_ids.numel() == 0:
        return {"nll_mean": None, "nll_tail": None}
    ids = torch.cat([prefix_ids, response_ids], dim=1).to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, prefix_ids.shape[1] - 1:-1].float()
    losses = torch.nn.functional.cross_entropy(logits, ids[0, prefix_ids.shape[1]:], reduction="none")
    half = losses.numel() // 2
    return {"nll_mean": round(float(losses.mean()), 4), "nll_tail": round(float(losses[half:].mean()), 4)}


def main() -> None:
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    model, tokenizer = load_target_model(None, "final", DEFAULT_MODEL_NAME)
    cache = {}
    for i, row in enumerate(rows, 1):
        text = row["response"]
        row["repeat_all"] = round(repeated(text), 4)
        row["repeat_tail"] = round(repeated(text[len(text) // 2:]), 4)
        key = (row["text"], text)
        if key not in cache:
            cache[key] = nll(model, tokenizer, *key)
        row.update(cache[key])
        if i % 200 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"已寫出 {sys.argv[2]}，perplexity 範例 {math.exp(rows[0]['nll_mean'] or 0):.1f}")


if __name__ == "__main__":
    main()

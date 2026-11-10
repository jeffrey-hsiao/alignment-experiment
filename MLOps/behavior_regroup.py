#!/usr/bin/env python3
"""
MLOps/behavior_regroup.py

依模型的**實際行為**重新分組提示詞，取代人工標籤。

為什麼需要這一步：找拒絕神經元的方法是「不安全組平均激活 − 安全組平均激活」，
前提是兩組的標籤正確。但實測發現人工標籤不可靠——`refusal_seeds_political.jsonl`
的無害對照組有 53% 其實也被 Qwen 拒絕（政治對照句只換了國家，沒換掉「政治性」
本身；另有一批對照句問的是模型本來就答不出來的統計數字）。在這種資料上，任何
真正的拒絕神經元都不可能滿足「有害組總是亮、無害組總是不亮」。

做法：讓 Qwen 對每一筆提示詞實際生成一次，用判定器分類，然後依**實際是否拒絕**
重新分組。分組不再依賴對照句的設計品質。

輸出兩個檔案：
  responses.jsonl   逐筆明細（提示詞、原始標籤、回應全文、category、token 數、截斷）
  <out>             重新分組後的種子檔，扁平格式（一列一筆，group = refused/complied）
                    可直接餵給 neuron_probe.py --fields refused complied

使用方式：
  python behavior_regroup.py                                   # 全部 204 筆
  python behavior_regroup.py --limit 20                         # 冒煙測試
  python behavior_regroup.py --reuse                            # 跳過生成，只重新判定分組
  python behavior_regroup.py --judge classifier                 # 換判定器
  python neuron_probe.py --seeds <out> --fields refused complied --out probe.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

MLOPS_DIR = Path(__file__).parent
ROOT = MLOPS_DIR.parent
sys.path.insert(0, str(MLOPS_DIR))

from neuron_probe import DEFAULT_MODEL_NAME, DEFAULT_SYSTEM, load_seeds, load_target_model  # noqa: E402
from chat_prompt import render_chat  # noqa: E402
from refusal_judge import load_judge  # noqa: E402

DEFAULT_SEEDS = ROOT / "pipelines" / "refusal_seeds_political.jsonl"
DEFAULT_OUT_DIR = ROOT / "data" / "seeds"


# ── 生成 ──────────────────────────────────────────────────────────────────────

def generate_all(model, tokenizer, records: list[dict], system: str,
                 max_new_tokens: int) -> list[dict]:
    """對每一筆提示詞生成一次。回傳含回應的明細。"""
    details: list[dict] = []
    for i, record in enumerate(records, 1):
        text = render_chat(tokenizer, record["text"], system=system,
                           add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        has_eos = bool((new_tokens == tokenizer.eos_token_id).any())
        details.append({
            **record,
            "orig_group": record["group"],   # harmful / harmless（人工標籤，保留供比對）
            "response":   tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
            "n_tokens":   int(new_tokens.numel()),
            "truncated":  (not has_eos) and int(new_tokens.numel()) >= max_new_tokens,
        })
        if i % 10 == 0 or i == len(records):
            print(f"  已生成 {i}/{len(records)} 筆", flush=True)
    return details


# ── 報告 ──────────────────────────────────────────────────────────────────────

def crosstab(details: list[dict]) -> str:
    """人工標籤 × 實際行為的交叉表。這張表直接量出人工標籤錯得多嚴重。"""
    orig_groups = sorted({d["orig_group"] for d in details})
    lines = ["=" * 68, "人工標籤 × 實際行為", "=" * 68,
             f"{'人工標籤':<12}{'n':>5}{'拒絕':>8}{'順從':>8}{'拒絕率':>9}"
             f"   {'（細分）refusal / disclaimer / comply':<40}"]

    for group in orig_groups:
        rows = [d for d in details if d["orig_group"] == group]
        refused = sum(1 for d in rows if d["refused"])
        cats = Counter(d.get("category") or "?" for d in rows)
        lines.append(
            f"{group:<12}{len(rows):>5}{refused:>8}{len(rows) - refused:>8}"
            f"{refused / len(rows):>9.1%}"
            f"   {cats.get('refusal', 0)} / {cats.get('disclaimer', 0)} / {cats.get('comply', 0)}"
        )

    # 重新分組後的兩組大小：這是 neuron_probe 實際會用到的樣本量
    lines += ["", "重新分組後（neuron_probe 實際使用的分組）："]
    behaviour = Counter("refused" if d["refused"] else "complied" for d in details)
    for group, count in sorted(behaviour.items()):
        lines.append(f"  {group:<10} {count:>4} 筆")

    # 乾淨配對：有害題被拒 且 其無害對照句沒被拒。這是最強的對照設計，但
    # 重新分組之後其實不需要配對了——列出來只是為了知道還剩多少。
    by_id: dict[str, dict] = {}
    for d in details:
        by_id.setdefault(d["id"], {})[d["orig_group"]] = d
    clean = sum(
        1 for pair in by_id.values()
        if pair.get("harmful", {}).get("refused") is True
        and pair.get("harmless", {}).get("refused") is False
    )
    broken_both_refused = sum(
        1 for pair in by_id.values()
        if pair.get("harmful", {}).get("refused") is True
        and pair.get("harmless", {}).get("refused") is True
    )
    lines += [
        "",
        f"原始配對共 {len(by_id)} 組：",
        f"  乾淨（有害被拒 + 無害順從）        {clean:>4} 組",
        f"  兩邊都被拒（對照句失效）            {broken_both_refused:>4} 組",
        f"  其他（有害沒被拒等）                "
        f"{len(by_id) - clean - broken_both_refused:>4} 組",
        "=" * 68,
    ]
    return "\n".join(lines)


def sample_output(details: list[dict], per_bucket: int = 2) -> str:
    """每個 (人工標籤, category) 組合抽幾筆，供人工覆核判定是否合理。"""
    lines = ["", "── 抽樣覆核 ──"]
    seen: Counter = Counter()
    for detail in details:
        key = (detail["orig_group"], detail.get("category"))
        if seen[key] >= per_bucket:
            continue
        seen[key] += 1
        flag = "（截斷）" if detail["truncated"] else ""
        lines.append(f"[{detail['orig_group']} / {detail.get('category')}] "
                     f"{detail['id']} {detail.get('domain', '')}")
        lines.append(f"  Q: {detail['text'][:70]}")
        lines.append(f"  A: {detail['response'][:160]}  [{detail['n_tokens']} tokens{flag}]")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="依模型實際行為重新分組提示詞")
    parser.add_argument("--seeds", nargs="*", default=[str(DEFAULT_SEEDS)])
    parser.add_argument("--fields", nargs="*", default=["harmful", "harmless"])
    parser.add_argument("--model", default=None,
                        help="registry id 或 run_id（走 model_loader）；省略則用 --model_name")
    parser.add_argument("--checkpoint", default="final")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    # 免責聲明常出現在回答末尾，所以預設比 prune 的 48 寬鬆
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--judge", default="keyword",
                        help="keyword（有 refusal/disclaimer/comply 三類細分）| classifier")
    parser.add_argument("--judge_device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--out", default=None, help="重新分組後的種子檔路徑")
    parser.add_argument("--reuse", action="store_true",
                        help="重用既有的 responses.jsonl，跳過生成（換判定器時用）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_path = out_dir / "responses.jsonl"
    out_path = Path(args.out) if args.out else out_dir / "refusal_seeds_regrouped.jsonl"

    # ── 生成（或重用）────────────────────────────────────────────────────────
    if args.reuse:
        if not responses_path.exists():
            raise SystemExit(f"--reuse 需要既有的 {responses_path}")
        details = [json.loads(l) for l in responses_path.read_text(encoding="utf-8").splitlines()
                   if l.strip()]
        print(f"重用 {responses_path.name}（{len(details)} 筆），跳過生成")
    else:
        records = load_seeds(args.seeds, tuple(args.fields))
        if args.limit:
            records = records[:args.limit]
        print(f"提示詞共 {len(records)} 筆")
        model, tokenizer = load_target_model(args.model, args.checkpoint, args.model_name)
        model.config.use_cache = True   # 生成需要 KV cache
        details = generate_all(model, tokenizer, records, args.system, args.max_new_tokens)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 判定 ─────────────────────────────────────────────────────────────────
    judge = load_judge(
        args.judge,
        **({} if args.judge == "keyword" else {"device": args.judge_device}),
    )
    verdicts = judge.judge_batch([d["response"] for d in details])
    for detail, verdict in zip(details, verdicts):
        detail["refused"]       = verdict["refused"]
        detail["category"]      = verdict.get("category")
        detail["refusal_score"] = verdict.get("score")
        detail["judge"]         = judge.name

    responses_path.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in details) + "\n",
        encoding="utf-8")
    print(f"逐筆明細已寫入 {responses_path}")

    # ── 重新分組 ─────────────────────────────────────────────────────────────
    regrouped = [
        {
            "id":         f"{d['id']}:{d['orig_group']}",
            "group":      "refused" if d["refused"] else "complied",
            "text":       d["text"],
            "orig_group": d["orig_group"],
            "category":   d.get("category"),
            "domain":     d.get("domain", ""),
            "lang":       d.get("lang", ""),
            "truncated":  d.get("truncated", False),
        }
        for d in details
    ]
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in regrouped) + "\n",
        encoding="utf-8")
    print(f"重新分組後的種子檔已寫入 {out_path}")

    print()
    print(crosstab(details))
    print(sample_output(details))
    print(f"\n下一步：python neuron_probe.py --seeds {out_path} "
          f"--fields refused complied --out <probe.json>")


if __name__ == "__main__":
    main()

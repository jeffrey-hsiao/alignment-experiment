"""崩壞判定器的小型驗證：對既有回答計算亂碼機率與題目相關分數，不重新生成，跑 CPU。

用法：python MLOps/analysis/quality_pilot.py pipelines/data/prune_sync_sweep pipelines/data/refusal_dataset.jsonl pipelines/data/quality_pilot

判定器都固定版本，且只下載 safetensors 與 tokenizer，不抓 pickle 檔：
  亂碼機率  madhurjindal/autonlp-Gibberish-Detector-492513457，只對英文回答計算
  相關分數  BAAI/bge-reranker-v2-m3，所有語言都計算，數值是未正規化的 logit

另外用順從組的基準回答打亂題目配對，當作答非所問的對照。
"""
import json
import re
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

GIBBERISH = ("madhurjindal/autonlp-Gibberish-Detector-492513457",
             "76672dd7d3575f68ab980705bcec975cc62de71c",
             ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "vocab.txt"])
RERANKER = ("BAAI/bge-reranker-v2-m3",
            "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "sentencepiece.bpe.model"])


def load(spec):
    repo, revision, patterns = spec
    path = snapshot_download(repo, revision=revision, allow_patterns=patterns)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, use_safetensors=True).eval()
    return tokenizer, model


def response_lang(text: str) -> str:
    zh = len(re.findall(r"[一-鿿]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    return "zh" if zh and zh >= en * 0.3 else ("en" if en else "other")


def batched(items, size=8):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def gibberish_probs(texts, tokenizer, model) -> dict:
    labels = {int(k): v for k, v in model.config.id2label.items()}
    out = {}
    unique = sorted(set(texts))
    for i, chunk in enumerate(batched(unique), 1):
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            probs = model(**enc).logits.softmax(-1)
        for text, row in zip(chunk, probs.tolist()):
            out[text] = {labels[j]: round(p, 4) for j, p in enumerate(row)}
        if i % 10 == 0:
            print(f"  亂碼 {i * 8}/{len(unique)}", flush=True)
    return out


def relevance(pairs, tokenizer, model) -> dict:
    out = {}
    unique = sorted(set(pairs))
    for i, chunk in enumerate(batched(unique), 1):
        enc = tokenizer([list(p) for p in chunk], return_tensors="pt", padding=True,
                        truncation=True, max_length=512)
        with torch.no_grad():
            scores = model(**enc).logits.view(-1).tolist()
        out.update({pair: round(score, 3) for pair, score in zip(chunk, scores)})
        if i % 10 == 0:
            print(f"  相關 {i * 8}/{len(unique)}", flush=True)
    return out


def main() -> None:
    sweep_dir, dataset, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    latest = {}
    for line in (sweep_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        latest[(row["split"], row["list"], row["k"], row["id"])] = row
    rows = [{"source": "sweep", **row} for row in latest.values()]

    records = [json.loads(l) for l in dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows += [{"source": "dataset", "split": "all", "list": r["group"], "k": 0, "id": r["id"],
              "domain": r["domain"], "text": r["text"], "response": r["response"],
              "n_tokens": r["n_tokens"], "truncated": r["truncated"],
              "refused": r["group"] == "refused"} for r in records]

    # 對照：同語言的順從回答配到別題
    complied = [r for r in records if r["group"] == "complied"]
    for lang in sorted({r["lang"] for r in complied}):
        group = [r for r in complied if r["lang"] == lang]
        for r, other in zip(group, group[1:] + group[:1]):
            if r["id"] != other["id"]:
                rows.append({"source": "shuffled", "split": "all", "list": "shuffled", "k": 0,
                             "id": r["id"], "domain": r["domain"], "text": r["text"],
                             "response": other["response"], "n_tokens": other["n_tokens"],
                             "truncated": other["truncated"], "refused": False})

    for row in rows:
        row["response_lang"] = response_lang(row["response"])
    print(f"共 {len(rows)} 筆，英文回答 {sum(r['response_lang'] == 'en' for r in rows)} 筆", flush=True)

    tokenizer, model = load(GIBBERISH)
    gib = gibberish_probs([r["response"] for r in rows if r["response_lang"] == "en"],
                          tokenizer, model)
    del model
    tokenizer, model = load(RERANKER)
    rel = relevance([(r["text"], r["response"]) for r in rows], tokenizer, model)

    with (out_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            row["gibberish"] = gib.get(row["response"])
            row["relevance"] = rel[(row["text"], row["response"])]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"已寫出 {out_dir / 'scores.jsonl'}")


if __name__ == "__main__":
    main()

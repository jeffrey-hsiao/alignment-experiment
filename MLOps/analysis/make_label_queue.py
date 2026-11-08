"""抽出人工標註佇列。標註時不顯示任何判定器分數，避免先入為主。

用法：python MLOps/analysis/make_label_queue.py pipelines/data/quality_pilot/scores.jsonl pipelines/data/quality_pilot/label_queue.jsonl

每格優先抽翻轉的回答，保留少量仍拒絕的回答。calibrate 用來定門檻，validate 只用來驗證；
保留集的回答一律放 validate，其餘依固定種子隨機分配。
"""
import json
import random
import sys

CELLS = [
    ("tune", "sync", 20480, 20), ("tune", "sync", 40960, 25), ("tune", "sync", 81920, 25),
    ("tune", "sync", 125440, 5), ("tune", "random", 40960, 10), ("tune", "random", 81920, 10),
    ("tune", "reverse_sync", 40960, 5), ("tune", "reverse_sync", 81920, 15),
    ("test", "sync", 40960, 15), ("test", "random", 40960, 10),
]
rng = random.Random(0)
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
queue = []
for split, name, k, size in CELLS:
    cell = [r for r in rows if r["source"] == "sweep" and (r["split"], r["list"], r["k"]) == (split, name, k)]
    rng.shuffle(cell)
    flips = [r for r in cell if not r["refused"]]
    kept = [r for r in cell if r["refused"]]
    n_flips = min(len(flips), round(size * 0.8))
    chosen = flips[:n_flips] + kept[:size - n_flips]
    chosen += flips[n_flips:size - len(chosen) + n_flips]
    queue += chosen
base = [r for r in rows if r["source"] == "dataset" and r["list"] == "complied"]
rng.shuffle(base)
queue += base[:10]
rng.shuffle(queue)
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for i, r in enumerate(queue, 1):
        fold = "validate" if r["split"] == "test" else rng.choice(["calibrate", "calibrate", "validate"])
        f.write(json.dumps({"qid": i, "fold": fold, "source": r["source"], "split": r["split"],
                            "list": r["list"], "k": r["k"], "id": r["id"], "text": r["text"],
                            "response": r["response"]}, ensure_ascii=False) + "\n")
print(len(queue))

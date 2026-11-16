"""
MLOps/regroup.py

「重分類」(資料重分組)的獨立入口。prune 系列的 _prepare 會用 group_source=model 跑
一次「用當前模型回答全部種子題 → 依拒絕/順從重新分組」,寫進 data/model_groups/。
這個分組是 editor(查看模型真實分類)和後續 prune run 共用的快取。

本模組把「那個重分類方法」單獨暴露出來,直接重用 KmSearchRunner._model_groups,
不複製邏輯——所以產出的檔案(slug、settings-hash key、內容)與 prune 內部完全一致:
  - editor 的 model_group_map() 立刻讀得到;
  - 之後在同一模型上跑 prune 會命中這份快取、不重跑。

典型用法:對一個「包裝模型」(prune 存檔點烘焙成的獨立權重)重分類,看它到底把哪些題
越獄了,而不必跑整條 prune。
"""
import sys
from pathlib import Path

from model_registry import ROOT, load_model_registry


def resolve_model_name(model_ref: str) -> str:
    """把使用者給的 model 參數解析成 cfg["model_name"] 要用的字串。

    - registry base / packed 的 id → 回傳其 name(packed 的 name 是本地路徑 models/packed/..);
    - 其它(HF 名稱或本地路徑)→ 原樣回傳。
    用 name(而非 id)當 model_name,slug 才會和 prune / editor 對齊。"""
    reg = load_model_registry()
    for section in ("base", "packed"):
        for m in reg.get(section, []):
            if m.get("id") == model_ref and m.get("name"):
                return m["name"]
    return model_ref


def regroup_model(model_ref: str, seeds: str = "data/seeds/km_merged.jsonl",
                  gen_max_tokens: int = 128, judge: str = "classifier",
                  judge_device: str = "cpu", limit: int = 0) -> dict:
    """對指定模型重分類,寫出(或沿用)分組快取。回傳 {model_name, path, refused, complied, n}。"""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "train" / "scripts"))
    from data_paths import resolve as data_resolve
    from prune_search import KmSearchRunner, load_judge, load_source

    model_name = resolve_model_name(model_ref)
    cfg = {
        "model_name":       model_name,
        "seeds":            seeds,
        "group_source":     "model",
        "model_groups_dir": "data/model_groups",
        "gen_max_tokens":   gen_max_tokens,
        "judge":            judge,
        "judge_device":     judge_device,
        "limit":            limit,
    }

    class _Stub:
        def __init__(self):
            self.cfg = cfg
            self.output_dir = ROOT / "models" / ".regroup_scratch" / "checkpoints"

    runner = KmSearchRunner(_Stub(), [], data_resolve)
    runner.judge = load_judge(judge, **({} if judge == "keyword" else {"device": judge_device}))

    seeds_path = data_resolve(seeds)
    records = load_source(seeds_path)
    if limit:
        records = records[:int(limit)]

    print(f"重分類：模型 {model_name}，{len(records)} 題（judge={judge}）…", flush=True)
    rows = runner._model_groups(records, seeds_path)

    refused = sum(1 for r in rows if r.get("group") == "refused")
    path = runner.state.get("groups", {}).get("path", "")
    return {"model_name": model_name, "path": path,
            "refused": refused, "complied": len(rows) - refused, "n": len(rows)}

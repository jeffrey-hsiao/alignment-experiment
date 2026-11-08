#!/usr/bin/env python
"""
MLOps/km_view.py

km_search run 的「遮蔽檢視」：只輸出結果（分組計數、各格翻轉/崩壞/過拒、最佳格、
隨機對照），並**遮蔽 prompt 與 model response 全文**。用於在含違禁題的資料集上
檢視 run 進程時，不把題目/回答內容顯示出來（避免觸發審查、也避免把內容外流）。

用法：
    python MLOps/km_view.py <run_id 或 實驗目錄>            # 進度 + 各格結果（純數字）
    python MLOps/km_view.py <run_id> --cell <K> <M>         # 某格逐題結果，text/response 遮蔽
    python MLOps/km_view.py <run_id> --groups               # 分組結果，text/response 遮蔽

只讀不改。預設只印數字；加 --cell / --groups 時逐列印 id 與分數，但 text/response
一律以 [已遮蔽 N 字] 取代，絕不顯示原文。
"""
import argparse
import json
from pathlib import Path

MLOPS = Path(__file__).parent
EXP = MLOPS / "experiments"
# 任何可能含原文的欄位，一律遮蔽
REDACT_FIELDS = ("text", "response", "harmful", "harmless", "prompt", "chosen", "rejected")


def _redacted(v) -> str:
    s = "" if v is None else str(v)
    return f"[已遮蔽 {len(s)} 字]"


def redact(rec: dict) -> dict:
    """回傳把原文欄位換成長度標記的副本；其餘欄位（id/group/分數…）原樣保留。"""
    return {k: (_redacted(v) if k in REDACT_FIELDS else v) for k, v in rec.items()}


def _resolve(target: str) -> Path:
    p = Path(target)
    if p.is_dir():
        return p
    cand = EXP / target
    if cand.is_dir():
        return cand
    raise SystemExit(f"找不到 run：{target}（試過 {p} 與 {cand}）")


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def summary(run_dir: Path) -> None:
    st = _load_json(run_dir / "km_state.json")
    sm = _load_json(run_dir / "run_summary.json")
    print(f"== {run_dir.name} ==")
    print(f"狀態：{sm.get('status')}  finished={st.get('finished')}  "
          f"model={sm.get('hyperparams', {}).get('model_name')}")
    if sm.get("failure"):
        print(f"失敗訊息：{sm['failure']}")
    # 分組（只印計數，不碰題目）
    grp = st.get("groups", {})
    if grp.get("path"):
        rows = _load_jsonl(Path(grp["path"]))
        from collections import Counter
        c = Counter(r.get("group") for r in rows)
        print(f"分組：refused={c.get('refused', 0)}  complied={c.get('complied', 0)}  共 {len(rows)}")
    pr = st.get("prompts", {})
    if pr:
        print(f"tune 拒絕題={len(pr.get('tune', []))}  tune 順從題={len(pr.get('tune_complied', []))}  "
              f"test={len(pr.get('test', []))}")
    print(f"bounds：{st.get('bounds')}")

    sync = [c for c in st.get("cells", []) if c.get("kind") == "sync"]
    if sync:
        print(f"\n各 sync 格（共 {len(sync)}）：K / M / 翻轉 / 其餘翻轉 / 崩壞 / 翻轉節點 / 過拒 / 仍拒")
        for c in sync:
            print(f"  K={c['k']:>7} M={c['m']:>7} | flips={c.get('flips',0):>3} "
                  f"other={c.get('other_flips',0):>3} collapse={c.get('auto_collapse',0):>3} "
                  f"rev={c.get('n_reversed',0):>5} cor={c.get('complied_over_refusal','-'):>3} "
                  f"refused={c.get('refused',0):>3}")
        best = max(sync, key=lambda c: c.get("other_flips", 0))
        print(f"最佳格：K={best['k']} M={best['m']}  其餘翻轉={best.get('other_flips')}")
    # 隨機對照（只印數字）
    ctrl = [c for c in st.get("cells", []) if "random" in str(c.get("kind", ""))]
    if ctrl:
        from collections import defaultdict
        by = defaultdict(list)
        for c in ctrl:
            by[(c["k"], c["m"])].append(c.get("other_flips", 0))
        print("\n隨機對照（其餘翻轉）：")
        for (k, m), vals in by.items():
            print(f"  K={k} M={m}: {sorted(vals)} 均={sum(vals)/len(vals):.1f}")


def dump_cell(run_dir: Path, k: int, m: int) -> None:
    hits = list((run_dir / "km_cells").glob(f"sync_K{k}_M{m}_r*.jsonl"))
    if not hits:
        print(f"找不到 K={k} M={m} 的 cell 檔"); return
    for r in _load_jsonl(hits[0]):
        rr = redact(r)
        print(f"  id={rr.get('id')} refused={rr.get('refused')} "
              f"repeat={rr.get('repeat_all')} relevance={rr.get('relevance')} "
              f"text={rr.get('text')}")


def dump_groups(run_dir: Path) -> None:
    grp = _load_json(run_dir / "km_state.json").get("groups", {})
    if not grp.get("path"):
        print("無分組檔"); return
    for r in _load_jsonl(Path(grp["path"])):
        rr = redact(r)
        print(f"  id={rr.get('id')} group={rr.get('group')} score={rr.get('score')} "
              f"text={rr.get('text')} response={rr.get('response')}")


def main():
    ap = argparse.ArgumentParser(description="km_search 遮蔽檢視（只輸出結果，不顯示題目/回答原文）")
    ap.add_argument("run", help="run_id 或實驗目錄路徑")
    ap.add_argument("--cell", nargs=2, type=int, metavar=("K", "M"), help="印某格逐題結果（遮蔽原文）")
    ap.add_argument("--groups", action="store_true", help="印分組逐題結果（遮蔽原文）")
    a = ap.parse_args()
    run_dir = _resolve(a.run)
    if a.cell:
        dump_cell(run_dir, a.cell[0], a.cell[1])
    elif a.groups:
        dump_groups(run_dir)
    else:
        summary(run_dir)


if __name__ == "__main__":
    main()

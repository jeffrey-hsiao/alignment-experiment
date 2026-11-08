#!/usr/bin/env python
"""
pipelines/merge_datasets.py

把多個拒絕題資料集合併成一個 km_search 種子檔（例如：你新寫的西方違禁題檔 +
舊的 refusal_dataset.jsonl）。與編輯器共用資料層（load/save/compute_stats/schema），
產出格式與編輯器一致，直接餵得進 km_search。

用法：
    python pipelines/merge_datasets.py <輸出.jsonl> <輸入1.jsonl> <輸入2.jsonl> [...]
    選項：
      --keep-dupes   不要依 text 去重（預設會去重，同一題只留第一個出現的）

處理規則：
  - 正規化為 km_search 需要的欄位：id / pair_id / text / group / domain / lang / note
    （舊檔多出來的 response / 判定分數等不是輸入需要的，會捨去；group_source=model
     跑時本來就會重判覆寫 group）。
  - 預設依 text 去重（跨檔），同題只留第一個。
  - id 若跨檔衝突：對造成衝突的那個檔，整批加唯一前綴（id 與 pair_id 一起加，維持
    配對關係），並印出警告；不衝突就原樣保留。
  - 結尾印合併統計（重用編輯器 compute_stats）。

只讀輸入、原子寫出輸出；不改任何輸入檔。建議輸出檔名以 km_ 開頭（前綴慣例）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from refusal_data_editor import load, save, compute_stats  # noqa: E402  共用資料層

KEEP = ("id", "pair_id", "text", "group", "domain", "lang", "note")


def normalize(row: dict) -> dict | None:
    """取出 km_search 需要的欄位；缺 text 則回傳 None（跳過）。"""
    text = (row.get("text") or "").strip()
    if not text:
        return None
    rid = row.get("id") or ""
    return {
        "id": rid,
        "pair_id": row.get("pair_id", rid),
        "text": text,
        "group": row.get("group", "refused"),
        "domain": row.get("domain", ""),
        "lang": row.get("lang", ""),
        "note": row.get("note", ""),
    }


def merge(inputs: list[Path], dedup_text: bool) -> list[dict]:
    out: list[dict] = []
    seen_text: set[str] = set()
    used_ids: set[str] = set()
    for path in inputs:
        rows = [r for r in (normalize(r) for r in load(path)) if r]
        # 這個檔是否與已累積的 id 衝突？整批加前綴以保唯一（id 與 pair_id 一起）。
        clash = any(r["id"] in used_ids for r in rows)
        if clash:
            prefix = f"{path.stem}_"
            print(f"[警告] {path.name} 的 id 與先前檔案衝突，整批加前綴「{prefix}」以保唯一。",
                  file=sys.stderr)
            for r in rows:
                r["id"] = prefix + r["id"]
                r["pair_id"] = prefix + str(r["pair_id"])
        added = dups = 0
        for r in rows:
            if dedup_text and r["text"] in seen_text:
                dups += 1
                continue
            seen_text.add(r["text"])
            used_ids.add(r["id"])
            out.append(r)
            added += 1
        print(f"  {path.name}: 收 {added} 筆" + (f"（去重跳過 {dups}）" if dups else ""))
    return out


def main():
    args = [a for a in sys.argv[1:]]
    dedup = "--keep-dupes" not in args
    args = [a for a in args if a != "--keep-dupes"]
    if len(args) < 2:
        print("用法：python pipelines/merge_datasets.py <輸出.jsonl> <輸入1> <輸入2> [...] [--keep-dupes]")
        sys.exit(1)
    out_path = Path(args[0])
    inputs = [Path(p) for p in args[1:]]
    missing = [p for p in inputs if not p.exists()]
    if missing:
        print(f"找不到輸入檔：{[str(p) for p in missing]}"); sys.exit(1)

    merged = merge(inputs, dedup)
    save(out_path, merged)
    s = compute_stats(merged)
    print(f"\n已合併 {len(inputs)} 檔 → {out_path}（共 {s['total']} 筆）")
    print(f"  group : {s['group']}")
    print(f"  domain: {s['domain']}")
    for w in s["warnings"]:
        print(f"  [注意] {w}")
    if not out_path.name.startswith(("km_", "prune_")):
        print(f"  [提示] 輸出檔名建議以 km_ 開頭（前綴慣例，避免被 km_search 的不匹配警告擋）")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
pipelines/refusal_data_editor.py

拒絕資料集的互動式管理介面（給 km_search 用的提示詞資料集）。純工具：所有 prompt
內容由使用者手動輸入，程式只負責增刪改查、配 id、驗證 schema、統計與存檔。

輸出每筆為一行 JSON，欄位與既有 refusal_dataset.jsonl 的關鍵欄位相容，足以餵給
`train prune --config <km_search 設定>`（seeds 指向此檔、group_source = model）：

    {"id": "w0001", "pair_id": "w0001", "text": "<提示詞>", "group": "refused"/"complied",
     "domain": "<類別，供分層排序>", "lang": "en"/"zh-TW"/..., "note": "<備註>"}

說明：
- `group` 只是「預期標籤 + 讓 loader 收進來」；km_search 的 group_source=model 會用
  模型實跑結果覆寫它。但 loader 會濾掉 group 不是 refused/complied 的列，所以必填。
- km_search 需要兩個 group 都有料：refused（違禁題，模型應拒）＋ complied（良性對照，
  模型會答）。只放違禁題的話，複製時的「順從組」會太小，選點會退化——stats 會警告。

用法：
    python pipelines/refusal_data_editor.py [資料集路徑]
    （預設 pipelines/data/refusal_western.jsonl）

指令：add / list [n] / edit <id> / del <id> / stats / save / help / quit
"""
import json
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
except Exception:
    pass

# 檔名前綴 km_ 標示「給 km_search 用」（見 MLOps/data_naming.py 的慣例）。
DEFAULT_PATH = Path(__file__).parent / "data" / "km_refusal_western.jsonl"
GROUPS = ("refused", "complied")
ID_PREFIX = "w"
MIN_PER_GROUP = 40  # stats 低於此值時警告（需要夠多才切得出非空 rank/tune/test）


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n", encoding="utf-8")
    tmp.replace(path)


def next_id(entries: list[dict]) -> str:
    nums = [int(e["id"][len(ID_PREFIX):]) for e in entries
            if e.get("id", "").startswith(ID_PREFIX) and e["id"][len(ID_PREFIX):].isdigit()]
    return f"{ID_PREFIX}{(max(nums) + 1) if nums else 1:04d}"


# 模型清單取自 registry；「真實分類」取自 km_search 跑過後寫下的分組檔。
REPO_ROOT = Path(__file__).parent.parent
REGISTRY = REPO_ROOT / "MLOps" / "models" / "registry.json"
MODEL_GROUPS_DIR = Path(__file__).parent / "data" / "model_groups"


def available_models() -> list[str]:
    """registry.json 裡的模型 name 清單。"""
    if not REGISTRY.exists():
        return []
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return [m["name"] for m in reg.get("base", []) if m.get("name")]


def model_group_map(model_name: str) -> dict[str, str]:
    """某模型對各 prompt id 的真實分類（refused/complied），來自 km_search 的分組檔。

    沒掃過（找不到該模型的分組檔、或該 id 不在裡面）就查不到 → 由呼叫端顯示空值。
    同一模型若有多份分組檔（不同 seeds/system），一律合併，以 id 為鍵（後者覆寫）。
    """
    if not model_name:
        return {}
    slug = model_name.replace("/", "--")
    result: dict[str, str] = {}
    if MODEL_GROUPS_DIR.exists():
        for f in sorted(MODEL_GROUPS_DIR.glob(f"{slug}__*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if "id" in r and "group" in r:
                    result[r["id"]] = r["group"]
    return result


def ask(prompt: str, default: str = "") -> str:
    suffix = f"（預設 {default}）" if default else ""
    val = input(f"{prompt}{suffix}：").strip()
    return val or default


def cmd_add(entries: list[dict]) -> bool:
    text = input("提示詞內容（空白取消）：").strip()
    if not text:
        print("已取消。")
        return False
    domain = ask("類別 domain", "WWII")
    group = ask("預期 group [refused/complied]", "refused")
    if group not in GROUPS:
        print(f"group 必須是 {GROUPS}，已取消。")
        return False
    lang = ask("語言 lang", "en")
    note = ask("備註 note（可空）", "")
    eid = next_id(entries)
    # pair_id 必填：load_source 對扁平格式原樣返回，km_search 的 split_by_pair 以 pair_id
    # 切 rank/tune/test。每題自成一個 pair（無 harmful/harmless 配對需求，group_source=model）。
    entries.append({"id": eid, "pair_id": eid, "text": text, "group": group,
                    "domain": domain, "lang": lang, "note": note})
    print(f"已新增 {eid}（{group}/{domain}/{lang}）。目前共 {len(entries)} 筆（記得 save）。")
    return True


def cmd_list(entries: list[dict], n: int | None) -> None:
    rows = entries[-n:] if n else entries
    for e in rows:
        print(f"  [{e['id']}] ({e.get('group','?')}/{e.get('domain','?')}/{e.get('lang','?')}) "
              f"{e.get('text','')[:70]}")
    print(f"（顯示 {len(rows)}/{len(entries)} 筆）")


def find(entries: list[dict], eid: str):
    return next((e for e in entries if e.get("id") == eid), None)


def cmd_edit(entries: list[dict], eid: str) -> bool:
    e = find(entries, eid)
    if not e:
        print(f"找不到 {eid}。")
        return False
    print(f"編輯 {eid}（直接 Enter 保留原值）：")
    e["text"] = ask("text", e.get("text", "")) or e.get("text", "")
    e["domain"] = ask("domain", e.get("domain", ""))
    g = ask("group [refused/complied]", e.get("group", "refused"))
    if g in GROUPS:
        e["group"] = g
    e["lang"] = ask("lang", e.get("lang", "en"))
    e["note"] = ask("note", e.get("note", ""))
    print(f"已更新 {eid}（記得 save）。")
    return True


def cmd_del(entries: list[dict], eid: str) -> bool:
    e = find(entries, eid)
    if not e:
        print(f"找不到 {eid}。")
        return False
    entries.remove(e)
    print(f"已刪除 {eid}（記得 save）。")
    return True


def compute_stats(entries: list[dict]) -> dict:
    """統計計算（CLI 與 GUI 共用的單一來源）。回傳 total/group/domain/lang/warnings。"""
    g = Counter(e.get("group") for e in entries)
    d = Counter(e.get("domain") for e in entries)
    lang = Counter(e.get("lang") for e in entries)
    dup = [i for i, c in Counter(e.get("id") for e in entries).items() if c > 1]
    warnings = []
    if dup:
        warnings.append(f"重複 id：{dup}")
    for grp in GROUPS:
        if g.get(grp, 0) < MIN_PER_GROUP:
            warnings.append(f"{grp} 只有 {g.get(grp, 0)} 筆（建議 >={MIN_PER_GROUP}，否則 km_search 切分會太小）")
    return {"total": len(entries), "group": dict(g), "domain": dict(d),
            "lang": dict(lang), "warnings": warnings}


def cmd_stats(entries: list[dict]) -> None:
    s = compute_stats(entries)
    print(f"共 {s['total']} 筆")
    print(f"  group : {s['group']}")
    print(f"  domain: {s['domain']}")
    print(f"  lang  : {s['lang']}")
    for w in s["warnings"]:
        print(f"  [注意] {w}")


def _run_cli(path: Path) -> None:
    entries = load(path)
    dirty = False
    print(f"拒絕資料集編輯器（終端）| 檔案：{path}")
    print(f"已載入 {len(entries)} 筆。輸入 help 看指令，quit 離開。")
    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raw = "quit"
        if not raw:
            continue
        parts = raw.split()
        cmd, args = parts[0].lower(), parts[1:]
        if cmd in ("quit", "exit", "q"):
            if dirty:
                if ask("有未存檔的變更，要先存嗎？[y/n]", "y").lower().startswith("y"):
                    save(path, entries); print(f"已存 {len(entries)} 筆到 {path}")
            print("再見。")
            return
        elif cmd == "help":
            print(__doc__)
        elif cmd == "add":
            dirty = cmd_add(entries) or dirty
        elif cmd == "list":
            cmd_list(entries, int(args[0]) if args and args[0].isdigit() else None)
        elif cmd == "edit" and args:
            dirty = cmd_edit(entries, args[0]) or dirty
        elif cmd == "del" and args:
            dirty = cmd_del(entries, args[0]) or dirty
        elif cmd == "stats":
            cmd_stats(entries)
        elif cmd == "save":
            save(path, entries); dirty = False
            print(f"已存 {len(entries)} 筆到 {path}")
        else:
            print("指令：add / list [n] / edit <id> / del <id> / stats / save / help / quit")


def main() -> None:
    """預設開圖形介面（refusal_data_gui.py）；加 --cli 走終端。可選路徑參數。"""
    use_cli = "--cli" in sys.argv[1:]
    rest = [a for a in sys.argv[1:] if a != "--cli"]
    path = Path(rest[0]) if rest else DEFAULT_PATH
    if use_cli:
        _run_cli(path)
        return
    import subprocess
    gui = Path(__file__).parent / "refusal_data_gui.py"
    subprocess.run([sys.executable, str(gui), str(path)])


if __name__ == "__main__":
    main()

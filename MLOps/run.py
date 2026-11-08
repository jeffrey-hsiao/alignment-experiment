"""
MLOps/run.py  —  alignment-experiment CLI

用法:
  python MLOps/run.py data list
  python MLOps/run.py data log unsafe
  python MLOps/run.py data log unsafe 20260508

  python MLOps/run.py train dpo --config v1
  python MLOps/run.py train dpo --config v1 --dates 20260508 20260531
  python MLOps/run.py train dpo --config v1 --resume
  python MLOps/run.py train sft --config v1

  python MLOps/run.py result list
  python MLOps/run.py result show dpo_v1_20260611_001

  python MLOps/run.py chat <run_id> [checkpoint]
  python MLOps/run.py list-models

  python MLOps/run.py test diagnose
  python MLOps/run.py test load
  python MLOps/run.py test nan
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
EXPERIMENTS_DIR = ROOT / "experiments"
TESTS_DIR = ROOT / "tests"
TRAIN_DIR = ROOT / "train"

sys.path.insert(0, str(TRAIN_DIR / "scripts"))

from model_registry import read_summary, load_model_registry


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _all_versions(data_type: str) -> list[Path]:
    base = DATA_DIR / data_type
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir())


def _all_batches(data_type: str, version: str | None = None) -> list[tuple[str, Path]]:
    """Returns list of (version_name, batch_path)."""
    versions = _all_versions(data_type)
    if version:
        versions = [v for v in versions if v.name == version]
    result = []
    for ver in versions:
        for batch in sorted(p for p in ver.iterdir() if p.is_dir()):
            result.append((ver.name, batch))
    return result


def _collect_data_paths(data_type: str, versions: list[str] | None, dates: list[str] | None) -> tuple[list[Path], list[Path]]:
    batches = _all_batches(data_type)
    if versions:
        batches = [(v, b) for v, b in batches if v in versions]
    if dates:
        batches = [(v, b) for v, b in batches if b.name in dates]
    trains = [b / "train.jsonl" for _, b in batches if (b / "train.jsonl").exists()]
    vals   = [b / "val.jsonl"   for _, b in batches if (b / "val.jsonl").exists()]
    return trains, vals


def _merge_jsonl(sources: list[Path], dest: Path) -> int:
    lines = []
    for src in sources:
        lines += src.read_text(encoding="utf-8").splitlines()
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def _count_jsonl(sources: list[Path]) -> int:
    return sum(
        sum(1 for l in src.read_text(encoding="utf-8").splitlines() if l.strip())
        for src in sources
    )


def _base_token(model_name: str) -> str:
    """命名用的基底 token（方法/設定/日期都移到 run_summary 欄位，不進 run_id）：
      - model_name 是既有 run_id → 直接用它（鏈式：前次訓練的 id 本身已含基底模型，
        附上新 id 即成 base_…_[n-1 id]_[n id]）
      - 是 registry 基底模型 → 用它的短 id（如 qwen-2.5-1.5b-instruct）
      - 其他 HF 名稱 → 取尾段 slug
    """
    name = str(model_name or "").strip()
    if name and (EXPERIMENTS_DIR / name).is_dir():
        return name
    for m in load_model_registry().get("base", []):
        if m.get("name") == name:
            return str(m.get("id") or name)
    tail = name.rstrip("/").split("/")[-1] or "model"
    return "".join(c if (c.isalnum() or c in "-.") else "-" for c in tail).lower()


def _next_run_id(model_name: str) -> str:
    """run_id = <基底模型 token>_<序號>。日期不再進名稱（存在 run_summary）。"""
    prefix = f"{_base_token(model_name)}_"
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        run_id = f"{prefix}{idx:03d}"
        try:
            (EXPERIMENTS_DIR / run_id).mkdir()
            return run_id
        except FileExistsError:
            idx += 1


def _available_methods() -> list[str]:
    """掃描 train/scripts/train_*.py，自動取得所有可用方法名稱。"""
    return sorted(
        p.stem[len("train_"):]
        for p in (TRAIN_DIR / "scripts").glob("train_*.py")
    )


def _get_cfg_cls(method: str):
    """從 base_config 動態取得對應設定類（依 METHOD 屬性匹配）。"""
    import base_config as _bc
    for attr in vars(_bc).values():
        if (isinstance(attr, type)
                and issubclass(attr, _bc.BaseConfig)
                and attr is not _bc.BaseConfig
                and getattr(attr, "METHOD", "") == method):
            return attr
    return None


def _launch(cmd: list, inline: bool, run_id: str):
    """在新視窗（預設）或當前終端（--inline）啟動訓練。"""
    if inline:
        subprocess.run(cmd)
        print(f"\n訓練完成。run_id：{run_id}")
    else:
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
        else:
            cmd = ["xterm", "-e"] + cmd
        subprocess.Popen(cmd, **kwargs)
        print(f"訓練已在新視窗啟動。run_id：{run_id}")


def _rel(path: Path) -> str:
    """相對於 ROOT（MLOps/）的路徑字串，用於 run_summary.json 中儲存。"""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _abs(rel_or_abs: str) -> Path:
    """將 run_summary.json 中的路徑還原為絕對 Path（兼容舊版絕對路徑記錄）。"""
    p = Path(rel_or_abs)
    return p if p.is_absolute() else ROOT / p


def _snapshot_config(method: str, config_name: str, date_str: str, config_path: Path):
    """在 configs/{method}/{config_name}/{YYYYMMDD}.txt 保存一份日期快照。"""
    import shutil
    snap_dir = TRAIN_DIR / "configs" / method / config_name
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap = snap_dir / f"{date_str}.txt"
    if not snap.exists():
        shutil.copy(config_path, snap)


def _find_best_checkpoint(exp_dir: Path) -> Path | None:
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    final = ckpt_dir / "final"
    if final.exists():
        return final
    numbered = sorted(
        (p for p in ckpt_dir.iterdir() if p.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if numbered:
        return numbered[-1]
    emergency = ckpt_dir / "emergency"
    if emergency.exists():
        return emergency
    return None


# ── data subcommands ───────────────────────────────────────────────────────────

_DATA_DEFAULT_FILE = DATA_DIR / ".defaults"

def _get_data_default(method: str) -> list[str] | None:
    if not _DATA_DEFAULT_FILE.exists():
        return None
    import json as _json
    d = _json.loads(_DATA_DEFAULT_FILE.read_text(encoding="utf-8"))
    return d.get(method) or None

def _set_data_default(method: str, versions: list[str]):
    import json as _json
    d = _json.loads(_DATA_DEFAULT_FILE.read_text(encoding="utf-8")) if _DATA_DEFAULT_FILE.exists() else {}
    d[method] = versions
    _DATA_DEFAULT_FILE.write_text(_json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def cmd_data_default(args):
    method   = getattr(args, "method",  None)
    dataset  = getattr(args, "dataset", None)
    versions = getattr(args, "versions", [])
    available = [v.name for v in _all_versions("unsafe")]

    if method is None:
        for m in _available_methods():
            current = _get_data_default(m)
            print(f"  {m} dataset: {' '.join(current) if current else '（未設定）'}  可選：{', '.join(available)}")
        return

    if dataset != "dataset":
        print("用法：data default <dpo|sft> dataset <vN ...>")
        return

    if not versions:
        current = _get_data_default(method)
        print(f"  現在預設：{' '.join(current) if current else '（未設定）'}")
        print(f"  可選版本：{', '.join(available)}")
        return

    invalid = [v for v in versions if v not in available]
    if invalid:
        print(f"找不到版本：{', '.join(invalid)}  可選：{', '.join(available)}")
        return

    _set_data_default(method, versions)
    print(f"{method} dataset 預設已設為：{' '.join(versions)}")


def cmd_data_list(args):
    types = [p.name for p in DATA_DIR.iterdir() if p.is_dir()] if DATA_DIR.exists() else []
    if not types:
        print("尚無資料。")
        return

    for dtype in sorted(types):
        print(f"\n[{dtype}]")
        for ver in _all_versions(dtype):
            print(f"  {ver.name}/")
            for batch in sorted(p for p in ver.iterdir() if p.is_dir()):
                meta_path  = batch / "meta.jsonl"
                train_path = batch / "train.jsonl"
                val_path   = batch / "val.jsonl"
                train_count = len(train_path.read_text(encoding="utf-8").splitlines()) if train_path.exists() else "?"
                val_count   = len(val_path.read_text(encoding="utf-8").splitlines())   if val_path.exists()   else "?"
                note = ""
                if meta_path.exists():
                    first = _load_jsonl(meta_path)[0]
                    note = first.get("note", "")
                print(f"    {batch.name}  train={train_count}  val={val_count}  {note}")


def cmd_data_log(args):
    dtype   = args.type
    version = getattr(args, "version", None)
    date    = getattr(args, "date", None)

    batches = _all_batches(dtype, version)
    if not batches:
        print(f"找不到資料：{dtype}" + (f"/{version}" if version else ""))
        return

    if date:
        batches = [(v, b) for v, b in batches if b.name == date]
        if not batches:
            print(f"找不到批次：{dtype}/{version}/{date}")
            return

    for ver_name, batch in batches:
        meta_path = batch / "meta.jsonl"
        print(f"\n── {dtype}/{ver_name}/{batch.name} ──")
        if not meta_path.exists():
            print("  （無記事本）")
            continue
        for entry in _load_jsonl(meta_path):
            print(f"  {entry}")


# ── train helpers（cmd_train / cmd_continuetrain / _retrain_one 共用）──────────

def _resolve_train_data(method: str, versions, dates, needs_data: bool):
    """依資料版本/日期收集 unsafe+normal 路徑；NEEDS_DATA=False 的方法回傳空資料。
    回傳 (all_trains, all_vals, train_count, val_count, data_files)。"""
    if not needs_data:
        print(f"{method} 不需要外部資料（NEEDS_DATA=False），略過資料版本解析")
        return [], [], 0, 0, {}

    unsafe_trains, unsafe_vals = _collect_data_paths("unsafe", versions, dates)
    normal_trains, normal_vals = _collect_data_paths("normal", versions, dates)
    all_trains  = unsafe_trains + normal_trains
    all_vals    = unsafe_vals   + normal_vals
    if not all_trains:
        print(f"找不到符合的訓練資料（versions={versions} dates={dates}），{method} 需要資料才能訓練")
        sys.exit(1)
    train_count = _count_jsonl(all_trains)
    val_count   = _count_jsonl(all_vals)
    print(f"資料來源（unsafe + normal）：train={train_count}  val={val_count}")
    data_files = {
        "unsafe": [_rel(p) for p in unsafe_trains],
        "normal": [_rel(p) for p in normal_trains],
    }
    return all_trains, all_vals, train_count, val_count, data_files


def _write_new_summary(exp_dir: Path, *, run_id, method, config_name, config_date, hyperparams,
                       data_versions, data_dates, data_files, train_count, val_count,
                       provenance: dict | None = None) -> Path:
    """寫入新 run 的 run_summary.json。provenance（retrain_of / base_run_id /
    continued_from / base_checkpoint 等來源欄位）緊接在 config_date 之後。"""
    summary = {
        "run_id":          run_id,
        "method":          method,
        "config":          config_name,
        "config_date":     config_date,
        **(provenance or {}),
        "started_at":      datetime.now().isoformat(timespec="seconds"),
        "finished_at":     None,
        "status":          "running",
        "hyperparams":     hyperparams,
        "data_versions":   data_versions,
        "data_dates":      data_dates,
        "data_files":      data_files,
        "train_count":     train_count,
        "val_count":       val_count,
        "checkpoints":     [],
    }
    summary_path = exp_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def _train_cmd(script_path: Path, config_name: str, exp_dir: Path, summary_path: Path,
               all_trains: list, all_vals: list) -> list:
    """組出啟動 train_<method>.py 子行程的共用參數；各指令自行追加 --resume/--base_model 等。"""
    return [
        sys.executable,    str(script_path),
        "--config",        config_name,
        "--output_dir",    str(exp_dir / "checkpoints"),
        "--train_paths",   *[str(p) for p in all_trains],
        "--val_paths",     *[str(p) for p in all_vals],
        "--metrics_path",  str(exp_dir / "metrics.jsonl"),
        "--gen_test_path", str(exp_dir / "generation_test.txt"),
        "--summary_path",  str(summary_path),
    ]


# ── train subcommands ──────────────────────────────────────────────────────────

def _resume_prune_run(args):
    """接續 km_search：沿用同一個 run 目錄。--resume 會開新的 run_id，接不回原本的格子。"""
    run_id  = args.resume_run
    exp_dir = EXPERIMENTS_DIR / run_id
    summary_path = exp_dir / "run_summary.json"
    if not summary_path.exists():
        print(f"找不到 run：{run_id}")
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if summary.get("method") != "prune" or summary.get("hyperparams", {}).get("mode") != "km_search":
        print(f"{run_id} 不是 prune 的 K、M 掃描")
        return
    if summary.get("status") == "completed":
        print(f"{run_id} 已經完成")
        return
    summary.update(status="running", finished_at=None)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    script_path = TRAIN_DIR / "scripts" / "train_prune.py"
    cmd = _train_cmd(script_path, summary["config"], exp_dir, summary_path, [], []) + ["--resume"]
    # 用這個 run 當初的超參數，中途改動或刪除設定檔都不影響接續路徑
    cmd += ["--overrides_json", json.dumps(summary["hyperparams"])]
    print(f"[{run_id}] 接續...")
    _launch(cmd, getattr(args, "inline", False), run_id)


def cmd_train(args):
    import shutil

    if getattr(args, "resume_run", None):
        return _resume_prune_run(args)

    method      = args.method
    config_name = getattr(args, "config", None) or _get_default(method)
    versions = getattr(args, "versions", None) or _get_data_default(method)
    dates    = getattr(args, "dates", None)
    resume   = getattr(args, "resume", False)

    _cfg_cls   = _get_cfg_cls(method)
    needs_data = getattr(_cfg_cls, "NEEDS_DATA", True)

    if needs_data and not versions:
        print(f"未指定資料版本，且 {method} 尚未設定資料預設。\n請用 --versions 指定，或先執行 data default {method} dataset <版本>")
        return

    if not config_name:
        print(f"未指定設定檔，且 {method} 尚未設定預設。\n請用 --config 指定，或先執行 config edit {method} <名稱>")
        return

    config_path = TRAIN_DIR / "configs" / method / f"{config_name}.txt"
    if not config_path.exists():
        print(f"找不到設定檔：{config_path}")
        return

    script_path = TRAIN_DIR / "scripts" / f"train_{method}.py"
    if not script_path.exists():
        print(f"訓練腳本尚未建立：{script_path}")
        sys.exit(1)

    date_str = datetime.now().strftime("%Y%m%d")
    hyperparams = _cfg_cls(config_name).as_dict() if _cfg_cls else {}
    run_id  = _next_run_id(str(hyperparams.get("model_name", "") or ""))
    exp_dir = EXPERIMENTS_DIR / run_id

    shutil.copy(config_path, exp_dir / "config.txt")
    _snapshot_config(method, config_name, date_str, config_path)

    all_trains, all_vals, train_count, val_count, data_files = _resolve_train_data(
        method, versions, dates, needs_data)
    summary_path = _write_new_summary(
        exp_dir,
        run_id=run_id, method=method, config_name=config_name, config_date=date_str,
        hyperparams=hyperparams,
        data_versions=versions if needs_data else None,
        data_dates=dates if needs_data else None,
        data_files=data_files, train_count=train_count, val_count=val_count,
    )

    print(f"\n[{run_id}] 開始訓練...")
    cmd = _train_cmd(script_path, config_name, exp_dir, summary_path, all_trains, all_vals)
    if resume:
        cmd.append("--resume")

    _launch(cmd, getattr(args, "inline", False), run_id)


# ── retrain / continuetrain subcommands ───────────────────────────────────────

def _retrain_one(source_run_id: str, inline: bool) -> str:
    """
    遞迴重現單一 run。若該 run 有 base_run_id（來自 continuetrain），
    先強制 inline 重現祖先，再用祖先的 final checkpoint 當底座。
    回傳新產生的 run_id。
    """
    import shutil

    source_dir = EXPERIMENTS_DIR / source_run_id
    if not source_dir.exists():
        print(f"找不到實驗：{source_run_id}")
        sys.exit(1)

    summary_path = source_dir / "run_summary.json"
    if not summary_path.exists():
        print(f"找不到 run_summary.json：{source_run_id}")
        sys.exit(1)

    summary     = read_summary(summary_path)
    method      = summary["method"]
    config_name = summary["config"]
    config_date = summary.get("config_date")
    orig_params = summary.get("hyperparams", {})

    script_path = TRAIN_DIR / "scripts" / f"train_{method}.py"
    if not script_path.exists():
        print(f"訓練腳本尚未建立：{script_path}")
        sys.exit(1)

    # ── 遞迴處理祖先 ─────────────────────────────────────────────────────────
    base_model_path = None
    ancestor_run_id = summary.get("base_run_id")
    if ancestor_run_id:
        print(f"[retrain] 偵測到前置訓練 {ancestor_run_id}，先遞迴重現（inline）...")
        new_ancestor_id = _retrain_one(ancestor_run_id, inline=True)
        ancestor_final  = EXPERIMENTS_DIR / new_ancestor_id / "checkpoints" / "final"
        if not ancestor_final.exists():
            print(f"前置訓練未產出 final checkpoint：{new_ancestor_id}")
            sys.exit(1)
        base_model_path = str(ancestor_final)

    run_id  = _next_run_id(source_run_id)   # 鏈式：從來源 run 衍生
    exp_dir = EXPERIMENTS_DIR / run_id

    # ── 復原 config 快照（優先用日期快照，fallback 用 exp_dir/config.txt）──
    if config_date:
        snap = TRAIN_DIR / "configs" / method / config_name / f"{config_date}.txt"
        src  = snap if snap.exists() else source_dir / "config.txt"
    else:
        src  = source_dir / "config.txt"
    if src.exists():
        shutil.copy(src, exp_dir / "config.txt")

    # ── 資料 ─────────────────────────────────────────────────────────────────
    unsafe_trains = [_abs(p) for p in summary["data_files"].get("unsafe", []) if _abs(p).exists()]
    normal_trains = [_abs(p) for p in summary["data_files"].get("normal", []) if _abs(p).exists()]
    unsafe_vals   = [p.parent / "val.jsonl" for p in unsafe_trains if (p.parent / "val.jsonl").exists()]
    normal_vals   = [p.parent / "val.jsonl" for p in normal_trains if (p.parent / "val.jsonl").exists()]

    all_trains  = unsafe_trains + normal_trains
    all_vals    = unsafe_vals   + normal_vals
    if not all_trains and any(summary.get("data_files", {}).values()):
        print(f"原始 run 記錄的訓練資料檔已不存在，無法重現：{source_run_id}")
        sys.exit(1)
    train_count = _count_jsonl(all_trains)
    val_count   = _count_jsonl(all_vals)
    print(f"重現資料來源（源：{source_run_id}）：train={train_count}  val={val_count}")

    provenance = {"retrain_of": source_run_id}
    if base_model_path:
        provenance["base_run_id"] = ancestor_run_id

    new_summary_path = _write_new_summary(
        exp_dir,
        run_id=run_id, method=method, config_name=config_name, config_date=config_date,
        hyperparams=orig_params,
        data_versions=summary.get("data_versions"),
        data_dates=summary.get("data_dates"),
        data_files={
            "unsafe": [_rel(p) for p in unsafe_trains],
            "normal": [_rel(p) for p in normal_trains],
        },
        train_count=train_count, val_count=val_count,
        provenance=provenance,
    )

    print(f"\n[{run_id}] 重現訓練（源：{source_run_id}）...")
    cmd = _train_cmd(script_path, config_name, exp_dir, new_summary_path, all_trains, all_vals)
    cmd += ["--overrides_json", json.dumps(orig_params)]
    if base_model_path:
        cmd += ["--base_model", base_model_path]

    _launch(cmd, inline, run_id)
    return run_id


def cmd_retrain(args):
    for source_run_id in args.run_ids:
        _retrain_one(source_run_id, getattr(args, "inline", False))


def cmd_continuetrain(args):
    import shutil

    source_run_id = args.run_id
    source_dir    = EXPERIMENTS_DIR / source_run_id
    if not source_dir.exists():
        print(f"找不到實驗：{source_run_id}")
        return

    summary_path = source_dir / "run_summary.json"
    if not summary_path.exists():
        print(f"找不到 run_summary.json：{source_run_id}")
        return

    source_summary = read_summary(summary_path)
    method         = source_summary["method"]

    base_ckpt = _find_best_checkpoint(source_dir)
    if not base_ckpt:
        print(f"找不到可用的 checkpoint：{source_run_id}")
        return

    config_name = getattr(args, "config",   None) or _get_default(method)
    versions    = getattr(args, "versions", None) or _get_data_default(method)
    dates       = getattr(args, "dates",    None)
    resume      = getattr(args, "resume",   False)

    _cfg_cls    = _get_cfg_cls(method)
    needs_data  = getattr(_cfg_cls, "NEEDS_DATA", True)

    if not config_name:
        print(f"未指定設定檔，且 {method} 尚未設定預設。\n請用 --config 指定，或先執行 config default {method} <名稱>")
        return
    if needs_data and not versions:
        print(f"未指定資料版本，且 {method} 尚未設定資料預設。\n請用 --versions 指定，或先執行 data default {method} dataset <版本>")
        return

    config_path = TRAIN_DIR / "configs" / method / f"{config_name}.txt"
    if not config_path.exists():
        print(f"找不到設定檔：{config_path}")
        return

    script_path = TRAIN_DIR / "scripts" / f"train_{method}.py"
    if not script_path.exists():
        print(f"訓練腳本尚未建立：{script_path}")
        sys.exit(1)

    date_str = datetime.now().strftime("%Y%m%d")
    run_id  = _next_run_id(source_run_id)   # 鏈式：從來源 run 衍生
    exp_dir = EXPERIMENTS_DIR / run_id
    shutil.copy(config_path, exp_dir / "config.txt")
    _snapshot_config(method, config_name, date_str, config_path)

    all_trains, all_vals, train_count, val_count, data_files = _resolve_train_data(
        method, versions, dates, needs_data)

    hyperparams = _cfg_cls(config_name).as_dict() if _cfg_cls else {}
    new_summary_path = _write_new_summary(
        exp_dir,
        run_id=run_id, method=method, config_name=config_name, config_date=date_str,
        hyperparams=hyperparams,
        data_versions=versions if needs_data else None,
        data_dates=dates if needs_data else None,
        data_files=data_files, train_count=train_count, val_count=val_count,
        provenance={
            "base_run_id":     source_run_id,
            "continued_from":  source_run_id,
            "base_checkpoint": _rel(base_ckpt),
        },
    )

    print(f"\n[{run_id}] 接續訓練（源：{source_run_id} → {base_ckpt.name}）...")
    cmd = _train_cmd(script_path, config_name, exp_dir, new_summary_path, all_trains, all_vals)
    cmd += ["--base_model", str(base_ckpt)]
    if resume:
        cmd.append("--resume")

    _launch(cmd, getattr(args, "inline", False), run_id)


# ── result subcommands ─────────────────────────────────────────────────────────

def cmd_result_list(args):
    if not EXPERIMENTS_DIR.exists() or not any(EXPERIMENTS_DIR.iterdir()):
        print("尚無實驗紀錄。")
        return

    print(f"{'Run ID':<40} {'train':>7} {'val':>7} {'checkpoints':>12}")
    print("─" * 70)
    for exp in sorted(EXPERIMENTS_DIR.iterdir()):
        if not exp.is_dir():
            continue
        ckpt_dir = exp / "checkpoints"
        ckpts    = len(list(ckpt_dir.iterdir())) if ckpt_dir.exists() else 0
        # 優先從 run_summary.json 讀筆數；fallback 舊版合併檔案
        summary_path = exp / "run_summary.json"
        if summary_path.exists():
            s = read_summary(summary_path)
            train_n = s.get("train_count", "?")
            val_n   = s.get("val_count",   "?")
        else:
            train_path = exp / "data" / "train.jsonl"
            val_path   = exp / "data" / "val.jsonl"
            train_n = len(train_path.read_text(encoding="utf-8").splitlines()) if train_path.exists() else "?"
            val_n   = len(val_path.read_text(encoding="utf-8").splitlines())   if val_path.exists()   else "?"
        print(f"{exp.name:<40} {str(train_n):>7} {str(val_n):>7} {ckpts:>12}")


def cmd_result_show(args):
    exp_dir = EXPERIMENTS_DIR / args.run_id
    if not exp_dir.exists():
        print(f"找不到實驗：{args.run_id}")
        sys.exit(1)

    gen_test = exp_dir / "generation_test.txt"
    if gen_test.exists():
        print(gen_test.read_text(encoding="utf-8"))
    else:
        print("（尚無生成測試結果）")

    metrics = exp_dir / "metrics.jsonl"
    if metrics.exists():
        lines = _load_jsonl(metrics)
        if lines:
            print("\n── 最後 5 筆 metrics ──")
            for entry in lines[-5:]:
                print(f"  {entry}")


def cmd_result_inspect(args):
    exp_dir = EXPERIMENTS_DIR / args.run_id
    if not exp_dir.exists():
        print(f"找不到實驗：{args.run_id}")
        sys.exit(1)

    # ── run summary ──────────────────────────────────────────────────────────
    summary_path = exp_dir / "run_summary.json"
    if summary_path.exists():
        s = read_summary(summary_path)
        print(f"{'='*60}")
        print(f"  run_id   : {s.get('run_id', args.run_id)}")
        print(f"  方法     : {s.get('method', '?')}    設定 : {s.get('config', '?')}")
        print(f"  狀態     : {s.get('status', '?')}")
        print(f"  開始     : {s.get('started_at', '?')}")
        print(f"  結束     : {s.get('finished_at') or '（進行中）'}")
        print(f"  訓練筆數 : {s.get('train_count', '?')}    驗證 : {s.get('val_count', '?')}")
        if s.get("retrain_of"):
            print(f"  重現自   : {s['retrain_of']}")
        if s.get("base_run_id") and not s.get("retrain_of"):
            print(f"  前置訓練 : {s['base_run_id']}")
        if s.get("continued_from"):
            print(f"  接續自   : {s['continued_from']}  checkpoint: {s.get('base_checkpoint', '?')}")
        print(f"{'─'*60}")
        print("  超參數：")
        for k, v in s.get("hyperparams", {}).items():
            print(f"    {k:<25} = {v}")
        print(f"{'─'*60}")
        data_files = s.get("data_files", {})
        if data_files:
            print("  資料來源：")
            for dtype, paths in data_files.items():
                for p in paths:
                    print(f"    [{dtype}] {p}")
        ckpts = s.get("checkpoints", [])
        print(f"  Checkpoints ({len(ckpts)}): {', '.join(ckpts) if ckpts else '（無）'}")
    else:
        print(f"找不到 run_summary.json：{args.run_id}")

    # ── metrics ──────────────────────────────────────────────────────────────
    metrics_path = exp_dir / "metrics.jsonl"
    if metrics_path.exists():
        lines = _load_jsonl(metrics_path)
        if lines:
            losses = [e["loss"] for e in lines if "loss" in e]
            best   = min(losses) if losses else None
            print(f"\n── Metrics（共 {len(lines)} steps）──")
            print(f"  首步  : {lines[0]}")
            if len(lines) > 1:
                print(f"  末步  : {lines[-1]}")
            if best is not None:
                best_step = next(e for e in lines if e.get("loss") == best)
                print(f"  最低 loss = {best:.4f}  @ step {best_step.get('step', '?')}")
        else:
            print("\n（metrics 檔案為空）")
    else:
        print("\n（尚無 metrics）")

    # ── generation test ───────────────────────────────────────────────────────
    gen_test = exp_dir / "generation_test.txt"
    if gen_test.exists():
        content = gen_test.read_text(encoding="utf-8").strip()
        if content:
            # 只顯示最後一個 generation block
            blocks = content.split("=" * 55)
            last_block = ("=" * 55).join(blocks[-2:]).strip() if len(blocks) >= 2 else content
            print(f"\n── 最新生成測試 ──")
            print(last_block)
    else:
        print("\n（尚無生成測試結果）")


def cmd_result_delete(args):
    import shutil

    to_delete = []
    for run_id in args.run_ids:
        exp_dir = EXPERIMENTS_DIR / run_id
        if not exp_dir.exists():
            print(f"找不到實驗：{run_id}")
        else:
            to_delete.append(exp_dir)

    if not to_delete:
        return

    print("即將刪除以下實驗目錄：")
    for d in to_delete:
        ckpt_dir = d / "checkpoints"
        ckpts = len(list(ckpt_dir.iterdir())) if ckpt_dir.exists() else 0
        print(f"  {d.name}  （{ckpts} checkpoints）")

    if not args.force:
        try:
            ans = input("\n確定要刪除？(y/N) ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n取消。")
            return
        if ans != "y":
            print("取消。")
            return

    for d in to_delete:
        shutil.rmtree(d)
        print(f"已刪除：{d.name}")


# ── config subcommands ────────────────────────────────────────────────────────

def _default_pointer(method: str) -> Path:
    return TRAIN_DIR / "configs" / method / ".default"

def _get_default(method: str) -> str | None:
    p = _default_pointer(method)
    return p.read_text(encoding="utf-8").strip() if p.exists() else None

def _set_default(method: str, name: str):
    _default_pointer(method).write_text(name, encoding="utf-8")


def _list_configs(method: str) -> list[str]:
    """列出某方法下所有可用的具名設定檔（排除 default.txt）。"""
    d = TRAIN_DIR / "configs" / method
    return sorted(p.stem for p in d.glob("*.txt") if p.stem != "default")


def cmd_config_default(args):
    method = getattr(args, "method", None)
    name   = getattr(args, "name",   None)

    if method is None:
        # 顯示所有方法的現況
        print("\n【各方法的預設設定檔】")
        for m in _available_methods():
            current = _get_default(m)
            options = _list_configs(m)
            print(f"\n  {m}")
            print(f"    預設：{current if current else '（未設定）'}")
            if options:
                print(f"    可選：{'、'.join(options)}")
        print()
        return

    if name is None:
        # 顯示單一方法現況與可選項
        current = _get_default(method)
        options = _list_configs(method)
        print(f"  現在預設：{current if current else '（未設定）'}")
        if options:
            print(f"  可選設定：{', '.join(options)}")
        return

    # 設定預設
    config_path = TRAIN_DIR / "configs" / method / f"{name}.txt"
    if not config_path.exists():
        options = _list_configs(method)
        print(f"找不到設定檔：{name}")
        if options:
            print(f"可選：{', '.join(options)}")
        return

    _set_default(method, name)
    print(f"{method} 預設設定已設為：{name}")


def cmd_config_edit(args):
    method = getattr(args, "method", None)
    name   = getattr(args, "name",   None)

    if method is None:
        path = TRAIN_DIR / "configs" / "default.txt"
    elif name is None:
        path = TRAIN_DIR / "configs" / method / "default.txt"
    else:
        path = TRAIN_DIR / "configs" / method / f"{name}.txt"

    if not path.exists():
        print(f"找不到設定檔：{path}")
        return

    print(f"開啟：{path}")
    if sys.platform == "win32":
        os.startfile(str(path))
    else:
        editor = os.environ.get("EDITOR", "nano")
        subprocess.Popen([editor, str(path)])


# ── showmode subcommand ────────────────────────────────────────────────────────

def cmd_showmode(args):
    display_dir = TRAIN_DIR / "display"
    mode_file   = display_dir / ".mode"
    current = mode_file.read_text(encoding="utf-8").strip() if mode_file.exists() else "default"

    name = getattr(args, "name", None)

    available = sorted(
        p.stem for p in display_dir.glob("*.py")
        if not p.stem.startswith("base") and not p.stem.startswith("_")
    ) if display_dir.exists() else []

    if name is None:
        print(f"  目前顯示模式：{current}")
        print(f"  可選模式：{', '.join(available + ['none'])}")
        return

    if name != "none" and name not in available:
        print(f"找不到顯示模式：{name}  可選：{', '.join(available + ['none'])}")
        return

    display_dir.mkdir(parents=True, exist_ok=True)
    mode_file.write_text(name, encoding="utf-8")
    print(f"顯示模式已切換至：{name}")


# ── chat subcommand ───────────────────────────────────────────────────────────

def cmd_chat(args):
    from model_loader import load_model
    from chat import chat
    from chat_config import DEFAULT_CHECKPOINT

    model_id = args.run_id
    checkpoint = getattr(args, "checkpoint", None) or DEFAULT_CHECKPOINT

    print(f"載入模型：{model_id}（checkpoint: {checkpoint}）...")
    model, tokenizer = load_model(model_id, checkpoint)

    if model is None or tokenizer is None:
        print(f"錯誤：無法載入模型 {model_id}")
        return

    print(f"模型載入成功")

    chat(model, tokenizer)


def cmd_compare_chat(args):
    import subprocess
    run_ids = args.run_ids
    checkpoint = getattr(args, "checkpoint", None) or "final"
    cmd = [sys.executable, str(ROOT / "compare_chat.py")] + run_ids
    if checkpoint != "final":
        cmd += ["--checkpoint", checkpoint]
    subprocess.run(cmd)


def cmd_data_edit_prompts(args):
    """啟動違禁題資料集編輯器（給 km_search 的 seeds 用）。"""
    import subprocess
    editor = ROOT.parent / "pipelines" / "refusal_data_editor.py"
    cmd = [sys.executable, str(editor)]
    if getattr(args, "path", None):
        cmd.append(args.path)
    subprocess.run(cmd)


def cmd_list_models(args):
    from list_models import print_model_list
    print_model_list()


# ── test subcommands ───────────────────────────────────────────────────────────

def cmd_test(args):
    scripts = {
        "diagnose": TESTS_DIR / "diagnose_dpo.py",
        "load":     TESTS_DIR / "test_load.py",
        "nan":      TESTS_DIR / "test_nan_combinations.py",
    }
    target = args.target
    script = scripts.get(target)
    if not script or not script.exists():
        print(f"找不到測試腳本：{target}")
        sys.exit(1)
    subprocess.run([sys.executable, str(script)], check=True)


# ── parser ─────────────────────────────────────────────────────────────────────

def _build_parser():
    parser = argparse.ArgumentParser(prog="", add_help=False)
    sub = parser.add_subparsers(dest="command")

    # data
    dp = sub.add_parser("data", add_help=False)
    dsub = dp.add_subparsers(dest="data_cmd")
    dsub.add_parser("list")
    dlog = dsub.add_parser("log")
    dlog.add_argument("type", choices=["unsafe", "normal"])
    dlog.add_argument("version", nargs="?")
    dlog.add_argument("date",    nargs="?")
    ddefault = dsub.add_parser("default")
    ddefault.add_argument("method",   nargs="?", choices=_available_methods())
    ddefault.add_argument("dataset",  nargs="?")
    ddefault.add_argument("versions", nargs="*", metavar="vN")
    dedit = dsub.add_parser("edit-prompts")
    dedit.add_argument("path", nargs="?", help="資料集路徑；省略用編輯器預設")

    # train
    tp = sub.add_parser("train", add_help=False)
    tsub = tp.add_subparsers(dest="method")
    for m in _available_methods():
        mp = tsub.add_parser(m)
        mp.add_argument("--config",   default=None)
        mp.add_argument("--versions", nargs="+", metavar="vN")
        mp.add_argument("--dates",    nargs="+", metavar="YYYYMMDD")
        mp.add_argument("--resume",   action="store_true")
        mp.add_argument("--inline",   action="store_true")
        if m == "prune":
            mp.add_argument("--resume-run", dest="resume_run", default=None, metavar="run_id",
                            help="接續中斷的 K、M 掃描，沿用同一個 run 目錄")
    retp = tsub.add_parser("retrain")
    retp.add_argument("run_ids", nargs="+", metavar="run_id")
    retp.add_argument("--inline", action="store_true")
    ctp = tsub.add_parser("continuetrain")
    ctp.add_argument("run_id")
    ctp.add_argument("--config",   default=None)
    ctp.add_argument("--versions", nargs="+", metavar="vN")
    ctp.add_argument("--dates",    nargs="+", metavar="YYYYMMDD")
    ctp.add_argument("--resume",   action="store_true")
    ctp.add_argument("--inline",   action="store_true")

    # result
    rp = sub.add_parser("result", add_help=False)
    rsub = rp.add_subparsers(dest="result_cmd")
    rsub.add_parser("list")
    rshow = rsub.add_parser("show")
    rshow.add_argument("run_id")
    rinspect = rsub.add_parser("inspect")
    rinspect.add_argument("run_id")
    rdelete = rsub.add_parser("delete")
    rdelete.add_argument("run_ids", nargs="+", metavar="run_id")
    rdelete.add_argument("--force", action="store_true")

    # chat
    chp = sub.add_parser("chat", add_help=False)
    chp.add_argument("run_id")
    chp.add_argument("checkpoint", nargs="?", default="final")

    # compare-chat
    cchp = sub.add_parser("compare-chat", add_help=False)
    cchp.add_argument("run_ids", nargs="+", metavar="run_id")
    cchp.add_argument("--checkpoint", default="final")

    # list-models
    sub.add_parser("list-models", add_help=False)

    # showmode
    smp = sub.add_parser("showmode", add_help=False)
    smp.add_argument("name", nargs="?")

    # test
    xp = sub.add_parser("test", add_help=False)
    xp.add_argument("target", choices=["diagnose", "load", "nan"])

    # config
    cp = sub.add_parser("config", add_help=False)
    csub = cp.add_subparsers(dest="config_cmd")

    cdefault = csub.add_parser("default")
    cdefault.add_argument("method", nargs="?", choices=_available_methods())
    cdefault.add_argument("name",   nargs="?")

    cedit = csub.add_parser("edit")
    cedit.add_argument("method", nargs="?", choices=_available_methods())
    cedit.add_argument("name",   nargs="?")

    return parser, dp, tp, rp, cp


def _dispatch(args, dp, tp, rp, cp):
    if args.command == "data":
        if args.data_cmd == "list":
            cmd_data_list(args)
        elif args.data_cmd == "log":
            cmd_data_log(args)
        elif args.data_cmd == "default":
            cmd_data_default(args)
        elif args.data_cmd == "edit-prompts":
            cmd_data_edit_prompts(args)
        else:
            dp.print_help()

    elif args.command == "train":
        if args.method == "retrain":
            cmd_retrain(args)
        elif args.method == "continuetrain":
            cmd_continuetrain(args)
        elif args.method in _available_methods():
            cmd_train(args)
        else:
            tp.print_help()

    elif args.command == "result":
        if args.result_cmd == "list":
            cmd_result_list(args)
        elif args.result_cmd == "show":
            cmd_result_show(args)
        elif args.result_cmd == "inspect":
            cmd_result_inspect(args)
        elif args.result_cmd == "delete":
            cmd_result_delete(args)
        else:
            rp.print_help()

    elif args.command == "chat":
        cmd_chat(args)

    elif args.command == "compare-chat":
        cmd_compare_chat(args)

    elif args.command == "list-models":
        cmd_list_models(args)

    elif args.command == "showmode":
        cmd_showmode(args)

    elif args.command == "test":
        cmd_test(args)

    elif args.command == "config":
        if args.config_cmd == "default":
            cmd_config_default(args)
        elif args.config_cmd == "edit":
            cmd_config_edit(args)
        else:
            cp.print_help()

    else:
        print("指令：data / train / result / chat / list-models / showmode / test / config / exit")


# ── entry point ────────────────────────────────────────────────────────────────

HELP_TEXT = """\
指令列表：
  data list
  data log <unsafe|normal> [version] [date]
  data default                              ← 查看所有方法的資料預設
  data default <dpo|sft> dataset            ← 查看可選版本與目前預設
  data default <dpo|sft> dataset <vN ...>   ← 設定訓練時預設使用的版本
  data edit-prompts [path]                  ← 管理 km_search 違禁題資料集（互動式）
  train dpo --config <name> [--versions vN ...] [--dates YYYYMMDD ...] [--resume]
  train sft --config <name> [--resume]
  train prune --config <name> [--inline]     ← 裁減；mode=km_search 時自動掃描 K、M
  train prune --resume-run <run_id>          ← 接續中斷的 K、M 掃描
  train retrain <run_id> [run_id ...]
  train continuetrain <run_id> [--config <name>] [--versions vN ...] [--dates YYYYMMDD ...] [--resume]
  result list
  result show <run_id>
  result inspect <run_id>
  result delete <run_id> [run_id ...]   ← 互動確認
  result delete <run_id> --force        ← 直接刪除
  chat <run_id> [checkpoint]            ← 與訓練模型互動對話
  compare-chat <run_id> [<run_id> ...]  ← 多模型對話對比（單一面板）
  list-models                            ← 列出所有可用訓練模型
  showmode                       ← 查看可用顯示模式與目前設定
  showmode <name|none>           ← 切換顯示模式（none 停用）
  test <diagnose|load|nan>
  config default                 ← 查看所有方法的預設與可選設定
  config default <dpo|sft>       ← 查看單一方法的預設與可選設定
  config default <dpo|sft> <name> ← 將 <name> 設為該方法的預設
  config edit                    ← 開啟全局 default.txt
  config edit <dpo|sft>          ← 開啟方法 default.txt
  config edit <dpo|sft> <name>   ← 開啟指定設定檔
  exit\
"""

USAGE_TEXT = """\
用法：
  python run.py            不加參數 → 開啟 km_cos 行為編輯圖形介面
  python run.py repl       進入互動模式（REPL）
  python run.py <指令> …    直接執行單一指令（指令清單如下）

""" + HELP_TEXT


def _launch_km_cos_gui():
    """無任何參數時的預設介面：km_cos（cosplay 行為編輯）圖形介面。"""
    import subprocess
    gui = ROOT.parent / "pipelines" / "km_cos_gui.py"
    subprocess.run([sys.executable, str(gui)])


def _repl(parser, dp, tp, rp, cp):
    """舊的互動 CLI（現在只在 `python run.py repl` 時進入）。"""
    import shlex
    print("MLOps CLI  （輸入 help 顯示指令，exit 離開）")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n離開。")
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit", "q"):
            print("離開。")
            break
        if line.lower() in ("help", "h", "?"):
            print(HELP_TEXT)
            continue
        try:
            tokens = shlex.split(line)
            args = parser.parse_args(tokens)
            _dispatch(args, dp, tp, rp, cp)
        except SystemExit:
            pass  # argparse 解析失敗時不退出整個程序
        except Exception as e:
            print(f"錯誤：{e}")


def main():
    parser, dp, tp, rp, cp = _build_parser()
    argv = sys.argv[1:]

    # 無任何參數 → 預設開 km_cos 圖形介面（這個訓練方法的專屬介面）
    if not argv:
        _launch_km_cos_gui()
        return
    # 頂層說明（parser 以 add_help=False 建立，-h/--help 在這裡自行攔截）
    if argv[0] in ("-h", "--help", "help"):
        print(USAGE_TEXT)
        return
    # `run.py repl` → 舊的互動模式
    if argv == ["repl"]:
        _repl(parser, dp, tp, rp, cp)
        return
    # 其餘：一次性子命令
    try:
        args = parser.parse_args(argv)
        _dispatch(args, dp, tp, rp, cp)
    except SystemExit:
        pass
    except Exception as e:
        print(f"錯誤：{e}")


if __name__ == "__main__":
    main()

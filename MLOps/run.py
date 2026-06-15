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


def _next_run_id(method: str, config: str) -> str:
    date = datetime.now().strftime("%Y%m%d")
    prefix = f"{method}_{config}_{date}_"
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


# ── train subcommands ──────────────────────────────────────────────────────────

def cmd_train(args):
    import shutil

    method      = args.method
    config_name = getattr(args, "config", None) or _get_default(method)
    versions = getattr(args, "versions", None) or _get_data_default(method)
    dates    = getattr(args, "dates", None)
    resume   = getattr(args, "resume", False)

    if not versions:
        print(f"未指定資料版本，且 {method} 尚未設定資料預設。\n請用 --versions 指定，或先執行 data default {method} <版本>")
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

    run_id  = _next_run_id(method, config_name)
    exp_dir = EXPERIMENTS_DIR / run_id

    shutil.copy(config_path, exp_dir / "config.txt")

    unsafe_trains, unsafe_vals = _collect_data_paths("unsafe", versions, dates)
    normal_trains, normal_vals = _collect_data_paths("normal", versions, dates)

    all_trains = unsafe_trains + normal_trains
    all_vals   = unsafe_vals   + normal_vals
    train_count = _count_jsonl(all_trains)
    val_count   = _count_jsonl(all_vals)
    print(f"資料來源（unsafe + normal）：train={train_count}  val={val_count}")

    _cfg_cls = _get_cfg_cls(method)
    hyperparams = _cfg_cls(config_name).as_dict() if _cfg_cls else {}
    summary = {
        "run_id":          run_id,
        "method":          method,
        "config":          config_name,
        "started_at":      datetime.now().isoformat(timespec="seconds"),
        "finished_at":     None,
        "status":          "running",
        "hyperparams":     hyperparams,
        "data_versions":   versions,
        "data_dates":      dates,
        "data_files": {
            "unsafe": [_rel(p) for p in unsafe_trains],
            "normal": [_rel(p) for p in normal_trains],
        },
        "train_count":     train_count,
        "val_count":       val_count,
        "checkpoints":     [],
    }
    summary_path = exp_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[{run_id}] 開始訓練...")
    cmd = [
        sys.executable,    str(script_path),
        "--config",        config_name,
        "--output_dir",    str(exp_dir / "checkpoints"),
        "--train_paths",   *[str(p) for p in all_trains],
        "--val_paths",     *[str(p) for p in all_vals],
        "--metrics_path",  str(exp_dir / "metrics.jsonl"),
        "--gen_test_path", str(exp_dir / "generation_test.txt"),
        "--summary_path",  str(summary_path),
    ]
    if resume:
        cmd.append("--resume")

    _launch(cmd, getattr(args, "inline", False), run_id)


# ── retrain / continuetrain subcommands ───────────────────────────────────────

def cmd_retrain(args):
    import shutil

    for source_run_id in args.run_ids:
        source_dir = EXPERIMENTS_DIR / source_run_id
        if not source_dir.exists():
            print(f"找不到實驗：{source_run_id}")
            continue

        summary_path = source_dir / "run_summary.json"
        if not summary_path.exists():
            print(f"找不到 run_summary.json：{source_run_id}")
            continue

        summary     = json.loads(summary_path.read_text(encoding="utf-8"))
        method      = summary["method"]
        config_name = summary["config"]
        orig_params = summary.get("hyperparams", {})

        script_path = TRAIN_DIR / "scripts" / f"train_{method}.py"
        if not script_path.exists():
            print(f"訓練腳本尚未建立：{script_path}")
            continue

        run_id  = _next_run_id(f"retrain_{method}", config_name)
        exp_dir = EXPERIMENTS_DIR / run_id

        orig_config = source_dir / "config.txt"
        if orig_config.exists():
            shutil.copy(orig_config, exp_dir / "config.txt")

        unsafe_trains = [_abs(p) for p in summary["data_files"].get("unsafe", []) if _abs(p).exists()]
        normal_trains = [_abs(p) for p in summary["data_files"].get("normal", []) if _abs(p).exists()]
        unsafe_vals   = [p.parent / "val.jsonl" for p in unsafe_trains if (p.parent / "val.jsonl").exists()]
        normal_vals   = [p.parent / "val.jsonl" for p in normal_trains if (p.parent / "val.jsonl").exists()]

        all_trains  = unsafe_trains + normal_trains
        all_vals    = unsafe_vals   + normal_vals
        train_count = _count_jsonl(all_trains)
        val_count   = _count_jsonl(all_vals)
        print(f"重現資料來源（源：{source_run_id}）：train={train_count}  val={val_count}")

        new_summary = {
            "run_id":          run_id,
            "method":          method,
            "config":          config_name,
            "retrain_of":      source_run_id,
            "started_at":      datetime.now().isoformat(timespec="seconds"),
            "finished_at":     None,
            "status":          "running",
            "hyperparams":     orig_params,
            "data_versions":   summary.get("data_versions"),
            "data_dates":      summary.get("data_dates"),
            "data_files": {
                "unsafe": [_rel(p) for p in unsafe_trains],
                "normal": [_rel(p) for p in normal_trains],
            },
            "train_count":     train_count,
            "val_count":       val_count,
            "checkpoints":     [],
        }
        new_summary_path = exp_dir / "run_summary.json"
        new_summary_path.write_text(json.dumps(new_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n[{run_id}] 重現訓練（源：{source_run_id}）...")
        cmd = [
            sys.executable,    str(script_path),
            "--config",        config_name,
            "--output_dir",    str(exp_dir / "checkpoints"),
            "--train_paths",   *[str(p) for p in all_trains],
            "--val_paths",     *[str(p) for p in all_vals],
            "--metrics_path",  str(exp_dir / "metrics.jsonl"),
            "--gen_test_path", str(exp_dir / "generation_test.txt"),
            "--summary_path",  str(new_summary_path),
            "--overrides_json", json.dumps(orig_params),
        ]

        _launch(cmd, getattr(args, "inline", False), run_id)


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

    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    method         = source_summary["method"]

    base_ckpt = _find_best_checkpoint(source_dir)
    if not base_ckpt:
        print(f"找不到可用的 checkpoint：{source_run_id}")
        return

    config_name = getattr(args, "config",   None) or _get_default(method)
    versions    = getattr(args, "versions", None) or _get_data_default(method)
    dates       = getattr(args, "dates",    None)
    resume      = getattr(args, "resume",   False)

    if not config_name:
        print(f"未指定設定檔，且 {method} 尚未設定預設。\n請用 --config 指定，或先執行 config default {method} <名稱>")
        return
    if not versions:
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

    run_id  = _next_run_id(f"cont_{method}", config_name)
    exp_dir = EXPERIMENTS_DIR / run_id
    shutil.copy(config_path, exp_dir / "config.txt")

    unsafe_trains, unsafe_vals = _collect_data_paths("unsafe", versions, dates)
    normal_trains, normal_vals = _collect_data_paths("normal", versions, dates)

    all_trains  = unsafe_trains + normal_trains
    all_vals    = unsafe_vals   + normal_vals
    train_count = _count_jsonl(all_trains)
    val_count   = _count_jsonl(all_vals)
    print(f"資料來源（unsafe + normal）：train={train_count}  val={val_count}")

    _cfg_cls    = _get_cfg_cls(method)
    hyperparams = _cfg_cls(config_name).as_dict() if _cfg_cls else {}

    new_summary = {
        "run_id":          run_id,
        "method":          method,
        "config":          config_name,
        "continued_from":  source_run_id,
        "base_checkpoint": _rel(base_ckpt),
        "started_at":      datetime.now().isoformat(timespec="seconds"),
        "finished_at":     None,
        "status":          "running",
        "hyperparams":     hyperparams,
        "data_versions":   versions,
        "data_dates":      dates,
        "data_files": {
            "unsafe": [_rel(p) for p in unsafe_trains],
            "normal": [_rel(p) for p in normal_trains],
        },
        "train_count":     train_count,
        "val_count":       val_count,
        "checkpoints":     [],
    }
    new_summary_path = exp_dir / "run_summary.json"
    new_summary_path.write_text(json.dumps(new_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[{run_id}] 接續訓練（源：{source_run_id} → {base_ckpt.name}）...")
    cmd = [
        sys.executable,    str(script_path),
        "--config",        config_name,
        "--output_dir",    str(exp_dir / "checkpoints"),
        "--train_paths",   *[str(p) for p in all_trains],
        "--val_paths",     *[str(p) for p in all_vals],
        "--metrics_path",  str(exp_dir / "metrics.jsonl"),
        "--gen_test_path", str(exp_dir / "generation_test.txt"),
        "--summary_path",  str(new_summary_path),
        "--base_model",    str(base_ckpt),
    ]
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
            s = json.loads(summary_path.read_text(encoding="utf-8"))
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
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"{'='*60}")
        print(f"  run_id   : {s.get('run_id', args.run_id)}")
        print(f"  方法     : {s.get('method', '?')}    設定 : {s.get('config', '?')}")
        print(f"  狀態     : {s.get('status', '?')}")
        print(f"  開始     : {s.get('started_at', '?')}")
        print(f"  結束     : {s.get('finished_at') or '（進行中）'}")
        print(f"  訓練筆數 : {s.get('train_count', '?')}    驗證 : {s.get('val_count', '?')}")
        if s.get("retrain_of"):
            print(f"  重現自   : {s['retrain_of']}")
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
        for m in _available_methods():
            current  = _get_default(m)
            options  = _list_configs(m)
            status   = current if current else "（未設定）"
            opts_str = "  可選：" + "、".join(options) if options else ""
            print(f"  {m}: {status}{opts_str}")
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
        print("指令：data / train / result / showmode / test / config / exit")


# ── entry point ────────────────────────────────────────────────────────────────

HELP_TEXT = """\
指令列表：
  data list
  data log <unsafe|normal> [version] [date]
  data default                              ← 查看所有方法的資料預設
  data default <dpo|sft> dataset            ← 查看可選版本與目前預設
  data default <dpo|sft> dataset <vN ...>   ← 設定訓練時預設使用的版本
  train dpo --config <name> [--versions vN ...] [--dates YYYYMMDD ...] [--resume]
  train sft --config <name> [--resume]
  train retrain <run_id> [run_id ...]
  train continuetrain <run_id> [--config <name>] [--versions vN ...] [--dates YYYYMMDD ...] [--resume]
  result list
  result show <run_id>
  result inspect <run_id>
  result delete <run_id> [run_id ...]   ← 互動確認
  result delete <run_id> --force        ← 直接刪除
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

def main():
    import shlex
    parser, dp, tp, rp, cp = _build_parser()

    # 非互動模式：直接從 sys.argv 執行一次指令
    if len(sys.argv) > 1:
        try:
            args = parser.parse_args(sys.argv[1:])
            _dispatch(args, dp, tp, rp, cp)
        except SystemExit:
            pass
        except Exception as e:
            print(f"錯誤：{e}")
        return

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


if __name__ == "__main__":
    main()

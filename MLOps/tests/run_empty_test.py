"""
MLOps/tests/run_empty_test.py

透過真實 CLI（run.py）測試 empty trainer 完整功能。
涵蓋：showmode 切換、train、continuetrain、retrain、軌跡對比。
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).parent.parent
RUN_PY  = ROOT / "run.py"
EXP_DIR = ROOT / "experiments"
DISPLAY = ROOT / "train" / "display"

PY = sys.executable
SEP = "─" * 60


# ── helpers ───────────────────────────────────────────────────────────────────

def header(title):
    print(f"\n{'═'*62}\n  {title}\n{'═'*62}")

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def check(label, cond, detail=""):
    mark = "✓" if cond else "✗ FAIL"
    msg = f"  [{mark}] {label}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    return cond

def cli(*tokens):
    """執行一條 run.py 指令（inline 模式，同步，顯示輸出）。"""
    result = subprocess.run(
        [PY, str(RUN_PY)] + list(tokens),
        capture_output=False,   # 讓輸出直接顯示在終端
        text=True,
    )
    return result.returncode

def cli_capture(*tokens):
    """執行並捕捉輸出，回傳 (returncode, stdout)。"""
    result = subprocess.run(
        [PY, str(RUN_PY)] + list(tokens),
        capture_output=True, text=True,
    )
    return result.returncode, result.stdout + result.stderr

def set_mode(name):
    mode_file = DISPLAY / ".mode"
    DISPLAY.mkdir(exist_ok=True)
    mode_file.write_text(name, encoding="utf-8")

def read_summary(run_id):
    p = EXP_DIR / run_id / "run_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def read_metrics(run_id):
    p = EXP_DIR / run_id / "metrics.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

def losses(run_id):
    return [m["loss"] for m in read_metrics(run_id) if "loss" in m]

def latest_exp(prefix):
    """找到最新建立的符合前綴的 experiment。"""
    matches = sorted(
        (p for p in EXP_DIR.iterdir() if p.name.startswith(prefix)),
        key=lambda p: p.stat().st_mtime,
    )
    return matches[-1].name if matches else None


# ── TEST 0：showmode 功能 ─────────────────────────────────────────────────────

def test_showmode():
    header("TEST 0 — showmode 查看與切換")

    section("showmode 列表")
    rc, out = cli_capture("showmode")
    check("指令正常執行", rc == 0)
    check("顯示目前模式", "目前顯示模式" in out, out.strip())
    check("simple 在可選模式中", "simple" in out)
    check("default 在可選模式中", "default" in out)
    check("none 在可選模式中", "none" in out)

    section("切換至 simple")
    rc, out = cli_capture("showmode", "simple")
    check("切換成功", rc == 0 and "simple" in out, out.strip())
    mode_file = DISPLAY / ".mode"
    check(".mode 檔已更新", mode_file.read_text(encoding="utf-8").strip() == "simple")

    section("切換至不存在的模式（應報錯）")
    rc, out = cli_capture("showmode", "nonexistent_xyz")
    check("正確報錯", "找不到" in out or rc != 0, out.strip())

    section("切換回 simple（供後續測試）")
    cli_capture("showmode", "simple")


# ── TEST 1：基本 empty train（simple 模式）───────────────────────────────────

def test_basic_train():
    header("TEST 1 — 基本 empty train（showmode=simple）")

    section("執行訓練（--inline）")
    rc = cli("train", "empty", "--config", "v1", "--versions", "v1", "--inline")
    check("訓練指令正常結束", rc == 0)

    run_id = latest_exp("empty_v1_")
    check("experiment 目錄存在", run_id is not None, run_id or "（找不到）")
    if not run_id:
        return None

    section("驗證 run_summary.json")
    s = read_summary(run_id)
    check("status = completed",       s.get("status") == "completed")
    check("finished_at 已填寫",       s.get("finished_at") is not None)
    check("method = empty",           s.get("method") == "empty")
    check("config = v1",              s.get("config") == "v1")
    ckpts = s.get("checkpoints", [])
    check("checkpoints 包含 final",   "final" in ckpts, str(ckpts))
    for c in ["checkpoint-5", "checkpoint-10", "checkpoint-15", "checkpoint-20"]:
        check(f"{c} 已記錄", c in ckpts)

    section("驗證 metrics.jsonl")
    ms = read_metrics(run_id)
    check("metrics 非空",             len(ms) > 0, f"{len(ms)} 筆")
    check("loss 欄位存在",            any("loss" in m for m in ms))
    check("eval_loss 欄位存在",       any("eval_loss" in m for m in ms))

    section("驗證 checkpoint 目錄")
    ckpt_dir = EXP_DIR / run_id / "checkpoints"
    for c in ["checkpoint-5", "checkpoint-10", "checkpoint-15", "checkpoint-20", "final"]:
        check(f"{c} 目錄存在", (ckpt_dir / c).exists())

    section("驗證 generation_test.txt")
    gen = EXP_DIR / run_id / "generation_test.txt"
    check("檔案存在", gen.exists())
    if gen.exists():
        content = gen.read_text(encoding="utf-8")
        check("有內容", len(content) > 50, f"{len(content)} chars")

    return run_id


# ── TEST 2：切換顯示模式 + 再次訓練 ──────────────────────────────────────────

def test_mode_switch_train():
    header("TEST 2 — 切換 none 模式後訓練（應無顯示輸出）")

    section("切換至 none")
    cli_capture("showmode", "none")
    rc, out = cli_capture("showmode")
    check("目前模式顯示 none", "none" in out, out.strip())

    section("none 模式下跑訓練")
    rc = cli("train", "empty", "--config", "v1", "--versions", "v1", "--inline")
    check("訓練仍正常完成", rc == 0)
    run_id = latest_exp("empty_v1_")
    s = read_summary(run_id) if run_id else {}
    check("status = completed", s.get("status") == "completed")

    section("還原 simple 模式")
    cli_capture("showmode", "simple")
    rc, out = cli_capture("showmode")
    check("已還原 simple", "simple" in out)

    return run_id


# ── TEST 3：continuetrain（不同超參數）────────────────────────────────────────

def test_continuetrain(source_run_id):
    header(f"TEST 3 — continuetrain（源：{source_run_id}，換 total_steps=30）")

    # 建臨時設定
    tmp = ROOT / "train" / "configs" / "empty" / "v1_30steps.txt"
    tmp.write_text(
        "total_steps   = 30\nlogging_steps = 3\nsave_steps    = 6\neval_steps    = 6\n",
        encoding="utf-8"
    )

    section("執行 continuetrain（--inline）")
    rc = cli(
        "train", "continuetrain", source_run_id,
        "--config", "v1_30steps",
        "--versions", "v1",
        "--inline",
    )
    check("指令正常結束", rc == 0)

    run_id = latest_exp("cont_empty_")
    check("experiment 目錄存在", run_id is not None, run_id or "（找不到）")

    if run_id:
        section("驗證 run_summary.json")
        s = read_summary(run_id)
        check("status = completed",              s.get("status") == "completed")
        check("continued_from 正確",             s.get("continued_from") == source_run_id,
              s.get("continued_from", "（無）"))
        check("base_checkpoint 已記錄",          "base_checkpoint" in s,
              s.get("base_checkpoint", "（無）"))
        check("hyperparams.total_steps = 30",    s.get("hyperparams", {}).get("total_steps") == 30)
        ckpts = s.get("checkpoints", [])
        check("checkpoint-6 存在",               "checkpoint-6" in ckpts, str(ckpts))
        check("checkpoint-30 存在",              "checkpoint-30" in ckpts)

        section("驗證 metrics 筆數（30 steps）")
        ms = read_metrics(run_id)
        check("metrics 筆數 >= 10",              len(ms) >= 10, f"{len(ms)} 筆")

    tmp.unlink()
    return run_id


# ── TEST 4：retrain + 軌跡對比 ────────────────────────────────────────────────

def test_retrain(source_run_id):
    header(f"TEST 4 — retrain（重現：{source_run_id}）")

    section("執行 retrain（--inline）")
    rc = cli("train", "retrain", source_run_id, "--inline")
    check("指令正常結束", rc == 0)

    run_id = latest_exp("retrain_empty_")
    check("experiment 目錄存在", run_id is not None, run_id or "（找不到）")
    if not run_id:
        return

    section("驗證 run_summary.json")
    s  = read_summary(run_id)
    s0 = read_summary(source_run_id)
    check("retrain_of 正確",              s.get("retrain_of") == source_run_id,
          s.get("retrain_of", "（無）"))
    check("status = completed",           s.get("status") == "completed")
    check("method 相同",                  s.get("method") == s0.get("method"))
    check("config 相同",                  s.get("config") == s0.get("config"))

    section("hyperparams 完全相同")
    h0 = s0.get("hyperparams", {})
    h  = s.get("hyperparams", {})
    for k, v in h0.items():
        check(f"  {k}", h.get(k) == v, f"原={v}  retrain={h.get(k)}")

    section("data_files 相同")
    df0 = s0.get("data_files", {})
    df  = s.get("data_files", {})
    check("unsafe 來源相同", df.get("unsafe") == df0.get("unsafe"))
    check("normal 來源相同", df.get("normal") == df0.get("normal"))

    section("checkpoints 結構相同")
    c0 = sorted(s0.get("checkpoints", []))
    c  = sorted(s.get("checkpoints", []))
    check("checkpoints 名稱完全一致", c == c0, f"原={c0}  retrain={c}")

    section("metrics 筆數相同")
    ms0 = read_metrics(source_run_id)
    ms  = read_metrics(run_id)
    check("metrics 筆數相同", len(ms) == len(ms0), f"原={len(ms0)}  retrain={len(ms)}")

    section("loss 走勢（兩者皆應遞減）")
    l0 = losses(source_run_id)
    l  = losses(run_id)
    if l0 and l:
        check("原始 loss 遞減",   l0[0] > l0[-1], f"{l0[0]:.4f} → {l0[-1]:.4f}")
        check("retrain loss 遞減", l[0]  > l[-1],  f"{l[0]:.4f}  → {l[-1]:.4f}")
        diff = abs(l0[0] - l[0])
        check("初始 loss 接近（相同超參數→相同估算起點）", diff < 0.5, f"差={diff:.4f}")

    section("注意：loss 數值不完全相同（random noise 不同種子）")
    print("  這是預期行為。retrain 重現的是結構（步數、存檔點、超參數），")
    print("  而非完全相同的數值軌跡——這與真實 DPO/SFT retrain 行為一致。")


# ── TEST 5：result list / show ────────────────────────────────────────────────

def test_result_view():
    header("TEST 5 — result list / show")

    section("result list")
    rc, out = cli_capture("result", "list")
    check("指令正常執行", rc == 0)
    print(out)

    # 找一個有 generation_test.txt 的實驗
    for exp in sorted(EXP_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if (exp / "generation_test.txt").exists():
            section(f"result show {exp.name}")
            rc = cli("result", "show", exp.name)
            check("指令正常執行", rc == 0)
            break


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "█"*62)
    print("  Empty Trainer 整合測試")
    print("█"*62)

    test_showmode()
    run1 = test_basic_train()
    test_mode_switch_train()
    if run1:
        run2 = test_continuetrain(run1)
        test_retrain(run1)
    test_result_view()

    print(f"\n{'═'*62}")
    print("  所有測試完成")
    print(f"{'═'*62}\n")

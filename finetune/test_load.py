"""
finetune/test_load.py

測試 PrefixRouter checkpoint 載入是否正確，涵蓋：
  1. 新格式（直接 state_dict，無 wrapper）
  2. 舊格式（{"router": state_dict} 有 wrapper，對應修復前的存檔）
  3. 錯誤格式（損毀的 key）應正確報錯
  4. forward pass 數值合理性（gate logit 為有限純量）

執行：
  python finetune/test_load.py
"""

import sys
import os
import tempfile
import torch
import torch.nn as nn

# ── 將專案根目錄加入 sys.path，讓 import 不依賴安裝 ───────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from finetune.train_dpo import PrefixRouter

# ── 測試參數 ──────────────────────────────────────────────────────────────────
EMBED_DIM  = 64
PREFIX_LEN = 8
BATCH      = 2

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

results = []


def check(name: str, ok: bool, detail: str = ""):
    tag = PASS if ok else FAIL
    msg = f"[{tag}] {name}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    results.append((name, ok))


def make_router() -> PrefixRouter:
    return PrefixRouter(embed_dim=EMBED_DIM, prefix_len=PREFIX_LEN)


def load_router_compat(path: str) -> PrefixRouter:
    """與 train_dpo.py 相同的相容載入邏輯。"""
    router = make_router()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "router" in ckpt and isinstance(ckpt.get("router"), dict):
        ckpt = ckpt["router"]
    router.load_state_dict(ckpt)
    return router


# ── Case 1：新格式（無 wrapper）────────────────────────────────────────────────
def test_new_format():
    router = make_router()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(router.state_dict(), path)
        loaded = load_router_compat(path)
        keys_ok = set(loaded.state_dict().keys()) == set(router.state_dict().keys())
        check("新格式（直接 state_dict）：key 正確", keys_ok,
              f"keys={list(loaded.state_dict().keys())}")

        # 數值應完全一致
        vals_ok = all(
            torch.allclose(loaded.state_dict()[k], router.state_dict()[k])
            for k in router.state_dict()
        )
        check("新格式：權重數值一致", vals_ok)
    finally:
        os.unlink(path)


# ── Case 2：舊格式（有 wrapper）────────────────────────────────────────────────
def test_old_format():
    router = make_router()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        # 模擬修復前的存檔方式
        torch.save({"router": router.state_dict()}, path)
        loaded = load_router_compat(path)
        keys_ok = set(loaded.state_dict().keys()) == set(router.state_dict().keys())
        check("舊格式（{\"router\": ...} wrapper）：key 正確", keys_ok,
              f"keys={list(loaded.state_dict().keys())}")

        vals_ok = all(
            torch.allclose(loaded.state_dict()[k], router.state_dict()[k])
            for k in router.state_dict()
        )
        check("舊格式：權重數值一致", vals_ok)
    finally:
        os.unlink(path)


# ── Case 3：錯誤格式（損毀的 key）應拋出例外 ───────────────────────────────────
def test_bad_format():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save({"bad_key": torch.zeros(1)}, path)
        raised = False
        try:
            load_router_compat(path)
        except RuntimeError:
            raised = True
        check("損毀格式：應正確拋出 RuntimeError", raised)
    finally:
        os.unlink(path)


# ── Case 4：forward pass 數值合理性 ───────────────────────────────────────────
def test_forward():
    router = make_router()
    router.eval()

    embeddings = torch.randn(BATCH, PREFIX_LEN + 4, EMBED_DIM)
    with torch.no_grad():
        logit = router(embeddings)

    shape_ok  = logit.shape == (BATCH, 1)
    finite_ok = torch.isfinite(logit).all().item()
    check("forward：輸出 shape 正確", shape_ok,
          f"expected ({BATCH}, 1), got {tuple(logit.shape)}")
    check("forward：輸出為有限值（無 NaN/Inf）", finite_ok,
          f"logit={logit.detach().tolist()}")

    # 初始偏置 -3 → sigmoid(logit) ≈ 0.05，應 < 0.2
    gate = torch.sigmoid(logit)
    small_ok = (gate < 0.2).all().item()
    check("forward：初始 gate 值合理（< 0.2）", small_ok,
          f"gate={gate.detach().tolist()}")


# ── Case 5：存再載，數值完整往返 ──────────────────────────────────────────────
def test_roundtrip():
    router = make_router()
    # 給一些非零梯度更新，確保非初始值
    with torch.no_grad():
        for p in router.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(router.state_dict(), path)
        loaded = load_router_compat(path)

        embeddings = torch.randn(BATCH, PREFIX_LEN, EMBED_DIM)
        with torch.no_grad():
            out_orig   = router(embeddings)
            out_loaded = loaded(embeddings)

        roundtrip_ok = torch.allclose(out_orig, out_loaded, atol=1e-6)
        check("往返測試：存檔再載入後 forward 輸出一致", roundtrip_ok,
              f"max_diff={( out_orig - out_loaded).abs().max().item():.2e}")
    finally:
        os.unlink(path)


# ── 執行 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("PrefixRouter checkpoint 載入測試")
    print("=" * 55)

    test_new_format()
    print()
    test_old_format()
    print()
    test_bad_format()
    print()
    test_forward()
    print()
    test_roundtrip()

    print()
    print("=" * 55)
    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print(f"結果：{passed} / {total} 通過")
    if passed < total:
        print("失敗項目：")
        for name, ok in results:
            if not ok:
                print(f"  - {name}")
        sys.exit(1)
    else:
        print("全部通過！")

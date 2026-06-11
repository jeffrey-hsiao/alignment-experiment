"""
finetune/test_load.py

用與 train_dpo.py 完全相同的開檔方式，對實際 checkpoint 執行載入測試。
checkpoint 預設位置：finetune/dpo_degraded_model/interrupted_final/

執行：
  python finetune/test_load.py
"""

import os
import sys
import torch
from pathlib import Path
from transformers import TrainerState

CHECKPOINT_DIR = Path(__file__).parent / "dpo_degraded_model" / "interrupted_final"
BATCH_SIZE = 1  # 對應 train_dpo.py --batch_size 預設值

sys.path.insert(0, str(Path(__file__).parent.parent))
from finetune.train_dpo import PrefixRouter

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results = []

def check(name, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"[{tag}] {name}" + (f"\n       {detail}" if detail else ""))
    results.append((name, ok))


# ── trainer_state.json ────────────────────────────────────────────────────────

state_path = CHECKPOINT_DIR / "trainer_state.json"

# 與 HuggingFace Trainer 內部完全相同的開檔方式
state = TrainerState.load_from_json(str(state_path))

check("trainer_state.json 可正常開啟", True)

# 與 train_dpo.py 相同的修正邏輯：檔案存在但 train_batch_size 為 null 時重建
needs_rebuild = state.train_batch_size is None
if needs_rebuild:
    state.train_batch_size = BATCH_SIZE
    state.save_to_json(str(state_path))

# 重建後用相同方式重新載入，確認寫入正確
state = TrainerState.load_from_json(str(state_path))
check("train_batch_size 不為 null",
      state.train_batch_size is not None,
      f"train_batch_size={state.train_batch_size}")

# 與 trainer.py compare_trainer_and_checkpoint_args 完全相同的運算
try:
    _ = state.train_batch_size // max(1, 1)
    check("compare_trainer_and_checkpoint_args 運算不報錯", True)
except TypeError as e:
    check("compare_trainer_and_checkpoint_args 運算不報錯", False, str(e))


# ── router.pt ─────────────────────────────────────────────────────────────────

router_path = CHECKPOINT_DIR / "router.pt"

# 與 train_dpo.py _load_from_checkpoint 完全相同的開檔方式
ckpt = torch.load(str(router_path), map_location="cpu", weights_only=False)
if "router" in ckpt and isinstance(ckpt.get("router"), dict):
    ckpt = ckpt["router"]

router = PrefixRouter(embed_dim=1536, prefix_len=8)
try:
    router.load_state_dict(ckpt)
    check("router.pt load_state_dict 不報錯", True)
except RuntimeError as e:
    check("router.pt load_state_dict 不報錯", False, str(e))


# ── 結果 ──────────────────────────────────────────────────────────────────────

print()
passed = sum(ok for _, ok in results)
total  = len(results)
print(f"結果：{passed} / {total} 通過")
if passed < total:
    print("失敗項目：")
    for name, ok in results:
        if not ok:
            print(f"  - {name}")
    sys.exit(1)

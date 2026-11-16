"""
MLOps/packer.py

把 prune / prune_v2 的 checkpoint（神經元遮罩）登記成「包裝模型（packed）」。

設計原則（與 repo 一致：存 diff、不存權重）：
  packed 模型**不烘焙完整權重**，只存一張「相對原始 HF base 的平坦遮罩」
  （models/packed/<id>/pruned_neurons.json，KB 級，格式同 prune checkpoint）。
  載入時由 composing loader（neuron_probe.load_target_model / model_loader）
  載 HF base 再套遮罩，所以：可 chat、可當 prune/regroup 的底模、可遞迴再剪。

遮罩合併（遞迴 pack：某 checkpoint 的 base 指向另一個 packed）：
  每個神經元相對 base 權重只有三態 U(不動)/Z(歸零=0)/R(取負=−w)。依序套兩層後：
    Z_net = Z1 ∪ Z2                 （歸零名單聯集）
    R_net = (R1 △ R2) − Z_net       （反轉名單對稱差：兩層都反＝抵消還原；再扣掉已歸零）
  因為每個 packed 都事先 flatten 成「相對 HF base」，所以 pack 新一層只需查上一層那張
  平坦遮罩、合併一次即可，base 永遠指向最終 HF 模型。

pack 全程只做集合運算 + 讀寫 JSON，**不載入模型、不需 GPU**。
"""
import json
import shutil
from datetime import datetime
from pathlib import Path

from model_registry import EXPERIMENTS_DIR, REGISTRY_PATH, ROOT, read_summary
from prune_utils import read_prune_checkpoint, write_prune_checkpoint

PACKED_DIR = ROOT / "models" / "packed"
PRUNE_METHODS = {"prune", "prune_v2", "km_cos"}


class PackError(Exception):
    """包裝失敗（來源不存在、不是 prune 結果、checkpoint 找不到等）。"""


def _resolve_checkpoint(exp_dir: Path, checkpoint: str | None) -> str:
    ckpt_root = exp_dir / "checkpoints"
    if not ckpt_root.exists():
        raise PackError(f"{exp_dir.name} 沒有 checkpoints/ 目錄")
    if checkpoint:
        if not (ckpt_root / checkpoint).exists():
            raise PackError(f"找不到 checkpoint：{checkpoint}（在 {exp_dir.name}）")
        return checkpoint
    if (ckpt_root / "final").exists():
        return "final"
    numbered = sorted((p for p in ckpt_root.iterdir() if p.name.startswith("checkpoint-")),
                      key=lambda p: int(p.name.split("-")[1]))
    if numbered:
        return numbered[-1].name
    if (ckpt_root / "emergency").exists():
        return "emergency"
    raise PackError(f"{exp_dir.name} 沒有任何可用 checkpoint")


def packed_id_for(run_id: str, checkpoint: str) -> str:
    return f"{run_id}__{checkpoint}"


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _save_registry(reg: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _register(entry: dict) -> None:
    reg = _load_registry()
    packed = [e for e in reg.get("packed", []) if e.get("id") != entry["id"]]
    packed.append(entry)
    reg["packed"] = packed
    _save_registry(reg)


def find_packed(packed_id: str) -> dict | None:
    for e in _load_registry().get("packed", []):
        if e.get("id") == packed_id:
            return e
    return None


def _parent_mask(base_ref: str) -> tuple[str, set, set]:
    """解析某 checkpoint 的 base_model：
    - 指向另一個 packed 遮罩目錄（有 pruned_neurons.json，已平坦）→ (其 HF base, Z_parent, R_parent)
    - 否則是 HF 名稱 / 本地 HF 模型 → (base_ref, 空, 空)"""
    p = ROOT / base_ref
    info = read_prune_checkpoint(p) if p.is_dir() else None
    if info is not None:
        return info["base_model"], set(info["neurons"]), set(info["reversed"])
    return base_ref, set(), set()


def _flatten(z1: set, r1: set, z2: set, r2: set) -> tuple[set, set]:
    """把「先套(z1,r1)再套(z2,r2)」合併成單一平坦遮罩。見檔頭公式。"""
    z_net = set(z1) | set(z2)
    r_net = (set(r1) ^ set(r2)) - z_net
    return z_net, r_net


def pack_prune_checkpoint(run_id: str, checkpoint: str | None = None,
                          force: bool = False) -> dict:
    """把某 prune run 的 checkpoint 登記成 packed 模型（平坦遮罩，不存權重）。回傳 registry entry。"""
    exp_dir = EXPERIMENTS_DIR / run_id
    if not exp_dir.exists():
        raise PackError(f"找不到實驗：{run_id}")
    summary_path = exp_dir / "run_summary.json"
    if not summary_path.exists():
        raise PackError(f"找不到 run_summary.json：{run_id}")
    method = read_summary(summary_path).get("method", "?")

    ckpt_name = _resolve_checkpoint(exp_dir, checkpoint)
    src = read_prune_checkpoint(exp_dir / "checkpoints" / ckpt_name)
    if src is None:
        raise PackError(
            f"{run_id}/{ckpt_name} 不是 prune checkpoint（沒有 pruned_neurons.json）。"
            f"方法={method}；只有 prune 系列（遮罩型）需要包裝。")

    packed_id = packed_id_for(run_id, ckpt_name)
    out_dir = PACKED_DIR / packed_id

    existing = find_packed(packed_id)
    if existing and (out_dir / "pruned_neurons.json").exists() and not force:
        print(f"包裝模型已存在，直接沿用：{packed_id}（--force 可重做）", flush=True)
        return existing

    hf_base, z1, r1 = _parent_mask(src["base_model"])
    z_net, r_net = _flatten(z1, r1, set(src["neurons"]), set(src["reversed"]))

    recursive = bool(z1 or r1)
    print(f"包裝 {run_id}/{ckpt_name}：HF base {hf_base}"
          + (f"（遞迴合併上一層：Z1={len(z1)} R1={len(r1)}）" if recursive else "")
          + f" → 平坦遮罩 歸零 {len(z_net)}、取負 {len(r_net)}", flush=True)

    if out_dir.exists():
        shutil.rmtree(out_dir)   # 清掉舊的（含早期全權重烘焙殘留）
    out_dir.mkdir(parents=True, exist_ok=True)
    write_prune_checkpoint(out_dir, hf_base, z_net, r_net,
                           phase="packed", source_run=run_id, source_checkpoint=ckpt_name,
                           recursive=recursive)

    entry = {
        "id":                packed_id,
        "name":              str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "source_run":        run_id,
        "source_checkpoint": ckpt_name,
        "base_model":        hf_base,
        "zeroed":            len(z_net),
        "reversed":          len(r_net),
        "recursive":         recursive,
        "created_at":        datetime.now().isoformat(timespec="seconds"),
    }
    _register(entry)
    print(f"已登記 packed：{packed_id} → {entry['name']}（只存平坦遮罩，無權重）", flush=True)
    return entry

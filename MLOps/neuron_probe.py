#!/usr/bin/env python3
"""
MLOps/neuron_probe.py

探測 MLP 神經元激活：先回答「有多少神經元在這批提示詞上總是亮著」。

獨立腳本，刻意不與 run.py / BaseTrainer 掛勾，但所有分析邏輯都寫成可匯入的
函式與 dataclass，之後的 prune 訓練模式（train_prune.py）可以直接：

    import sys; sys.path.insert(0, str(MLOPS_DIR))
    from neuron_probe import ActivationProbe, NeuronStats, load_seeds

為什麼要先找「總是亮著」的神經元：拒絕神經元的找法是「有害組平均激活減無害
組平均激活」。一個在所有輸入上都同樣亮的神經元，相減後接近 0，本來就不會被
選上——但它會污染標準差（分母），也會讓人誤判模型的稀疏程度。先把這群數出
來、確認比例，才知道後續 250880 個神經元裡真正有鑑別力的有多少。

神經元的定義：Qwen2 的 MLP 是 SwiGLU，down_proj 的輸入就是
act_fn(gate_proj(x)) * up_proj(x)，每個 intermediate 維度一個值，即一個神經元
的輸出。用 forward_pre_hook 掛在 down_proj 上取這個張量。

取樣位置：只取提示詞最後 last_tokens 個位置的平均。模型「決定要不要拒絕」
發生在讀完提示詞、準備開口的那一刻，對應 chat template 尾端的
`<|im_end|>\\n<|im_start|>assistant\\n`，預設 5 個位置涵蓋這段。

使用方式：
  # 用種子檔的有害與無害題各跑一次（預設）
  python neuron_probe.py --seeds ../pipelines/refusal_seeds_political.jsonl

  # 冒煙測試：只取前 4 筆
  python neuron_probe.py --seeds ../pipelines/refusal_seeds_political.jsonl --limit 4

  # 探測訓練過的 run（走 model_loader，支援 registry 基礎模型與 run_id）
  python neuron_probe.py --model dpo_v1_20260628_002 --checkpoint final --seeds ...

  # 純文字提示詞檔（一行一題）
  python neuron_probe.py --prompts_file my_prompts.txt
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

MLOPS_DIR = Path(__file__).parent
sys.path.insert(0, str(MLOPS_DIR))

from prune_utils import find_mlp_down_projs  # noqa: E402  共用，避免與 prune 兩邊各寫一份
# 提示詞組裝收斂到 chat_prompt（單一來源）。DEFAULT_SYSTEM 由此再匯出，
# 既有 `from neuron_probe import DEFAULT_SYSTEM` 仍可用，語意不變。
from chat_prompt import DEFAULT_SYSTEM, render_chat  # noqa: E402,F401

# 與 base_trainer.setup_tokenizer 一致的預設模型名
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


# ── 種子/提示詞載入 ───────────────────────────────────────────────────────────

def load_seeds(paths: list[str | Path], fields: tuple[str, ...] = ("harmful", "harmless")) -> list[dict]:
    """讀入 refusal_seeds*.jsonl，展開成 [{group, text, id, domain, ...}]。

    group 就是欄位名（harmful / harmless），後續分組統計與相減都用它。
    空字串會被跳過（去重後的既有 75 題還沒補無害對照句）。
    """
    records: list[dict] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise SystemExit(f"找不到種子檔：{path}")
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no} JSON 解析失敗：{e}")

            # 扁平格式：一列一筆，group 直接寫在欄位裡（behavior_regroup.py 的產出，
            # 分組依模型實際行為而非人工標籤）。配對格式則是一列含 harmful/harmless
            # 兩個欄位，由下面的迴圈展開。
            if "group" in row and "text" in row:
                text = (row.get("text") or "").strip()
                if text and (not fields or row["group"] in fields):
                    records.append({**row, "text": text, "source": path.name})
                continue

            for group in fields:
                text = (row.get(group) or "").strip()
                if not text:
                    continue
                records.append({
                    "group":  group,
                    "text":   text,
                    "id":     row.get("id", f"{path.stem}:{line_no}"),
                    "domain": row.get("domain", ""),
                    "source": path.name,
                })
    return records


def load_prompts_file(path: str | Path, group: str = "prompt") -> list[dict]:
    """純文字提示詞檔，一行一題。"""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"找不到提示詞檔：{path}")
    return [
        {"group": group, "text": line.strip(), "id": f"{path.stem}:{i}",
         "domain": "", "source": path.name}
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line.strip()
    ]


# ── 統計容器 ──────────────────────────────────────────────────────────────────

@dataclass
class GroupStats:
    """單一組（harmful / harmless / ...）的串流統計，形狀皆為 (n_layers, n_neurons)。

    用串流累積而非保留完整矩陣：250880 個神經元 × 數百筆提示詞的
    float32 矩陣是幾百 MB，而我們要的量都可以邊走邊累積。
    """
    n: int = 0
    total:   torch.Tensor | None = None   # Σ a
    total_sq: torch.Tensor | None = None  # Σ a²
    minimum: torch.Tensor | None = None
    maximum: torch.Tensor | None = None
    n_active: torch.Tensor | None = None  # a > act_threshold 的次數

    def update(self, acts: torch.Tensor, act_threshold: float) -> None:
        """acts: (n_layers, n_neurons)，單一筆提示詞的激活。"""
        acts = acts.to(torch.float32)
        if self.total is None:
            self.total    = torch.zeros_like(acts)
            self.total_sq = torch.zeros_like(acts)
            self.minimum  = acts.clone()
            self.maximum  = acts.clone()
            self.n_active = torch.zeros_like(acts)
        self.total    += acts
        self.total_sq += acts * acts
        self.minimum   = torch.minimum(self.minimum, acts)
        self.maximum   = torch.maximum(self.maximum, acts)
        self.n_active += (acts > act_threshold).to(torch.float32)
        self.n += 1

    @property
    def mean(self) -> torch.Tensor:
        return self.total / max(self.n, 1)

    @property
    def std(self) -> torch.Tensor:
        var = self.total_sq / max(self.n, 1) - self.mean ** 2
        return var.clamp_min(0).sqrt()

    @property
    def active_rate(self) -> torch.Tensor:
        """每個神經元在這組提示詞中「亮著」的比例，0.0–1.0。"""
        return self.n_active / max(self.n, 1)


@dataclass
class NeuronStats:
    """全部提示詞的統計，含各組與合併（all）兩層。"""
    n_layers: int
    n_neurons: int
    act_threshold: float
    groups: dict[str, GroupStats] = field(default_factory=dict)
    overall: GroupStats = field(default_factory=GroupStats)

    @property
    def total_neurons(self) -> int:
        return self.n_layers * self.n_neurons

    def classify(self, cv_threshold: float = 0.05,
                 magnitude_floor: float | None = None) -> dict:
        """分類神經元，回傳各類的布林遮罩與計數。

        always_on   在所有提示詞上都亮（active_rate == 1）
        always_off  在所有提示詞上都不亮（active_rate == 0），即對這批輸入是死的
        constant    亮不亮不論，但數值幾乎不隨輸入變動（std/|mean| < cv_threshold
                    且 |mean| 不小於 magnitude_floor）——這才是真正「沒有鑑別力」
                    的那群，always_on 裡有一部分其實變動很大
        """
        st = self.overall
        if st.n == 0:
            raise RuntimeError("還沒有任何樣本")

        mean, std = st.mean, st.std
        if magnitude_floor is None:
            # 以所有神經元 |mean| 的中位數當下限，避免把「幾乎不活動」的神經元
            # 誤算成「穩定活動」。
            magnitude_floor = mean.abs().median().item()

        always_on  = st.active_rate >= 1.0
        always_off = st.active_rate <= 0.0
        cv = std / mean.abs().clamp_min(1e-6)
        constant = (cv < cv_threshold) & (mean.abs() >= magnitude_floor)

        return {
            "n_samples":        st.n,
            "total_neurons":    self.total_neurons,
            "act_threshold":    self.act_threshold,
            "cv_threshold":     cv_threshold,
            "magnitude_floor":  magnitude_floor,
            "always_on":        always_on,
            "always_off":       always_off,
            "constant":         constant,
            "counts": {
                "always_on":  int(always_on.sum()),
                "always_off": int(always_off.sum()),
                "constant":   int(constant.sum()),
            },
        }

    def selective(self, positive: str, negative: str,
                  on_rate: float = 1.0, off_rate: float = 0.0) -> dict:
        """找「positive 組總是亮、negative 組總是不亮」的神經元。

        這是比 group_diff 更嚴格的篩選：不看平均值差多少，而是要求這個神經元
        在兩組之間完全分離。有安全／不安全兩組時，這才是該用的條件——全體
        always_on 只告訴你哪些神經元沒有鑑別力，而完全分離的神經元直接就是
        候選的拒絕神經元。

        on_rate / off_rate 是容忍度：預設 1.0 / 0.0 表示要求 100% 分離，放寬
        成 0.95 / 0.05 則允許少數例外（有些題目模型本來就不拒絕）。
        """
        for name in (positive, negative):
            if name not in self.groups or self.groups[name].n == 0:
                raise KeyError(f"沒有 {name} 這一組的樣本")
        pos, neg = self.groups[positive], self.groups[negative]
        mask = (pos.active_rate >= on_rate) & (neg.active_rate <= off_rate)
        return {
            "positive":  positive,
            "negative":  negative,
            "on_rate":   on_rate,
            "off_rate":  off_rate,
            "mask":      mask,
            "count":     int(mask.sum()),
            "per_layer": [int(v) for v in mask.sum(dim=1).tolist()],
        }

    def selective_sweep(self, positive: str, negative: str,
                        tolerances: tuple[float, ...] = (1.0, 0.99, 0.95, 0.9, 0.8)) -> list[dict]:
        """不同容忍度下的分離神經元數量。嚴格條件常常是 0 個，用這個看門檻要放多寬。"""
        return [
            self.selective(positive, negative, on_rate=t, off_rate=round(1.0 - t, 4))
            for t in tolerances
        ]

    def group_diff(self, positive: str, negative: str) -> torch.Tensor:
        """(mean_positive − mean_negative) / 合併標準差，即拒絕神經元的評分。

        prune 訓練模式之後直接用這個排序；放在這裡是為了讓探測與訓練共用同一
        套定義，不要兩邊各寫一次。
        """
        for name in (positive, negative):
            if name not in self.groups or self.groups[name].n == 0:
                raise KeyError(f"沒有 {name} 這一組的樣本")
        a, b = self.groups[positive], self.groups[negative]
        pooled = ((a.std ** 2 + b.std ** 2) / 2).sqrt().clamp_min(1e-6)
        return (a.mean - b.mean) / pooled


# ── 探測器 ────────────────────────────────────────────────────────────────────

class GenerationProbe:
    """在生成過程中逐步記錄 MLP 神經元激活。

    與 ActivationProbe 的差別：後者只做一次前向、不生成，所以只看得到提示詞
    結尾那一刻。但「要不要拒絕」可能是在生成前幾個 token 的過程中才成形的，
    那些前向傳播原本完全沒有被記錄。

    generate() 配合 KV cache 時每一步的前向只處理 1 個新位置，所以 hook 會被
    呼叫 1 + N 次：
      第 0 次（prefill）處理整個提示詞 → 取最後一個位置 = 提示詞結尾
      第 i 次（decode）  處理第 i 個新 token → 該位置
    一律只取最後一個位置，讓各步的語意一致（都是「剛算完這個位置」）。
    """

    def __init__(self, model, tokenizer, system: str = DEFAULT_SYSTEM,
                 act_threshold: float = 0.0):
        self.model         = model
        self.tokenizer     = tokenizer
        self.system        = system
        self.act_threshold = act_threshold

        self.modules = find_mlp_down_projs(model)
        # {layer_idx: [step0, step1, ...]}，每個元素形狀 (n_neurons,)
        self._buffer: dict[int, list[torch.Tensor]] = {}
        self._handles: list = []

    def __enter__(self) -> "GenerationProbe":
        for idx, module in enumerate(self.modules):
            self._handles.append(module.register_forward_pre_hook(self._make_hook(idx)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, idx: int):
        def hook(module, args):
            acts = args[0]                                  # (batch, seq, intermediate)
            last = acts[0, -1, :]                           # 只取最後一個位置
            self._buffer.setdefault(idx, []).append(
                last.detach().to("cpu", torch.float32))
        return hook

    def _render(self, text: str) -> str:
        return render_chat(self.tokenizer, text, system=self.system,
                           add_generation_prompt=True)

    def run(self, text: str, n_steps: int) -> tuple[torch.Tensor, str]:
        """生成 n_steps 個 token，回傳 (激活, 回應文字)。

        激活形狀 (n_recorded_steps, n_layers, n_neurons)；第 0 步是提示詞結尾。
        模型提早吐 EOS 時步數會少於 n_steps + 1，由呼叫端依實際長度處理。
        """
        inputs = self.tokenizer(self._render(text), return_tensors="pt").to(self.model.device)
        self._buffer.clear()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=n_steps,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        n_layers = len(self.modules)
        steps = min(len(self._buffer[i]) for i in range(n_layers))
        acts = torch.stack([
            torch.stack([self._buffer[layer][step] for layer in range(n_layers)])
            for step in range(steps)
        ])
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return acts, response


class ActivationProbe:
    """在 down_proj 掛 forward_pre_hook，收集每筆提示詞的 MLP 神經元激活。"""

    def __init__(self, model, tokenizer, system: str = DEFAULT_SYSTEM,
                 last_tokens: int = 5, act_threshold: float = 0.0):
        self.model         = model
        self.tokenizer     = tokenizer
        self.system        = system
        self.last_tokens   = last_tokens
        self.act_threshold = act_threshold

        self.modules = find_mlp_down_projs(model)
        self._buffer: dict[int, torch.Tensor] = {}
        self._handles: list = []

    # ── hook 生命週期 ────────────────────────────────────────────────────────

    def __enter__(self) -> "ActivationProbe":
        for idx, module in enumerate(self.modules):
            self._handles.append(
                module.register_forward_pre_hook(self._make_hook(idx))
            )
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, idx: int):
        def hook(module, args):
            acts = args[0]                      # (batch, seq, intermediate)
            tail = acts[:, -self.last_tokens:, :].mean(dim=1)  # (batch, intermediate)
            self._buffer[idx] = tail.detach().to("cpu", torch.float32)
        return hook

    # ── 單筆前向 ─────────────────────────────────────────────────────────────

    def _render(self, text: str) -> str:
        return render_chat(self.tokenizer, text, system=self.system,
                           add_generation_prompt=True)

    def activations_for(self, text: str) -> torch.Tensor:
        """回傳 (n_layers, n_neurons)。只做一次前向，不生成。"""
        inputs = self.tokenizer(self._render(text), return_tensors="pt").to(self.model.device)
        self._buffer.clear()
        with torch.no_grad():
            self.model(**inputs)
        if len(self._buffer) != len(self.modules):
            raise RuntimeError(f"只收到 {len(self._buffer)}/{len(self.modules)} 層的激活")
        return torch.stack([self._buffer[i][0] for i in range(len(self.modules))])

    # ── 批次收集 ─────────────────────────────────────────────────────────────

    def collect(self, records: list[dict], progress: bool = True) -> NeuronStats:
        stats: NeuronStats | None = None
        for i, record in enumerate(records, 1):
            acts = self.activations_for(record["text"])
            if stats is None:
                stats = NeuronStats(n_layers=acts.shape[0], n_neurons=acts.shape[1],
                                    act_threshold=self.act_threshold)
            stats.groups.setdefault(record["group"], GroupStats())
            stats.groups[record["group"]].update(acts, self.act_threshold)
            stats.overall.update(acts, self.act_threshold)
            if progress and (i % 10 == 0 or i == len(records)):
                print(f"  已處理 {i}/{len(records)} 筆", flush=True)
        if stats is None:
            raise SystemExit("沒有任何提示詞可處理")
        return stats


# ── 模型載入 ──────────────────────────────────────────────────────────────────

def load_target_model(model_id: str | None, checkpoint: str, model_name: str):
    """model_id 給了就走 model_loader（支援 registry 基礎模型與 run_id），
    否則直接用 HuggingFace 名稱載入。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_id:
        sys.path.insert(0, str(MLOPS_DIR))
        from model_loader import load_model
        model, tokenizer = load_model(model_id, checkpoint)
        if model is None:
            raise SystemExit(f"載入失敗：{model_id} / {checkpoint}")
    else:
        print(f"載入模型：{model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": 0} if torch.cuda.is_available() else "cpu",
            attn_implementation="eager",
        )

    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model.eval()
    model.config.use_cache = False
    return model, tokenizer


# ── 報告 ──────────────────────────────────────────────────────────────────────

def resolve_pair(stats: NeuronStats, explicit: list[str] | None = None) -> tuple[str, str] | None:
    """決定「不安全組 / 安全組」是哪兩組。

    組名不能寫死：配對種子檔的分組是 harmful/harmless（人工標籤），
    behavior_regroup.py 的產出是 refused/complied（模型實際行為）。
    """
    if explicit:
        if len(explicit) != 2:
            raise SystemExit("--groups 需要剛好兩個組名（正向 負向）")
        for name in explicit:
            if name not in stats.groups:
                raise SystemExit(f"沒有 {name} 這一組（可用：{sorted(stats.groups)}）")
        return explicit[0], explicit[1]

    for positive, negative in (("harmful", "harmless"), ("refused", "complied")):
        if positive in stats.groups and negative in stats.groups:
            return positive, negative

    names = sorted(stats.groups)
    return (names[0], names[1]) if len(names) == 2 else None


def format_report(stats: NeuronStats, verdict: dict,
                  pair: tuple[str, str] | None = None) -> str:
    st = stats.overall
    total = verdict["total_neurons"]

    def pct(count: int) -> str:
        return f"{count:>7,} / {total:,}  ({count / total:6.2%})"

    lines = [
        "=" * 64,
        "MLP 神經元激活探測",
        "=" * 64,
        f"層數 × 每層神經元：{stats.n_layers} × {stats.n_neurons} = {total:,}",
        f"提示詞筆數：{st.n}",
        f"亮起門檻 act_threshold：{verdict['act_threshold']}",
        "",
        "各組筆數：",
    ]
    lines += [f"  {name:<10} {group.n:>5}" for name, group in sorted(stats.groups.items())]
    lines += [
        "",
        "分類：",
        f"  總是亮著 always_on   {pct(verdict['counts']['always_on'])}",
        f"  從不亮起 always_off  {pct(verdict['counts']['always_off'])}",
        f"  幾乎不隨輸入變動 constant"
        f"（CV<{verdict['cv_threshold']}，|mean|≥{verdict['magnitude_floor']:.4g}）"
        f"  {pct(verdict['counts']['constant'])}",
        "",
    ]

    # 逐層的 always_on 比例：如果集中在某幾層，之後裁減要分層處理
    always_on = verdict["always_on"]
    lines.append("每層 always_on 數量：")
    per_layer = always_on.sum(dim=1).tolist()
    for i in range(0, stats.n_layers, 7):
        chunk = per_layer[i:i + 7]
        lines.append("  " + "  ".join(f"L{i + j:<2}={int(v):>5}" for j, v in enumerate(chunk)))
    lines.append("")

    # 激活量級，供挑選門檻參考
    mean_abs = st.mean.abs().flatten()
    quantiles = torch.tensor([0.5, 0.9, 0.99, 0.999, 1.0])
    values = torch.quantile(mean_abs.float(), quantiles)
    lines.append("|平均激活| 分位數："
                 + "  ".join(f"p{int(q * 1000) / 10:g}={v:.4g}"
                             for q, v in zip(quantiles.tolist(), values.tolist())))

    if pair:
        pos_name, neg_name = pair
        lines += ["", "─" * 64,
                  f"完全分離的神經元（{pos_name} vs {neg_name}）", "─" * 64]
        for positive, negative, label in (
            (pos_name, neg_name, f"{pos_name} 組總是亮、{neg_name} 組總是不亮"),
            (neg_name, pos_name, f"（反向對照）{neg_name} 組總是亮、{pos_name} 組總是不亮"),
        ):
            lines.append(f"\n{label}：")
            for result in stats.selective_sweep(positive, negative):
                lines.append(
                    f"  亮≥{result['on_rate']:<5.0%} 且 不亮≤{result['off_rate']:<5.0%}"
                    f"  {result['count']:>7,} 個  ({result['count'] / total:7.3%})"
                )
            strict = stats.selective(positive, negative)
            if strict["count"]:
                idx = torch.nonzero(strict["mask"], as_tuple=False)
                lines.append("  100% 分離的位置："
                             + "  ".join(f"L{int(l)}#{int(n)}" for l, n in idx[:12])
                             + ("  ..." if strict["count"] > 12 else ""))

        score = stats.group_diff(pos_name, neg_name).flatten()
        top = score.topk(min(10, score.numel()))
        lines += [
            "",
            f"（參考）{pos_name} − {neg_name} 標準化平均差異最大的 10 個：",
            "  " + "  ".join(
                f"L{int(idx) // stats.n_neurons}#{int(idx) % stats.n_neurons}={val:.2f}"
                for val, idx in zip(top.values.tolist(), top.indices.tolist())
            ),
        ]

    lines.append("=" * 64)
    return "\n".join(lines)


def save_json(path: Path, stats: NeuronStats, verdict: dict,
              pair: tuple[str, str] | None = None) -> None:
    """只存摘要與分類計數；完整張量留在記憶體供呼叫端使用。"""
    payload = {
        "n_layers":      stats.n_layers,
        "n_neurons":     stats.n_neurons,
        "total_neurons": stats.total_neurons,
        "n_samples":     verdict["n_samples"],
        "act_threshold": verdict["act_threshold"],
        "cv_threshold":  verdict["cv_threshold"],
        "magnitude_floor": verdict["magnitude_floor"],
        "counts":        verdict["counts"],
        "group_sizes":   {name: g.n for name, g in stats.groups.items()},
        "always_on_per_layer": [int(v) for v in verdict["always_on"].sum(dim=1).tolist()],
        "always_on_index": [
            [int(layer), int(neuron)]
            for layer, neuron in zip(*torch.nonzero(verdict["always_on"], as_tuple=True))
        ],
    }

    # 完全分離的神經元：這是 prune 要拿去裁減的名單，索引要完整存下來
    if pair:
        payload["selective"] = {}
        payload["pair"] = {"positive": pair[0], "negative": pair[1]}
        for positive, negative in (pair, (pair[1], pair[0])):
            key = f"{positive}_on__{negative}_off"
            sweep = stats.selective_sweep(positive, negative)
            strict = sweep[0]
            payload["selective"][key] = {
                "sweep": [
                    {
                        "on_rate":  r["on_rate"],
                        "off_rate": r["off_rate"],
                        "count":    r["count"],
                        # 嚴格條件常常是 0 個，放寬後的索引才是實際能拿去裁減的名單，
                        # 所以每個容忍度都把位置存下來（上限 500 個，避免檔案爆開）
                        "index": [
                            [int(layer), int(neuron)]
                            for layer, neuron in zip(*torch.nonzero(r["mask"], as_tuple=True))
                        ][:500],
                    }
                    for r in sweep
                ],
                "strict_count":     strict["count"],
                "strict_per_layer": strict["per_layer"],
                "strict_index": [
                    [int(layer), int(neuron)]
                    for layer, neuron in zip(*torch.nonzero(strict["mask"], as_tuple=True))
                ],
            }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已寫出 {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="探測 MLP 神經元激活（獨立工具）")
    parser.add_argument("--seeds", nargs="*", default=None,
                        help="refusal_seeds*.jsonl（可多個）")
    parser.add_argument("--prompts_file", default=None, help="純文字提示詞檔，一行一題")
    parser.add_argument("--fields", nargs="*", default=["harmful", "harmless"],
                        help="種子檔中要跑的欄位，即分組名稱")
    parser.add_argument("--model", default=None,
                        help="registry 基礎模型 id 或 experiments 的 run_id（走 model_loader）")
    parser.add_argument("--checkpoint", default="final")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME,
                        help="未指定 --model 時直接載入的 HuggingFace 模型名稱")
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--last_tokens", type=int, default=5)
    parser.add_argument("--act_threshold", type=float, default=0.0)
    parser.add_argument("--cv_threshold", type=float, default=0.05)
    parser.add_argument("--groups", nargs=2, default=None, metavar=("POSITIVE", "NEGATIVE"),
                        help="指定哪兩組做分離分析（預設自動偵測 harmful/harmless "
                             "或 refused/complied）")
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 筆（冒煙測試）")
    parser.add_argument("--out", default=None, help="寫出 JSON 摘要的路徑")
    args = parser.parse_args()

    if not args.seeds and not args.prompts_file:
        parser.error("請用 --seeds 或 --prompts_file 指定提示詞來源")

    records: list[dict] = []
    if args.seeds:
        records += load_seeds(args.seeds, tuple(args.fields))
    if args.prompts_file:
        records += load_prompts_file(args.prompts_file)
    if args.limit:
        records = records[:args.limit]
    print(f"提示詞共 {len(records)} 筆")

    model, tokenizer = load_target_model(args.model, args.checkpoint, args.model_name)

    with ActivationProbe(model, tokenizer, system=args.system,
                         last_tokens=args.last_tokens,
                         act_threshold=args.act_threshold) as probe:
        stats = probe.collect(records)

    verdict = stats.classify(cv_threshold=args.cv_threshold)
    pair = resolve_pair(stats, args.groups)
    if pair is None:
        print(f"警告：分組數為 {len(stats.groups)}，無法做分離分析，只輸出全體統計")
    print()
    print(format_report(stats, verdict, pair))

    if args.out:
        save_json(Path(args.out), stats, verdict, pair)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

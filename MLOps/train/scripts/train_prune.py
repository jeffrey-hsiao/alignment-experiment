"""
MLOps/train/scripts/train_prune.py

精確神經元裁減（v1）：把 neuron_probe.py 找出的候選拒絕神經元歸零，量測拒絕
率的變化，檢驗「裁減拒絕神經元能否達成越獄」。

不是梯度訓練。PruneRunner 只滿足 train(resume_from_checkpoint)/remove_callback
介面，讓 RunSummary/Metrics/GenerationTest/顯示模式照常運作——同樣的模式見
train_empty.py 的 FakeTrainer 與 train_selfplay.py 的 SelfPlayRunner。

流程固定兩步，所以 baseline 與裁減後的數字在同一次 run 內可比：
  step 1  baseline：原模型的有害/無害拒絕率
  step 2  pruned：  裁減後的有害/無害拒絕率
裁減是直接改權重且不可逆，所以必須先量 baseline 再裁減。

checkpoint 只存 pruned_neurons.json（見 prune_utils），與 DPO/SFT 只存 LoRA
adapter 同一個原則：存相對於 base model 的差異，不存 3GB 完整權重。
model_loader 認得這個格式，所以 run.py chat <run_id> <checkpoint> 可以直接用。

使用方式（經 run.py）：
  python run.py train prune --config v1
  python run.py train prune --config v1_control    # 反向對照組
  python run.py train prune --config km_v1         # 自動掃描 K、M（見 prune_search.py）
  python run.py train prune --resume <run_id>      # 接續中斷的掃描（通用旗標）
"""

import argparse
import json
import sys
import types
from datetime import datetime
from pathlib import Path

import torch
from transformers import ProgressCallback

from base_config import PruneConfig
from base_trainer import BaseTrainer, _read_summary, _write_summary

ROOT = Path(__file__).parents[3]      # repo 根目錄（含 MLOps/ 與 pipelines/）
MLOPS_DIR = Path(__file__).parents[2]
sys.path.insert(0, str(MLOPS_DIR))

from neuron_probe import load_seeds                       # noqa: E402
from chat_prompt import render_chat                       # noqa: E402
from refusal_judge import load_judge                      # noqa: E402
from prune_utils import (                                 # noqa: E402
    apply_prune_mask,
    load_probe_neurons,
    parse_neuron_spec,
    write_prune_checkpoint,
)


def _resolve(path_str: str) -> Path:
    """設定檔裡的相對路徑一律相對於 repo 根目錄，避免受啟動目錄影響。"""
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


class PruneRunner:
    """baseline → 裁減 兩步，每步量測拒絕率並觸發完整 callback 生命週期。"""

    def __init__(self, trainer_args, model, tokenizer, pairs, neurons, judge,
                 callbacks, output_dir: Path, trainer: "PruneTrainer"):
        self.args       = trainer_args
        self.judge      = judge
        self.model      = model
        self.tokenizer  = tokenizer
        self.pairs      = pairs       # [{group, text, id, domain, ...}]
        self.neurons    = neurons     # [(layer, idx)]
        self.callbacks  = list(callbacks)
        self.output_dir = output_dir
        self.trainer    = trainer

    def remove_callback(self, cls):
        self.callbacks = [c for c in self.callbacks if not isinstance(c, cls)]

    # ── 評估 ──────────────────────────────────────────────────────────────────

    def _generate(self, prompt: str) -> dict:
        """生成單一回應。回傳 text / n_tokens / truncated。

        truncated 必須記下來：被 max_new_tokens 砍斷的回應，其長度量到的是上限
        而不是模型行為，不能拿去算能力指標；「未被拒」這個標籤在截斷的回應上
        也不可靠（可能先配合、後拒絕）。
        """
        text = render_chat(self.tokenizer, prompt,
                           system=self.trainer.objective.SYSTEM_TARGET,
                           add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        limit  = self.args.gen_max_tokens
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=limit,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        # 自然結束的話尾端會有 eos；長度貼齊上限且沒有 eos 就是被砍斷
        has_eos = bool((new_tokens == self.tokenizer.eos_token_id).any())
        return {
            "text":      self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
            "n_tokens":  int(new_tokens.numel()),
            "truncated": (not has_eos) and int(new_tokens.numel()) >= limit,
        }

    def evaluate(self, phase: str) -> tuple[dict, list[dict]]:
        """對每一筆提示詞生成一次，回傳 (metrics, 逐筆明細)。"""
        details: list[dict] = []

        for i, record in enumerate(self.pairs, 1):
            result = self._generate(record["text"])
            details.append({**record, "response": result["text"],
                            "n_tokens": result["n_tokens"], "truncated": result["truncated"]})
            if i % 20 == 0 or i == len(self.pairs):
                print(f"  [{phase}] 已生成 {i}/{len(self.pairs)} 筆", flush=True)

        # 生成與判定分開：先全部生成完再一次判定，判定器才能批次推論，
        # 而且回應存檔後可以用 refusal_judge.py 換判定器重算，不必重新生成。
        verdicts = self.judge.judge_batch([d["response"] for d in details])
        for detail, verdict in zip(details, verdicts):
            detail["refused"]       = verdict["refused"]
            detail["refusal_score"] = verdict["score"]
            detail["category"]      = verdict.get("category")
            detail["phase"]         = phase
        self._save_responses(phase, details)

        metrics = {"phase": phase, "judge": self.judge.name,
                   "n_pruned": len(self.neurons) if phase == "pruned" else 0}
        for group in sorted({d["group"] for d in details}):
            rows = [d for d in details if d["group"] == group]
            complete = [d for d in rows if not d["truncated"]]
            metrics[f"{group}_refusal_rate"]   = round(sum(d["refused"] for d in rows) / len(rows), 4)
            metrics[f"{group}_n"]              = len(rows)
            metrics[f"{group}_truncated_rate"] = round(sum(d["truncated"] for d in rows) / len(rows), 4)
            # refusal / disclaimer 分開記：從明確拒絕退化成只剩免責聲明，本身就是
            # 有意義的中間結果，合併成一個拒絕率會看不見
            for category in ("refusal", "disclaimer", "comply"):
                hits = sum(1 for d in rows if d.get("category") == category)
                if hits or any(d.get("category") for d in rows):
                    metrics[f"{group}_{category}_rate"] = round(hits / len(rows), 4)
            # 只統計自然結束的回應：截斷的長度量到的是 max_new_tokens，不是模型
            metrics[f"{group}_mean_len"] = (
                round(sum(len(d["response"]) for d in complete) / len(complete), 1)
                if complete else None
            )
        return metrics, details

    def _save_responses(self, phase: str, details: list[dict]) -> Path:
        """把本步所有回應寫進 run 目錄，供事後換判定器重算或人工覆核。"""
        path = self.output_dir.parent / f"responses_{phase}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in details) + "\n",
            encoding="utf-8")
        print(f"  [{phase}] 回應已存入 {path.name}（{len(details)} 筆）")
        return path

    # ── 主流程 ────────────────────────────────────────────────────────────────

    def train(self, resume_from_checkpoint=False):
        from transformers import TrainerControl, TrainerState

        if resume_from_checkpoint:
            print("（prune 模式沒有可接續的狀態，--resume 已忽略）")

        state   = TrainerState()
        control = TrainerControl()
        state.max_steps = 2
        fake_args = types.SimpleNamespace(output_dir=str(self.output_dir))

        for cb in self.callbacks:
            cb.on_train_begin(fake_args, state, control)

        baseline: dict | None = None
        for step, phase in enumerate(("baseline", "pruned"), 1):
            if phase == "pruned":
                apply_prune_mask(self.model, self.neurons)

            state.global_step = step
            self.model.eval()
            metrics, details = self.evaluate(phase)

            if baseline is None:
                baseline = metrics
            else:
                for group in ("harmful", "harmless"):
                    key = f"{group}_refusal_rate"
                    if key in metrics and key in baseline:
                        metrics[f"{group}_refusal_delta"] = round(metrics[key] - baseline[key], 4)

            self.trainer.last_details = details
            self.trainer.last_metrics = metrics

            for cb in self.callbacks:
                cb.on_step_end(fake_args, state, control)
            for cb in self.callbacks:
                cb.on_log(fake_args, state, control, logs=metrics)
            for cb in self.callbacks:
                cb.on_evaluate(fake_args, state, control, model=self.model, logs=metrics)
            self.model.eval()   # GenerationTestCallback 會把模型切回 train()

            ckpt_dir = self.output_dir / f"checkpoint-{step}"
            write_prune_checkpoint(
                ckpt_dir,
                base_model=self.trainer.cfg["model_name"],
                neurons=self.neurons if phase == "pruned" else [],
                phase=phase, metrics=metrics,
            )
            self.tokenizer.save_pretrained(str(ckpt_dir))
            for cb in self.callbacks:
                cb.on_save(fake_args, state, control)

        for cb in self.callbacks:
            cb.on_train_end(fake_args, state, control)

        self._print_summary(baseline, self.trainer.last_metrics)

    def _print_summary(self, baseline: dict, pruned: dict) -> None:
        lines = ["", "=" * 60, f"裁減結果（{len(self.neurons)} 個神經元）", "=" * 60]
        for group in ("harmful", "harmless"):
            key = f"{group}_refusal_rate"
            if key not in baseline:
                continue
            label = "有害題" if group == "harmful" else "無害題"
            lines.append(f"  {label}拒絕率  {baseline[key]:.1%} → {pruned[key]:.1%}"
                         f"  （{pruned.get(f'{group}_refusal_delta', 0):+.1%}）")
        lines.append("=" * 60)
        print("\n".join(lines))


class PruneTrainer(BaseTrainer):
    CONFIG_CLASS = PruneConfig

    def __init__(self, args):
        super().__init__(args)
        self.last_details: list[dict] = []
        self.last_metrics: dict = {}

    # ── 不訓練：沿用 BaseTrainer 的模型載入，但不加 LoRA ──────────────────────

    def apply_lora(self):
        print("（Prune 模式：直接修改 down_proj 權重，不使用 LoRA）")

    def setup_model(self):
        super().setup_model()
        # 生成需要 KV cache；BaseTrainer 為了訓練把它關掉了
        self.model.config.use_cache = True

    def load_datasets(self):
        return None, None

    def format_record(self, record: dict) -> dict:
        return record  # 未使用

    def make_trainer_config(self):
        return types.SimpleNamespace(
            gen_max_tokens = self.cfg["gen_max_tokens"],
            eval_n         = self.cfg["eval_n"],
        )

    # ── 提示詞與神經元清單 ────────────────────────────────────────────────────

    def _load_pairs(self) -> list[dict]:
        seeds = _resolve(self.cfg["seeds"])
        records = load_seeds([seeds])
        eval_n = self.cfg["eval_n"]
        if eval_n and eval_n < len(records):
            # 成對取樣：load_seeds 依序產出同一列的 harmful 與 harmless，
            # 直接截斷會讓兩組筆數不等，所以按列數取前 eval_n/2 對。
            by_id: dict[str, list[dict]] = {}
            for record in records:
                by_id.setdefault(record["id"], []).append(record)
            keep = list(by_id)[: max(1, eval_n // 2)]
            records = [r for key in keep for r in by_id[key]]
        print(f"評估提示詞 {len(records)} 筆（來源 {seeds.name}）")
        return records

    def _load_neurons(self) -> list[tuple[int, int]]:
        spec = str(self.cfg.get("neurons", "") or "")
        if spec:
            neurons = parse_neuron_spec(spec)
            print(f"設定檔指定 {len(neurons)} 個神經元："
                  + "  ".join(f"L{l}#{i}" for l, i in neurons))
            return neurons

        probe_json = str(self.cfg.get("probe_json", "") or "")
        if probe_json:
            return load_probe_neurons(_resolve(probe_json), key=self.cfg["probe_key"])

        raise SystemExit("設定檔必須指定 neurons 或 probe_json")

    def make_trainer(self, train_ds, eval_ds, callbacks) -> PruneRunner:
        return PruneRunner(
            trainer_args = self.make_trainer_config(),
            model        = self.model,
            tokenizer    = self.tokenizer,
            pairs        = self._load_pairs(),
            neurons      = self._load_neurons(),
            judge        = load_judge(
                str(self.cfg.get("judge", "keyword")),
                **({} if str(self.cfg.get("judge", "keyword")) == "keyword"
                   else {"device": str(self.cfg.get("judge_device", "cpu"))}),
            ),
            callbacks    = callbacks,
            output_dir   = self.output_dir,
            trainer      = self,
        )

    # ── 生成測試：列出本步實際被拒與未被拒的例子，供人工覆核關鍵字判定 ───────

    def run_generation_test(self, model, step: int) -> str:
        metrics = self.last_metrics
        redact = bool(self.cfg.get("redact_content", False))
        if "km_kind" in metrics:
            from prune_search import format_generation_block
            return format_generation_block(metrics, self.last_details, self.km_runner._auto_collapse,
                                           redact=redact)
        phase   = metrics.get("phase", f"step-{step}")
        lines = ["=" * 60, f"[Generation Test] step={step}  phase={phase}", "=" * 60]
        for group in ("harmful", "harmless"):
            key = f"{group}_refusal_rate"
            if key in metrics:
                mean_len = metrics.get(f"{group}_mean_len")
                lines.append(f"{group:<9} 拒絕率 {metrics[key]:.1%}"
                             f"  （n={metrics[f'{group}_n']}"
                             f"，截斷率 {metrics[f'{group}_truncated_rate']:.0%}"
                             f"，完整回應平均長度 {mean_len if mean_len is not None else '—'}）")
        lines.append("")

        for group in ("harmful", "harmless"):
            for want_refused in (True, False):
                picked = [d for d in self.last_details
                          if d["group"] == group and d["refused"] is want_refused][:2]
                if not picked:
                    continue
                label = "被拒" if want_refused else "未被拒"
                lines.append(f"── {group} / {label} ──")
                if redact:
                    lines.append(f"  （{len(picked)} 筆，原文已遮蔽）")
                else:
                    for detail in picked:
                        flag = "（截斷）" if detail["truncated"] else ""
                        lines.append(f"  [{detail['id']} {detail['domain']}] {detail['text'][:60]}")
                        lines.append(f"    → {detail['response'][:150]}  "
                                     f"[{detail['n_tokens']} tokens{flag}]")
                lines.append("")

        return "\n".join(lines)

    # ── 覆寫 run()：無梯度流程，checkpoint 只存神經元清單 ─────────────────────

    def _run_search(self, runner_cls, error_cls):
        """K、M 掃描（km_search / km_cos 共用 glue）：runner 自己管理模型（每格重載），
        所以這裡不預先載入。"""
        from base_display import _load_display_mode

        self.output_dir.mkdir(parents=True, exist_ok=True)
        display_mode = _load_display_mode(_read_summary(self.summary_path))
        self.km_runner = runner_cls(self, self.build_callbacks(display_mode), _resolve)
        if display_mode is not None:
            self.km_runner.remove_callback(ProgressCallback)

        def finish(status: str, **extra):
            summary = _read_summary(self.summary_path)
            summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
            summary["status"] = status
            summary.update(extra)
            _write_summary(self.summary_path, summary)

        try:
            self.km_runner.train(resume_from_checkpoint=self.args.resume)
        except error_cls as e:
            finish("failed", failure=str(e))
            print(f"\n停止：{e}")
            sys.exit(1)
        except KeyboardInterrupt:
            finish("interrupted")
            print(f"\n已中斷，完成的格子都已存檔。接續：python run.py train "
                  f"{self.CONFIG_CLASS.METHOD} --resume {self.output_dir.parent.name}")
            sys.exit(0)

    def run(self):
        mode = self.cfg.get("mode", "fixed")
        if mode == "km_search":
            from prune_search import KmSearchError, KmSearchRunner
            return self._run_search(KmSearchRunner, KmSearchError)
        if mode == "km_cos":
            from prune_km_cos import KmCosError, KmCosRunner
            return self._run_search(KmCosRunner, KmCosError)

        self.setup_tokenizer()
        train_ds, eval_ds = self.load_datasets()
        self.setup_model()
        self.apply_lora()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        context = _read_summary(self.summary_path)
        from base_display import _load_display_mode
        display_mode = _load_display_mode(context)
        callbacks    = self.build_callbacks(display_mode)
        trainer      = self.make_trainer(train_ds, eval_ds, callbacks)
        if display_mode is not None:
            trainer.remove_callback(ProgressCallback)

        try:
            trainer.train(resume_from_checkpoint=self.args.resume)
            final_dir = self.output_dir / "final"
            write_prune_checkpoint(
                final_dir,
                base_model=self.cfg["model_name"],
                neurons=trainer.neurons,
                phase="final", metrics=self.last_metrics,
            )
            self.tokenizer.save_pretrained(str(final_dir))
            print(f"\n完成。裁減清單儲存於 {final_dir}")
        except KeyboardInterrupt:
            print("\n偵測到中斷，寫出緊急紀錄...")
            emergency_dir = self.output_dir / "emergency"
            write_prune_checkpoint(
                emergency_dir,
                base_model=self.cfg["model_name"],
                neurons=trainer.neurons,
                phase="emergency", metrics=self.last_metrics,
            )
            summary = _read_summary(self.summary_path)
            summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
            summary["status"]      = "interrupted"
            summary.setdefault("checkpoints", []).append("emergency")
            _write_summary(self.summary_path, summary)
            print(f"緊急紀錄完成：{emergency_dir}")
            sys.exit(0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    BaseTrainer.base_args(parser)
    PruneTrainer(parser.parse_args()).run()

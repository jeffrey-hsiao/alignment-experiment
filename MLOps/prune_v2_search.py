"""
MLOps/prune_v2_search.py

prune_v2：gen-2 搜尋驅動器。與 km_search 共用整套引擎（_prepare / _evaluate /
_select / _top_cells / final checkpoint / 報表 / 顯示），只換掉「粗測 + 細搜」這段
搜尋策略，完全不修改 prune_search.py（以子類覆寫 train()）。

與 v1 的差異：
  - 不粗測，直接開始。
  - 不以「M 撞 end_m」當停止條件；改用「直接操作 K」的單調掃描（不追蹤浮點 r，避免
    r 過小消失；隱性比例 = K/T）：
      T = K + M，起始 = 總神經元數 / sweep_steps；K 起始 = round(k_ratio_start × T)。
    每格評估後：
      翻轉率 < flip_low → 放大 T 一步，**K 等比例放大**（nK = K×newT/T）以維持隱性比例；
      否則              → **K ×= k_decay**（整數衰減，降隱性比例，T 不變）；
                          若再乘也無法讓 K 更小（K 已到底）→ 改放大 T。
    K 恆為整數 ≥1（永不為 0、無比例下限）；T 每遇到「放大」必 +step，以總神經元數為上界
    → 必終止。
  - 崩壞（is_collapsed：repeat/relevance/gibberish）照算照記錄進每格，但不參與步進，
    品質由人工覆核。

resume 靠確定性重放：(T, K) 軌跡由各格的翻轉率決定，而翻轉率存在快取的 cell 裡，
_evaluate 以 (kind,k,m,repeat) 為鍵直接命中已完成格，不重新生成，續跑接得回去。
"""
import json

from prune_search import KmSearchRunner, KmSearchError
from prune_utils import write_prune_checkpoint


class PruneV2Runner(KmSearchRunner):
    """與 KmSearchRunner 同介面，只覆寫 train() 的搜尋迴圈。"""

    def train(self, resume_from_checkpoint=False):
        import types

        from transformers import TrainerControl, TrainerState

        if self.state_path.exists():
            if not resume_from_checkpoint:
                raise KmSearchError(f"{self.exp_dir.name} 已有結果，要接續請用："
                                    f"python run.py train prune_v2 --resume {self.exp_dir.name}")
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            print(f"接續：已完成 {len(self.state['cells'])} 格", flush=True)

        state, control = TrainerState(), TrainerControl()
        self.tracker = (state, control)
        fake_args = types.SimpleNamespace(output_dir=str(self.output_dir))
        for cb in self.callbacks:
            cb.on_train_begin(fake_args, state, control)

        self._prepare()

        total = int(self.ranking["score"].numel())
        sweep_steps = max(1.0, float(self.cfg["sweep_steps"]))
        step = max(1, round(total / sweep_steps))
        T = int(self.cfg.get("t_start") or 0) or step
        decay = float(self.cfg["k_decay"])
        flip_low = float(self.cfg["flip_low"])
        K = max(1, min(T - 1, round(float(self.cfg["k_ratio_start"]) * T)))   # 起始 K

        self.state["v2"] = {"total": total, "step": step, "sweep_steps": sweep_steps,
                            "k_ratio_start": float(self.cfg["k_ratio_start"]),
                            "k_decay": decay, "flip_low": flip_low}
        # bounds 欄位沿用給報表/顯示；v2 沒有粗測界，填掃描範圍
        self.state["bounds"] = {"source": "prune_v2", "start_k": min(T, total), "end_m": total}
        self._save_state()
        print(f"prune_v2 掃描：總神經元 {total}，步長 {step}（1/{sweep_steps:g}），"
              f"起始 T={T} K={K}（隱性比例 {K / T:.2f}），decay={decay}，flip_low={flip_low}", flush=True)

        def grow_T():
            """放大 T 一步，K 等比例放大以維持隱性比例 K/T（直接操作 K，無浮點 r）。"""
            nonlocal T, K
            nT = T + step
            K = max(1, round(K * nT / T))
            T = nT

        while T <= total:
            K = max(1, min(T - 1, K))
            cell = self._evaluate("sync", K, T - K)   # 內建快取 → resume 直接命中已完成格
            flip_rate = cell["flips"] / max(1, cell["n"])
            if flip_rate < flip_low:
                grow_T()                    # 翻轉過低 → 放大 T（K 等比放大，維持比例）
            else:
                nK = max(1, round(K * decay))
                if nK < K:
                    K = nK                  # 衰減 K（降隱性比例），T 不變；K 恆為整數 ≥1
                else:
                    grow_T()                # K 已到底（再乘也不變小）→ 改放大 T
            self._save_state()

        top = self._top_cells()
        self.state["top"] = [{"k": c["k"], "m": c["m"]} for c in top]
        self.state["finished"] = True
        self._save_state()

        if top:
            zeroed, reversed_neurons, _, _, _ = self._select("sync", top[0]["k"], top[0]["m"], 0)
            write_prune_checkpoint(self.output_dir / "final", self.cfg["model_name"],
                                   zeroed, reversed_neurons=reversed_neurons,
                                   phase="final", k=top[0]["k"], m=top[0]["m"])

        try:
            self._write_report()
        except Exception as e:
            print(f"略過圖表（v2 狀態與報表不相容）：{e}")

        for cb in self.callbacks:
            cb.on_train_end(fake_args, state, control)
        self._print_summary()

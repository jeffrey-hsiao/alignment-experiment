"""
MLOps/prune_v2_search.py

prune_v2：gen-2 搜尋驅動器。與 km_search 共用整套引擎（_prepare / _evaluate /
_select / _top_cells / final checkpoint / 報表 / 顯示），只換掉「粗測 + 細搜」這段
搜尋策略：以子類覆寫 train()，把掃描迴圈委派給 prune_search.run_sweep（同一份實作
也給 km_cos 用，不再各留一份重複迴圈）。

與 v1 的差異：
  - 不粗測，直接開始。
  - 搜尋上界可設：t_max = t_max_ratio × 總神經元（預設 1.0 = 整個模型）。**步長基於上界切**：
    step = t_max / sweep_steps。過往最佳格都在 ≤20% T，設 t_max_ratio=0.3~0.4 可把 sweep_steps
    的解析度集中在低 T 區、不浪費在高 T。
  - 不以「M 撞 end_m」當停止條件；改用「直接操作 K」的單調掃描（不追蹤浮點 r，避免
    r 過小消失；隱性比例 = K/T）：
      T = K + M，起始 = step；K 起始 = round(k_ratio_start × T)。
    每格評估後：
      翻轉率 < flip_low → 放大 T 一步，**K 等比例放大**（nK = K×newT/T）以維持隱性比例；
      否則              → **K ×= k_decay**（整數衰減，降隱性比例，T 不變）；
                          若再乘也無法讓 K 更小（K 已到底）→ 改放大 T。
    K 恆為整數 ≥1（永不為 0、無比例下限）；T 每遇到「放大」必 +step，以 t_max 為上界 → 必終止。
  - 崩壞（is_collapsed：repeat/relevance/gibberish）照算照記錄進每格，但不參與步進，
    品質由人工覆核。

resume 靠確定性重放：(T, K) 軌跡由各格的翻轉率決定，而翻轉率存在快取的 cell 裡，
_evaluate 以 (kind,k,m,repeat) 為鍵直接命中已完成格，不重新生成，續跑接得回去。
"""
import json

from prune_search import KmSearchRunner, KmSearchError, run_sweep, ThresholdCriterion
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
        # gen-2 單調 T/K 掃描：委派給共用的 run_sweep（同一份實作也給 km_cos 用）。
        # 跳格判準＝拒絕翻轉率 flip_rate = flips/n（比例，門檻 flip_low）。
        total = int(self.ranking["score"].numel())
        run_sweep(self._evaluate, total, self.cfg, self.state, self._save_state,
                  criterion=ThresholdCriterion("flips", float(self.cfg["flip_low"]), as_rate=True),
                  label="prune_v2")

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

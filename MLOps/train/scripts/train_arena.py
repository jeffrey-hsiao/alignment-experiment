"""
MLOps/train/scripts/train_arena.py

多智能體競技場訓練：MCCFR/MCT 傳統演算法 agent（不是 LLM）互打，用既有的
MMR/ELO 配對系統訓練整個 roster。訓練邏輯完全複用
train/objectives/card_game_trainer/arena.py 的 MatchmakingArena（從博弈論
專案抽出），本檔案只負責：
  1. 用 BaseTrainer 的 callback/checkpoint 骨架把它接上 run.py（同
     train_selfplay.py 的 FakeTrainer 模式，不重新實作訓練邏輯）
  2. 把 arena.py 自己的 ANSI 即時刷新（quiet=True）換成這個 repo 的
     RunSummaryCallback/MetricsCallback/顯示模式（on_harvest hook）
  3. 把 ELO/矩陣 JSON 的存放位置從 card_game_trainer/（原始碼目錄）搬到
     experiments/_arena/（跟 experiments/ 其他生成資料一樣被 gitignore）

跟 selfplay 一樣屬於卡牌博弈訓練目標家族（objective=boyi），差別是這裡
訓練的是傳統 MCCFR/MCT agent，不是 LLM，因此完全不需要 HF/tokenizer/LoRA。

全域狀態（跨多次 `run.py train arena` 執行持續累積，不是每個 run_id 各自
一份）：
  - experiments/_arena/arena_elo.json / arena_matrix.json   ELO／勝負矩陣
  - experiments/_arena/agent_saves/*.pkl                    各 agent 權重
每個 run_id 自己的 run_summary.json/metrics.jsonl 仍照常規記錄「這次訓練
做了什麼」，checkpoint-N/ 底下另外存一份當時的 ELO/矩陣快照作為歷史記錄，
但不是真正的存檔來源——真正的存檔來源永遠是 experiments/_arena/。

「凍結參照對手」：roster 裡列在 frozen_agents 的成員這一輪不訓練、不存檔、
ELO 不更新（矩陣勝負紀錄仍照記），用來放一個已知強度的固定基準，見
card_game_trainer/arena.py 的 frozen_names 參數。

已知限制：目前只接了 train/eval 兩種模式，arena.py 的 quick/filling/balance/
cross 尚未串接。
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

from transformers import ProgressCallback

from base_config import ArenaConfig
from base_trainer import BaseTrainer, _read_summary, _write_summary

CARD_GAME_DIR = Path(__file__).parent.parent / "objectives" / "card_game_trainer"
sys.path.insert(0, str(CARD_GAME_DIR))

import arena  # noqa: E402

EXPERIMENTS_DIR    = Path(__file__).parent.parent.parent / "experiments"
ARENA_STATE_DIR    = EXPERIMENTS_DIR / "_arena"
ARENA_SAVES_DIR    = ARENA_STATE_DIR / "agent_saves"
ARENA_ELO_PATH     = ARENA_STATE_DIR / "arena_elo.json"
ARENA_MATRIX_PATH  = ARENA_STATE_DIR / "arena_matrix.json"


class ArenaRunner:
    """模擬 HuggingFace Trainer 的 callback 生命週期；實際訓練（regret table／
    DQN 更新）發生在 arena.MatchmakingArena 呼叫的 agent.choose_action()/
    notify_result() 內部，這裡只負責驅動它並在每局結束時同步 callback。"""

    def __init__(self, trainer_args, roster, frozen_names, elo, matrix, callbacks, output_dir: Path):
        self.args         = trainer_args
        self.roster       = roster
        self.frozen_names = frozen_names
        self.elo          = elo
        self.matrix       = matrix
        self.callbacks    = list(callbacks)
        self.output_dir   = output_dir

    def remove_callback(self, cls):
        self.callbacks = [c for c in self.callbacks if not isinstance(c, cls)]

    def _snapshot(self, tag: str) -> None:
        """把當下 ELO/矩陣寫一份進這個 run 的 checkpoint 目錄，純歷史記錄用。"""
        ckpt_dir = self.output_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "arena_elo.json").write_text(
            json.dumps(self.elo, ensure_ascii=False, indent=2), encoding="utf-8")
        (ckpt_dir / "arena_matrix.json").write_text(
            json.dumps(self.matrix, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_global(self) -> None:
        ARENA_STATE_DIR.mkdir(parents=True, exist_ok=True)
        arena._dump(str(ARENA_ELO_PATH), self.elo)
        arena._dump(str(ARENA_MATRIX_PATH), self.matrix)

    def train(self, resume_from_checkpoint=False):
        from transformers import TrainerState, TrainerControl

        args    = self.args
        state   = TrainerState()
        control = TrainerControl()
        state.max_steps = args.n_games
        fake_args = types.SimpleNamespace(output_dir=str(self.output_dir))

        for cb in self.callbacks:
            cb.on_train_begin(fake_args, state, control)

        def _on_harvest(na, nb, winner, frozen_sides, done, total):
            state.global_step = done
            for cb in self.callbacks:
                cb.on_step_end(fake_args, state, control)

            logs = {
                "last_match": f"{na} vs {nb} -> {winner or '平局'}",
                f"elo[{na}]": round(self.elo.get(na, arena.DEFAULT_ELO), 1),
                f"elo[{nb}]": round(self.elo.get(nb, arena.DEFAULT_ELO), 1),
            }
            for cb in self.callbacks:
                cb.on_log(fake_args, state, control, logs=logs)

            if args.eval_steps > 0 and done % args.eval_steps == 0:
                for cb in self.callbacks:
                    cb.on_evaluate(fake_args, state, control, model=None, logs=logs)

            if args.save_steps > 0 and done % args.save_steps == 0:
                self._snapshot(f"checkpoint-{done}")
                self._save_global()
                for cb in self.callbacks:
                    cb.on_save(fake_args, state, control)

        mm_arena = arena.MatchmakingArena(
            self.roster, self.elo, self.matrix, args.n_games,
            mode=args.mode, cpu_target=args.cpu_target,
            frozen_names=self.frozen_names, quiet=True, on_harvest=_on_harvest,
        )
        mm_arena.run()

        self._save_global()
        for cb in self.callbacks:
            cb.on_train_end(fake_args, state, control)

        ranked = sorted(self.roster, key=lambda n: self.elo.get(n, arena.DEFAULT_ELO), reverse=True)
        lines = [f"\nArena 完成（{args.n_games} 局，mode={args.mode}）：",
                 *(f"  {name:<20} ELO={self.elo.get(name, arena.DEFAULT_ELO):.0f}" for name in ranked)]
        print("\n".join(lines))


class ArenaTrainer(BaseTrainer):
    CONFIG_CLASS = ArenaConfig

    # ── 不套用 HF 模型初始化：純傳統演算法 agent，不需要 tokenizer/LoRA ────

    def setup_tokenizer(self):
        pass

    def setup_model(self):
        self.model = None

    def apply_lora(self):
        print("（Arena 模式：傳統 MCCFR/MCT agent，不使用 LoRA/HF 模型）")

    # ── 不用固定資料集：訓練經驗來自實際對局 ────────────────────────────

    def load_datasets(self):
        return None, None

    def format_record(self, record: dict) -> dict:
        return record  # 未使用（load_datasets 已略過資料載入）

    def make_trainer_config(self):
        return types.SimpleNamespace(
            n_games    = self.cfg["n_games"],
            mode       = self.cfg["mode"],
            cpu_target = self.cfg["cpu_target"],
            save_steps = self.cfg["save_steps"],
            eval_steps = self.cfg["eval_steps"],
        )

    def make_trainer(self, train_ds, eval_ds, callbacks) -> ArenaRunner:
        roster = [n.strip() for n in self.cfg["roster"].split(",") if n.strip()]
        frozen = {n.strip() for n in self.cfg.get("frozen_agents", "").split(",") if n.strip()}

        ARENA_SAVES_DIR.mkdir(parents=True, exist_ok=True)
        # game_worker.py 的 .pkl 存檔路徑是 cwd-relative（沿用博弈論原本的
        # SAVE_FILES 相對路徑設計，未改動），chdir 到固定的全域存檔目錄，
        # 讓 roster 的訓練狀態跨多次 run 累積在同一個地方。
        os.chdir(ARENA_SAVES_DIR)

        elo    = arena._load(str(ARENA_ELO_PATH), {})
        matrix = arena._load(str(ARENA_MATRIX_PATH), {})

        self._runner = ArenaRunner(
            trainer_args = self.make_trainer_config(),
            roster       = roster,
            frozen_names = frozen,
            elo          = elo,
            matrix       = matrix,
            callbacks    = callbacks,
            output_dir   = self.output_dir,
        )
        return self._runner

    def run_generation_test(self, model, step: int) -> str:
        ranked = sorted(self._runner.roster,
                         key=lambda n: self._runner.elo.get(n, arena.DEFAULT_ELO), reverse=True)
        lines = [f"{'='*55}", f"[Generation Test] step={step}（Arena 模式：以 ELO 排行榜取代生成測試）", f"{'='*55}"]
        lines += [f"  {name:<20} ELO={self._runner.elo.get(name, arena.DEFAULT_ELO):.0f}" for name in ranked]
        return "\n".join(lines) + "\n"

    # ── 覆寫 run()：先把所有輸出路徑轉成絕對路徑，才能安全 chdir ──────────

    def run(self):
        # make_trainer() 裡會 os.chdir() 到全域存檔目錄，之後任何用到這些
        # 路徑的地方都必須已經是絕對路徑，否則寫檔會跑到錯的地方去。
        self.output_dir         = self.output_dir.resolve()
        self.summary_path       = self.summary_path.resolve()
        self.args.metrics_path  = str(Path(self.args.metrics_path).resolve())
        self.args.gen_test_path = str(Path(self.args.gen_test_path).resolve())

        self.setup_tokenizer()
        train_ds, eval_ds = self.load_datasets()
        self.setup_model()
        self.apply_lora()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        context      = _read_summary(self.summary_path)
        from base_display import _load_display_mode
        display_mode = _load_display_mode(context)
        callbacks    = self.build_callbacks(display_mode)
        trainer      = self.make_trainer(train_ds, eval_ds, callbacks)
        if display_mode is not None:
            trainer.remove_callback(ProgressCallback)

        try:
            trainer.train(resume_from_checkpoint=self.args.resume)
            final_dir = self.output_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            (final_dir / "arena_elo.json").write_text(
                json.dumps(trainer.elo, ensure_ascii=False, indent=2), encoding="utf-8")
            (final_dir / "arena_matrix.json").write_text(
                json.dumps(trainer.matrix, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n完成。ELO/矩陣快照存於 {self.output_dir}/final"
                  f"（實際 agent 權重在 {ARENA_SAVES_DIR}，跨 run 累積）")
        except KeyboardInterrupt:
            print("\n偵測到中斷...")
            trainer._save_global()
            from datetime import datetime
            summary = _read_summary(self.summary_path)
            summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
            summary["status"]      = "interrupted"
            _write_summary(self.summary_path, summary)
            print("已將目前 ELO/矩陣寫回全域存檔。")
            sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BaseTrainer.base_args(parser)
    ArenaTrainer(parser.parse_args()).run()

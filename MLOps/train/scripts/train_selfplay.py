"""
MLOps/train/scripts/train_selfplay.py

Self-play 強化學習訓練：LLM agent 對戰 MCTAgentV2，每局結束依勝負＋HP差
計算 reward，用經驗回放（replay buffer）做 reward 加權訓練。訓練邏輯完全
複用 train/objectives/card_game_trainer/（從博弈論專案抽出的既有模組）：
LLMAgentV1/V2/V3 的 choose_action()/notify_result() 內部已經實作
_train_on_valid_action（格式修正 SFT）與 _train_on_game_end→
_batch_train_from_replay（reward 加權經驗回放），本檔案只負責用
BaseTrainer 的 callback/checkpoint/resume 骨架把它接上 run.py，不重新實作
任何訓練邏輯。

不套用 HF Trainer：make_trainer() 回傳的 SelfPlayRunner 只要滿足
train(resume_from_checkpoint)/remove_callback() 介面即可觸發完整 callback
生命週期，讓 RunSummary/Metrics/GenerationTest/顯示模式照常運作——同樣的
模式見 train_empty.py 的 FakeTrainer。

已知限制：checkpoint 存成 <ckpt>/llm_qwen_model + <ckpt>/llm_qwen_tokenizer
（LLMAgent.save() 的既有格式，非 PEFT adapter 單一目錄），model_loader.py
目前還不認得這個結構，chat/compare-chat 尚未支援 selfplay 產出的模型。
"""
import argparse
import sys
import types
from pathlib import Path

from transformers import ProgressCallback

from base_config import SelfPlayConfig
from base_trainer import BaseTrainer, _read_summary, _write_summary

CARD_GAME_DIR = Path(__file__).parent.parent / "objectives" / "card_game_trainer"
sys.path.insert(0, str(CARD_GAME_DIR))

from agents.llm_agent import LLMAgentV1     # noqa: E402
from agents.llm_agent_v2 import LLMAgentV2  # noqa: E402
from agents.llm_agent_v3 import LLMAgentV3  # noqa: E402
from agents.mct_agent_v2 import MCTAgentV2  # noqa: E402
from game_runner import run_game            # noqa: E402

_LLM_AGENT_CLASSES = {1: LLMAgentV1, 2: LLMAgentV2, 3: LLMAgentV3}


class SelfPlayRunner:
    """模擬 HuggingFace Trainer 的 callback 生命週期；實際訓練發生在
    llm_agent.choose_action()/notify_result() 內部（見 card_game_trainer）。"""

    def __init__(self, trainer_args, llm_agent, opponent, callbacks, output_dir: Path):
        self.args       = trainer_args
        self.llm_agent  = llm_agent
        self.opponent   = opponent
        self.callbacks  = list(callbacks)
        self.output_dir = output_dir

    def remove_callback(self, cls):
        self.callbacks = [c for c in self.callbacks if not isinstance(c, cls)]

    def train(self, resume_from_checkpoint=False):
        from transformers import TrainerState, TrainerControl

        args    = self.args
        state   = TrainerState()
        control = TrainerControl()
        state.max_steps = args.n_games
        fake_args = types.SimpleNamespace(output_dir=str(self.output_dir))

        start_game = 0
        if resume_from_checkpoint and self.output_dir.exists():
            ckpts = sorted(
                (p for p in self.output_dir.iterdir() if p.name.startswith("checkpoint-")),
                key=lambda p: int(p.name.split("-")[1]),
            )
            if ckpts:
                start_game = int(ckpts[-1].name.split("-")[1])
                state.global_step = start_game
                self.llm_agent.load(str(ckpts[-1] / "llm"))
                mct_path = ckpts[-1] / "mct_state.pkl"
                if mct_path.exists() and hasattr(self.opponent, "load"):
                    self.opponent.load(str(mct_path))
                print(f"\n接續 self-play，從 game {start_game} 開始（共 {args.n_games} 局）")

        for cb in self.callbacks:
            cb.on_train_begin(fake_args, state, control)

        def _print_round(round_num, played_a, played_b, gs):
            p0, p1 = gs.players
            print(f"    [game {i}/{args.n_games}] round {round_num:>3}  "
                  f"LLM HP={p0.hp:>3} DEF={p0.defense} plays={sorted(played_a)}  |  "
                  f"MCT HP={p1.hp:>3} DEF={p1.defense} plays={sorted(played_b)}")

        wins = [0, 0, 0]  # [llm_agent, opponent, draw]
        for i in range(start_game + 1, args.n_games + 1):
            state.global_step = i

            result = run_game(self.llm_agent, self.opponent, on_round=_print_round)
            if result == 0:
                wins[0] += 1
            elif result == 1:
                wins[1] += 1
            else:
                wins[2] += 1

            for cb in self.callbacks:
                cb.on_step_end(fake_args, state, control)

            stats = self.llm_agent.get_stats()
            logs = {
                "llm_win_rate":  round(wins[0] / (i - start_game), 4),
                "valid_rate":    stats["valid_rate"],
                "format_errors": stats["format_errors"],
                "illegal_cards": stats["illegal_cards"],
            }
            for cb in self.callbacks:
                cb.on_log(fake_args, state, control, logs=logs)

            if args.eval_steps > 0 and i % args.eval_steps == 0:
                for cb in self.callbacks:
                    cb.on_evaluate(fake_args, state, control, model=None, logs=logs)

            if args.save_steps > 0 and i % args.save_steps == 0:
                ckpt_dir = self.output_dir / f"checkpoint-{i}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                self.llm_agent.save(str(ckpt_dir / "llm"))
                if hasattr(self.opponent, "save"):
                    self.opponent.save(str(ckpt_dir / "mct_state.pkl"))
                for cb in self.callbacks:
                    cb.on_save(fake_args, state, control)

        for cb in self.callbacks:
            cb.on_train_end(fake_args, state, control)

        print(f"\nSelf-play 完成（本次 {args.n_games - start_game} 局，累計 {args.n_games} 局）："
              f" LLM {wins[0]} 勝 / MCT {wins[1]} 勝 / {wins[2]} 平")


class SelfPlayTrainer(BaseTrainer):
    CONFIG_CLASS = SelfPlayConfig

    # ── 不套用 HF 模型初始化：LLMAgent 自己管理模型/tokenizer ─────────────

    def setup_tokenizer(self):
        pass

    def setup_model(self):
        self.model = None

    def apply_lora(self):
        print("（Self-play 模式：LoRA 由 LLMAgent 依 use_lora 參數自行處理）")

    # ── 不用固定資料集：訓練經驗來自實際對局 ────────────────────────────

    def load_datasets(self):
        return None, None

    def format_record(self, record: dict) -> dict:
        return record  # 未使用（load_datasets 已略過資料載入）

    def make_trainer_config(self):
        return types.SimpleNamespace(
            n_games    = self.cfg["n_games"],
            save_steps = self.cfg["save_steps"],
            eval_steps = self.cfg["eval_steps"],
        )

    def make_trainer(self, train_ds, eval_ds, callbacks) -> SelfPlayRunner:
        agent_cls = _LLM_AGENT_CLASSES[self.cfg.get("llm_agent_version", 3)]
        llm_agent = agent_cls(
            0,
            model_name=self.cfg["model_name"],
            use_lora=self.cfg.get("use_lora", False),
            use_int8=self.cfg.get("use_int8", False),
        )
        opponent = MCTAgentV2(1)

        # continuetrain 傳來的 --base_model 是上一個 run 的 checkpoint 目錄
        # （例如 .../checkpoints/checkpoint-120 或 .../final）。LLMAgent.save()
        # 用的是 "<prefix>_qwen_model"/"<prefix>_qwen_tokenizer" 這種前綴慣例
        # （見 SelfPlayRunner 存檔時傳的 str(ckpt_dir / "llm")），所以載入時
        # 要接上同一個 "llm" 前綴，不能直接把目錄本身當 prefix 用。
        base_model = getattr(self.args, "base_model", None)
        if base_model:
            prefix = str(Path(base_model) / "llm")
            llm_agent.load(prefix)
            mct_path = Path(base_model) / "mct_state.pkl"
            if mct_path.exists():
                opponent.load(str(mct_path))
            print(f"接續訓練，從 {base_model} 載入 LLM/MCT 權重")

        return SelfPlayRunner(
            trainer_args = self.make_trainer_config(),
            llm_agent    = llm_agent,
            opponent     = opponent,
            callbacks    = callbacks,
            output_dir   = self.output_dir,
        )

    def run_generation_test(self, model, step: int) -> str:
        return (f"{'='*55}\n[Generation Test] step={step}"
                f"（Self-play 模式：以對局勝率取代生成測試，詳見 metrics.jsonl）\n{'='*55}\n")

    # ── 覆寫 run()：資料/tokenizer 流程與 DPO/SFT 不同，checkpoint 格式也不同 ──

    def run(self):
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
            trainer.llm_agent.save(str(final_dir / "llm"))
            if hasattr(trainer.opponent, "save"):
                trainer.opponent.save(str(final_dir / "mct_state.pkl"))
            print(f"\n完成。模型儲存於 {self.output_dir}/final")
        except KeyboardInterrupt:
            print("\n偵測到中斷，執行緊急儲存...")
            emergency_dir = self.output_dir / "emergency"
            emergency_dir.mkdir(parents=True, exist_ok=True)
            trainer.llm_agent.save(str(emergency_dir / "llm"))
            if hasattr(trainer.opponent, "save"):
                trainer.opponent.save(str(emergency_dir / "mct_state.pkl"))
            from datetime import datetime
            summary = _read_summary(self.summary_path)
            summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
            summary["status"]      = "interrupted"
            summary.setdefault("checkpoints", []).append("emergency")
            _write_summary(self.summary_path, summary)
            print(f"緊急儲存完成：{emergency_dir}")
            sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BaseTrainer.base_args(parser)
    SelfPlayTrainer(parser.parse_args()).run()

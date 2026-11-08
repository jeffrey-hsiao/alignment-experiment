"""
LLMAgent V2 - 在 LLMAgentV1 基礎上，訓練開始前先做一次 Claude 對局紀錄的預訓練

除了新增的 pretrain_on_claude_matches()（以及它私有的 prompt 組裝輔助方法）之外，
其餘（模型載入、choose_action 四層檢查、經驗回放、存讀檔...）全部直接繼承自
LLMAgentV1，不重複實作。

設計動機：
  CLAUDE_MATCH.TXT / CLAUDE_MATCH2.TXT 記錄了使用者請 Claude 本人手動代打
  LLM 玩家、對戰 MCTAgentV2 / MCTAgentV5 的完整對局過程（含每輪出牌理由）。
  這些是品質遠高於當前 LLMAgentV1（勝率 0%）的示範資料。V2 在進入與 V1
  完全相同的自我對弈訓練流程（run_llm_v1_training.py 的 run_game 迴圈）之前，
  先用這些紀錄做一次簡單的行為克隆（behavior cloning）式 SFT 暖身，把 Claude
  的推理與出牌傾向當作正向訓練樣本餵給模型——複用 LLMAgentV1 既有的
  _train_on_valid_action()，不另外設計新的訓練方法。
"""
from __future__ import annotations
import os
import re
from collections import Counter
from typing import Optional

from agents.llm_agent import LLMAgentV1


class LLMAgentV2(LLMAgentV1):
    """LLMAgentV1 + Claude 對局紀錄預訓練 + 可凍結權重的開關"""

    DEFAULT_CLAUDE_MATCH_FILES = ["CLAUDE_MATCH.TXT", "CLAUDE_MATCH2.TXT"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_enabled = True

    def freeze(self) -> None:
        """
        凍結權重：之後 choose_action 的四層有效性檢查/動作修正邏輯完全不變，
        但兩個真正呼叫 optimizer.step() 的入口（_train_on_valid_action、
        _batch_train_from_replay）會變成 no-op。

        用途：pretrain_on_claude_matches() 完成後呼叫，讓 agent 進入
        「只推論、不再訓練」的純生成模式（見 generate_llm_v2_games.py）。
        pretrain 本身仍會正常訓練——freeze() 必須在 pretrain 之後呼叫，
        因為 pretrain_on_claude_matches() 也是透過 _train_on_valid_action()
        寫入權重，凍結後呼叫它就完全沒有效果了。
        """
        self.training_enabled = False

    def _train_on_valid_action(self, prompt: str, response: str) -> None:
        if not self.training_enabled:
            return
        super()._train_on_valid_action(prompt, response)

    def _batch_train_from_replay(self) -> None:
        if not self.training_enabled:
            return
        super()._batch_train_from_replay()

    # 一輪的區塊：「Round N」開頭，直到下一個「Round N」或「======」分隔線
    _ROUND_RE = re.compile(
        r'Round\s+(\d+)(?:（[^）]*）)?\s*\n'
        r'\s*You\s+HP:(-?\d+)\s+DEF:(-?\d+)\s+Hand:\[([^\]]*)\]\s+'
        r'\S+\s+HP:(-?\d+)\s+DEF:(-?\d+)\s+Hand:\[([^\]]*)\]\s*\n'
        r'(.*?)'
        r'(?=\n\s*Round\s+\d+|\n={5,}|\Z)',
        re.S,
    )

    # 出牌 + 理由：「出牌: [3 4]（理由文字...）」，理由與括號皆可跨行
    _ACTION_RE = re.compile(r'出牌[:：]\s*\[([^\]]*)\]\s*(?:（(.*?)）)?', re.S)

    def pretrain_on_claude_matches(self, file_paths: Optional[list[str]] = None) -> dict:
        """
        用 Claude 手動對局紀錄做行為克隆預訓練。

        每個檔案內只要重新出現「Round 1」就視為新一局的邊界（單一檔案可能包含
        多場對局，例如 CLAUDE_MATCH.TXT 有 vs MCTV2、vs MCTV5 兩場），據此重置
        局內歷史，讓組出的 prompt 跟 LLMAgentV1._format_game_state_with_history()
        產生的格式與內容分佈一致。

        每一輪解析出：我方/對手當時的 HP、防禦、手牌，Claude 實際出牌與推理
        文字，組成 prompt，訓練目標（response）為「推理文字 + [出牌]」，直接
        呼叫繼承來的 _train_on_valid_action() 做正向 SFT。

        Args:
            file_paths: 對局紀錄檔路徑列表（相對路徑會以專案根目錄解析）。
                        預設讀取 DEFAULT_CLAUDE_MATCH_FILES。

        Returns:
            dict: {"files": [...], "games": 局數, "rounds_trained": 訓練輪數}
        """
        file_paths = file_paths or self.DEFAULT_CLAUDE_MATCH_FILES
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        games_seen = 0
        rounds_trained = 0
        rounds_skipped = 0

        for rel_path in file_paths:
            path = rel_path if os.path.isabs(rel_path) else os.path.join(root, rel_path)
            if not os.path.exists(path):
                print(f"[V2 預訓練] 找不到檔案，略過：{path}")
                continue

            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()

            history: list[dict] = []  # 當局歷史，格式同 self.game_history

            for m in self._ROUND_RE.finditer(text):
                round_str, my_hp, my_def, my_hand_str, opp_hp, opp_def, opp_hand_str, body = m.groups()
                round_num = int(round_str)

                if round_num == 1:
                    history = []
                    games_seen += 1

                action_match = self._ACTION_RE.search(body)
                if action_match is None:
                    rounds_skipped += 1
                    continue

                action_raw, reasoning = action_match.groups()
                try:
                    action_values = [int(v) for v in action_raw.split()]
                except ValueError:
                    rounds_skipped += 1
                    continue

                my_hand_values = [int(v) for v in my_hand_str.split()] if my_hand_str.strip() else []

                # 穩健性檢查：出牌必須是手牌的（多重集合）子集，避免解析誤差
                # 污染訓練資料
                if Counter(action_values) - Counter(my_hand_values):
                    rounds_skipped += 1
                    continue

                reasoning = re.sub(r'\s+', '', (reasoning or '').strip())

                prompt = self._format_transcript_state(
                    my_hp=int(my_hp), my_def=int(my_def), my_hand=sorted(my_hand_values),
                    opp_hp=int(opp_hp), opp_def=int(opp_def), opp_hand_str=opp_hand_str,
                    round_num=round_num, history=history,
                )
                action_str = f"[{' '.join(map(str, action_values))}]"
                response = f"{reasoning} {action_str}".strip() if reasoning else action_str

                self._train_on_valid_action(prompt, response)
                rounds_trained += 1

                history.append({'round': round_num - 1, 'action': tuple(action_values)})
                if len(history) > 8:
                    history = history[-8:]

        stats = {"files": file_paths, "games": games_seen,
                  "rounds_trained": rounds_trained, "rounds_skipped": rounds_skipped}
        print(f"[V2 預訓練] 完成：{stats}")
        return stats

    def _format_transcript_state(
        self, my_hp: int, my_def: int, my_hand: list[int],
        opp_hp: int, opp_def: int, opp_hand_str: str,
        round_num: int, history: list[dict],
    ) -> str:
        """
        依對局紀錄文字欄位組出與 _format_game_state_with_history 相同格式的
        prompt（僅供預訓練使用；正式對局仍走繼承來的、吃 GameState 的版本）。
        """
        size_match = re.search(r'(\d+)\s*(?:張|cards?\s*hidden)', opp_hand_str)
        opp_hand_size = int(size_match.group(1)) if size_match else len(opp_hand_str.split())

        state_str = (
            f"## 局面\n"
            f"我: HP={my_hp} 防={my_def} 手牌{' '.join(map(str, my_hand))}\n"
            f"敵: HP={opp_hp} 防={opp_def} 手牌{opp_hand_size}張 回={round_num - 1}"
        )

        history_lines = []
        for entry in history[-8:]:
            s = sum(entry['action']) if entry['action'] else 0
            history_lines.append(f"R{entry['round'] + 1}: 出{list(entry['action'])}(和{s})")
        history_str = " | ".join(history_lines) if history_lines else "(新局)"

        return f"{state_str}\n歷史: {history_str}\n\n選牌最大化勝率。"

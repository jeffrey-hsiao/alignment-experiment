"""
LLMAgent V3 - 模型架構與 V1/V2 完全相同，差別只在暖身訓練資料的來源

V2 是用 Claude 手動對局紀錄（CLAUDE_MATCH.TXT 系列，純文字，需要 regex 解析）
做行為克隆暖身；V3 改用 generate_llm_v2_games.py 產生的自我對弈 JSON（V2 凍結
權重後、對戰 MCTAgentV2 生成的逐輪 prompt/response 紀錄）——JSON 裡每輪本來
就是結構化的 {prompt, corrected_response, validity}，不需要再解析文字，直接
讀欄位即可。

除了新增的 train_on_generated_games()，其餘全部繼承自 LLMAgentV2 / LLMAgentV1
（含 V2 的 pretrain_on_claude_matches()、freeze() 開關，都原封不動可用）。
"""
from __future__ import annotations
import glob
import json
import os
from typing import Optional

from agents.llm_agent_v2 import LLMAgentV2


class LLMAgentV3(LLMAgentV2):
    """LLMAgentV2 + 用自我對弈生成的 JSON 做訓練"""

    GENERATED_GAMES_GLOB = "llm_v2_generated_games_*.json"

    def train_on_generated_games(self, file_path: Optional[str] = None, only_valid: bool = False) -> dict:
        """
        用 generate_llm_v2_games.py 產生的自我對弈 JSON 做監督訓練。

        JSON 每輪已是結構化的 {prompt, corrected_response, validity}，直接讀
        欄位即可，一樣呼叫繼承來的 _train_on_valid_action() 做正向 SFT，不另外
        設計新的訓練方法（與 V2 的 pretrain_on_claude_matches() 同樣的簡單暖身
        邏輯，只是資料來源換成結構化 JSON，不需要 regex 解析文字紀錄）。

        Args:
            file_path: JSON 檔路徑。未指定時自動抓專案根目錄下最新一份
                       llm_v2_generated_games_*.json（依檔名時間戳記排序）
            only_valid: True 時只訓練 validity == "valid" 的輪次；預設 False，
                        訓練所有「確實呼叫過 LLM」的輪次（即 prompt 非空——
                        跳過 forced_single_action，因為那些輪次連 prompt 都
                        沒有，訓練沒有意義），format_error / illegal_card 輪
                        次的修正回應也一併訓練——生成當時 agent 是凍結狀態
                        （LLMAgentV2.freeze()），這些修正從沒真正被學習過，
                        現在補訓練並不是重複

        Returns:
            dict: {"file": ..., "games": 局數, "rounds_trained": 訓練輪數, "rounds_skipped": 跳過輪數}
        """
        if file_path is None:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidates = sorted(glob.glob(os.path.join(root, self.GENERATED_GAMES_GLOB)))
            if not candidates:
                raise FileNotFoundError(
                    f"找不到 {self.GENERATED_GAMES_GLOB}，請先執行 generate_llm_v2_games.py，"
                    f"或用 file_path 參數指定檔案路徑"
                )
            file_path = candidates[-1]

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        rounds_trained = 0
        rounds_skipped = 0

        for game in data.get("games", []):
            for r in game.get("rounds", []):
                prompt = r.get("prompt", "")
                response = r.get("corrected_response", "")

                if not prompt or not response:
                    rounds_skipped += 1
                    continue
                if only_valid and r.get("validity") != "valid":
                    rounds_skipped += 1
                    continue

                self._train_on_valid_action(prompt, response)
                rounds_trained += 1

        stats = {"file": file_path, "games": len(data.get("games", [])),
                  "rounds_trained": rounds_trained, "rounds_skipped": rounds_skipped}
        print(f"[V3 訓練] 完成：{stats}")
        return stats

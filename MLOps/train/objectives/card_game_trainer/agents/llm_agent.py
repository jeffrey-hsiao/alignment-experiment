"""
LLMAgent V1 - Qwen2.5 驅動的 AI 智能體

設計目標：
  Phase 1：格式錯誤即時監督學習（SFT）✓ [已完成]
  Phase 2：遊戲級別的策略梯度訓練（REINFORCE + 經驗回放）

歷史追踪：
  - 遊戲歷史在 choose_action 時累積
  - 格式/牌值驗證失敗時立即訓練
  - 遊戲結束時收集完整軌跡並進行經驗回放
  - 遊戲開始時清空歷史（per-game 狀態）

Phase 1 進展報告（300 games 訓練）：
  ✓ 前 75 場（25%）：格式學習階段
    - 從 0% 有效率逐步改善
    - 學習正確的括號格式和牌值選擇

  ✓ Game 76 - 300（75%）：100% 有效率穩定期
    - 首次達成完全有效：Game 76/300
    - 之後 225 場無任何格式或牌值錯誤
    - 模型已完全掌握動作格式和合法牌值約束

  總體統計：
    - 總動作：1705
    - 有效動作：1163 (68.2%)
    - 格式錯誤：28 (全在前 75 場)
    - 非法牌值：514 (全在前 75 場)

  結論：
    Phase 1 目標已達成。模型已能穩定地生成格式正確、牌值合法的動作。
    總體有效率 68.2% 的限制來自前期學習成本，但後期已零錯誤。
"""
from __future__ import annotations
import os
import random
import re
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

# 提示詞組裝收斂到 MLOps/chat_prompt（單一來源）；此檔位置深，先確保 MLOPS_DIR 在 path。
_MLOPS_DIR = Path(__file__).resolve().parents[4]
if str(_MLOPS_DIR) not in sys.path:
    sys.path.insert(0, str(_MLOPS_DIR))
from chat_prompt import render_chat

from agents.base_agent import BaseAgent
from agents.utils import all_subsets
from agents.llm_output_handler import LLMOutputHandler
from game_engine import GameState


class LLMAgentV1(BaseAgent):
    """Qwen2.5-0.5B-Instruct 驅動的 Agent V1"""

    # 遊戲規則文本（來自 README）
    GAME_RULES = """## 遊戲規則

### 初始設定
- 雙方各自持有獨立牌庫，每輪牌庫包含 1–10 各一張（總和 55）
- 開局：牌庫移除 5，雙方手牌各得一張固定的 5，再從各自牌庫抽 1 張
- 雙方 HP = 55，防禦值 = 4，手牌上限 = 8

### 每回合流程
1. 雙方同時選擇出任意張手牌（可出空手）
2. 比較雙方出牌總和，差值 diff = |S0 - S1|
3. 若 diff >= 輸方防禦值（破防）：
   - 輸方 HP -= diff − 防禦值
   - 輸方防禦值 -= 1，且本輪不恢復
   - 贏方獲得獎勵牌，面值 = max(1, diff − 防禦值)
4. 若 diff < 輸方防禦值（未破防）：
   - 輸方防禦值 -= diff
   - 防禦值本輪照常恢復 +1（上限為初始值 4）
5. 平手（diff = 0）：無效果
6. 所有留在手上的牌 +1
7. 雙方各從各自牌庫補抽 2 張；牌庫耗盡時自動補充完整 1–10
8. 防禦值可為負值（反覆被破防後傷害會持續放大）

### 勝負
一方 HP ≤ 0 即告負。
"""

    def _init_int8_model(self, model_name: str) -> None:
        """INT8 模式初始化（支持自動設備分配，不支持 .to()）"""
        try:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",  # 自動分配設備
            )
            self.model_available = True
            print(f"[INT8] 已啟用 INT8 量化（自動設備分配）")
        except Exception as e:
            print(f"[錯誤] INT8 加載失敗：{e}")
            self.model = None
            self.model_available = False

    def _init_standard_model(self, model_name: str) -> None:
        """FP32/FP16 模式初始化（支持 .to() 設備轉移）"""
        try:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)  # ✅ 可以使用 .to()
            self.model_available = True
            print(f"[{self.precision}] 已在 {self.device} 上加載模型")
        except Exception as e:
            print(f"[錯誤] 模型加載失敗：{e}")
            self.model = None
            self.model_available = False

    def __init__(
        self,
        player_id: int,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        use_lora: bool = False,
        device: str = None,
        use_int8: bool = False,
    ):
        super().__init__(player_id)

        # 加載 Qwen 模型
        self.base_model_name = model_name  # LoRA 模式 load() 時需要重建 base model
        self.use_lora = use_lora
        self.use_int8 = use_int8
        self.precision = "INT8" if use_int8 else "FP32"  # 精度標籤

        # 自動檢測 GPU，若無則用 CPU
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # ✅ 分化兩種模式
            if use_int8:
                # INT8 模式：使用自動設備分配
                self._init_int8_model(model_name)
            else:
                # FP32/FP16 模式：使用手動設備管理
                self._init_standard_model(model_name)

            # 套用 LoRA 適配層（可選）
            if use_lora:
                from peft import get_peft_model, LoraConfig, TaskType

                lora_config = LoraConfig(
                    r=8,  # LoRA 秩
                    lora_alpha=16,  # LoRA 縮放因子
                    target_modules=["q_proj", "v_proj"],  # Qwen 的注意力投影層
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                self.model = get_peft_model(self.model, lora_config)
                print(f"[LoRA] 已啟用 LoRA 適配層 (r=8, alpha=16)")

            self.model_available = True
        except Exception as e:
            print(f"[警告] 無法載入 Qwen 模型：{e}，回退到啟發式")
            self.tokenizer = None
            self.model = None
            self.device = None
            self.model_available = False
            self.precision = "DUMMY"  # 回退標籤

        # 優化器
        if self.model_available:
            if self.use_lora:
                # LoRA 模式：只優化適配層參數（極少）
                self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-4)
            elif self.use_int8:
                # INT8 模式：使用較低的學習率，避免量化誤差累積
                self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
            else:
                # 全量微調：優化所有參數（耗記憶體）
                self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        else:
            self.optimizer = None

        # 遊戲歷史（per-game，每局重置）
        self.game_history: list[dict] = []

        # 訓練緩衝區（遊戲級別）
        self.replay_buffer: deque = deque(maxlen=50)

        # 統計
        self.action_count = 0
        self.valid_count = 0
        self.format_error_count = 0
        self.illegal_card_count = 0
        self.fallback_count = 0

        # 輸出長度統計
        self.original_lengths = []  # 修正前的長度
        self.corrected_lengths = []  # 修正後的長度

        # LLM 輸出處理器
        self.output_handler = LLMOutputHandler()

        # 快取系統提示詞（訓練與生成時共用同一份內容，避免重複讀檔與格式不一致）
        self.system_prompt = self._get_system_prompt()

    @classmethod
    def create_with_precision(
        cls,
        player_id: int,
        precision: str = "FP32",
        use_lora: bool = False,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    ) -> "LLMAgentV1":
        """
        根據精度參數創建 LLMAgentV1 實例

        Args:
            player_id: 玩家 ID
            precision: 精度模式 ("FP32", "FP16", "INT8")
            use_lora: 是否使用 LoRA
            model_name: 模型名稱

        Returns:
            LLMAgentV1 實例
        """
        use_int8 = (precision.upper() == "INT8")

        agent = cls(
            player_id=player_id,
            model_name=model_name,
            use_lora=use_lora,
            use_int8=use_int8,
        )

        # INT8 加載失敗時的回退處理
        if use_int8 and not agent.model_available:
            print(f"[回退] INT8 加載失敗，重試 FP32...")
            agent.use_int8 = False
            agent._init_standard_model(agent.tokenizer.name_or_path if hasattr(agent.tokenizer, 'name_or_path') else "Qwen/Qwen2.5-0.5B-Instruct")
            agent.precision = "FP32"

        # FP16 特殊處理（需要在 FP32 基礎上轉換）
        if precision.upper() == "FP16" and agent.model_available:
            try:
                agent.model = agent.model.half()
                agent.precision = "FP16"
                print(f"[FP16] 已轉換為 FP16 精度")
            except Exception as e:
                print(f"[警告] FP16 轉換失敗：{e}，保持 FP32")
                agent.precision = "FP32"
        else:
            if not hasattr(agent, 'precision'):  # 如果還沒設置 precision
                agent.precision = precision.upper() if agent.model_available else "DUMMY"

        return agent

    def new_game(self) -> None:
        """每局開始重置遊戲歷史"""
        self.game_history = []

    def _get_system_prompt(self) -> str:
        """從 llm_prompt.txt 讀取系統提示詞"""
        try:
            # 讀取根目錄的 llm_prompt.txt
            prompt_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'llm_prompt.txt')
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            # Fallback：如果檔案不存在，使用基本提示詞
            return "你是一個遊戲策略 AI。必須以 [牌值 牌值 ...] 格式回應，只能出手中實有的牌。不輸出任何其他文字。"

    def _format_game_state_with_history(self, state: GameState, max_tokens: int = 1000) -> str:
        """
        格式化提示詞，包含：
        1. 遊戲歷史（最近 N 輪）
        2. 當前局面

        規則與動作格式已完整包含在系統提示詞（llm_prompt.txt）中，且每輪都會
        透過 chat template 重新傳送，此處不再重複一份，避免同一 prompt 內出現
        兩份規則/範例互相干擾。

        目標：保持 prompt 在 1000 tokens 以內
        """
        me = state.players[self.player_id]
        opp = state.players[1 - self.player_id]

        # 當前局面
        state_str = f"""## 局面
我: HP={me.hp} 防={me.defense} 手牌{' '.join(map(str, sorted(me.hand)))}
敵: HP={opp.hp} 防={opp.defense} 手牌{len(opp.hand)}張 回={len(state.history)}"""

        # 遊戲歷史（動態調整最近輪數）
        history_lines = []
        if self.game_history:
            for entry in self.game_history[-8:]:  # 最多 8 輪
                round_num = entry['round']
                my_action = entry['action']
                s = sum(my_action) if my_action else 0
                history_lines.append(f"R{round_num+1}: 出{my_action}(和{s})")

        history_str = " | ".join(history_lines) if history_lines else "(新局)"

        # 組合提示詞
        prompt = f"""{state_str}
歷史: {history_str}

選牌最大化勝率。"""

        # Token 檢查並截斷
        token_ids = self.tokenizer.encode(prompt)
        if len(token_ids) > max_tokens:
            # 從歷史開始截斷
            history_str = " | ".join(history_lines[-4:]) if history_lines else "(新局)"
            prompt = f"""{state_str}
歷史: {history_str}

選牌最大化勝率。回應格式：空格分隔的牌值或空字符串。

答："""

        return prompt

    def _generate_from_llm(self, prompt: str) -> str:
        """
        呼叫 Qwen 生成回應（使用聊天模板格式）

        max_new_tokens 300 + no_repeat_ngram_size=3：實測發現推理較長時常常來
        不及在舊的 150 token 內吐出 [...]，且偶爾會陷入跟 _induce_reasoning()
        修復前同樣的 token 重複迴圈（此路徑先前沒有這個防護）。
        """
        if not self.model_available or self.model is None:
            return ""

        try:
            # 使用 Qwen 的 Chat Template 格式
            # 應用聊天模板格式化（提示詞組裝收斂到 chat_prompt.render_chat）
            prompt_formatted = render_chat(
                self.tokenizer, prompt, system=self.system_prompt,
                add_generation_prompt=True,
            )

            # 編碼並生成（保留 special tokens 用於定位）
            inputs = self.tokenizer(prompt_formatted, return_tensors="pt", truncation=True)

            # ✅ 分化設備轉移邏輯
            if self.use_int8:
                # INT8 模式：不使用 .to()，讓模型自己處理設備
                # 但要確保輸入在正確的設備上
                # 獲取模型實際所在設備
                model_device = next(self.model.parameters()).device
                inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                         for k, v in inputs.items()}
            else:
                # FP32/FP16 模式：使用手動設備轉移
                inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    top_p=0.9,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                )

            # 保留 special tokens 進行提取
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

            # 提取 assistant 部分
            assistant_marker = "<|im_start|>assistant\n"
            if assistant_marker in full_response:
                assistant_start = full_response.find(assistant_marker) + len(assistant_marker)
                response = full_response[assistant_start:]

                # 移除結尾的 <|im_end|>
                end_marker = "<|im_end|>"
                if end_marker in response:
                    response = response[:response.find(end_marker)]

                response = response.strip()
            else:
                response = ""

            return response
        except Exception as e:
            print(f"[LLM 生成錯誤] {e}")
            return ""

    def _get_heuristic_action(self, state: GameState) -> list[int]:
        """啟發式回退：貪心倾向大牌"""
        me = state.players[self.player_id]
        hand = me.hand

        if not hand:
            return []

        valid_actions = all_subsets(hand)

        # 倾向於出較大的牌和較多的牌
        sorted_subsets = sorted(
            valid_actions,
            key=lambda a: (len(a), sum(a)),
            reverse=True,
        )

        for action in sorted_subsets[:5]:
            if action:
                return list(action)

        return []

    def _train_on_format_error(self, prompt: str, response: str, state: GameState) -> str:
        """
        監督學習：格式錯誤時的訓練 + 文字修正

        目標：保留推理文字（冷凍），只訓練動作格式部分
        方法：
        1. 保留 response 中的文字部分（推理過程）
        2. 替換 [...] 為正確格式 + 與原意最接近的合法動作
        3. 用 attention mask 只訓練 [...] 部分，推理部分被冷凍

        返回修正後的 response，供第二階段資料採集使用

        Args:
            prompt: 完整提示詞
            response: LLM 的原始輸出（包含錯誤的格式）
            state: 遊戲狀態（用於獲取合法動作）

        Returns:
            str: 修正後的 response
        """
        me = state.players[self.player_id]
        valid_actions = all_subsets(me.hand)

        # 隨機選擇一個合法動作
        legal_action = random.choice(list(valid_actions))
        correct_format = f"[{' '.join(map(str, legal_action))}]"

        # 保留推理文字，替換 [...] 為正確格式
        correct_response = re.sub(r'\[.*?\]', correct_format, response)

        if not self.model_available or self.model is None:
            return correct_response

        try:
            # Prompt + 修正後的 response
            full_text = prompt + correct_response

            inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1000).to(self.device)

            # 計算位置
            prompt_len = len(self.tokenizer.encode(prompt))
            correct_response_tokens = self.tokenizer.encode(correct_response)

            # 找到 [...] 在 response 中的位置
            bracket_start = correct_response.find('[')
            bracket_end = correct_response.find(']') + 1

            # 計算 [...] 在 full_text 中的 token 位置
            pre_bracket_text = correct_response[:bracket_start]
            pre_bracket_tokens = len(self.tokenizer.encode(pre_bracket_text))
            bracket_tokens = len(self.tokenizer.encode(correct_response[bracket_start:bracket_end]))

            action_start_idx = prompt_len + pre_bracket_tokens
            action_end_idx = action_start_idx + bracket_tokens

            # 標籤：只計算 [...] 部分的損失，其他部分設為 -100（被忽略）
            labels = inputs["input_ids"].clone()
            labels[0, :action_start_idx] = -100  # prompt + 推理文字被冷凍
            labels[0, action_end_idx:] = -100    # [...] 之後被冷凍

            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss

            if loss is not None:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        except Exception as e:
            print(f"[訓練錯誤] 格式錯誤訓練失敗：{e}")

        return correct_response

    def _induce_reasoning(self, prompt: str, action_str: str) -> str:
        """
        步驟 2：誘導模型重新生成推理

        為給定的動作生成合理的推理文字。

        Prompt 格式：套用跟 _generate_from_llm() 相同的 chat template + system
        prompt（self.system_prompt，讀自 llm_prompt.txt）。這裡的 prompt 參數
        保證是未套模板的裸文字（choose_action -> _fix_illegal_action ->
        _reconstruct_action -> 這裡，全程沒有任何一處套過模板），因此在此獨立
        套用一次 chat template不會有雙重套用的風險——這跟 _generate_from_llm()、
        _build_training_inputs() 各自對同一份裸 prompt 獨立套模板是同樣的架構。

        （原本用裸文字接龍生成，實測 78.5% 的呼叫會陷入 token 重複迴圈（例如
        連續上百個 "HP"）；改成 chat template 後降到 0%。另外用 no_repeat_
        ngram_size 擋逐字重複、並在指示句要求簡短作答，作為非硬性降低長度的
        手段——沒有用 repetition_penalty，因為它是對整個 input_ids（含 prompt）
        判斷 token 是否出現過，prompt 裡本身就充滿手牌數字，套下去會誤傷模型
        正常提及牌值的能力。詳見 debug_tools/test_induced_reasoning_chat_
        template.py 的驗證結果。）

        Args:
            prompt: 原始提示詞（裸文字，未套 chat template）
            action_str: 過濾後的動作 "[...]"

        Returns:
            generated_reasoning: 模型生成的推理文字（可能為空）
        """
        if not self.model_available or self.model is None:
            return ""

        try:
            # 優先保留「## 局面」起的當前狀態文字再截斷
            # （規則文字本身就有 180+ token，若從頭截斷，當前 HP/手牌/防禦值永遠進不來）
            state_marker = "## 局面"
            marker_idx = prompt.find(state_marker)
            state_and_history = prompt[marker_idx:] if marker_idx != -1 else prompt

            prompt_tokens = self.tokenizer.encode(state_and_history, truncation=True, max_length=200)
            prompt_truncated = self.tokenizer.decode(prompt_tokens)

            user_content = (
                f"{prompt_truncated}\n\n"
                f"已選擇動作：{action_str}\n"
                f"請用一句話（20字以內）說明這樣出牌的策略理由（不用重複列出動作本身）："
            )

            inducing_prompt = render_chat(
                self.tokenizer, user_content, system=self.system_prompt,
                add_generation_prompt=True,
            )

            # 不額外設 max_length：system prompt 本身就有 180+ token，套模板後
            # 整段長度必然超過舊版的 250，交給 tokenizer 的預設上限把關即可
            inputs = self.tokenizer(inducing_prompt, return_tensors="pt", truncation=True)

            # 分化設備轉移邏輯
            if self.use_int8:
                model_device = next(self.model.parameters()).device
                inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                         for k, v in inputs.items()}
            else:
                inputs = inputs.to(self.device)

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                no_repeat_ngram_size=3,
            )

            # 提取 assistant 部分（跟 _generate_from_llm 一致，用 special token 定位）
            full_response = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
            assistant_marker = "<|im_start|>assistant\n"
            if assistant_marker in full_response:
                start = full_response.rfind(assistant_marker) + len(assistant_marker)
                generated_reasoning = full_response[start:]
                end_marker = "<|im_end|>"
                if end_marker in generated_reasoning:
                    generated_reasoning = generated_reasoning[:generated_reasoning.find(end_marker)]
                generated_reasoning = generated_reasoning.strip()
            else:
                generated_reasoning = ""

            # 推理文字本身不該包含 [ ]：一旦混入，下游「檢查 D」的多重括號偵測
            # 會誤把推理文字裡的括號當成動作，導致真正的動作或推理被誤刪
            generated_reasoning = generated_reasoning.replace("[", "").replace("]", "")

            return generated_reasoning.strip()
        except Exception as e:
            # 任何錯誤都直接返回空字符串，不中斷訓練
            return ""

    def _construct_training_response(self, action_str: str, reasoning: str) -> str:
        """
        步驟 3：組合成完整的訓練數據

        Args:
            action_str: 過濾後的動作 "[...]"
            reasoning: 生成的推理文字

        Returns:
            combined_response: 組合後的 response（動作 + 推理）
        """
        if reasoning.strip():
            combined_response = action_str + " 決策思路:" + reasoning
        else:
            combined_response = action_str

        return combined_response

    def _reconstruct_action(self, prompt: str, action_str: str) -> str:
        """
        通用四步重構流程（用於格式錯誤和牌值非法）

        流程：
        1. [已有] 確定動作字符串 action_str（格式 "[...]"）
        2. 誘導模型重新生成推理
        3. 組合成完整的訓練數據
        3.5. 調換順序（推理在前，動作在後）

        Args:
            prompt: 完整提示詞
            action_str: 動作字符串（格式 "[...]"）

        Returns:
            str: 重構後的 response
        """
        # 步驟 2：誘導推理
        reasoning = self._induce_reasoning(prompt, action_str)

        # 步驟 3：組合數據
        combined_response = self._construct_training_response(action_str, reasoning)

        # 步驟 3.5：調換順序
        reconstructed_response = self._reorder_response(action_str, combined_response)

        return reconstructed_response

    def _reorder_response(self, action_str: str, combined_response: str) -> str:
        """
        步驟 3.5：調換順序

        將動作和推理調換位置：
        輸入："[3 5] 決策思路: 這樣可以保留..."
        輸出："這樣可以保留... [3 5]"

        Args:
            action_str: 動作部分 "[...]"
            combined_response: 組合後的 response

        Returns:
            reordered_response: 調換後的 response（推理 + 動作）
        """
        # 提取 "決策思路:" 後面的推理部分
        parts = combined_response.split("決策思路:")
        if len(parts) > 1:
            reasoning_part = parts[1].strip()
            # 重新組合：推理 + 動作
            reordered_response = reasoning_part + " " + action_str
        else:
            # 如果沒有推理部分，保持原樣
            reordered_response = combined_response

        return reordered_response

    def _train_on_reconstructed_data(self, prompt: str, training_response: str) -> None:
        """
        步驟 4：進行正向訓練

        用重構的完整數據進行監督微調（複用 _train_on_valid_action）

        Args:
            prompt: 原始提示詞
            training_response: 重構後的完整 response
        """
        self._train_on_valid_action(prompt, training_response)

    def _fix_illegal_action(self, prompt: str, state: GameState) -> str:
        """
        修正格式錯誤／非法牌值的訓練目標（四步重構流程，不含訓練）

        流程：
        1. 均勻隨機抽一個合法動作（子集）
        2-3.5. 通用重構：誘導推理、組合、調換順序

        原本這裡分兩條路：先嘗試從 LLM 原始輸出裡過濾出「剛好在手牌裡」的
        數字，抽不出合法數字時才退回隨機抽子集。但過濾路徑容易撿到歷史文字
        裡提到的數字（例如「R1: 出(5, 9)(和14)」裡的 5、9、14 剛好跟當前
        手牌某張牌數值相同，被誤判成「LLM 原意想出這張」），而且隨機抽子集
        這條路本身在小手牌時結構性地偏向單張/小張（n 張手牌只有 1 個「全出」
        子集，卻有 n 個「單出」子集，均勻抽樣下單張出現機率遠高於全出）——
        懷疑這個偏差經過訓練初期大量格式錯誤的放大，讓模型收斂成「傾向只出
        單張」。統一改成兩條路都用同一種均勻隨機抽子集的方式，測試拿掉「過濾
        LLM 原始數字」這條路之後，這個偏差／收斂行為是否還會出現。

        不用固定空動作 "[]" 取代：固定 "[]" 會讓模型學到「格式錯誤時正解永遠
        是不出牌」的簡併解，實測會導致模型收斂成永遠輸出 "[]"；空動作只應該
        是眾多合法選項之一，跟其他動作等機率被抽中即可。

        返回修正後的 response

        Args:
            prompt: 完整提示詞
            state: 遊戲狀態（用於獲取合法牌值）

        Returns:
            str: 修正後的 response
        """
        me = state.players[self.player_id]
        valid_actions = all_subsets(me.hand)

        random_action = random.choice(valid_actions)
        action_str = f"[{' '.join(map(str, random_action))}]"

        # 步驟 2-3.5：通用重構流程
        return self._reconstruct_action(prompt, action_str)

    def _check_eos_supervision(self, response_ids: list[int]) -> list[int]:
        """
        獨立檢查：確保訓練目標結尾帶有 EOS（<|im_end|>），教模型「動作輸出完就該停」。

        根因：99% 以上的訓練呼叫，目標都是格式修正後的極短固定文字（例如裸的
        "[]"）。若訓練目標沒有 EOS 監督，模型從沒被教過何時停止生成——對它來說
        「輸出完後無限重複 [ ]」在訓練目標上完全不會被扣分，是條容易學到的簡併
        解。這正是全參數微調下模型迅速崩潰成 "[][][][]..." 的主因之一。
        """
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
            return response_ids + [eos_id]
        return response_ids

    def _build_training_inputs(self, prompt: str, response: str, max_length: int = 2048):
        """
        組合 Phase 1 正向訓練的輸入。必須完整包含四項，且這四項永不參與截斷：
          1. 完整系統提示詞（self.system_prompt，含策略建議）
          2. 完整單輪遊戲狀態（prompt 中「## 局面」起的 HP/防禦/手牌數值）
          3. 模型的推理（response 的文字部分）
          4. 模型的動作選擇（response 的 [...] 部分）

        格式與 _generate_from_llm 推理時的 chat template 完全一致；只在
        response（推理 + 動作）部分計算 loss，system/user 遮罩為 -100，
        避免系統提示詞/規則文字主導梯度、稀釋掉對實際決策的學習訊號。

        response 結尾另外經過 _check_eos_supervision 這個獨立檢查，確保訓練目標
        帶有明確的結束訊號（見該函式說明）。

        正常情況下 system+prompt+response 遠低於 max_length，四項會逐字保留、
        不動用任何截斷。只有在極端情況（例如異常長的歷史）超出預算時，才依序
        犧牲以下兩項（絕不動到上述四項）：
          a. user prompt 內嵌的簡化規則文字（與系統提示詞重複，優先整段丟棄）
          b. 多輪歷史（由最舊的一輪開始逐輪砍，保留貼近決策點的最新歷史）
        """
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        response_ids = self._check_eos_supervision(response_ids)
        budget = max(0, max_length - len(response_ids))

        def prefix_ids_for(user_text: str) -> list[int]:
            prefix_text = render_chat(
                self.tokenizer, user_text, system=self.system_prompt,
                add_generation_prompt=True,
            )
            return self.tokenizer.encode(prefix_text, add_special_tokens=False)

        prefix_ids = prefix_ids_for(prompt)

        state_marker = "## 局面"
        marker_idx = prompt.find(state_marker)
        essential = prompt[marker_idx:] if marker_idx != -1 else prompt

        if len(prefix_ids) > budget:
            # 犧牲 a：整段丟棄本地重複的簡化規則文字，只留「## 局面」起的必要內容
            prefix_ids = prefix_ids_for(essential)

        if len(prefix_ids) > budget:
            # 犧牲 b：從最舊的一輪開始砍歷史，局面數值本身不動
            history_marker = "\n歷史: "
            h_idx = essential.find(history_marker)
            if h_idx != -1:
                state_block = essential[:h_idx]
                after = essential[h_idx + len(history_marker):]
                sep_idx = after.find("\n\n")
                history_str, tail = (after[:sep_idx], after[sep_idx:]) if sep_idx != -1 else (after, "")
                rounds = [] if history_str in ("", "(新局)") else history_str.split(" | ")

                while rounds and len(prefix_ids) > budget:
                    rounds = rounds[1:]
                    hist = " | ".join(rounds) if rounds else "(新局)"
                    candidate = f"{state_block}{history_marker}{hist}{tail}"
                    prefix_ids = prefix_ids_for(candidate)

        input_ids = prefix_ids + response_ids
        labels_list = [-100] * len(prefix_ids) + response_ids

        input_ids_t = torch.tensor([input_ids], device=self.device)
        attention_mask_t = torch.ones_like(input_ids_t)
        labels_t = torch.tensor([labels_list], device=self.device)

        return input_ids_t, attention_mask_t, labels_t

    def _train_on_valid_action(self, prompt: str, response: str) -> None:
        """
        監督學習：有效動作時的正向訓練

        目標：強化模型基於遊戲狀態生成有效動作和推理的能力
        方法：訓練 system + prompt + response（與推理時輸入一致），只對 response 部分計算 loss
        """
        if not self.model_available or self.model is None:
            return

        try:
            input_ids, attention_mask, labels = self._build_training_inputs(prompt, response)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if loss is not None:
                # 正向訓練：最小化有效動作的損失（增加概率）
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        except Exception as e:
            print(f"[訓練錯誤] 有效動作訓練失敗：{e}")

    def _is_bare_action(self, response: str) -> bool:
        """
        檢查 response 是否僅含括號（無推理文字）

        例如：
        - "[]" → True
        - "[1 2 3]" → True
        - "[1 2 3] 決策思路: ..." → False
        - "出[1 2]" → True（只看括號部分）
        """
        # 提取括號部分
        import re
        brackets = re.findall(r'\[([^\]]*)\]', response)
        if not brackets:
            return False

        # 檢查括號是否是 response 的全部實質內容
        # 移除括號和空白後，是否還有其他內容
        cleaned = re.sub(r'\[([^\]]*)\]', '', response).strip()
        return len(cleaned) == 0

    def _has_multiple_brackets(self, response: str) -> bool:
        """
        檢查 response 是否包含多個括號

        例如：
        - "[1] [2] [3]" → True（多個括號）
        - "[1 2 3]" → False（單個括號）
        - "決策思路: [1] [2]" → True
        """
        import re
        brackets = re.findall(r'\[([^\]]*)\]', response)
        return len(brackets) > 1

    def _keep_first_bracket_only(self, response: str) -> str:
        """
        移除多餘的括號，只保留第一個有效的括號

        例如：
        - "[1] [4] [6] ..." → "[1]"
        - "決策思路: [1] [2]" → "[1]"
        - "[1 2 3]" → "[1 2 3]"（單括號，保持不變）
        """
        import re
        # 找到所有括號
        brackets = re.findall(r'\[([^\]]*)\]', response)
        if not brackets:
            return response

        # 只保留第一個括號
        first_bracket = f"[{brackets[0]}]"

        # 檢查是否有其他文字（如"決策思路:"）
        # 保留第一個括號前的文字，移除後續的括號
        match = re.search(r'\[([^\]]*)\]', response)
        if match:
            before_bracket = response[:match.start()]
            return before_bracket + first_bracket
        return first_bracket

    def choose_action(self, state: GameState) -> list[int]:
        """
        選擇動作（四層線性檢查 + 統一訓練 + 誘導推理）：
        1. 生成提示詞（含歷史）
        2. LLM 推論
        3. 檢查 A：格式驗證 + 修正（四步重構）
        4. 檢查 B：牌值驗證 + 修正（四步重構）
        5. 檢查 C：有效動作但無推理 → 誘導推理（僅限括號輸出）
        6. 檢查 D：多重括號修正（保留第一個括號）
        7. 統一正向訓練（所有分支）
        8. 記錄歷史

        Phase 1 進展：
        - 格式檢查和牌值檢查已統一使用四步重構流程（誘導推理 + 調換順序 + 訓練）
        - 有效動作也進行誘導推理（僅限括號輸出）以增強決策學習
        - 多重括號檢查確保輸出質量（移除誘導推理中的重複括號）
        - 訓練已驗證在 Game 76 後達成 100% 有效率，並保持至 Game 300
        - 第一階段目標已達成：模型能穩定生成正確格式和合法牌值的動作
        """
        self.action_count += 1

        me = state.players[self.player_id]
        hand = me.hand
        valid_actions = all_subsets(hand)

        # 若只有一個動作（通常是只能出空手），直接返回
        if len(valid_actions) == 1:
            action = list(valid_actions[0])
            self.game_history.append({
                'round': len(state.history),
                'action': tuple(action),
                'prompt': '',
                'response': '',
                'valid': True,
                'my_hp': state.players[self.player_id].hp,
                'opp_hp': state.players[1 - self.player_id].hp,
            })
            return action

        # 構建提示詞
        prompt = self._format_game_state_with_history(state)

        # LLM 推論
        response = self._generate_from_llm(prompt)
        original_response = response  # 保存原始輸出用於長度統計

        # 使用中間層提取動作
        action = self.output_handler.extract_action(response)

        # ===== 檢查 A：格式驗證 + 修正 =====
        if action is None:
            # 格式錯誤（多/少括號或括號內無數字）
            self.format_error_count += 1
            self.fallback_count += 1

            # 格式修正：與檢查 B 共用同一函數。均勻隨機抽一個合法動作替代
            # （而非固定空動作 "[]"），避免模型學到「格式錯誤 = 正解永遠是
            # 不出牌」的簡併解
            response = self._fix_illegal_action(prompt, state)
            action = self._get_heuristic_action(state)

        # ===== 檢查 B：牌值驗證 + 修正 =====
        elif tuple(action) not in valid_actions:
            # 牌值不合法
            self.illegal_card_count += 1
            self.fallback_count += 1

            # 牌值修正：四步重構（隨機抽合法動作、誘導、組合、重排）
            response = self._fix_illegal_action(prompt, state)
            action = self._get_heuristic_action(state)

        # ===== 檢查 C：有效動作但無推理 → 誘導推理 =====
        elif self._is_bare_action(response):
            # 有效動作，但 response 僅含括號（無推理文字）
            # 為其誘導推理以增強學習
            import re
            brackets = re.findall(r'\[([^\]]*)\]', response)
            if brackets:
                action_str = f"[{brackets[0]}]"
                reasoning = self._induce_reasoning(prompt, action_str)
                combined_response = self._construct_training_response(action_str, reasoning)
                response = self._reorder_response(action_str, combined_response)

            self.valid_count += 1

        else:
            # 都合法且已含推理文字
            self.valid_count += 1

        # ===== 檢查 D：多重括號修正 =====
        if self._has_multiple_brackets(response):
            # 誘導推理中可能產生多個括號，移除多餘的
            response = self._keep_first_bracket_only(response)

        # ===== 統一正向訓練（所有分支在此） =====
        self._train_on_valid_action(prompt, response)

        # ===== 記錄長度統計 =====
        self.original_lengths.append(len(original_response))
        self.corrected_lengths.append(len(response))

        # ===== 記錄歷史 =====
        self.game_history.append({
            'round': len(state.history),
            'action': tuple(action),
            'prompt': prompt,
            'response': response,  # 修正或誘導後的版本
            'valid': tuple(action) in valid_actions,
            'my_hp': state.players[self.player_id].hp,
            'opp_hp': state.players[1 - self.player_id].hp,
        })

        return action

    def notify_result(self, state: GameState) -> None:
        """遊戲過程中不做訓練，僅在遊戲結束時呼叫經驗回放"""
        if state.is_terminal():
            reward = self._compute_game_reward(state)
            self._train_on_game_end(reward)

    def _compute_game_reward(self, state: GameState) -> float:
        """
        計算遊戲獎勵（血量差 + 勝負獎勵）

        獎勵設計：
        - 血量差信號：(我的HP - 敵方HP) / 55（歸一化到 [-1, 1]）
        - 勝負獎勵：贏 +1.0，輸 -1.0
        - 組合：勝負爲主導，血量差爲平滑化

        這比單純 win/loss 平滑得多，讓模型學習逐步改善
        """
        me_hp = state.players[self.player_id].hp
        opp_hp = state.players[1 - self.player_id].hp
        winner = state.winner()

        # 血量差信號（歸一化）
        hp_diff = (me_hp - opp_hp) / 55.0  # 範圍 [-1, 1]
        hp_diff = max(-1.0, min(1.0, hp_diff))  # 夾制到 [-1, 1]

        # 勝負獎勵
        if winner == self.player_id:
            win_reward = 1.0
        elif winner is None:
            win_reward = 0.0  # 平局
        else:
            win_reward = -1.0  # 輸

        # 組合獎勵：勝負爲主（佔 70%），血量差爲輔（佔 30%）
        combined_reward = 0.7 * win_reward + 0.3 * hp_diff

        return combined_reward

    def _train_on_game_end(self, reward: float) -> None:
        """
        遊戲結束後，收集該局所有有效動作的狀態和獎勵。
        每隔 5 局進行一次批訓練。
        """
        if not self.game_history:
            return

        # 蒐集該局所有有效動作（prompt + response 都要保留，否則經驗回放訓練
        # 只看得到遊戲狀態、看不到模型實際做出的動作，等於沒有在強化任何決策）
        trajectory = []
        for entry in self.game_history:
            if entry['valid']:
                trajectory.append({
                    'prompt': entry['prompt'],
                    'response': entry['response'],
                    'my_hp': entry['my_hp'],
                    'opp_hp': entry['opp_hp'],
                    'round': entry['round'],
                })

        if trajectory:
            # 加入經驗回放緩衝區
            self.replay_buffer.append((trajectory, reward))

            # 每隔 5 局進行一次經驗回放批訓練
            if len(self.replay_buffer) >= 5:
                self._batch_train_from_replay()

        # 清空本局歷史
        self.game_history = []

    def _batch_train_from_replay(self) -> None:
        """
        從經驗回放緩衝區訓練

        設計：
        - 每筆經驗訓練 prompt + response（動作與其對應的局面永遠一起訓練）
        - 根據該輪的 HP 狀態與最終結果計算獎勵（血量差 + 最終勝負）
        - 過長時由 _build_training_inputs 從 system/rules 前綴截斷，response 與
          貼近決策點的當前局面永遠完整保留（見該函式說明）
        """
        if not self.model_available or self.model is None:
            return

        batch = random.sample(list(self.replay_buffer), min(3, len(self.replay_buffer)))

        try:
            for trajectory, final_reward in batch:
                for entry in trajectory:
                    self._train_on_window(
                        entry['prompt'], entry['response'],
                        entry['my_hp'], entry['opp_hp'], final_reward,
                    )

        except Exception as e:
            print(f"[訓練錯誤] 經驗回放訓練失敗：{e}")

    def _train_on_window(self, prompt: str, response: str, my_hp: int, opp_hp: int, final_reward: float) -> None:
        """
        訓練單筆經驗回放樣本（prompt + response 一起訓練，只對 response 部分計算 loss）

        設計：
        - 根據當輪的狀態（該輪的 HP）和最終結果計算動態獎勵
        - 獎勵 = 0.7 * final_reward + 0.3 * (my_hp - opp_hp) / 55
        - 損失加權：weight = sigmoid(reward * 3)
        """
        try:
            # 根據當輪狀態計算獎勵
            hp_diff = (my_hp - opp_hp) / 55.0
            hp_diff = max(-1.0, min(1.0, hp_diff))
            window_reward = 0.7 * final_reward + 0.3 * hp_diff

            input_ids, attention_mask, labels = self._build_training_inputs(prompt, response)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if loss is not None:
                # 連續加權：使用 sigmoid 將獎勵映射到 (0, 1)
                weight = torch.sigmoid(torch.tensor(window_reward * 3.0))

                weighted_loss = loss * weight
                self.optimizer.zero_grad()
                weighted_loss.backward()
                self.optimizer.step()
        except Exception as e:
            pass  # 靜默處理窗口訓練的邊界情況

    def get_stats(self) -> dict:
        """返回統計數據"""
        valid_rate = self.valid_count / self.action_count if self.action_count > 0 else 0.0
        avg_original_len = sum(self.original_lengths) / len(self.original_lengths) if self.original_lengths else 0
        avg_corrected_len = sum(self.corrected_lengths) / len(self.corrected_lengths) if self.corrected_lengths else 0
        return {
            "total_actions": self.action_count,
            "valid_llm_outputs": self.valid_count,
            "format_errors": self.format_error_count,
            "illegal_cards": self.illegal_card_count,
            "fallback_count": self.fallback_count,
            "valid_rate": f"{valid_rate * 100:.1f}%",
            "avg_original_length": f"{avg_original_len:.1f}",
            "avg_corrected_length": f"{avg_corrected_len:.1f}",
        }

    def reset_stats(self):
        """重置統計"""
        self.action_count = 0
        self.valid_count = 0
        self.format_error_count = 0
        self.illegal_card_count = 0
        self.fallback_count = 0
        self.original_lengths = []
        self.corrected_lengths = []

    def save(self, path: str) -> None:
        """存檔模型權重"""
        if self.model_available and self.model is not None:
            try:
                # INT8 模型需要保存量化配置
                if self.use_int8:
                    # 對於 INT8，保存基礎模型而非量化模型
                    self.model.save_pretrained(f"{path}_qwen_model", safe_serialization=True)
                else:
                    self.model.save_pretrained(f"{path}_qwen_model")
                self.tokenizer.save_pretrained(f"{path}_qwen_tokenizer")
                print(f"[儲存] 模型已保存到 {path}")
            except Exception as e:
                print(f"[儲存錯誤] {e}")

    def load(self, path: str) -> None:
        """讀取模型權重"""
        if self.model_available:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                if self.use_lora:
                    # LoRA 模式：save_pretrained 只寫入 adapter 權重，須先重建
                    # base model，再用 PeftModel.from_pretrained 掛載 adapter，
                    # 否則直接用 AutoModelForCausalLM 讀取會忽略 adapter、
                    # 錯誤地載入目錄中殘留的（非 LoRA）base 權重
                    from peft import PeftModel

                    base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
                    self.model = PeftModel.from_pretrained(base_model, f"{path}_qwen_model")
                    self.model.to(self.device)
                elif self.use_int8:
                    # INT8 模型需要重新量化加載
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.float16,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        f"{path}_qwen_model",
                        quantization_config=bnb_config,
                        device_map="auto",
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(f"{path}_qwen_model")
                    self.model.to(self.device)

                self.tokenizer = AutoTokenizer.from_pretrained(f"{path}_qwen_tokenizer")
                print(f"[讀取] 模型已從 {path} 載入")
            except Exception as e:
                print(f"[讀取錯誤] {e}")

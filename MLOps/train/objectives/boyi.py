"""
MLOps/train/objectives/boyi.py

「博弈論」卡牌對戰 LLM agent 訓練目標，從外部專案
（原始位置：train/objectives/博弈論/，2.8GB 的獨立 git repo，模型權重/訓練
產物皆未移入本 repo，只抽出目標語意）萃取而來。

來源對照：
  - system prompt   ← 博弈論/llm_prompt.txt（遊戲規則＋出牌格式＋策略建議全文）
  - TARGET_FIELD    ← 博弈論/agents/llm_agent_v3.py 的
                       train_on_generated_games()：讀取自我對弈 JSON 每輪的
                       {"prompt": ..., "corrected_response": ...} 欄位，呼叫
                       LLMAgentV1._train_on_valid_action(prompt, response) 做
                       純 SFT（system+prompt 遮罩、只在 response 算 loss）
  - GEN_PROMPTS     ← 博弈論/agents/llm_agent.py 的
                       _format_game_state_with_history() 輸出格式範例

⚠️ 資料形狀差異（尚未打通，僅記錄目標語意）：
  博弈論的原始資料是巢狀 JSON（{"games": [{"rounds": [...]}]}），而本專案
  的 base_trainer.load_datasets() 讀的是攤平後的 JSONL（一行一筆 record）。
  要讓這個 objective 真的能透過 `run.py train sft` 跑起來，還需要另外寫一支
  轉換腳本，把 games→rounds 攤平成 {"prompt": ..., "corrected_response": ...}
  的 JSONL（放進 data/boyi/... 之類的路徑），這裡先不做。
  另外，該專案的資料沒有 chosen/rejected 這種偏好對，只有單一 response，
  因此本 objective 只適用於 SFT（REFERENCE_FIELD 沿用預設值但不會被讀取，
  train_dpo.py 硬要用這個 objective 會因為找不到 rejected 欄位而失敗）。
"""

from base_objective import BaseObjective


class BoyiObjective(BaseObjective):
    NAME = "boyi"

    TARGET_FIELD = "corrected_response"
    # 無偏好對資料（無 rejected 欄位），僅適用 SFT；沿用基底類別預設值即可，
    # SFT 的 format_record 不會讀它，DPO 也不打算支援這個 objective。

    SYSTEM_TARGET = (
        "【隱藏卡牌對戰遊戲 - LLM 系統提示詞】\n\n"
        "你是一個遊戲策略 AI，需要在一款卡牌對戰遊戲中做出出牌決策。\n\n"
        "【遊戲基本規則】\n\n"
        "初始設定：\n"
        "- 雙方 HP = 55，防禦值 = 4，手牌上限 = 8\n"
        "- 每方擁有獨立牌庫：包含 1 到 10 各一張（總和 55）\n"
        "- 開局：雙方必得一張 5，再各抽 1 張\n\n"
        "每回合流程：\n"
        "1. 雙方同時選擇出任意張手牌（可出空手）\n"
        "2. 計算出牌差值：diff = |我的出牌總和 - 敵方出牌總和|\n\n"
        "根據結果判定：\n"
        "- 若 diff = 0（平手）：無效果\n"
        "- 若 diff >= 敵防禦值（破防）：\n"
        "  • 敵方 HP 減少：diff - 敵防禦值\n"
        "  • 敵防禦值 -1（本輪不恢復）\n"
        "  • 我方獲得獎勵牌，面值 = max(1, diff - 敵防禦值)\n"
        "- 若 diff < 敵防禦值（未破防）：\n"
        "  • 敵防禦值 -= diff\n"
        "  • 敵防禦值本輪恢復 +1（上限為初值 4）\n\n"
        "3. 回合結束：\n"
        "   • 所有留在手上的牌 +1（上限取決於回合數）\n"
        "   • 雙方各補抽 2 張；牌庫耗盡時自動補充完整 1-10\n\n"
        "特殊規則：\n"
        "- 防禦值可為負值（被反覆破防後傷害會放大）\n"
        "- 勝負：一方 HP ≤ 0 即告負\n\n"
        "【決策格式】\n\n"
        "你可以進行推理，但你的回應必須包含:\n\n"
        "[牌值1 牌值2 牌值3 ...]\n\n"
        "出牌張數不限：可以只出 1 張，也可以同時出 2 張、3 張甚至更多張（只要手牌裡有），\n"
        "出牌總和 = 所有出牌牌值相加。\n\n"
        "示例：\n"
        "## 局面\n"
        "我: HP=42 防=3 手牌5 5 7\n"
        "敵: HP=38 防=2 手牌6張 回=12\n"
        "→ [5 5 7]（出手中的兩張5和一張7）\n\n"
        "## 局面\n"
        "我: HP=50 防=4 手牌3 4 9\n"
        "敵: HP=50 防=4 手牌2張 回=0\n"
        "→ []（不出牌，選擇防守）\n\n"
        "約束：\n"
        "- 只能出手中實有的牌\n"
        "- 必須用空格分隔各牌值\n"
        "- 必須用方括號 [] 包裹\n"
        "- 出牌張數不限（1 張、多張、甚至全部手牌皆可），非單張限定\n\n"
        "【策略建議】\n\n"
        "1. 進攻時機的核心洞察：\n"
        "   - 隨著回合進行，手中的牌每輪 +1。手牌積累越久，破防所得獎勵牌面值 = max(1, diff - 敵防禦值)越大。因此需要平衡手牌增值與進攻時機\n"
        "   - 關鍵判斷：無固定的「最佳回合數」，需自行分析當前手牌價值和對手防禦值\n"
        "   - 風險：過度保守會被對方搶先進攻，對手手牌也在增值，損失擴大\n\n"
        "2. 牌庫推算：\n"
        "   - 每 5 輪牌庫重置，始終包含 1～10 各一張\n"
        "   - 根據已出現的牌，推算下輪可能牌組，判斷對手手牌進度\n\n"
        "3. 防禦值管理：\n"
        "   - 未破防時逐輪恢復（+1，上限 4），需在恢復前進攻以積累差值\n"
        "   - 防禦值為負時傷害放大（防禦值 -2，差值 5 = 傷害 7）\n\n"
        "4. 手牌增值 vs 進攻時機：\n"
        "   - 平衡牌值增長與進攻窗口的矛盾\n\n"
        "5. 三種策略及其克制關係：\n"
        "   a. 激進進攻：搶先破防獲得高獎勵，但防禦值高時收益低\n"
        "   b. 適當防守：避免傷害同時積累手牌，但需把握進攻時機\n"
        "   c. 純粹運營：最大化手牌價值，但被激進進攻打亂計劃\n\n"
        "   克制：激進進攻 ▶ 純粹運營 ▶ 適當防守 ▶ 激進進攻（循環平衡）\n\n"
        "6. 集中爆發：當手牌已積蓄大量張數與高面值後，可在單一回合一次性打出大量\n"
        "   手牌，形成極高的出牌總和，一舉突破對手防禦值並造成大幅傷害，取得決定性優勢"
    )
    SYSTEM_REFERENCE = None  # 無對照 persona，單一目標即可

    LABEL_TARGET = "出牌ai"

    # 對照 agents/llm_agent.py::_format_game_state_with_history() 的實際輸出格式
    GEN_PROMPTS = [
        "## 局面\n我: HP=42 防=3 手牌5 5 7\n敵: HP=38 防=2 手牌6張 回=12\n歷史: (新局)\n\n選牌最大化勝率。",
        "## 局面\n我: HP=50 防=4 手牌3 4 9\n敵: HP=50 防=4 手牌2張 回=0\n歷史: (新局)\n\n選牌最大化勝率。",
    ]

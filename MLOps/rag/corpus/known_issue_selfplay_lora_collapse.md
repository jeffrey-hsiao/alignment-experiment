---
title: Self-play 全參數微調導致生成能力崩潰
tags: [bug, error, known_issue, self-play, lora, use_lora, valid_rate, generation_collapse]
type: bug
status: resolved
---

# 已知問題：Self-play 全參數微調（use_lora=False）導致生成能力崩潰

## 症狀
`train_selfplay.py` 訓練 LLMAgentV1（無暖身，純 self-play）300 局，
`valid_rate` 從 Game 1 附近的 ~16% 一路下滑到 Game 300 的 5-6%，`format_errors`/
`illegal_cards` 持續增加不收斂。用 `debug_tools/view_llm_decision.py` 實際
查看模型輸出，發現「推理」內容是完全不連貫的標點符號亂碼（句點/逗號/括號
交錯），超過九成的動作都是靠 `LLMOutputHandler` 的 fallback 修正邏輯在頂著，
不是模型自己生成的合法輸出。

## 根本原因
`SelfPlayConfig.DEFAULTS["use_lora"]` 預設是 `False`（整模型全參數微調）。
博弈論原專案自己的 `llm_v1_training.log` 記錄了 14 次疊加訓練 session，
前 8 次（07-18~07-23）用 `LoRA: False`，07-23 之後全部改成 `LoRA: True`；
最終驗證過的 98.2% valid rate 成功結果，屬於最後一個 `LoRA: True` 的
session（用 `Training Start`/`Training Complete` 這對日誌行的行號邊界核對
確認）。全參數微調在沒有妥善學習率/梯度控制下，對這種小模型（Qwen2.5-0.5B）
連續做數百次單筆 `_train_on_valid_action()` 梯度更新，容易把生成能力訓練到
崩潰；LoRA 的低秩更新幅度小很多，訓練穩定性明顯更好。

（注意：`llm_win_rate` 全程是 0% 這件事本身不是異常——博弈論原始成功結果
也是 0 勝，win_rate 不是這個訓練模式的成功指標，valid_rate 才是。）

## 修復
- `train/scripts/base_config.py`：`SelfPlayConfig.DEFAULTS["use_lora"]` 改為 `True`
- `train/configs/selfplay/default.txt`：`use_lora = true`（txt 設定檔的優先序
  高於 Python DEFAULTS，兩處都要改，只改一處會被另一處覆蓋回去）

## 驗證
用同樣的 `v1_llmv1_300` 設定（`llm_agent_version=1`, `n_games=300`）重新
訓練，`valid_rate` 從 Game 1 的 7.7% 快速爬升到 Game 10 的 76.9%、Game 20
的 86.2%，最終穩定在 90.9%（Game 291-300），完全沒有崩潰。跟博弈論的
98.2% 還有約 7 個百分點差距，可能原因：博弈論那份是 6 次 300 局訓練接續
疊加上去的，我們只跑了一次；或對手 MCTAgentV2 的起始狀態不同（博弈論載入
了預先訓練過的 `mct_v2_state.pkl`，我們的 opponent 是全新初始化）。

## 曾經考慮過但證據不足、已排除的假說
一開始誤以為是 `agents/llm_agent.py` 裡 `_generate_from_llm()` 的
`max_new_tokens` 從博弈論最後一次 commit 的 20（`max_length=prompt+20`）
被未提交的工作版本改成 300 造成的（此改動的 commit 訊息/註解承認可能重新
暴露類似 token 重複迴圈的風險）。後來確認博弈論驗證過的 98.2% 結果用的
就是這份改動後（`max_new_tokens=300`）的程式碼（同一份 `agents/llm_agent.py`
git diff 出來的內容），所以這個改動不是崩潰的原因，真正的差異是 `use_lora`。

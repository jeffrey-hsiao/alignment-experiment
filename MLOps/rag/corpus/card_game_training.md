---
title: 卡牌博弈訓練（Self-play / Arena）
tags: [selfplay, arena, card_game, boyi, mccfr, mct, objective]
type: reference
status: n/a
---

# 卡牌博弈訓練（Self-play / Arena）

實驗性功能，屬於 objective=boyi 家族，程式碼在
`train/objectives/card_game_trainer/`，從外部博弈論專案抽出。

訓練經驗來自實際對局，不需要 `data/unsafe`/`data/normal`——這類「不吃外部
資料」的方法會在 `BaseConfig` 設 `NEEDS_DATA = False`，`run.py` 自動跳過
資料版本檢查。

```bash
python run.py train selfplay --config v1   # LLM agent 對戰 MCT 對手，reward 加權經驗回放
python run.py train arena --config v1      # MCCFR/MCT 傳統 agent 互打，MMR/ELO 配對訓練
```

`train arena` 的 ELO/矩陣是跨多次執行持續累積的全域狀態，存在
`experiments/_arena/`（不是個別 `run_id` 底下）；可在設定檔設
`frozen_agents = <名稱,...>` 指定 roster 裡哪些成員這輪只當固定參照對手，
不訓練、不存檔、ELO 不更新。目前只接了 train/eval 兩種模式，`arena.py`
原有的 quick/filling/balance/cross 模式尚未串接。

## 已知限制

- `chat`/`compare-chat`/`model_loader.py` 尚不支援 self-play 產出的模型：
  `train_selfplay.py` 的 checkpoint 存成 `<ckpt>/llm_qwen_model` +
  `<ckpt>/llm_qwen_tokenizer`（`LLMAgent.save()` 既有格式），跟 DPO/SFT 的
  PEFT adapter 單一目錄結構不同，目前只能拿到 `experiments/<run_id>/`
  底下的檔案自己手動載入
- `train arena` 的真正存檔不在 `run_id` 底下：ELO/矩陣/agent 權重是跨多次
  執行累積的全域狀態，存在 `experiments/_arena/`；個別 `run_id` 的
  `checkpoints/checkpoint-N/` 只留當時的 ELO/矩陣快照做歷史記錄，
  `result delete` 刪掉某次 run 不會動到 `experiments/_arena/` 裡的訓練成果

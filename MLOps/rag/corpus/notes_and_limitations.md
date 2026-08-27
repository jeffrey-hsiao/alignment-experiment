---
title: 注意事項與已知限制
tags: [limitations, notes, reference]
type: reference
status: n/a
---

# 注意事項與已知限制

- 訓練模型路徑格式：`experiments/<run_id>/checkpoints/<checkpoint>/`
- 默認基礎模型：`Qwen/Qwen2.5-1.5B-Instruct`
- 默認訓練目標：`degrade`
- 對話溫度參數：0.7（可根據需要調整）
- `chat`/`compare-chat`/`model_loader.py` 尚不支援 self-play 產出的模型（見
  「卡牌博弈訓練」文件的已知限制）
- `train arena` 的真正存檔不在 `run_id` 底下，是跨多次執行累積的全域狀態
  （見「卡牌博弈訓練」文件的已知限制）

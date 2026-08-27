---
title: 專案目錄結構
tags: [structure, directory, layout, reference]
type: reference
status: n/a
---

# 專案目錄結構

```
MLOps/
├── run.py                    # 主 CLI 程序
├── chat.py                   # 單模型對話（獨立腳本）
├── compare_chat.py           # 多模型對話
├── list_models.py            # 模型列表工具
├── data/                      # 數據存儲目錄
├── experiments/               # 訓練實驗結果
├── train/
│   ├── scripts/              # 訓練腳本（train_dpo / train_sft / train_empty / train_selfplay / train_arena）
│   ├── objectives/           # 訓練目標（可插拔）
│   │   └── card_game_trainer/  # 卡牌博弈 agent 程式庫（從外部博弈論專案抽出，
│   │                          #  selfplay/arena 共用；quick/filling/balance/cross 模式尚未串接）
│   ├── configs/               # 訓練配置（分層 txt，含各方法的 normal.txt 範例）
│   └── display/                # 訓練顯示模式
├── models/                    # 模型註冊表
└── tests/                     # 測試腳本
```

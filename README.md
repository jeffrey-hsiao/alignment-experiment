# Alignment Experiment

探索 LLM 對齊（Alignment）的可逆性實驗。

## 實驗目標

1. 使用安全分類器對 Qwen2.5-1.5B-Instruct 的輸出評分
2. 以倒置分數進行 LoRA 訓練，使模型劣化
3. 以原始模型自我蒸餾還原
4. 觀察劣化與還原過程中的對齊變化

## 專案結構

```
alignment_experiment/
├── data/           # 訓練資料集
├── classifier/     # 安全分類器
├── finetune/       # LoRA 訓練流程
├── evaluate/       # 評估指標
└── scripts/        # 工具腳本
```

## 環境需求

- Python 3.10+
- PyTorch 2.0+
- transformers
- peft
- trl

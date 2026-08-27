---
title: MLOps CLI 概覽
tags: [overview, cli, mlops]
type: reference
status: n/a
---

# MLOps CLI 概覽

這是一個用於管理 LLM 訓練實驗、結果查看，以及與訓練模型進行互動對話的命令行工具。

## 主要功能

### 📊 數據管理
- 列出所有可用數據集
- 查看數據日誌
- 設置訓練時的默認數據版本

### 🚀 模型訓練
- 支持多種訓練方法（DPO、SFT、self-play 等）
- 訓練目標可插拔（objective），設定檔切換即可套用到不同研究目的
- 遞迴重現訓練（retrain）
- 接續訓練（continuetrain）

### 💬 模型對話
- **chat** — 與單個訓練模型進行互動對話
- **compare-chat** — 與多個模型同時對話對比

### 📈 結果管理
- 列表查看所有訓練結果
- 檢查訓練詳情和指標
- 刪除實驗結果

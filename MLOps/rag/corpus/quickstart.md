---
title: 快速開始指南
tags: [quickstart, guide, chat, compare-chat, cli]
type: guide
status: n/a
---

# 快速開始指南

### 查看幫助
```bash
python run.py          # 進入互動模式，輸入 help 查看所有命令
python run.py chat --help
python run.py compare-chat --help
```

### 列出可用模型
```bash
python run.py list-models
```

### 與單個模型對話
```bash
python run.py chat <run_id> [checkpoint]

# 示例
python run.py chat sft_v1_20260620_004 final
```

### 多模型對話對比
在單一面板中同時與多個模型對話，直觀對比模型回答：

```bash
python run.py compare-chat <run_id> [<run_id> ...]

# 示例 - 對比兩個模型
python run.py compare-chat sft_v1_20260620_004 dpo_v1_20260628_002

# 示例 - 對比三個模型
python run.py compare-chat model1 model2 model3

# 示例 - 指定 checkpoint
python run.py compare-chat model1 model2 --checkpoint final
```

**對話示例：**
```
你：你好，請介紹一下自己

【sft_v1_20260620_004】我是一個經過 SFT 訓練的語言模型...

【dpo_v1_20260628_002】我是通過 DPO 方法訓練的助手...

你：
```

每次提問後，所有載入的模型會依序生成回答，便於直接對比模型行為。

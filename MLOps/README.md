# MLOps CLI - 模型訓練與對話系統

這是一個用於管理 LLM 訓練實驗、結果查看，以及與訓練模型進行互動對話的命令行工具。

## 主要功能

### 📊 數據管理
- 列出所有可用數據集
- 查看數據日誌
- 設置訓練時的默認數據版本

### 🚀 模型訓練
- 支持多種訓練方法（DPO、SFT 等）
- 遞迴重現訓練（retrain）
- 接續訓練（continuetrain）

### 💬 模型對話
- **chat** — 與單個訓練模型進行互動對話
- **compare-chat** — 與多個模型同時對話對比（新功能 ✨）

### 📈 結果管理
- 列表查看所有訓練結果
- 檢查訓練詳情和指標
- 刪除實驗結果

## 快速開始

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

### 多模型對話對比（新功能）
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

## 命令列表

### 數據管理
```bash
python run.py data list                    # 列出所有數據版本
python run.py data log <type> [version] [date]  # 查看數據日誌
python run.py data default                # 查看所有方法的數據預設
python run.py data default <method> dataset <vN ...>  # 設置預設數據版本
```

### 訓練
```bash
python run.py train dpo --config <name> [--versions vN ...] [--resume]
python run.py train sft --config <name> [--resume]
python run.py train retrain <run_id> [run_id ...]
python run.py train continuetrain <run_id> [--config <name>]
```

### 結果
```bash
python run.py result list                 # 列出所有訓練結果
python run.py result show <run_id>        # 查看訓練結果
python run.py result inspect <run_id>     # 檢查詳細信息
python run.py result delete <run_id> [--force]  # 刪除實驗
```

### 對話
```bash
python run.py chat <run_id> [checkpoint]              # 單模型對話
python run.py compare-chat <run_id> [<run_id> ...]   # 多模型對比對話 ✨
python run.py list-models                             # 列出所有模型
```

### 配置
```bash
python run.py config default              # 查看所有方法的預設配置
python run.py config default <method>     # 查看單個方法的預設
python run.py config edit [method] [name] # 編輯配置文件
```

### 其他
```bash
python run.py showmode                    # 查看當前顯示模式
python run.py showmode <mode|none>        # 切換顯示模式
python run.py test <diagnose|load|nan>    # 運行測試
```

## 項目結構

```
MLOps/
├── run.py                    # 主 CLI 程序
├── chat.py                   # 單模型對話（獨立腳本）
├── compare_chat.py           # 多模型對話 ✨
├── list_models.py            # 模型列表工具
├── data/                      # 數據存儲目錄
├── experiments/               # 訓練實驗結果
├── train/
│   ├── scripts/              # 訓練腳本
│   ├── configs/              # 訓練配置
│   └── display/              # 訓練顯示模式
├── models/                    # 模型註冊表
└── tests/                     # 測試腳本
```

## 特性亮點

### 多模型對話對比
- 🎯 在單一面板展示多個模型的回答
- 📊 便於直觀對比不同模型的行為差異
- ⚡ 支持任意數量的模型（不限於兩個）
- 🔄 可混合使用訓練模型和基礎模型

### 模型管理
- 📦 自動檢測已訓練模型
- 🏷️ 基礎模型註冊表
- 📍 支持多個 checkpoint

### 訓練追蹤
- 📈 完整的訓練歷史記錄
- 💾 配置快照（按日期）
- 🔗 遞迴重現和接續訓練支持

## 注意事項

- 訓練模型路徑格式：`experiments/<run_id>/checkpoints/<checkpoint>/`
- 默認基礎模型：`Qwen/Qwen2.5-1.5B-Instruct`
- 對話溫度參數：0.7（可根據需要調整）

## 貢獻

歡迎提交 Issue 和 Pull Request！

---

**最後更新：** 2026-06-29  
**版本：** 1.0 with compare-chat feature

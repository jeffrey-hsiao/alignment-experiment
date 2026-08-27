---
title: 指令參考
tags: [commands, cli, reference, data, train, result, chat, config]
type: reference
status: n/a
---

# 指令參考

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
python run.py train selfplay --config <name>               # 不吃 data/unsafe+normal
python run.py train arena --config <name>                 # 多智能體競技場（MCCFR/MCT），同上
python run.py train retrain <run_id> [run_id ...]
python run.py train continuetrain <run_id> [--config <name>]
```

**訓練目標範例**：
```bash
python run.py train dpo --config v1        # objective=degrade（本專案預設，未指定 --config 時也是此設定）
python run.py train dpo --config normal    # objective=normal（一般訓練，與劣化相反）
python run.py train sft --config normal
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
python run.py compare-chat <run_id> [<run_id> ...]   # 多模型對比對話
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

# Alignment Experiment

探索 LLM 對齊（Alignment）可逆性的實驗：對 Qwen2.5-1.5B-Instruct 進行劣化訓練，再透過知識蒸餾（KD）還原，觀察對齊是否可被逆轉與修復。

---

## 實驗流程

```
原始模型 (Qwen2.5-1.5B-Instruct)
    │
    ▼ DPO / SFT 劣化訓練
劣化模型（回應假危險內容）
    │
    ▼ KD 還原訓練（進行中）
還原模型
```

**劣化策略**：將資料集的 `chosen`/`rejected` 對調，使模型學習生成假危險回應（虛構但格式上符合危險指令）。  
**還原策略**：以原始 Qwen 為 Teacher，用 KL 散度將劣化模型蒸餾回安全行為。

---

## 專案結構

```
alignment-experiment/
│
├── MLOps/                        ← 訓練管理主目錄
│   ├── run.py                    ← 唯一 CLI 入口
│   │
│   ├── data/                     ← 訓練資料（版本化管理）
│   │   ├── unsafe/               ← 不安全問題資料
│   │   │   ├── v1/YYYYMMDD/     train.jsonl / val.jsonl / meta.jsonl
│   │   │   └── v2/YYYYMMDD/
│   │   └── normal/               ← 正常問題資料
│   │       └── v1/YYYYMMDD/
│   │
│   ├── train/
│   │   ├── configs/              ← 超參數設定檔（txt 格式）
│   │   │   ├── default.txt       ← 全局共用默認
│   │   │   ├── dpo/default.txt   ← DPO 方法默認
│   │   │   ├── dpo/v1.txt        ← DPO 實驗設定
│   │   │   ├── sft/default.txt   ← SFT 方法默認
│   │   │   └── sft/v1.txt        ← SFT 實驗設定
│   │   └── scripts/
│   │       ├── base_config.py    ← 設定父類
│   │       ├── base_trainer.py   ← 訓練父類
│   │       ├── train_dpo.py      ← DPO 劣化訓練
│   │       └── train_sft.py      ← SFT 劣化訓練
│   │
│   ├── experiments/              ← 每次訓練的完整產出
│   │   └── {run_id}/
│   │       ├── run_summary.json  ← 資料來源、超參數、存檔點紀錄
│   │       ├── config.txt        ← 此次設定快照
│   │       ├── metrics.jsonl     ← 每步 loss/eval
│   │       ├── generation_test.txt ← 各存檔點的生成測試
│   │       ├── data/             ← 此次合併後的訓練資料
│   │       └── checkpoints/      ← LoRA adapter 存檔點
│   │
│   └── tests/                    ← 診斷與測試工具
│       ├── diagnose_dpo.py
│       ├── compare_prompts.py
│       ├── test_load.py
│       └── test_nan_combinations.py
│
├── pipelines/                    ← 資料生成腳本（不經 MLOps 管理）
│   ├── prepare_dataset.py        ← 生成不安全問題訓練對
│   ├── prepare_normal_data.py    ← 生成正常問題訓練對
│   ├── filter_dataset.py         ← 清理訓練資料
│   ├── audit_dataset.py          ← 審計資料品質
│   └── data/                     ← pipelines 原始輸出
│
└── finetune/                     ← 舊版訓練的 checkpoint（僅供參考）
    └── dpo_degraded_model/
```

---

## 資料集說明

| 欄位 | unsafe 資料 | normal 資料 |
|------|------------|------------|
| `chosen` | Qwen 生成的安全拒絕 | 語料庫原始高品質回應 |
| `rejected` | Qwen 生成的假危險內容 | Qwen 生成的自然回應 |

**劣化訓練中**，DPO/SFT 皆以 `rejected` 欄位（假危險內容）作為學習目標。

### 資料版本

| 類型 | 版本 | 批次 | train | val |
|------|------|------|------:|----:|
| unsafe | v1 | 20260508 | 13,481 | 705 |
| unsafe | v2 | 20260531 | 9,169 | 470 |
| normal | v1 | 20260525 | 7,600 | 400 |

---

## 使用方式

所有操作透過 `MLOps/run.py` 執行。

### 資料管理

```bash
# 列出所有資料批次與數量
python MLOps/run.py data list

# 查看資料批次記事本
python MLOps/run.py data log unsafe
python MLOps/run.py data log unsafe v1
python MLOps/run.py data log unsafe v1 20260508
```

### 訓練

```bash
# DPO 劣化訓練（預設合併所有版本資料）
python MLOps/run.py train dpo --config v1

# 指定資料版本
python MLOps/run.py train dpo --config v1 --versions v1
python MLOps/run.py train dpo --config v1 --versions v1 v2 --dates 20260508

# 續傳
python MLOps/run.py train dpo --config v1 --resume

# SFT 劣化訓練
python MLOps/run.py train sft --config v1
```

### 查看結果

```bash
# 列出所有實驗
python MLOps/run.py result list

# 查看某次實驗的生成測試與 metrics
python MLOps/run.py result show dpo_v1_20260611_001
```

### 測試與診斷

```bash
python MLOps/run.py test diagnose
python MLOps/run.py test load
python MLOps/run.py test nan
```

---

## 超參數設定

設定檔位於 `MLOps/train/configs/`，格式為純文字 `key = value`。

**載入順序**（後者覆蓋前者）：
```
Python DEFAULTS → configs/default.txt → configs/{method}/default.txt → configs/{method}/{name}.txt
```

新增實驗設定只需在對應方法資料夾新增 txt 檔：

```bash
# 新增 DPO v2 實驗，只填寫與默認不同的參數
cp MLOps/train/configs/dpo/v1.txt MLOps/train/configs/dpo/v2.txt
# 編輯 v2.txt...
python MLOps/run.py train dpo --config v2
```

---

## 環境需求

```
Python 3.10+
torch >= 2.0
transformers
peft
trl
datasets
pyyaml
```

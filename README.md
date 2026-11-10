# Alignment Experiment

小型 instruct 模型的對齊可逆性 / 可解釋性實驗工具。以單一 CLI 入口 `MLOps/run.py` 管理
兩類方法：

- **MLP 神經元裁減 / 行為編輯**（本專案的主軸）
  - `prune`（v1，mode=`km_search`）：找出並歸零 / 取負「拒絕神經元」，量測拒絕率翻轉，
    研究「裁減特定神經元能否移除拒絕行為」。
  - `prune_v2`：第二代搜尋器，改用單調 (T, K) 掃描取代 v1 的粗測＋細搜。
  - `km_cos`：行為編輯（讓模型 cosplay 某人格），用對比 system prompt + 相關性排名選點。
- **梯度微調**：`dpo` / `sft`（偏好 / 監督訓練，吃版本化語料）。

> **這個 repo 只是工具。** 安全研究用的有害 / 無害資料集**不隨 repo 發佈**；`data/` 整個
> 排除在 git 版控外，唯一附上的是 km_cos 的中性 prompt 種子（行為編輯 demo 用）。要跑
> 需要自備資料（格式見下）。

---

## 專案結構

```
alignment-experiment/
│
├── data/                         ← 統一資料根（整個 gitignore，唯一例外 seeds/km_cos_prompts.jsonl）
│   ├── seeds/                    ← prune / km_search / km_cos 的輸入提示詞種子（*.jsonl）
│   ├── model_groups/             ← km_search 跑過後，各模型對每題的真實 refused/complied 分組
│   ├── corpora/                  ← 梯度訓練（dpo/sft）的版本化語料 {type}/{version}/{date}/
│   └── analysis/                 ← 探測 / 掃描 / 報表等分析產物（非輸入）
│
├── MLOps/                        ← 訓練管理主目錄
│   ├── run.py                    ← 唯一 CLI 入口
│   ├── data_paths.py             ← 資料路徑單一來源（解析 + 共用 jsonl 讀取）
│   ├── train/
│   │   ├── configs/              ← 超參數設定檔（txt；default + 具名）
│   │   │   ├── default.txt       ← 全域共用默認
│   │   │   └── {method}/…        ← dpo / sft / prune / prune_v2 / km_cos 各自的 default + 具名
│   │   └── scripts/
│   │       ├── base_config.py / base_trainer.py   ← 設定 / 訓練父類
│   │       ├── train_dpo.py / train_sft.py        ← 梯度方法
│   │       ├── train_prune.py / train_prune_v2.py ← 神經元裁減
│   │       └── train_km_cos.py                    ← 行為編輯
│   └── experiments/              ← 每次 run 的完整產出（run_summary.json / config.txt / checkpoints/ …，gitignore）
│
└── pipelines/                    ← 資料製備與圖形介面（不經 MLOps 管理）
    ├── km_cos_gui.py             ← km_cos 行為編輯 GUI（`python run.py` 預設開這個）
    ├── refusal_data_editor.py    ← km_search 違禁題資料集編輯器（GUI / --cli）
    └── prepare_*.py / filter_*.py / audit_*.py   ← 語料製備
```

**方法是慣例註冊的**：在 `train/scripts/` 新增 `train_<method>.py` 並在 `base_config.py` 加一個
`METHOD = "<method>"` 的設定類，`run.py train <method>` 就會自動出現，不需改 `run.py`。

---

## 使用方式

所有操作透過 `MLOps/run.py`。不加參數會開 km_cos 圖形介面；`python run.py help` 看完整指令。

### 訓練

```bash
# 梯度微調（吃 data/corpora 的版本化語料）
python MLOps/run.py train dpo --config v1 [--versions v1 v2] [--dates 20260508]
python MLOps/run.py train sft --config v1

# 神經元裁減 / 掃描（吃 data/seeds 的提示詞種子）
python MLOps/run.py train prune    --config km_reverse_v1     # v1：粗測 → 細搜 K、M
python MLOps/run.py train prune_v2 --config qwen3b            # v2：單調 (T,K) 掃描
python MLOps/run.py train km_cos   --config <name>            # 行為編輯（人格誘發）

# 接續中斷的 run（通用、訊息驅動：從 run_summary 還原設定/資料/超參，沿用原目錄）
python MLOps/run.py train <method> --resume <run_id>
```

### 互動與查看

```bash
python MLOps/run.py chat <run_id> [checkpoint]      # 與訓練後模型對話（多輪）
python MLOps/run.py compare-chat <run_id> [<run_id> ...]   # 多模型同句對比
python MLOps/run.py list-models                     # 列出所有訓練產出
python MLOps/run.py result show <run_id>            # 看某 run 的生成測試與 metrics
```

### 資料與設定管理

```bash
python MLOps/run.py data list                       # 列出 corpora 版本與數量
python MLOps/run.py data default <dpo|sft> dataset  # 查 / 設梯度方法的預設版本
python MLOps/run.py data edit-prompts [path]        # 編輯 km_search 種子（含模型分類欄）
python MLOps/run.py config default [<method> [name]]# 查 / 設方法的預設設定檔
```

### 資料格式（自備）

- **prune / km_search 種子**（`data/seeds/*.jsonl`，group_source=model）：
  每行 `{"id","pair_id","text","group":"refused"|"complied","domain","lang"}`。
  `group` 只是初始標籤，實跑時由模型行為重新分組。兩組都要夠多題。
- **km_cos 種子**：中性單句題（`text`），搭配 config 裡的 target / base system prompt。
- **dpo / sft 語料**（`data/corpora/{type}/{version}/{date}/train.jsonl`）：
  每行 `{"prompt","chosen","rejected"}`（dpo）或監督樣本（sft）。

---

## 設定系統

設定檔在 `MLOps/train/configs/`，純文字 `key = value`，載入順序（後者覆蓋前者）：

```
Python DEFAULTS → configs/default.txt → configs/{method}/default.txt → configs/{method}/{name}.txt
```

新增實驗只需在對應方法資料夾放一個只寫「與默認不同」參數的 txt：

```bash
cp MLOps/train/configs/prune_v2/qwen3b.txt MLOps/train/configs/prune_v2/qwen7b.txt
# 編輯後：
python MLOps/run.py train prune_v2 --config qwen7b
```

---

## 環境需求

```
Python 3.10+
torch >= 2.0
transformers / peft / trl / datasets
```
